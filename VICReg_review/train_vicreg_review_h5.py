"""Train VICReg review model from prebuilt HDF5 data.

This avoids reparsing giant JSON files in the training loop. Batches are built
from a streamable H5 file produced by build_review_h5.py. Each game is loaded as
one contiguous vector block, two 60 percent review-level views are sampled, and
the final VICReg loss is computed over a large batch of games.
"""

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VICReg_review.model import (  # noqa: E402
    LatentArrayMLP,
    SentimentAdversarialLoss,
    load_mlp4_a_sentiment_head,
    vicreg_loss,
)

DEFAULT_H5 = SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_SST_CHECKPOINT = PROJECT_ROOT / "sst" / "heads" / "mlp4_1024_128_32_8_1_best.pt"
DEFAULT_HEADS_DIR = SCRIPT_DIR / "heads"
DEFAULT_SMOKE_RESULT = DEFAULT_HEADS_DIR / "vicreg_review_h5_worst_case_smoke.json"


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_text_write(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def sample_view(game_vectors, review_offsets, sample_fraction, rng, cache_dtype):
    review_count = len(review_offsets) - 1
    take = max(1, int(math.ceil(review_count * sample_fraction)))
    selected = rng.choice(review_count, size=take, replace=False)
    selected.sort()

    lengths = review_offsets[selected + 1] - review_offsets[selected]
    total = int(lengths.sum())
    out = np.empty((total, game_vectors.shape[1]), dtype=cache_dtype)
    cursor = 0
    for review_index, length in zip(selected, lengths):
        start = int(review_offsets[review_index])
        end = int(review_offsets[review_index + 1])
        out[cursor : cursor + int(length)] = game_vectors[start:end]
        cursor += int(length)
    return torch.from_numpy(out)


def load_game_views(h5, game_index, sample_fraction, rng, cache_dtype, pin_cache):
    game_review_offsets = h5["game_review_offsets"]
    review_offsets_ds = h5["review_offsets"]
    vectors_ds = h5["vectors"]

    review_start = int(game_review_offsets[game_index])
    review_end = int(game_review_offsets[game_index + 1])
    review_offsets = review_offsets_ds[review_start : review_end + 1].astype(np.int64)
    sentence_start = int(review_offsets[0])
    sentence_end = int(review_offsets[-1])
    game_vectors = vectors_ds[sentence_start:sentence_end]
    if game_vectors.dtype != cache_dtype:
        game_vectors = game_vectors.astype(cache_dtype, copy=False)

    relative_offsets = review_offsets - sentence_start
    view_a = sample_view(game_vectors, relative_offsets, sample_fraction, rng, cache_dtype)
    view_b = sample_view(game_vectors, relative_offsets, sample_fraction, rng, cache_dtype)
    if pin_cache and torch.cuda.is_available():
        view_a = view_a.pin_memory()
        view_b = view_b.pin_memory()
    return view_a, view_b


def game_review_sentence_counts(h5):
    game_review_offsets = h5["game_review_offsets"][:]
    review_offsets = h5["review_offsets"]
    review_counts = np.diff(game_review_offsets).astype(np.int64)
    sentence_counts = np.empty((len(review_counts),), dtype=np.int64)
    for game_index in range(len(review_counts)):
        review_start = int(game_review_offsets[game_index])
        review_end = int(game_review_offsets[game_index + 1])
        sentence_start = int(review_offsets[review_start])
        sentence_end = int(review_offsets[review_end])
        sentence_counts[game_index] = sentence_end - sentence_start
    return review_counts, sentence_counts


def game_sentence_counts(h5):
    _, sentence_counts = game_review_sentence_counts(h5)
    return sentence_counts


def select_worst_case_game(h5, mode):
    review_counts, sentence_counts = game_review_sentence_counts(h5)
    if mode == "reviews":
        game_index = int(np.argmax(review_counts))
    elif mode == "sentences":
        game_index = int(np.argmax(sentence_counts))
    else:
        raise ValueError(f"Unknown worst-case mode: {mode}")
    return {
        "game_index": game_index,
        "game_name": decode_name(h5["game_names"][game_index]),
        "reviews": int(review_counts[game_index]),
        "sentences": int(sentence_counts[game_index]),
        "max_reviews": int(review_counts.max()),
        "max_sentences": int(sentence_counts.max()),
    }


def pack_by_sentence_budget(order, counts, max_games, max_batch_sentences):
    batches = []
    current = []
    current_sentences = 0
    for game_index in order:
        game_sentences = int(counts[game_index])
        would_exceed_games = len(current) >= max_games
        would_exceed_sentences = (
            current
            and max_batch_sentences > 0
            and current_sentences + game_sentences > max_batch_sentences
        )
        if would_exceed_games or would_exceed_sentences:
            batches.append(current)
            current = []
            current_sentences = 0
        current.append(int(game_index))
        current_sentences += game_sentences
    if current:
        batches.append(current)
    return batches


def make_epoch_indices(
    num_games,
    epoch_seed,
    batch_size,
    steps_per_epoch,
    game_order="random",
    counts=None,
    max_batch_sentences=0,
):
    rng = np.random.default_rng(epoch_seed)
    if game_order == "random":
        base_order = None
    elif game_order == "largest_first":
        if counts is None:
            raise ValueError("largest_first order requires game sentence counts.")
        base_order = np.argsort(-counts).tolist()
    elif game_order == "smallest_first":
        if counts is None:
            raise ValueError("smallest_first order requires game sentence counts.")
        base_order = np.argsort(counts).tolist()
    elif game_order == "file":
        base_order = list(range(num_games))
    else:
        raise ValueError(f"Unknown game order: {game_order}")

    if max_batch_sentences > 0:
        if counts is None:
            raise ValueError("max_batch_sentences requires game sentence counts.")
        if steps_per_epoch <= 0:
            order = rng.permutation(num_games).tolist() if base_order is None else base_order
            return pack_by_sentence_budget(order, counts, batch_size, max_batch_sentences)

        batches = []
        while len(batches) < steps_per_epoch:
            order = rng.permutation(num_games).tolist() if base_order is None else base_order
            batches.extend(pack_by_sentence_budget(order, counts, batch_size, max_batch_sentences))
        return batches[:steps_per_epoch]

    if steps_per_epoch <= 0:
        indices = rng.permutation(num_games).tolist() if base_order is None else base_order
    else:
        needed = steps_per_epoch * batch_size
        indices = []
        while len(indices) < needed:
            if base_order is None:
                indices.extend(rng.permutation(num_games).tolist())
            else:
                indices.extend(base_order)
        indices = indices[:needed]
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def prepare_batch(h5, batch_indices, sample_fraction, rng, cache_dtype, pin_cache):
    game_names = h5["game_names"]
    views_a = []
    views_b = []
    names = []
    lengths_a = []
    lengths_b = []
    for game_index in batch_indices:
        view_a, view_b = load_game_views(h5, game_index, sample_fraction, rng, cache_dtype, pin_cache)
        views_a.append(view_a)
        views_b.append(view_b)
        names.append(decode_name(game_names[game_index]))
        lengths_a.append(view_a.shape[0])
        lengths_b.append(view_b.shape[0])
    return {
        "view_a": views_a,
        "view_b": views_b,
        "games": names,
        "len_a": torch.tensor(lengths_a, dtype=torch.long),
        "len_b": torch.tensor(lengths_b, dtype=torch.long),
    }


def prepare_epoch_batches(
    h5_path,
    epoch,
    batch_size,
    steps_per_epoch,
    sample_fraction,
    seed,
    cache_dtype,
    pin_cache,
    game_order,
    max_batch_sentences,
):
    batches = []
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    with h5py.File(h5_path, "r") as h5:
        num_games = int(h5["game_names"].shape[0])
        counts = game_sentence_counts(h5) if (game_order != "random" or max_batch_sentences > 0) else None
        epoch_indices = make_epoch_indices(
            num_games,
            seed + epoch,
            batch_size,
            steps_per_epoch,
            game_order=game_order,
            counts=counts,
            max_batch_sentences=max_batch_sentences,
        )
        for batch_indices in epoch_indices:
            batches.append(prepare_batch(h5, batch_indices, sample_fraction, rng, cache_dtype, pin_cache))
    return batches


class QueueEpochIterator:
    def __init__(
        self,
        h5_path,
        epoch,
        batch_size,
        steps_per_epoch,
        sample_fraction,
        seed,
        cache_dtype,
        pin_cache,
        prefetch_batches,
        game_order,
        max_batch_sentences,
    ):
        self.queue = queue.Queue(maxsize=max(1, prefetch_batches))
        self.thread = threading.Thread(
            target=self._worker,
            args=(
                h5_path,
                epoch,
                batch_size,
                steps_per_epoch,
                sample_fraction,
                seed,
                cache_dtype,
                pin_cache,
                game_order,
                max_batch_sentences,
            ),
            daemon=True,
        )
        self.thread.start()

    def _worker(
        self,
        h5_path,
        epoch,
        batch_size,
        steps_per_epoch,
        sample_fraction,
        seed,
        cache_dtype,
        pin_cache,
        game_order,
        max_batch_sentences,
    ):
        try:
            rng = np.random.default_rng(seed + epoch * 1_000_003)
            with h5py.File(h5_path, "r") as h5:
                num_games = int(h5["game_names"].shape[0])
                counts = game_sentence_counts(h5) if (game_order != "random" or max_batch_sentences > 0) else None
                epoch_indices = make_epoch_indices(
                    num_games,
                    seed + epoch,
                    batch_size,
                    steps_per_epoch,
                    game_order=game_order,
                    counts=counts,
                    max_batch_sentences=max_batch_sentences,
                )
                for batch_indices in epoch_indices:
                    self.queue.put(prepare_batch(h5, batch_indices, sample_fraction, rng, cache_dtype, pin_cache))
            self.queue.put(None)
        except BaseException as exc:
            self.queue.put(exc)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.thread.join()
                break
            if isinstance(item, BaseException):
                self.thread.join()
                raise item
            yield item


def iter_epoch(args, epoch, next_epoch_future, executor, cache_dtype):
    if args.cache_mode == "full":
        batches = next_epoch_future.result()
        if epoch < args.epochs:
            future = executor.submit(
                prepare_epoch_batches,
                args.input_h5,
                epoch + 1,
                args.batch_size,
                args.steps_per_epoch,
                args.sample_fraction,
                args.seed,
                cache_dtype,
                args.pin_cache,
                args.game_order,
                args.max_batch_sentences,
            )
        else:
            future = None
        return batches, future

    iterator = QueueEpochIterator(
        args.input_h5,
        epoch,
        args.batch_size,
        args.steps_per_epoch,
        args.sample_fraction,
        args.seed,
        cache_dtype,
        args.pin_cache,
        args.prefetch_batches,
        args.game_order,
        args.max_batch_sentences,
    )
    return iterator, None


def cuda_device_index(device):
    if device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def capture_rng_state(device):
    state = {"cpu": torch.get_rng_state(), "cuda": None}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(cuda_device_index(device))
    return state


def restore_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], cuda_device_index(device))


def move_view_to_device(view_cpu, device, pin_transfer):
    return view_cpu.unsqueeze(0).to(device, non_blocking=pin_transfer)


def forward_view(model, view_cpu, device, amp_enabled, pin_transfer):
    view = move_view_to_device(view_cpu, device, pin_transfer)
    try:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            return model(view, key_padding_mask=None)
    finally:
        del view


def compute_latent_loss(z_a, z_b, adversary, args, amp_enabled):
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        vic = vicreg_loss(
            z_a,
            z_b,
            invariance_weight=args.vicreg_invariance_weight,
            variance_weight=args.vicreg_variance_weight,
            covariance_weight=args.vicreg_covariance_weight,
        )
        adv_a, stats_a = adversary(z_a)
        adv_b, stats_b = adversary(z_b)
        adv_loss = 0.5 * (adv_a + adv_b)
        loss = vic["loss"] + args.adversary_weight * adv_loss
    return loss, vic, adv_loss, stats_a, stats_b


def make_metrics(loss, vic, adv_loss, stats_a, stats_b, batch):
    return {
        "loss": float(loss.detach().cpu()),
        "vicreg": float(vic["loss"].detach().cpu()),
        "invariance": float(vic["invariance"].detach().cpu()),
        "variance": float(vic["variance"].detach().cpu()),
        "covariance": float(vic["covariance"].detach().cpu()),
        "adversary_entropy_loss": float(adv_loss.detach().cpu()),
        "sentiment_mean": float((stats_a["sentiment_mean"] + stats_b["sentiment_mean"]).mul(0.5).cpu()),
        "sentiment_std": float((stats_a["sentiment_std"] + stats_b["sentiment_std"]).mul(0.5).cpu()),
        "sentiment_entropy": float((stats_a["sentiment_entropy"] + stats_b["sentiment_entropy"]).mul(0.5).cpu()),
        "sentences_a": float(batch["len_a"].float().mean().item()),
        "sentences_b": float(batch["len_b"].float().mean().item()),
    }


def finish_optimizer_step(model, optimizer, scaler, args):
    if args.grad_clip > 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()


def run_training_batch_standard(batch, model, adversary, optimizer, scaler, args, device, amp_enabled, pin_transfer):
    optimizer.zero_grad(set_to_none=True)
    z_a_parts = []
    z_b_parts = []
    for view_a_cpu, view_b_cpu in zip(batch["view_a"], batch["view_b"]):
        z_a_parts.append(forward_view(model, view_a_cpu, device, amp_enabled, pin_transfer))
        z_b_parts.append(forward_view(model, view_b_cpu, device, amp_enabled, pin_transfer))

    z_a = torch.cat(z_a_parts, dim=0)
    z_b = torch.cat(z_b_parts, dim=0)
    loss, vic, adv_loss, stats_a, stats_b = compute_latent_loss(z_a, z_b, adversary, args, amp_enabled)
    scaler.scale(loss).backward()
    finish_optimizer_step(model, optimizer, scaler, args)
    return make_metrics(loss, vic, adv_loss, stats_a, stats_b, batch)


def collect_recompute_latents(view_list, model, device, amp_enabled, pin_transfer):
    latents = []
    rng_states = []
    with torch.no_grad():
        for view_cpu in view_list:
            rng_states.append(capture_rng_state(device))
            latents.append(forward_view(model, view_cpu, device, amp_enabled, pin_transfer).detach())
    return latents, rng_states


def replay_latent_grads(view_list, latent_grads, rng_states, model, device, amp_enabled, pin_transfer):
    for view_cpu, latent_grad, rng_state in zip(view_list, latent_grads, rng_states):
        restore_rng_state(rng_state, device)
        z = forward_view(model, view_cpu, device, amp_enabled, pin_transfer)
        z.backward(latent_grad.unsqueeze(0))


def run_training_batch_recompute(batch, model, adversary, optimizer, scaler, args, device, amp_enabled, pin_transfer):
    optimizer.zero_grad(set_to_none=True)
    z_a_parts, rng_a = collect_recompute_latents(batch["view_a"], model, device, amp_enabled, pin_transfer)
    z_b_parts, rng_b = collect_recompute_latents(batch["view_b"], model, device, amp_enabled, pin_transfer)

    z_a = torch.cat(z_a_parts, dim=0).detach().requires_grad_(True)
    z_b = torch.cat(z_b_parts, dim=0).detach().requires_grad_(True)
    loss, vic, adv_loss, stats_a, stats_b = compute_latent_loss(z_a, z_b, adversary, args, amp_enabled)
    scaler.scale(loss).backward()

    z_a_grads = [grad.detach().clone() for grad in z_a.grad.unbind(0)]
    z_b_grads = [grad.detach().clone() for grad in z_b.grad.unbind(0)]
    metrics = make_metrics(loss, vic, adv_loss, stats_a, stats_b, batch)

    del z_a, z_b, z_a_parts, z_b_parts, loss, vic, adv_loss, stats_a, stats_b
    replay_latent_grads(batch["view_a"], z_a_grads, rng_a, model, device, amp_enabled, pin_transfer)
    replay_latent_grads(batch["view_b"], z_b_grads, rng_b, model, device, amp_enabled, pin_transfer)
    finish_optimizer_step(model, optimizer, scaler, args)
    return metrics


def run_training_batch(batch, model, adversary, optimizer, scaler, args, device, amp_enabled, pin_transfer):
    if args.backward_mode == "standard":
        return run_training_batch_standard(
            batch, model, adversary, optimizer, scaler, args, device, amp_enabled, pin_transfer
        )
    if args.backward_mode == "recompute":
        return run_training_batch_recompute(
            batch, model, adversary, optimizer, scaler, args, device, amp_enabled, pin_transfer
        )
    raise ValueError(f"Unknown backward mode: {args.backward_mode}")


def cuda_memory_summary(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 2),
    }


def build_training_components(args, input_dim, device):
    model = LatentArrayMLP(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    if args.latent_dim != 1024:
        raise ValueError("SST MLP4-A adversary requires --latent-dim 1024.")
    sentiment_head = load_mlp4_a_sentiment_head(args.sst_checkpoint, map_location=device).to(device)
    adversary = SentimentAdversarialLoss(sentiment_head, grl_lambda=args.grl_lambda).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, adversary, optimizer


def run_worst_case_smoke(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dtype = np.dtype(args.cache_dtype)
    rng = np.random.default_rng(args.seed + 17)

    with h5py.File(args.input_h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
        worst = select_worst_case_game(h5, args.smoke_worst_case_by)
        batch_indices = [worst["game_index"]] * args.batch_size
        batch = prepare_batch(h5, batch_indices, args.sample_fraction, rng, cache_dtype, args.pin_cache)

    model, adversary, optimizer = build_training_components(args, input_dim, device)
    model.train()
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pin_transfer = args.pin_cache and device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    result = {
        "status": "running",
        "finished_at": None,
        "input_h5": str(Path(args.input_h5).resolve()),
        "device": str(device),
        "amp": amp_enabled,
        "batch_size": args.batch_size,
        "sample_fraction": args.sample_fraction,
        "backward_mode": args.backward_mode,
        "cache_dtype": str(cache_dtype),
        "worst_case_by": args.smoke_worst_case_by,
        "game_index": worst["game_index"],
        "game_name": worst["game_name"],
        "game_reviews": worst["reviews"],
        "game_sentences": worst["sentences"],
        "batch_reviews": args.batch_size * worst["reviews"],
        "batch_sentences": args.batch_size * worst["sentences"],
        "view_a_sentences": batch["len_a"].tolist(),
        "view_b_sentences": batch["len_b"].tolist(),
        "metrics": None,
        "cuda_memory": None,
        "error": None,
    }

    print(
        f"worst_case_smoke: by={args.smoke_worst_case_by} game={worst['game_name']} "
        f"reviews={worst['reviews']} sentences={worst['sentences']} "
        f"batch_size={args.batch_size} batch_reviews={result['batch_reviews']} "
        f"batch_sentences={result['batch_sentences']} sample_fraction={args.sample_fraction} "
        f"backward_mode={args.backward_mode}",
        flush=True,
    )
    print(
        f"views: mean_a={float(batch['len_a'].float().mean()):.0f} "
        f"mean_b={float(batch['len_b'].float().mean()):.0f} "
        f"total_a={int(batch['len_a'].sum())} total_b={int(batch['len_b'].sum())}",
        flush=True,
    )

    try:
        metrics = run_training_batch(
            batch,
            model,
            adversary,
            optimizer,
            scaler,
            args,
            device,
            amp_enabled,
            pin_transfer,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["status"] = "done"
        result["metrics"] = metrics
        result["cuda_memory"] = cuda_memory_summary(device)
        print(
            f"smoke done: loss={metrics['loss']:.4f} "
            f"sentences=({metrics['sentences_a']:.0f},{metrics['sentences_b']:.0f}) "
            f"cuda={result['cuda_memory']}",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["cuda_memory"] = cuda_memory_summary(device)
        print(f"smoke OOM: {result['cuda_memory']}", flush=True)
        raise
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["cuda_memory"] = cuda_memory_summary(device)
        raise
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_text_write(json.dumps(result, ensure_ascii=False, indent=2), args.smoke_result_json)


def write_history(rows, path):
    if not rows:
        return
    columns = list(rows[0].keys())
    lines = ["\t".join(columns)]
    for row in rows:
        line = []
        for column in columns:
            value = row[column]
            line.append(f"{value:.10g}" if isinstance(value, float) else str(value))
        lines.append("\t".join(line))
    atomic_text_write("\n".join(lines) + "\n", path)


def write_manifest(path, status, args, epoch, step, metrics=None, error=None):
    payload = {
        "status": status,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": epoch,
        "step": step,
        "input_h5": str(Path(args.input_h5).resolve()),
        "checkpoint_out": str(Path(args.checkpoint_out).resolve()),
        "sample_fraction": args.sample_fraction,
        "batch_size": args.batch_size,
        "max_batch_sentences": args.max_batch_sentences,
        "cache_mode": args.cache_mode,
        "backward_mode": args.backward_mode,
        "game_order": args.game_order,
        "metrics": metrics or {},
        "error": error,
    }
    atomic_text_write(json.dumps(payload, ensure_ascii=False, indent=2), path)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dtype = np.dtype(args.cache_dtype)

    with h5py.File(args.input_h5, "r") as h5:
        num_games = int(h5["game_names"].shape[0])
        input_dim = int(h5.attrs["input_dim"])
        total_sentences = int(h5["vectors"].shape[0])
        if args.max_batch_sentences > 0:
            counts = game_sentence_counts(h5)
            default_steps = len(
                make_epoch_indices(
                    num_games,
                    args.seed + 1,
                    args.batch_size,
                    0,
                    game_order=args.game_order,
                    counts=counts,
                    max_batch_sentences=args.max_batch_sentences,
                )
            )
        else:
            default_steps = math.ceil(num_games / args.batch_size)
    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = default_steps

    model, adversary, optimizer = build_training_components(args, input_dim, device)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pin_transfer = args.pin_cache and device.type == "cuda"
    history_rows = []
    best_loss = float("inf")
    global_step = 0

    print(
        f"device={device} games={num_games} sentences={total_sentences} "
        f"batch_size={args.batch_size} steps_per_epoch={args.steps_per_epoch} "
        f"max_batch_sentences={args.max_batch_sentences} "
        f"cache_mode={args.cache_mode} backward_mode={args.backward_mode} "
        f"prefetch_batches={args.prefetch_batches} "
        f"sample_fraction={args.sample_fraction} game_order={args.game_order}",
        flush=True,
    )
    print(f"model=LatentArrayMLP params={sum(p.numel() for p in model.parameters())}", flush=True)

    executor = None
    next_epoch_future = None
    if args.cache_mode == "full":
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        next_epoch_future = executor.submit(
            prepare_epoch_batches,
            args.input_h5,
            1,
            args.batch_size,
            args.steps_per_epoch,
            args.sample_fraction,
            args.seed,
            cache_dtype,
            args.pin_cache,
            args.game_order,
            args.max_batch_sentences,
        )

    last_metrics = None
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_sums = {}
            epoch_batches, next_epoch_future = iter_epoch(args, epoch, next_epoch_future, executor, cache_dtype)

            for batch_index, batch in enumerate(epoch_batches, start=1):
                metrics = run_training_batch(
                    batch,
                    model,
                    adversary,
                    optimizer,
                    scaler,
                    args,
                    device,
                    amp_enabled,
                    pin_transfer,
                )
                global_step += 1
                last_metrics = metrics
                for key, value in metrics.items():
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + value

                if batch_index == 1 or batch_index % args.log_every == 0:
                    print(
                        f"epoch={epoch:03d} step={batch_index:04d}/{args.steps_per_epoch} "
                        f"global={global_step} loss={metrics['loss']:.4f} "
                        f"vic={metrics['vicreg']:.4f} inv={metrics['invariance']:.4f} "
                        f"var={metrics['variance']:.4f} cov={metrics['covariance']:.4f} "
                        f"adv_entropy={metrics['sentiment_entropy']:.4f} "
                        f"sent_mean={metrics['sentiment_mean']:.4f} "
                        f"sentences=({metrics['sentences_a']:.0f},{metrics['sentences_b']:.0f}) "
                        f"games={','.join(batch['games'][:3])}",
                        flush=True,
                    )

            averaged = {key: value / args.steps_per_epoch for key, value in epoch_sums.items()}
            history_rows.append({"epoch": epoch, "global_step": global_step, **averaged})
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                "metrics": averaged,
                "model_class": "LatentArrayMLP",
                "num_latents": args.num_latents,
                "latent_dim": args.latent_dim,
                "input_h5": str(Path(args.input_h5).resolve()),
                "sst_checkpoint": str(Path(args.sst_checkpoint).resolve()),
            }
            if not args.no_save:
                atomic_torch_save(checkpoint, args.checkpoint_out)
            if averaged["loss"] < best_loss:
                best_loss = averaged["loss"]
                if not args.no_save:
                    atomic_torch_save(checkpoint, args.best_checkpoint_out)

            write_history(history_rows, args.history_tsv)
            write_manifest(args.manifest_json, "running", args, epoch, global_step, averaged)

        write_manifest(args.manifest_json, "done", args, args.epochs, global_step, last_metrics)
    except KeyboardInterrupt:
        write_manifest(args.manifest_json, "interrupted", args, epoch if "epoch" in locals() else 0, global_step, last_metrics)
        raise
    except BaseException as exc:
        write_manifest(
            args.manifest_json,
            "error",
            args,
            epoch if "epoch" in locals() else 0,
            global_step,
            last_metrics,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", default=str(DEFAULT_H5))
    parser.add_argument("--sst-checkpoint", default=str(DEFAULT_SST_CHECKPOINT))
    parser.add_argument("--checkpoint-out", default=str(DEFAULT_HEADS_DIR / "vicreg_review_h5_latest.pt"))
    parser.add_argument("--best-checkpoint-out", default=str(DEFAULT_HEADS_DIR / "vicreg_review_h5_best.pt"))
    parser.add_argument("--history-tsv", default=str(DEFAULT_HEADS_DIR / "vicreg_review_h5_history.tsv"))
    parser.add_argument("--manifest-json", default=str(DEFAULT_HEADS_DIR / "vicreg_review_h5_manifest.json"))
    parser.add_argument("--smoke-result-json", default=str(DEFAULT_SMOKE_RESULT))
    parser.add_argument("--no-save", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means one full pass over game IDs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-batch-sentences",
        type=int,
        default=0,
        help=(
            "Optional per-batch budget on original sentence count. Use with a large "
            "--batch-size so small games pack together while large games form smaller batches."
        ),
    )
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--cache-mode", choices=["queue", "full"], default="queue")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--pin-cache", action="store_true")
    parser.add_argument(
        "--backward-mode",
        choices=["recompute", "standard"],
        default="recompute",
        help="recompute caches only latents, then replays one view at a time for backward to reduce VRAM.",
    )
    parser.add_argument(
        "--game-order",
        choices=["random", "largest_first", "smallest_first", "file"],
        default="random",
        help="Order game IDs inside each epoch. largest_first is useful for worst-case memory tests.",
    )
    parser.add_argument(
        "--smoke-worst-case",
        action="store_true",
        help="Run one worst-case batch_size * max-game smoke step instead of a full training run.",
    )
    parser.add_argument(
        "--smoke-worst-case-by",
        choices=["reviews", "sentences"],
        default="reviews",
        help="Select the repeated worst-case game by review count or sentence count.",
    )

    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--num-latents", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--adversary-weight", type=float, default=1.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_worst_case:
        run_worst_case_smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
