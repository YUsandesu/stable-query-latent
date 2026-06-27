"""Train VICReg review model from prebuilt HDF5 data.

This avoids reparsing giant JSON files in the training loop. Batches are built
from the streamable H5 file produced by ``game_review_data/build.py``. Each game
is loaded as one contiguous vector block, two 60 percent review-level views are
sampled, and the final VICReg loss is computed over a large batch of games.
"""

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VICReg_review.model import (  # noqa: E402
    GameCentroidExpander,
    HierarchicalLatentArrayMLP,
    LatentArrayMLP,
    SentimentAdversarialLoss,
    game_centroid,
    load_mlp4_a_sentiment_head,
    vicreg_centroid_loss,
    vicreg_loss,
)

DEFAULT_H5 = PROJECT_ROOT / "game_review_data" / "embedding_h5.h5"
DEFAULT_SST_CHECKPOINT = PROJECT_ROOT / "sst" / "heads" / "mlp4_1024_128_32_8_1_best.pt"
DEFAULT_HEADS_DIR = SCRIPT_DIR / "heads"
DEFAULT_SMOKE_RESULT = DEFAULT_HEADS_DIR / "vicreg_review_h5_worst_case_smoke.json"
DEFAULT_DESCRIPTION_DIR = SCRIPT_DIR / "tags" / "game_descriptions"
DEFAULT_DESCRIPTION_CACHE = DEFAULT_HEADS_DIR / "description_embedding_cache.npz"
REQUIRED_TRAINING_H5_DATASETS = (
    "vectors",
    "review_offsets",
    "game_review_offsets",
    "game_names",
)


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def split_text_for_embedding(text, max_sentences):
    import re

    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(part) for part in value)
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def parse_string_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(",")
    return [str(part).strip() for part in parts if str(part).strip()]


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


def validate_training_h5(h5, path=None):
    missing = [name for name in REQUIRED_TRAINING_H5_DATASETS if name not in h5]
    if missing:
        hint = ""
        if "texts" in h5 and "vectors" not in h5:
            hint = " This looks like text_h5.h5; run the embed-h5 stage and pass embedding_h5.h5."
        raise ValueError(f"Training H5 is missing datasets {missing}.{hint}")
    vectors = h5["vectors"]
    review_offsets = h5["review_offsets"]
    game_review_offsets = h5["game_review_offsets"]
    game_names = h5["game_names"]
    if vectors.ndim != 2:
        raise ValueError(f"{path or 'H5'}: vectors must be 2D, got shape {vectors.shape}")
    if review_offsets.ndim != 1 or game_review_offsets.ndim != 1:
        raise ValueError(f"{path or 'H5'}: review/game offsets must be 1D")
    if int(review_offsets.shape[0]) < 1 or int(game_review_offsets.shape[0]) != int(game_names.shape[0]) + 1:
        raise ValueError(f"{path or 'H5'}: invalid game/review offset lengths")
    if int(review_offsets[-1]) != int(vectors.shape[0]):
        raise ValueError(
            f"{path or 'H5'}: review_offsets[-1]={int(review_offsets[-1])} "
            f"but vectors rows={int(vectors.shape[0])}"
        )
    if int(game_review_offsets[-1]) != int(review_offsets.shape[0] - 1):
        raise ValueError(
            f"{path or 'H5'}: game_review_offsets[-1]={int(game_review_offsets[-1])} "
            f"but review count={int(review_offsets.shape[0] - 1)}"
        )
    if "appids" in h5 and int(h5["appids"].shape[0]) != int(game_names.shape[0]):
        raise ValueError(f"{path or 'H5'}: appids length does not match game_names")
    return True


def default_extra_description_sources():
    return [
        ("1091500", "Cyberpunk 2077", "neutral", PROJECT_ROOT / "2077_text.txt"),
        ("1091500", "Cyberpunk 2077", "positive", PROJECT_ROOT / "2077_text_postive.txt"),
        ("1091500", "Cyberpunk 2077", "negative", PROJECT_ROOT / "2077_text_negative.txt"),
        ("1385380", "Across the Obelisk", "neutral", PROJECT_ROOT / "AO_text.txt"),
        ("1385380", "Across the Obelisk", "positive", PROJECT_ROOT / "AO_text_postive.txt"),
        ("1385380", "Across the Obelisk", "negative", PROJECT_ROOT / "AO_text_negative.txt"),
    ]


def collect_description_sources(args):
    sources = []
    description_dir = Path(args.description_dir)
    if description_dir.exists():
        for path in sorted(description_dir.glob("*.txt")):
            sources.append((path.stem, path.stem, "game_description", path))
    if args.description_include_extra_cases:
        for appid, title, variant, path in default_extra_description_sources():
            if path.exists():
                sources.append((appid, title, variant, path))
    return sources


def build_description_cache(args):
    from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder

    sources = collect_description_sources(args)
    if not sources:
        raise FileNotFoundError(
            f"No description text files found in {args.description_dir} "
            "and no extra description cases are available."
        )

    embedder = LocalEmbedder(
        args.description_local_model or DEFAULT_LOCAL_MODEL,
        device=args.device,
        batch_size=args.description_embed_batch_size,
    )
    vectors = []
    offsets = [0]
    appids = []
    titles = []
    variants = []
    paths = []
    for index, (appid, title, variant, path) in enumerate(sources, start=1):
        text = Path(path).read_text(encoding="utf-8")
        sentences = split_text_for_embedding(text, args.description_max_sentences)
        if not sentences:
            continue
        embedded = np.asarray(embedder.embed(sentences), dtype=np.float32)
        vectors.append(embedded)
        offsets.append(offsets[-1] + embedded.shape[0])
        appids.append(str(appid))
        titles.append(str(title))
        variants.append(str(variant))
        paths.append(str(Path(path).resolve()))
        if index % 25 == 0 or index == len(sources):
            print(
                f"description embeddings {index}/{len(sources)} "
                f"appid={appid} variant={variant} sentences={len(sentences)}",
                flush=True,
            )

    if not vectors:
        raise ValueError("Description source collection produced no non-empty texts.")
    flat = np.concatenate(vectors, axis=0).astype(np.float32)
    cache_path = Path(args.description_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                vectors=flat,
                offsets=np.asarray(offsets, dtype=np.int64),
                appids=np.asarray(appids, dtype=object),
                titles=np.asarray(titles, dtype=object),
                variants=np.asarray(variants, dtype=object),
                paths=np.asarray(paths, dtype=object),
                local_model=args.description_local_model or DEFAULT_LOCAL_MODEL,
                max_sentences=int(args.description_max_sentences),
            )
        tmp_path.replace(cache_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"wrote description embedding cache -> {cache_path}", flush=True)
    return cache_path


def load_description_bank(args, h5_appids):
    cache_path = Path(args.description_cache)
    if args.overwrite_description_cache or not cache_path.exists():
        build_description_cache(args)

    data = np.load(cache_path, allow_pickle=True)
    vectors = data["vectors"].astype(np.float32, copy=False)
    offsets = data["offsets"].astype(np.int64, copy=False)
    appids = [str(x) for x in data["appids"]]
    variants = [str(x) for x in data["variants"]]
    paths = [str(x) for x in data["paths"]]
    by_appid = {}
    for item_index, appid in enumerate(appids):
        start = int(offsets[item_index])
        end = int(offsets[item_index + 1])
        if end <= start:
            continue
        by_appid.setdefault(appid, []).append({
            "vectors": vectors[start:end],
            "variant": variants[item_index],
            "path": paths[item_index],
        })

    bank = []
    missing = []
    for game_index, appid in enumerate(h5_appids):
        variants_for_game = by_appid.get(str(appid), [])
        bank.append(variants_for_game)
        if not variants_for_game:
            missing.append((game_index, str(appid)))
    covered = len(bank) - len(missing)
    print(
        f"description bank: covered={covered}/{len(bank)} "
        f"text_variants={sum(len(items) for items in bank)} cache={cache_path}",
        flush=True,
    )
    if covered == 0:
        raise ValueError("Description alignment requested but no H5 appids were covered by the cache.")
    return bank


def load_recommendation_targets(args, num_games):
    from backheads.train_recommendation_head import DEFAULT_REVIEWS_DIR, load_labels_for_h5

    rows, keep_indices, missing = load_labels_for_h5(
        Path(args.input_h5),
        Path(args.recommendation_reviews_dir or DEFAULT_REVIEWS_DIR),
        label_min_length=args.recommendation_label_min_length,
        min_label_count=args.recommendation_min_label_count,
    )
    targets = np.full((num_games,), np.nan, dtype=np.float32)
    for row, game_index in zip(rows, keep_indices):
        targets[int(game_index)] = float(row.positive_rate)
    print(
        f"recommendation targets: covered={len(rows)}/{num_games} "
        f"missing={len(missing)} transform={args.recommendation_target_transform}",
        flush=True,
    )
    return targets


def resolve_train_game_indices(args, h5) -> np.ndarray | None:
    if args.train_game_count <= 0:
        args.train_game_indices = None
        args.train_game_appids = []
        return None

    num_games = int(h5["game_names"].shape[0])
    if args.train_game_count > num_games:
        raise ValueError(f"--train-game-count {args.train_game_count} exceeds H5 games={num_games}.")
    appids = [decode_name(x) for x in h5["appids"][:]] if "appids" in h5 else [
        decode_name(x).split("_", 1)[0] for x in h5["game_names"][:]
    ]
    anchors = parse_string_list(args.train_game_anchor_appids)
    anchor_indices = []
    missing = []
    for appid in anchors:
        try:
            anchor_indices.append(appids.index(appid))
        except ValueError:
            missing.append(appid)
    if missing:
        raise ValueError(f"Anchor appids not found in H5: {missing}")
    if len(set(anchor_indices)) > args.train_game_count:
        raise ValueError("--train-game-count is smaller than the unique anchor appid count.")

    rng = np.random.default_rng(args.train_game_seed)
    order = rng.permutation(num_games).tolist()
    selected = []
    seen = set()
    for index in anchor_indices + order:
        index = int(index)
        if index in seen:
            continue
        selected.append(index)
        seen.add(index)
        if len(selected) >= args.train_game_count:
            break

    selected = np.asarray(selected, dtype=np.int64)
    args.train_game_indices = selected
    args.train_game_appids = [appids[int(i)] for i in selected]
    return selected


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


def limit_view_sentences(view, max_view_sentences, rng):
    if max_view_sentences <= 0 or view.shape[0] <= max_view_sentences:
        return view
    selected = rng.choice(int(view.shape[0]), size=int(max_view_sentences), replace=False)
    selected.sort()
    return view[torch.from_numpy(selected.astype(np.int64))].contiguous()


def load_game_views(h5, game_index, sample_fraction, rng, cache_dtype, pin_cache, max_view_sentences=0):
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
    view_a = limit_view_sentences(view_a, max_view_sentences, rng)
    view_b = limit_view_sentences(view_b, max_view_sentences, rng)
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
    game_indices=None,
):
    rng = np.random.default_rng(epoch_seed)
    if game_indices is None:
        available = np.arange(num_games, dtype=np.int64)
    else:
        available = np.asarray(game_indices, dtype=np.int64)
        if available.ndim != 1 or available.size == 0:
            raise ValueError("game_indices must be a non-empty 1D sequence.")
    if game_order == "random":
        base_order = None
    elif game_order == "largest_first":
        if counts is None:
            raise ValueError("largest_first order requires game sentence counts.")
        base_order = available[np.argsort(-counts[available])].tolist()
    elif game_order == "smallest_first":
        if counts is None:
            raise ValueError("smallest_first order requires game sentence counts.")
        base_order = available[np.argsort(counts[available])].tolist()
    elif game_order == "file":
        base_order = available.tolist()
    else:
        raise ValueError(f"Unknown game order: {game_order}")

    if max_batch_sentences > 0:
        if counts is None:
            raise ValueError("max_batch_sentences requires game sentence counts.")
        if steps_per_epoch <= 0:
            order = rng.permutation(available).tolist() if base_order is None else base_order
            return pack_by_sentence_budget(order, counts, batch_size, max_batch_sentences)

        batches = []
        while len(batches) < steps_per_epoch:
            order = rng.permutation(available).tolist() if base_order is None else base_order
            batches.extend(pack_by_sentence_budget(order, counts, batch_size, max_batch_sentences))
        return batches[:steps_per_epoch]

    if steps_per_epoch <= 0:
        indices = rng.permutation(available).tolist() if base_order is None else base_order
    else:
        if game_indices is not None and base_order is None:
            take = min(int(batch_size), int(available.size))
            return [rng.choice(available, size=take, replace=False).tolist() for _ in range(steps_per_epoch)]
        needed = steps_per_epoch * batch_size
        indices = []
        while len(indices) < needed:
            if base_order is None:
                indices.extend(rng.permutation(available).tolist())
            else:
                indices.extend(base_order)
        indices = indices[:needed]
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def prepare_batch(h5, batch_indices, sample_fraction, rng, cache_dtype, pin_cache, max_view_sentences=0):
    game_names = h5["game_names"]
    views_a = []
    views_b = []
    names = []
    lengths_a = []
    lengths_b = []
    for game_index in batch_indices:
        view_a, view_b = load_game_views(
            h5,
            game_index,
            sample_fraction,
            rng,
            cache_dtype,
            pin_cache,
            max_view_sentences=max_view_sentences,
        )
        views_a.append(view_a)
        views_b.append(view_b)
        names.append(decode_name(game_names[game_index]))
        lengths_a.append(view_a.shape[0])
        lengths_b.append(view_b.shape[0])
    return {
        "view_a": views_a,
        "view_b": views_b,
        "games": names,
        "indices": [int(game_index) for game_index in batch_indices],
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
    max_view_sentences,
    game_indices=None,
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
            game_indices=game_indices,
        )
        for batch_indices in epoch_indices:
            batches.append(
                prepare_batch(
                    h5,
                    batch_indices,
                    sample_fraction,
                    rng,
                    cache_dtype,
                    pin_cache,
                    max_view_sentences=max_view_sentences,
                )
            )
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
        max_view_sentences,
        game_indices=None,
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
                max_view_sentences,
                game_indices,
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
        max_view_sentences,
        game_indices,
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
                    game_indices=game_indices,
                )
                for batch_indices in epoch_indices:
                    self.queue.put(
                        prepare_batch(
                            h5,
                            batch_indices,
                            sample_fraction,
                            rng,
                            cache_dtype,
                            pin_cache,
                            max_view_sentences=max_view_sentences,
                        )
                    )
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
                args.max_view_sentences,
                args.train_game_indices,
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
        args.max_view_sentences,
        args.train_game_indices,
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


def forward_view_stem(model, view_cpu, device, amp_enabled, pin_transfer):
    view = move_view_to_device(view_cpu, device, pin_transfer)
    try:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            return model.forward_stem(view, key_padding_mask=None)
    finally:
        del view


def forward_stems_to_outputs(model, stems, amp_enabled):
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        return model.forward_tail(torch.cat(stems, dim=0))


def sample_description_views(batch, description_bank, rng, cache_dtype, pin_cache):
    if description_bank is None:
        return [], []
    views = []
    positions = []
    for position, game_index in enumerate(batch["indices"]):
        variants = description_bank[int(game_index)]
        if not variants:
            continue
        item = variants[int(rng.integers(0, len(variants)))]
        arr = item["vectors"]
        if arr.dtype != cache_dtype:
            arr = arr.astype(cache_dtype, copy=False)
        tensor = torch.from_numpy(arr)
        if pin_cache and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        views.append(tensor)
        positions.append(position)
    return views, positions


def description_alignment_loss(description_centroids, review_centroids, temperature):
    if description_centroids.numel() == 0 or review_centroids.numel() == 0:
        return description_centroids.new_tensor(0.0)
    description_centroids = F.normalize(description_centroids.float(), dim=-1)
    review_centroids = F.normalize(review_centroids.float(), dim=-1)
    logits = description_centroids @ review_centroids.T
    logits = logits / max(float(temperature), 1e-6)
    target = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def recommendation_label_vector(batch, recommendation_targets, args, device):
    if recommendation_targets is None:
        return None
    values = np.asarray([recommendation_targets[int(i)] for i in batch["indices"]], dtype=np.float32)
    target = torch.from_numpy(values).to(device=device)
    if args.recommendation_target_transform == "logit":
        target = target.clamp(1e-4, 1.0 - 1e-4).logit()
    return target


def linear_label_decorrelation_loss(features, target, eps=1e-6):
    mask = torch.isfinite(target)
    if int(mask.sum().item()) < 3:
        return features.new_tensor(0.0)
    x = features[mask].float()
    y = target[mask].float().view(-1, 1)
    y = y - y.mean(dim=0, keepdim=True)
    if float(y.pow(2).mean().detach().cpu()) < eps:
        return features.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    x_std = torch.sqrt(x.pow(2).mean(dim=0, keepdim=True) + eps)
    y_std = torch.sqrt(y.pow(2).mean(dim=0, keepdim=True) + eps)
    corr = (x * y).mean(dim=0, keepdim=True) / (x_std * y_std)
    return corr.pow(2).mean()


def compute_latent_loss(
    z_a,
    z_b,
    expander,
    adversary,
    args,
    amp_enabled,
    z_description=None,
    description_positions=None,
    recommendation_targets=None,
):
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        centroid_a = game_centroid(z_a)
        centroid_b = game_centroid(z_b)
        if args.vicreg_scope == "slot":
            vic = vicreg_loss(
                z_a,
                z_b,
                invariance_weight=args.vicreg_invariance_weight,
                variance_weight=args.vicreg_variance_weight,
                covariance_weight=args.vicreg_covariance_weight,
            )
            adv_input_a = z_a
            adv_input_b = z_b
        elif args.vicreg_scope == "game":
            vic = vicreg_centroid_loss(
                centroid_a,
                centroid_b,
                expander,
                invariance_weight=args.vicreg_invariance_weight,
                variance_weight=args.vicreg_variance_weight,
                covariance_weight=args.vicreg_covariance_weight,
                compact_variance_weight=args.compact_variance_weight,
                compact_covariance_weight=args.compact_covariance_weight,
            )
            adv_input_a = centroid_a
            adv_input_b = centroid_b
        else:
            raise ValueError(f"Unknown vicreg scope: {args.vicreg_scope}")
        adv_a, stats_a = adversary(adv_input_a)
        adv_b, stats_b = adversary(adv_input_b)
        adv_loss = 0.5 * (adv_a + adv_b)
        loss = vic["loss"] + args.adversary_weight * adv_loss

        extra = {}
        review_centroid = 0.5 * (centroid_a + centroid_b)
        if (
            z_description is not None
            and description_positions
            and (args.description_align_weight > 0 or args.description_mse_weight > 0)
        ):
            description_centroid = game_centroid(z_description)
            positions = torch.as_tensor(description_positions, device=review_centroid.device, dtype=torch.long)
            paired_review = review_centroid.index_select(0, positions)
            align = description_alignment_loss(
                description_centroid,
                paired_review,
                args.description_align_temperature,
            )
            mse = F.mse_loss(description_centroid.float(), paired_review.float())
            loss = loss + args.description_align_weight * align + args.description_mse_weight * mse
            extra["description_align"] = align
            extra["description_mse"] = mse
            extra["description_count"] = description_centroid.new_tensor(float(description_centroid.size(0)))

        if recommendation_targets is not None and args.recommendation_decorr_weight > 0:
            reco = linear_label_decorrelation_loss(review_centroid, recommendation_targets)
            loss = loss + args.recommendation_decorr_weight * reco
            extra["recommendation_decorr"] = reco
    return loss, vic, adv_loss, stats_a, stats_b, extra


def make_metrics(loss, vic, adv_loss, stats_a, stats_b, batch, extra=None):
    metrics = {
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
    if "compact_variance" in vic:
        metrics["compact_variance"] = float(vic["compact_variance"].detach().cpu())
        metrics["compact_covariance"] = float(vic["compact_covariance"].detach().cpu())
    for key, value in (extra or {}).items():
        metrics[key] = float(value.detach().cpu())
    return metrics


def finish_optimizer_step(model, optimizer, scaler, args):
    if args.grad_clip > 0:
        scaler.unscale_(optimizer)
        # Clip every optimized parameter (encoder + the learnable adversary probe),
        # not just the encoder, or the probe can explode to NaN.
        params = [p for group in optimizer.param_groups for p in group["params"]]
        clip_grad_norm_(params, args.grad_clip)
    scaler.step(optimizer)
    scaler.update()


def run_training_batch_standard(
    batch,
    model,
    expander,
    adversary,
    optimizer,
    scaler,
    args,
    device,
    amp_enabled,
    pin_transfer,
    description_bank,
    description_rng,
    recommendation_targets_np,
    cache_dtype,
):
    optimizer.zero_grad(set_to_none=True)
    z_a_parts = []
    z_b_parts = []
    for view_a_cpu, view_b_cpu in zip(batch["view_a"], batch["view_b"]):
        z_a_parts.append(forward_view(model, view_a_cpu, device, amp_enabled, pin_transfer))
        z_b_parts.append(forward_view(model, view_b_cpu, device, amp_enabled, pin_transfer))

    z_a = torch.cat(z_a_parts, dim=0)
    z_b = torch.cat(z_b_parts, dim=0)
    description_views, description_positions = sample_description_views(
        batch, description_bank, description_rng, cache_dtype, pin_cache=False
    )
    z_description = None
    if description_views:
        z_description = torch.cat(
            [forward_view(model, view_cpu, device, amp_enabled, pin_transfer) for view_cpu in description_views],
            dim=0,
        )
    recommendation_targets = recommendation_label_vector(batch, recommendation_targets_np, args, device)
    loss, vic, adv_loss, stats_a, stats_b, extra = compute_latent_loss(
        z_a,
        z_b,
        expander,
        adversary,
        args,
        amp_enabled,
        z_description=z_description,
        description_positions=description_positions,
        recommendation_targets=recommendation_targets,
    )
    scaler.scale(loss).backward()
    finish_optimizer_step(model, optimizer, scaler, args)
    return make_metrics(loss, vic, adv_loss, stats_a, stats_b, batch, extra)


def collect_recompute_latents(view_list, model, device, amp_enabled, pin_transfer):
    latents = []
    rng_states = []
    with torch.no_grad():
        for view_cpu in view_list:
            rng_states.append(capture_rng_state(device))
            latents.append(forward_view(model, view_cpu, device, amp_enabled, pin_transfer).detach())
    return latents, rng_states


def collect_split_stems(view_list, model, device, amp_enabled, pin_transfer):
    stems = []
    rng_states = []
    with torch.no_grad():
        for view_cpu in view_list:
            rng_states.append(capture_rng_state(device))
            stems.append(forward_view_stem(model, view_cpu, device, amp_enabled, pin_transfer).detach().cpu())
    return stems, rng_states


def replay_stem_grads(view_list, stem_grads, rng_states, model, device, amp_enabled, pin_transfer):
    for view_cpu, stem_grad, rng_state in zip(view_list, stem_grads, rng_states):
        restore_rng_state(rng_state, device)
        stem = forward_view_stem(model, view_cpu, device, amp_enabled, pin_transfer)
        stem.backward(stem_grad.to(device, non_blocking=pin_transfer).unsqueeze(0))
        del stem, stem_grad
        if device.type == "cuda" and view_cpu.shape[0] >= 20000:
            torch.cuda.empty_cache()


def replay_latent_grads(view_list, latent_grads, rng_states, model, device, amp_enabled, pin_transfer):
    for view_cpu, latent_grad, rng_state in zip(view_list, latent_grads, rng_states):
        restore_rng_state(rng_state, device)
        z = forward_view(model, view_cpu, device, amp_enabled, pin_transfer)
        z.backward(latent_grad.unsqueeze(0))
        del z
        if device.type == "cuda" and view_cpu.shape[0] >= 20000:
            torch.cuda.empty_cache()


def run_training_batch_split_recompute(
    batch,
    model,
    expander,
    adversary,
    optimizer,
    scaler,
    args,
    device,
    amp_enabled,
    pin_transfer,
    description_bank,
    description_rng,
    recommendation_targets_np,
    cache_dtype,
):
    if not (hasattr(model, "forward_stem") and hasattr(model, "forward_tail")):
        return run_training_batch_recompute(
            batch,
            model,
            expander,
            adversary,
            optimizer,
            scaler,
            args,
            device,
            amp_enabled,
            pin_transfer,
            description_bank,
            description_rng,
            recommendation_targets_np,
            cache_dtype,
        )

    optimizer.zero_grad(set_to_none=True)
    stem_a_parts, rng_a = collect_split_stems(batch["view_a"], model, device, amp_enabled, pin_transfer)
    stem_b_parts, rng_b = collect_split_stems(batch["view_b"], model, device, amp_enabled, pin_transfer)
    description_views, description_positions = sample_description_views(
        batch, description_bank, description_rng, cache_dtype, pin_cache=False
    )
    if description_views:
        stem_description_parts, rng_description = collect_split_stems(
            description_views, model, device, amp_enabled, pin_transfer
        )
        stem_description = torch.cat(stem_description_parts, dim=0).to(device, non_blocking=pin_transfer).requires_grad_(True)
        z_description = forward_stems_to_outputs(model, [stem_description], amp_enabled)
    else:
        stem_description_parts = []
        rng_description = []
        stem_description = None
        z_description = None

    stem_a = torch.cat(stem_a_parts, dim=0).to(device, non_blocking=pin_transfer).requires_grad_(True)
    stem_b = torch.cat(stem_b_parts, dim=0).to(device, non_blocking=pin_transfer).requires_grad_(True)
    z_a = forward_stems_to_outputs(model, [stem_a], amp_enabled)
    z_b = forward_stems_to_outputs(model, [stem_b], amp_enabled)
    recommendation_targets = recommendation_label_vector(batch, recommendation_targets_np, args, device)
    loss, vic, adv_loss, stats_a, stats_b, extra = compute_latent_loss(
        z_a,
        z_b,
        expander,
        adversary,
        args,
        amp_enabled,
        z_description=z_description,
        description_positions=description_positions,
        recommendation_targets=recommendation_targets,
    )
    scaler.scale(loss).backward()

    stem_a_grads = [grad.detach().cpu().clone() for grad in stem_a.grad.unbind(0)]
    stem_b_grads = [grad.detach().cpu().clone() for grad in stem_b.grad.unbind(0)]
    if stem_description is not None and stem_description.grad is not None:
        stem_description_grads = [grad.detach().cpu().clone() for grad in stem_description.grad.unbind(0)]
    else:
        stem_description_grads = []
    metrics = make_metrics(loss, vic, adv_loss, stats_a, stats_b, batch, extra)

    del (
        z_a,
        z_b,
        z_description,
        stem_a,
        stem_b,
        stem_description,
        stem_a_parts,
        stem_b_parts,
        stem_description_parts,
        loss,
        vic,
        adv_loss,
        stats_a,
        stats_b,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    replay_stem_grads(batch["view_a"], stem_a_grads, rng_a, model, device, amp_enabled, pin_transfer)
    del stem_a_grads, rng_a
    replay_stem_grads(batch["view_b"], stem_b_grads, rng_b, model, device, amp_enabled, pin_transfer)
    del stem_b_grads, rng_b
    if stem_description_grads:
        replay_stem_grads(
            description_views,
            stem_description_grads,
            rng_description,
            model,
            device,
            amp_enabled,
            pin_transfer,
        )
        del stem_description_grads, rng_description
    finish_optimizer_step(model, optimizer, scaler, args)
    return metrics


def run_training_batch_recompute(
    batch,
    model,
    expander,
    adversary,
    optimizer,
    scaler,
    args,
    device,
    amp_enabled,
    pin_transfer,
    description_bank,
    description_rng,
    recommendation_targets_np,
    cache_dtype,
):
    optimizer.zero_grad(set_to_none=True)
    z_a_parts, rng_a = collect_recompute_latents(batch["view_a"], model, device, amp_enabled, pin_transfer)
    z_b_parts, rng_b = collect_recompute_latents(batch["view_b"], model, device, amp_enabled, pin_transfer)
    description_views, description_positions = sample_description_views(
        batch, description_bank, description_rng, cache_dtype, pin_cache=False
    )
    if description_views:
        z_description_parts, rng_description = collect_recompute_latents(
            description_views, model, device, amp_enabled, pin_transfer
        )
        z_description = torch.cat(z_description_parts, dim=0).detach().requires_grad_(True)
    else:
        z_description_parts = []
        rng_description = []
        z_description = None

    z_a = torch.cat(z_a_parts, dim=0).detach().requires_grad_(True)
    z_b = torch.cat(z_b_parts, dim=0).detach().requires_grad_(True)
    recommendation_targets = recommendation_label_vector(batch, recommendation_targets_np, args, device)
    loss, vic, adv_loss, stats_a, stats_b, extra = compute_latent_loss(
        z_a,
        z_b,
        expander,
        adversary,
        args,
        amp_enabled,
        z_description=z_description,
        description_positions=description_positions,
        recommendation_targets=recommendation_targets,
    )
    scaler.scale(loss).backward()

    z_a_grads = [grad.detach().clone() for grad in z_a.grad.unbind(0)]
    z_b_grads = [grad.detach().clone() for grad in z_b.grad.unbind(0)]
    if z_description is not None and z_description.grad is not None:
        z_description_grads = [grad.detach().clone() for grad in z_description.grad.unbind(0)]
    else:
        z_description_grads = []
    metrics = make_metrics(loss, vic, adv_loss, stats_a, stats_b, batch, extra)

    del z_a, z_b, z_a_parts, z_b_parts, z_description, z_description_parts, loss, vic, adv_loss, stats_a, stats_b
    replay_latent_grads(batch["view_a"], z_a_grads, rng_a, model, device, amp_enabled, pin_transfer)
    del z_a_grads, rng_a
    replay_latent_grads(batch["view_b"], z_b_grads, rng_b, model, device, amp_enabled, pin_transfer)
    del z_b_grads, rng_b
    if z_description_grads:
        replay_latent_grads(
            description_views,
            z_description_grads,
            rng_description,
            model,
            device,
            amp_enabled,
            pin_transfer,
        )
        del z_description_grads, rng_description
    finish_optimizer_step(model, optimizer, scaler, args)
    return metrics


def run_training_batch(
    batch,
    model,
    expander,
    adversary,
    optimizer,
    scaler,
    args,
    device,
    amp_enabled,
    pin_transfer,
    description_bank=None,
    description_rng=None,
    recommendation_targets_np=None,
    cache_dtype=np.dtype("float16"),
):
    if description_rng is None:
        description_rng = np.random.default_rng(args.seed)
    if args.backward_mode == "standard":
        return run_training_batch_standard(
            batch,
            model,
            expander,
            adversary,
            optimizer,
            scaler,
            args,
            device,
            amp_enabled,
            pin_transfer,
            description_bank,
            description_rng,
            recommendation_targets_np,
            cache_dtype,
        )
    if args.backward_mode == "recompute":
        return run_training_batch_recompute(
            batch,
            model,
            expander,
            adversary,
            optimizer,
            scaler,
            args,
            device,
            amp_enabled,
            pin_transfer,
            description_bank,
            description_rng,
            recommendation_targets_np,
            cache_dtype,
        )
    if args.backward_mode == "split_recompute":
        return run_training_batch_split_recompute(
            batch,
            model,
            expander,
            adversary,
            optimizer,
            scaler,
            args,
            device,
            amp_enabled,
            pin_transfer,
            description_bank,
            description_rng,
            recommendation_targets_np,
            cache_dtype,
        )
    raise ValueError(f"Unknown backward mode: {args.backward_mode}")


def grl_lambda_at(global_step, steps_per_epoch, args):
    """GRL strength schedule: hold at 0 during warmup so the encoder learns pure
    VICReg (and the probe warms up), then linearly ramp to args.grl_lambda.

    warmup/ramp are measured in epochs. ramp <= 0 means a hard switch to full
    strength right after warmup.
    """
    steps_per_epoch = max(1, steps_per_epoch)
    progress_epochs = global_step / steps_per_epoch
    warmup = args.grl_warmup_epochs
    ramp = args.grl_ramp_epochs
    if progress_epochs < warmup:
        return 0.0
    if ramp <= 0:
        return args.grl_lambda
    frac = (progress_epochs - warmup) / ramp
    return args.grl_lambda if frac >= 1.0 else args.grl_lambda * frac


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
    model_cls = HierarchicalLatentArrayMLP if args.encoder_arch == "hierarchical" else LatentArrayMLP
    model = model_cls(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        num_heads=args.num_heads,
        dropout=args.dropout,
        output_dim=args.output_dim,
        reduce_hidden=args.reduce_hidden,
    ).to(device)
    expander = None
    if args.vicreg_scope == "game":
        expander = GameCentroidExpander(
            input_dim=model.output_dim,
            hidden_dims=args.expander_hidden,
            output_dim=args.expander_dim,
            dropout=args.expander_dropout,
        ).to(device)
    sentiment_head = load_mlp4_a_sentiment_head(args.sst_checkpoint, map_location=device).to(device)
    adversary = SentimentAdversarialLoss(
        sentiment_head,
        input_dim=model.output_dim,
        probe_hidden=args.probe_hidden,
        probe_dim=1024,
        grl_lambda=args.grl_lambda,
    ).to(device)
    # The probe is a learnable adversary on the head side, so it must be optimized
    # too. The frozen SST head has requires_grad=False and is filtered out.
    trainable = list(model.parameters())
    if expander is not None:
        trainable += list(expander.parameters())
    trainable += [p for p in adversary.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, expander, adversary, optimizer


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
        batch = prepare_batch(
            h5,
            batch_indices,
            args.sample_fraction,
            rng,
            cache_dtype,
            args.pin_cache,
            max_view_sentences=args.max_view_sentences,
        )

    model, expander, adversary, optimizer = build_training_components(args, input_dim, device)
    model.train()
    if expander is not None:
        expander.train()
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
        "vicreg_scope": args.vicreg_scope,
        "expander_dim": args.expander_dim,
        "compact_variance_weight": args.compact_variance_weight,
        "compact_covariance_weight": args.compact_covariance_weight,
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
        f"vicreg_scope={args.vicreg_scope} expander_dim={args.expander_dim} "
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
            expander,
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


def read_history(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    columns = [column.lstrip("\ufeff") for column in lines[0].split("\t")]
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        row = {}
        for column, value in zip(columns, values):
            try:
                parsed = float(value)
                row[column] = int(parsed) if parsed.is_integer() and column in {"epoch", "global_step"} else parsed
            except ValueError:
                row[column] = value
        rows.append(row)
    return rows


def merge_resume_history(history_rows, resume_epoch, global_step, metrics):
    if resume_epoch <= 0 or not metrics:
        return history_rows
    rows = [row for row in history_rows if int(row.get("epoch", -1)) != int(resume_epoch)]
    rows.append({"epoch": int(resume_epoch), "global_step": int(global_step), **metrics})
    return sorted(rows, key=lambda row: int(row.get("epoch", 0)))


def write_manifest(path, status, args, epoch, step, metrics=None, error=None):
    train_game_indices = getattr(args, "train_game_indices", None)
    payload = {
        "status": status,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": epoch,
        "step": step,
        "input_h5": str(Path(args.input_h5).resolve()),
        "checkpoint_out": str(Path(args.checkpoint_out).resolve()),
        "sample_fraction": args.sample_fraction,
        "batch_size": args.batch_size,
        "vicreg_scope": args.vicreg_scope,
        "expander_dim": args.expander_dim,
        "compact_variance_weight": args.compact_variance_weight,
        "compact_covariance_weight": args.compact_covariance_weight,
        "max_batch_sentences": args.max_batch_sentences,
        "max_view_sentences": args.max_view_sentences,
        "cache_mode": args.cache_mode,
        "backward_mode": args.backward_mode,
        "game_order": args.game_order,
        "description_align_weight": args.description_align_weight,
        "description_mse_weight": args.description_mse_weight,
        "description_cache": str(Path(args.description_cache).resolve()),
        "adversary_weight": args.adversary_weight,
        "recommendation_decorr_weight": args.recommendation_decorr_weight,
        "recommendation_target_transform": args.recommendation_target_transform,
        "train_game_count": int(args.train_game_count),
        "train_game_seed": int(args.train_game_seed),
        "train_game_anchor_appids": parse_string_list(args.train_game_anchor_appids),
        "train_game_indices": [] if train_game_indices is None else [int(i) for i in train_game_indices],
        "train_game_appids": list(getattr(args, "train_game_appids", [])),
        "metrics": metrics or {},
        "error": error,
    }
    atomic_text_write(json.dumps(payload, ensure_ascii=False, indent=2), path)


def should_run_probe(epoch: int, args) -> bool:
    every = int(getattr(args, "probe_every", 0))
    if every <= 0:
        return False
    start = max(1, int(getattr(args, "probe_start_epoch", 1)))
    return epoch >= start and (epoch - start) % every == 0


def run_dual_probe(model, args, device, epoch, global_step, probe_rows):
    """Periodic full validation on the current encoder.

    This replaces the old dual_probe path. It measures the sweep tasks we care
    about during training: sentiment probe, recommendation probe, held-out
    anchor TAG generalization, and real-text TAG/cosine behavior. Failures are
    caught so a probe never kills a training run.
    """
    try:
        from VICReg_review.train_tag_probe import extract_features, pool_features
        from VICReg_review.run_data_view_sweep import recommendation_probe, sentiment_r2
        from VICReg_review import text_variant_eval
    except Exception:
        from train_tag_probe import extract_features, pool_features  # type: ignore
        from run_data_view_sweep import recommendation_probe, sentiment_r2  # type: ignore
        import text_variant_eval  # type: ignore
    try:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                feats, names = extract_features(
                    model,
                    args.input_h5,
                    args.probe_sample_fraction,
                    args.probe_feature_views,
                    args.seed,
                    args.cache_dtype,
                    device,
                    args.amp and device.type == "cuda",
                )
        finally:
            if was_training:
                model.train()

        eval_dir = Path(args.probe_history_tsv).resolve().parent
        eval_args = SimpleNamespace(
            h5=Path(args.input_h5),
            out_dir=eval_dir,
            device=args.device,
            seed=args.seed,
            probe_folds=args.probe_folds,
            text_variant_dir=args.text_variant_dir,
            text_variant_cache=args.text_variant_cache,
            rebuild_text_variant_cache=args.rebuild_text_variant_cache,
            text_variant_feature_views=args.text_variant_feature_views,
            text_variant_sample_fraction=args.text_variant_sample_fraction,
            tag_text_split_json=args.tag_text_split_json,
            tag_text_train_frac=args.tag_text_train_frac,
            tag_text_val_frac=args.tag_text_val_frac,
            tag_text_split_seed=args.tag_text_split_seed,
            tag_text_threshold_steps=args.tag_text_threshold_steps,
            local_model=args.text_variant_local_model,
            embed_batch_size=args.text_variant_embed_batch_size,
            max_text_sentences=args.text_variant_max_sentences,
            eval_feature_views=args.text_variant_feature_views,
            eval_sample_fraction=args.text_variant_sample_fraction,
            amp_eval=args.amp and device.type == "cuda",
        )
        X_stats = pool_features(feats, "stats").astype(np.float32)
        report = {
            "epoch": epoch,
            "global_step": global_step,
            "sentiment_probe": sentiment_r2(eval_args, X_stats, names),
            "recommendation_probe": recommendation_probe(eval_args, X_stats, names),
            "text_variant_eval": text_variant_eval.evaluate(eval_args, model, feats, names, eval_dir),
        }
    except BaseException as exc:  # noqa: BLE001
        print(f"[probe] epoch={epoch} skipped: {type(exc).__name__}: {exc}", flush=True)
        return

    text_eval = report.get("text_variant_eval") or {}
    anchor_test = (text_eval.get("tag_generalization") or {}).get("anchor_test") or {}
    real_text_tag = text_eval.get("real_text_tag") or {}
    sentiment = report.get("sentiment_probe") or {}
    reco = report.get("recommendation_probe") or {}
    row = {
        "epoch": epoch,
        "global_step": global_step,
        "sentiment_r2": sentiment.get("r2"),
        "sentiment_pearson": sentiment.get("pearson"),
        "recommendation_pearson": reco.get("pearson_mean"),
        "recommendation_mae": reco.get("mae_mean"),
        "anchor_test_tag_micro_f1": anchor_test.get("micro_f1"),
        "anchor_test_tag_recall": anchor_test.get("recall"),
    }
    for variant in ("positive", "neutral", "negative"):
        payload = real_text_tag.get(variant) or {}
        metrics = payload.get("variant") or {}
        row[f"{variant}_tag_micro_f1"] = metrics.get("micro_f1")
        row[f"{variant}_tag_recall"] = metrics.get("recall")
        row[f"{variant}_drop_micro_f1"] = payload.get("drop_micro_f1")
        row[f"{variant}_drop_recall"] = payload.get("drop_recall")
    probe_rows.append(row)
    write_history(probe_rows, args.probe_history_tsv)
    jsonl_path = Path(args.probe_history_tsv).with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(
        f"[probe] epoch={epoch} anchor_tag_f1={row['anchor_test_tag_micro_f1']} "
        f"sent_r2={row['sentiment_r2']} reco_pearson={row['recommendation_pearson']}",
        flush=True,
    )


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dtype = np.dtype(args.cache_dtype)
    resume_checkpoint = None
    probe_rows = []

    with h5py.File(args.input_h5, "r") as h5:
        validate_training_h5(h5, args.input_h5)
        num_games = int(h5["game_names"].shape[0])
        input_dim = int(h5.attrs["input_dim"])
        total_sentences = int(h5["vectors"].shape[0])
        h5_appids = [decode_name(x) for x in h5["appids"][:]] if "appids" in h5 else [
            decode_name(x).split("_", 1)[0] for x in h5["game_names"][:]
        ]
        train_game_indices = resolve_train_game_indices(args, h5)
        effective_num_games = len(train_game_indices) if train_game_indices is not None else num_games
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
                    game_indices=train_game_indices,
                )
            )
        else:
            default_steps = math.ceil(effective_num_games / args.batch_size)
    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = default_steps

    description_bank = None
    if args.description_align_weight > 0 or args.description_mse_weight > 0:
        description_bank = load_description_bank(args, h5_appids)
        if args.train_game_indices is not None:
            keep = set(int(i) for i in args.train_game_indices)
            description_bank = [items if index in keep else [] for index, items in enumerate(description_bank)]
    recommendation_targets = None
    if args.recommendation_decorr_weight > 0:
        recommendation_targets = load_recommendation_targets(args, num_games)
        if args.train_game_indices is not None:
            mask = np.ones_like(recommendation_targets, dtype=bool)
            mask[np.asarray(args.train_game_indices, dtype=np.int64)] = False
            recommendation_targets[mask] = np.nan
    description_rng = np.random.default_rng(args.seed + 704_971)

    model, expander, adversary, optimizer = build_training_components(args, input_dim, device)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pin_transfer = args.pin_cache and device.type == "cuda"
    history_rows = []
    best_loss = float("inf")
    global_step = 0
    start_epoch = 1
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if expander is not None:
            expander_state = resume_checkpoint.get("expander_state_dict")
            if expander_state is not None:
                incompatible = expander.load_state_dict(expander_state, strict=False)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    print(
                        "resume expander state loaded with differences: "
                        f"missing={list(incompatible.missing_keys)} "
                        f"unexpected={list(incompatible.unexpected_keys)}",
                        flush=True,
                    )
            else:
                print("resume checkpoint has no expander_state_dict; initialized a fresh expander.", flush=True)
        incompatible = adversary.load_state_dict(resume_checkpoint["adversary_state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "resume adversary state loaded with differences: "
                f"missing={list(incompatible.missing_keys)} "
                f"unexpected={list(incompatible.unexpected_keys)}",
                flush=True,
            )
        if args.reset_optimizer_on_resume:
            print("resume optimizer state skipped by --reset-optimizer-on-resume.", flush=True)
        elif "optimizer_state_dict" in resume_checkpoint:
            try:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            except ValueError as exc:
                print(f"resume optimizer state skipped: {exc}", flush=True)
        start_epoch = int(resume_checkpoint.get("epoch") or 0) + 1
        global_step = int(resume_checkpoint.get("global_step") or 0)
        metrics = resume_checkpoint.get("metrics") or {}
        if "loss" in metrics:
            best_loss = float(metrics["loss"])
        history_rows = merge_resume_history(
            read_history(args.history_tsv),
            int(resume_checkpoint.get("epoch") or 0),
            global_step,
            metrics,
        )
        print(
            f"resumed checkpoint={args.resume_checkpoint} "
            f"from_epoch={start_epoch - 1} global_step={global_step}",
            flush=True,
        )

    print(
        f"device={device} games={num_games} sentences={total_sentences} "
        f"train_games={effective_num_games} "
        f"batch_size={args.batch_size} steps_per_epoch={args.steps_per_epoch} "
        f"max_batch_sentences={args.max_batch_sentences} "
        f"max_view_sentences={args.max_view_sentences} "
        f"cache_mode={args.cache_mode} backward_mode={args.backward_mode} "
        f"prefetch_batches={args.prefetch_batches} "
        f"sample_fraction={args.sample_fraction} game_order={args.game_order}",
        flush=True,
    )
    expander_params = sum(p.numel() for p in expander.parameters()) if expander is not None else 0
    print(
        f"model={model.__class__.__name__} params={sum(p.numel() for p in model.parameters())} "
        f"vicreg_scope={args.vicreg_scope} expander_params={expander_params} "
        f"expander_dim={args.expander_dim if expander is not None else 0}",
        flush=True,
    )
    print(
        f"grl_schedule: lambda_target={args.grl_lambda} warmup_epochs={args.grl_warmup_epochs} "
        f"ramp_epochs={args.grl_ramp_epochs} (full GRL at epoch "
        f"{args.grl_warmup_epochs + args.grl_ramp_epochs:g})",
        flush=True,
    )

    executor = None
    next_epoch_future = None
    if args.cache_mode == "full":
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        next_epoch_future = executor.submit(
            prepare_epoch_batches,
            args.input_h5,
            start_epoch,
            args.batch_size,
            args.steps_per_epoch,
            args.sample_fraction,
            args.seed,
            cache_dtype,
            args.pin_cache,
            args.game_order,
            args.max_batch_sentences,
            args.train_game_indices,
        )

    last_metrics = None
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            if expander is not None:
                expander.train()
            epoch_sums = {}
            epoch_batches, next_epoch_future = iter_epoch(args, epoch, next_epoch_future, executor, cache_dtype)

            for batch_index, batch in enumerate(epoch_batches, start=1):
                current_grl = grl_lambda_at(global_step, args.steps_per_epoch, args)
                adversary.grl.lambda_ = current_grl
                metrics = run_training_batch(
                    batch,
                    model,
                    expander,
                    adversary,
                    optimizer,
                    scaler,
                    args,
                    device,
                    amp_enabled,
                    pin_transfer,
                    description_bank=description_bank,
                    description_rng=description_rng,
                    recommendation_targets_np=recommendation_targets,
                    cache_dtype=cache_dtype,
                )
                metrics["grl_lambda"] = current_grl
                global_step += 1
                last_metrics = metrics
                for key, value in metrics.items():
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + value

                if batch_index == 1 or batch_index % args.log_every == 0:
                    compact_msg = ""
                    if "compact_variance" in metrics:
                        compact_msg = (
                            f" cvar={metrics['compact_variance']:.4f} "
                            f"ccov={metrics['compact_covariance']:.4f}"
                        )
                    extra_msg = ""
                    if "description_align" in metrics:
                        extra_msg += (
                            f" desc_align={metrics['description_align']:.4f} "
                            f"desc_mse={metrics['description_mse']:.4f}"
                        )
                    if "recommendation_decorr" in metrics:
                        extra_msg += f" reco_decorr={metrics['recommendation_decorr']:.4f}"
                    print(
                        f"epoch={epoch:03d} step={batch_index:04d}/{args.steps_per_epoch} "
                        f"global={global_step} grl={current_grl:.3f} loss={metrics['loss']:.4f} "
                        f"vic={metrics['vicreg']:.4f} inv={metrics['invariance']:.4f} "
                        f"var={metrics['variance']:.4f} cov={metrics['covariance']:.4f}{compact_msg} "
                        f"{extra_msg} "
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
                "expander_state_dict": expander.state_dict() if expander is not None else None,
                "adversary_state_dict": adversary.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "resumed_from": str(Path(args.resume_checkpoint).resolve()) if args.resume_checkpoint else None,
                "args": vars(args),
                "metrics": averaged,
                "model_class": model.__class__.__name__,
                "vicreg_scope": args.vicreg_scope,
                "num_latents": args.num_latents,
                "latent_dim": args.latent_dim,
                "output_dim": args.output_dim,
                "expander_dim": args.expander_dim if expander is not None else None,
                "expander_hidden": tuple(args.expander_hidden) if expander is not None else None,
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

            if should_run_probe(epoch, args):
                run_dual_probe(model, args, device, epoch, global_step, probe_rows)

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
    parser.add_argument("--probe-every", type=int, default=1,
                        help="Run the aligned full probe every N epochs after --probe-start-epoch. "
                             "0 = off (e.g. for smoke tests).")
    parser.add_argument("--probe-start-epoch", type=int, default=3,
                        help="First epoch that may run the periodic full probe.")
    parser.add_argument("--probe-feature-views", type=int, default=2,
                        help="Views per game when extracting probe features (fewer = faster probe).")
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--probe-sample-fraction", type=float, default=0.6)
    parser.add_argument("--probe-history-tsv", default=str(DEFAULT_HEADS_DIR / "dual_probe_history.tsv"))
    parser.add_argument("--text-variant-dir", default=None)
    parser.add_argument("--text-variant-cache", default=None)
    parser.add_argument("--rebuild-text-variant-cache", action="store_true")
    parser.add_argument("--text-variant-feature-views", type=int, default=4)
    parser.add_argument("--text-variant-sample-fraction", type=float, default=1.0)
    parser.add_argument("--text-variant-local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--text-variant-embed-batch-size", type=int, default=32)
    parser.add_argument("--text-variant-max-sentences", type=int, default=4096)
    parser.add_argument("--tag-text-split-json", default=None)
    parser.add_argument("--tag-text-train-frac", type=float, default=0.7)
    parser.add_argument("--tag-text-val-frac", type=float, default=0.15)
    parser.add_argument("--tag-text-split-seed", type=int, default=20260627)
    parser.add_argument("--tag-text-threshold-steps", type=int, default=33)
    parser.add_argument("--smoke-result-json", default=str(DEFAULT_SMOKE_RESULT))
    parser.add_argument("--resume-checkpoint", default=None,
                        help="Resume model, adversary, optimizer, epoch, and global step from this checkpoint.")
    parser.add_argument("--reset-optimizer-on-resume", action="store_true",
                        help="Load model/adversary/expander weights from --resume-checkpoint but start a fresh optimizer.")
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
    parser.add_argument(
        "--max-view-sentences",
        type=int,
        default=0,
        help=(
            "Optional cap on sampled sentences for each single-game training view. "
            "This prevents rare ultra-long games from OOMing the attention block."
        ),
    )
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument(
        "--train-game-count",
        type=int,
        default=0,
        help="If >0, train on a fixed subset of this many H5 games while keeping original H5 indices.",
    )
    parser.add_argument(
        "--train-game-seed",
        type=int,
        default=20260626,
        help="Seed for the fixed training-game subset used by --train-game-count.",
    )
    parser.add_argument(
        "--train-game-anchor-appids",
        default="1086940,1091500,1385380",
        help="Comma-separated appids that are forced into every fixed training subset.",
    )
    parser.add_argument("--cache-mode", choices=["queue", "full"], default="queue")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--pin-cache", action="store_true")
    parser.add_argument(
        "--backward-mode",
        choices=["recompute", "split_recompute", "standard"],
        default="recompute",
        help=(
            "recompute caches final latents, then replays one view at a time. "
            "split_recompute caches the latent-array stem and replays only the sentence->latent stem, "
            "which keeps full windows while reducing long-sequence backward memory."
        ),
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

    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument(
        "--encoder-arch",
        choices=["latent_mlp", "hierarchical"],
        default="latent_mlp",
        help=(
            "latent_mlp: current single cross-attention + per-latent funnel. "
            "hierarchical: cross-attention, self-attention, then attentional width reductions."
        ),
    )
    parser.add_argument("--output-dim", type=int, default=18,
                        help="Final per-latent code width after the reduction funnel.")
    parser.add_argument("--reduce-hidden", type=parse_int_list, default=(128, 64, 32),
                        help="Comma-separated hidden widths between latent-dim and output-dim, e.g. 128,64,32.")
    parser.add_argument("--probe-hidden", type=int, default=256,
                        help="Hidden width of the adversary up-projection probe (output_dim -> probe_hidden -> 1024).")
    parser.add_argument("--num-latents", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--compact-variance-weight", type=float, default=0.0,
                        help="Optional auxiliary variance pressure directly on the 18-d game centroid.")
    parser.add_argument("--compact-covariance-weight", type=float, default=0.0,
                        help="Optional auxiliary covariance pressure directly on the 18-d game centroid.")
    parser.add_argument(
        "--vicreg-scope",
        choices=["game", "slot"],
        default="game",
        help=(
            "game: mean-pool latent slots to one centroid per game, run invariance on "
            "centroids and variance/covariance on an expanded centroid. slot: legacy "
            "VICReg over (batch*num_latents, output_dim)."
        ),
    )
    parser.add_argument("--expander-dim", type=int, default=1024,
                        help="High-dimensional projection width for game-level variance/covariance.")
    parser.add_argument("--expander-hidden", type=parse_int_list, default=(128, 512),
                        help="Comma-separated hidden widths for centroid expander, e.g. 128,512.")
    parser.add_argument("--expander-dropout", type=float, default=0.0)
    parser.add_argument("--description-align-weight", type=float, default=0.0,
                        help="InfoNCE weight aligning description-text centroids to same-game review centroids.")
    parser.add_argument("--description-mse-weight", type=float, default=0.0,
                        help="Auxiliary MSE weight between description centroids and same-game review centroids.")
    parser.add_argument("--description-align-temperature", type=float, default=0.07)
    parser.add_argument("--description-dir", default=str(DEFAULT_DESCRIPTION_DIR))
    parser.add_argument("--description-cache", default=str(DEFAULT_DESCRIPTION_CACHE))
    parser.add_argument("--overwrite-description-cache", action="store_true")
    parser.add_argument("--description-include-extra-cases", action=argparse.BooleanOptionalAction, default=True,
                        help="Include local full-text Cyberpunk/AO neutral/positive/negative cases in the cache.")
    parser.add_argument("--description-max-sentences", type=int, default=512)
    parser.add_argument("--description-embed-batch-size", type=int, default=16)
    parser.add_argument("--description-local-model", default=None)
    parser.add_argument("--recommendation-decorr-weight", type=float, default=0.0,
                        help="Minimize squared linear correlation between compact game centroids and Steam positive rate.")
    parser.add_argument("--recommendation-reviews-dir", default=None)
    parser.add_argument("--recommendation-label-min-length", type=int, default=0)
    parser.add_argument("--recommendation-min-label-count", type=int, default=10)
    parser.add_argument("--recommendation-target-transform", choices=["identity", "logit"], default="logit")
    parser.add_argument("--adversary-weight", type=float, default=10.0,
                        help="Sweep (TAG_PROBE_RESULTS.md) found 10 maximizes content/sentiment "
                             "selectivity (gap +0.439); weight 1 is too weak, >=20 over-regularizes.")
    parser.add_argument("--grl-lambda", type=float, default=1.0,
                        help="Target GRL strength (reached after warmup + ramp).")
    parser.add_argument("--grl-warmup-epochs", type=float, default=5.0,
                        help="Epochs to hold GRL at 0 so the encoder learns pure VICReg first. 0 = on from step 1.")
    parser.add_argument("--grl-ramp-epochs", type=float, default=10.0,
                        help="Epochs to linearly ramp GRL from 0 to --grl-lambda after warmup. 0 = hard switch.")

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
