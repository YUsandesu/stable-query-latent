"""Train LatentArrayMLP on Steam game reviews with VICReg + GRL sentiment loss.

Each sample is one game file from game_review_cleaned_3_sentences. For that game,
view A and view B independently sample 60 percent of reviews, flatten their
sentence vectors, and pass both views through the same encoder.
"""

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

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

DEFAULT_INPUT_DIR = PROJECT_ROOT / "game_review_data" / "game_review_cleaned_3_sentences"
DEFAULT_SST_CHECKPOINT = PROJECT_ROOT / "sst" / "heads" / "mlp4_1024_128_32_8_1_best.pt"
DEFAULT_HEADS_DIR = SCRIPT_DIR / "heads"


def _numeric_suffix(value, prefix):
    text = str(value)
    if text.startswith(prefix):
        text = text[len(prefix):]
    try:
        return int(text)
    except ValueError:
        return text


def load_game_review_vectors(path):
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a review_id -> sentences mapping.")

    reviews = []
    review_items = sorted(raw.items(), key=lambda item: _numeric_suffix(item[0], ""))
    for _, sentence_map in review_items:
        if not isinstance(sentence_map, dict):
            continue
        vectors = []
        sentence_items = sorted(
            sentence_map.items(),
            key=lambda item: _numeric_suffix(item[0], "sentence_"),
        )
        for _, payload in sentence_items:
            if not isinstance(payload, dict) or "vector" not in payload:
                continue
            vector = payload["vector"]
            if isinstance(vector, list) and vector:
                vectors.append(vector)
        if vectors:
            reviews.append(torch.tensor(vectors, dtype=torch.float32))

    if not reviews:
        raise ValueError(f"{path} contains no sentence vectors.")
    return reviews


class GameReviewVicRegDataset(Dataset):
    def __init__(
        self,
        input_dir,
        sample_fraction=0.6,
        max_sentences=4096,
        cache_games=1,
        limit_games=0,
    ):
        self.input_dir = Path(input_dir)
        self.files = sorted(self.input_dir.glob("*.json"))
        if limit_games and limit_games > 0:
            self.files = self.files[:limit_games]
        if not self.files:
            raise ValueError(f"No JSON game files found in {self.input_dir}.")
        if not (0.0 < sample_fraction <= 1.0):
            raise ValueError("--sample-fraction must be in (0, 1].")
        self.sample_fraction = float(sample_fraction)
        self.max_sentences = int(max_sentences)
        self.cache_games = max(0, int(cache_games))
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.files)

    def _get_reviews(self, path):
        key = str(path)
        if self.cache_games > 0 and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        reviews = load_game_review_vectors(path)
        if self.cache_games > 0:
            self._cache[key] = reviews
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_games:
                self._cache.popitem(last=False)
        return reviews

    def _sample_view(self, reviews):
        review_count = len(reviews)
        take = max(1, int(math.ceil(review_count * self.sample_fraction)))
        indices = torch.randperm(review_count)[:take].tolist()
        view = torch.cat([reviews[index] for index in indices], dim=0)

        if self.max_sentences > 0 and view.size(0) > self.max_sentences:
            sentence_indices = torch.randperm(view.size(0))[: self.max_sentences]
            view = view[sentence_indices]
        return view

    def __getitem__(self, index):
        path = self.files[index]
        reviews = self._get_reviews(path)
        return {
            "view_a": self._sample_view(reviews),
            "view_b": self._sample_view(reviews),
            "game": path.stem,
            "review_count": len(reviews),
        }


def collate_review_views(batch):
    def pad(key):
        views = [item[key] for item in batch]
        lengths = torch.tensor([view.size(0) for view in views], dtype=torch.long)
        max_length = int(lengths.max().item())
        dim = views[0].size(1)
        padded = views[0].new_zeros((len(views), max_length, dim))
        mask = torch.ones((len(views), max_length), dtype=torch.bool)
        for row, view in enumerate(views):
            length = view.size(0)
            padded[row, :length] = view
            mask[row, :length] = False
        return padded, mask, lengths

    view_a, mask_a, len_a = pad("view_a")
    view_b, mask_b, len_b = pad("view_b")
    return {
        "view_a": view_a,
        "mask_a": mask_a,
        "len_a": len_a,
        "view_b": view_b,
        "mask_b": mask_b,
        "len_b": len_b,
        "games": [item["game"] for item in batch],
        "review_counts": torch.tensor([item["review_count"] for item in batch], dtype=torch.long),
    }


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


def write_manifest(path, status, args, epoch, step, metrics=None, error=None):
    payload = {
        "status": status,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": epoch,
        "step": step,
        "input_dir": str(Path(args.input_dir).resolve()),
        "checkpoint_out": str(Path(args.checkpoint_out).resolve()),
        "sst_checkpoint": str(Path(args.sst_checkpoint).resolve()),
        "sample_fraction": args.sample_fraction,
        "max_sentences": args.max_sentences,
        "metrics": metrics or {},
        "error": error,
    }
    atomic_text_write(json.dumps(payload, ensure_ascii=False, indent=2), path)


def make_loader(args, device):
    dataset = GameReviewVicRegDataset(
        args.input_dir,
        sample_fraction=args.sample_fraction,
        max_sentences=args.max_sentences,
        cache_games=args.cache_games,
        limit_games=args.limit_games,
    )
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_review_views,
    )
    return dataset, loader


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset, loader = make_loader(args, device)
    steps_per_epoch = args.steps_per_epoch or len(loader)

    model = LatentArrayMLP(
        input_dim=args.input_dim,
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
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history_rows = []
    best_loss = float("inf")
    global_step = 0
    pin_memory = device.type == "cuda"

    print(
        f"device={device} games={len(dataset)} batch_size={args.batch_size} "
        f"steps_per_epoch={steps_per_epoch} sample_fraction={args.sample_fraction} "
        f"max_sentences={args.max_sentences}"
    )
    print(f"model=LatentArrayMLP params={sum(p.numel() for p in model.parameters())}")
    print(f"sentiment_head={args.sst_checkpoint}")

    last_metrics = None
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            iterator = iter(loader)
            epoch_sums = {}

            for step in range(1, steps_per_epoch + 1):
                batch, iterator = next_batch(loader, iterator)
                view_a = batch["view_a"].to(device, non_blocking=pin_memory)
                mask_a = batch["mask_a"].to(device, non_blocking=pin_memory)
                view_b = batch["view_b"].to(device, non_blocking=pin_memory)
                mask_b = batch["mask_b"].to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    z_a = model(view_a, key_padding_mask=mask_a)
                    z_b = model(view_b, key_padding_mask=mask_b)
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

                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                metrics = {
                    "loss": float(loss.detach().cpu()),
                    "vicreg": float(vic["loss"].detach().cpu()),
                    "invariance": float(vic["invariance"].detach().cpu()),
                    "variance": float(vic["variance"].detach().cpu()),
                    "covariance": float(vic["covariance"].detach().cpu()),
                    "adversary_entropy_loss": float(adv_loss.detach().cpu()),
                    "sentiment_mean": float((stats_a["sentiment_mean"] + stats_b["sentiment_mean"]).mul(0.5).cpu()),
                    "sentiment_std": float((stats_a["sentiment_std"] + stats_b["sentiment_std"]).mul(0.5).cpu()),
                    "sentiment_entropy": float(
                        (stats_a["sentiment_entropy"] + stats_b["sentiment_entropy"]).mul(0.5).cpu()
                    ),
                    "sentences_a": float(batch["len_a"].float().mean().item()),
                    "sentences_b": float(batch["len_b"].float().mean().item()),
                }
                last_metrics = metrics
                for key, value in metrics.items():
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + value

                if step == 1 or step % args.log_every == 0:
                    game_text = ",".join(batch["games"][:3])
                    print(
                        f"epoch={epoch:03d} step={step:04d}/{steps_per_epoch} "
                        f"loss={metrics['loss']:.4f} vic={metrics['vicreg']:.4f} "
                        f"inv={metrics['invariance']:.4f} var={metrics['variance']:.4f} "
                        f"cov={metrics['covariance']:.4f} adv_entropy={metrics['sentiment_entropy']:.4f} "
                        f"sent_mean={metrics['sentiment_mean']:.4f} games={game_text}"
                    )

            averaged = {key: value / steps_per_epoch for key, value in epoch_sums.items()}
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


def write_history(rows, path):
    if not rows:
        return
    columns = list(rows[0].keys())
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(f"{row[column]:.10g}" if isinstance(row[column], float) else str(row[column]) for column in columns))
    atomic_text_write("\n".join(lines) + "\n", path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--sst-checkpoint", default=str(DEFAULT_SST_CHECKPOINT))
    parser.add_argument("--checkpoint-out", default=str(DEFAULT_HEADS_DIR / "vicreg_review_latent_array_mlp_latest.pt"))
    parser.add_argument("--best-checkpoint-out", default=str(DEFAULT_HEADS_DIR / "vicreg_review_latent_array_mlp_best.pt"))
    parser.add_argument("--history-tsv", default=str(DEFAULT_HEADS_DIR / "vicreg_review_history.tsv"))
    parser.add_argument("--manifest-json", default=str(DEFAULT_HEADS_DIR / "vicreg_review_manifest.json"))
    parser.add_argument("--no-save", action="store_true", help="Run training without writing model checkpoints.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means one pass over game files.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--max-sentences", type=int, default=4096, help="0 disables the sentence cap after review sampling.")
    parser.add_argument("--cache-games", type=int, default=1)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--input-dim", type=int, default=1024)
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
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
