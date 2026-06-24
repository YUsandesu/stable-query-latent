"""Predict recommendation rates directly from game description/review text files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "game_review_data") not in sys.path:
    sys.path.insert(0, str(ROOT / "game_review_data"))

from backheads.model import RecommendationRateHead  # noqa: E402
from game_review_data.embedding_data import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    CloudEmbedder,
    LocalEmbedder,
)

DEFAULT_CHECKPOINT = SCRIPT_DIR / "heads" / "recommendation_linear_probe.pt"


def split_text(text: str, max_sentences: int) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


def summarize_vectors(vectors: np.ndarray, mode: str) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    if mode == "mean":
        return mean
    if mode == "mean_std":
        return np.concatenate([mean, std], axis=0)
    if mode == "mean_std_extrema":
        return np.concatenate([mean, std, vectors.min(axis=0), vectors.max(axis=0)], axis=0)
    raise ValueError(f"Unknown feature mode in checkpoint: {mode}")


def make_embedder(args):
    if args.backend == "local":
        return LocalEmbedder(
            args.local_model,
            device=args.device,
            batch_size=args.batch_size,
        ), None
    embedder = CloudEmbedder(
        base_url=args.base_url,
        token_file=args.token_file,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_in_flight=args.max_in_flight,
        normalize=False,
    )
    return embedder, embedder.close


def predict_rates(checkpoint: dict, feature: np.ndarray) -> np.ndarray:
    feature_mean = checkpoint["feature_mean"].astype(np.float32)
    feature_std = np.maximum(checkpoint["feature_std"].astype(np.float32), 1e-6)
    normalized = (feature.astype(np.float32) - feature_mean) / feature_std
    if checkpoint.get("kind") == "linear_recommendation_probe":
        value = float(normalized @ checkpoint["coef"].astype(np.float32) + float(checkpoint["intercept"]))
        if checkpoint.get("target_transform") == "logit":
            positive = 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))
        else:
            positive = np.clip(value, 0.0, 1.0)
        return np.asarray([positive, 1.0 - positive], dtype=np.float32)

    model = RecommendationRateHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return model.predict_rates(torch.from_numpy(normalized).unsqueeze(0))[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_files", nargs="+", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", default=256, type=int)
    parser.add_argument("--max-in-flight", default=None, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-sentences", default=4096, type=int)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_mode = str(checkpoint.get("feature_mode", "mean_std"))

    embedder, closer = make_embedder(args)
    try:
        print("file\tsentences\tpositive_rate\tnegative_rate")
        for path in args.text_files:
            text = path.read_text(encoding="utf-8")
            sentences = split_text(text, args.max_sentences)
            if not sentences:
                raise ValueError(f"{path} contains no text.")
            vectors = np.asarray(embedder.embed(sentences), dtype=np.float32)
            feature = summarize_vectors(vectors, feature_mode)
            expected_dim = int(checkpoint.get("input_dim", checkpoint["feature_mean"].shape[0]))
            if feature.shape[0] != expected_dim:
                raise ValueError(
                    f"{path}: feature dim {feature.shape[0]} != checkpoint dim {expected_dim}"
                )
            rates = predict_rates(checkpoint, feature)
            print(f"{path}\t{len(sentences)}\t{rates[0]:.4f}\t{rates[1]:.4f}")
    finally:
        if closer:
            closer()


if __name__ == "__main__":
    main()
