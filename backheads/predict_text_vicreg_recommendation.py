"""Predict recommendation rates from text via the frozen VICReg encoder."""

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

from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features  # noqa: E402

DEFAULT_CHECKPOINT = SCRIPT_DIR / "heads" / "recommendation_vicreg_linear_probe.pt"
DEFAULT_H5 = ROOT / "VICReg_review" / "h5" / "game_review_cleaned_3_sentences.h5"


def split_text(text: str, max_sentences: int) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


def predict_rates(checkpoint: dict, feature: np.ndarray) -> np.ndarray:
    mean = checkpoint["feature_mean"].astype(np.float32)
    std = np.maximum(checkpoint["feature_std"].astype(np.float32), 1e-6)
    normalized = (feature.astype(np.float32) - mean) / std
    value = float(normalized @ checkpoint["coef"].astype(np.float32) + float(checkpoint["intercept"]))
    if checkpoint.get("target_transform") == "logit":
        positive = 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))
    else:
        positive = np.clip(value, 0.0, 1.0)
    return np.asarray([positive, 1.0 - positive], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_files", nargs="+", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--h5", default=DEFAULT_H5, type=Path)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-sentences", default=4096, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    import h5py

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "vicreg_linear_recommendation_probe":
        raise ValueError(f"{args.checkpoint} is not a vicreg_linear_recommendation_probe.")

    embedder = LocalEmbedder(
        args.local_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    device = torch.device(embedder.device)
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder_path = Path(checkpoint["encoder_checkpoint"])
    if not encoder_path.is_absolute():
        encoder_path = ROOT / encoder_path
    encoder, _, _, _ = load_frozen_encoder(encoder_path, input_dim, device)
    encoder.float().eval()

    views = max(1, int(checkpoint.get("feature_views") or 4))
    fraction = float(checkpoint.get("sample_fraction") or 0.6)
    pool = checkpoint.get("pool", "stats")

    print("file\tsentences\tpositive_rate\tnegative_rate")
    for path in args.text_files:
        text = path.read_text(encoding="utf-8")
        sentences = split_text(text, args.max_sentences)
        if not sentences:
            raise ValueError(f"{path} contains no text.")
        vectors = np.asarray(embedder.embed(sentences), dtype=np.float32)
        vt = torch.from_numpy(vectors).to(device)
        rng = np.random.default_rng(args.seed)
        codes = []
        with torch.no_grad():
            for _ in range(views):
                if vt.shape[0] > 2:
                    size = max(1, int(np.ceil(vt.shape[0] * fraction)))
                    indices = np.sort(rng.choice(vt.shape[0], size=size, replace=False))
                    sub = vt[indices]
                else:
                    sub = vt
                code = encoder(sub.unsqueeze(0).float(), key_padding_mask=None)
                codes.append(code.squeeze(0).float())
        feats = torch.stack(codes, dim=0).mean(dim=0).cpu().numpy()
        pooled = pool_features(feats[None, ...], pool)[0].astype(np.float32)
        rates = predict_rates(checkpoint, pooled)
        print(f"{path}\t{len(sentences)}\t{rates[0]:.4f}\t{rates[1]:.4f}")


if __name__ == "__main__":
    main()
