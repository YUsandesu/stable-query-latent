"""Predict Steam positive/negative recommendation rates from cached features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backheads.model import RecommendationRateHead  # noqa: E402

DEFAULT_CACHE = SCRIPT_DIR / "heads" / "recommendation_features_mean_std.npz"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "heads" / "recommendation_linear_probe.pt"


def load_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def predict_rates(checkpoint: dict, X: np.ndarray) -> np.ndarray:
    mean = checkpoint["feature_mean"].astype(np.float32)
    std = np.maximum(checkpoint["feature_std"].astype(np.float32), 1e-6)
    Xn = (X.astype(np.float32) - mean) / std
    if checkpoint.get("kind") == "linear_recommendation_probe":
        z = Xn @ checkpoint["coef"].astype(np.float32) + float(checkpoint["intercept"])
        if checkpoint.get("target_transform") == "logit":
            pos = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        else:
            pos = np.clip(z, 0.0, 1.0)
        return np.stack([pos, 1.0 - pos], axis=1).astype(np.float32)

    model = RecommendationRateHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return model.predict_rates(torch.from_numpy(Xn)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--top", default=20, type=int)
    parser.add_argument("--sort", choices=["pred", "true", "error"], default="error")
    args = parser.parse_args()

    cache = load_cache(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    X = cache["X"].astype(np.float32)
    pred = predict_rates(checkpoint, X)

    true = cache["y"].astype(np.float32)
    error = np.abs(pred[:, 0] - true[:, 0])
    if args.sort == "pred":
        order = np.argsort(-pred[:, 0])
    elif args.sort == "true":
        order = np.argsort(-true[:, 0])
    else:
        order = np.argsort(-error)

    titles = [str(value) for value in cache["titles"]]
    appids = [str(value) for value in cache["appids"]]
    print("appid\ttitle\tpred_positive\tpred_negative\ttrue_positive\tabs_error")
    for index in order[: args.top]:
        print(
            f"{appids[index]}\t{titles[index]}\t{pred[index, 0]:.4f}\t"
            f"{pred[index, 1]:.4f}\t{true[index, 0]:.4f}\t{error[index]:.4f}"
        )


if __name__ == "__main__":
    main()
