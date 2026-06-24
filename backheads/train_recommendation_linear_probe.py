"""Train a linear probe for Steam recommendation-rate recovery.

This differs from train_recommendation_head.py on purpose:

    frozen aggregate features -> one linear readout -> positive rate

There are no hidden layers. The probe answers how much of the Steam positive
rate is linearly recoverable from the cached game-review features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backheads.train_recommendation_head import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_H5,
    DEFAULT_REPORT,
    DEFAULT_REVIEWS_DIR,
    apply_normalizer,
    atomic_json_write,
    fit_normalizer,
    kfold_indices,
    load_or_build_cache,
    metrics,
    split_indices,
)

DEFAULT_OUT = SCRIPT_DIR / "heads" / "recommendation_linear_probe.pt"
DEFAULT_LINEAR_REPORT = SCRIPT_DIR / "heads" / "recommendation_linear_probe_report.json"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit(y: np.ndarray, eps: float) -> np.ndarray:
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def transform_target(y: np.ndarray, args) -> np.ndarray:
    pos = y[:, 0].astype(np.float64)
    if args.target_transform == "logit":
        return logit(pos, args.logit_eps)
    return pos


def inverse_target(z: np.ndarray, args) -> np.ndarray:
    if args.target_transform == "logit":
        return sigmoid(z)
    return np.clip(z, 0.0, 1.0)


def fit_ridge_linear(X: np.ndarray, z: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Fit ridge with an unpenalized intercept using the dual form.

    X is already standardized by the training split. Since X has many more
    columns than rows, solving the n x n dual system is much cheaper and more
    stable than solving a D x D system.
    """

    X = X.astype(np.float64, copy=False)
    z = z.astype(np.float64, copy=False)
    intercept = float(z.mean())
    centered = z - intercept
    gram = X @ X.T
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    dual = np.linalg.solve(gram, centered)
    coef = X.T @ dual
    return coef.astype(np.float32), intercept


def predict_positive(X: np.ndarray, coef: np.ndarray, intercept: float, args) -> np.ndarray:
    z = X.astype(np.float32) @ coef.astype(np.float32) + float(intercept)
    return inverse_target(z.astype(np.float64), args).astype(np.float32)


def inner_select_alpha(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, args) -> float:
    if len(args.alphas) == 1:
        return float(args.alphas[0])

    scores = []
    for alpha in args.alphas:
        fold_mae = []
        for _, inner_train_rel, inner_val_rel in kfold_indices(len(train_idx), args.inner_folds, args.seed + 123):
            inner_train = train_idx[inner_train_rel]
            inner_val = train_idx[inner_val_rel]
            mean, std = fit_normalizer(X[inner_train])
            Xtr = apply_normalizer(X[inner_train], mean, std)
            Xva = apply_normalizer(X[inner_val], mean, std)
            ztr = transform_target(y[inner_train], args)
            coef, intercept = fit_ridge_linear(Xtr, ztr, float(alpha))
            pred_pos = predict_positive(Xva, coef, intercept, args)
            fold_mae.append(float(np.abs(pred_pos - y[inner_val, 0]).mean()))
        scores.append((float(np.mean(fold_mae)), float(alpha)))
    scores.sort(key=lambda item: item[0])
    return scores[0][1]


def train_one_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    save_path: Path | None = None,
) -> dict:
    alpha = inner_select_alpha(X, y, train_idx, args)
    mean, std = fit_normalizer(X[train_idx])
    Xtr = apply_normalizer(X[train_idx], mean, std)
    Xva = apply_normalizer(X[val_idx], mean, std)
    ztr = transform_target(y[train_idx], args)
    coef, intercept = fit_ridge_linear(Xtr, ztr, alpha)
    pred_pos = predict_positive(Xva, coef, intercept, args)
    pred = np.stack([pred_pos, 1.0 - pred_pos], axis=1).astype(np.float32)
    result = {
        "alpha": float(alpha),
        **metrics(pred, y[val_idx].astype(np.float32)),
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "linear_recommendation_probe",
                "coef": coef.astype(np.float32),
                "intercept": float(intercept),
                "feature_mean": mean.astype(np.float32),
                "feature_std": std.astype(np.float32),
                "alpha": float(alpha),
                "target_transform": args.target_transform,
                "logit_eps": float(args.logit_eps),
                "feature_mode": args.feature_mode,
                "label_order": ["positive_rate", "negative_rate"],
                "args": vars(args),
            },
            save_path,
        )
    return result


def cross_validate(X: np.ndarray, y: np.ndarray, args) -> list[dict]:
    rows = []
    for fold, train_idx, val_idx in kfold_indices(len(X), args.folds, args.seed):
        fold_args = argparse.Namespace(**vars(args))
        fold_args.seed = args.seed + fold
        result = train_one_split(X, y, train_idx, val_idx, fold_args, save_path=None)
        result["fold"] = int(fold)
        result["train_games"] = int(len(train_idx))
        result["val_games"] = int(len(val_idx))
        rows.append(result)
        print(
            f"fold {fold}: mae={result['mae']:.4f} rmse={result['rmse']:.4f} "
            f"pearson={result['pearson']:.4f} alpha={result['alpha']}",
            flush=True,
        )
    return rows


def summarize_cv(folds: list[dict]) -> dict:
    summary = {}
    for key in ("mae", "rmse", "max_abs_error", "pearson"):
        values = np.asarray([row[key] for row in folds], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5, type=Path)
    parser.add_argument("--reviews-dir", default=DEFAULT_REVIEWS_DIR, type=Path)
    parser.add_argument("--cache", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--report", default=DEFAULT_LINEAR_REPORT, type=Path)
    parser.add_argument("--feature-mode", choices=["mean", "mean_std", "mean_std_extrema"], default="mean_std")
    parser.add_argument("--label-min-length", default=0, type=int)
    parser.add_argument("--min-label-count", default=20, type=int)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--target-transform", choices=["identity", "logit"], default="logit")
    parser.add_argument("--logit-eps", default=1e-4, type=float)
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        type=float,
    )
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--inner-folds", default=3, type=int)
    parser.add_argument("--final-val-fraction", default=0.15, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = load_or_build_cache(args)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.float32)
    print(
        f"linear probe dataset: games={len(X)} feature_dim={X.shape[1]} "
        f"positive_rate_mean={float(y[:, 0].mean()):.4f} std={float(y[:, 0].std()):.4f}",
        flush=True,
    )

    cv_rows = cross_validate(X, y, args)
    cv_summary = summarize_cv(cv_rows)
    print(
        "linear cv: "
        f"mae={cv_summary['mae_mean']:.4f}+/-{cv_summary['mae_std']:.4f} "
        f"rmse={cv_summary['rmse_mean']:.4f}+/-{cv_summary['rmse_std']:.4f} "
        f"pearson={cv_summary['pearson_mean']:.4f}+/-{cv_summary['pearson_std']:.4f}",
        flush=True,
    )

    train_idx, val_idx = split_indices(len(X), args.final_val_fraction, args.seed + 999)
    final_result = train_one_split(X, y, train_idx, val_idx, args, save_path=args.out)
    print(
        f"linear final holdout: mae={final_result['mae']:.4f} "
        f"rmse={final_result['rmse']:.4f} pearson={final_result['pearson']:.4f} "
        f"alpha={final_result['alpha']} -> {args.out}",
        flush=True,
    )

    report = {
        "dataset": {
            "games": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "positive_rate_mean": float(y[:, 0].mean()),
            "positive_rate_std": float(y[:, 0].std()),
            "total_reviews": int(np.asarray(cache["totals"]).sum()),
            "feature_cache": str(Path(args.cache).resolve()),
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cv_folds": cv_rows,
        "cv_summary": cv_summary,
        "final_holdout": final_result,
        "checkpoint": str(Path(args.out).resolve()),
    }
    atomic_json_write(report, args.report)
    print(f"linear report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
