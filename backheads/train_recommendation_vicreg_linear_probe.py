"""Train a linear recommendation-rate probe on frozen VICReg features.

This is the probe to use when the question is:

    after the review text passes through the frozen VICReg encoder, how much of
    the Steam positive-rate signal is still linearly recoverable?

The encoder is frozen. Only a ridge linear readout is fit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backheads.train_recommendation_head import (  # noqa: E402
    DEFAULT_H5,
    DEFAULT_REVIEWS_DIR,
    apply_normalizer,
    atomic_json_write,
    fit_normalizer,
    load_labels_for_h5,
    metrics,
    split_indices,
)
from backheads.train_recommendation_linear_probe import (  # noqa: E402
    fit_ridge_linear,
    inverse_target,
    kfold_indices,
    transform_target,
)
from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features, sample_game_views  # noqa: E402

DEFAULT_CACHE = SCRIPT_DIR / "heads" / "recommendation_vicreg_features.npz"
DEFAULT_OUT = SCRIPT_DIR / "heads" / "recommendation_vicreg_linear_probe.pt"
DEFAULT_REPORT = SCRIPT_DIR / "heads" / "recommendation_vicreg_linear_probe_report.json"


def newest_existing(patterns: list[str]) -> Path | None:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    paths = [path for path in paths if path.is_file()]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def resolve_encoder_checkpoint(value: str | None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    tag_head = newest_existing([
        "VICReg_review/heads/tag_probe_linear*.pt",
        "VICReg_review/heads/**/tag_probe_linear*.pt",
    ])
    if tag_head is not None:
        try:
            payload = torch.load(tag_head, map_location="cpu", weights_only=False)
            encoder_value = payload.get("encoder_checkpoint")
            if encoder_value:
                path = Path(encoder_value)
                if not path.is_absolute():
                    path = ROOT / path
                if path.exists():
                    return path
        except Exception:
            pass

    encoder = newest_existing([
        "VICReg_review/heads/sweep_adv/vicreg_adv*_best*.pt",
        "VICReg_review/heads/gui_run/vicreg_review_h5_best*.pt",
        "VICReg_review/heads/vicreg_review_h5_best*.pt",
    ])
    if encoder is None:
        raise FileNotFoundError("No VICReg encoder checkpoint found.")
    return encoder


@torch.no_grad()
def build_vicreg_feature_cache(args, rows, keep_indices: np.ndarray, encoder_path: Path) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dtype = np.dtype(args.cache_dtype)
    rng = np.random.default_rng(args.seed)
    started = time.time()

    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    print(f"loading frozen VICReg encoder on {device}: {encoder_path}", flush=True)
    encoder, cfg, epoch, global_step = load_frozen_encoder(encoder_path, input_dim, device)
    encoder.float().eval()

    features = []
    with h5py.File(args.h5, "r") as h5:
        for row_number, (label_row, game_index) in enumerate(zip(rows, keep_indices), start=1):
            views = sample_game_views(
                h5,
                int(game_index),
                args.sample_fraction,
                args.feature_views,
                rng,
                cache_dtype,
            )
            codes = []
            for view in views:
                tensor = view.unsqueeze(0).to(device).float()
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    code = encoder(tensor, key_padding_mask=None)
                codes.append(code.squeeze(0).float())
            mean_code = torch.stack(codes, dim=0).mean(dim=0).cpu().numpy()
            pooled = pool_features(mean_code[None, ...], args.pool)[0].astype(np.float32)
            features.append(pooled)
            if row_number % 25 == 0 or row_number == len(rows):
                elapsed = time.time() - started
                print(
                    f"vicreg features {row_number}/{len(rows)} {label_row.appid} "
                    f"dim={pooled.shape[0]} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    X = np.stack(features, axis=0).astype(np.float32)
    y = np.asarray([[row.positive_rate, row.negative_rate] for row in rows], dtype=np.float32)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.cache.with_suffix(args.cache.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                X=X,
                y=y,
                totals=np.asarray([row.total for row in rows], dtype=np.int32),
                appids=np.asarray([row.appid for row in rows], dtype=object),
                game_names=np.asarray([row.game_name for row in rows], dtype=object),
                titles=np.asarray([row.title for row in rows], dtype=object),
                positive_counts=np.asarray([row.positive_count for row in rows], dtype=np.int32),
                negative_counts=np.asarray([row.negative_count for row in rows], dtype=np.int32),
                encoder_checkpoint=str(encoder_path.resolve()),
                encoder_epoch=-1 if epoch is None else int(epoch),
                encoder_global_step=-1 if global_step is None else int(global_step),
                encoder_cfg=np.asarray([repr(cfg)], dtype=object),
                h5=str(Path(args.h5).resolve()),
                reviews_dir=str(Path(args.reviews_dir).resolve()),
                pool=args.pool,
                sample_fraction=float(args.sample_fraction),
                feature_views=int(args.feature_views),
                label_min_length=int(args.label_min_length),
                min_label_count=int(args.min_label_count),
            )
        tmp_path.replace(args.cache)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"wrote VICReg feature cache -> {args.cache}", flush=True)
    return load_feature_cache(args.cache)


def load_feature_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def cache_matches(cache: dict, args, rows, encoder_path: Path) -> bool:
    try:
        return (
            cache["X"].shape[0] == len(rows)
            and str(cache.get("pool", "")) == args.pool
            and abs(float(cache.get("sample_fraction", -1.0)) - float(args.sample_fraction)) < 1e-8
            and int(cache.get("feature_views", -1)) == int(args.feature_views)
            and int(cache.get("label_min_length", -1)) == int(args.label_min_length)
            and int(cache.get("min_label_count", -1)) == int(args.min_label_count)
            and str(Path(str(cache.get("h5", ""))).resolve()) == str(Path(args.h5).resolve())
            and str(Path(str(cache.get("encoder_checkpoint", ""))).resolve()) == str(encoder_path.resolve())
        )
    except Exception:
        return False


def load_or_build_vicreg_cache(args, encoder_path: Path) -> dict:
    rows, keep_indices, missing = load_labels_for_h5(
        args.h5,
        args.reviews_dir,
        label_min_length=args.label_min_length,
        min_label_count=args.min_label_count,
    )
    print(
        f"labels: aligned_games={len(rows)} missing_or_filtered={len(missing)} "
        f"label_min_length={args.label_min_length}",
        flush=True,
    )
    if args.cache.exists() and not args.overwrite_cache:
        cache = load_feature_cache(args.cache)
        if cache_matches(cache, args, rows, encoder_path):
            print(f"loaded VICReg feature cache -> {args.cache}", flush=True)
            return cache
        print("VICReg feature cache metadata does not match current args; rebuilding.", flush=True)
    return build_vicreg_feature_cache(args, rows, keep_indices, encoder_path)


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


def train_one_split(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, args, save_path=None):
    alpha = inner_select_alpha(X, y, train_idx, args)
    mean, std = fit_normalizer(X[train_idx])
    Xtr = apply_normalizer(X[train_idx], mean, std)
    Xva = apply_normalizer(X[val_idx], mean, std)
    ztr = transform_target(y[train_idx], args)
    coef, intercept = fit_ridge_linear(Xtr, ztr, alpha)
    pred_pos = predict_positive(Xva, coef, intercept, args)
    pred = np.stack([pred_pos, 1.0 - pred_pos], axis=1).astype(np.float32)
    result = {"alpha": float(alpha), **metrics(pred, y[val_idx].astype(np.float32))}
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "vicreg_linear_recommendation_probe",
                "coef": coef.astype(np.float32),
                "intercept": float(intercept),
                "feature_mean": mean.astype(np.float32),
                "feature_std": std.astype(np.float32),
                "alpha": float(alpha),
                "target_transform": args.target_transform,
                "logit_eps": float(args.logit_eps),
                "pool": args.pool,
                "sample_fraction": float(args.sample_fraction),
                "feature_views": int(args.feature_views),
                "encoder_checkpoint": str(Path(args.encoder_checkpoint_resolved).resolve()),
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
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--cache", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--pool", choices=["flatten", "mean", "stats"], default="stats")
    parser.add_argument("--sample-fraction", default=0.6, type=float)
    parser.add_argument("--feature-views", default=4, type=int)
    parser.add_argument("--cache-dtype", default="float16")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
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
    encoder_path = resolve_encoder_checkpoint(args.encoder_checkpoint)
    args.encoder_checkpoint_resolved = str(encoder_path)
    cache = load_or_build_vicreg_cache(args, encoder_path)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.float32)
    print(
        f"VICReg linear probe dataset: games={len(X)} feature_dim={X.shape[1]} "
        f"positive_rate_mean={float(y[:, 0].mean()):.4f} std={float(y[:, 0].std()):.4f}",
        flush=True,
    )

    cv_rows = cross_validate(X, y, args)
    cv_summary = summarize_cv(cv_rows)
    print(
        "VICReg linear cv: "
        f"mae={cv_summary['mae_mean']:.4f}+/-{cv_summary['mae_std']:.4f} "
        f"rmse={cv_summary['rmse_mean']:.4f}+/-{cv_summary['rmse_std']:.4f} "
        f"pearson={cv_summary['pearson_mean']:.4f}+/-{cv_summary['pearson_std']:.4f}",
        flush=True,
    )

    train_idx, val_idx = split_indices(len(X), args.final_val_fraction, args.seed + 999)
    final_result = train_one_split(X, y, train_idx, val_idx, args, save_path=args.out)
    print(
        f"VICReg linear final holdout: mae={final_result['mae']:.4f} "
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
            "encoder_checkpoint": str(encoder_path.resolve()),
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cv_folds": cv_rows,
        "cv_summary": cv_summary,
        "final_holdout": final_result,
        "checkpoint": str(Path(args.out).resolve()),
    }
    atomic_json_write(report, args.report)
    print(f"VICReg linear report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
