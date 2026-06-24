"""Train a head that predicts Steam positive/negative recommendation rates.

Input features come from the game-review H5 built by VICReg_review. Labels come
from the raw Steam review CSV ``recommend`` column:

    Recommended     -> positive
    Not Recommended -> negative

The saved head outputs two probabilities: [positive_rate, negative_rate].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backheads.model import RecommendationRateHead  # noqa: E402

DEFAULT_H5 = ROOT / "VICReg_review" / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_REVIEWS_DIR = (
    ROOT
    / "game_review_data"
    / "Steam Games Metadata and Player Reviews (2020–2024"
    / "Game Reviews"
)
DEFAULT_CACHE = SCRIPT_DIR / "heads" / "recommendation_features_mean_std.npz"
DEFAULT_OUT = SCRIPT_DIR / "heads" / "recommendation_head.pt"
DEFAULT_REPORT = SCRIPT_DIR / "heads" / "recommendation_head_report.json"

POSITIVE_VALUE = "recommended"
NEGATIVE_VALUE = "not recommended"


@dataclass
class LabelRow:
    game_name: str
    appid: str
    title: str
    positive_count: int
    negative_count: int

    @property
    def total(self) -> int:
        return self.positive_count + self.negative_count

    @property
    def positive_rate(self) -> float:
        return self.positive_count / max(1, self.total)

    @property
    def negative_rate(self) -> float:
        return self.negative_count / max(1, self.total)


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def review_csv_path(reviews_dir: Path, game_name: str) -> Path:
    exact = reviews_dir / f"{game_name}.csv"
    if exact.exists():
        return exact
    appid = game_name.split("_", 1)[0]
    matches = sorted(reviews_dir.glob(f"{appid}_*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No review CSV found for {game_name} in {reviews_dir}")
    raise FileNotFoundError(f"Multiple review CSVs match appid={appid}: {matches[:5]}")


def count_recommendations(csv_path: Path, label_min_length: int = 0) -> tuple[int, int]:
    positive = 0
    negative = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "recommend" not in (reader.fieldnames or []):
            raise KeyError(f"{csv_path} has no 'recommend' column")
        for row in reader:
            if label_min_length > 0 and len(row.get("review") or "") < label_min_length:
                continue
            value = (row.get("recommend") or "").strip().lower()
            if value == POSITIVE_VALUE:
                positive += 1
            elif value == NEGATIVE_VALUE:
                negative += 1
    return positive, negative


def load_labels_for_h5(h5_path: Path, reviews_dir: Path, label_min_length: int, min_label_count: int):
    rows: list[LabelRow] = []
    keep_indices: list[int] = []
    missing: list[str] = []
    with h5py.File(h5_path, "r") as h5:
        game_names = [decode(value) for value in h5["game_names"][:]]
        appids = [decode(value) for value in h5["appids"][:]] if "appids" in h5 else [
            name.split("_", 1)[0] for name in game_names
        ]
        titles = [decode(value) for value in h5["game_titles"][:]] if "game_titles" in h5 else appids

    for index, (game_name, appid, title) in enumerate(zip(game_names, appids, titles)):
        try:
            csv_path = review_csv_path(reviews_dir, game_name)
            pos, neg = count_recommendations(csv_path, label_min_length=label_min_length)
        except Exception as exc:
            missing.append(f"{game_name}: {exc}")
            continue
        if pos + neg < min_label_count:
            missing.append(f"{game_name}: only {pos + neg} labeled reviews")
            continue
        rows.append(LabelRow(game_name, appid, title, pos, neg))
        keep_indices.append(index)
    if not rows:
        raise ValueError("No labeled games aligned between H5 and raw CSVs.")
    return rows, np.asarray(keep_indices, dtype=np.int64), missing


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
    raise ValueError(f"Unknown feature mode: {mode}")


def build_feature_cache(args, rows: list[LabelRow], keep_indices: np.ndarray) -> dict:
    started = time.time()
    feats: list[np.ndarray] = []
    sentence_counts: list[int] = []
    review_counts: list[int] = []

    with h5py.File(args.h5, "r") as h5:
        game_offsets = h5["game_review_offsets"]
        review_offsets = h5["review_offsets"]
        vectors = h5["vectors"]
        for row_number, (label_row, game_index) in enumerate(zip(rows, keep_indices), start=1):
            review_start = int(game_offsets[game_index])
            review_end = int(game_offsets[game_index + 1])
            sentence_start = int(review_offsets[review_start])
            sentence_end = int(review_offsets[review_end])
            game_vectors = vectors[sentence_start:sentence_end]
            feature = summarize_vectors(game_vectors, args.feature_mode)
            feats.append(feature.astype(np.float32))
            sentence_counts.append(sentence_end - sentence_start)
            review_counts.append(review_end - review_start)
            if row_number % 25 == 0 or row_number == len(rows):
                elapsed = time.time() - started
                print(
                    f"features {row_number}/{len(rows)} {label_row.appid} "
                    f"sentences={sentence_end - sentence_start} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.asarray([[row.positive_rate, row.negative_rate] for row in rows], dtype=np.float32)
    totals = np.asarray([row.total for row in rows], dtype=np.int32)
    appids = np.asarray([row.appid for row in rows], dtype=object)
    game_names = np.asarray([row.game_name for row in rows], dtype=object)
    titles = np.asarray([row.title for row in rows], dtype=object)
    positive_counts = np.asarray([row.positive_count for row in rows], dtype=np.int32)
    negative_counts = np.asarray([row.negative_count for row in rows], dtype=np.int32)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.cache.with_suffix(args.cache.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                X=X,
                y=y,
                totals=totals,
                appids=appids,
                game_names=game_names,
                titles=titles,
                positive_counts=positive_counts,
                negative_counts=negative_counts,
                sentence_counts=np.asarray(sentence_counts, dtype=np.int32),
                review_counts=np.asarray(review_counts, dtype=np.int32),
                h5=str(Path(args.h5).resolve()),
                reviews_dir=str(Path(args.reviews_dir).resolve()),
                feature_mode=args.feature_mode,
                label_min_length=int(args.label_min_length),
                min_label_count=int(args.min_label_count),
            )
        tmp_path.replace(args.cache)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"wrote feature cache -> {args.cache}", flush=True)
    return load_feature_cache(args.cache)


def load_feature_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_or_build_cache(args) -> dict:
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
    if missing[:5]:
        print("label skips:", " | ".join(missing[:5]), flush=True)
    if args.cache.exists() and not args.overwrite_cache:
        cache = load_feature_cache(args.cache)
        cache_ok = (
            cache["X"].shape[0] == len(rows)
            and str(cache.get("feature_mode", "")) == args.feature_mode
            and int(cache.get("label_min_length", -1)) == int(args.label_min_length)
            and int(cache.get("min_label_count", -1)) == int(args.min_label_count)
            and str(Path(str(cache.get("h5", ""))).resolve()) == str(Path(args.h5).resolve())
            and str(Path(str(cache.get("reviews_dir", ""))).resolve())
            == str(Path(args.reviews_dir).resolve())
        )
        if cache_ok:
            print(f"loaded feature cache -> {args.cache}", flush=True)
            return cache
        print("cache metadata does not match current args; rebuilding.", flush=True)
    return build_feature_cache(args, rows, keep_indices)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_n = max(1, int(round(n * val_fraction)))
    return perm[val_n:], perm[:val_n]


def kfold_indices(n: int, folds: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    chunks = np.array_split(perm, folds)
    for fold in range(folds):
        val = chunks[fold]
        train = np.concatenate([chunks[i] for i in range(folds) if i != fold])
        yield fold, train, val


def fit_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred_pos = pred[:, 0]
    true_pos = target[:, 0]
    err = pred_pos - true_pos
    centered_pred = pred_pos - pred_pos.mean()
    centered_true = true_pos - true_pos.mean()
    pearson = float(
        (centered_pred * centered_true).sum()
        / (np.linalg.norm(centered_pred) * np.linalg.norm(centered_true) + 1e-12)
    )
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "pearson": pearson,
    }


def train_one_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    save_path: Path | None = None,
) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mean, std = fit_normalizer(X[train_idx])
    Xtr = apply_normalizer(X[train_idx], mean, std)
    Xva = apply_normalizer(X[val_idx], mean, std)
    ytr = y[train_idx].astype(np.float32)
    yva = y[val_idx].astype(np.float32)

    model = RecommendationRateHead(
        input_dim=X.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    best = {"val_loss": float("inf"), "epoch": -1, "state_dict": None}
    no_improve = 0
    for epoch in range(args.max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        total_loss = 0.0
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            logits = model(Xtr_t[idx])
            target = ytr_t[idx]
            kl = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
            pos_mse = F.mse_loss(torch.softmax(logits, dim=-1)[:, 0], target[:, 0])
            loss = kl + args.mse_weight * pos_mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t)
            val_rates = torch.softmax(val_logits, dim=-1)
            val_loss = (
                F.kl_div(F.log_softmax(val_logits, dim=-1), yva_t, reduction="batchmean")
                + args.mse_weight * F.mse_loss(val_rates[:, 0], yva_t[:, 0])
            ).item()
        if val_loss < best["val_loss"] - args.min_delta:
            best = {
                "val_loss": float(val_loss),
                "epoch": int(epoch),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1
        if args.verbose and (epoch % args.log_every == 0 or epoch == args.max_epochs - 1):
            print(
                f"epoch {epoch:03d}: train_loss={total_loss / len(Xtr_t):.5f} "
                f"val_loss={val_loss:.5f} no_improve={no_improve}/{args.patience}",
                flush=True,
            )
        if no_improve >= args.patience:
            break

    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model.predict_rates(Xva_t).cpu().numpy()
    result = {
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        **metrics(pred, yva),
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "recommendation_rate_head",
                "state_dict": best["state_dict"],
                "input_dim": int(X.shape[1]),
                "hidden_dims": tuple(args.hidden_dims),
                "dropout": float(args.dropout),
                "feature_mean": mean,
                "feature_std": std,
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
        result["fold"] = fold
        result["train_games"] = int(len(train_idx))
        result["val_games"] = int(len(val_idx))
        rows.append(result)
        print(
            f"fold {fold}: mae={result['mae']:.4f} rmse={result['rmse']:.4f} "
            f"pearson={result['pearson']:.4f} epoch={result['best_epoch']}",
            flush=True,
        )
    return rows


def summarize_cv(folds: list[dict]) -> dict:
    summary = {}
    for key in ("mae", "rmse", "max_abs_error", "pearson", "val_loss"):
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
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--feature-mode", choices=["mean", "mean_std", "mean_std_extrema"], default="mean_std")
    parser.add_argument("--label-min-length", default=0, type=int)
    parser.add_argument("--min-label-count", default=20, type=int)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--hidden-dims", nargs="+", default=[256, 64], type=int)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--mse-weight", default=5.0, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-epochs", default=500, type=int)
    parser.add_argument("--patience", default=60, type=int)
    parser.add_argument("--min-delta", default=1e-5, type=float)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--final-val-fraction", default=0.15, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", default=25, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = load_or_build_cache(args)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.float32)
    print(
        f"dataset: games={len(X)} feature_dim={X.shape[1]} "
        f"positive_rate_mean={float(y[:, 0].mean()):.4f} std={float(y[:, 0].std()):.4f}",
        flush=True,
    )

    cv_rows = cross_validate(X, y, args)
    cv_summary = summarize_cv(cv_rows)
    print(
        "cv: "
        f"mae={cv_summary['mae_mean']:.4f}+/-{cv_summary['mae_std']:.4f} "
        f"rmse={cv_summary['rmse_mean']:.4f}+/-{cv_summary['rmse_std']:.4f} "
        f"pearson={cv_summary['pearson_mean']:.4f}+/-{cv_summary['pearson_std']:.4f}",
        flush=True,
    )

    train_idx, val_idx = split_indices(len(X), args.final_val_fraction, args.seed + 999)
    final_result = train_one_split(X, y, train_idx, val_idx, args, save_path=args.out)
    print(
        f"final holdout: mae={final_result['mae']:.4f} rmse={final_result['rmse']:.4f} "
        f"pearson={final_result['pearson']:.4f} -> {args.out}",
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
    print(f"report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
