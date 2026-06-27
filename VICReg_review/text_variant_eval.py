"""Text-variant evaluation for the review VICReg sweep.

Protocol:
* The H5/game feature is the anchor distribution: "description + reviews".
* Train a fresh Ridge TAG probe on anchor features for train games.
* Select alpha and a global decision threshold on anchor validation games.
* First evaluate held-out anchor features to measure normal test-set TAG
  generalization.
* Then evaluate every available positive/neutral/negative/noname real-text
  variant with the same fixed probe to measure text generalization, sentiment
  impact, and whether identity depends on proper-name shortcuts.
* Report cosine from each real-text variant to its anchor.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from VICReg_review.identity_diagnostic import encode_text_centroid, l2_normalize, split_text
from VICReg_review.train_tag_probe import load_labels


VARIANTS = ("positive", "neutral", "negative", "noname")
VARIANT_ALIASES = {
    "positive": ("positive", "postive", "pos"),
    "neutral": ("neutral", "middle", "base"),
    "negative": ("negative", "neg"),
    "noname": ("noname", "no_name", "nameless", "deidentified", "deidentified_names"),
}
DEFAULT_ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


@dataclass(frozen=True)
class VariantRecord:
    appid: str
    name: str
    variant: str
    path: Path


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_h5_names(h5_path: Path) -> tuple[list[str], list[str]]:
    with h5py.File(h5_path, "r") as h5:
        names = [decode(x) for x in h5["game_names"][:]]
        if "appids" in h5:
            appids = [decode(x) for x in h5["appids"][:]]
        else:
            appids = [name.split("_", 1)[0] for name in names]
    return names, appids


def legacy_text_path(root: Path, appid: str, variant: str) -> Path | None:
    legacy = {
        "1091500": {
            "positive": root / "2077_text_postive.txt",
            "neutral": root / "2077_text.txt",
            "negative": root / "2077_text_negative.txt",
            "noname": root / "2077_noname.txt",
        },
        "1385380": {
            "positive": root / "AO_text_postive.txt",
            "neutral": root / "AO_text.txt",
            "negative": root / "AO_text_negative.txt",
            "noname": root / "AO_text_noname.txt",
        },
    }
    path = legacy.get(str(appid), {}).get(variant)
    return path if path is not None and path.exists() else None


def find_variant_path(root: Path, appid: str, variant: str) -> Path | None:
    aliases = VARIANT_ALIASES[variant]
    candidates = []
    for alias in aliases:
        candidates.extend([
            root / str(appid) / f"{alias}.txt",
            root / str(appid) / f"text_{alias}.txt",
            root / f"{appid}_{alias}.txt",
            root / f"{appid}_text_{alias}.txt",
            root / f"{appid}_{alias}_text.txt",
        ])
    for path in candidates:
        if path.exists():
            return path
    return legacy_text_path(root, appid, variant)


def discover_variant_records(root: Path, names: list[str], appids: list[str]) -> list[VariantRecord]:
    if not root.exists():
        return []
    records = []
    for name, appid in zip(names, appids):
        for variant in VARIANTS:
            path = find_variant_path(root, appid, variant)
            if path is not None:
                records.append(VariantRecord(str(appid), name, variant, path))
    return records


def load_or_embed_variant_texts(args, records: list[VariantRecord], cache_path: Path) -> dict:
    source_paths = [str(record.path.resolve()) for record in records]
    source_mtime_ns = [int(record.path.stat().st_mtime_ns) for record in records]
    source_sizes = [int(record.path.stat().st_size) for record in records]
    source_variants = [record.variant for record in records]
    if cache_path.exists() and not getattr(args, "rebuild_text_variant_cache", False):
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = [str(x) for x in data["paths"]] if "paths" in data else []
        cached_mtime_ns = [int(x) for x in data["source_mtime_ns"]] if "source_mtime_ns" in data else []
        cached_sizes = [int(x) for x in data["source_sizes"]] if "source_sizes" in data else []
        cached_variants = [str(x) for x in data["variants"]] if "variants" in data else []
        if (
            cached_paths == source_paths
            and cached_mtime_ns == source_mtime_ns
            and cached_sizes == source_sizes
            and cached_variants == source_variants
        ):
            return {key: data[key] for key in data.files}
        print("text variant embedding cache is stale; rebuilding.", flush=True)

    from game_review_data.embedding_data import LocalEmbedder

    embedder = LocalEmbedder(
        getattr(args, "local_model", "Qwen/Qwen3-Embedding-0.6B"),
        device=getattr(args, "device", None),
        batch_size=int(getattr(args, "embed_batch_size", 32)),
    )
    vectors = []
    offsets = [0]
    for record in records:
        text = record.path.read_text(encoding="utf-8")
        sentences = split_text(text, int(getattr(args, "max_text_sentences", 4096)))
        embedded = np.asarray(embedder.embed(sentences), dtype=np.float32)
        vectors.append(embedded)
        offsets.append(offsets[-1] + embedded.shape[0])
        print(
            f"text variant embedded appid={record.appid} variant={record.variant} sentences={len(sentences)}",
            flush=True,
        )

    payload = {
        "vectors": np.concatenate(vectors, axis=0).astype(np.float32) if vectors else np.zeros((0, 1024), dtype=np.float32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "appids": np.asarray([r.appid for r in records], dtype=object),
        "names": np.asarray([r.name for r in records], dtype=object),
        "variants": np.asarray([r.variant for r in records], dtype=object),
        "paths": np.asarray([str(r.path) for r in records], dtype=object),
        "source_mtime_ns": np.asarray(source_mtime_ns, dtype=np.int64),
        "source_sizes": np.asarray(source_sizes, dtype=np.int64),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return payload


def encode_variant_features(args, encoder, cache: dict, device) -> dict[tuple[str, str], np.ndarray]:
    vectors = cache["vectors"].astype(np.float32)
    offsets = cache["offsets"].astype(np.int64)
    names = [str(x) for x in cache["names"]]
    variants = [str(x) for x in cache["variants"]]
    encode_args = SimpleNamespace(
        feature_views=int(getattr(args, "text_variant_feature_views", getattr(args, "eval_feature_views", 4))),
        sample_fraction=float(getattr(args, "text_variant_sample_fraction", getattr(args, "eval_sample_fraction", 0.6))),
        amp=bool(getattr(args, "amp_eval", True)),
        seed=int(getattr(args, "seed", 42)),
    )
    out = {}
    for index, (name, variant) in enumerate(zip(names, variants)):
        block = vectors[int(offsets[index]): int(offsets[index + 1])]
        if block.size == 0:
            continue
        out[(name, variant)] = encode_text_centroid(encoder, block, encode_args, device)
    return out


def align_labels(h5_path: Path, names: list[str]) -> tuple[np.ndarray, list[str]]:
    tags, label_names, labels = load_labels(None, str(h5_path))
    index = {name: i for i, name in enumerate(label_names)}
    y = np.zeros((len(names), labels.shape[1]), dtype=np.int8)
    keep_names = []
    for row, name in enumerate(names):
        if name in index:
            y[row] = labels[index[name]]
            keep_names.append(name)
    return y, list(tags)


def make_or_load_split(path: Path, names: list[str], args) -> dict[str, list[str]]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: [str(x) for x in payload.get(key, [])] for key in ("train", "val", "test")}

    train_frac = float(getattr(args, "tag_text_train_frac", 0.7))
    val_frac = float(getattr(args, "tag_text_val_frac", 0.15))
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("--tag-text-train-frac and --tag-text-val-frac must leave a positive test split.")
    rng = np.random.default_rng(int(getattr(args, "tag_text_split_seed", getattr(args, "seed", 42))))
    ordered = np.asarray(list(names), dtype=object)
    perm = rng.permutation(len(ordered))
    n_train = max(1, int(round(len(ordered) * train_frac)))
    n_val = max(1, int(round(len(ordered) * val_frac)))
    if n_train + n_val >= len(ordered):
        n_train = max(1, len(ordered) - 2)
        n_val = 1
    payload = {
        "train": [str(x) for x in ordered[perm[:n_train]]],
        "val": [str(x) for x in ordered[perm[n_train:n_train + n_val]]],
        "test": [str(x) for x in ordered[perm[n_train + n_val:]]],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": int(getattr(args, "tag_text_split_seed", getattr(args, "seed", 42))),
    }
    atomic_json_write(payload, path)
    return {key: payload[key] for key in ("train", "val", "test")}


def micro_prf(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= float(threshold)
    truth = y_true > 0
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"micro_f1": f1, "precision": precision, "recall": recall}


def threshold_grid(scores: np.ndarray, steps: int) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        return np.asarray([0.5], dtype=np.float32)
    return np.linspace(float(finite.min()), float(finite.max()), max(2, int(steps)), dtype=np.float32)


def train_anchor_ridge(args, X_anchor: np.ndarray, y: np.ndarray, name_to_index: dict[str, int], split: dict[str, list[str]]):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    train_idx = np.asarray([name_to_index[n] for n in split["train"] if n in name_to_index], dtype=np.int64)
    val_idx = np.asarray([name_to_index[n] for n in split["val"] if n in name_to_index], dtype=np.int64)
    if train_idx.size < 2 or val_idx.size < 1:
        raise ValueError("Text TAG split has too few train/val games.")

    scaler = StandardScaler().fit(X_anchor[train_idx])
    Xtr = scaler.transform(X_anchor[train_idx])
    Xva = scaler.transform(X_anchor[val_idx])
    ytr = y[train_idx].astype(np.float32)
    yva = y[val_idx]

    best = None
    for alpha in DEFAULT_ALPHAS:
        model = Ridge(alpha=float(alpha))
        model.fit(Xtr, ytr)
        val_scores = model.predict(Xva)
        for threshold in threshold_grid(val_scores, int(getattr(args, "tag_text_threshold_steps", 33))):
            metrics = micro_prf(yva, val_scores, float(threshold))
            key = (metrics["micro_f1"], metrics["recall"], -float(alpha))
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "model": model,
                }
    return scaler, best["model"], best["alpha"], best["threshold"], best["metrics"]


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    metrics = micro_prf(y_true, scores, threshold)
    metrics["n_games"] = int(y_true.shape[0])
    return metrics


def evaluate(args, encoder, feats: np.ndarray, feature_names: list[str], combo_dir: Path) -> dict:
    root_value = getattr(args, "text_variant_dir", None)
    if not root_value:
        return {"status": "disabled"}
    root = Path(root_value)

    h5_path = Path(args.h5)
    names, appids = load_h5_names(h5_path)
    records = discover_variant_records(root, names, appids)
    if not records:
        return {"status": "skipped", "reason": f"no variant text files found under {root}"}

    cache_path = Path(getattr(args, "text_variant_cache", "") or (Path(args.out_dir) / "text_variant_embedding_cache.npz"))
    cache = load_or_embed_variant_texts(args, records, cache_path)
    device = torch.device(args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    variant_features = encode_variant_features(args, encoder, cache, device)

    anchor = feats.mean(axis=1).astype(np.float32)
    name_to_index = {name: i for i, name in enumerate(feature_names)}
    y, tags = align_labels(h5_path, feature_names)
    split_path = Path(getattr(args, "tag_text_split_json", "") or (Path(args.out_dir) / "tag_text_eval_split.json"))
    split = make_or_load_split(split_path, feature_names, args)
    scaler, ridge, alpha, threshold, val_metrics = train_anchor_ridge(args, anchor, y, name_to_index, split)

    test_names = [n for n in split["test"] if n in name_to_index]
    test_idx = np.asarray([name_to_index[n] for n in test_names], dtype=np.int64)
    if test_idx.size == 0:
        return {"status": "skipped", "reason": "empty text TAG test split"}

    anchor_test_scores = ridge.predict(scaler.transform(anchor[test_idx]))
    tag_generalization = {
        "anchor_test": evaluate_scores(y[test_idx], anchor_test_scores, threshold),
    }
    real_text_tag = {}
    cosine_rows = []
    anchor_by_name = {feature_names[i]: anchor[i] for i in range(len(feature_names))}
    split_membership = {}
    for split_name in ("train", "val", "test"):
        for name in split.get(split_name, []):
            split_membership[name] = split_name

    for variant in VARIANTS:
        rows = []
        labels = []
        anchor_rows = []
        variant_names = []
        for name in feature_names:
            feat = variant_features.get((name, variant))
            if feat is None or name not in name_to_index:
                continue
            rows.append(feat)
            labels.append(y[name_to_index[name]])
            anchor_rows.append(anchor_by_name[name])
            variant_names.append(name)
            cosine = float((l2_normalize(anchor_by_name[name][None, :]) @ l2_normalize(feat[None, :]).T)[0, 0])
            cosine_rows.append({
                "name": name,
                "variant": variant,
                "split": split_membership.get(name, "unknown"),
                "anchor_cosine": cosine,
            })
        if not rows:
            real_text_tag[variant] = {
                "variant": {"n_games": 0, "micro_f1": float("nan"), "precision": float("nan"), "recall": float("nan")},
                "anchor_subset": {"n_games": 0, "micro_f1": float("nan"), "precision": float("nan"), "recall": float("nan")},
                "drop_micro_f1": float("nan"),
                "drop_recall": float("nan"),
                "split_counts": {},
            }
            continue
        Xv = np.stack(rows, axis=0).astype(np.float32)
        Xa = np.stack(anchor_rows, axis=0).astype(np.float32)
        yv = np.stack(labels, axis=0).astype(np.int8)
        variant_scores = ridge.predict(scaler.transform(Xv))
        anchor_subset_scores = ridge.predict(scaler.transform(Xa))
        variant_metrics = evaluate_scores(yv, variant_scores, threshold)
        anchor_subset_metrics = evaluate_scores(yv, anchor_subset_scores, threshold)
        split_counts = {}
        for name in variant_names:
            key = split_membership.get(name, "unknown")
            split_counts[key] = split_counts.get(key, 0) + 1
        real_text_tag[variant] = {
            "variant": variant_metrics,
            "anchor_subset": anchor_subset_metrics,
            "drop_micro_f1": float(anchor_subset_metrics["micro_f1"] - variant_metrics["micro_f1"]),
            "drop_recall": float(anchor_subset_metrics["recall"] - variant_metrics["recall"]),
            "split_counts": split_counts,
        }

    cos_by_variant = {}
    for variant in VARIANTS:
        values = [row["anchor_cosine"] for row in cosine_rows if row["variant"] == variant]
        cos_by_variant[variant] = {
            "n_games": len(values),
            "mean_anchor_cosine": float(np.mean(values)) if values else float("nan"),
            "median_anchor_cosine": float(np.median(values)) if values else float("nan"),
        }

    def metric_delta(a: str, b: str, key: str) -> float:
        av = real_text_tag.get(a, {}).get("variant", {}).get(key, float("nan"))
        bv = real_text_tag.get(b, {}).get("variant", {}).get(key, float("nan"))
        return float(av - bv) if not (math.isnan(av) or math.isnan(bv)) else float("nan")

    return {
        "status": "done",
        "split_json": str(split_path),
        "embedding_cache": str(cache_path),
        "ridge_alpha": float(alpha),
        "ridge_threshold": float(threshold),
        "val_anchor": val_metrics,
        "tag_generalization": tag_generalization,
        "real_text_tag": real_text_tag,
        "sentiment_effect": {
            "positive_minus_neutral_micro_f1": metric_delta("positive", "neutral", "micro_f1"),
            "negative_minus_neutral_micro_f1": metric_delta("negative", "neutral", "micro_f1"),
            "noname_minus_neutral_micro_f1": metric_delta("noname", "neutral", "micro_f1"),
            "positive_minus_neutral_recall": metric_delta("positive", "neutral", "recall"),
            "negative_minus_neutral_recall": metric_delta("negative", "neutral", "recall"),
            "noname_minus_neutral_recall": metric_delta("noname", "neutral", "recall"),
        },
        "anchor_cosine": cos_by_variant,
        "cosine_rows": cosine_rows,
        "num_tags": int(len(tags)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
