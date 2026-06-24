"""Train the deployable PXI head from VICReg(cleaned_3 H5) features.

This is the real-text PXI path:

    H5 game vectors from game_review_cleaned_3_sentences.h5
      (cleaned_3 prepends detailed_description/about_the_game/short_description
       before the review texts)
      -> frozen VICReg encoder
      -> cached VICReg code
      -> cross-validated linear PXI mean-regression head

The old pxi_probe.py can evaluate an existing feature cache. This script owns
the feature extraction step so the PXI head is trained on the exact H5 input
distribution used by VICReg training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
if str(ROOT / "game_review_data") not in sys.path:
    sys.path.insert(0, str(ROOT / "game_review_data"))

from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features, sample_game_views  # noqa: E402

DEFAULT_H5 = ROOT / "VICReg_review" / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_DESCRIPTIONS = ROOT / "VICReg_review" / "tags" / "game_descriptions"
DEFAULT_OVERLAP = SCRIPT_DIR / "pxi_vicreg_overlap.json"
DEFAULT_CACHE = ROOT / "VICReg_review" / "tags" / "pxi_feat_h5_cleaned3_vicreg_adv10_best_fv4_sf0.6.npz"
DEFAULT_RAW_CACHE = ROOT / "VICReg_review" / "tags" / "pxi_feat_h5_cleaned3_raw_direct.npz"
DEFAULT_DESC_CACHE = ROOT / "VICReg_review" / "tags" / "pxi_feat_game_descriptions_raw_direct.npz"
DEFAULT_OUT = ROOT / "VICReg_review" / "heads" / "pxi_probe_linear.pt"
DEFAULT_MLP_OUT = ROOT / "VICReg_review" / "heads" / "pxi_probe_mlp.pt"
DEFAULT_REPORT = SCRIPT_DIR / "pxi_reviews_desc_probe_report.json"
DEFAULT_MARKDOWN = ROOT / "VICReg_PXI_result.md"

FUNCTIONAL = [
    "functional_progress_feedback",
    "functional_ease_of_control",
    "functional_audiovisual_appeal",
    "functional_goals_and_rules",
    "functional_challenge",
]
PSYCHOLOGICAL = [
    "psychological_meaning",
    "psychological_mastery",
    "psychological_curiosity",
    "psychological_autonomy",
    "psychological_immersion",
]


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def split_text(text: str, max_sentences: int) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


def newest_existing(patterns: list[str]) -> Path | None:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    paths = [path for path in paths if path.is_file()]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def resolve_encoder_from_tag_head() -> tuple[Path, Path]:
    tag_head = newest_existing([
        "VICReg_review/heads/tag_probe_linear*.pt",
        "VICReg_review/heads/**/tag_probe_linear*.pt",
    ])
    if tag_head is None:
        raise FileNotFoundError("No tag_probe_linear*.pt found.")
    tag_probe = torch.load(tag_head, map_location="cpu", weights_only=False)
    value = tag_probe.get("encoder_checkpoint")
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path, tag_head
    encoder = newest_existing([
        "VICReg_review/heads/sweep_adv/vicreg_adv*_best*.pt",
        "VICReg_review/heads/gui_run/vicreg_review_h5_best*.pt",
        "VICReg_review/heads/vicreg_review_h5_best*.pt",
    ])
    if encoder is None:
        raise FileNotFoundError("No VICReg encoder checkpoint found.")
    return encoder, tag_head


def load_overlap(overlap_path: Path):
    payload = json.loads(overlap_path.read_text(encoding="utf-8"))
    dims = list(payload["dims"])
    rows = list(payload["matches"])
    if not rows:
        raise ValueError(f"No PXI overlap games found in {overlap_path}")
    return dims, rows


def appid_to_h5_index(h5: h5py.File) -> dict[str, int]:
    if "appids" in h5:
        appids = [decode(x) for x in h5["appids"][:]]
    else:
        appids = [decode(x).split("_")[0] for x in h5["game_names"][:]]
    return {appid: index for index, appid in enumerate(appids)}


@torch.no_grad()
def build_feature_cache(args, encoder_path: Path, rows: list[dict]) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dtype = np.dtype(args.cache_dtype)

    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
        index_by_appid = appid_to_h5_index(h5)

    print(f"loading VICReg encoder on {device}: {encoder_path}", flush=True)
    encoder, cfg, epoch, global_step = load_frozen_encoder(encoder_path, input_dim, device)
    encoder.float().eval()

    appids, names, feats = [], [], []
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.h5, "r") as h5:
        for row_index, row in enumerate(rows, start=1):
            appid = row["appid"]
            if appid not in index_by_appid:
                print(f"skip {appid}: not in H5", flush=True)
                continue
            game_index = index_by_appid[appid]
            review_views = sample_game_views(
                h5,
                game_index,
                args.sample_fraction,
                args.feature_views,
                rng,
                cache_dtype,
            )

            codes = []
            for review_view in review_views:
                # H5 cleaned_3 already prepends Steam metadata/description text
                # vectors before reviews, so this view is the desired
                # "comments + intro" input distribution.
                tensor = review_view.unsqueeze(0).to(device).float()
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    code = encoder(tensor, key_padding_mask=None)
                codes.append(code.squeeze(0).float())
            mean_code = torch.stack(codes, dim=0).mean(dim=0).cpu().numpy()
            appids.append(appid)
            names.append(row["pxi_name"])
            feats.append(mean_code.astype(np.float32))
            print(
                f"features {row_index}/{len(rows)} {appid} {row['pxi_name']}: "
                f"h5_view_sentences={[int(view.shape[0]) for view in review_views]} "
                f"code={tuple(mean_code.shape)}",
                flush=True,
            )

    feats_np = np.stack(feats, axis=0).astype(np.float32)
    cache = {
        "appids": np.asarray(appids, dtype=object),
        "names": np.asarray(names, dtype=object),
        "feats": feats_np,
        "encoder_checkpoint": str(encoder_path.resolve()),
        "encoder_epoch": -1 if epoch is None else int(epoch),
        "encoder_global_step": -1 if global_step is None else int(global_step),
        "input_source": "h5_cleaned3_metadata_plus_reviews",
        "sample_fraction": float(args.sample_fraction),
        "feature_views": int(args.feature_views),
        "seed": int(args.seed),
        "h5": str(Path(args.h5).resolve()),
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.cache.with_suffix(args.cache.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **cache)
        tmp.replace(args.cache)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"wrote feature cache -> {args.cache}", flush=True)
    return cache


def load_or_build_cache(args, encoder_path: Path, rows: list[dict]) -> dict:
    if args.cache.exists() and not args.overwrite_cache:
        data = np.load(args.cache, allow_pickle=True)
        cache = {key: data[key] for key in data.files}
        print(f"loaded feature cache -> {args.cache}", flush=True)
        return cache
    return build_feature_cache(args, encoder_path, rows)


def build_raw_direct_cache(args, rows: list[dict]) -> dict:
    """Cache direct Qwen embedding baselines from the full cleaned_3 H5 game set.

    For each overlap game, read every sentence vector already stored in H5
    (metadata/description entries first, then reviews) and compute mean and
    mean+std features. This bypasses VICReg and is the raw-embedding ceiling.
    """
    appids, names, mean_rows, stats_rows, sentence_counts = [], [], [], [], []
    with h5py.File(args.h5, "r") as h5:
        index_by_appid = appid_to_h5_index(h5)
        game_offsets = h5["game_review_offsets"]
        review_offsets = h5["review_offsets"]
        vectors = h5["vectors"]
        for row_index, row in enumerate(rows, start=1):
            appid = row["appid"]
            if appid not in index_by_appid:
                continue
            game_index = index_by_appid[appid]
            review_start = int(game_offsets[game_index])
            review_end = int(game_offsets[game_index + 1])
            sentence_start = int(review_offsets[review_start])
            sentence_end = int(review_offsets[review_end])
            game_vectors = vectors[sentence_start:sentence_end].astype(np.float32)
            mean = game_vectors.mean(axis=0)
            std = game_vectors.std(axis=0)
            appids.append(appid)
            names.append(row["pxi_name"])
            mean_rows.append(mean)
            stats_rows.append(np.concatenate([mean, std], axis=0))
            sentence_counts.append(sentence_end - sentence_start)
            print(
                f"raw baseline {row_index}/{len(rows)} {appid} {row['pxi_name']}: "
                f"sentences={sentence_end - sentence_start}",
                flush=True,
            )
    cache = {
        "appids": np.asarray(appids, dtype=object),
        "names": np.asarray(names, dtype=object),
        "mean": np.stack(mean_rows, axis=0).astype(np.float32),
        "stats": np.stack(stats_rows, axis=0).astype(np.float32),
        "sentence_counts": np.asarray(sentence_counts, dtype=np.int32),
        "input_source": "h5_cleaned3_raw_qwen_direct",
        "h5": str(Path(args.h5).resolve()),
    }
    args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.raw_cache.with_suffix(args.raw_cache.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **cache)
        tmp.replace(args.raw_cache)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"wrote raw direct cache -> {args.raw_cache}", flush=True)
    return cache


def load_or_build_raw_cache(args, rows: list[dict]) -> dict:
    if args.raw_cache.exists() and not args.overwrite_raw_cache:
        data = np.load(args.raw_cache, allow_pickle=True)
        cache = {key: data[key] for key in data.files}
        print(f"loaded raw direct cache -> {args.raw_cache}", flush=True)
        return cache
    return build_raw_direct_cache(args, rows)


def build_description_raw_cache(args, rows: list[dict]) -> dict:
    embedder = LocalEmbedder(
        args.local_model,
        device=args.embed_device,
        batch_size=args.embed_batch_size,
    )
    appids, names, mean_rows, stats_rows, sentence_counts, char_counts = [], [], [], [], [], []
    for row_index, row in enumerate(rows, start=1):
        appid = row["appid"]
        path = args.descriptions_dir / f"{appid}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        sentences = split_text(text, args.max_description_sentences)
        if not sentences:
            continue
        vectors = np.asarray(embedder.embed(sentences), dtype=np.float32)
        mean = vectors.mean(axis=0)
        std = vectors.std(axis=0)
        appids.append(appid)
        names.append(row["pxi_name"])
        mean_rows.append(mean)
        stats_rows.append(np.concatenate([mean, std], axis=0))
        sentence_counts.append(len(sentences))
        char_counts.append(len(text))
        print(
            f"description baseline {row_index}/{len(rows)} {appid} {row['pxi_name']}: "
            f"sentences={len(sentences)} chars={len(text)}",
            flush=True,
        )
    cache = {
        "appids": np.asarray(appids, dtype=object),
        "names": np.asarray(names, dtype=object),
        "mean": np.stack(mean_rows, axis=0).astype(np.float32),
        "stats": np.stack(stats_rows, axis=0).astype(np.float32),
        "sentence_counts": np.asarray(sentence_counts, dtype=np.int32),
        "char_counts": np.asarray(char_counts, dtype=np.int32),
        "input_source": "game_descriptions_raw_qwen_direct",
        "descriptions_dir": str(Path(args.descriptions_dir).resolve()),
    }
    args.description_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.description_cache.with_suffix(args.description_cache.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **cache)
        tmp.replace(args.description_cache)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"wrote description raw cache -> {args.description_cache}", flush=True)
    return cache


def load_or_build_description_cache(args, rows: list[dict]) -> dict:
    if args.description_cache.exists() and not args.overwrite_description_cache:
        data = np.load(args.description_cache, allow_pickle=True)
        cache = {key: data[key] for key in data.files}
        print(f"loaded description raw cache -> {args.description_cache}", flush=True)
        return cache
    return build_description_raw_cache(args, rows)


def make_xy(cache: dict, rows: list[dict], dims: list[str]):
    feat_by_appid = {str(appid): cache["feats"][i] for i, appid in enumerate(cache["appids"])}
    appids, names, y_rows = [], [], []
    for row in rows:
        appid = row["appid"]
        if appid in feat_by_appid:
            appids.append(appid)
            names.append(row["pxi_name"])
            y_rows.append([row["pxi"][dim] for dim in dims])
    X_code = np.stack([feat_by_appid[appid] for appid in appids], axis=0).astype(np.float32)
    Y = np.asarray(y_rows, dtype=np.float32)
    return appids, names, X_code, Y


def make_raw_matrices(raw_cache: dict, appids: list[str]) -> dict[str, np.ndarray]:
    index = {str(appid): i for i, appid in enumerate(raw_cache["appids"])}
    rows = [index[appid] for appid in appids]
    return {
        "raw_mean": np.asarray(raw_cache["mean"][rows], dtype=np.float32),
        "raw_stats": np.asarray(raw_cache["stats"][rows], dtype=np.float32),
    }


def transform_fit(X_train, X_test, normalizer: str, pca_components: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    state = {"normalizer": normalizer, "scaler": None, "pca": None}
    if normalizer == "standard":
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        state["scaler"] = scaler
    elif normalizer == "l2":
        X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
        X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
    elif normalizer == "none":
        pass
    else:
        raise ValueError(f"unknown normalizer {normalizer}")

    if pca_components > 0 and pca_components < X_train.shape[1]:
        pca = PCA(n_components=min(pca_components, X_train.shape[0] - 1)).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        state["pca"] = pca
    return X_train, X_test, state


def transform_apply(X, state):
    X = np.asarray(X, dtype=np.float32)
    if state["normalizer"] == "standard":
        X = state["scaler"].transform(X)
    elif state["normalizer"] == "l2":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    if state["pca"] is not None:
        X = state["pca"].transform(X)
    return X


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    pearson = float("nan")
    if y_true.size >= 2 and np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        pearson = float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1])
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "r2": r2}


def loo_predict(X, Y, alpha: float, normalizer: str, pca: int):
    from sklearn.linear_model import Ridge

    n = X.shape[0]
    pred = np.zeros_like(Y, dtype=np.float32)
    for i in range(n):
        train = np.asarray([j for j in range(n) if j != i])
        Xtr, Xte, _ = transform_fit(X[train], X[i:i + 1], normalizer, pca)
        model = Ridge(alpha=alpha)
        model.fit(Xtr, Y[train])
        pred[i] = model.predict(Xte)[0]
    return pred


def evaluate_grid(X_code, Y, args):
    pools = args.pools
    normalizers = args.normalizers
    alphas = args.alphas
    pcas = args.pcas
    target_min = Y.min(axis=0)
    target_max = Y.max(axis=0)
    results = []
    best = None
    for pool in pools:
        X = pool_features(X_code, pool)
        for normalizer in normalizers:
            for pca in pcas:
                if pca >= X.shape[0] - 1:
                    continue
                for alpha in alphas:
                    pred = loo_predict(X, Y, alpha, normalizer, pca)
                    clipped = np.clip(pred, target_min, target_max)
                    raw_metrics = metrics(Y, pred)
                    clipped_metrics = metrics(Y, clipped)
                    clip_fraction = float(np.mean(np.abs(pred - clipped) > 1e-6))
                    raw_excess = np.maximum(pred - target_max, 0.0) + np.maximum(target_min - pred, 0.0)
                    mean_excess = float(raw_excess.mean())
                    objective = raw_metrics["mae"] + 0.15 * clip_fraction + 0.02 * mean_excess
                    row = {
                        "pool": pool,
                        "normalizer": normalizer,
                        "pca": int(pca),
                        "alpha": float(alpha),
                        "objective": float(objective),
                        "raw": raw_metrics,
                        "clipped": clipped_metrics,
                        "clip_fraction": clip_fraction,
                        "mean_excess": mean_excess,
                    }
                    results.append(row)
                    if best is None or row["objective"] < best["objective"]:
                        best = row
    results.sort(key=lambda row: row["objective"])
    return best, results


def evaluate_matrix_grid(matrices: dict[str, np.ndarray], Y, args):
    target_min = Y.min(axis=0)
    target_max = Y.max(axis=0)
    results = []
    best = None
    for feature_set, X in matrices.items():
        for normalizer in args.normalizers:
            for pca in args.pcas:
                if pca >= X.shape[0] - 1:
                    continue
                for alpha in args.alphas:
                    pred = loo_predict(X, Y, alpha, normalizer, pca)
                    clipped = np.clip(pred, target_min, target_max)
                    raw_metrics = metrics(Y, pred)
                    clipped_metrics = metrics(Y, clipped)
                    clip_fraction = float(np.mean(np.abs(pred - clipped) > 1e-6))
                    raw_excess = np.maximum(pred - target_max, 0.0) + np.maximum(target_min - pred, 0.0)
                    mean_excess = float(raw_excess.mean())
                    objective = raw_metrics["mae"] + 0.15 * clip_fraction + 0.02 * mean_excess
                    row = {
                        "feature_set": feature_set,
                        "normalizer": normalizer,
                        "pca": int(pca),
                        "alpha": float(alpha),
                        "objective": float(objective),
                        "raw": raw_metrics,
                        "clipped": clipped_metrics,
                        "clip_fraction": clip_fraction,
                        "mean_excess": mean_excess,
                    }
                    results.append(row)
                    if best is None or row["objective"] < best["objective"]:
                        best = row
    results.sort(key=lambda row: row["objective"])
    return best, results


class MLPProbe(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
            ])
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev = hidden
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    value = str(value).strip()
    if value in {"", "0", "none", "None"}:
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def _standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    scale = X_train.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (X_train - mean) / scale, (X_test - mean) / scale, mean, scale


def _fit_mlp_once(X_train, Y_train, X_test, config, seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    Xtr, Xte, x_mean, x_scale = _standardize_train_test(X_train.astype(np.float32), X_test.astype(np.float32))
    y_mean = Y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_scale = Y_train.std(axis=0, keepdims=True).astype(np.float32)
    y_scale = np.where(y_scale < 1e-6, 1.0, y_scale)
    Ytr = ((Y_train - y_mean) / y_scale).astype(np.float32)

    model = MLPProbe(
        input_dim=Xtr.shape[1],
        output_dim=Ytr.shape[1],
        hidden_dims=config["hidden_dims"],
        dropout=float(config["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    x_tensor = torch.from_numpy(Xtr).to(device)
    y_tensor = torch.from_numpy(Ytr).to(device)
    best_state = None
    best_loss = float("inf")
    patience = int(config["patience"])
    stale = 0
    for epoch in range(int(config["epochs"])):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x_tensor)
        loss = torch.nn.functional.mse_loss(pred, y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        value = float(loss.detach().cpu())
        if value + 1e-7 < best_loss:
            best_loss = value
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
    return (out * y_scale + y_mean).astype(np.float32)


def loo_predict_mlp(X, Y, config, seed, device):
    n = X.shape[0]
    pred = np.zeros_like(Y, dtype=np.float32)
    for i in range(n):
        train = np.asarray([j for j in range(n) if j != i])
        pred[i] = _fit_mlp_once(X[train], Y[train], X[i:i + 1], config, seed + i * 1009, device)[0]
    return pred


def evaluate_mlp_grid(X_code, Y, args):
    target_min = Y.min(axis=0)
    target_max = Y.max(axis=0)
    device = torch.device(args.mlp_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    best = None
    for pool in args.mlp_pools:
        X = pool_features(X_code, pool)
        for hidden in args.mlp_hidden_dims:
            hidden_dims = parse_hidden_dims(hidden)
            for dropout in args.mlp_dropouts:
                for weight_decay in args.mlp_weight_decays:
                    for lr in args.mlp_lrs:
                        config = {
                            "pool": pool,
                            "hidden_dims": hidden_dims,
                            "dropout": float(dropout),
                            "weight_decay": float(weight_decay),
                            "lr": float(lr),
                            "epochs": int(args.mlp_epochs),
                            "patience": int(args.mlp_patience),
                        }
                        pred = loo_predict_mlp(X, Y, config, args.seed, device)
                        clipped = np.clip(pred, target_min, target_max)
                        raw_metrics = metrics(Y, pred)
                        clipped_metrics = metrics(Y, clipped)
                        clip_fraction = float(np.mean(np.abs(pred - clipped) > 1e-6))
                        raw_excess = np.maximum(pred - target_max, 0.0) + np.maximum(target_min - pred, 0.0)
                        mean_excess = float(raw_excess.mean())
                        objective = raw_metrics["mae"] + 0.15 * clip_fraction + 0.02 * mean_excess
                        row = {
                            "pool": pool,
                            "hidden_dims": list(hidden_dims),
                            "dropout": float(dropout),
                            "weight_decay": float(weight_decay),
                            "lr": float(lr),
                            "epochs": int(args.mlp_epochs),
                            "patience": int(args.mlp_patience),
                            "objective": float(objective),
                            "raw": raw_metrics,
                            "clipped": clipped_metrics,
                            "clip_fraction": clip_fraction,
                            "mean_excess": mean_excess,
                        }
                        results.append(row)
                        print(
                            "mlp "
                            f"pool={pool} hidden={hidden_dims or 'linear'} dropout={dropout} "
                            f"wd={weight_decay} lr={lr}: mae={raw_metrics['mae']:.3f} "
                            f"r={raw_metrics['pearson']:.3f} clip={clip_fraction:.3f}",
                            flush=True,
                        )
                        if best is None or row["objective"] < best["objective"]:
                            best = row
    results.sort(key=lambda row: row["objective"])
    return best, results


def fit_final_mlp_head(X_code, Y, dims, appids, names, config, args):
    pool = config["pool"]
    X = pool_features(X_code, pool).astype(np.float32)
    x_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    x_scale = X.std(axis=0, keepdims=True).astype(np.float32)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale)
    y_mean = Y.mean(axis=0, keepdims=True).astype(np.float32)
    y_scale = Y.std(axis=0, keepdims=True).astype(np.float32)
    y_scale = np.where(y_scale < 1e-6, 1.0, y_scale)
    Xn = ((X - x_mean) / x_scale).astype(np.float32)
    Yn = ((Y - y_mean) / y_scale).astype(np.float32)
    device = torch.device(args.mlp_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = MLPProbe(
        input_dim=Xn.shape[1],
        output_dim=Yn.shape[1],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    x_tensor = torch.from_numpy(Xn).to(device)
    y_tensor = torch.from_numpy(Yn).to(device)
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(int(config["epochs"])):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x_tensor)
        loss = torch.nn.functional.mse_loss(pred, y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        value = float(loss.detach().cpu())
        if value + 1e-7 < best_loss:
            best_loss = value
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if int(config.get("patience", args.mlp_patience)) > 0 and stale >= int(config.get("patience", args.mlp_patience)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "kind": "mlp_pxi_probe",
        "target_kind": "mean_regression",
        "input_source": "h5_cleaned3_metadata_plus_reviews",
        "dims": list(dims),
        "functional_dims": list(FUNCTIONAL),
        "psychological_dims": list(PSYCHOLOGICAL),
        "appid_order": list(appids),
        "game_names": list(names),
        "pool": pool,
        "feature_dim": int(X.shape[1]),
        "hidden_dims": list(config["hidden_dims"]),
        "dropout": float(config["dropout"]),
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "epochs": int(config["epochs"]),
        "x_mean": x_mean.squeeze(0).astype(np.float32),
        "x_scale": x_scale.squeeze(0).astype(np.float32),
        "y_mean": y_mean.squeeze(0).astype(np.float32),
        "y_scale": y_scale.squeeze(0).astype(np.float32),
        "model_state_dict": {key: val.detach().cpu() for key, val in model.state_dict().items()},
        "target_mean": Y.mean(axis=0).astype(np.float32),
        "target_std": Y.std(axis=0).astype(np.float32),
        "target_min": Y.min(axis=0).astype(np.float32),
        "target_max": Y.max(axis=0).astype(np.float32),
        "n_games": int(len(appids)),
        "cv_best": config,
        "caveat": "Experimental MLP probe; selected by leave-one-out on N=21.",
    }
    args.export_mlp_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.export_mlp_head)
    print(f"exported experimental MLP PXI head -> {args.export_mlp_head}", flush=True)
    return payload


def fit_final_head(X_code, Y, dims, appids, names, config, cache, args):
    from sklearn.linear_model import Ridge

    X = pool_features(X_code, config["pool"])
    X_fit, _, state = transform_fit(X, X[:1], config["normalizer"], int(config["pca"]))
    model = Ridge(alpha=float(config["alpha"]))
    model.fit(X_fit, Y)

    payload = {
        "kind": "linear_pxi_probe",
        "target_kind": "mean_regression",
        "input_source": "h5_cleaned3_metadata_plus_reviews",
        "dims": list(dims),
        "functional_dims": list(FUNCTIONAL),
        "psychological_dims": list(PSYCHOLOGICAL),
        "appid_order": list(appids),
        "game_names": list(names),
        "pool": config["pool"],
        "normalizer": config["normalizer"],
        "norm_eps": 1e-8,
        "feature_dim": int(X.shape[1]),
        "scaler_mean": None,
        "scaler_scale": None,
        "pca_components": None,
        "pca_mean": None,
        "ridge_coef": model.coef_.astype(np.float32),
        "ridge_intercept": model.intercept_.astype(np.float32),
        "target_mean": Y.mean(axis=0).astype(np.float32),
        "target_std": Y.std(axis=0).astype(np.float32),
        "target_min": Y.min(axis=0).astype(np.float32),
        "target_max": Y.max(axis=0).astype(np.float32),
        "alpha": float(config["alpha"]),
        "pca": int(config["pca"]),
        "n_games": int(len(appids)),
        "encoder_checkpoint": str(cache.get("encoder_checkpoint", "")),
        "feature_cache": str(args.cache.resolve()),
        "overlap_json": str(args.overlap.resolve()),
        "cv_best": config,
        "caveat": "N=21; leave-one-out tuned small-sample PXI probe.",
    }
    if state["normalizer"] == "standard":
        payload["scaler_mean"] = state["scaler"].mean_.astype(np.float32)
        payload["scaler_scale"] = state["scaler"].scale_.astype(np.float32)
    if state["pca"] is not None:
        payload["pca_components"] = state["pca"].components_.astype(np.float32)
        payload["pca_mean"] = state["pca"].mean_.astype(np.float32)

    args.export_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.export_head)
    print(f"exported deployable PXI head -> {args.export_head}", flush=True)
    return payload


def predict_with_payload(X_code, payload):
    X = pool_features(X_code, payload["pool"])
    state = {
        "normalizer": payload.get("normalizer", "standard"),
        "scaler": None,
        "pca": None,
    }
    if state["normalizer"] == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler_mean"])
        scaler.scale_ = np.asarray(payload["scaler_scale"])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
        state["scaler"] = scaler
    if payload.get("pca_components") is not None:
        class PcaState:
            pass

        pca = PcaState()
        pca.components_ = np.asarray(payload["pca_components"])
        pca.mean_ = np.asarray(payload["pca_mean"])
        pca.transform = lambda Z: (Z - pca.mean_) @ pca.components_.T
        state["pca"] = pca
    Xt = transform_apply(X, state)
    return Xt @ payload["ridge_coef"].T + payload["ridge_intercept"]


def group_metrics(Y, pred, dims, group):
    cols = [dims.index(dim) for dim in group]
    return metrics(Y[:, cols].reshape(-1), pred[:, cols].reshape(-1))


def write_reports(args, dims, rows, appids, names, Y, cv_pred, final_pred, best, grid, payload,
                  raw_best=None, raw_grid=None, desc_best=None, desc_grid=None,
                  mlp_best=None, mlp_grid=None):
    target_min = Y.min(axis=0)
    target_max = Y.max(axis=0)
    cv_clipped = np.clip(cv_pred, target_min, target_max)
    final_clipped = np.clip(final_pred, target_min, target_max)
    cv_clip_count = int(np.sum(np.abs(cv_pred - cv_clipped) > 1e-6))
    final_clip_count = int(np.sum(np.abs(final_pred - final_clipped) > 1e-6))
    cv_all = metrics(Y.reshape(-1), cv_pred.reshape(-1))
    cv_func = group_metrics(Y, cv_pred, dims, FUNCTIONAL)
    cv_psy = group_metrics(Y, cv_pred, dims, PSYCHOLOGICAL)
    final_all = metrics(Y.reshape(-1), final_pred.reshape(-1))
    final_func = group_metrics(Y, final_pred, dims, FUNCTIONAL)
    final_psy = group_metrics(Y, final_pred, dims, PSYCHOLOGICAL)

    report = {
        "n_games": int(len(appids)),
        "dims": dims,
        "best_config": best,
        "raw_direct_baseline_best": raw_best,
        "raw_direct_baseline_top_grid": (raw_grid or [])[:20],
        "description_only_baseline_best": desc_best,
        "description_only_baseline_top_grid": (desc_grid or [])[:20],
        "mlp_probe_best": mlp_best,
        "mlp_probe_top_grid": (mlp_grid or [])[:20],
        "top_grid": grid[:20],
        "cv": {
            "all": cv_all,
            "functional": cv_func,
            "psychological": cv_psy,
            "clip_count": cv_clip_count,
            "clip_total": int(cv_pred.size),
        },
        "final_in_sample": {
            "all": final_all,
            "functional": final_func,
            "psychological": final_psy,
            "clip_count": final_clip_count,
            "clip_total": int(final_pred.size),
        },
        "artifact": str(args.export_head.resolve()),
        "feature_cache": str(args.cache.resolve()),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    row_by_appid = {row["appid"]: row for row in rows}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# VICReg PXI Result",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "",
        f"- Input: `{rel(args.h5)}` game vectors. This H5 comes from cleaned_3, where Steam metadata/description fields are prepended before reviews.",
        f"- Intersection: {len(appids)} PXIbenchmark games with VICReg H5 rows.",
        "- Path: cleaned_3 H5 game view -> frozen VICReg -> cached VICReg code -> PXI mean-regression head.",
        f"- Feature cache: `{rel(args.cache)}`.",
        f"- Raw direct baseline cache: `{rel(args.raw_cache)}`.",
        f"- Description-only baseline cache: `{rel(args.description_cache)}`.",
        f"- Exported head: `{rel(args.export_head)}`.",
        f"- Best LOO config: pool={best['pool']}, normalizer={best['normalizer']}, pca={best['pca']}, alpha={best['alpha']}.",
        "",
        "## Raw Embedding Baseline",
        "",
        "This baseline bypasses VICReg. It directly pools every cleaned_3 sentence embedding "
        "for each game (the prepended Steam description metadata plus all reviews), then "
        "fits a linear/Ridge PXI head with leave-one-out testing. It is the raw-Qwen "
        "reference for how much PXI signal is available before the VICReg bottleneck.",
        "",
        "| feature set | normalizer | pca | alpha | LOO MAE | RMSE | Pearson | R^2 | clipped raw values |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if raw_best:
        raw_clip_count = int(round(raw_best["clip_fraction"] * Y.size))
        lines.append(
            f"| {raw_best['feature_set']} | {raw_best['normalizer']} | {raw_best['pca']} | "
            f"{raw_best['alpha']:.5g} | {raw_best['raw']['mae']:.3f} | "
            f"{raw_best['raw']['rmse']:.3f} | {raw_best['raw']['pearson']:.3f} | "
            f"{raw_best['raw']['r2']:.3f} | {raw_clip_count}/{Y.size} |"
        )
    else:
        lines.append("| unavailable | - | - | - | - | - | - | - | - |")
    lines += [
        "",
        "## Description-Only Raw Baseline",
        "",
        "This baseline embeds only `VICReg_review/tags/game_descriptions/{appid}.txt` "
        "for each PXI overlap game, pools the resulting Qwen sentence embeddings, "
        "and fits a linear/Ridge PXI head with leave-one-out testing. It answers "
        "how far the public game description text alone can go without reviews or VICReg.",
        "",
        "| feature set | normalizer | pca | alpha | LOO MAE | RMSE | Pearson | R^2 | clipped raw values |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if desc_best:
        desc_clip_count = int(round(desc_best["clip_fraction"] * Y.size))
        lines.append(
            f"| {desc_best['feature_set']} | {desc_best['normalizer']} | {desc_best['pca']} | "
            f"{desc_best['alpha']:.5g} | {desc_best['raw']['mae']:.3f} | "
            f"{desc_best['raw']['rmse']:.3f} | {desc_best['raw']['pearson']:.3f} | "
            f"{desc_best['raw']['r2']:.3f} | {desc_clip_count}/{Y.size} |"
        )
    else:
        lines.append("| unavailable | - | - | - | - | - | - | - | - |")

    lines += [
        "",
        "## MLP Probe Experiment",
        "",
        "This probe puts a small MLP after the VICReg code, matching the idea that "
        "downstream tasks often need a learned adapter. The table reports strict "
        "leave-one-out performance, so improvements here count only if they beat "
        "the linear VICReg probe on held-out games.",
        "",
        "| pool | hidden dims | dropout | weight decay | lr | LOO MAE | RMSE | Pearson | R^2 | clipped raw values |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if mlp_best:
        mlp_clip_count = int(round(mlp_best["clip_fraction"] * Y.size))
        hidden = ",".join(str(v) for v in mlp_best["hidden_dims"]) or "linear"
        lines.append(
            f"| {mlp_best['pool']} | {hidden} | {mlp_best['dropout']:.3g} | "
            f"{mlp_best['weight_decay']:.5g} | {mlp_best['lr']:.5g} | "
            f"{mlp_best['raw']['mae']:.3f} | {mlp_best['raw']['rmse']:.3f} | "
            f"{mlp_best['raw']['pearson']:.3f} | {mlp_best['raw']['r2']:.3f} | "
            f"{mlp_clip_count}/{Y.size} |"
        )
    else:
        lines.append("| unavailable | - | - | - | - | - | - | - | - | - |")

    lines += [
        "",
        "## Training Fit Metrics",
        "",
        "These numbers are computed after fitting the exported PXI head on all 21 games. "
        "They describe how well the head fits the available cleaned_3 input "
        "(Steam description metadata plus review text). Lower MAE is better.",
        "",
        "| subset | N values | MAE | RMSE | Pearson | R^2 | clipped raw values |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in [("all dimensions", final_all), ("functional", final_func), ("psychological", final_psy)]:
        n = Y.size if label == "all dimensions" else len(appids) * len(FUNCTIONAL)
        if label == "psychological":
            n = len(appids) * len(PSYCHOLOGICAL)
        lines.append(
            f"| {label} | {n} | {value['mae']:.3f} | {value['rmse']:.3f} | "
            f"{value['pearson']:.3f} | {value['r2']:.3f} | "
            f"{final_clip_count}/{final_pred.size} |"
        )
    lines += [
        "",
        "## Leave-One-Out Test Metrics",
        "",
        "These numbers are the stricter test estimate: each game is predicted by a "
        "head trained on the other 20 games, using the same selected configuration.",
        "",
        "| subset | N values | MAE | RMSE | Pearson | R^2 | clipped raw values |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in [("all dimensions", cv_all), ("functional", cv_func), ("psychological", cv_psy)]:
        n = Y.size if label == "all dimensions" else len(appids) * len(FUNCTIONAL)
        if label == "psychological":
            n = len(appids) * len(PSYCHOLOGICAL)
        lines.append(
            f"| {label} | {n} | {value['mae']:.3f} | {value['rmse']:.3f} | "
            f"{value['pearson']:.3f} | {value['r2']:.3f} | "
            f"{cv_clip_count}/{cv_pred.size} |"
        )

    lines += [
        "",
        "## Per-Game LOO Predictions",
        "",
        "| appid | PXI game | Steam game | match | PXI samples | MAE | functional MAE | psychological MAE |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    func_cols = [dims.index(dim) for dim in FUNCTIONAL]
    psy_cols = [dims.index(dim) for dim in PSYCHOLOGICAL]
    per_game = []
    for i, appid in enumerate(appids):
        err = cv_pred[i] - Y[i]
        per_game.append((float(np.abs(err).mean()), i, err))
    for _, i, err in sorted(per_game):
        appid = appids[i]
        row = row_by_appid[appid]
        lines.append(
            f"| {appid} | {row['pxi_name']} | {row['steam_name']} | {row['match']} | "
            f"{row.get('n_pxi_samples', '')} | {np.abs(err).mean():.3f} | "
            f"{np.abs(err[func_cols]).mean():.3f} | {np.abs(err[psy_cols]).mean():.3f} |"
        )

    lines += [
        "",
        "## Per-Dimension LOO Metrics",
        "",
        "| dimension | group | MAE | RMSE | Pearson | R^2 | actual mean | predicted mean | raw min | raw max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for j, dim in enumerate(dims):
        m = metrics(Y[:, j], cv_pred[:, j])
        group = "functional" if dim in FUNCTIONAL else "psychological"
        lines.append(
            f"| {dim} | {group} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['pearson']:.3f} | "
            f"{m['r2']:.3f} | {Y[:, j].mean():.3f} | {cv_pred[:, j].mean():.3f} | "
            f"{cv_pred[:, j].min():.3f} | {cv_pred[:, j].max():.3f} |"
        )

    lines += [
        "",
        "## Top Description-Only Baseline Configs",
        "",
        "| rank | feature set | normalizer | pca | alpha | objective | raw MAE | raw Pearson | clip fraction | mean excess |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate((desc_grid or [])[:10], start=1):
        lines.append(
            f"| {rank} | {row['feature_set']} | {row['normalizer']} | {row['pca']} | {row['alpha']:.5g} | "
            f"{row['objective']:.3f} | {row['raw']['mae']:.3f} | {row['raw']['pearson']:.3f} | "
            f"{row['clip_fraction']:.3f} | {row['mean_excess']:.3f} |"
        )

    lines += [
        "",
        "## Top MLP Probe Configs",
        "",
        "| rank | pool | hidden dims | dropout | weight decay | lr | objective | raw MAE | raw Pearson | clip fraction | mean excess |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate((mlp_grid or [])[:10], start=1):
        hidden = ",".join(str(v) for v in row["hidden_dims"]) or "linear"
        lines.append(
            f"| {rank} | {row['pool']} | {hidden} | {row['dropout']:.3g} | "
            f"{row['weight_decay']:.5g} | {row['lr']:.5g} | {row['objective']:.3f} | "
            f"{row['raw']['mae']:.3f} | {row['raw']['pearson']:.3f} | "
            f"{row['clip_fraction']:.3f} | {row['mean_excess']:.3f} |"
        )

    lines += [
        "",
        "## Top Raw Baseline Configs",
        "",
        "| rank | feature set | normalizer | pca | alpha | objective | raw MAE | raw Pearson | clip fraction | mean excess |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate((raw_grid or [])[:10], start=1):
        lines.append(
            f"| {rank} | {row['feature_set']} | {row['normalizer']} | {row['pca']} | {row['alpha']:.5g} | "
            f"{row['objective']:.3f} | {row['raw']['mae']:.3f} | {row['raw']['pearson']:.3f} | "
            f"{row['clip_fraction']:.3f} | {row['mean_excess']:.3f} |"
        )

    lines += [
        "",
        "## Top Grid Configs",
        "",
        "| rank | pool | normalizer | pca | alpha | objective | raw MAE | raw Pearson | clip fraction | mean excess |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(grid[:10], start=1):
        lines.append(
            f"| {rank} | {row['pool']} | {row['normalizer']} | {row['pca']} | {row['alpha']:.5g} | "
            f"{row['objective']:.3f} | {row['raw']['mae']:.3f} | {row['raw']['pearson']:.3f} | "
            f"{row['clip_fraction']:.3f} | {row['mean_excess']:.3f} |"
        )

    lines += [
        "",
        "## Caveats",
        "",
        "- N is only 21 games, so leave-one-out estimates have high variance.",
        "- The final exported head is fit on all 21 games after selecting the configuration by LOO.",
        "- This head is calibrated for VICReg features built from the cleaned_3 H5 input distribution.",
        "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.report_json}", flush=True)
    print(f"wrote {args.markdown}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--descriptions-dir", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--overlap", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--raw-cache", type=Path, default=DEFAULT_RAW_CACHE)
    parser.add_argument("--description-cache", type=Path, default=DEFAULT_DESC_CACHE)
    parser.add_argument("--export-head", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--export-mlp-head", type=Path, default=DEFAULT_MLP_OUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--device", default=None)
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--max-description-sentences", type=int, default=256)
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--feature-views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dtype", default="float16")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-raw-cache", action="store_true")
    parser.add_argument("--overwrite-description-cache", action="store_true")
    parser.add_argument("--pools", nargs="+", default=["stats", "mean", "flatten"])
    parser.add_argument("--normalizers", nargs="+", default=["l2", "standard"])
    parser.add_argument("--pcas", nargs="+", type=int, default=[0, 2, 4, 6, 8, 12, 16])
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    )
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--mlp-device", default=None)
    parser.add_argument("--mlp-pools", nargs="+", default=["mean", "stats"])
    parser.add_argument("--mlp-hidden-dims", nargs="+", default=["4", "8", "16", "8,4"])
    parser.add_argument("--mlp-dropouts", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--mlp-weight-decays", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    parser.add_argument("--mlp-lrs", nargs="+", type=float, default=[0.001, 0.003])
    parser.add_argument("--mlp-epochs", type=int, default=600)
    parser.add_argument("--mlp-patience", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    dims, rows = load_overlap(args.overlap)
    encoder_path, tag_head = resolve_encoder_from_tag_head()
    cache = load_or_build_cache(args, encoder_path, rows)
    appids, names, X_code, Y = make_xy(cache, rows, dims)
    raw_cache = load_or_build_raw_cache(args, rows)
    raw_matrices = make_raw_matrices(raw_cache, appids)
    description_cache = load_or_build_description_cache(args, rows)
    description_matrices = make_raw_matrices(description_cache, appids)
    print(f"training PXI head: games={len(appids)} code_shape={X_code.shape}", flush=True)
    raw_best, raw_grid = evaluate_matrix_grid(raw_matrices, Y, args)
    print("best raw baseline:", json.dumps(raw_best, indent=2), flush=True)
    desc_best, desc_grid = evaluate_matrix_grid(description_matrices, Y, args)
    print("best description-only baseline:", json.dumps(desc_best, indent=2), flush=True)
    mlp_best, mlp_grid = None, []
    if not args.skip_mlp:
        mlp_best, mlp_grid = evaluate_mlp_grid(X_code, Y, args)
        print("best MLP probe:", json.dumps(mlp_best, indent=2), flush=True)
        fit_final_mlp_head(X_code, Y, dims, appids, names, mlp_best, args)
    best, grid = evaluate_grid(X_code, Y, args)
    print("best config:", json.dumps(best, indent=2), flush=True)
    payload = fit_final_head(X_code, Y, dims, appids, names, best, cache, args)
    X = pool_features(X_code, best["pool"])
    cv_pred = loo_predict(X, Y, best["alpha"], best["normalizer"], best["pca"])
    final_pred = predict_with_payload(X_code, payload)
    report = write_reports(
        args, dims, rows, appids, names, Y, cv_pred, final_pred, best, grid, payload,
        raw_best=raw_best, raw_grid=raw_grid, desc_best=desc_best, desc_grid=desc_grid,
        mlp_best=mlp_best, mlp_grid=mlp_grid,
    )
    print("CV all:", report["cv"]["all"], flush=True)
    print(
        f"CV clipped: {report['cv']['clip_count']}/{report['cv']['clip_total']} | "
        f"final clipped: {report['final_in_sample']['clip_count']}/{report['final_in_sample']['clip_total']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
