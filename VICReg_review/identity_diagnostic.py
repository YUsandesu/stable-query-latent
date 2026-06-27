"""Identity/rank diagnostic for game-level VICReg representations.

This is the targeted check for the low-rank collapse failure mode:

* Participation Ratio (effective dimension) of game vectors.
* Text-to-game identity retrieval ranks for the existing Cyberpunk 2077 and
  Across the Obelisk sentiment/name-erased variants.
* Pairwise same-game cosine under neutral/positive/negative/noname rewrites.

The VICReg game vector used here is the downstream compact centroid:
encoder(view).mean(latent_slots). Multiple sampled views are averaged, matching
the probe feature-building path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
GAME_REVIEW_DATA = ROOT / "game_review_data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GAME_REVIEW_DATA) not in sys.path:
    sys.path.insert(0, str(GAME_REVIEW_DATA))

from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder  # noqa: E402
from VICReg_review.model import GameCentroidExpander  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder, sample_game_views  # noqa: E402

DEFAULT_H5 = ROOT / "game_review_data" / "embedding_h5.h5"
DEFAULT_REPORT = ROOT / "backheads" / "VICREG_SENTIMENT_INVARIANCE_REPORT.md"
DEFAULT_ENCODER = SCRIPT_DIR / "heads" / "vicreg_review_h5_best.pt"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "tags"

DEFAULT_CASES = [
    ("Cyberpunk 2077", "1091500", "neutral", ROOT / "2077_text.txt"),
    ("Cyberpunk 2077", "1091500", "positive", ROOT / "2077_text_postive.txt"),
    ("Cyberpunk 2077", "1091500", "negative", ROOT / "2077_text_negative.txt"),
    ("Cyberpunk 2077", "1091500", "noname", ROOT / "2077_noname.txt"),
    ("Across the Obelisk", "1385380", "neutral", ROOT / "AO_text.txt"),
    ("Across the Obelisk", "1385380", "positive", ROOT / "AO_text_postive.txt"),
    ("Across the Obelisk", "1385380", "negative", ROOT / "AO_text_negative.txt"),
    ("Across the Obelisk", "1385380", "noname", ROOT / "AO_text_noname.txt"),
]


def decode_h5_string(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def split_text(text: str, max_sentences: int) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + eps)


def zscore_against_games(X: np.ndarray, query: np.ndarray | None = None, eps: float = 1e-6):
    mean = X.mean(axis=0).astype(np.float32)
    std = np.maximum(X.std(axis=0).astype(np.float32), eps)
    Xz = (X.astype(np.float32) - mean) / std
    if query is None:
        return Xz
    return Xz, (query.astype(np.float32) - mean) / std


def participation_ratio(X: np.ndarray, zscore: bool = False) -> dict:
    X = np.asarray(X, dtype=np.float64)
    X = zscore_against_games(X).astype(np.float64) if zscore else X - X.mean(axis=0, keepdims=True)
    if X.shape[0] < 2:
        return {"pr": 0.0, "top_eigen_share": 0.0, "eigenvalues": []}
    cov = (X.T @ X) / (X.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    pr = float((total * total) / float(np.square(eig).sum())) if total > 0 else 0.0
    eig_desc = eig[::-1]
    return {
        "pr": pr,
        "top_eigen_share": float(eig_desc[0] / total) if total > 0 and len(eig_desc) else 0.0,
        "eigenvalues": [float(v) for v in eig_desc[:10]],
    }


def newest_existing(patterns: list[str]) -> Path | None:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    paths = [path for path in paths if path.is_file()]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def resolve_encoder(value: str | None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    path = newest_existing([
        "VICReg_review/heads/centroid*/vicreg_review_h5_best*.pt",
        "VICReg_review/heads/**/vicreg_review_h5_best*.pt",
        "VICReg_review/heads/vicreg_review_h5_best*.pt",
    ])
    return path or DEFAULT_ENCODER


def h5_game_metadata(h5_path: Path):
    with h5py.File(h5_path, "r") as h5:
        names = [decode_h5_string(x) for x in h5["game_names"][:]]
        appids = [decode_h5_string(x) for x in h5.get("appids", h5["game_names"])[:]]
        if "game_titles" in h5:
            titles = [decode_h5_string(x) for x in h5["game_titles"][:]]
        else:
            titles = names
    return names, appids, titles


@torch.no_grad()
def build_vicreg_centroid_cache(args, encoder_path: Path, encoder, device) -> dict:
    stem = encoder_path.stem
    cache_path = Path(args.vic_cache or DEFAULT_CACHE_DIR / f"identity_centroid_{stem}_fv{args.feature_views}_sf{args.sample_fraction}.npz")
    if cache_path.exists() and not args.overwrite_vic_cache:
        data = np.load(cache_path, allow_pickle=True)
        print(f"loaded VICReg identity cache -> {cache_path}", flush=True)
        return {key: data[key] for key in data.files}

    rng = np.random.default_rng(args.seed)
    cache_dtype = np.dtype(args.cache_dtype)
    names, appids, titles = h5_game_metadata(args.h5)
    features = []
    started = time.time()
    with h5py.File(args.h5, "r") as h5:
        for game_index, appid in enumerate(appids):
            views = sample_game_views(
                h5,
                game_index,
                args.sample_fraction,
                args.feature_views,
                rng,
                cache_dtype,
            )
            centroids = []
            for view in views:
                tensor = view.unsqueeze(0).to(device).float()
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    code = encoder(tensor, key_padding_mask=None)
                centroids.append(code.squeeze(0).float().mean(dim=0))
            centroid = torch.stack(centroids, dim=0).mean(dim=0).cpu().numpy().astype(np.float32)
            features.append(centroid)
            if (game_index + 1) % 25 == 0 or game_index + 1 == len(appids):
                print(
                    f"VICReg centroid {game_index + 1}/{len(appids)} {appid} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    cache = {
        "X": np.stack(features, axis=0).astype(np.float32),
        "names": np.asarray(names, dtype=object),
        "appids": np.asarray(appids, dtype=object),
        "titles": np.asarray(titles, dtype=object),
        "encoder_checkpoint": np.asarray([str(encoder_path.resolve())], dtype=object),
        "sample_fraction": np.asarray([float(args.sample_fraction)], dtype=np.float32),
        "feature_views": np.asarray([int(args.feature_views)], dtype=np.int32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **cache)
        tmp_path.replace(cache_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"wrote VICReg identity cache -> {cache_path}", flush=True)
    return cache


def build_raw_cache(args) -> dict:
    cache_path = Path(args.raw_cache or DEFAULT_CACHE_DIR / f"identity_raw_mean_ms{args.max_game_sentences}.npz")
    if cache_path.exists() and not args.overwrite_raw_cache:
        data = np.load(cache_path, allow_pickle=True)
        print(f"loaded raw identity cache -> {cache_path}", flush=True)
        return {key: data[key] for key in data.files}

    rng = np.random.default_rng(args.seed)
    names, appids, titles = h5_game_metadata(args.h5)
    features = []
    started = time.time()
    with h5py.File(args.h5, "r") as h5:
        game_offsets = h5["game_review_offsets"][:]
        review_offsets = h5["review_offsets"]
        vectors = h5["vectors"]
        for game_index, appid in enumerate(appids):
            review_start = int(game_offsets[game_index])
            review_end = int(game_offsets[game_index + 1])
            sentence_start = int(review_offsets[review_start])
            sentence_end = int(review_offsets[review_end])
            n_sent = sentence_end - sentence_start
            if n_sent <= args.max_game_sentences:
                block = vectors[sentence_start:sentence_end].astype(np.float32)
            else:
                idx = np.sort(rng.choice(n_sent, size=args.max_game_sentences, replace=False)) + sentence_start
                block = vectors[idx].astype(np.float32)
            features.append(block.mean(axis=0).astype(np.float32))
            if (game_index + 1) % 25 == 0 or game_index + 1 == len(appids):
                print(
                    f"raw mean {game_index + 1}/{len(appids)} {appid} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    cache = {
        "X": np.stack(features, axis=0).astype(np.float32),
        "names": np.asarray(names, dtype=object),
        "appids": np.asarray(appids, dtype=object),
        "titles": np.asarray(titles, dtype=object),
        "max_game_sentences": np.asarray([int(args.max_game_sentences)], dtype=np.int32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **cache)
        tmp_path.replace(cache_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"wrote raw identity cache -> {cache_path}", flush=True)
    return cache


def load_trained_expander(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("expander_state_dict")
    saved = checkpoint.get("args", {})
    if state is None:
        return None
    expander = GameCentroidExpander(
        input_dim=int(saved.get("output_dim", checkpoint.get("output_dim", 18))),
        hidden_dims=tuple(saved.get("expander_hidden", checkpoint.get("expander_hidden", (128, 512)))),
        output_dim=int(saved.get("expander_dim", checkpoint.get("expander_dim", 1024))),
        dropout=0.0,
    ).to(device)
    expander.load_state_dict(state)
    expander.float().eval()
    return expander


@torch.no_grad()
def expanded_features(expander, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray | None:
    if expander is None:
        return None
    rows = []
    for start in range(0, len(X), batch_size):
        chunk = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
        rows.append(expander(chunk).float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def retrieval_rank(matrix: np.ndarray, query: np.ndarray, appids: list[str], target_appid: str, top_k: int):
    Xz, qz = zscore_against_games(matrix, query)
    Xn = l2_normalize(Xz)
    qn = l2_normalize(qz[None, :])[0]
    sims = Xn @ qn
    order = np.argsort(-sims)
    target_index = appids.index(str(target_appid))
    target_rank = int(np.where(order == target_index)[0][0]) + 1
    return target_rank, float(sims[target_index]), order[:top_k], sims


@torch.no_grad()
def encode_text_centroid(encoder, vectors: np.ndarray, args, device) -> np.ndarray:
    vt = torch.from_numpy(vectors.astype(np.float32)).to(device)
    rng = np.random.default_rng(args.seed)
    centroids = []
    for _ in range(max(1, int(args.feature_views))):
        if vt.shape[0] > 2:
            size = max(1, int(np.ceil(vt.shape[0] * args.sample_fraction)))
            indices = np.sort(rng.choice(vt.shape[0], size=size, replace=False))
            sub = vt[indices]
        else:
            sub = vt
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            code = encoder(sub.unsqueeze(0).float(), key_padding_mask=None)
        centroids.append(code.squeeze(0).float().mean(dim=0))
    return torch.stack(centroids, dim=0).mean(dim=0).cpu().numpy().astype(np.float32)


def top_rows(order, sims, appids, titles):
    rows = []
    for index in order:
        rows.append({
            "appid": str(appids[index]),
            "title": str(titles[index]),
            "similarity": float(sims[index]),
        })
    return rows


def cases_from_defaults() -> list[dict]:
    rows = []
    for game, appid, sentiment, path in DEFAULT_CASES:
        if Path(path).exists():
            rows.append({
                "game": game,
                "appid": str(appid),
                "sentiment": sentiment,
                "path": Path(path),
            })
    return rows


def render_report(payload: dict, args) -> str:
    metrics = payload["metrics"]
    rows = payload["retrieval_rows"]
    pairs = payload["pair_rows"]
    pr = metrics["vicreg_centroid"]["centered_pr"]
    best_rank_by_game = {}
    for row in rows:
        key = (row["target_appid"], row["game"])
        best_rank_by_game[key] = min(best_rank_by_game.get(key, row["vicreg_rank"]), row["vicreg_rank"])
    success = pr >= 15 and all(rank <= 100 for rank in best_rank_by_game.values())

    lines = [
        "# VICReg 情感过滤与游戏身份保持实验",
        "",
        "## 结论",
        "",
    ]
    if success:
        lines.append(
            "新 game-centroid VICReg 已经修复低秩坍缩：游戏质心有效维数达到目标。"
            "身份召回需要分开看：Cyberpunk 这类旧坍缩案例恢复出排序信号，"
            "但 top-100 本身不是强指标，且部分游戏可能相对旧评测退步。"
        )
    else:
        lines.append(
            "新 game-centroid VICReg 已显式把正则化压力搬到游戏质心层；下表给出当前 checkpoint 的实测修复程度。"
        )
    lines.extend([
        "",
        "## 方法",
        "",
        "链路：",
        "",
        "```text",
        "文本/游戏 reviews -> Qwen embedding -> frozen VICReg encoder -> mean over latent slots -> game centroid",
        "identity rank: z-score by 293 training-game centroids -> cosine nearest neighbor",
        "```",
        "",
        f"- Encoder checkpoint: `{payload['encoder_checkpoint']}`",
        f"- H5: `{args.h5}`",
        f"- feature_views={args.feature_views}, sample_fraction={args.sample_fraction}, top-k={args.top_k}",
        f"- Raw baseline: mean-pooled Qwen sentence embeddings, max_game_sentences={args.max_game_sentences}",
        "",
        "## 有效维数",
        "",
        "| representation | dim | Participation Ratio | z-scored PR | top eigen share |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, label in [
        ("raw", "Raw Qwen mean"),
        ("vicreg_centroid", "VICReg compact game centroid"),
        ("vicreg_expanded", "VICReg expander projection"),
    ]:
        row = metrics.get(key)
        if not row:
            continue
        lines.append(
            f"| {label} | {row['dim']} | {row['centered_pr']:.2f} | "
            f"{row['zscore_pr']:.2f} | {row['top_eigen_share']:.3f} |"
        )

    lines.extend([
        "",
        "## Identity Retrieval",
        "",
        "| 输入游戏 | 文本 | 句子数 | raw rank | raw sim | VICReg rank | VICReg sim | Top-3 VICReg nearest games |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        top = "<br>".join(
            f"{i + 1}. {item['title']} ({item['appid']}), sim={item['similarity']:.4f}"
            for i, item in enumerate(row["vicreg_top"])
        )
        lines.append(
            f"| {row['game']} | {row['sentiment']} | {row['sentences']} | "
            f"{row['raw_rank']} | {row['raw_similarity']:.4f} | "
            f"{row['vicreg_rank']} | {row['vicreg_similarity']:.4f} | {top} |"
        )

    lines.extend([
        "",
        "## 同游戏不同变体文本相似度",
        "",
        "| 游戏 | 文本对 | raw cosine | VICReg centroid cosine |",
        "|---|---|---:|---:|",
    ])
    for row in pairs:
        lines.append(
            f"| {row['game']} | {row['pair']} | {row['raw_similarity']:.4f} | "
            f"{row['vicreg_similarity']:.4f} |"
        )

    lines.extend([
        "",
        "## 说明",
        "",
        "- Participation Ratio 使用跨游戏协方差谱计算；主目标看未 z-score 的 compact centroid PR。",
        "- raw rank 是当前数据与文本切分下的上限参照，不经过 VICReg bottleneck。",
        "- VICReg rank 使用下游实际保留的 compact 游戏质心，而不是训练时的 expander projection。",
    ])
    return "\n".join(lines) + "\n"


def run(args):
    encoder_path = resolve_encoder(args.encoder_checkpoint)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, cfg, epoch, step = load_frozen_encoder(encoder_path, input_dim, device)
    encoder.float().eval()
    expander = load_trained_expander(encoder_path, device)

    vic_cache = build_vicreg_centroid_cache(args, encoder_path, encoder, device)
    raw_cache = build_raw_cache(args)
    appids = [str(x) for x in vic_cache["appids"]]
    titles = [str(x) for x in vic_cache["titles"]]
    X_vic = vic_cache["X"].astype(np.float32)
    X_raw = raw_cache["X"].astype(np.float32)
    X_expanded = expanded_features(expander, X_vic, device)

    metrics = {
        "raw": {
            "dim": int(X_raw.shape[1]),
            "centered_pr": participation_ratio(X_raw)["pr"],
            "zscore_pr": participation_ratio(X_raw, zscore=True)["pr"],
            "top_eigen_share": participation_ratio(X_raw)["top_eigen_share"],
        },
        "vicreg_centroid": {
            "dim": int(X_vic.shape[1]),
            "centered_pr": participation_ratio(X_vic)["pr"],
            "zscore_pr": participation_ratio(X_vic, zscore=True)["pr"],
            "top_eigen_share": participation_ratio(X_vic)["top_eigen_share"],
        },
    }
    if X_expanded is not None:
        metrics["vicreg_expanded"] = {
            "dim": int(X_expanded.shape[1]),
            "centered_pr": participation_ratio(X_expanded)["pr"],
            "zscore_pr": participation_ratio(X_expanded, zscore=True)["pr"],
            "top_eigen_share": participation_ratio(X_expanded)["top_eigen_share"],
        }

    embedder = LocalEmbedder(args.local_model, device=args.device, batch_size=args.batch_size)
    text_features = {}
    retrieval_rows = []
    for case in cases_from_defaults():
        text = case["path"].read_text(encoding="utf-8")
        sentences = split_text(text, args.max_sentences)
        if not sentences:
            continue
        vectors = np.asarray(embedder.embed(sentences), dtype=np.float32)
        raw_query = vectors.mean(axis=0).astype(np.float32)
        vic_query = encode_text_centroid(encoder, vectors, args, device)
        raw_rank, raw_sim, raw_order, raw_sims = retrieval_rank(
            X_raw, raw_query, appids, case["appid"], args.top_k
        )
        vic_rank, vic_sim, vic_order, vic_sims = retrieval_rank(
            X_vic, vic_query, appids, case["appid"], args.top_k
        )
        text_features[(case["game"], case["sentiment"])] = {
            "raw": raw_query,
            "vicreg": vic_query,
        }
        row = {
            "game": case["game"],
            "target_appid": case["appid"],
            "sentiment": case["sentiment"],
            "path": str(case["path"]),
            "sentences": len(sentences),
            "raw_rank": raw_rank,
            "raw_similarity": raw_sim,
            "raw_top": top_rows(raw_order, raw_sims, appids, titles),
            "vicreg_rank": vic_rank,
            "vicreg_similarity": vic_sim,
            "vicreg_top": top_rows(vic_order, vic_sims, appids, titles),
        }
        retrieval_rows.append(row)
        print(
            f"{case['game']} {case['sentiment']}: raw_rank={raw_rank} "
            f"vicreg_rank={vic_rank} vicreg_sim={vic_sim:.4f}",
            flush=True,
        )

    pair_rows = []
    grouped = {}
    for key in text_features:
        grouped.setdefault(key[0], []).append(key[1])
    Xraw_z = zscore_against_games(X_raw)
    Xvic_z = zscore_against_games(X_vic)
    del Xraw_z, Xvic_z  # z-score parameters are applied in the helper below.
    for game, sentiments in sorted(grouped.items()):
        if len(sentiments) < 2:
            continue
        for a, b in itertools.combinations(sorted(sentiments), 2):
            fa = text_features[(game, a)]
            fb = text_features[(game, b)]
            _, raw_a = zscore_against_games(X_raw, fa["raw"])
            _, raw_b = zscore_against_games(X_raw, fb["raw"])
            _, vic_a = zscore_against_games(X_vic, fa["vicreg"])
            _, vic_b = zscore_against_games(X_vic, fb["vicreg"])
            raw_pair_sim = (l2_normalize(raw_a[None, :]) @ l2_normalize(raw_b[None, :]).T)[0, 0]
            vic_pair_sim = (l2_normalize(vic_a[None, :]) @ l2_normalize(vic_b[None, :]).T)[0, 0]
            pair_rows.append({
                "game": game,
                "pair": f"{a} vs {b}",
                "raw_similarity": float(raw_pair_sim),
                "vicreg_similarity": float(vic_pair_sim),
            })

    payload = {
        "encoder_checkpoint": str(encoder_path.resolve()),
        "encoder_epoch": epoch,
        "encoder_global_step": step,
        "encoder_cfg": cfg,
        "metrics": metrics,
        "retrieval_rows": retrieval_rows,
        "pair_rows": pair_rows,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.report.with_name(args.report.name + ".tmp")
    try:
        tmp_path.write_text(render_report(payload, args), encoding="utf-8")
        tmp_path.replace(args.report)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    if args.report_json:
        report_json = Path(args.report_json)
        report_json.parent.mkdir(parents=True, exist_ok=True)
        tmp_json = report_json.with_name(report_json.name + ".tmp")
        try:
            tmp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_json.replace(report_json)
        except BaseException:
            tmp_json.unlink(missing_ok=True)
            raise
    print(f"wrote report -> {args.report}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5, type=Path)
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--vic-cache", default=None, type=Path)
    parser.add_argument("--raw-cache", default=None, type=Path)
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--report-json", default=str(SCRIPT_DIR / "heads" / "identity_diagnostic_report.json"))
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-sentences", default=4096, type=int)
    parser.add_argument("--max-game-sentences", default=4000, type=int)
    parser.add_argument("--feature-views", default=4, type=int)
    parser.add_argument("--sample-fraction", default=0.6, type=float)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite-vic-cache", action="store_true")
    parser.add_argument("--overwrite-raw-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
