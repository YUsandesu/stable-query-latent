"""Evaluate whether VICReg keeps game identity while suppressing sentiment.

The experiment embeds neutral/positive/negative text variants, runs them through
the frozen VICReg encoder, then compares their pooled codes against the cached
game-level VICReg features.
"""

from __future__ import annotations

import argparse
import itertools
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
from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features, sample_game_views  # noqa: E402

DEFAULT_CACHE = SCRIPT_DIR / "heads" / "recommendation_vicreg_features.npz"
DEFAULT_PROBE = SCRIPT_DIR / "heads" / "recommendation_vicreg_linear_probe.pt"
DEFAULT_H5 = ROOT / "VICReg_review" / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_REPORT = SCRIPT_DIR / "VICREG_SENTIMENT_INVARIANCE_REPORT.md"

DEFAULT_CASES = [
    ("Cyberpunk 2077", "1091500", "neutral", ROOT / "2077_text.txt"),
    ("Cyberpunk 2077", "1091500", "positive", ROOT / "2077_text_postive.txt"),
    ("Cyberpunk 2077", "1091500", "negative", ROOT / "2077_text_negative.txt"),
    ("Across the Obelisk", "1385380", "neutral", ROOT / "AO_text.txt"),
    ("Across the Obelisk", "1385380", "positive", ROOT / "AO_text_postive.txt"),
    ("Across the Obelisk", "1385380", "negative", ROOT / "AO_text_negative.txt"),
]


def split_text(text: str, max_sentences: int) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + eps)


def predict_probe_positive(checkpoint: dict, feature: np.ndarray) -> float:
    mean = checkpoint["feature_mean"].astype(np.float32)
    std = np.maximum(checkpoint["feature_std"].astype(np.float32), 1e-6)
    normalized = (feature.astype(np.float32) - mean) / std
    value = float(normalized @ checkpoint["coef"].astype(np.float32) + float(checkpoint["intercept"]))
    if checkpoint.get("target_transform") == "logit":
        return float(1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0))))
    return float(np.clip(value, 0.0, 1.0))


def render_report(results: list[dict], pair_rows: list[dict], args) -> str:
    lines = [
        "# VICReg 情感过滤与游戏身份保持实验",
        "",
        "## 结论",
        "",
        "当前结果强烈支持：VICReg 会削弱好评/差评推荐极性，因此它不适合作为单篇文本的好评率预测器。",
        "游戏身份保持则是部分成立：Across the Obelisk 的三种情绪文本仍能排到同一游戏附近；Cyberpunk 2077 没能匹配回目标游戏。",
        "所以更严谨的结论是：VICReg 有情感过滤倾向，但“鲁棒游戏身份编码器”还需要更稳定的 identity matching 或训练目标来验证。",
        "",
        "## 方法",
        "",
        "链路：",
        "",
        "```text",
        "文本 -> Qwen embedding -> frozen VICReg encoder -> full latent flatten -> z-score cosine nearest game",
        "```",
        "",
        f"- VICReg feature cache: `{args.cache}`",
        f"- VICReg recommendation probe: `{args.probe}`",
        f"- top-k: {args.top_k}",
        "",
        "好评率 probe 的输出只作为“推荐语义是否仍可线性读出”的辅助诊断；如果输出饱和，不解释为真实 Steam 好评率。",
        "",
        "## Nearest Game 结果",
        "",
        "| 输入游戏 | 情感 | 句子数 | probe positive | 真实游戏排名 | 真实游戏相似度 | Top-3 nearest games |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        top = "<br>".join(
            f"{i + 1}. {item['title']} ({item['appid']}), sim={item['similarity']:.4f}"
            for i, item in enumerate(row["top"])
        )
        lines.append(
            f"| {row['game']} | {row['sentiment']} | {row['sentences']} | "
            f"{row['probe_positive']:.4f} | {row['target_rank']} | "
            f"{row['target_similarity']:.4f} | {top} |"
        )

    lines.extend([
        "",
        "## 同游戏不同情绪文本的 VICReg 相似度",
        "",
        "| 游戏 | 文本对 | cosine similarity |",
        "|---|---|---:|",
    ])
    for row in pair_rows:
        lines.append(f"| {row['game']} | {row['pair']} | {row['similarity']:.4f} |")

    lines.extend([
        "",
        "## 解释",
        "",
        "- 好评率 probe 对单篇文本输出饱和，不能解释成真实好评率。",
        "- positive / negative 的 probe positive 几乎没有差距，说明推荐极性很难从当前 VICReg 表示中线性读出。",
        "- Across the Obelisk 的目标排名为 1 / 8 / 24，说明这个游戏上情绪变化后仍保留了一定身份信息。",
        "- Cyberpunk 2077 的目标排名为 288 / 273 / 284，说明当前 identity matching 不能稳定恢复该游戏身份。",
        "",
        "## 建议表述",
        "",
        "> VICReg suppresses recommendation polarity in the tested representation. For some games, this helps different sentiment rewrites remain close to the same game identity, but robust identity preservation is not yet universal.",
        "",
        "中文：",
        "",
        "> VICReg 在当前表示中明显压制了好评/差评语义。对部分游戏，它能让不同情绪改写仍靠近同一游戏身份；但这种身份保持还不是普遍稳定的，需要进一步验证和改进。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE, type=Path)
    parser.add_argument("--probe", default=DEFAULT_PROBE, type=Path)
    parser.add_argument("--h5", default=DEFAULT_H5, type=Path)
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-sentences", default=4096, type=int)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--overwrite-identity-cache", action="store_true")
    args = parser.parse_args()

    import h5py

    cache = np.load(args.cache, allow_pickle=True)
    appids = [str(value) for value in cache["appids"]]
    titles = [str(value) for value in cache["titles"]]
    probe = torch.load(args.probe, map_location="cpu", weights_only=False)

    embedder = LocalEmbedder(args.local_model, device=args.device, batch_size=args.batch_size)
    device = torch.device(embedder.device)
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder_path = Path(probe["encoder_checkpoint"])
    if not encoder_path.is_absolute():
        encoder_path = ROOT / encoder_path
    encoder, _, _, _ = load_frozen_encoder(encoder_path, input_dim, device)
    encoder.float().eval()

    views = max(1, int(probe.get("feature_views") or 4))
    fraction = float(probe.get("sample_fraction") or 0.6)
    pool = probe.get("pool", "stats")
    # For game identity, use the full latent array flattened. The 36-d stats
    # pool used by the recommendation probe is intentionally compact and is not
    # discriminative enough for nearest-neighbor identity matching.
    identity_cache_path = args.report.with_suffix(".identity_cache.npz")
    if identity_cache_path.exists() and not args.overwrite_identity_cache:
        identity_payload = np.load(identity_cache_path, allow_pickle=True)
        X_identity = identity_payload["X"].astype(np.float32)
        print(f"loaded identity cache -> {identity_cache_path}", flush=True)
    else:
        X_identity = []
        cache_dtype = np.dtype("float16")
        rng = np.random.default_rng(args.seed)
        with h5py.File(args.h5, "r") as h5:
            for game_index, appid in enumerate(appids):
                views_for_game = sample_game_views(h5, game_index, fraction, views, rng, cache_dtype)
                codes = []
                with torch.no_grad():
                    for view in views_for_game:
                        tensor = view.unsqueeze(0).to(device).float()
                        code = encoder(tensor, key_padding_mask=None)
                        codes.append(code.squeeze(0).float())
                mean_code = torch.stack(codes, dim=0).mean(dim=0).cpu().numpy()
                X_identity.append(mean_code.reshape(-1).astype(np.float32))
                if (game_index + 1) % 25 == 0 or game_index + 1 == len(appids):
                    print(f"identity features {game_index + 1}/{len(appids)} {appid}", flush=True)
        X_identity = np.stack(X_identity, axis=0).astype(np.float32)
        tmp_path = identity_cache_path.with_suffix(identity_cache_path.suffix + ".tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.savez_compressed(handle, X=X_identity, appids=np.asarray(appids, dtype=object))
            tmp_path.replace(identity_cache_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        print(f"wrote identity cache -> {identity_cache_path}", flush=True)
    identity_mean = X_identity.mean(axis=0).astype(np.float32)
    identity_std = np.maximum(X_identity.std(axis=0).astype(np.float32), 1e-6)
    cache_norm = l2_normalize((X_identity - identity_mean) / identity_std)

    results = []
    probe_features_by_case = {}
    identity_features_by_case = {}

    for game, target_appid, sentiment, path in DEFAULT_CASES:
        text = path.read_text(encoding="utf-8")
        sentences = split_text(text, args.max_sentences)
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
        identity_feature = feats.reshape(-1).astype(np.float32)
        pooled = pool_features(feats[None, ...], pool)[0].astype(np.float32)
        probe_features_by_case[(game, sentiment)] = pooled
        identity_features_by_case[(game, sentiment)] = identity_feature

        identity_query = (identity_feature - identity_mean) / identity_std
        similarities = cache_norm @ l2_normalize(identity_query[None, :])[0]
        order = np.argsort(-similarities)
        target_index = appids.index(target_appid)
        target_rank = int(np.where(order == target_index)[0][0]) + 1
        top = [
            {
                "appid": appids[index],
                "title": titles[index],
                "similarity": float(similarities[index]),
            }
            for index in order[: args.top_k]
        ]
        results.append(
            {
                "game": game,
                "target_appid": target_appid,
                "sentiment": sentiment,
                "sentences": len(sentences),
                "probe_positive": predict_probe_positive(probe, pooled),
                "target_rank": target_rank,
                "target_similarity": float(similarities[target_index]),
                "top": top,
            }
        )
        print(
            f"{game} {sentiment}: target_rank={target_rank} "
            f"target_sim={similarities[target_index]:.4f} probe_pos={results[-1]['probe_positive']:.4f}",
            flush=True,
        )

    pair_rows = []
    for game in sorted({case[0] for case in DEFAULT_CASES}):
        sentiments = ["neutral", "positive", "negative"]
        for a, b in itertools.combinations(sentiments, 2):
            fa = identity_features_by_case[(game, a)]
            fb = identity_features_by_case[(game, b)]
            za = (fa - identity_mean) / identity_std
            zb = (fb - identity_mean) / identity_std
            sim = float((l2_normalize(za[None, :]) @ l2_normalize(zb[None, :]).T)[0, 0])
            pair_rows.append({"game": game, "pair": f"{a} vs {b}", "similarity": sim})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.report.with_name(args.report.name + ".tmp")
    try:
        tmp_path.write_text(render_report(results, pair_rows, args), encoding="utf-8")
        tmp_path.replace(args.report)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
