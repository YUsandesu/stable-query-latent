"""Headless: run the deployed tag probe on a real text file and score it.

Replicates validation.py's inference path (embed -> multi-view encode -> pool ->
standardize -> linear -> sigmoid) without the GUI, and scores the predicted tags
against the game's true non-emotional Steam tags so we can measure generalization
on real mechanism/story descriptions.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (str(ROOT), str(ROOT / "game_review_data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features  # noqa: E402
try:
    from VICReg_review.coarse_tags import coarsen_tag_dict, keyword_scores
except ImportError:  # pragma: no cover
    from coarse_tags import coarsen_tag_dict, keyword_scores


def split_text(text, max_sentences=256):
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", text.strip())
    sents = [p.strip() for p in parts if p.strip()]
    return sents[:max_sentences] if sents else [text.strip()]


def normalize_for_probe(pooled, probe):
    normalizer = probe.get("normalizer", "standard")
    if normalizer == "l2":
        eps = float(probe.get("norm_eps", 1e-8))
        return pooled / (np.linalg.norm(pooled) + eps)
    return np.clip((pooled - probe["scaler_mean"]) / probe["scaler_scale"], -10, 10)


def apply_keyword_prior(text, probs, tags, probe):
    weight = float(probe.get("keyword_weight", 0.0) or 0.0)
    if weight <= 0 or not probe.get("coarse_aliases"):
        return probs
    prior = keyword_scores(text, tags)
    weight = min(max(weight, 0.0), 1.0)
    return ((1.0 - weight) * probs + weight * prior).astype(np.float32)


def predict(text, probe, encoder, embedder, device, seed=0):
    sents = split_text(text)
    vectors = embedder.embed(sents)
    vt = torch.tensor(vectors, dtype=torch.float32, device=device)
    n = vt.shape[0]
    views = max(1, int(probe.get("feature_views") or 4))
    frac = float(probe.get("sample_fraction") or 0.6)
    rng = np.random.default_rng(seed)
    codes = []
    with torch.no_grad():
        for _ in range(views):
            if n > 2:
                k = max(1, int(np.ceil(n * frac)))
                idx = np.sort(rng.choice(n, size=k, replace=False))
                sub = vt[idx]
            else:
                sub = vt
            codes.append(encoder(sub.unsqueeze(0), key_padding_mask=None).squeeze(0).float())
    feats = torch.stack(codes, 0).mean(0).cpu().numpy()
    pooled = pool_features(feats[None, ...], probe["pool"])[0]
    xs = normalize_for_probe(pooled, probe)
    probs = 1.0 / (1.0 + np.exp(-(xs @ probe["coef"].T + probe["intercept"])))
    probs = np.where(probe["trained_mask"], probs, 0.0).astype(np.float32)
    return apply_keyword_prior(text, probs, probe["tags"], probe), len(sents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("--appid", required=True, help="Steam appid for ground-truth tags.")
    ap.add_argument("--probe", default=str(SCRIPT_DIR / "heads" / "tag_probe_linear.pt"))
    ap.add_argument("--encoder", default=None, help="Override; default = the probe's encoder.")
    ap.add_argument("--topk", type=int, default=0, help="0 = use the number of true labels for accuracy@K.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    probe = torch.load(args.probe, map_location="cpu", weights_only=False)
    tags = probe["tags"]
    enc_ckpt = args.encoder or probe["encoder_checkpoint"]
    import h5py
    with h5py.File(SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5", "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, _, _, _ = load_frozen_encoder(enc_ckpt, input_dim, device)

    from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder
    embedder = LocalEmbedder(DEFAULT_LOCAL_MODEL, device=str(device), batch_size=16)

    # ground-truth non-emotional tags for this game
    groups = json.loads((SCRIPT_DIR / "tags" / "tag_groups.json").read_text(encoding="utf-8"))
    subjective = set(groups["subjective"])
    games = json.loads((ROOT / "game_review_data" / "Steam Games Metadata and Player Reviews (2020–2024" / "games.json").read_text(encoding="utf-8"))
    raw = games[args.appid].get("tags", {})
    if probe.get("coarse_aliases"):
        raw = coarsen_tag_dict(raw)
    true_tags = [t for t in raw if t in set(tags) and t not in subjective]

    keep = np.array([t not in subjective for t in tags])
    text = Path(args.text_file).read_text(encoding="utf-8")
    probs, n_sent = predict(text, probe, encoder, embedder, device)
    order = [i for i in np.argsort(-probs) if keep[i]]
    k = len(true_tags) if args.topk <= 0 else args.topk
    topk = order[:k]
    pred_tags = [tags[i] for i in topk]

    true_set = set(true_tags)
    hits = [t for t in pred_tags if t in true_set]
    prec = len(hits) / len(pred_tags) if pred_tags else 0.0
    # recall of the true tags within top-K
    rec = len([t for t in true_tags if t in set(pred_tags)]) / len(true_tags) if true_tags else 0.0

    print(f"=== {args.text_file} (appid {args.appid}, {n_sent} sentences) ===")
    print(f"true non-emotional tags ({len(true_tags)}): {true_tags}")
    print(f"top-{k} predicted: {[ (t, round(float(probs[tags.index(t)]),2)) for t in pred_tags ]}")
    print(f"hits ({len(hits)}/{k}): {hits}")
    print(f"precision@{k}={prec:.3f}  recall@{k}={rec:.3f}")


if __name__ == "__main__":
    main()
