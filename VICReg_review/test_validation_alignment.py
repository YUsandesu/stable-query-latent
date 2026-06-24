"""Headless check that validation.py's inference math is aligned with the probe.

No GUI / no embedder. Loads the exported linear artifact and the cached per-game
encoder features, runs the EXACT numpy block validation.predict uses, and verifies
the predicted top tags overlap each game's true tags.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from VICReg_review.train_tag_probe import pool_features  # noqa: E402

artifact = torch.load(SCRIPT_DIR / "heads" / "tag_probe_linear.pt", map_location="cpu", weights_only=False)
assert artifact["kind"] == "linear_tag_probe"
tags = artifact["tags"]
print(f"artifact: pool={artifact['pool']} tags={len(tags)} "
      f"trained={int(artifact['trained_mask'].sum())} content={int(artifact['content_mask'].sum())}")
print(f"encoder_checkpoint={artifact['encoder_checkpoint']}")

feat_cache = SCRIPT_DIR / "tags" / "probe_feat_vicreg_adv10_best_fv4_sf0.6.npz"
data = np.load(feat_cache, allow_pickle=True)
feats_all, names = data["feats"], [str(n) for n in data["names"]]  # (G, num_latents, output_dim)

labnpz = np.load(SCRIPT_DIR / "tags" / "tag_labels.npz", allow_pickle=True)
lab_names = [str(n) for n in labnpz["game_names"]]
labels = (labnpz["labels"] > 0).astype(int)
lab_idx = {n: i for i, n in enumerate(lab_names)}


def predict_one(feats):  # mirrors validation.PredictorWorker.predict
    pooled = pool_features(feats[None, ...], artifact["pool"])[0]
    if artifact.get("normalizer", "standard") == "l2":
        x_probe = pooled / (np.linalg.norm(pooled) + float(artifact.get("norm_eps", 1e-8)))
    else:
        x_probe = (pooled - artifact["scaler_mean"]) / artifact["scaler_scale"]
    logits = x_probe @ artifact["coef"].T + artifact["intercept"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    return np.where(artifact["trained_mask"], probs, 0.0).astype(np.float32)


hits, total = 0, 0
for gi in [0, 25, 60, 120, 200, 280]:
    name = names[gi]
    if name not in lab_idx:
        continue
    probs = predict_one(feats_all[gi])
    assert probs.shape == (len(tags),) and 0.0 <= probs.min() and probs.max() <= 1.0
    top = np.argsort(-probs)[:10]
    true = set(np.flatnonzero(labels[lab_idx[name]]))
    overlap = [tags[t] for t in top if t in true]
    hits += len(overlap); total += 1
    print(f"\n{name}: top-10 predicted = {[tags[t] for t in top]}")
    print(f"  overlap with true tags ({len(overlap)}/10): {overlap}")

print(f"\nOK: inference math runs, probs in [0,1], mean top-10 hits/game = {hits / max(total,1):.1f}")
