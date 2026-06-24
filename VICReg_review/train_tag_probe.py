"""Fair tag probe: cross-validated diagnostic on a frozen VICReg encoder.

This is a redesign of the original single-split probe. It answers "can a frozen
VICReg code predict a game's Steam tags?" honestly, given that the dataset is
tiny (293 games) and the tag set is long-tailed (many tags appear in only a
handful of games, some are effectively unique to one game).

What changed vs. the old probe and *why*:

  * K-fold cross-validation instead of one 80/20 split. With ~290 games a single
    val split of ~57 games is high variance; one lucky/unlucky split moved
    micro-F1 by several points. We report mean +/- std over K folds so the
    number is trustworthy.

  * A fairness rule. A tag is scored on a fold only if it has >= --min-train-pos
    positive games in that fold's TRAIN split and >= 1 in val. A tag that is
    unique to one game (or appears only in the val split) is unpredictable by
    construction; including it just adds guaranteed-zero cells that drag micro-F1
    down and tell us nothing about the encoder. We exclude those cells from the
    metric and report how many tags survive.

  * A per-tag linear classifier (logistic regression) trained per fold, with the
    decision threshold tuned on TRAIN only (no val leakage). The encoder is
    frozen; this just reads its code.

  * A frequency-floor breakdown: micro-F1 restricted to tags seen in >= F games.
    This quantifies the "labels are too discrete" problem directly.

Run tag_build.py first. Compare the reported micro-F1 against ceiling_diagnostic.py
(the same pipeline on the raw 1024-d Qwen embeddings), which is the upper bound
any frozen-encoder probe can reach.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VICReg_review.model import LatentArrayMLP  # noqa: E402
from VICReg_review.train_vicreg_review_h5 import load_game_views  # noqa: E402
try:
    from VICReg_review.coarse_tags import COARSE_TAG_ALIASES
except ImportError:  # pragma: no cover - allows direct script execution
    from coarse_tags import COARSE_TAG_ALIASES

DEFAULT_H5 = SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "heads" / "vicreg_review_h5_best.pt"
DEFAULT_TAGS_DIR = SCRIPT_DIR / "tags"
DEFAULT_OUT_DIR = SCRIPT_DIR / "heads"


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_text_write(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_frozen_encoder(checkpoint_path, input_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint.get("args", {})
    defaults = dict(latent_dim=256, num_latents=256, num_heads=8,
                    dropout=0.1, output_dim=18, reduce_hidden=(128, 64, 32))
    cfg = {key: saved.get(key, defaults[key]) for key in defaults}
    model = LatentArrayMLP(
        input_dim=input_dim,
        latent_dim=cfg["latent_dim"],
        num_latents=cfg["num_latents"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        output_dim=cfg["output_dim"],
        reduce_hidden=tuple(cfg["reduce_hidden"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.float()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, cfg, checkpoint.get("epoch"), checkpoint.get("global_step")


def sample_game_views(h5, game_index, sample_fraction, num_views, rng, cache_dtype):
    views = []
    while len(views) < num_views:
        view_a, view_b = load_game_views(
            h5, game_index, sample_fraction, rng, cache_dtype, pin_cache=False
        )
        views.append(view_a)
        if len(views) < num_views:
            views.append(view_b)
    return views[:num_views]


@torch.no_grad()
def extract_features(encoder, h5_path, sample_fraction, feature_views, seed, cache_dtype, device, amp):
    rng = np.random.default_rng(seed)
    cache_np = np.dtype(cache_dtype)
    with h5py.File(h5_path, "r") as h5:
        game_names = [decode_name(name) for name in h5["game_names"][:]]
        num_games = len(game_names)
        feats = None
        for game_index in range(num_games):
            views = sample_game_views(h5, game_index, sample_fraction, feature_views, rng, cache_np)
            stacked = []
            for view in views:
                tensor = view.unsqueeze(0).to(device).float()
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    code = encoder(tensor, key_padding_mask=None)
                stacked.append(code.squeeze(0).float())
            mean_code = torch.stack(stacked, dim=0).mean(dim=0)  # (num_latents, output_dim)
            if feats is None:
                feats = torch.empty((num_games, *mean_code.shape), dtype=torch.float32)
            feats[game_index] = mean_code.cpu()
            if (game_index + 1) % 50 == 0 or game_index + 1 == num_games:
                print(f"features {game_index + 1}/{num_games}", flush=True)
    return feats.numpy(), game_names


def pool_features(feats, pool):
    """feats: (num_games, num_latents, output_dim) -> (num_games, D)."""
    if pool == "flatten":
        return feats.reshape(feats.shape[0], -1)
    if pool == "mean":
        return feats.mean(axis=1)
    if pool == "stats":
        return np.concatenate([feats.mean(axis=1), feats.std(axis=1)], axis=1)
    raise ValueError(f"unknown pool: {pool}")


def l2_normalize(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def load_labels(tags_dir):
    vocab = json.loads((Path(tags_dir) / "tag_vocab.json").read_text(encoding="utf-8"))
    npz = np.load(Path(tags_dir) / "tag_labels.npz", allow_pickle=True)
    names = [str(n) for n in npz["game_names"]]
    labels = (npz["labels"] > 0).astype(np.int8)
    return vocab["tags"], names, labels


def align(feature_names, label_names, labels):
    index = {n: i for i, n in enumerate(label_names)}
    out = np.zeros((len(feature_names), labels.shape[1]), dtype=np.int8)
    keep = np.zeros(len(feature_names), dtype=bool)
    for row, name in enumerate(feature_names):
        if name in index:
            out[row] = labels[index[name]]
            keep[row] = True
    return out, keep


def kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


def micro_prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, p, r


def cross_validate(X, y, tags, args):
    from sklearn.linear_model import LogisticRegression

    n, num_tags = y.shape
    per_tag_tp = np.zeros(num_tags)
    per_tag_fp = np.zeros(num_tags)
    per_tag_fn = np.zeros(num_tags)
    scored_tags = np.zeros(num_tags, dtype=bool)
    fold_f1s = []

    for fold, (tr, va) in enumerate(kfold_indices(n, args.folds, args.seed)):
        Xtr, Xva = l2_normalize(X[tr], args.norm_eps), l2_normalize(X[va], args.norm_eps)
        f_tp = f_fp = f_fn = 0.0
        learnable = 0
        for t in range(num_tags):
            if int(y[tr, t].sum()) < args.min_train_pos or int(y[va, t].sum()) < 1:
                continue
            learnable += 1
            scored_tags[t] = True
            clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, y[tr, t])
            tr_prob = clf.predict_proba(Xtr)[:, 1]
            va_prob = clf.predict_proba(Xva)[:, 1]
            best_thr, best = 0.5, -1.0
            for thr in np.linspace(0.1, 0.9, 33):
                f1, _, _ = micro_prf(
                    float(((tr_prob >= thr) & (y[tr, t] > 0)).sum()),
                    float(((tr_prob >= thr) & (y[tr, t] == 0)).sum()),
                    float(((tr_prob < thr) & (y[tr, t] > 0)).sum()),
                )
                if f1 > best:
                    best, best_thr = f1, thr
            pred = va_prob >= best_thr
            tru = y[va, t] > 0
            tp = float((pred & tru).sum()); fp = float((pred & ~tru).sum()); fn = float((~pred & tru).sum())
            per_tag_tp[t] += tp; per_tag_fp[t] += fp; per_tag_fn[t] += fn
            f_tp += tp; f_fp += fp; f_fn += fn
        f1, p, r = micro_prf(f_tp, f_fp, f_fn)
        fold_f1s.append(f1)
        print(f"fold {fold}: learnable_tags={learnable} micro_f1={f1:.4f} P={p:.3f} R={r:.3f}", flush=True)

    return per_tag_tp, per_tag_fp, per_tag_fn, scored_tags, fold_f1s


def summarize(per_tag_tp, per_tag_fp, per_tag_fn, scored_tags, fold_f1s, y, tags, args):
    F1, P, R = micro_prf(per_tag_tp[scored_tags].sum(), per_tag_fp[scored_tags].sum(), per_tag_fn[scored_tags].sum())
    per_tag_f1 = np.zeros(len(tags))
    for t in np.flatnonzero(scored_tags):
        per_tag_f1[t], _, _ = micro_prf(per_tag_tp[t], per_tag_fp[t], per_tag_fn[t])
    macro = float(per_tag_f1[scored_tags].mean()) if scored_tags.any() else 0.0

    doc_freq = y.sum(axis=0)
    floor_rows = []
    for floor in args.freq_floors:
        sel = scored_tags & (doc_freq >= floor)
        if not sel.any():
            continue
        f1, p, r = micro_prf(per_tag_tp[sel].sum(), per_tag_fp[sel].sum(), per_tag_fn[sel].sum())
        floor_rows.append({"floor": int(floor), "tags": int(sel.sum()), "micro_f1": f1, "precision": p, "recall": r})

    ranked = sorted(
        [(tags[t], float(per_tag_f1[t]), int(doc_freq[t])) for t in np.flatnonzero(scored_tags)],
        key=lambda x: -x[1],
    )
    return {
        "micro_f1": F1,
        "precision": P,
        "recall": R,
        "macro_f1": macro,
        "fold_micro_f1_mean": float(np.mean(fold_f1s)),
        "fold_micro_f1_std": float(np.std(fold_f1s)),
        "fold_micro_f1": [float(x) for x in fold_f1s],
        "scored_tags": int(scored_tags.sum()),
        "total_tags": int(len(tags)),
        "freq_floor_breakdown": floor_rows,
        "top_tags": [[t, round(f, 4), n] for t, f, n in ranked[:15]],
        "bottom_tags": [[t, round(f, 4), n] for t, f, n in ranked[-15:]],
    }


def export_linear_probe(X, y, tags, doc_freq, args, encoder_path):
    """Fit the FINAL deployable probe on ALL labeled games and save a portable
    linear artifact (scaler + per-tag logistic weights). This is the exact same
    method used in cross_validate, just trained on everything for inference. The
    artifact carries no sklearn dependency: inference is sigmoid(x_std @ W.T + b).
    """
    from sklearn.linear_model import LogisticRegression

    Xs = l2_normalize(X, args.norm_eps)
    num_tags = y.shape[1]
    coef = np.zeros((num_tags, X.shape[1]), dtype=np.float32)
    intercept = np.zeros(num_tags, dtype=np.float32)
    threshold = np.full(num_tags, 0.5, dtype=np.float32)
    trained = np.zeros(num_tags, dtype=bool)
    for t in range(num_tags):
        if int(y[:, t].sum()) < args.min_train_pos:
            continue  # too rare to fit; leave at zero (prob ~ sigmoid(0)=0.5 floor handled below)
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
        clf.fit(Xs, y[:, t])
        coef[t] = clf.coef_[0]
        intercept[t] = float(clf.intercept_[0])
        trained[t] = True
        prob = clf.predict_proba(Xs)[:, 1]
        best_thr, best = 0.5, -1.0
        for thr in np.linspace(0.1, 0.9, 33):
            f1, _, _ = micro_prf(
                float(((prob >= thr) & (y[:, t] > 0)).sum()),
                float(((prob >= thr) & (y[:, t] == 0)).sum()),
                float(((prob < thr) & (y[:, t] > 0)).sum()),
            )
            if f1 > best:
                best, best_thr = f1, thr
        threshold[t] = best_thr

    content_mask = np.zeros(num_tags, dtype=bool)
    groups_path = Path(args.tags_dir) / "tag_groups.json"
    if groups_path.exists():
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        content_set = set(groups.get("content", []))
        content_mask = np.array([t in content_set for t in tags], dtype=bool)

    artifact = {
        "kind": "linear_tag_probe",
        "encoder_checkpoint": str(Path(encoder_path).resolve()),
        "tags": list(tags),
        "normalizer": "l2",
        "norm_eps": float(args.norm_eps),
        "pool": args.pool,
        "feature_views": args.feature_views,
        "sample_fraction": args.sample_fraction,
        "C": args.C,
        # Backward-compatible identity fields for older readers. New readers use
        # normalizer=l2 and ignore these.
        "scaler_mean": np.zeros(X.shape[1], dtype=np.float32),
        "scaler_scale": np.ones(X.shape[1], dtype=np.float32),
        "coef": coef,
        "intercept": intercept,
        "threshold": threshold,
        "trained_mask": trained,
        "content_mask": content_mask,
        "doc_freq": doc_freq.astype(np.int32),
        "coarse_aliases": {tag: COARSE_TAG_ALIASES[tag] for tag in tags if tag in COARSE_TAG_ALIASES},
        "keyword_weight": float(args.keyword_weight),
    }
    torch.save(artifact, args.export_head)
    print(f"exported deployable linear probe ({int(trained.sum())}/{num_tags} tags fit) -> {args.export_head}",
          flush=True)


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    encoder, cfg, enc_epoch, enc_step = load_frozen_encoder(args.checkpoint, input_dim, device)
    print(f"encoder loaded from {args.checkpoint} (epoch={enc_epoch} step={enc_step}) cfg={cfg}", flush=True)

    stem = Path(args.checkpoint).stem
    cache = Path(args.tags_dir) / f"probe_feat_{stem}_fv{args.feature_views}_sf{args.sample_fraction}.npz"
    if cache.exists() and not args.rebuild_features:
        data = np.load(cache, allow_pickle=True)
        feats, feature_names = data["feats"], [str(n) for n in data["names"]]
        print(f"loaded cached features {feats.shape} from {cache}", flush=True)
    else:
        feats, feature_names = extract_features(
            encoder, args.h5, args.sample_fraction, args.feature_views,
            args.seed, args.cache_dtype, device, args.amp,
        )
        np.savez(cache, feats=feats, names=np.asarray(feature_names))
        print(f"cached features {feats.shape} -> {cache}", flush=True)

    tags, label_names, labels = load_labels(args.tags_dir)
    y, keep = align(np.asarray(feature_names), label_names, labels)
    feats, y = feats[keep], y[keep]
    X = pool_features(feats, args.pool)
    print(f"games_with_labels={X.shape[0]} pooled_dim={X.shape[1]} tags={y.shape[1]} pool={args.pool}", flush=True)

    per_tag_tp, per_tag_fp, per_tag_fn, scored, fold_f1s = cross_validate(X, y, tags, args)
    summary = summarize(per_tag_tp, per_tag_fp, per_tag_fn, scored, fold_f1s, y, tags, args)

    if args.export_head:
        export_linear_probe(X, y, tags, y.sum(axis=0), args, args.checkpoint)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "encoder_epoch": enc_epoch,
        "encoder_global_step": enc_step,
        "encoder_cfg": cfg,
        "eval": "kfold_cv_fair",
        "folds": args.folds,
        "min_train_pos": args.min_train_pos,
        "pool": args.pool,
        "normalizer": "l2",
        "classifier": "per_tag_logreg_balanced",
        "C": args.C,
        "feature_views": args.feature_views,
        "sample_fraction": args.sample_fraction,
        **summary,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_text_write(json.dumps(report, ensure_ascii=False, indent=2), args.report_json)

    print("=" * 64, flush=True)
    print(f"OVERALL (fair, {args.folds}-fold) micro_f1={summary['micro_f1']:.4f} "
          f"P={summary['precision']:.3f} R={summary['recall']:.3f}", flush=True)
    print(f"per-fold micro_f1={summary['fold_micro_f1_mean']:.4f} +/- {summary['fold_micro_f1_std']:.4f}", flush=True)
    print(f"macro_f1={summary['macro_f1']:.4f} over {summary['scored_tags']}/{summary['total_tags']} scored tags", flush=True)
    print("frequency-floor breakdown:", flush=True)
    for row in summary["freq_floor_breakdown"]:
        print(f"  freq>={row['floor']:3d}: tags={row['tags']:3d} micro_f1={row['micro_f1']:.4f} "
              f"P={row['precision']:.3f} R={row['recall']:.3f}", flush=True)
    print(f"wrote {args.report_json}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tags-dir", default=str(DEFAULT_TAGS_DIR))
    parser.add_argument("--report-json", default=str(DEFAULT_OUT_DIR / "tag_probe_report.json"))
    parser.add_argument("--export-head", default=None,
                        help="If set, also fit the final probe on ALL games and save a portable "
                             "linear artifact here (used by validation.py).")
    parser.add_argument("--rebuild-features", action="store_true")

    parser.add_argument("--feature-views", type=int, default=4,
                        help="Views sampled per game; their codes are averaged into one feature.")
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--pool", choices=["flatten", "mean", "stats"], default="flatten",
                        help="flatten (full num_latents*output_dim code) scores highest; "
                             "stats (mean+std over latents, 2*output_dim) is much faster, ~0.04 lower.")

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-pos", type=int, default=2,
                        help="A tag is scored on a fold only if it has >= this many train positives.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 strength for per-tag logistic regression.")
    parser.add_argument("--norm-eps", type=float, default=1e-8, help="Epsilon for row-wise L2 probe normalization.")
    parser.add_argument("--keyword-weight", type=float, default=0.6,
                        help="Inference-time lexical prior weight for coarse real-description probes.")
    parser.add_argument("--freq-floors", type=int, nargs="*", default=[5, 10, 20, 30, 40, 60, 80])
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
