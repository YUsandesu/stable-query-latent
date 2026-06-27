"""Run the final data-size x view-fraction sweep for the Steam review encoder.

The sweep varies four axes:

* train-game-count: how many games are visible during self-supervised training.
* sample-fraction: the random review-view fraction used for the two VICReg views.
* output-dim: compact game-vector width after the hierarchical encoder.
* arm: GRL adversary enabled vs disabled. The no-GRL arm also disables the
  recommendation loss for the corrected ablation.

Evaluation is always performed against the full 293-game H5 candidate pool. The
six long diagnostic texts are test-only here; the default description cache is
the train-only cache that excludes those texts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VICReg_review.identity_diagnostic import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    cases_from_defaults,
    encode_text_centroid,
    l2_normalize,
    participation_ratio,
    retrieval_rank,
    split_text,
    zscore_against_games,
)
from VICReg_review.train_tag_probe import (  # noqa: E402
    cross_validate as tag_cross_validate,
    extract_features,
    load_frozen_encoder,
    load_labels,
    pool_features,
    sample_game_views,
    summarize as tag_summarize,
)

DEFAULT_H5 = SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_OUT_DIR = SCRIPT_DIR / "heads" / "data_view_sweep"
DEFAULT_DESCRIPTION_CACHE = SCRIPT_DIR / "heads" / "description_embedding_cache_train_only.npz"
DEFAULT_PYTHON = Path(sys.executable)


def atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_json_write(payload: dict, path: Path) -> None:
    atomic_text_write(json.dumps(payload, ensure_ascii=False, indent=2), path)


def decode_h5(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("RUN " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def combo_name(output_dim: int, arm: str, train_games: int, view: float) -> str:
    return f"dim{output_dim:03d}_{arm_label(arm)}_n{train_games:03d}_view{int(round(view * 100)):02d}"


def combo_arm_from_dir(combo_dir: Path) -> str | None:
    parts = combo_dir.name.split("_")
    if len(parts) >= 4 and parts[0].startswith("dim") and parts[2].startswith("n") and parts[3].startswith("view"):
        return parts[1]
    return None


def arm_label(arm: str) -> str:
    return str(arm).strip().lower()


def arm_adversary_weight(arm: str) -> float:
    arm = arm_label(arm)
    if arm == "grl":
        return 10.0
    if arm in {"nogrl", "no_grl", "no-grl"}:
        return 0.0
    raise ValueError(f"Unknown arm: {arm}")


def arm_recommendation_weight(arm: str) -> float:
    arm = arm_label(arm)
    if arm == "grl":
        return 30.0
    if arm in {"nogrl", "no_grl", "no-grl"}:
        return 0.0
    raise ValueError(f"Unknown arm: {arm}")


def manifest_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def manifest_matches_arm(path: Path, arm: str) -> bool:
    payload = manifest_payload(path)
    if not payload:
        return False
    try:
        actual_reco = float(payload.get("recommendation_decorr_weight"))
    except (TypeError, ValueError):
        return False
    if not math.isclose(actual_reco, float(arm_recommendation_weight(arm)), rel_tol=0.0, abs_tol=1e-6):
        return False
    if "adversary_weight" in payload:
        try:
            actual_adv = float(payload.get("adversary_weight"))
        except (TypeError, ValueError):
            return False
        if not math.isclose(actual_adv, float(arm_adversary_weight(arm)), rel_tol=0.0, abs_tol=1e-6):
            return False
    return True


def report_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def checkpoint_fingerprint(path: Path) -> dict | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def report_is_current(report_path: Path, checkpoint_path: Path, manifest_path: Path | None = None, arm: str | None = None) -> bool:
    report = report_payload(report_path)
    if not report:
        return False
    current = checkpoint_fingerprint(checkpoint_path)
    if current is None:
        return False
    if arm is not None and manifest_path is not None and not manifest_matches_arm(manifest_path, arm):
        return False
    observed = report.get("checkpoint_file")
    if not isinstance(observed, dict):
        return False
    try:
        observed_size = int(observed.get("size"))
        observed_mtime = int(observed.get("mtime_ns"))
    except (TypeError, ValueError):
        return False
    return observed_size == current["size"] and observed_mtime == current["mtime_ns"]


def combo_dir_for(args, output_dim: int, arm: str, train_games: int, view: float) -> Path:
    return args.out_dir / combo_name(output_dim, arm, train_games, view)


def combo_paths(args, output_dim: int, arm: str, train_games: int, view: float) -> dict[str, Path]:
    combo_dir = combo_dir_for(args, output_dim, arm, train_games, view)
    return {
        "dir": combo_dir,
        "checkpoint": combo_dir / "vicreg_review_h5_latest.pt",
        "best_checkpoint": combo_dir / "vicreg_review_h5_best.pt",
        "history": combo_dir / "vicreg_review_h5_history.tsv",
        "manifest": combo_dir / "vicreg_review_h5_manifest.json",
        "probe_history": combo_dir / "dual_probe_history.tsv",
    }


def manifest_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except json.JSONDecodeError:
        return "bad_json"


def combo_needs_train(args, output_dim: int, arm: str, train_games: int, view: float) -> bool:
    paths = combo_paths(args, output_dim, arm, train_games, view)
    if args.force_train:
        return True
    if not paths["checkpoint"].exists():
        return True
    payload = manifest_payload(paths["manifest"])
    if payload is None:
        return True
    if payload.get("status") != "done":
        return True
    return not manifest_matches_arm(paths["manifest"], arm)


def is_resumable_partial(args, output_dim: int, arm: str, train_games: int, view: float) -> bool:
    paths = combo_paths(args, output_dim, arm, train_games, view)
    payload = manifest_payload(paths["manifest"])
    if not paths["checkpoint"].exists() or args.force_train or payload is None:
        return False
    return payload.get("status") not in {None, "done"} and manifest_matches_arm(paths["manifest"], arm)


def should_try_paired_training(output_dim: int, train_games: int, view: float) -> bool:
    # The n=100/view=0.8 pair repeatedly OOMs on the 20GB Windows GPU.
    # Single-arm runs use the same seed and sampled batches while avoiding the
    # doubled activation/optimizer footprint.
    if view >= 0.8 and train_games >= 100:
        return False
    return True


def max_view_sentences_for(args, view: float) -> int:
    if view >= 0.8:
        return int(args.max_view_sentences_80)
    if view >= 0.6:
        return int(args.max_view_sentences_60)
    return int(args.max_view_sentences_default)


def build_train_command(args, output_dim: int, arm: str, train_games: int, view: float, combo_dir: Path) -> list[str]:
    checkpoint = combo_dir / "vicreg_review_h5_latest.pt"
    cmd = [
        str(args.python),
        str(SCRIPT_DIR / "train_vicreg_review_h5.py"),
        "--input-h5", str(args.h5),
        "--device", args.device,
        "--amp",
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--sample-fraction", f"{view:g}",
        "--train-game-count", str(train_games),
        "--train-game-seed", str(args.train_game_seed),
        "--train-game-anchor-appids", args.train_game_anchor_appids,
        "--encoder-arch", "hierarchical",
        "--output-dim", str(output_dim),
        "--reduce-hidden", "128",
        "--vicreg-scope", "game",
        "--expander-dim", "512",
        "--expander-hidden", "256,512",
        "--compact-variance-weight", "25",
        "--compact-covariance-weight", "25",
        "--description-cache", str(args.description_cache),
        "--no-description-include-extra-cases",
        "--description-align-weight", "5",
        "--description-mse-weight", "10",
        "--recommendation-decorr-weight", f"{arm_recommendation_weight(arm):g}",
        "--recommendation-target-transform", "logit",
        "--adversary-weight", f"{arm_adversary_weight(arm):g}",
        "--cache-mode", "queue",
        "--cache-dtype", args.cache_dtype,
        "--backward-mode", "split_recompute",
        "--probe-every", "0",
        "--checkpoint-out", str(combo_dir / "vicreg_review_h5_latest.pt"),
        "--best-checkpoint-out", str(combo_dir / "vicreg_review_h5_best.pt"),
        "--history-tsv", str(combo_dir / "vicreg_review_h5_history.tsv"),
        "--manifest-json", str(combo_dir / "vicreg_review_h5_manifest.json"),
        "--seed", str(args.seed),
    ]
    max_view_sentences = max_view_sentences_for(args, view)
    if max_view_sentences > 0:
        cmd.extend(["--max-view-sentences", str(max_view_sentences)])
    if is_resumable_partial(args, output_dim, arm, train_games, view):
        cmd.extend(["--resume-checkpoint", str(checkpoint)])
        cmd.append("--reset-optimizer-on-resume")
    if arm_label(arm) == "nogrl":
        cmd.extend(["--grl-lambda", "0"])
    return cmd


def build_paired_train_command(args, output_dim: int, train_games: int, view: float) -> list[str]:
    grl = combo_paths(args, output_dim, "grl", train_games, view)
    nogrl = combo_paths(args, output_dim, "nogrl", train_games, view)
    cmd = [
        str(args.python),
        str(SCRIPT_DIR / "train_vicreg_review_h5_paired.py"),
        "--input-h5", str(args.h5),
        "--device", args.device,
        "--amp",
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--sample-fraction", f"{view:g}",
        "--train-game-count", str(train_games),
        "--train-game-seed", str(args.train_game_seed),
        "--train-game-anchor-appids", args.train_game_anchor_appids,
        "--encoder-arch", "hierarchical",
        "--output-dim", str(output_dim),
        "--reduce-hidden", "128",
        "--vicreg-scope", "game",
        "--expander-dim", "512",
        "--expander-hidden", "256,512",
        "--compact-variance-weight", "25",
        "--compact-covariance-weight", "25",
        "--description-cache", str(args.description_cache),
        "--no-description-include-extra-cases",
        "--description-align-weight", "5",
        "--description-mse-weight", "10",
        "--recommendation-target-transform", "logit",
        "--grl-adversary-weight", f"{arm_adversary_weight('grl'):g}",
        "--nogrl-adversary-weight", f"{arm_adversary_weight('nogrl'):g}",
        "--grl-recommendation-decorr-weight", f"{arm_recommendation_weight('grl'):g}",
        "--nogrl-recommendation-decorr-weight", f"{arm_recommendation_weight('nogrl'):g}",
        "--cache-mode", "queue",
        "--cache-dtype", args.cache_dtype,
        "--backward-mode", "split_recompute",
        "--probe-every", "0",
        "--grl-checkpoint-out", str(grl["checkpoint"]),
        "--grl-best-checkpoint-out", str(grl["best_checkpoint"]),
        "--grl-history-tsv", str(grl["history"]),
        "--grl-manifest-json", str(grl["manifest"]),
        "--grl-probe-history-tsv", str(grl["probe_history"]),
        "--nogrl-checkpoint-out", str(nogrl["checkpoint"]),
        "--nogrl-best-checkpoint-out", str(nogrl["best_checkpoint"]),
        "--nogrl-history-tsv", str(nogrl["history"]),
        "--nogrl-manifest-json", str(nogrl["manifest"]),
        "--nogrl-probe-history-tsv", str(nogrl["probe_history"]),
        "--seed", str(args.seed),
    ]
    max_view_sentences = max_view_sentences_for(args, view)
    if max_view_sentences > 0:
        cmd.extend(["--max-view-sentences", str(max_view_sentences)])
    return cmd


def h5_game_metadata(h5_path: Path) -> tuple[list[str], list[str], list[str]]:
    with h5py.File(h5_path, "r") as h5:
        names = [decode_h5(x) for x in h5["game_names"][:]]
        appids = [decode_h5(x) for x in h5["appids"][:]]
        titles = [decode_h5(x) for x in h5["game_titles"][:]] if "game_titles" in h5 else names
    return names, appids, titles


def cache_raw_game_vectors(args, appids: list[str], names: list[str], titles: list[str]) -> dict:
    cache_path = args.out_dir / "raw_identity_cache_ms4000.npz"
    if cache_path.exists() and not args.rebuild_shared_eval:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    rng = np.random.default_rng(args.seed)
    X = []
    with h5py.File(args.h5, "r") as h5:
        game_offsets = h5["game_review_offsets"][:]
        review_offsets = h5["review_offsets"]
        vectors = h5["vectors"]
        for gi in range(len(appids)):
            review_start = int(game_offsets[gi])
            review_end = int(game_offsets[gi + 1])
            sentence_start = int(review_offsets[review_start])
            sentence_end = int(review_offsets[review_end])
            n_sent = sentence_end - sentence_start
            if n_sent <= args.max_game_sentences:
                block = vectors[sentence_start:sentence_end].astype(np.float32)
            else:
                selected = np.sort(rng.choice(n_sent, size=args.max_game_sentences, replace=False)) + sentence_start
                block = vectors[selected].astype(np.float32)
            X.append(block.mean(axis=0).astype(np.float32))
            if (gi + 1) % 50 == 0 or gi + 1 == len(appids):
                print(f"raw cache {gi + 1}/{len(appids)}", flush=True)
    payload = {
        "X": np.stack(X, axis=0).astype(np.float32),
        "appids": np.asarray(appids, dtype=object),
        "names": np.asarray(names, dtype=object),
        "titles": np.asarray(titles, dtype=object),
    }
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return payload


def embed_test_cases(args) -> dict:
    cache_path = args.out_dir / "test_case_embeddings.npz"
    if cache_path.exists() and not args.rebuild_shared_eval:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    from game_review_data.embedding_data import LocalEmbedder

    embedder = LocalEmbedder(args.local_model, device=args.device, batch_size=args.embed_batch_size)
    vectors = []
    offsets = [0]
    games = []
    appids = []
    sentiments = []
    paths = []
    sentence_counts = []
    for case in cases_from_defaults():
        text = case["path"].read_text(encoding="utf-8")
        sentences = split_text(text, args.max_text_sentences)
        embedded = np.asarray(embedder.embed(sentences), dtype=np.float32)
        vectors.append(embedded)
        offsets.append(offsets[-1] + embedded.shape[0])
        games.append(case["game"])
        appids.append(case["appid"])
        sentiments.append(case["sentiment"])
        paths.append(str(case["path"]))
        sentence_counts.append(len(sentences))
        print(f"embedded test text {case['game']} {case['sentiment']} sentences={len(sentences)}", flush=True)
    payload = {
        "vectors": np.concatenate(vectors, axis=0).astype(np.float32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "games": np.asarray(games, dtype=object),
        "appids": np.asarray(appids, dtype=object),
        "sentiments": np.asarray(sentiments, dtype=object),
        "paths": np.asarray(paths, dtype=object),
        "sentence_counts": np.asarray(sentence_counts, dtype=np.int32),
    }
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return payload


def eval_feature_cache_path(combo_dir: Path) -> Path:
    return combo_dir / "eval_features_full293_fv4.npz"


def build_vicreg_feature_cache(args, checkpoint: Path, combo_dir: Path) -> tuple[np.ndarray, list[str]]:
    cache_path = eval_feature_cache_path(combo_dir)
    if cache_path.exists() and not args.rebuild_eval:
        data = np.load(cache_path, allow_pickle=True)
        return data["feats"].astype(np.float32), [str(n) for n in data["names"]]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, _, _, _ = load_frozen_encoder(checkpoint, input_dim, device)
    with torch.no_grad():
        feats, names = extract_features(
            encoder,
            str(args.h5),
            args.eval_sample_fraction,
            args.eval_feature_views,
            args.seed,
            "float16",
            device,
            args.amp_eval and device.type == "cuda",
        )
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, feats=feats.astype(np.float32), names=np.asarray(names, dtype=object))
    return feats.astype(np.float32), list(names)


def build_vicreg_feature_caches_paired(args, targets: list[tuple[Path, Path]]) -> dict[Path, tuple[np.ndarray, list[str]]]:
    """Build missing GRL/no-GRL eval feature caches in one H5 pass.

    The paired trainer uses identical training batches.  This paired evaluator
    mirrors that idea for evaluation: it samples each evaluation view once, then
    forwards the exact same tensors through each frozen encoder.
    """
    loaded: dict[Path, tuple[np.ndarray, list[str]]] = {}
    pending: list[tuple[Path, Path, Path]] = []
    for checkpoint, combo_dir in targets:
        cache_path = eval_feature_cache_path(combo_dir)
        if cache_path.exists() and not args.rebuild_eval:
            data = np.load(cache_path, allow_pickle=True)
            loaded[combo_dir] = (data["feats"].astype(np.float32), [str(n) for n in data["names"]])
        else:
            pending.append((checkpoint, combo_dir, cache_path))

    if not pending:
        return loaded
    if len(pending) == 1:
        checkpoint, combo_dir, _ = pending[0]
        loaded[combo_dir] = build_vicreg_feature_cache(args, checkpoint, combo_dir)
        return loaded

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    encoders = []
    for checkpoint, _, _ in pending:
        encoder, _, _, _ = load_frozen_encoder(checkpoint, input_dim, device)
        encoders.append(encoder)

    rng = np.random.default_rng(args.seed)
    cache_np = np.dtype(args.cache_dtype)
    feats_by_combo: list[torch.Tensor | None] = [None for _ in pending]
    with torch.no_grad(), h5py.File(args.h5, "r") as h5:
        game_names = [decode_h5(name) for name in h5["game_names"][:]]
        num_games = len(game_names)
        for game_index in range(num_games):
            views = sample_game_views(
                h5,
                game_index,
                args.eval_sample_fraction,
                args.eval_feature_views,
                rng,
                cache_np,
            )
            per_encoder_codes = [[] for _ in encoders]
            for view in views:
                tensor = view.unsqueeze(0).to(device).float()
                for encoder_index, encoder in enumerate(encoders):
                    with torch.amp.autocast("cuda", enabled=args.amp_eval and device.type == "cuda"):
                        code = encoder(tensor, key_padding_mask=None)
                    per_encoder_codes[encoder_index].append(code.squeeze(0).float())
                del tensor
            for encoder_index, codes in enumerate(per_encoder_codes):
                mean_code = torch.stack(codes, dim=0).mean(dim=0)
                if feats_by_combo[encoder_index] is None:
                    feats_by_combo[encoder_index] = torch.empty(
                        (num_games, *mean_code.shape),
                        dtype=torch.float32,
                    )
                feats_by_combo[encoder_index][game_index] = mean_code.cpu()
            if (game_index + 1) % 50 == 0 or game_index + 1 == num_games:
                print(f"paired features {game_index + 1}/{num_games} models={len(pending)}", flush=True)

    for index, (_, combo_dir, cache_path) in enumerate(pending):
        feats = feats_by_combo[index]
        if feats is None:
            raise RuntimeError("paired feature extraction produced no features")
        array = feats.numpy().astype(np.float32)
        with cache_path.open("wb") as handle:
            np.savez_compressed(handle, feats=array, names=np.asarray(game_names, dtype=object))
        loaded[combo_dir] = (array, list(game_names))
    return loaded


def tag_probe_metrics(args, feats: np.ndarray, feature_names: list[str]) -> dict:
    tags, label_names, labels = load_labels(None, str(args.h5))
    index = {n: i for i, n in enumerate(label_names)}
    y = np.zeros((len(feature_names), labels.shape[1]), dtype=np.int8)
    keep = np.zeros(len(feature_names), dtype=bool)
    for row, name in enumerate(feature_names):
        if name in index:
            y[row] = labels[index[name]]
            keep[row] = True
    X = pool_features(feats[keep], "flatten")
    y = y[keep]
    probe_args = SimpleNamespace(
        folds=args.probe_folds,
        seed=args.seed,
        min_train_pos=2,
        C=1.0,
        norm_eps=1e-8,
        freq_floors=[5, 10, 20, 30, 40, 60, 80],
    )
    per_tag_tp, per_tag_fp, per_tag_fn, scored, fold_f1s = tag_cross_validate(X, y, tags, probe_args)
    return tag_summarize(per_tag_tp, per_tag_fp, per_tag_fn, scored, fold_f1s, y, tags, probe_args)


def sentiment_r2(args, X: np.ndarray, names: list[str]) -> dict:
    cache = np.load(SCRIPT_DIR / "tags" / "game_sentiment.npz", allow_pickle=True)
    targets = {str(n): float(s) for n, s in zip(cache["names"], cache["sent"])}
    rows = [i for i, n in enumerate(names) if n in targets]
    y = np.asarray([targets[names[i]] for i in rows], dtype=np.float64)
    X = X[rows].astype(np.float32)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros(len(y), dtype=np.float64)
    for tr, va in kfold_indices(len(y), args.probe_folds, args.seed):
        scaler = StandardScaler().fit(X[tr])
        model = Ridge(alpha=10.0).fit(scaler.transform(X[tr]), y[tr])
        pred[va] = model.predict(scaler.transform(X[va]))
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 and np.std(pred) > 0 else float("nan")
    return {"r2": float(r2), "pearson": corr, "n": int(len(y))}


def kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


def recommendation_probe(args, X: np.ndarray, names: list[str]) -> dict:
    from backheads.train_recommendation_head import DEFAULT_REVIEWS_DIR, load_labels_for_h5
    from backheads.train_recommendation_vicreg_linear_probe import cross_validate as reco_cv, summarize_cv

    rows, keep_indices, _ = load_labels_for_h5(
        args.h5,
        DEFAULT_REVIEWS_DIR,
        label_min_length=0,
        min_label_count=10,
    )
    name_to_row = {name: i for i, name in enumerate(names)}
    with h5py.File(args.h5, "r") as h5:
        h5_names = [decode_h5(x) for x in h5["game_names"][:]]
    selected = []
    labels = []
    for row, game_index in zip(rows, keep_indices):
        game_name = h5_names[int(game_index)]
        if game_name not in name_to_row:
            continue
        selected.append(name_to_row[game_name])
        labels.append([row.positive_rate, row.negative_rate])
    reco_args = SimpleNamespace(
        folds=args.probe_folds,
        inner_folds=3,
        seed=args.seed,
        target_transform="logit",
        logit_eps=1e-4,
        alphas=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    )
    folds = reco_cv(X[np.asarray(selected)].astype(np.float32), np.asarray(labels, dtype=np.float32), reco_args)
    return summarize_cv(folds)


def identity_metrics(args, checkpoint: Path, feats: np.ndarray, names: list[str], raw_cache: dict, text_cache: dict) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, _, _, _ = load_frozen_encoder(checkpoint, input_dim, device)
    encode_args = SimpleNamespace(
        feature_views=args.eval_feature_views,
        sample_fraction=args.eval_sample_fraction,
        amp=args.amp_eval,
        seed=args.seed,
    )
    appids = [str(x) for x in raw_cache["appids"]]
    titles = [str(x) for x in raw_cache["titles"]]
    X_raw = raw_cache["X"].astype(np.float32)
    X_vic = feats.mean(axis=1).astype(np.float32)
    rows = []
    text_features = {}
    offsets = text_cache["offsets"].astype(np.int64)
    for i, (game, appid, sentiment) in enumerate(zip(text_cache["games"], text_cache["appids"], text_cache["sentiments"])):
        vectors = text_cache["vectors"][int(offsets[i]): int(offsets[i + 1])].astype(np.float32)
        raw_query = vectors.mean(axis=0).astype(np.float32)
        vic_query = encode_text_centroid(encoder, vectors, encode_args, device)
        raw_rank, raw_sim, _, _ = retrieval_rank(X_raw, raw_query, appids, str(appid), 3)
        vic_rank, vic_sim, _, _ = retrieval_rank(X_vic, vic_query, appids, str(appid), 3)
        rows.append({
            "game": str(game),
            "appid": str(appid),
            "sentiment": str(sentiment),
            "raw_rank": int(raw_rank),
            "raw_similarity": float(raw_sim),
            "vicreg_rank": int(vic_rank),
            "vicreg_similarity": float(vic_sim),
        })
        text_features[(str(game), str(sentiment))] = {"raw": raw_query, "vicreg": vic_query}

    pair_rows = []
    for game in sorted({key[0] for key in text_features}):
        sentiments = sorted(key[1] for key in text_features if key[0] == game)
        for a_index in range(len(sentiments)):
            for b_index in range(a_index + 1, len(sentiments)):
                a = sentiments[a_index]
                b = sentiments[b_index]
                fa = text_features[(game, a)]
                fb = text_features[(game, b)]
                _, raw_a = zscore_against_games(X_raw, fa["raw"])
                _, raw_b = zscore_against_games(X_raw, fb["raw"])
                _, vic_a = zscore_against_games(X_vic, fa["vicreg"])
                _, vic_b = zscore_against_games(X_vic, fb["vicreg"])
                raw_cos = float((l2_normalize(raw_a[None, :]) @ l2_normalize(raw_b[None, :]).T)[0, 0])
                vic_cos = float((l2_normalize(vic_a[None, :]) @ l2_normalize(vic_b[None, :]).T)[0, 0])
                pair_rows.append({"game": game, "pair": f"{a} vs {b}", "raw_cosine": raw_cos, "vicreg_cosine": vic_cos})
    ranks = [row["vicreg_rank"] for row in rows]
    return {
        "participation_ratio": participation_ratio(X_vic)["pr"],
        "zscore_participation_ratio": participation_ratio(X_vic, zscore=True)["pr"],
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "hit_at_1": float(np.mean([rank <= 1 for rank in ranks])),
        "hit_at_5": float(np.mean([rank <= 5 for rank in ranks])),
        "hit_at_100": float(np.mean([rank <= 100 for rank in ranks])),
        "mean_vicreg_cosine": float(np.mean([row["vicreg_cosine"] for row in pair_rows])) if pair_rows else float("nan"),
        "retrieval_rows": rows,
        "pair_rows": pair_rows,
    }


def evaluate_combo_from_features(
    args,
    checkpoint: Path,
    combo_dir: Path,
    feats: np.ndarray,
    feature_names: list[str],
    raw_cache: dict,
    text_cache: dict,
) -> dict:
    report_path = combo_dir / "eval_report.json"
    arm = combo_arm_from_dir(combo_dir)
    manifest_path = combo_dir / "vicreg_review_h5_manifest.json"
    if report_path.exists() and not args.rebuild_eval and report_is_current(report_path, checkpoint, manifest_path, arm):
        payload = report_payload(report_path)
        if payload is not None:
            return payload

    X_stats = pool_features(feats, "stats").astype(np.float32)
    report = {
        "report_version": 2,
        "arm": arm,
        "adversary_weight": float(arm_adversary_weight(arm)) if arm is not None else None,
        "recommendation_decorr_weight": float(arm_recommendation_weight(arm)) if arm is not None else None,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_file": checkpoint_fingerprint(checkpoint),
        "tag_probe": tag_probe_metrics(args, feats, feature_names),
        "sentiment_probe": sentiment_r2(args, X_stats, feature_names),
        "recommendation_probe": recommendation_probe(args, X_stats, feature_names),
        "identity": identity_metrics(args, checkpoint, feats, feature_names, raw_cache, text_cache),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_json_write(report, report_path)
    return report


def evaluate_combo(args, checkpoint: Path, combo_dir: Path) -> dict:
    report_path = combo_dir / "eval_report.json"
    arm = combo_arm_from_dir(combo_dir)
    manifest_path = combo_dir / "vicreg_review_h5_manifest.json"
    if report_path.exists() and not args.rebuild_eval and report_is_current(report_path, checkpoint, manifest_path, arm):
        payload = report_payload(report_path)
        if payload is not None:
            return payload

    names, appids, titles = h5_game_metadata(args.h5)
    raw_cache = cache_raw_game_vectors(args, appids, names, titles)
    text_cache = embed_test_cases(args)
    feats, feature_names = build_vicreg_feature_cache(args, checkpoint, combo_dir)
    return evaluate_combo_from_features(args, checkpoint, combo_dir, feats, feature_names, raw_cache, text_cache)


def evaluate_targets(args, targets: list[tuple[Path, Path]]) -> None:
    pending = [
        (checkpoint, combo_dir)
        for checkpoint, combo_dir in targets
        if checkpoint.exists()
        and (
            args.rebuild_eval
            or not report_is_current(
                combo_dir / "eval_report.json",
                checkpoint,
                combo_dir / "vicreg_review_h5_manifest.json",
                combo_arm_from_dir(combo_dir),
            )
        )
    ]
    if not pending:
        return

    names, appids, titles = h5_game_metadata(args.h5)
    raw_cache = cache_raw_game_vectors(args, appids, names, titles)
    text_cache = embed_test_cases(args)
    for checkpoint, combo_dir in pending:
        feats, feature_names = build_vicreg_feature_cache(args, checkpoint, combo_dir)
        evaluate_combo_from_features(args, checkpoint, combo_dir, feats, feature_names, raw_cache, text_cache)
        del feats, feature_names
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def scalar_from_report(report: dict) -> dict:
    return {
        "tag_micro_f1": float(report["tag_probe"]["micro_f1"]),
        "tag_fold_std": float(report["tag_probe"]["fold_micro_f1_std"]),
        "sentiment_r2": float(report["sentiment_probe"]["r2"]),
        "recommendation_pearson": float(report["recommendation_probe"]["pearson_mean"]),
        "recommendation_mae": float(report["recommendation_probe"]["mae_mean"]),
        "pr": float(report["identity"]["participation_ratio"]),
        "mean_rank": float(report["identity"]["mean_rank"]),
        "median_rank": float(report["identity"]["median_rank"]),
        "hit_at_1": float(report["identity"]["hit_at_1"]),
        "hit_at_5": float(report["identity"]["hit_at_5"]),
        "hit_at_100": float(report["identity"]["hit_at_100"]),
        "mean_text_cosine": float(report["identity"]["mean_vicreg_cosine"]),
    }


def composite_score(row: dict) -> float:
    tag = row["tag_micro_f1"]
    identity = row["hit_at_5"]
    cos = (row["mean_text_cosine"] + 1.0) / 2.0
    sent_penalty = max(0.0, row["sentiment_r2"])
    reco_penalty = abs(row["recommendation_pearson"])
    pr_bonus = min(row["pr"] / 25.0, 1.0)
    return float(0.30 * tag + 0.30 * identity + 0.15 * cos + 0.15 * pr_bonus - 0.05 * sent_penalty - 0.05 * reco_penalty)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_jsonl(rows: list[dict], path: Path) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def flatten_manifest(manifest: dict, base: dict) -> dict:
    metrics = manifest.get("metrics") or {}
    row = {
        **base,
        "status": manifest.get("status"),
        "finished_at": manifest.get("finished_at"),
        "epoch": manifest.get("epoch"),
        "step": manifest.get("step"),
        "input_h5": manifest.get("input_h5"),
        "checkpoint_out": manifest.get("checkpoint_out"),
        "train_game_count": manifest.get("train_game_count"),
        "train_game_seed": manifest.get("train_game_seed"),
        "train_game_appids": ",".join(str(x) for x in manifest.get("train_game_appids", [])),
        "train_game_indices": ",".join(str(x) for x in manifest.get("train_game_indices", [])),
        "error": manifest.get("error"),
    }
    for key, value in metrics.items():
        row[f"metric_{key}"] = value
    return row


def export_raw_detail_tables(args, rows: list[dict]) -> None:
    raw_dir = args.out_dir / "raw_test_data"
    eval_jsonl = []
    manifest_rows = []
    retrieval_rows = []
    pair_rows = []
    tag_floor_rows = []
    tag_rank_rows = []
    tag_fold_rows = []
    probe_rows = []
    reco_rows = []
    identity_summary_rows = []

    for row in rows:
        base = {
            "output_dim": row.get("output_dim"),
            "arm": row.get("arm"),
            "combo": row.get("combo"),
            "train_games": row.get("train_games"),
            "view_fraction": row.get("view_fraction"),
        }
        combo_dir = combo_dir_for(
            args,
            int(row["output_dim"]),
            str(row["arm"]),
            int(row["train_games"]),
            float(row["view_fraction"]),
        )
        manifest_path = combo_dir / "vicreg_review_h5_manifest.json"
        if manifest_path.exists():
            try:
                manifest_rows.append(flatten_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), base))
            except json.JSONDecodeError as exc:
                manifest_rows.append({**base, "status": "bad_json", "error": str(exc)})

        eval_path = combo_dir / "eval_report.json"
        arm = str(row["arm"])
        manifest_path = combo_dir / "vicreg_review_h5_manifest.json"
        if not report_is_current(eval_path, combo_dir / "vicreg_review_h5_latest.pt", manifest_path, arm):
            continue
        report = report_payload(eval_path)
        if report is None:
            continue
        eval_jsonl.append({**base, "report": report})

        tag = report.get("tag_probe") or {}
        for index, value in enumerate(tag.get("fold_micro_f1", [])):
            tag_fold_rows.append({**base, "fold": index, "micro_f1": value})
        for floor in tag.get("freq_floor_breakdown", []):
            tag_floor_rows.append({**base, **floor})
        for kind in ("top_tags", "bottom_tags"):
            for rank, item in enumerate(tag.get(kind, []), start=1):
                if len(item) >= 3:
                    tag_rank_rows.append({
                        **base,
                        "kind": kind,
                        "rank": rank,
                        "tag": item[0],
                        "f1": item[1],
                        "doc_freq": item[2],
                    })

        sentiment = report.get("sentiment_probe") or {}
        reco = report.get("recommendation_probe") or {}
        probe_rows.append({
            **base,
            "tag_micro_f1": tag.get("micro_f1"),
            "tag_precision": tag.get("precision"),
            "tag_recall": tag.get("recall"),
            "tag_macro_f1": tag.get("macro_f1"),
            "tag_fold_mean": tag.get("fold_micro_f1_mean"),
            "tag_fold_std": tag.get("fold_micro_f1_std"),
            "tag_scored_tags": tag.get("scored_tags"),
            "tag_total_tags": tag.get("total_tags"),
            "sentiment_r2": sentiment.get("r2"),
            "sentiment_pearson": sentiment.get("pearson"),
            "sentiment_n": sentiment.get("n"),
        })
        reco_rows.append({**base, **reco})

        identity = report.get("identity") or {}
        identity_summary_rows.append({
            **base,
            "participation_ratio": identity.get("participation_ratio"),
            "zscore_participation_ratio": identity.get("zscore_participation_ratio"),
            "mean_rank": identity.get("mean_rank"),
            "median_rank": identity.get("median_rank"),
            "hit_at_1": identity.get("hit_at_1"),
            "hit_at_5": identity.get("hit_at_5"),
            "hit_at_100": identity.get("hit_at_100"),
            "mean_vicreg_cosine": identity.get("mean_vicreg_cosine"),
        })
        for detail in identity.get("retrieval_rows", []):
            retrieval_rows.append({**base, **detail})
        for detail in identity.get("pair_rows", []):
            pair_rows.append({**base, **detail})

    write_jsonl(eval_jsonl, raw_dir / "eval_reports_full.jsonl")
    write_csv(manifest_rows, raw_dir / "training_manifests.csv")
    write_csv(probe_rows, raw_dir / "probe_summary.csv")
    write_csv(reco_rows, raw_dir / "recommendation_probe.csv")
    write_csv(identity_summary_rows, raw_dir / "identity_summary.csv")
    write_csv(retrieval_rows, raw_dir / "identity_retrieval_details.csv")
    write_csv(pair_rows, raw_dir / "identity_pair_cosine_details.csv")
    write_csv(tag_floor_rows, raw_dir / "tag_freq_floor_details.csv")
    write_csv(tag_rank_rows, raw_dir / "tag_top_bottom_details.csv")
    write_csv(tag_fold_rows, raw_dir / "tag_fold_details.csv")


def render_report(rows: list[dict], args) -> str:
    complete = [row for row in rows if row["status"] == "done"]
    best = max(complete, key=lambda row: row["composite_score"]) if complete else None
    paired = []
    by_key = {(row.get("output_dim"), row.get("arm"), row["train_games"], row["view_fraction"]): row for row in complete}
    for output_dim in args.output_dims:
        for train_games in args.train_game_counts:
            for view in args.sample_fractions:
                grl = by_key.get((output_dim, "grl", train_games, view))
                nogrl = by_key.get((output_dim, "nogrl", train_games, view))
                if not grl or not nogrl:
                    continue
                paired.append({
                    "output_dim": output_dim,
                    "train_games": train_games,
                    "view_fraction": view,
                    "delta_score": grl["composite_score"] - nogrl["composite_score"],
                    "delta_tag": grl["tag_micro_f1"] - nogrl["tag_micro_f1"],
                    "delta_sentiment_r2": grl["sentiment_r2"] - nogrl["sentiment_r2"],
                    "delta_reco_pearson_abs": abs(grl["recommendation_pearson"]) - abs(nogrl["recommendation_pearson"]),
                    "delta_hit_at_5": grl["hit_at_5"] - nogrl["hit_at_5"],
                    "delta_pr": grl["pr"] - nogrl["pr"],
                })
    lines = [
        "# 数据量 x View Fraction 收尾实验报告",
        "",
        f"- 日期：{time.strftime('%Y-%m-%d')}",
        f"- 输出维度轴：{', '.join(str(x) for x in args.output_dims)}",
        f"- 总游戏数：293；实验数据量轴：{', '.join(str(x) for x in args.train_game_counts)}",
        f"- view fraction：{', '.join(f'{v:.1f}' for v in args.sample_fractions)}",
        f"- 对照 arm：{', '.join(args.arms)}（grl=GRL 10 + reco 30；nogrl=GRL 0 + reco 0）。",
        "- 评估候选池：始终使用全量 293 款游戏。",
        "- 测试文本：BG3、Cyberpunk、Across the Obelisk 的官方描述/长文本只在测试阶段使用；训练使用 train-only description cache。",
        f"- 每组合训练预算：epochs={args.epochs}, steps_per_epoch={args.steps_per_epoch}, batch_size={args.batch_size}。",
        "- 训练显存策略：split_recompute，把句嵌入 -> latentArray 的长序列段与后续层次降维分段反传，默认不截断 view。",
        "",
    ]
    if best:
        lines.extend([
            "## 结论",
            "",
            f"当前 sweep 的最佳综合窗口是 **dim={best['output_dim']}、arm={best['arm']}、N={best['train_games']}、view={best['view_fraction']:.1f}**。",
            f"综合分 {best['composite_score']:.3f}，TAG micro-F1 {best['tag_micro_f1']:.3f}，"
            f"身份 Hit@5 {best['hit_at_5']:.3f}，PR {best['pr']:.2f}，"
            f"情感 R² {best['sentiment_r2']:.3f}，好评率 Pearson {best['recommendation_pearson']:.3f}。",
            "",
            "综合分权重为：TAG 0.30、身份 Hit@5 0.30、同游戏情绪文本 cosine 0.15、PR 0.15，"
            "并对情感 R² 与好评率 Pearson 各扣 0.05。它不是论文指标，只用于窗口选择。",
            "",
        ])
    lines.extend([
        "## 数据量-性能曲线",
        "",
        "| dim | arm | N | view | score | TAG F1 | sentiment R² | reco Pearson | PR | mean rank | Hit@1 | Hit@5 | Hit@100 | text cosine |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(complete, key=lambda x: (x["output_dim"], x["arm"], x["train_games"], x["view_fraction"])):
        lines.append(
            f"| {row['output_dim']} | {row['arm']} | {row['train_games']} | {row['view_fraction']:.1f} | {row['composite_score']:.3f} | "
            f"{row['tag_micro_f1']:.3f} | {row['sentiment_r2']:.3f} | {row['recommendation_pearson']:.3f} | "
            f"{row['pr']:.2f} | {row['mean_rank']:.1f} | {row['hit_at_1']:.3f} | {row['hit_at_5']:.3f} | "
            f"{row['hit_at_100']:.3f} | {row['mean_text_cosine']:.3f} |"
        )

    lines.extend([
        "",
        "## GRL 对照差值",
        "",
        "| dim | N | view | Δscore | ΔTAG F1 | Δsentiment R² | Δabs(reco Pearson) | ΔHit@5 | ΔPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired:
        lines.append(
            f"| {row['output_dim']} | {row['train_games']} | {row['view_fraction']:.1f} | {row['delta_score']:+.3f} | "
            f"{row['delta_tag']:+.3f} | {row['delta_sentiment_r2']:+.3f} | "
            f"{row['delta_reco_pearson_abs']:+.3f} | {row['delta_hit_at_5']:+.3f} | {row['delta_pr']:+.2f} |"
        )
    lines.extend([
        "",
        "说明：Δ = GRL+reco - no-GRL+no-reco。对情感 R² 和 abs(reco Pearson)，负数表示前者更好；"
        "对 TAG、Hit@5、PR，正数表示前者更好。",
        "",
        "## View 最佳窗口预测",
        "",
        "| dim | arm | view | 平均 score | 平均 TAG F1 | 平均 Hit@5 | 平均 PR | 平均 sentiment R² |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for output_dim in args.output_dims:
        for arm in args.arms:
            for view in args.sample_fractions:
                subset = [
                    row for row in complete
                    if row["output_dim"] == output_dim
                    and row["arm"] == arm
                    and abs(row["view_fraction"] - view) < 1e-8
                ]
                if not subset:
                    continue
                lines.append(
                    f"| {output_dim} | {arm} | {view:.1f} | {np.mean([r['composite_score'] for r in subset]):.3f} | "
                    f"{np.mean([r['tag_micro_f1'] for r in subset]):.3f} | "
                    f"{np.mean([r['hit_at_5'] for r in subset]):.3f} | "
                    f"{np.mean([r['pr'] for r in subset]):.2f} | "
                    f"{np.mean([r['sentiment_r2'] for r in subset]):.3f} |"
                )

    lines.extend([
        "",
        "## 备注",
        "",
        "- N 是训练阶段可见的游戏数量，不是每款游戏的评论条数。",
        "- BG3、Cyberpunk、Across the Obelisk 三个锚点在每个训练子集里固定保留，以免身份召回测试变成“目标未见过”的外推问题。",
        "- 高 view 下少数游戏的随机窗口会超过十万句；训练端使用 split_recompute 分段反传来保留全量窗口，max_view_sentences 默认为 0。",
        "- 若某组合 status 不是 done，它不会进入上面的曲线均值；原始 JSON 保存在各组合目录。",
    ])
    return "\n".join(lines) + "\n"


def summarize(args) -> list[dict]:
    rows = []
    for output_dim in args.output_dims:
        for arm in args.arms:
            arm = arm_label(arm)
            for train_games in args.train_game_counts:
                for view in args.sample_fractions:
                    name = combo_name(output_dim, arm, train_games, view)
                    combo_dir = combo_dir_for(args, output_dim, arm, train_games, view)
                    eval_path = combo_dir / "eval_report.json"
                    manifest_path = combo_dir / "vicreg_review_h5_manifest.json"
                    status = manifest_status(manifest_path)
                    row = {
                        "output_dim": output_dim,
                        "arm": arm,
                        "combo": name,
                        "train_games": train_games,
                        "view_fraction": view,
                    }
                    checkpoint_path = combo_dir / "vicreg_review_h5_latest.pt"
                    report_current = report_is_current(eval_path, checkpoint_path, manifest_path, arm)
                    manifest_current = status == "done" and manifest_matches_arm(manifest_path, arm)
                    if report_current and manifest_current:
                        report = report_payload(eval_path)
                        if report is None:
                            row["status"] = "stale_eval"
                            rows.append(row)
                            continue
                        row.update(scalar_from_report(report))
                        row["composite_score"] = composite_score(row)
                        if status == "done":
                            row["status"] = "done"
                        elif status is None:
                            row["status"] = "evaluated_no_manifest"
                        else:
                            row["status"] = f"evaluated_{status}"
                    elif eval_path.exists():
                        row["status"] = "stale_eval"
                    elif manifest_path.exists():
                        if status == "done" and not manifest_matches_arm(manifest_path, arm):
                            row["status"] = "stale_manifest"
                        else:
                            row["status"] = "trained_done_missing_eval" if status == "done" else (status or "missing_eval")
                    else:
                        row["status"] = "missing"
                    rows.append(row)
    write_csv(rows, args.out_dir / "data_view_sweep_summary.csv")
    export_raw_detail_tables(args, rows)
    atomic_text_write(render_report(rows, args), args.out_dir / "DATA_VIEW_SWEEP_REPORT.md")
    atomic_json_write({"rows": rows, "args": vars_for_json(args)}, args.out_dir / "data_view_sweep_summary.json")
    return rows


def vars_for_json(args) -> dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            payload[key] = value
    return payload


def summarize_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_sweep_manifest(args, status: str, rows: list[dict] | None = None, current: dict | None = None, error: str | None = None) -> None:
    rows = rows or []
    payload = {
        "status": status,
        "started_at": getattr(args, "sweep_started_at", None),
        "updated_at": timestamp(),
        "out_dir": str(args.out_dir),
        "expected_combinations": len(args.output_dims) * len(args.train_game_counts) * len(args.sample_fractions) * len(args.arms),
        "done_combinations": sum(1 for row in rows if row.get("status") == "done"),
        "status_counts": summarize_counts(rows),
        "current": current,
        "error": error,
        "args": vars_for_json(args),
    }
    atomic_json_write(payload, args.out_dir / "sweep_manifest.json")


def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sweep_started_at = timestamp()
    write_sweep_manifest(args, "running", summarize(args))
    current = None
    try:
        for output_dim in args.output_dims:
            for train_games in args.train_game_counts:
                for view in args.sample_fractions:
                    current = {
                        "output_dim": output_dim,
                        "train_games": train_games,
                        "view_fraction": view,
                        "arms": [arm_label(arm) for arm in args.arms],
                    }
                    write_sweep_manifest(args, "running", summarize(args), current=current)
                    arms = [arm_label(arm) for arm in args.arms]
                    for arm in arms:
                        combo_paths(args, output_dim, arm, train_games, view)["dir"].mkdir(parents=True, exist_ok=True)

                    if not args.skip_train:
                        grl_pair = {"grl", "nogrl"}.issubset(set(arms))
                        both_need_fresh = (
                            grl_pair
                            and should_try_paired_training(output_dim, train_games, view)
                            and combo_needs_train(args, output_dim, "grl", train_games, view)
                            and combo_needs_train(args, output_dim, "nogrl", train_games, view)
                            and not is_resumable_partial(args, output_dim, "grl", train_games, view)
                            and not is_resumable_partial(args, output_dim, "nogrl", train_games, view)
                        )
                        if both_need_fresh:
                            try:
                                run_command(build_paired_train_command(args, output_dim, train_games, view), ROOT)
                            except subprocess.CalledProcessError as exc:
                                print(
                                    f"paired training failed for {combo_name(output_dim, 'grl', train_games, view)} / "
                                    f"{combo_name(output_dim, 'nogrl', train_games, view)}: {exc}; "
                                    "falling back to single-arm training",
                                    flush=True,
                                )
                                time.sleep(20)
                                for arm in arms:
                                    if combo_needs_train(args, output_dim, arm, train_games, view):
                                        paths = combo_paths(args, output_dim, arm, train_games, view)
                                        run_command(
                                            build_train_command(args, output_dim, arm, train_games, view, paths["dir"]),
                                            ROOT,
                                        )
                        else:
                            for arm in arms:
                                if combo_needs_train(args, output_dim, arm, train_games, view):
                                    paths = combo_paths(args, output_dim, arm, train_games, view)
                                    run_command(
                                        build_train_command(args, output_dim, arm, train_games, view, paths["dir"]),
                                        ROOT,
                                    )

                    if not args.skip_eval:
                        eval_targets = []
                        for arm in arms:
                            paths = combo_paths(args, output_dim, arm, train_games, view)
                            if paths["checkpoint"].exists():
                                eval_targets.append((paths["checkpoint"], paths["dir"]))
                        if {"grl", "nogrl"}.issubset(set(arms)):
                            evaluate_targets(args, eval_targets)
                        else:
                            for checkpoint, combo_dir in eval_targets:
                                evaluate_combo(args, checkpoint, combo_dir)
                    write_sweep_manifest(args, "running", summarize(args), current=current)
        rows = summarize(args)
        done = sum(1 for row in rows if row["status"] == "done")
        final_status = "done" if done == len(rows) else "incomplete"
        write_sweep_manifest(args, final_status, rows)
        print(f"sweep summary: {done}/{len(rows)} combinations done -> {args.out_dir}", flush=True)
    except KeyboardInterrupt:
        rows = summarize(args)
        write_sweep_manifest(args, "interrupted", rows, current=current, error="KeyboardInterrupt")
        raise
    except BaseException as exc:
        rows = summarize(args)
        write_sweep_manifest(args, "error", rows, current=current, error=repr(exc))
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON, type=Path, help="Python executable for child training runs.")
    parser.add_argument("--train-game-counts", type=int, nargs="+", default=[50, 100, 150, 200, 250, 293])
    parser.add_argument("--sample-fractions", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--output-dims", type=int, nargs="+", default=[18, 36, 72])
    parser.add_argument("--arms", nargs="+", default=["grl", "nogrl"], choices=["grl", "nogrl"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--max-view-sentences-80",
        type=int,
        default=0,
        help="Training-time per-game view sentence cap for view fractions >= 0.8. 0 disables the cap.",
    )
    parser.add_argument(
        "--max-view-sentences-60",
        type=int,
        default=0,
        help="Training-time per-game view sentence cap for view fractions >= 0.6 and < 0.8. 0 disables the cap.",
    )
    parser.add_argument(
        "--max-view-sentences-default",
        type=int,
        default=0,
        help="Training-time per-game view sentence cap for lower view fractions. 0 disables the cap.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-game-seed", type=int, default=20260626)
    parser.add_argument("--train-game-anchor-appids", default="1086940,1091500,1385380")
    parser.add_argument("--description-cache", default=DEFAULT_DESCRIPTION_CACHE, type=Path)
    parser.add_argument("--eval-feature-views", type=int, default=4)
    parser.add_argument("--eval-sample-fraction", type=float, default=0.6)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--max-game-sentences", type=int, default=4000)
    parser.add_argument("--max-text-sentences", type=int, default=4096)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--amp-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-eval", action="store_true")
    parser.add_argument(
        "--rebuild-shared-eval",
        action="store_true",
        help="Rebuild sweep-level raw/text evaluation caches. Per-combo eval caches still use --rebuild-eval.",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
