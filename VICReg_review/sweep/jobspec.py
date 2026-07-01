"""Translate a declarative Combo + memory plan into trainer argv + output paths.

This is the single place that turns the YAML config into the legacy trainer's
flags, so the 40-flag command line lives here (generated, not hand-typed).
Each job is single-arm; grl vs nogrl is encoded as the adversary / recommendation
weights, and the two arms share seed + sampled batches so the comparison stays
controlled without a separate paired trainer.
"""

from __future__ import annotations

import os
from pathlib import Path

# Loss constants for the current experiment (legacy run_data_view_sweep values).
# TODO: lift into the YAML `loss:` section if these become swept.
ADVERSARY_WEIGHT = 10.0
RECO_DECORR_WEIGHT = 30.0
COMPACT_VAR_WEIGHT = 25.0
COMPACT_COV_WEIGHT = 25.0


def effective_cpu_count() -> int:
    """Visible CPU cores, respecting cgroup/affinity (RunPod may pin a subset).
    Kept torch-free so the supervisor can call it without a CUDA import."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def auto_data_workers(n_lanes: int = 1) -> int:
    """Per-lane parallel H5-read procs, auto-scaled to the machine: the trainer's
    convention (cores-1, capped at 16) divided across the GPU lanes, so N lanes
    don't collectively oversubscribe the CPU. Not a YAML knob -- it follows the
    hardware and how many GPUs the sweep is spread over."""
    n_lanes = max(1, int(n_lanes))
    return max(1, min(16, (effective_cpu_count() - 1) // n_lanes))


def combo_dir(config, combo) -> Path:
    return Path(config.out_dir) / combo.combo_id


def combo_paths(config, combo) -> dict:
    d = combo_dir(config, combo)
    return {
        "dir": d,
        "checkpoint": d / "vicreg_review_h5_latest.pt",
        "best": d / "vicreg_review_h5_best.pt",
        "history": d / "vicreg_review_h5_history.tsv",
        "manifest": d / "vicreg_review_h5_manifest.json",
        "probe_history": d / "dual_probe_history.tsv",
    }


def build_trainer_argv(config, combo, settings: dict, *, device: str = "cuda",
                       probe_queue_dir=None, resume: bool = True,
                       data_workers: int | None = None, vm_name: str | None = None) -> list[str]:
    paths = combo_paths(config, combo)
    grl = combo.arm == "grl"
    cache_mode = settings.get("cache_mode", "full")
    pin_cache = bool(settings.get("pin_cache", True))
    prefetch = int(settings.get("prefetch_batches", 2) or 2)
    # data_workers is auto-scaled by the supervisor (cores / lane-count); it is
    # not read from the YAML. Fall back to auto for a single lane if unspecified.
    dw = int(data_workers) if data_workers is not None else auto_data_workers(1)
    argv = [
        "--input-h5", str(config.h5),
        "--device", str(device), "--amp",
        "--epochs", str(config.train.epochs),
        "--steps-per-epoch", "0",
        "--batch-size", str(config.train.batch_size),
        "--sample-fraction", f"{combo.view:g}",
        "--train-game-count", str(combo.train_games),
        "--train-game-seed", str(config.data_seed.train_game_seed),
        "--train-game-anchor-appids", ",".join(str(a) for a in config.data_seed.anchors),
        "--encoder-arch", "hierarchical",
        "--latent-dim", str(config.model.latent_dim),
        "--num-latents", str(combo.num_latents),
        "--output-dim", str(combo.output_dim),
        "--reduce-hidden", ",".join(str(h) for h in config.model.reduce_hidden),
        "--vicreg-scope", "game",
        "--expander-dim", str(config.model.expander_dim),
        "--expander-hidden", ",".join(str(h) for h in config.model.expander_hidden),
        "--compact-variance-weight", str(COMPACT_VAR_WEIGHT),
        "--compact-covariance-weight", str(COMPACT_COV_WEIGHT),
        "--recommendation-target-transform", "logit",
        "--adversary-weight", str(ADVERSARY_WEIGHT if grl else 0.0),
        "--recommendation-decorr-weight", str(RECO_DECORR_WEIGHT if grl else 0.0),
        "--backward-mode", str(settings.get("backward_mode", "standard")),
        "--stem-chunk-size", str(int(settings.get("stem_chunk_size", 0) or 0)),
        "--cache-mode", cache_mode,
        "--cache-dtype", "float16",
        "--prefetch-batches", str(prefetch),
        "--data-workers", str(dw),
        "--seed", str(config.train.seed),
        "--checkpoint-out", str(paths["checkpoint"]),
        "--best-checkpoint-out", str(paths["best"]),
        "--history-tsv", str(paths["history"]),
        "--manifest-json", str(paths["manifest"]),
        "--probe-history-tsv", str(paths["probe_history"]),
        "--probe-every", str(config.probe.every),
        "--probe-start-epoch", str(config.probe.start_epoch),
    ]
    if pin_cache:
        argv.append("--pin-cache")
    if probe_queue_dir:
        argv += ["--probe-queue-dir", str(probe_queue_dir)]
    if resume and paths["checkpoint"].exists():
        argv += ["--resume-checkpoint", str(paths["checkpoint"])]
    if vm_name:
        # Per-epoch fence: the trainer aborts if this combo's status.json stops
        # naming us (a peer reclaimed it after our lease lapsed). Status lives next
        # to the checkpoint, written by the coordinator.
        argv += ["--fence-status", str(paths["dir"] / "status.json"), "--fence-vm", str(vm_name)]
    return argv
