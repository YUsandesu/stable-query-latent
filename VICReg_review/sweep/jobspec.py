"""Translate a declarative Combo + memory plan into trainer argv + output paths.

This is the single place that turns the YAML config into the legacy trainer's
flags, so the 40-flag command line lives here (generated, not hand-typed).
Each job is single-arm; grl vs nogrl is encoded as the adversary / recommendation
weights, and the two arms share seed + sampled batches so the comparison stays
controlled without a separate paired trainer.
"""

from __future__ import annotations

from pathlib import Path

# Loss constants for the current experiment (legacy run_data_view_sweep values).
# TODO: lift into the YAML `loss:` section if these become swept.
ADVERSARY_WEIGHT = 10.0
RECO_DECORR_WEIGHT = 30.0
COMPACT_VAR_WEIGHT = 25.0
COMPACT_COV_WEIGHT = 25.0


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
                       probe_queue_dir=None, resume: bool = True) -> list[str]:
    paths = combo_paths(config, combo)
    grl = combo.arm == "grl"
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
        "--cache-mode", "full",
        "--cache-dtype", "float16",
        "--pin-cache",
        "--seed", str(config.train.seed),
        "--checkpoint-out", str(paths["checkpoint"]),
        "--best-checkpoint-out", str(paths["best"]),
        "--history-tsv", str(paths["history"]),
        "--manifest-json", str(paths["manifest"]),
        "--probe-history-tsv", str(paths["probe_history"]),
        "--probe-every", str(config.probe.every),
        "--probe-start-epoch", str(config.probe.start_epoch),
    ]
    if probe_queue_dir:
        argv += ["--probe-queue-dir", str(probe_queue_dir)]
    if resume and paths["checkpoint"].exists():
        argv += ["--resume-checkpoint", str(paths["checkpoint"])]
    return argv
