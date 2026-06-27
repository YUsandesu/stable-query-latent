"""Cloud/A100 entrypoint for the full review VICReg sweep.

This reuses run_data_view_sweep.py for evaluation, reporting, manifests, and
resume behavior, but replaces the local Windows training command defaults:

* full sweep grid by default: N increases by --train-game-step up to the H5
  pool, then adds full, views=0.8/0.6/0.4/0.2, dims=18/36/72,
  arms=grl/nogrl.
* no split_recompute path. The default backward mode is recompute; standard is
  available for an A100 if full-graph backprop fits.
* paired GRL/no-GRL training is attempted by default on fresh combinations.
* periodic in-training probes start at --train-probe-start-epoch and then run
  every --train-probe-every epochs by default, using the full aligned eval:
  sentiment, recommendation, anchor TAG generalization, and real-text TAG/cosine
  behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VICReg_review import run_data_view_sweep as sweep


DEFAULT_OUT_DIR = SCRIPT_DIR / "heads" / "cloud_full_sweep"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_TEXT_VARIANT_DIR = ROOT


_BASE_BUILD_TRAIN_COMMAND = sweep.build_train_command
_BASE_BUILD_PAIRED_TRAIN_COMMAND = sweep.build_paired_train_command
_BASE_SHOULD_TRY_PAIRED_TRAINING = sweep.should_try_paired_training
_PAIRED_MODE = "always"


def _set_option(cmd: list[str], option: str, value) -> None:
    value = str(value)
    try:
        index = cmd.index(option)
    except ValueError:
        cmd.extend([option, value])
    else:
        cmd[index + 1] = value


def _append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled and flag not in cmd:
        cmd.append(flag)


def _apply_cloud_train_options(cmd: list[str], args) -> list[str]:
    tag_split_json = args.tag_text_split_json or (args.out_dir / "tag_text_eval_split.json")
    text_variant_cache = args.text_variant_cache or (args.out_dir / "text_variant_embedding_cache.npz")
    _set_option(cmd, "--backward-mode", args.backward_mode)
    _set_option(cmd, "--prefetch-batches", args.prefetch_batches)
    _set_option(cmd, "--probe-every", args.train_probe_every)
    _set_option(cmd, "--probe-start-epoch", args.train_probe_start_epoch)
    _set_option(cmd, "--probe-feature-views", args.train_probe_feature_views)
    _set_option(cmd, "--probe-folds", args.train_probe_folds)
    _set_option(cmd, "--probe-sample-fraction", args.eval_sample_fraction)
    _set_option(cmd, "--text-variant-dir", args.text_variant_dir)
    _set_option(cmd, "--text-variant-feature-views", args.text_variant_feature_views)
    _set_option(cmd, "--text-variant-sample-fraction", args.text_variant_sample_fraction)
    _set_option(cmd, "--text-variant-local-model", args.local_model)
    _set_option(cmd, "--text-variant-embed-batch-size", args.embed_batch_size)
    _set_option(cmd, "--text-variant-max-sentences", args.max_text_sentences)
    _set_option(cmd, "--tag-text-split-json", tag_split_json)
    _set_option(cmd, "--tag-text-train-frac", args.tag_text_train_frac)
    _set_option(cmd, "--tag-text-val-frac", args.tag_text_val_frac)
    _set_option(cmd, "--tag-text-split-seed", args.tag_text_split_seed)
    _set_option(cmd, "--tag-text-threshold-steps", args.tag_text_threshold_steps)
    _set_option(cmd, "--text-variant-cache", text_variant_cache)
    _append_flag(cmd, "--pin-cache", args.pin_cache)
    _append_flag(cmd, "--rebuild-text-variant-cache", args.rebuild_text_variant_cache)
    return cmd


def build_train_command(args, output_dim: int, arm: str, train_games: int, view: float, combo_dir: Path) -> list[str]:
    cmd = _BASE_BUILD_TRAIN_COMMAND(args, output_dim, arm, train_games, view, combo_dir)
    _set_option(cmd, "--probe-history-tsv", combo_dir / "dual_probe_history.tsv")
    return _apply_cloud_train_options(cmd, args)


def build_paired_train_command(args, output_dim: int, train_games: int, view: float) -> list[str]:
    cmd = _BASE_BUILD_PAIRED_TRAIN_COMMAND(args, output_dim, train_games, view)
    return _apply_cloud_train_options(cmd, args)


def should_try_paired_training(output_dim: int, train_games: int, view: float) -> bool:
    if _PAIRED_MODE == "always":
        return True
    if _PAIRED_MODE == "never":
        return False
    return _BASE_SHOULD_TRY_PAIRED_TRAINING(output_dim, train_games, view)


def install_cloud_overrides(args) -> None:
    global _PAIRED_MODE
    _PAIRED_MODE = args.paired_mode
    sweep.build_train_command = build_train_command
    sweep.build_paired_train_command = build_paired_train_command
    sweep.should_try_paired_training = should_try_paired_training


def stepped_train_game_counts(pool: int, start: int, step: int, include_full: bool) -> list[int]:
    if pool <= 0:
        raise ValueError(f"H5 pool must be positive, got {pool}.")
    if start <= 0:
        raise ValueError("--train-game-start must be positive.")
    if step <= 0:
        raise ValueError("--train-game-step must be positive.")

    counts = []
    current = int(start)
    while current < pool:
        counts.append(current)
        current += int(step)
    if include_full:
        counts.append(0)
    return counts


def expand_train_game_counts(args) -> None:
    if args.train_game_counts is not None:
        return
    if not Path(args.h5).exists():
        raise SystemExit(
            f"H5 not found: {args.h5}\n"
            "Pass --h5 on the A100 server, or build game_review_data/embedding_h5.h5 first."
        )
    with sweep.h5py.File(args.h5, "r") as h5:
        sweep.validate_training_h5(h5, args.h5)
        pool = int(h5["game_names"].shape[0])
    args.train_game_counts = stepped_train_game_counts(
        pool,
        args.train_game_start,
        args.train_game_step,
        not args.no_full_count,
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=sweep.DEFAULT_H5, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON, type=Path, help="Python executable for child training runs.")
    parser.add_argument(
        "--train-game-counts",
        type=int,
        nargs="+",
        default=None,
        help="Explicit games-visible grid. Overrides --train-game-start/--train-game-step. 0 means full H5 pool.",
    )
    parser.add_argument("--train-game-start", type=int, default=50)
    parser.add_argument("--train-game-step", type=int, default=50)
    parser.add_argument("--no-full-count", action="store_true", help="Do not append full-pool training after stepped counts.")
    parser.add_argument("--sample-fractions", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--output-dims", type=int, nargs="+", default=[18, 36, 72])
    parser.add_argument("--arms", nargs="+", default=["grl", "nogrl"], choices=["grl", "nogrl"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--pin-cache", action="store_true")
    parser.add_argument(
        "--backward-mode",
        choices=["recompute", "standard"],
        default="recompute",
        help="Cloud training path. split_recompute is intentionally not available here.",
    )
    parser.add_argument(
        "--paired-mode",
        choices=["always", "auto", "never"],
        default="always",
        help="Whether to try paired GRL/no-GRL training for fresh combinations.",
    )
    parser.add_argument(
        "--train-probe-every",
        type=int,
        default=5,
        help="Run aligned periodic eval every N epochs. 0 disables periodic eval.",
    )
    parser.add_argument(
        "--train-probe-start-epoch",
        type=int,
        default=3,
        help="First epoch for aligned periodic eval; default gives 3,8,13,... with --train-probe-every 5.",
    )
    parser.add_argument("--train-probe-feature-views", type=int, default=2)
    parser.add_argument("--train-probe-folds", type=int, default=5)
    parser.add_argument(
        "--text-variant-dir",
        type=Path,
        default=DEFAULT_TEXT_VARIANT_DIR,
        help=(
            "Directory with real-text variants. Expected layout: "
            "<appid>/positive.txt, <appid>/neutral.txt, <appid>/negative.txt. "
            "The project-root AO/Cyberpunk legacy files are also recognized."
        ),
    )
    parser.add_argument("--text-variant-cache", type=Path, default=None)
    parser.add_argument("--rebuild-text-variant-cache", action="store_true")
    parser.add_argument("--text-variant-feature-views", type=int, default=4)
    parser.add_argument("--text-variant-sample-fraction", type=float, default=1.0)
    parser.add_argument("--tag-text-split-json", type=Path, default=None)
    parser.add_argument("--tag-text-train-frac", type=float, default=0.7)
    parser.add_argument("--tag-text-val-frac", type=float, default=0.15)
    parser.add_argument("--tag-text-split-seed", type=int, default=20260627)
    parser.add_argument("--tag-text-threshold-steps", type=int, default=33)
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
    parser.add_argument("--train-game-anchor-appids", default="1091500,1385380")
    parser.add_argument("--description-cache", default=sweep.DEFAULT_DESCRIPTION_CACHE, type=Path)
    parser.add_argument("--eval-feature-views", type=int, default=4)
    parser.add_argument("--eval-sample-fraction", type=float, default=0.6)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--max-game-sentences", type=int, default=4000)
    parser.add_argument("--max-text-sentences", type=int, default=4096)
    parser.add_argument("--local-model", default=sweep.DEFAULT_LOCAL_MODEL)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    expand_train_game_counts(args)
    if args.tag_text_split_json is None:
        args.tag_text_split_json = args.out_dir / "tag_text_eval_split.json"
    if args.text_variant_cache is None:
        args.text_variant_cache = args.out_dir / "text_variant_embedding_cache.npz"
    install_cloud_overrides(args)
    sweep.run(args)


if __name__ == "__main__":
    main()
