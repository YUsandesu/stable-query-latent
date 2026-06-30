"""Cloud/A100 entrypoint for the full review VICReg sweep.

This reuses run_data_view_sweep.py for evaluation, reporting, manifests, and
resume behavior, but replaces the local Windows training command defaults:

* full sweep grid by default: N increases by --train-game-step up to the H5
  pool, then adds full, views=0.8/0.6/0.4/0.2, dims=18/36/72,
  arms=grl/nogrl.
* full-window training defaults to auto backward mode: try standard first for
  A100 throughput, then retry CUDA OOM combinations with split_recompute.
* paired GRL/no-GRL training is attempted by default on fresh combinations.
* periodic in-training probes are disabled by default for throughput. Enable
  --train-probe-every when you want learning curves. The expensive raw-Qwen /
  VICReg comparison runs once at the end on the best full-train checkpoint by
  default; use --eval-mode per_combo for the old behavior.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.logging_tee import run_with_optional_tee
from VICReg_review import run_data_view_sweep as sweep
from VICReg_review import oom_proxy
from VICReg_review.train_vicreg_review_h5 import parse_int_list


DEFAULT_OUT_DIR = SCRIPT_DIR / "heads" / "cloud_full_sweep"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_TEXT_VARIANT_DIR = ROOT
DEFAULT_DESCRIPTIONS_DIR = SCRIPT_DIR / "tags" / "game_descriptions"


_BASE_BUILD_TRAIN_COMMAND = sweep.build_train_command
_BASE_BUILD_PAIRED_TRAIN_COMMAND = sweep.build_paired_train_command
_BASE_SHOULD_TRY_PAIRED_TRAINING = sweep.should_try_paired_training
_BASE_RUN_COMMAND = sweep.run_command
_PAIRED_MODE = "always"
_STANDARD_OOM_RETRY = True
_STANDARD_OOM_FALLBACK = "split_recompute"
_VRAM_FALLBACK_REPORTED = False
# Calibration / OOM-budget state (populated by install_calib when --calib-mode != off).
_CALIB = None
_GAME_STATS = None
_FREE_VRAM_BYTES = 0.0
_VRAM_SAFETY = 0.85
_CALIB_ARGS = None
AUTO_FULL_PASS_MIN_TOTAL_VRAM_GIB = 60.0
AUTO_FULL_PASS_MIN_FREE_VRAM_GIB = 50.0
AUTO_STEP_TIERS = (
    (100.0, 0),
    (70.0, 0),
    (40.0, 8),
    (20.0, 4),
)


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


def _extract_option(cmd: list[str], option: str) -> str | None:
    try:
        index = cmd.index(option)
    except ValueError:
        return None
    if index + 1 >= len(cmd):
        return None
    return str(cmd[index + 1])


def _cuda_memory_gib(device_text: str | None) -> tuple[float, float] | None:
    device_text = str(device_text or "")
    if not device_text.startswith("cuda"):
        return None
    torch = sweep.torch
    if not torch.cuda.is_available():
        return None
    try:
        if ":" in device_text:
            device_index = int(device_text.split(":", 1)[1])
        else:
            device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        total_bytes = max(int(total_bytes), int(props.total_memory))
        return free_bytes / 1024**3, total_bytes / 1024**3
    except (RuntimeError, ValueError) as exc:
        print(f"auto backward-mode: could not read CUDA memory for {device_text!r}: {exc}", flush=True)
        return None


def _read_int_file(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _system_memory_gib() -> tuple[float | None, float | None]:
    """Return (available/free-ish, total) RAM in GiB, respecting cgroup limits when visible."""
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None
    phys_pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None
    avail_pages = os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") else None
    total_bytes = int(page_size * phys_pages) if page_size and phys_pages else None
    avail_bytes = int(page_size * avail_pages) if page_size and avail_pages else None

    cgroup_limit = _read_int_file("/sys/fs/cgroup/memory.max")
    if cgroup_limit is None:
        cgroup_limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cgroup_limit is not None and 0 < cgroup_limit < (1 << 60):
        total_bytes = min(total_bytes, cgroup_limit) if total_bytes else cgroup_limit
        current = _read_int_file("/sys/fs/cgroup/memory.current")
        if current is None:
            current = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if current is not None:
            avail_bytes = max(0, cgroup_limit - current)

    if total_bytes is None:
        return None, None
    if avail_bytes is None:
        avail_bytes = total_bytes
    return avail_bytes / 1024**3, total_bytes / 1024**3


def _steps_for_resource_gib(resource_gib: float) -> int:
    for threshold_gib, steps in AUTO_STEP_TIERS:
        if resource_gib >= threshold_gib:
            return steps
    return 4


def resolve_auto_steps_per_epoch(args) -> None:
    if args.steps_per_epoch is not None:
        return

    cuda_memory = _cuda_memory_gib(args.device)
    ram_free_gib, ram_total_gib = _system_memory_gib()
    resource_candidates = []
    if cuda_memory is not None:
        gpu_free_gib, gpu_total_gib = cuda_memory
        resource_candidates.extend([gpu_free_gib, gpu_total_gib])
    else:
        gpu_free_gib = gpu_total_gib = None
    if ram_free_gib is not None:
        resource_candidates.append(ram_free_gib)
    if ram_total_gib is not None:
        resource_candidates.append(ram_total_gib)

    resource_gib = min(resource_candidates) if resource_candidates else 0.0
    args.steps_per_epoch = _steps_for_resource_gib(resource_gib)
    mode = "full-pass" if args.steps_per_epoch <= 0 else f"{args.steps_per_epoch} steps/epoch"
    if gpu_free_gib is not None:
        print(f"auto steps_per_epoch: gpu_free={gpu_free_gib:.1f}GiB ", end="", flush=True)
    else:
        print("auto steps_per_epoch: gpu=unavailable ", end="", flush=True)
    if gpu_total_gib is not None:
        print(f"gpu_total={gpu_total_gib:.1f}GiB ", end="", flush=True)
    if ram_free_gib is not None:
        print(f"ram_free={ram_free_gib:.1f}GiB ", end="", flush=True)
    if ram_total_gib is not None:
        print(f"ram_total={ram_total_gib:.1f}GiB ", end="", flush=True)
    print(f"-> {mode}", flush=True)


def _select_backward_mode(args, view: float, latent_scale: float) -> str:
    global _VRAM_FALLBACK_REPORTED

    requested = str(args.backward_mode)
    if requested != "auto":
        return requested

    memory = _cuda_memory_gib(args.device)
    if memory is None:
        if not _VRAM_FALLBACK_REPORTED:
            print(
                "auto backward-mode: CUDA memory unavailable; using "
                f"{_STANDARD_OOM_FALLBACK}",
                flush=True,
            )
            _VRAM_FALLBACK_REPORTED = True
        return _STANDARD_OOM_FALLBACK

    free_gib, total_gib = memory
    if total_gib < AUTO_FULL_PASS_MIN_TOTAL_VRAM_GIB or free_gib < AUTO_FULL_PASS_MIN_FREE_VRAM_GIB:
        if not _VRAM_FALLBACK_REPORTED:
            print(
                "auto backward-mode: CUDA memory below standard threshold "
                f"(free={free_gib:.1f}GiB total={total_gib:.1f}GiB; "
                f"need free>={AUTO_FULL_PASS_MIN_FREE_VRAM_GIB:g}GiB "
                f"and total>={AUTO_FULL_PASS_MIN_TOTAL_VRAM_GIB:g}GiB); "
                f"using {_STANDARD_OOM_FALLBACK}",
                flush=True,
            )
            _VRAM_FALLBACK_REPORTED = True
        return _STANDARD_OOM_FALLBACK

    return "standard"


def _num_latents_for(args, latent_scale: float) -> int:
    return sweep.effective_num_latents(args, latent_scale)


def install_calib(args) -> None:
    """Build or load the (num_latents, mode) -> (C, R) calibration and game stats.

    See VICReg_review/oom_proxy.py for the memory model. When --calib-mode is
    'off' (or calibration is unavailable) the sweep falls back to the legacy
    VRAM-snapshot backward-mode selection.
    """
    global _CALIB, _GAME_STATS, _FREE_VRAM_BYTES, _VRAM_SAFETY, _CALIB_ARGS
    _CALIB_ARGS = args
    _VRAM_SAFETY = float(getattr(args, "vram_safety", 0.85))
    mode = str(getattr(args, "calib_mode", "off"))
    if mode == "off":
        _CALIB = None
        return

    calib_json = getattr(args, "calib_json", None) or (args.out_dir / "calib.json")
    args.calib_json = calib_json
    nl_list = oom_proxy._num_latents_list(args.base_num_latents, args.latent_scales)
    modes = ("standard", _STANDARD_OOM_FALLBACK)

    try:
        if mode == "measure":
            print(f"calib: measuring (C,R) for num_latents={nl_list} modes={list(modes)}", flush=True)
            _CALIB = oom_proxy.calibrate(
                args.h5, nl_list, modes,
                device=args.device, amp=True,
                batch_size=args.batch_size, output_dim=max(args.output_dims),
            )
            oom_proxy.save_calib(_CALIB, calib_json)
            print(f"calib: written {calib_json}", flush=True)
        else:  # load
            _CALIB = oom_proxy.load_calib(calib_json)
            if _CALIB is None:
                print(f"calib: {calib_json} not found; falling back to VRAM-snapshot mode", flush=True)
                return

        _GAME_STATS = oom_proxy.GameStats.from_h5(args.h5)
        _FREE_VRAM_BYTES = oom_proxy._free_vram_bytes(args.device)
    except BaseException as exc:  # never let calibration abort the sweep
        print(f"calib: disabled after error ({type(exc).__name__}: {exc}); "
              "using VRAM-snapshot backward-mode selection", flush=True)
        _CALIB = None
        _GAME_STATS = None
        return
    print(
        f"calib: ready (free_vram={_FREE_VRAM_BYTES / oom_proxy.GIB:.1f}GiB "
        f"safety={_VRAM_SAFETY})",
        flush=True,
    )


def _plan_for(view: float, train_games: int, latent_scale: float, try_paired: bool):
    """Per-combo memory plan, or None when calibration is unavailable."""
    if _CALIB is None or _GAME_STATS is None or _CALIB_ARGS is None:
        return None
    args = _CALIB_ARGS
    num_latents = _num_latents_for(args, latent_scale)
    worst = _GAME_STATS.subset_worst_sentences(
        train_games, args.train_game_seed, args.train_game_anchor_appids
    )
    return oom_proxy.plan_combo(
        _CALIB, worst, _FREE_VRAM_BYTES, num_latents, view, args.batch_size,
        modes=("standard", _STANDARD_OOM_FALLBACK), safety=_VRAM_SAFETY,
        try_paired=try_paired,
    )


def _apply_plan_caps(cmd: list[str], plan) -> None:
    if not plan:
        return
    if int(plan.get("max_batch_sentences", 0)) > 0:
        _set_option(cmd, "--max-batch-sentences", plan["max_batch_sentences"])
    if int(plan.get("max_view_sentences", 0)) > 0:
        _set_option(cmd, "--max-view-sentences", plan["max_view_sentences"])


def _manifest_error_is_oom(path: str | None) -> bool:
    if not path:
        return False
    payload = sweep.manifest_payload(Path(path))
    if not payload:
        return False
    error = str(payload.get("error", "")).lower()
    return "outofmemory" in error or "out of memory" in error or "cuda oom" in error


def _command_had_cuda_oom(cmd: list[str]) -> bool:
    manifest_options = (
        "--manifest-json",
        "--grl-manifest-json",
        "--nogrl-manifest-json",
    )
    return any(_manifest_error_is_oom(_extract_option(cmd, option)) for option in manifest_options)


def _log_standard_oom(cmd: list[str]) -> None:
    try:
        num_latents = int(_extract_option(cmd, "--num-latents") or 0)
        view = float(_extract_option(cmd, "--sample-fraction") or 0.0)
    except ValueError:
        return
    details = []
    if num_latents > 0:
        details.append(f"num_latents={num_latents}")
    if view > 0:
        details.append(f"view={view:g}")
    suffix = f" ({', '.join(details)})" if details else ""
    print(
        "auto backward-mode: standard OOM observed for current combo"
        f"{suffix}; retrying this combo with {_STANDARD_OOM_FALLBACK}. "
        "Next combo will re-evaluate auto mode from scratch.",
        flush=True,
    )


def run_command_with_standard_oom_retry(cmd: list[str], cwd: Path) -> None:
    try:
        _BASE_RUN_COMMAND(cmd, cwd)
        return
    except sweep.subprocess.CalledProcessError:
        if (
            not _STANDARD_OOM_RETRY
            or _extract_option(cmd, "--backward-mode") != "standard"
            or _STANDARD_OOM_FALLBACK == "standard"
            or not _command_had_cuda_oom(cmd)
        ):
            raise

    _log_standard_oom(cmd)
    retry_cmd = list(cmd)
    _set_option(retry_cmd, "--backward-mode", _STANDARD_OOM_FALLBACK)
    print(
        f"auto backward-mode: retrying CUDA OOM combo with --backward-mode {_STANDARD_OOM_FALLBACK}",
        flush=True,
    )
    _BASE_RUN_COMMAND(retry_cmd, cwd)


def _apply_cloud_train_options(cmd: list[str], args, backward_mode: str) -> list[str]:
    tag_split_json = args.tag_text_split_json or (args.out_dir / "tag_text_eval_split.json")
    text_variant_cache = args.text_variant_cache or (args.out_dir / "text_variant_embedding_cache.npz")
    test_case_cache = args.test_case_cache or (args.out_dir / "test_case_embeddings.npz")
    _set_option(cmd, "--cache-mode", args.cache_mode)
    _set_option(cmd, "--backward-mode", backward_mode)
    _set_option(cmd, "--prefetch-batches", args.prefetch_batches)
    if args.max_batch_sentences > 0:
        _set_option(cmd, "--max-batch-sentences", args.max_batch_sentences)
    _set_option(cmd, "--probe-every", args.train_probe_every)
    if args.probe_queue_dir:
        _set_option(cmd, "--probe-queue-dir", args.probe_queue_dir)
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
    _set_option(cmd, "--test-case-cache", test_case_cache)
    _append_flag(cmd, "--pin-cache", args.pin_cache)
    _append_flag(cmd, "--rebuild-text-variant-cache", args.rebuild_text_variant_cache)
    _append_flag(cmd, "--rebuild-test-case-cache", args.rebuild_shared_eval)
    return cmd


def build_train_command(args, output_dim: int, arm: str, train_games: int, view: float, combo_dir: Path, latent_scale: float = 1.0) -> list[str]:
    cmd = _BASE_BUILD_TRAIN_COMMAND(args, output_dim, arm, train_games, view, combo_dir, latent_scale)
    _set_option(cmd, "--probe-history-tsv", combo_dir / "dual_probe_history.tsv")
    plan = _plan_for(view, train_games, latent_scale, try_paired=False)
    backward_mode = plan["backward_mode"] if plan else _select_backward_mode(args, view, latent_scale)
    cmd = _apply_cloud_train_options(cmd, args, backward_mode)
    _apply_plan_caps(cmd, plan)
    return cmd


def build_paired_train_command(args, output_dim: int, train_games: int, view: float, latent_scale: float = 1.0) -> list[str]:
    cmd = _BASE_BUILD_PAIRED_TRAIN_COMMAND(args, output_dim, train_games, view, latent_scale)
    plan = _plan_for(view, train_games, latent_scale, try_paired=True)
    backward_mode = plan["backward_mode"] if plan else _select_backward_mode(args, view, latent_scale)
    cmd = _apply_cloud_train_options(cmd, args, backward_mode)
    _apply_plan_caps(cmd, plan)
    return cmd


def should_try_paired_training(output_dim: int, train_games: int, view: float, latent_scale: float = 1.0) -> bool:
    if _PAIRED_MODE == "never":
        return False
    plan = _plan_for(view, train_games, latent_scale, try_paired=True)
    if plan is not None:
        # Calibration decides: only pair when the budget fits two arms.
        return bool(plan["paired"])
    if _PAIRED_MODE == "always":
        return True
    return _BASE_SHOULD_TRY_PAIRED_TRAINING(output_dim, train_games, view)


def install_cloud_overrides(args) -> None:
    global _PAIRED_MODE
    _PAIRED_MODE = args.paired_mode
    install_calib(args)
    sweep.build_train_command = build_train_command
    sweep.build_paired_train_command = build_paired_train_command
    sweep.should_try_paired_training = should_try_paired_training
    sweep.run_command = run_command_with_standard_oom_retry


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
    parser.add_argument(
        "--latent-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Multipliers for --base-num-latents. Example: 1 2 4 runs "
            "256/512/1024 latent slots when --base-num-latents is 256."
        ),
    )
    parser.add_argument("--base-num-latents", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--expander-dim", type=int, default=128)
    parser.add_argument(
        "--expander-hidden",
        type=parse_int_list,
        default=(128,),
        help="Comma-separated hidden widths for the game-centroid expander.",
    )
    parser.add_argument("--arms", nargs="+", default=["grl", "nogrl"], choices=["grl", "nogrl"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help=(
            "Batches per epoch. Omit to auto-tier by RAM/VRAM: <40GiB keeps 4, "
            "40-70GiB uses 8, >=70GiB uses full-pass. Explicit 0 forces full-pass."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--cache-mode", choices=["queue", "full"], default="full")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--pin-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-batch-sentences",
        type=int,
        default=0,
        help=(
            "Optional original-sentence budget per training batch. Use 0 for fixed "
            "--batch-size; set this to smooth extreme long-game batches."
        ),
    )
    parser.add_argument(
        "--backward-mode",
        choices=["auto", "split_recompute", "recompute", "standard"],
        default="auto",
        help=(
            "Cloud training path. auto tries standard for A100 throughput and "
            "falls back to split_recompute after CUDA OOM."
        ),
    )
    parser.add_argument(
        "--paired-mode",
        choices=["always", "auto", "never"],
        default="always",
        help="Whether to try paired GRL/no-GRL training for fresh combinations.",
    )
    parser.add_argument(
        "--calib-mode",
        choices=["off", "load", "measure"],
        default="measure",
        help=(
            "OOM-budget calibration (VICReg_review/oom_proxy.py). 'measure' runs a "
            "pseudo-batch warm-up per num_latents/mode at startup to fit peak=R+C*S, "
            "then plans per-combo backward-mode / paired / sentence caps. 'load' reuses "
            "a prior calib.json. 'off' falls back to the legacy VRAM-snapshot selection."
        ),
    )
    parser.add_argument(
        "--calib-json",
        type=Path,
        default=None,
        help="Calibration cache path. Defaults to <out-dir>/calib.json.",
    )
    parser.add_argument(
        "--vram-safety",
        type=float,
        default=0.85,
        help="Fraction of free VRAM the planner may budget (leaves room for context/fragmentation).",
    )
    parser.add_argument(
        "--train-probe-every",
        type=int,
        default=0,
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
        "--probe-queue-dir",
        type=Path,
        default=None,
        help=(
            "If set, in-training probes run decoupled: each probe epoch emits a "
            "slim checkpoint + queue marker here instead of probing inline. Run "
            "VICReg_review/probe_worker.py against the same dir to consume them. "
            "Forwarded to both single-arm and paired training commands."
        ),
    )
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
    parser.add_argument("--test-case-cache", type=Path, default=None)
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
    parser.add_argument("--eval-feature-views", type=int, default=4)
    parser.add_argument("--eval-sample-fraction", type=float, default=0.6)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--max-game-sentences", type=int, default=4000)
    parser.add_argument(
        "--raw-cache-workers",
        type=int,
        default=0,
        help="Workers for raw mean-pool cache. 0 uses CPU cores minus 1; 1 disables multiprocessing.",
    )
    parser.add_argument("--descriptions-dir", type=Path, default=DEFAULT_DESCRIPTIONS_DIR)
    parser.add_argument("--max-description-sentences", type=int, default=256)
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
    parser.add_argument(
        "--prepare-shared-eval-only",
        action="store_true",
        help="Build shared raw/text evaluation caches, then exit before training or per-combo evaluation.",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--eval-mode",
        choices=["per_combo", "final_best", "none"],
        default="final_best",
        help=(
            "Cloud default is final_best: train the sweep first, then run the "
            "expensive raw-Qwen/VICReg comparison once on the best full-train checkpoint."
        ),
    )
    parser.add_argument("--skip-eval", action="store_true", help="Deprecated alias for --eval-mode none.")
    parser.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    args = parser.parse_args(argv)
    if args.skip_eval:
        args.eval_mode = "none"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_with_optional_tee(args.logout_address, run_main, args)


def run_main(args) -> None:
    expand_train_game_counts(args)
    resolve_auto_steps_per_epoch(args)
    if args.tag_text_split_json is None:
        args.tag_text_split_json = args.out_dir / "tag_text_eval_split.json"
    if args.text_variant_cache is None:
        args.text_variant_cache = args.out_dir / "text_variant_embedding_cache.npz"
    if args.test_case_cache is None:
        args.test_case_cache = args.out_dir / "test_case_embeddings.npz"
    install_cloud_overrides(args)
    sweep.run(args)


if __name__ == "__main__":
    main()
