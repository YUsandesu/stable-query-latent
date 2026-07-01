"""OOM budget planner + calibration for the cloud VICReg review sweep.

Memory model (per combo, single arm, one training step)::

    peak_bytes ~= R + C * S

where ``S`` is the total number of sentences forwarded through the encoder in
one batch, counting *both* augmented views (``S = sum(len_a) + sum(len_b)``).

* ``R`` (intercept) is the fixed footprint: model weights + optimizer state +
  resident buffers measured by the allocator. It does **not** include the CUDA
  context / cuDNN workspace (those show up in ``mem_get_info`` but not in
  ``max_memory_allocated``), which is why the budget keeps a safety margin.
* ``C`` (slope) is the activation bytes retained per forwarded sentence.

Both ``R`` and ``C`` depend on ``(num_latents, backward_mode)``. ``C`` grows
with ``num_latents`` because the sentence->latent cross-attention is
``O(sentences * num_latents)``; ``standard`` keeps the whole graph so its ``C``
is much larger than ``split_recompute``'s.

Why this needs measurement, not derivation: the *input* (a batch of fixed
``input_dim`` vectors) is exactly computable, but GPU peak is dominated by
intermediate activations that are many multiples of the raw input, and that
multiplier is set by model internals / AMP / the autograd graph. So we measure
the slope once and extrapolate.

Two entry points:

* :func:`calibrate` — pseudo-batch warm-up. For each ``(num_latents,
  backward_mode)`` it runs the **real** training step (:func:`run_training_batch`)
  at a few known ``S`` values and least-squares fits ``(C, R)``. Synthetic random
  data is valid because GPU memory depends on tensor *shapes*, not values.
* :func:`plan_combo` — from ``(C, R)`` + measured free VRAM + the worst single
  game in the train subset, decide per combo: ``paired?``, ``backward_mode``,
  ``max_batch_sentences``, ``max_view_sentences``. ``paired`` is modelled as
  ~2x the single-arm footprint (two models + two optimizers + both graphs).

Runnable as a CLI to (a) calibrate standalone and (b) print a dry-run memory
plan over a grid before committing to a multi-hour sweep::

    python VICReg_review/oom_proxy.py --h5 game_review_data/embedding_h5.h5 \
        --calib-out VICReg_review/heads/cloud_full_sweep_a100/calib.json \
        --measure --plan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from VICReg_review.train_vicreg_review_h5 import (  # noqa: E402
    build_training_components,
    decode_name,
    game_review_sentence_counts,
    parse_string_list,
    parse_args as train_parse_args,
    run_training_batch,
)

GIB = 1024.0 ** 3

# Architecture / loss flags that affect the memory footprint. These mirror the
# constants emitted by run_data_view_sweep.build_train_command so the calibrated
# numbers match the real training runs. output_dim has a minor effect; calibrate
# at the largest grid value for a conservative (slightly high) C.
ARCH_FLAGS = {
    "--encoder-arch": "hierarchical",
    "--latent-dim": "256",
    "--reduce-hidden": "128",
    "--vicreg-scope": "game",
    "--expander-dim": "128",
    "--expander-hidden": "128",
    "--compact-variance-weight": "25",
    "--compact-covariance-weight": "25",
    "--recommendation-decorr-weight": "30",
    "--recommendation-target-transform": "logit",
    "--adversary-weight": "10",
    "--cache-dtype": "float16",
}

DEFAULT_MODES = ("standard", "split_recompute")
# Conservative stem chunk used when a combo has no calibration entry (keeps the
# chunked stem bounded rather than running an unchunked, possibly-OOM forward).
NO_CALIB_CHUNK = 2048


# Host-RAM accounting lives in the torch-free mem_budget module so the notebook
# monitor can share the exact same definition without importing torch/the trainer.
from VICReg_review.mem_budget import (  # noqa: E402
    _read_cgroup_stat,
    _read_int_file,
    available_ram_bytes,
)


def estimate_full_cache_bytes(total_sentences: int, view: float, input_dim: int,
                              *, prefetch_factor: float = 2.0, dtype_bytes: int = 2) -> float:
    """Approx host RAM for --cache-mode full: ~one epoch of sampled views (both
    a/b) materialised, plus the prefetched next epoch. dtype float16 = 2 bytes."""
    return float(total_sentences) * float(view) * 2.0 * int(input_dim) * dtype_bytes * float(prefetch_factor)
# Total forwarded sentences (both views) probed per calibration point. Kept
# small so calibration never OOMs even at num_latents=1024; >=2 successes are
# enough to fit the line.
DEFAULT_CALIB_POINTS = (2000, 5000, 9000)
CALIB_N_GAMES = 4  # held fixed so R (intercept) is consistent across points


def _calib_key(num_latents: int, mode: str) -> str:
    return f"{int(num_latents)}|{mode}"


def build_calib_args(h5_path, num_latents: int, backward_mode: str, batch_size: int,
                     device: str, amp: bool, output_dim: int):
    """A fully-populated trainer args namespace for a calibration combo.

    Reuses the real ``train_vicreg_review_h5`` parser so every field exists with
    its production default; only the memory-relevant knobs are overridden.
    """
    argv = ["--input-h5", str(h5_path),
            "--device", str(device),
            "--num-latents", str(int(num_latents)),
            "--output-dim", str(int(output_dim)),
            "--batch-size", str(int(batch_size)),
            "--backward-mode", str(backward_mode)]
    for flag, value in ARCH_FLAGS.items():
        argv += [flag, value]
    if amp:
        argv.append("--amp")
    return train_parse_args(argv)


def _pseudo_batch(total_per_view: int, input_dim: int, n_games: int, dtype):
    n_games = max(1, min(n_games, total_per_view))
    base = total_per_view // n_games
    counts = [base] * (n_games - 1) + [total_per_view - base * (n_games - 1)]
    counts = [max(1, c) for c in counts]

    def views():
        return [torch.randn(c, input_dim, dtype=dtype) for c in counts]

    return {
        "view_a": views(),
        "view_b": views(),
        "len_a": torch.tensor(counts, dtype=torch.long),
        "len_b": torch.tensor(counts, dtype=torch.long),
    }


def measure_peak_bytes(args, input_dim: int, device: torch.device,
                       total_per_view: int, n_games: int = CALIB_N_GAMES) -> int:
    """Run one real training step on a synthetic batch; return peak alloc bytes."""
    dtype = torch.float16 if str(args.cache_dtype) == "float16" else torch.float32
    batch = _pseudo_batch(total_per_view, input_dim, n_games, dtype)
    model, expander, adversary, optimizer = build_training_components(args, input_dim, device)
    model.train()
    if expander is not None:
        expander.train()
    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pin_transfer = bool(args.pin_cache) and device.type == "cuda"
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        run_training_batch(batch, model, expander, adversary, optimizer, scaler,
                            args, device, amp_enabled, pin_transfer, None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        else:
            peak = 0
    finally:
        del batch, model, expander, adversary, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return peak


def fit_line(s_values, peak_values) -> tuple[float, float]:
    """Least-squares fit peak = R + C*S. Returns (C_bytes_per_sentence, R_bytes)."""
    s = np.asarray(s_values, dtype=np.float64)
    p = np.asarray(peak_values, dtype=np.float64)
    if len(s) < 2:
        raise ValueError("need >=2 points to fit a line")
    C, R = np.polyfit(s, p, 1)
    C = max(float(C), 1.0)        # slope must be positive
    R = max(float(R), 0.0)        # intercept can't be negative
    return C, R


def calibrate(h5_path, num_latents_list, modes=DEFAULT_MODES, *,
              device="cuda", amp=True, batch_size=128, output_dim=72,
              points=DEFAULT_CALIB_POINTS) -> dict:
    """Measure (C, R) for every (num_latents, backward_mode). See module docstring."""
    import h5py

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    with h5py.File(str(h5_path), "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    calib = {"_meta": {"input_dim": input_dim, "device": str(dev), "amp": bool(amp),
                       "batch_size": int(batch_size), "output_dim": int(output_dim),
                       "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    for num_latents in num_latents_list:
        for mode in modes:
            args = build_calib_args(h5_path, num_latents, mode, batch_size, str(dev), amp, output_dim)
            xs, ys = [], []
            for total_per_view in points:
                S = 2 * total_per_view  # both views forwarded
                try:
                    peak = measure_peak_bytes(args, input_dim, dev, total_per_view)
                except torch.cuda.OutOfMemoryError:
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"calib {num_latents}|{mode}: S={S} OOM (skipped)", flush=True)
                    continue
                xs.append(S)
                ys.append(peak)
                print(f"calib {num_latents}|{mode}: S={S} peak={peak/GIB:.2f}GiB", flush=True)
            if len(xs) < 2:
                print(f"calib {num_latents}|{mode}: too few points; skipping", flush=True)
                continue
            C, R = fit_line(xs, ys)
            calib[_calib_key(num_latents, mode)] = {
                "C": C, "R": R,
                "C_kib_per_sentence": round(C / 1024.0, 3),
                "R_gib": round(R / GIB, 3),
                "points": [{"S": int(s), "peak_gib": round(p / GIB, 3)} for s, p in zip(xs, ys)],
            }
            print(f"calib {num_latents}|{mode}: C={C/1024:.2f}KiB/sent R={R/GIB:.2f}GiB", flush=True)
    return calib


def save_calib(calib: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_calib(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass
class GameStats:
    """Per-game sentence counts + appids, loaded once from the H5 (offsets only)."""
    sentence_counts: np.ndarray
    appids: list[str]
    input_dim: int
    num_games: int = field(init=False)

    def __post_init__(self):
        self.num_games = int(len(self.sentence_counts))

    @classmethod
    def from_h5(cls, h5_path) -> "GameStats":
        import h5py

        with h5py.File(str(h5_path), "r") as h5:
            _reviews, sentence_counts = game_review_sentence_counts(h5)
            input_dim = int(h5.attrs["input_dim"])
            if "appids" in h5:
                appids = [decode_name(x) for x in h5["appids"][:]]
            else:
                appids = [decode_name(x).split("_", 1)[0] for x in h5["game_names"][:]]
        return cls(sentence_counts=np.asarray(sentence_counts, dtype=np.int64),
                   appids=appids, input_dim=input_dim)

    def subset_indices(self, train_game_count: int, seed: int, anchor_appids) -> np.ndarray:
        """Replicates train_vicreg_review_h5.resolve_train_game_indices selection."""
        if train_game_count <= 0 or train_game_count >= self.num_games:
            return np.arange(self.num_games, dtype=np.int64)
        anchors = parse_string_list(anchor_appids) if isinstance(anchor_appids, str) else list(anchor_appids or [])
        anchor_indices = [self.appids.index(a) for a in anchors if a in self.appids]
        rng = np.random.default_rng(seed)
        order = rng.permutation(self.num_games).tolist()
        selected, seen = [], set()
        for index in anchor_indices + order:
            index = int(index)
            if index in seen:
                continue
            selected.append(index)
            seen.add(index)
            if len(selected) >= train_game_count:
                break
        return np.asarray(selected, dtype=np.int64)

    def subset_worst_sentences(self, train_game_count: int, seed: int, anchor_appids) -> int:
        idx = self.subset_indices(train_game_count, seed, anchor_appids)
        return int(self.sentence_counts[idx].max())

    def subset_total_sentences(self, train_game_count: int, seed: int, anchor_appids) -> int:
        idx = self.subset_indices(train_game_count, seed, anchor_appids)
        return int(self.sentence_counts[idx].sum())


def plan_combo(calib: dict, worst_game_sentences: int, free_vram_bytes: float,
               num_latents: int, view: float, batch_size: int, *,
               modes=DEFAULT_MODES, safety: float = 0.85,
               try_paired: bool = True) -> dict:
    """Decide backward_mode / paired / sentence caps for one combo.

    ``worst_game_sentences`` is the raw sentence count of the biggest single
    game in the train subset (a single game cannot be split across batches, so
    it sets the hard floor a batch must accommodate).
    """
    budget = float(free_vram_bytes) * float(safety)
    view = float(view)
    # One game across both views forwards worst_game * view * 2 sentences.
    s_floor = worst_game_sentences * view * 2.0
    # Above this raw batch-sentence count a cap can never bind (a batch holds at
    # most batch_size games, each <= the biggest game), so we report 0 = no cap.
    natural_max = float(batch_size) * float(worst_game_sentences)

    def _batch_cap(usable_S: float) -> int:
        # usable_S is the affordable forwarded-sentence budget (both views).
        raw = usable_S / (view * 2.0)
        return 0 if raw >= natural_max else max(int(raw), 1)

    def evaluate(mode: str):
        entry = calib.get(_calib_key(num_latents, mode))
        if not entry:
            return None
        C, R = float(entry["C"]), float(entry["R"])
        usable_bytes = budget - R
        if usable_bytes <= 0:
            return None
        usable_S = usable_bytes / C                  # affordable forwarded sentences
        fits_floor = s_floor <= usable_S             # biggest single game fits uncapped?
        max_view = 0 if fits_floor else max(int(usable_S / 2.0), 1)
        return {
            "mode": mode, "C": C, "R": R, "usable_S": usable_S,
            "max_batch_sentences": _batch_cap(usable_S),
            "max_view_sentences": max_view, "fits_floor": fits_floor,
            "est_peak_bytes": R + C * s_floor,       # peak of the biggest single game
        }

    ordered = [m for m in ("standard",) if m in modes] + [m for m in modes if m != "standard"]
    chosen = None
    for mode in ordered:
        cand = evaluate(mode)
        if cand is None:
            continue
        chosen = cand
        if cand["fits_floor"]:
            break  # prefer the fastest mode that fits the biggest game uncapped

    if chosen is None:
        return {"num_latents": int(num_latents), "view": view, "paired": False,
                "backward_mode": "split_recompute", "max_batch_sentences": 0,
                "max_view_sentences": 0, "note": "no calibration; fall back to split_recompute"}

    C, R = chosen["C"], chosen["R"]
    paired = False
    if try_paired:
        # Paired holds two arms simultaneously: ~2R fixed + 2C activations.
        paired_usable_S = (budget - 2.0 * R) / (2.0 * C) if (budget - 2.0 * R) > 0 else -1.0
        paired = paired_usable_S >= s_floor
        if paired:
            chosen["max_batch_sentences"] = _batch_cap(paired_usable_S)
            chosen["max_view_sentences"] = 0  # paired fits the biggest game by construction

    note = "ok" if chosen["fits_floor"] else "biggest game capped via max_view_sentences"
    return {
        "num_latents": int(num_latents),
        "view": view,
        "worst_game_sentences": int(worst_game_sentences),
        "backward_mode": chosen["mode"],
        "paired": bool(paired),
        "max_batch_sentences": int(chosen["max_batch_sentences"]),
        "max_view_sentences": int(chosen["max_view_sentences"]),
        "est_peak_gib": round(chosen["est_peak_bytes"] / GIB, 2),
        "budget_gib": round(budget / GIB, 2),
        "note": note,
    }


def plan_combo_chunked(calib: dict, worst_game_sentences: int, free_vram_bytes: float,
                       num_latents: int, view: float, batch_size: int, *,
                       safety: float = 0.85, try_paired: bool = True,
                       mode: str = "standard", total_sentences: int = 0,
                       cache_bytes: float = 0.0, ram_budget: float = 0.0) -> dict:
    """Chunked-stem plan: never drop sentences, pick the backward mode + stem chunk.

    Memory model (both from the same calibrated ``peak = R + C*S``, S = both-view
    forwarded sentences):

    * ``standard`` holds EVERY game's graph until one backward, so its peak scales
      with the WHOLE batch: ``R + C_std * (2*view*total_sentences)``. It grows with
      the game count -- chunking the stem does not help (the retained per-game
      latents still stack).
    * ``split_recompute`` processes one game at a time (stems parked on CPU, backward
      replayed per game), so its peak is set by the WORST single game, independent of
      the game count. The chunked stem bounds that game further to ``~R + C*chunk``.

    So we pick ``standard`` only when the whole batch's forward actually fits, else
    start in ``split_recompute`` (the worst-game-bounded mode) directly -- instead of
    OOMing in standard and reactively downgrading. ``total_sentences<=0`` (unknown)
    keeps the caller's ``mode``.
    """
    budget = float(free_vram_bytes) * float(safety)
    # Host-RAM decision (independent of VRAM): if the full pinned cache would not
    # fit in the RAM budget, stream instead. ram_budget<=0 means "unknown" -> keep
    # full and let the reactive SIGKILL->RAM downgrade handle it.
    cache_mode, pin_cache = "full", True
    if ram_budget and ram_budget > 0 and cache_bytes and cache_bytes > ram_budget:
        cache_mode, pin_cache = "queue", False

    # --- backward-mode selection from the memory model ---
    std = calib.get(_calib_key(num_latents, "standard"))
    chosen_mode = mode
    standard_peak = None
    if total_sentences and total_sentences > 0 and std:
        s_total = 2.0 * float(view) * float(total_sentences)     # both views, whole batch
        standard_peak = float(std["R"]) + float(std["C"]) * s_total
        chosen_mode = "standard" if standard_peak <= budget else "split_recompute"

    entry = calib.get(_calib_key(num_latents, chosen_mode)) or calib.get(_calib_key(num_latents, "split_recompute"))
    if not entry:
        # No calibration for this num_latents: fall back to a small conservative
        # chunk (never 0 -- 0 means "don't chunk" and would risk OOM).
        return {"num_latents": int(num_latents), "view": float(view),
                "worst_game_sentences": int(worst_game_sentences),
                "backward_mode": chosen_mode, "paired": False,
                "stem_chunk_size": NO_CALIB_CHUNK,
                "chunk_full": NO_CALIB_CHUNK >= float(worst_game_sentences) * float(view),
                "cache_mode": cache_mode, "pin_cache": pin_cache,
                "budget_gib": round(budget / GIB, 2),
                "note": "no calibration; conservative chunk"}
    C, R = float(entry["C"]), float(entry["R"])
    # split's slope was fit against the whole CALIB_N_GAMES-game pseudo-batch total,
    # but a split STEP's peak is one game (~1/N of that S), so the true cost per
    # forwarded sentence is ~N x the fitted slope. Correct it so the chunk is sized
    # for a single game, not the whole calib batch.
    c_sent = C * float(CALIB_N_GAMES) if chosen_mode == "split_recompute" else C

    def chunk_for(resident_R: float) -> int:
        # Per-chunk peak ~ resident_R + c_sent * chunk_sentences; /2 keeps a margin
        # for both views held. Floor at 1 so it always fits.
        return max(1, int((budget - resident_R) / (2.0 * c_sent)))

    paired = bool(try_paired) and (budget - 2.0 * R) > 0
    resident_R = 2.0 * R if paired else R
    chunk = chunk_for(resident_R)
    if budget - resident_R <= 0:           # model itself doesn't fit: can't pair
        paired = False
        chunk = chunk_for(R)
    # "full" = one chunk covers the biggest game's forwarded sentences (view*worst,
    # one view) -> the stem runs in a single online-softmax pass. This is the fast,
    # small-VRAM-friendly class the scheduler prioritises.
    fwd_worst = float(worst_game_sentences) * float(view)
    chunk_full = int(chunk) >= fwd_worst
    return {
        "num_latents": int(num_latents),
        "view": float(view),
        "worst_game_sentences": int(worst_game_sentences),
        "backward_mode": chosen_mode,
        "paired": bool(paired),
        "stem_chunk_size": int(chunk),
        "chunk_full": bool(chunk_full),
        "cache_mode": cache_mode,
        "pin_cache": pin_cache,
        "budget_gib": round(budget / GIB, 2),
        "standard_peak_gib": None if standard_peak is None else round(standard_peak / GIB, 2),
        "note": (f"{chosen_mode}: " + ("full (1 pass)" if chunk_full else "chunked")),
    }


def _free_vram_bytes(device: str) -> float:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return float(free)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, type=Path)
    p.add_argument("--calib-out", type=Path, default=None, help="Where to write/read calib.json.")
    p.add_argument("--measure", action="store_true", help="Run pseudo-batch calibration now.")
    p.add_argument("--plan", action="store_true", help="Print the dry-run memory plan table.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--base-num-latents", type=int, default=256)
    p.add_argument("--latent-scales", type=float, nargs="+", default=[1, 2, 4])
    p.add_argument("--output-dims", type=int, nargs="+", default=[18, 36, 64, 72])
    p.add_argument("--sample-fractions", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2])
    p.add_argument("--train-game-counts", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 1500, 2000, 0])
    p.add_argument("--train-game-seed", type=int, default=20260626)
    p.add_argument("--train-game-anchor-appids", default="1091500,1385380")
    p.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    p.add_argument("--safety", type=float, default=0.85)
    p.add_argument("--free-vram-gib", type=float, default=None,
                   help="Override measured free VRAM (e.g. plan on a different card).")
    p.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    return p.parse_args(argv)


def _num_latents_list(base: int, scales) -> list[int]:
    return sorted({max(1, int(round(base * s))) for s in scales})


def _run(args) -> None:
    nl_list = _num_latents_list(args.base_num_latents, args.latent_scales)

    calib = None
    if args.measure:
        calib = calibrate(args.h5, nl_list, tuple(args.modes), device=args.device,
                          amp=args.amp, batch_size=args.batch_size,
                          output_dim=max(args.output_dims))
        if args.calib_out:
            save_calib(calib, args.calib_out)
            print(f"calib written: {args.calib_out}", flush=True)
    elif args.calib_out:
        calib = load_calib(args.calib_out)
        if calib is None:
            raise SystemExit(f"calib not found: {args.calib_out}; pass --measure first.")

    if not args.plan:
        return
    if calib is None:
        raise SystemExit("nothing to plan with; pass --measure and/or --calib-out.")

    free = (args.free_vram_gib * GIB) if args.free_vram_gib is not None else _free_vram_bytes(args.device)
    stats = GameStats.from_h5(args.h5)
    print(f"\nmemory plan | free_vram={free/GIB:.1f}GiB safety={args.safety} "
          f"pool={stats.num_games} games\n", flush=True)
    header = ("num_lat", "view", "games", "worst_g", "mode", "paired",
              "max_batch", "max_view", "est_peak", "note")
    print("  ".join(f"{h:>9}" for h in header), flush=True)
    for nl in nl_list:
        for games in args.train_game_counts:
            worst = stats.subset_worst_sentences(games, args.train_game_seed, args.train_game_anchor_appids)
            for view in args.sample_fractions:
                plan = plan_combo(calib, worst, free, nl, view, args.batch_size,
                                  modes=tuple(args.modes), safety=args.safety)
                row = (nl, f"{view:g}", games or "all", worst, plan["backward_mode"],
                       "Y" if plan["paired"] else "n", plan["max_batch_sentences"],
                       plan["max_view_sentences"], f"{plan.get('est_peak_gib', '?')}", plan["note"])
                print("  ".join(f"{str(c):>9}" for c in row), flush=True)


def main(argv=None) -> None:
    args = _parse_args(argv)
    from tools.logging_tee import run_with_optional_tee
    run_with_optional_tee(args.logout_address, _run, args)


if __name__ == "__main__":
    main()
