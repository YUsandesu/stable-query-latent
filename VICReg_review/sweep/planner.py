"""Per-combo memory plan for the sweep: chunk, never cap.

Wraps VICReg_review.oom_proxy. For each combo it resolves
``{backward_mode, paired, stem_chunk_size}`` from the calibrated (C, R), the
measured free VRAM, and the worst single game in that combo's train subset.

The chunked stem (model.HierarchicalLatentArrayMLP, chunk_size) bounds the
per-game stem peak, so every game trains in FULL -- there is no sentence cap and
no "this combo OOMs" branch; the only levers are chunk size and pairing.

Runnable as a dry-run plan table::

    python -m VICReg_review.sweep.planner --config VICReg_review/sweep/sweep.yaml \
        --calib-json .../calib.json --free-vram-gib 79
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.logging_tee import run_with_optional_tee  # noqa: E402
from VICReg_review import oom_proxy  # noqa: E402
from VICReg_review.sweep.config import SweepConfig  # noqa: E402

GIB = oom_proxy.GIB


def plan_for_combo(config: SweepConfig, calib: dict, stats: "oom_proxy.GameStats",
                   free_vram_bytes: float, combo, *, try_paired: bool = True,
                   ram_budget: float = 0.0) -> dict:
    ds = config.data_seed
    worst = stats.subset_worst_sentences(combo.train_games, ds.train_game_seed, ds.anchors)
    total = stats.subset_total_sentences(combo.train_games, ds.train_game_seed, ds.anchors)
    cache_bytes = oom_proxy.estimate_full_cache_bytes(total, combo.view, stats.input_dim)
    plan = oom_proxy.plan_combo_chunked(
        calib, worst, free_vram_bytes, combo.num_latents, combo.view,
        config.train.batch_size, safety=config.memory.vram_safety, try_paired=try_paired,
        total_sentences=total, cache_bytes=cache_bytes, ram_budget=ram_budget,
    )
    plan["combo_id"] = combo.combo_id
    plan["train_games"] = combo.train_games
    plan["cache_gib"] = round(cache_bytes / GIB, 1)
    return plan


def plan_grid(config: SweepConfig, calib: dict, stats: "oom_proxy.GameStats",
              free_vram_bytes: float) -> list[dict]:
    # paired only matters when both arms are present in the grid.
    arms = {str(a) for a in config.grid.arms}
    try_paired = {"grl", "nogrl"}.issubset(arms)
    ram_budget = oom_proxy.available_ram_bytes() * config.memory.ram_safety
    # One plan per (output_dim, latent_scale, train_games, view) -- arm-independent
    # for memory, so dedupe across arms to keep the table readable.
    seen = set()
    rows = []
    for combo in config.iter_combos():
        key = (combo.output_dim, combo.num_latents, combo.train_games, combo.view)
        if key in seen:
            continue
        seen.add(key)
        rows.append(plan_for_combo(config, calib, stats, free_vram_bytes, combo,
                                   try_paired=try_paired, ram_budget=ram_budget))
    return rows


def format_table(rows: list[dict], free_vram_bytes: float, safety: float, pool: int,
                 ram_budget_gib: float = 0.0) -> str:
    out = [f"memory plan | free_vram={free_vram_bytes / GIB:.1f}GiB safety={safety} "
           f"ram_budget={ram_budget_gib:.0f}GiB pool={pool} games", ""]
    header = ("num_lat", "view", "games", "worst_g", "mode", "paired", "chunk", "std_req", "cache_gib", "cache", "note")
    out.append("  ".join(f"{h:>9}" for h in header))
    for r in rows:
        row = (r["num_latents"], f"{r['view']:g}", r["train_games"] or "all",
               r["worst_game_sentences"], r["backward_mode"],
               "Y" if r["paired"] else "n", r["stem_chunk_size"],
               r.get("standard_required_gib", "?"), r.get("cache_gib", "?"),
               r.get("cache_mode", "full"), r["note"])
        out.append("  ".join(f"{str(c):>9}" for c in row))
    return "\n".join(out)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--h5", default=None, help="Override config.h5 (e.g. a local-disk copy).")
    p.add_argument("--calib-json", type=Path, default=None,
                   help="Calibration cache. Defaults to <out_dir>/calib.json.")
    p.add_argument("--free-vram-gib", type=float, default=None,
                   help="Override measured free VRAM (plan for a different card / no GPU).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    return p.parse_args(argv)


def _run(args) -> None:
    config = SweepConfig.load(args.config)
    if args.h5:
        config.h5 = str(args.h5)
    calib_json = args.calib_json or (Path(config.out_dir) / "calib.json")
    calib = oom_proxy.load_calib(calib_json)
    if calib is None:
        raise SystemExit(f"calib not found: {calib_json}; run oom_proxy --measure first.")
    stats = oom_proxy.GameStats.from_h5(config.h5)
    free = (args.free_vram_gib * GIB) if args.free_vram_gib is not None \
        else oom_proxy._free_vram_bytes(args.device)
    rows = plan_grid(config, calib, stats, free)
    ram_budget = oom_proxy.available_ram_bytes() * config.memory.ram_safety
    print(format_table(rows, free, config.memory.vram_safety, stats.num_games, ram_budget / GIB))


def main(argv=None) -> None:
    args = _parse_args(argv)
    run_with_optional_tee(args.logout_address, _run, args)


if __name__ == "__main__":
    main()
