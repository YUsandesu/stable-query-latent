"""Soft-prune selected combos from a running coordinated sweep.

This does not stop workers or delete checkpoints. It writes failed.json markers
for unfinished combos matching the prune rules, so the existing
Coordinator treats them as terminal and future claims skip them.

Use --dry-run first. Use --apply to write markers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in os.sys.path:
    os.sys.path.insert(0, str(REPO))

from VICReg_review.sweep.config import SweepConfig  # noqa: E402
from VICReg_review.sweep import coordination as coord  # noqa: E402


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def manifest_done(combo_dir: Path) -> bool:
    payload = read_json(combo_dir / "vicreg_review_h5_manifest.json")
    return isinstance(payload, dict) and payload.get("status") == "done"


def atomic_create_json(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        os.write(fd, b"\n")
    finally:
        os.close(fd)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO / "VICReg_review/sweep/sweep.yaml"))
    p.add_argument("--out-dir", default=None, help="Override config.out_dir.")
    p.add_argument("--threshold", type=int, default=128,
                   help="Prune combos with 0 < train_games < threshold. Use 0 to disable.")
    p.add_argument(
        "--drop-latent-view",
        action="append",
        default=[],
        metavar="LATENTS:VIEW",
        help="Prune combos with this num_latents and sample_fraction/view. Example: 1024:0.8",
    )
    p.add_argument("--drop-all", action="store_true", help="Prune train_game_count=all combos.")
    p.add_argument("--apply", action="store_true", help="Actually write failed.json markers.")
    p.add_argument(
        "--include-live",
        action="store_true",
        help="Also mark combos that currently have status.json claims. This does not stop the worker; it only prevents future reclaim.",
    )
    return p.parse_args()


def parse_latent_view_rules(values: list[str]) -> list[tuple[int, float]]:
    rules = []
    for value in values:
        left, sep, right = str(value).partition(":")
        if not sep:
            raise SystemExit(f"--drop-latent-view must be LATENTS:VIEW, got {value!r}")
        rules.append((int(left), float(right)))
    return rules


def prune_reasons(combo, threshold: int, latent_view_rules: list[tuple[int, float]], drop_all: bool) -> list[str]:
    reasons = []
    train_games = int(combo.train_games)
    if drop_all and train_games <= 0:
        reasons.append("train_games=all")
    if threshold > 0 and 0 < train_games < threshold:
        reasons.append(f"train_games={train_games} < threshold={threshold}")
    for latents, view in latent_view_rules:
        if int(combo.num_latents) == int(latents) and abs(float(combo.view) - float(view)) < 1e-9:
            reasons.append(f"num_latents={latents} view={view:g}")
    return reasons


def main() -> None:
    args = parse_args()
    cfg = SweepConfig.load(args.config)
    if args.out_dir:
        cfg.out_dir = args.out_dir
    out_dir = Path(cfg.out_dir)
    threshold = int(args.threshold)
    latent_view_rules = parse_latent_view_rules(args.drop_latent_view)
    if threshold <= 0 and not latent_view_rules and not args.drop_all:
        raise SystemExit("No prune rules active. Pass --threshold N and/or --drop-latent-view LATENTS:VIEW.")

    counts = {
        "eligible": 0,
        "already_done": 0,
        "already_failed": 0,
        "live_skipped": 0,
        "would_prune": 0,
        "pruned": 0,
    }
    samples: dict[str, list[str]] = {key: [] for key in counts}

    def sample(key: str, combo_id: str, limit: int = 12) -> None:
        if len(samples[key]) < limit:
            samples[key].append(combo_id)

    now = time.time()
    for combo in cfg.iter_combos():
        reasons = prune_reasons(combo, threshold, latent_view_rules, args.drop_all)
        if not reasons:
            continue
        counts["eligible"] += 1
        sample("eligible", combo.combo_id)

        combo_dir = out_dir / combo.combo_id
        if manifest_done(combo_dir) or (combo_dir / coord.DONE).exists():
            counts["already_done"] += 1
            sample("already_done", combo.combo_id)
            continue
        if (combo_dir / coord.FAILED).exists():
            counts["already_failed"] += 1
            sample("already_failed", combo.combo_id)
            continue
        if (combo_dir / coord.STATUS).exists() and not args.include_live:
            counts["live_skipped"] += 1
            sample("live_skipped", combo.combo_id)
            continue

        counts["would_prune"] += 1
        sample("would_prune", combo.combo_id)
        if args.apply:
            payload = {
                "vm": "manual_prune",
                "ts": now,
                "reason": "manual_prune",
                "error": f"pruned: {'; '.join(reasons)}",
                "combo_id": combo.combo_id,
                "train_games": int(combo.train_games),
                "num_latents": int(combo.num_latents),
                "view": float(combo.view),
                "threshold": threshold,
                "drop_all": bool(args.drop_all),
                "drop_latent_view": [{"num_latents": int(n), "view": float(v)} for n, v in latent_view_rules],
            }
            if atomic_create_json(combo_dir / coord.FAILED, payload):
                counts["pruned"] += 1
                sample("pruned", combo.combo_id)
            else:
                counts["already_failed"] += 1
                sample("already_failed", combo.combo_id)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: prune selected combos")
    print(f"out_dir: {out_dir}")
    print(f"threshold: {threshold} ({'enabled' if threshold > 0 else 'disabled'})")
    print(f"drop_all: {bool(args.drop_all)}")
    print(f"drop_latent_view: {latent_view_rules or '[]'}")
    for key, value in counts.items():
        print(f"{key:14}: {value}")
        if samples[key]:
            print(f"  sample: {samples[key]}")
    if not args.apply:
        print("\nNo files changed. Re-run with --apply to write failed.json prune markers.")
    elif counts["live_skipped"]:
        print("\nNote: live claimed combos were left alone. Re-run later, or use --include-live if you only want to prevent future reclaim.")


if __name__ == "__main__":
    main()
