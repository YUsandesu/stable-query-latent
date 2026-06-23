"""Parallel multi-seed sweep: v1 vs v2 (and a param-matched v2) across N seeds.

Same experiment as seed_sweep.py, but runs several (config, seed) jobs concurrently
on the single GPU. Each tiny model only uses ~2 GB and ~44% SM, so packing a few
processes onto one card pushes utilization up and cuts wall-clock roughly N-fold.

Each job runs in its own process (CUDA needs a fresh context per process); a
ProcessPoolExecutor with --workers caps how many share the GPU at once.

Writes heads/seed_sweep_parallel.json with every run plus aggregates.
"""

import argparse
import contextlib
import io
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
META_SEED = 20240622
NUM_SEEDS = 10
EPOCHS = 800
CONFIGS = [("v1", 64), ("v2", 64), ("v2", 56), ("base", 64)]
H5 = str(SCRIPT_DIR / "pesudo_data" / "benchmark_sentence_latent_query_multi.h5")


def _worker(job):
    """Top-level so it is picklable for the process pool. Trains one config/seed."""
    model_name, hidden_dim, seed, epochs = job
    # Each process imports the trainer fresh (spawn) and builds its own CUDA context.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_latent_query_model as T

    with contextlib.redirect_stdout(io.StringIO()):
        metrics = T.train_and_test(
            h5_path=H5, epochs=epochs, batch_size=128, learning_rate=3e-4,
            min_learning_rate=1e-5, test_ratio=0.2, seed=seed,
            hidden_dim=hidden_dim, flat_dim=128, query_sizes=(32, 16, 8),
            num_heads=8, dropout=0.0, device_name=None,
            model_out=None, history_txt=None, per_dim_txt=None,
            preload_data=True, split_by="score_combo", model_name=model_name,
        )
    return (model_name, hidden_dim, seed,
            metrics["test_accuracy"], metrics["test_mae"], metrics["test_ce"])


def key_of(model_name, hidden_dim):
    return f"{model_name}_h{hidden_dim}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent training processes on the GPU.")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to these config keys (e.g. base_h64). Default: all CONFIGS.")
    parser.add_argument("--out", default="seed_sweep_parallel.json",
                        help="Output filename under heads/.")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in np.random.default_rng(META_SEED).integers(1, 1_000_000, args.num_seeds)]
    configs = [(m, h) for m, h in CONFIGS if args.models is None or key_of(m, h) in args.models]
    jobs = [(m, h, seed, args.epochs) for seed in seeds for (m, h) in configs]
    print(f"seeds={seeds}\njobs={len(jobs)} workers={args.workers} epochs={args.epochs}", flush=True)

    results = {key_of(m, h): [] for m, h in configs}
    started = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for future in as_completed(futures):
            model_name, hidden_dim, seed, acc, mae, ce = future.result()
            results[key_of(model_name, hidden_dim)].append(
                {"seed": seed, "acc": acc, "mae": mae, "ce": ce}
            )
            done += 1
            print(f"[{time.time() - started:6.0f}s] ({done}/{len(jobs)}) "
                  f"{key_of(model_name, hidden_dim):7s} seed={seed:6d}  "
                  f"acc={acc:.4f} mae={mae:.4f} ce={ce:.4f}", flush=True)

    summary = {}
    for key, runs in results.items():
        summary[key] = {
            metric: {
                "mean": st.mean([r[metric] for r in runs]),
                "std": st.stdev([r[metric] for r in runs]),
            }
            for metric in ("acc", "mae", "ce")
        }

    out = {"seeds": seeds, "epochs": args.epochs, "results": results, "summary": summary,
           "elapsed_seconds": round(time.time() - started, 1)}
    (SCRIPT_DIR / "heads" / args.out).write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n===== SUMMARY (mean +/- std, {args.num_seeds} seeds) =====")
    for key in results:
        s = summary[key]
        print(f"{key:7s}  acc={s['acc']['mean']:.4f}+/-{s['acc']['std']:.4f}  "
              f"mae={s['mae']['mean']:.4f}+/-{s['mae']['std']:.4f}  "
              f"ce={s['ce']['mean']:.4f}+/-{s['ce']['std']:.4f}")
    print(f"elapsed {out['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
