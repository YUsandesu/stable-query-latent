"""Persistent training worker.

Loads torch / CUDA context once, optionally calibrates, then drains the job
queue: each job calls train_vicreg_review_h5.train() in-process for one combo.
The embedding cache is H5/OS-page backed (no resident RAM cache to hoist), so
the win is process reuse + crash isolation, not an in-RAM cache.

OOM handling: a per-combo torch.cuda.OutOfMemoryError is caught, reported as a
'oom' result, and the worker frees + continues. If freeing fails (corrupt CUDA
context) the worker exits non-zero so the supervisor restarts it cleanly.

Note: train() blocks for the whole combo, so the heartbeat is refreshed before
each combo, not mid-training -- the supervisor uses PID liveness (not heartbeat
staleness) as the death signal so a long, legitimate combo is never killed.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tools.logging_tee import run_with_optional_tee  # noqa: E402
from VICReg_review import oom_proxy  # noqa: E402
from VICReg_review import train_vicreg_review_h5 as tvh  # noqa: E402
from VICReg_review.sweep import protocol  # noqa: E402
from VICReg_review.sweep.config import SweepConfig  # noqa: E402


def _peak_gib():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / oom_proxy.GIB, 2)
    return None


def run_job(job: dict, device: str) -> dict:
    args = tvh.parse_args(list(job["argv"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tvh.train(args)
    return {"status": "done", "peak_mem_gib": _peak_gib(), "ckpt": job.get("ckpt")}


def maybe_calibrate(config: SweepConfig, device: str) -> None:
    calib_path = Path(config.out_dir) / "calib.json"
    mode = config.memory.calib
    if mode == "off":
        return
    if mode == "load" and calib_path.exists():
        return
    nl_list = oom_proxy._num_latents_list(config.grid.base_num_latents, config.grid.latent_scales)
    print(f"worker: calibrating (C,R) for num_latents={nl_list}", flush=True)
    calib = oom_proxy.calibrate(
        config.h5, nl_list, ("standard", "split_recompute"),
        device=device, amp=True, batch_size=config.train.batch_size,
        output_dim=max(config.grid.output_dims),
    )
    oom_proxy.save_calib(calib, calib_path)
    print(f"worker: wrote {calib_path}", flush=True)


def main_loop(args) -> int:
    config = SweepConfig.load(args.config)
    if getattr(args, "h5", None):
        config.h5 = str(args.h5)
    if getattr(args, "out_dir", None):
        config.out_dir = str(args.out_dir)   # keep calib.json + checkpoints on the same shard dir
    qdir = args.queue_dir or protocol.default_qdir(config.out_dir)
    pid = os.getpid()
    poll = max(0.5, float(args.poll_interval))

    # Calibration is produced once and shared via calib.json. In multi-GPU runs
    # the supervisor pre-calibrates (--calib-only worker) and starts lane workers
    # with --no-calib so they don't all race to rewrite it.
    if not args.no_calib:
        try:
            maybe_calibrate(config, args.device)
        except BaseException as exc:
            print(f"worker: calibration failed ({type(exc).__name__}: {exc})", flush=True)
    if args.calib_only:
        print("worker: calib-only done; exiting", flush=True)
        return 0
    protocol.write_ready(qdir, pid)
    print(f"worker: ready pid={pid} qdir={qdir}", flush=True)

    while True:
        if protocol.stop_path(qdir).exists():
            print("worker: STOP seen; exiting", flush=True)
            return 0
        jobs = protocol.pending_jobs(qdir)
        if not jobs:
            protocol.write_heartbeat(qdir, pid, None)
            time.sleep(poll)
            continue
        job_file = jobs[0]
        job = protocol.read_json(job_file)
        if job is None:
            time.sleep(poll)
            continue
        combo_id = job["combo_id"]
        protocol.write_heartbeat(qdir, pid, combo_id)
        print(f"worker: training {combo_id}", flush=True)
        try:
            result = run_job(job, args.device)
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = {"combo_id": combo_id, "status": "oom", "peak_mem_gib": _peak_gib(), "error": f"{exc}"}
            protocol.write_result(qdir, result)
            protocol.mark_job_consumed(job_file)
            # Exit so the supervisor respawns a fresh worker with a clean CUDA
            # context (an OOM can leave the allocator fragmented/wedged). The
            # supervisor downgrades the chunk before the retry.
            print(f"worker: {combo_id} -> OOM; exiting for a clean restart", flush=True)
            return 2
        except BaseException as exc:
            result = {"combo_id": combo_id, "status": "error",
                      "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            protocol.write_result(qdir, result)
            protocol.mark_job_consumed(job_file)
            print(f"worker: {combo_id} -> error: {exc}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        result["combo_id"] = combo_id
        protocol.write_result(qdir, result)
        protocol.mark_job_consumed(job_file)
        print(f"worker: {combo_id} -> done", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", default=None, help="Override config.out_dir (rarely needed).")
    p.add_argument("--h5", default=None, help="Override config.h5 (e.g. a local-disk copy).")
    p.add_argument("--queue-dir", default=None, help="Job queue dir to poll (default <out_dir>/sweep_jobs).")
    p.add_argument("--calib-only", action="store_true", help="Only calibrate + write calib.json, then exit.")
    p.add_argument("--no-calib", action="store_true", help="Skip calibration (supervisor pre-calibrated).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_with_optional_tee(args.logout_address, main_loop, args)


if __name__ == "__main__":
    raise SystemExit(main())
