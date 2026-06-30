"""Supervisor: owns the ledger, drives the worker, recovers from crashes.

The supervisor never touches CUDA, so it cannot OOM. It:

* reconciles the ledger on startup (dead-PID 'running' -> 'interrupted');
* spawns the worker and waits for it to calibrate (worker.ready);
* for each combo, plans memory settings, emits a job, marks it 'running' with
  the worker PID, and waits for a result OR worker death;
* on success marks 'done'; on OOM/error downgrades the settings and retries; on
  worker death it polls until VRAM is reclaimed, respawns, and retries;
* isolates failures per combo (log + continue) so one combo never aborts the
  sweep; a combo that exhausts its attempts is marked 'failed' and skipped.

``spawn_worker``, ``free_vram_fn``, ``calib`` and ``stats`` are injectable so the
recovery loop can be tested without a GPU.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.logging_tee import run_with_optional_tee  # noqa: E402
from VICReg_review import oom_proxy  # noqa: E402
from VICReg_review.sweep import jobspec, protocol  # noqa: E402
from VICReg_review.sweep.config import SweepConfig  # noqa: E402
from VICReg_review.sweep.ledger import Ledger, pid_alive  # noqa: E402

MAX_ATTEMPTS = 4
MIN_CHUNK = 64


def _free_vram_bytes_smi() -> float:
    """Free VRAM via nvidia-smi so the supervisor never initialises a CUDA
    context of its own (it must stay un-OOM-able). Falls back to torch only if
    nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            return float(out.splitlines()[0].strip()) * 1024 * 1024
    except Exception:
        pass
    return oom_proxy._free_vram_bytes("cuda")


class Supervisor:
    def __init__(self, config: SweepConfig, config_path=None, *, spawn_worker=None,
                 free_vram_fn=None, calib=None, stats=None, poll=2.0,
                 ready_timeout=3600.0, reclaim_timeout=600.0, logout_address=None,
                 h5_override=None, retry_failed=False):
        self.config = config
        self.config_path = config_path
        self.logout_address = logout_address
        self.h5_override = h5_override
        self.retry_failed = retry_failed
        self.out_dir = config.out_dir
        self.ledger = Ledger(Path(config.out_dir) / "ledger.jsonl")
        self.probe_queue = Path(config.out_dir) / "probe_queue"
        self._spawn = spawn_worker or self._default_spawn
        self._free_vram = free_vram_fn or _free_vram_bytes_smi
        self.poll = poll
        self.ready_timeout = ready_timeout
        self.reclaim_timeout = reclaim_timeout
        self.worker = None
        self.calib = calib
        self.stats = stats
        self._tally = {"done": 0, "failed": 0, "skipped": 0}

    # --- worker process management ----------------------------------------
    def _default_spawn(self):
        argv = [sys.executable, "-u", str(SCRIPT_DIR / "worker.py"),
                "--config", str(self.config_path), "--device", "cuda"]
        if self.h5_override:
            argv += ["--h5", str(self.h5_override)]
        if self.logout_address:
            argv += ["--logout-address", str(self.logout_address)]
        # start_new_session so the worker + its data-pool children form one
        # process group we can kill cleanly. expandable_segments reduces the
        # allocator fragmentation the OOM messages keep flagging.
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        return subprocess.Popen(argv, cwd=str(ROOT), start_new_session=True, env=env)

    def _kill_worker(self) -> None:
        """Make sure the previous worker (and its data-pool children) are dead
        before respawning, so a worker that OOM'd but is slow to release its CUDA
        context does not leave the GPU full for the next one."""
        w = self.worker
        if w is None or w.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(w.pid), signal.SIGKILL)   # whole process group
        except Exception:
            try:
                w.kill()
            except Exception:
                pass
        try:
            w.wait(timeout=30)
        except Exception:
            pass

    def _worker_alive(self) -> bool:
        return bool(self.worker) and self.worker.poll() is None

    def _worker_pid(self):
        return getattr(self.worker, "pid", None)

    def _wait_ready(self) -> None:
        self._kill_worker()           # never leave an old worker holding the GPU
        protocol.clear_signals(self.out_dir)
        self.worker = self._spawn()
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if protocol.ready_path(self.out_dir).exists():
                return
            if not self._worker_alive():
                raise RuntimeError("worker exited before signalling ready")
            time.sleep(self.poll)
        raise TimeoutError("worker did not become ready in time")

    def _poll_until_free(self) -> None:
        deadline = time.time() + self.reclaim_timeout
        first = self._free_vram()
        while time.time() < deadline:
            time.sleep(self.poll)
            if self._free_vram() >= first:        # memory came back after the crash
                return
        # best effort: proceed even if we couldn't confirm reclamation

    def _recover_worker(self) -> None:
        self._kill_worker()           # kill the stuck/old worker first ...
        self._poll_until_free()       # ... then wait for its GPU memory to return
        self._wait_ready()            # ... then spawn a fresh one

    # --- planning / downgrade ---------------------------------------------
    def _ensure_inputs(self) -> None:
        if self.calib is None:
            self.calib = oom_proxy.load_calib(Path(self.out_dir) / "calib.json") or {}
        if self.stats is None:
            self.stats = oom_proxy.GameStats.from_h5(self.config.h5)

    def _ram_budget(self) -> float:
        return oom_proxy.available_ram_bytes() * float(getattr(self.config.memory, "ram_safety", 0.8))

    def plan_settings(self, combo) -> dict:
        ds = self.config.data_seed
        worst = self.stats.subset_worst_sentences(combo.train_games, ds.train_game_seed, ds.anchors)
        total = self.stats.subset_total_sentences(combo.train_games, ds.train_game_seed, ds.anchors)
        cache_bytes = oom_proxy.estimate_full_cache_bytes(total, combo.view, self.stats.input_dim)
        plan = oom_proxy.plan_combo_chunked(
            self.calib, worst, self._free_vram(), combo.num_latents, combo.view,
            self.config.train.batch_size, safety=self.config.memory.vram_safety, try_paired=False,
            cache_bytes=cache_bytes, ram_budget=self._ram_budget())
        return {"backward_mode": plan["backward_mode"],
                "stem_chunk_size": int(plan["stem_chunk_size"]),
                "paired": False,
                "cache_mode": plan["cache_mode"],
                "pin_cache": plan["pin_cache"]}

    @staticmethod
    def downgrade(settings: dict) -> dict:
        """VRAM downgrade after a CUDA OOM. No cap -- shrink the stem chunk, then
        fall back to split_recompute, then shrink further."""
        chunk = int(settings.get("stem_chunk_size", 0) or 0)
        mode = settings.get("backward_mode", "standard")
        if chunk > MIN_CHUNK * 4:
            return {**settings, "stem_chunk_size": max(MIN_CHUNK, chunk // 2)}
        if mode != "split_recompute":
            return {**settings, "backward_mode": "split_recompute"}
        return {**settings, "stem_chunk_size": max(MIN_CHUNK // 2, chunk // 2)}

    @staticmethod
    def downgrade_ram(settings: dict) -> dict:
        """Host-RAM downgrade after a SIGKILL (OOM-killer): stop materialising +
        pinning the full cache (stream via the bounded queue, shrink prefetch).
        If already streaming + unpinned, fall back to the VRAM chunk downgrade."""
        if settings.get("cache_mode") != "queue" or settings.get("pin_cache", True):
            return {**settings, "cache_mode": "queue", "pin_cache": False, "prefetch_batches": 1}
        return Supervisor.downgrade(settings)

    # --- main loop ---------------------------------------------------------
    def _tally_str(self) -> str:
        t = self._tally
        return f"done={t['done']} failed={t['failed']} skipped={t['skipped']}"

    def run(self) -> dict:
        self.ledger.reconcile_running()
        if self.retry_failed:
            n = 0
            for cid, rec in self.ledger.load().items():
                if rec.get("status") == "failed":
                    self.ledger.update(cid, status="pending", attempts=0, error=None)
                    n += 1
            if n:
                print(f"supervisor: reset {n} failed combos for retry", flush=True)
        self._wait_ready()
        self._ensure_inputs()
        total = self.config.combo_count()
        self._tally = {"done": 0, "failed": 0, "skipped": 0}
        print(f"supervisor: {total} combos to process", flush=True)
        for position, combo in enumerate(self.config.iter_combos(), 1):
            self._run_combo(combo, position, total)
        protocol.write_stop(self.out_dir)
        summary = self.ledger.summary()
        print(f"supervisor: sweep complete [{total}/{total}] -> {summary}", flush=True)
        return summary

    def _run_combo(self, combo, position=0, total=0) -> None:
        combo_id = combo.combo_id
        config_hash = self.config.config_hash(combo)
        paths = jobspec.combo_paths(self.config, combo)
        prog = f"[{position}/{total}]"

        if self.ledger.is_done(combo_id, config_hash) and paths["checkpoint"].exists():
            self._tally["skipped"] += 1
            print(f"supervisor: {prog} {combo_id} skip (already done) | {self._tally_str()}", flush=True)
            return  # verified done -- skip

        settings = self.plan_settings(combo)
        while True:
            attempts = int((self.ledger.get(combo_id) or {}).get("attempts", 0))
            if attempts >= MAX_ATTEMPTS:
                self.ledger.mark_failed(combo_id, f"exhausted {attempts} attempts")
                self._tally["failed"] += 1
                print(f"supervisor: {prog} {combo_id} FAILED after {attempts} attempts | {self._tally_str()}", flush=True)
                return

            # Single recovery point: if the worker died (mid-combo crash, or it
            # exited itself after an OOM for a clean CUDA context), bring up a
            # fresh one before emitting the next attempt.
            if not self._worker_alive():
                print(f"supervisor: {prog} worker not alive before {combo_id}; recovering", flush=True)
                self._recover_worker()

            protocol.clear_combo(self.out_dir, combo_id)
            argv = jobspec.build_trainer_argv(self.config, combo, settings, probe_queue_dir=self.probe_queue)
            protocol.write_job(self.out_dir, {
                "combo_id": combo_id, "config_hash": config_hash,
                "argv": argv, "settings": settings, "ckpt": str(paths["checkpoint"]),
            })
            self.ledger.mark_running(combo_id, config_hash, self._worker_pid(), settings)
            result = self._await_result(combo_id)

            if result is None:  # worker died mid-combo without writing a result
                exit_code = self.worker.poll() if self.worker else None
                self.ledger.mark_interrupted(combo_id, f"worker died (exit={exit_code}) during combo")
                # Clear the stale job so the fresh worker (spawned at the top of
                # the next iteration) cannot pick up the job that killed it.
                protocol.clear_combo(self.out_dir, combo_id)
                if exit_code == -9:   # SIGKILL -> almost certainly the host OOM-killer
                    print(f"supervisor: {prog} {combo_id} worker SIGKILLed (host RAM OOM); RAM-downgrade", flush=True)
                    settings = self.downgrade_ram(settings)
                else:
                    print(f"supervisor: {prog} worker died on {combo_id} (exit={exit_code}); VRAM-downgrade", flush=True)
                    settings = self.downgrade(settings)
                continue
            if result.get("status") == "done":
                self.ledger.mark_done(combo_id, result.get("peak_mem_gib"), result.get("ckpt"))
                protocol.clear_combo(self.out_dir, combo_id)
                self._tally["done"] += 1
                print(f"supervisor: {prog} {combo_id} done (peak={result.get('peak_mem_gib')}GiB) | {self._tally_str()}", flush=True)
                return
            # oom / error -> downgrade and retry. The worker exits after an OOM,
            # so the top-of-loop liveness check recovers it on the next pass.
            print(f"supervisor: {prog} {combo_id} {result.get('status')}: {result.get('error')}; downgrading", flush=True)
            settings = self.downgrade(settings)

    def _await_result(self, combo_id: str):
        while True:
            result = protocol.read_result(self.out_dir, combo_id)
            if result is not None:
                return result
            if not self._worker_alive():
                # give a just-written result one last chance before declaring death
                return protocol.read_result(self.out_dir, combo_id)
            time.sleep(self.poll)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--h5", default=None,
                   help="Override config.h5 (e.g. a fast local-disk copy of the embedding H5). "
                        "Only the H5 input is redirected; checkpoints/ledger stay at out_dir.")
    p.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    p.add_argument("--retry-failed", action="store_true",
                   help="Reset 'failed' combos (attempts exhausted) back to pending so they "
                        "are retried -- e.g. after a code fix that should make them fit.")
    return p.parse_args(argv)


def run_main(args) -> None:
    config = SweepConfig.load(args.config)
    if args.h5:
        config.h5 = str(args.h5)
    summary = Supervisor(config, config_path=args.config, logout_address=args.logout_address,
                         h5_override=args.h5, retry_failed=args.retry_failed).run()
    print(f"sweep done: {summary}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_with_optional_tee(args.logout_address, run_main, args)


if __name__ == "__main__":
    main()
