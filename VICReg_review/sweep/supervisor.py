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
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.logging_tee import run_with_optional_tee  # noqa: E402
from VICReg_review import oom_proxy  # noqa: E402
from VICReg_review.sweep import jobspec, protocol  # noqa: E402
from VICReg_review.sweep.coordination import Coordinator, LeaseLiveness  # noqa: E402
from VICReg_review.sweep.config import SweepConfig  # noqa: E402
from VICReg_review.sweep.ledger import Ledger, pid_alive  # noqa: E402

MAX_ATTEMPTS = 4
MIN_CHUNK = 64
_SIGKILL = getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", 15))  # Windows has no SIGKILL


def _default_vm_name() -> str:
    try:
        return socket.gethostname() or "vm"
    except Exception:
        return "vm"


def _iter_proc_pids():
    """All live PIDs from /proc (Linux). Empty on platforms without /proc so the
    stale-worker reap is a harmless no-op off the training host."""
    try:
        return [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []


def _free_vram_bytes_smi(gpu: int = 0) -> float:
    """Free VRAM of the given physical GPU via nvidia-smi so the supervisor never
    initialises a CUDA context of its own (it must stay un-OOM-able). Querying by
    --id keeps this correct on multi-GPU hosts (no hardcoded GPU 0). Falls back to
    torch only if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             f"--id={int(gpu)}"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            return float(out.splitlines()[0].strip()) * 1024 * 1024
    except Exception:
        pass
    return oom_proxy._free_vram_bytes("cuda")


class Supervisor:
    """One GPU lane: owns a worker pinned to ``gpu`` polling ``qdir``, and drives
    combos claimed from a (possibly shared) grid. A single-GPU sweep is one lane
    via ``run()``; a multi-GPU sweep is N lanes sharing one ledger/grid/tally,
    driven by ``run_sweep`` (see below)."""

    def __init__(self, config: SweepConfig, config_path=None, *, spawn_worker=None,
                 free_vram_fn=None, calib=None, stats=None, poll=2.0,
                 ready_timeout=3600.0, reclaim_timeout=600.0, logout_address=None,
                 h5_override=None, retry_failed=False, gpu=0, qdir=None,
                 no_calib=False, ram_divisor=1, shard=(0, 1), work_dir=None,
                 vm_name=None, liveness=None):
        self.config = config
        self.config_path = config_path
        self.logout_address = logout_address
        self.h5_override = h5_override
        self.retry_failed = retry_failed
        self.gpu = int(gpu)
        # Disjoint slice of the sweep for this machine: (index, count). (0, 1) =
        # whole sweep. Two VMs use shard (0, 2) and (1, 2) with separate out_dirs.
        self._shard = (int(shard[0]), max(1, int(shard[1])))
        self.no_calib = bool(no_calib)
        self.out_dir = config.out_dir
        # out_dir  -> DURABLE, may be on the shared FS: ledger, checkpoints, probes.
        # work_dir -> MACHINE-LOCAL scratch: calib.json + the job queue (transient
        # same-machine IPC + a GPU-specific memory calibration). Defaults to out_dir
        # so single-machine behaviour is unchanged.
        self.work_dir = Path(work_dir) if work_dir else Path(self.out_dir)
        self.qdir = Path(qdir) if qdir is not None else protocol.default_qdir(self.work_dir)
        # ledger = MACHINE-LOCAL forensics/attempts (the cross-VM done-truth is the
        # coordinator's per-combo status/done markers). Keep it in work_dir so many
        # VMs sharing one out_dir never append to one ledger on the network FS.
        self.ledger = Ledger(Path(self.work_dir) / "ledger.jsonl")
        self.probe_queue = Path(config.out_dir) / "probe_queue"   # dir of per-probe files -> shared-safe
        self._spawn_fn = spawn_worker or self._default_spawn
        self._free_vram_fn = free_vram_fn or _free_vram_bytes_smi
        self.poll = poll
        self.ready_timeout = ready_timeout
        self.reclaim_timeout = reclaim_timeout
        self.worker = None
        self.calib = calib
        self.stats = stats
        # Host RAM is shared across lanes (VRAM is not -- each lane reads its own
        # card). Divide the RAM budget so N lanes don't each plan a full pinned
        # cache and blow the OOM-killer. Data-loader procs are likewise auto-scaled
        # to cores / lane-count (not a YAML knob).
        self._ram_divisor = max(1, int(ram_divisor))
        self._data_workers = jobspec.auto_data_workers(self._ram_divisor)
        # Coordination state -- per-instance by default; run_sweep replaces these
        # refs on every lane so they share one grid/tally/ledger.
        self.total = 0
        self._grid_lock = threading.Lock()
        self._tally = {"done": 0, "failed": 0, "skipped": 0}
        self._tally_lock = threading.Lock()
        # Cross-VM coordination (file-based, shared out_dir). One Coordinator per VM,
        # shared by all lanes. Claims are atomic (O_EXCL status.json) so lanes AND VMs
        # never train the same combo; done = checkpoint exists (migration) OR done.json.
        self._vm_name_req = vm_name
        self._liveness = liveness
        self.coordinator = None
        self.vm_name = None
        self._ordered_ids = []
        self._combo_by_id = {}
        self._fits_fn = None
        # Capability routing (heterogeneous VMs: L4 / A100 / B200): each combo's
        # portable standard-mode peak vs this card's VRAM and everyone else's.
        self._std_peak_gib = {}
        self._my_vram_gib = None
        self._vram_cache = None
        self._vram_cache_ts = 0.0

    def _share_from(self, primary: "Supervisor") -> None:
        """Adopt the primary lane's shared coordination state (multi-GPU)."""
        self.ledger = primary.ledger
        self.stats = primary.stats
        self.calib = primary.calib
        self.total = primary.total
        self._tally = primary._tally
        self._tally_lock = primary._tally_lock
        self.coordinator = primary.coordinator
        self.vm_name = primary.vm_name
        self._ordered_ids = primary._ordered_ids
        self._combo_by_id = primary._combo_by_id
        self._fits_fn = primary._fits_fn

    def _free_vram(self) -> float:
        return self._free_vram_fn(self.gpu)

    def _spawn(self):
        return self._spawn_fn(self.gpu, self.qdir)

    # --- worker process management ----------------------------------------
    def _default_spawn(self, gpu, qdir):
        argv = [sys.executable, "-u", str(SCRIPT_DIR / "worker.py"),
                "--config", str(self.config_path), "--device", "cuda",
                "--queue-dir", str(qdir), "--out-dir", str(self.out_dir),
                "--work-dir", str(self.work_dir)]
        if self.no_calib:
            argv += ["--no-calib"]           # calib was pre-computed once by _ensure_calib
        if self.h5_override:
            argv += ["--h5", str(self.h5_override)]
        if self.logout_address:
            argv += ["--logout-address", str(self.logout_address)]
        # start_new_session so the worker + its data-pool children form one
        # process group we can kill cleanly. expandable_segments reduces the
        # allocator fragmentation the OOM messages keep flagging. CUDA_VISIBLE_DEVICES
        # pins the worker to this GPU (its cuda:0 == physical gpu), so this lane's
        # per-GPU free-VRAM read stays consistent on multi-GPU hosts.
        env = {**os.environ,
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "CUDA_VISIBLE_DEVICES": str(gpu)}
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
        protocol.clear_signals(self.qdir)
        self.worker = self._spawn()
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if protocol.ready_path(self.qdir).exists():
                return
            if not self._worker_alive():
                raise RuntimeError("worker exited before signalling ready")
            time.sleep(self.poll)
        raise TimeoutError("worker did not become ready in time")

    def _total_vram(self) -> float:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                 f"--id={int(self.gpu)}"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if out:
                return float(out.splitlines()[0].strip()) * 1024 * 1024
        except Exception:
            pass
        return 0.0

    def _poll_until_free(self) -> None:
        """Wait until the just-killed worker's VRAM is actually reclaimed before
        respawning. The old ``free >= first`` test returned on the FIRST poll
        (free never dips below itself), so the fresh worker could allocate on top
        of a not-yet-released context and OOM immediately. Now we wait until this
        lane's GPU is mostly empty again (it owns the card), or free plateaus."""
        total = self._total_vram()
        target = 0.85 * total if total else 0.0
        deadline = time.time() + self.reclaim_timeout
        best, stall = -1.0, 0
        while time.time() < deadline:
            time.sleep(self.poll)
            free = self._free_vram()
            if target and free >= target:
                return                                 # GPU reclaimed -> safe to respawn
            margin = 0.02 * (total or free or 1.0)
            if free > best + margin:
                best, stall = free, 0                  # still climbing (memory coming back)
            else:
                stall += 1
                if stall >= 5:                         # plateaued -> as free as it will get
                    return
        # best effort: proceed even if we couldn't confirm reclamation

    def _recover_worker(self) -> None:
        self._kill_worker()           # kill the stuck/old worker first ...
        self._poll_until_free()       # ... then wait for its GPU memory to return
        self._wait_ready()            # ... then spawn a fresh one

    @staticmethod
    def _proc_cmdline(pid):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            return None

    def _protected_pids(self) -> set:
        """Ourselves + our whole ancestor chain -- the notebook kernel may hold a
        CUDA context and appear on the GPU; killing it would abort the run."""
        protected, pid = set(), os.getpid()
        for _ in range(64):
            if pid <= 1 or pid in protected:
                break
            protected.add(pid)
            try:
                with open(f"/proc/{pid}/stat") as f:
                    pid = int(f.read().rsplit(")", 1)[1].split()[1])   # ppid
            except (OSError, IndexError, ValueError):
                break
        return protected

    def _gpu_pids(self, gpus) -> set:
        """PIDs currently holding compute contexts on our target GPUs (nvidia-smi
        reads this without the supervisor ever creating a CUDA context)."""
        pids = set()
        for g in gpus:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
                     f"--id={int(g)}"], capture_output=True, text=True, timeout=5).stdout
                pids.update(int(x) for x in out.split() if x.strip().isdigit())
            except Exception:
                pass
        return pids

    def _reap_stale_workers(self, gpus) -> None:
        """Automatically clear the target GPUs of stale workers before starting.

        Workers spawn in a new session (start_new_session) so the supervisor can
        kill a worker + its data-pool children as one group -- but that also
        DETACHES them from the supervisor, so a supervisor/notebook-kernel restart
        leaves them running and holding ~30 GiB of VRAM each. The next run then
        OOMs on top of these ghosts. So on startup (before spawning anything) we
        find every process pinned to OUR GPUs, and SIGKILL the ones that are our
        worker script -- confirmed via /proc cmdline so an unrelated GPU job is
        never touched, and skipping our own ancestor chain (the kernel). This is
        the automatic replacement for a manual `pkill -f worker.py`."""
        worker_script = str(SCRIPT_DIR / "worker.py")
        protected = self._protected_pids()
        candidates = self._gpu_pids(gpus)
        # Also catch our worker.py that isn't momentarily pinned to the GPU (e.g.
        # mid-startup, or a ghost whose out_dir differs from this run's), via a
        # /proc scan. Matching the script PATH means only THIS repo's workers are
        # ever touched -- not probe_worker, not some other job. One supervisor runs
        # per machine, so any pre-existing sweep worker here is a stale ghost.
        for pid in _iter_proc_pids():
            cmd = self._proc_cmdline(pid)
            if cmd and worker_script in cmd:
                candidates.add(pid)
        killed = []
        for pid in candidates:
            if pid in protected:
                continue
            cmd = self._proc_cmdline(pid)
            if cmd is None or worker_script not in cmd:
                continue          # gone, or not one of our workers -> leave it
            try:
                os.kill(pid, _SIGKILL)
                killed.append(pid)
            except OSError:
                pass
        if killed:
            print(f"supervisor: auto-killed {len(killed)} stale GPU worker(s) from a "
                  f"previous run (gpus={list(gpus)}): {killed}", flush=True)
            time.sleep(3)   # let the driver reclaim their VRAM before we spawn

    # --- one-off calibration ----------------------------------------------
    def _ensure_calib(self) -> None:
        """Produce/load calib.json once, before any lane worker starts. In a
        multi-GPU run this avoids every lane racing to rewrite it; lane workers
        then run with --no-calib."""
        if self.calib is not None:            # injected (tests) or already loaded
            return
        calib_path = Path(self.work_dir) / "calib.json"
        mode = getattr(self.config.memory, "calib", "measure")
        if mode == "off":
            self.calib = {}
            return
        if not (mode == "measure" or not calib_path.exists()):
            self.calib = oom_proxy.load_calib(calib_path) or {}
            return
        self._pre_calibrate(calib_path)
        self.calib = oom_proxy.load_calib(calib_path) or {}

    def _pre_calibrate(self, calib_path) -> None:
        print(f"supervisor: calibrating once on gpu {self.gpu} ...", flush=True)
        argv = [sys.executable, "-u", str(SCRIPT_DIR / "worker.py"),
                "--config", str(self.config_path), "--device", "cuda", "--calib-only",
                "--out-dir", str(self.out_dir), "--work-dir", str(self.work_dir)]
        if self.h5_override:
            argv += ["--h5", str(self.h5_override)]
        if self.logout_address:
            argv += ["--logout-address", str(self.logout_address)]
        env = {**os.environ,
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "CUDA_VISIBLE_DEVICES": str(self.gpu)}
        p = subprocess.Popen(argv, cwd=str(ROOT), start_new_session=True, env=env)
        try:
            p.wait(timeout=self.ready_timeout)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass

    # --- planning / downgrade ---------------------------------------------
    def _ensure_inputs(self) -> None:
        if self.stats is None:
            self.stats = oom_proxy.GameStats.from_h5(self.config.h5)

    def _ram_budget(self) -> float:
        ram = oom_proxy.available_ram_bytes() * float(getattr(self.config.memory, "ram_safety", 0.8))
        return ram / self._ram_divisor

    def plan_settings(self, combo) -> dict:
        ds = self.config.data_seed
        worst = self.stats.subset_worst_sentences(combo.train_games, ds.train_game_seed, ds.anchors)
        total = self.stats.subset_total_sentences(combo.train_games, ds.train_game_seed, ds.anchors)
        cache_bytes = oom_proxy.estimate_full_cache_bytes(total, combo.view, self.stats.input_dim)
        plan = oom_proxy.plan_combo_chunked(
            self.calib, worst, self._free_vram(), combo.num_latents, combo.view,
            self.config.train.batch_size, safety=self.config.memory.vram_safety, try_paired=False,
            total_sentences=total, cache_bytes=cache_bytes, ram_budget=self._ram_budget())
        # Surface the memory model so it can be sanity-checked against real peaks.
        # Two ORTHOGONAL axes:
        #  * backward_mode (across games): standard = keep every game's graph for one
        #    backward (peak ~ R + C*total); split_recompute = one game at a time (peak
        #    ~ R + C*worst).
        #  * stem chunk (within one game): the latent cross-attention is ALWAYS the
        #    online-softmax + checkpoint path in training (chunk_size>0), so this is
        #    just the CHUNK COUNT -- 1 chunk when it covers the biggest game, more when
        #    it's split. (The fused nn.MultiheadAttention path is eval-only.)
        n = int(plan["stem_chunk_size"])
        fwd_worst = max(1, int(worst * combo.view))          # forwarded sentences, biggest game, one view
        n_chunks = -(-fwd_worst // n)                          # ceil
        stem = "full" if plan.get("chunk_full") else f"chunk={n} (~{n_chunks}/worst-game)"
        print(f"supervisor: gpu{self.gpu} {combo.combo_id} plan: backward={plan['backward_mode']} "
              f"stem={stem} view={combo.view:g} worst={worst} total={total} "
              f"std_peak={plan.get('standard_peak_gib')}GiB budget={plan.get('budget_gib')}GiB "
              f"cache={plan['cache_mode']}", flush=True)
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
    def _progress(self) -> str:
        """Monotonic progress from the shared tally (not the grid dispatch index,
        which jumps around when lanes interleave). Shows how many combos are
        finished and how many remain across ALL lanes."""
        with self._tally_lock:
            d, f, s = self._tally["done"], self._tally["failed"], self._tally["skipped"]
        fin = d + f + s
        left = max(0, self.total - fin)
        return f"[{fin}/{self.total}] done={d} failed={f} skipped={s} ({left} left)"

    def _bump(self, key) -> None:
        with self._tally_lock:
            self._tally[key] += 1

    def _reset_failed(self) -> None:
        n = 0
        for cid, rec in self.ledger.load().items():
            if rec.get("status") == "failed":
                self.ledger.update(cid, status="pending", attempts=0, error=None)
                n += 1
        if n:
            print(f"supervisor: reset {n} failed combos for retry", flush=True)

    def _sharded_combos(self) -> list:
        """This machine's disjoint slice of the grid (combo idx %% N == i)."""
        combos = list(self.config.iter_combos())
        i, n = self._shard
        if n > 1:
            combos = [c for idx, c in enumerate(combos) if idx % n == i]
        return combos

    def _order_combos(self, combos: list) -> list:
        """Schedule order: run the FAST tier first -- 'backward=standard, stem=full'
        (fits in one pass) -- descending batch size so the biggest fast combos start
        early; defer the SLOW tier (split_recompute / multi-chunk) to the end. Batch
        size S = 2*view*total is a machine-independent proxy for peak, so the order
        is stable across GPUs. One plan per combo (cheap; a single VRAM snapshot)."""
        ds = self.config.data_seed
        free = self._free_vram()          # snapshot once (avoid one nvidia-smi per combo)
        ram = self._ram_budget()
        self._std_peak_gib = {}

        def scored(c):
            total = self.stats.subset_total_sentences(c.train_games, ds.train_game_seed, ds.anchors)
            worst = self.stats.subset_worst_sentences(c.train_games, ds.train_game_seed, ds.anchors)
            cache_bytes = oom_proxy.estimate_full_cache_bytes(total, c.view, self.stats.input_dim)
            plan = oom_proxy.plan_combo_chunked(
                self.calib, worst, free, c.num_latents, c.view, self.config.train.batch_size,
                safety=self.config.memory.vram_safety, try_paired=False,
                total_sentences=total, cache_bytes=cache_bytes, ram_budget=ram)
            self._std_peak_gib[c.combo_id] = plan.get("standard_peak_gib")   # portable across VMs
            light = plan["backward_mode"] == "standard" and bool(plan.get("chunk_full"))
            return (0 if light else 1, -(2.0 * float(c.view) * float(total)))

        keyed = [(scored(c), c) for c in combos]
        keyed.sort(key=lambda t: t[0])
        n_light = sum(1 for (tier, _), _ in keyed if tier == 0)
        print(f"supervisor: scheduling {n_light} fast (standard+full) then "
              f"{len(keyed) - n_light} slow (split/chunked), each by descending size", flush=True)
        return [c for _, c in keyed]

    def _prepare(self, gpus=None) -> None:
        """Shared, run once (by the primary lane): reconcile, calibrate, load
        stats, and build the shared grid iterator. Safe to call before spawning
        any lane worker. ``gpus`` are the physical cards this run will use -- they
        get auto-cleared of stale workers first."""
        self._reap_stale_workers(gpus or [self.gpu])   # clear VRAM ghosts before anything
        self.ledger.reconcile_running()
        if self.retry_failed:
            self._reset_failed()
        self._ensure_calib()
        self._ensure_inputs()
        combos = self._order_combos(self._sharded_combos())   # fast tier first, slow tier last
        self.total = len(combos)
        self._ordered_ids = [c.combo_id for c in combos]
        self._combo_by_id = {c.combo_id: c for c in combos}
        self._start_coordinator(gpus or [self.gpu])
        i, n = self._shard
        shard_note = f" (shard {i}/{n} of {self.config.combo_count()})" if n > 1 else ""
        print(f"supervisor: {self.total} combos to process{shard_note} as vm={self.vm_name}", flush=True)

    def _start_coordinator(self, gpus) -> None:
        """One Coordinator per VM (shared by all lanes): file-based, cross-VM claims on
        the shared out_dir. done = checkpoint exists (recognises the CURRENT run's
        finished combos -> seamless migration) OR done.json. Atomic O_EXCL claim means
        no two lanes -- or VMs -- ever train the same combo."""
        def _done_fn(cid):
            c = self._combo_by_id.get(cid)
            return bool(c) and jobspec.combo_paths(self.config, c)["checkpoint"].exists()
        liveness = self._liveness or LeaseLiveness(lease=600.0, refresh=120.0)
        base = self._vm_name_req or _default_vm_name()
        total_vram = 0.0
        try:
            total_vram = self._total_vram()
        except Exception:
            pass
        info = {"gpus": list(gpus), "cores": jobspec.effective_cpu_count(),
                "vram_gib": round(total_vram / oom_proxy.GIB, 1) if total_vram else None}
        self.coordinator = Coordinator(self.out_dir, base, done_fn=_done_fn, liveness=liveness, info=info)
        self.vm_name = self.coordinator.vm_name
        self._my_vram_gib = info["vram_gib"]
        self._fits_fn = self._capability_fits   # skip combos better run on a bigger VM

    def _max_alive_vram(self) -> float | None:
        """Biggest single-card VRAM among ALIVE VMs (registry). Cached ~3s so
        next_claim's per-combo fits check doesn't re-glob the registry each time."""
        now = time.time()
        if now - self._vram_cache_ts > 3.0:
            vrams = [i.get("vram_gib") for _, i in self.coordinator.alive_vms() if i.get("vram_gib")]
            self._vram_cache = max(vrams) if vrams else None
            self._vram_cache_ts = now
        return self._vram_cache

    def _capability_fits(self, cid) -> bool:
        """Heterogeneous routing: claim a combo iff it fits MY card in fast standard,
        OR I'm the biggest alive VM (so a combo too big for EVERY card still gets done
        -- on the biggest card, in split -- never starved). Unknown peak/VRAM -> claim
        (safe default; degrades to no routing, e.g. homogeneous or single VM)."""
        peak = self._std_peak_gib.get(cid)
        my = self._my_vram_gib
        if peak is None or not my:
            return True
        safety = float(getattr(self.config.memory, "vram_safety", 0.85))
        if float(peak) <= my * safety:
            return True                              # fits me -> fast standard here
        biggest = self._max_alive_vram()
        return biggest is None or my >= biggest      # else only the biggest takes it

    def close(self) -> None:
        if self.coordinator is not None:
            self.coordinator.close()

    def _claim(self):
        """Claim the next combo THIS vm should run -- atomic across lanes AND VMs.
        None means nothing is claimable-by-us right now (all done, or in flight on a
        live peer we could later reclaim)."""
        cid = self.coordinator.next_claim(self._ordered_ids, self._fits_fn, lane=self.gpu)
        return self._combo_by_id.get(cid) if cid else None

    def _all_done(self) -> bool:
        return all(self.coordinator.is_terminal(cid) for cid in self._ordered_ids)

    def _lane_loop(self) -> None:
        """Bring this lane's worker up, then drain the shared queue until every combo
        is done (a None claim just means 'nothing for me now' -> wait for peers)."""
        protocol.clear_queue(self.qdir)   # drop stale jobs from a previous crashed run
        self._wait_ready()
        try:
            while not self._all_done():
                combo = self._claim()
                if combo is None:
                    time.sleep(self.poll)      # nothing for us now; wait for peers/reclaim
                    continue
                self._run_combo(combo)
        finally:
            protocol.write_stop(self.qdir)
            self._kill_worker()

    def run(self) -> dict:
        """Single-lane sweep (single GPU / tests): prepare + drain."""
        self._prepare()
        self.no_calib = True       # calib already produced by _prepare; worker skips it
        try:
            self._lane_loop()
        finally:
            self.close()
        summary = self.ledger.summary()
        print(f"supervisor: sweep complete [{self.total}/{self.total}] -> {summary}", flush=True)
        return summary

    def _run_combo(self, combo, position=0, total=0) -> None:
        combo_id = combo.combo_id
        config_hash = self.config.config_hash(combo)
        paths = jobspec.combo_paths(self.config, combo)
        tag = f"gpu{self.gpu}"

        settings = self.plan_settings(combo)
        while True:
            attempts = int((self.ledger.get(combo_id) or {}).get("attempts", 0))
            if attempts >= MAX_ATTEMPTS:
                self.ledger.mark_failed(combo_id, f"exhausted {attempts} attempts")
                self.coordinator.mark_failed(combo_id, f"exhausted {attempts} attempts")  # terminal marker
                self._bump("failed")
                print(f"supervisor: {tag} {combo_id} FAILED after {attempts} attempts | {self._progress()}", flush=True)
                return

            # Single recovery point: if the worker died (mid-combo crash, or it
            # exited itself after an OOM for a clean CUDA context), bring up a
            # fresh one before emitting the next attempt.
            if not self._worker_alive():
                print(f"supervisor: {tag} worker not alive before {combo_id}; recovering", flush=True)
                self._recover_worker()

            protocol.clear_combo(self.qdir, combo_id)
            argv = jobspec.build_trainer_argv(self.config, combo, settings,
                                              probe_queue_dir=self.probe_queue,
                                              data_workers=self._data_workers,
                                              vm_name=self.vm_name)
            protocol.write_job(self.qdir, {
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
                protocol.clear_combo(self.qdir, combo_id)
                if exit_code == -9:   # SIGKILL -> almost certainly the host OOM-killer
                    print(f"supervisor: {tag} {combo_id} worker SIGKILLed (host RAM OOM); RAM-downgrade", flush=True)
                    settings = self.downgrade_ram(settings)
                else:
                    print(f"supervisor: {tag} worker died on {combo_id} (exit={exit_code}); VRAM-downgrade", flush=True)
                    settings = self.downgrade(settings)
                continue
            if result.get("status") == "done":
                self.ledger.mark_done(combo_id, result.get("peak_mem_gib"), result.get("ckpt"))
                protocol.clear_combo(self.qdir, combo_id)
                # Fence at commit: only publish done if we STILL own the claim. If a
                # peer reclaimed it (our lease lapsed), the checkpoint is idempotent and
                # theirs is authoritative -- don't double-count.
                if self.coordinator.fenced_done(combo_id, lane=self.gpu):
                    self._bump("done")
                    print(f"supervisor: {tag} {combo_id} done (peak={result.get('peak_mem_gib')}GiB) | {self._progress()}", flush=True)
                else:
                    print(f"supervisor: {tag} {combo_id} done but claim was reclaimed by a peer; not double-counting", flush=True)
                return
            # oom / error -> downgrade and retry.
            status = result.get("status")
            print(f"supervisor: {tag} {combo_id} {status}: {result.get('error')}; downgrading", flush=True)
            settings = self.downgrade(settings)
            if status == "oom":
                # The worker exits itself after an OOM (clean CUDA context). Bring
                # up a fresh one NOW -- after its VRAM is actually reclaimed
                # (_poll_until_free) -- so the next attempt never allocates on top
                # of the dying worker's memory. Doing it here (vs waiting for the
                # top-of-loop liveness check) avoids dispatching to a worker that
                # is mid-teardown and getting a spurious second downgrade.
                self._recover_worker()

    def _await_result(self, combo_id: str):
        while True:
            result = protocol.read_result(self.qdir, combo_id)
            if result is not None:
                return result
            if not self._worker_alive():
                # give a just-written result one last chance before declaring death
                return protocol.read_result(self.qdir, combo_id)
            time.sleep(self.poll)


def run_sweep(config: SweepConfig, config_path, gpus, *, logout_address=None,
              h5_override=None, retry_failed=False, poll=2.0, ready_timeout=3600.0,
              reclaim_timeout=600.0, spawn_worker=None, free_vram_fn=None,
              calib=None, stats=None, shard=(0, 1), work_dir=None,
              vm_name=None, liveness=None) -> dict:
    """Drive the sweep across ``gpus``. One lane per GPU, each with its own worker
    (pinned via CUDA_VISIBLE_DEVICES) and its own job queue subdir; all lanes share
    one ledger + grid + tally so every combo is trained exactly once. A single GPU
    is just one lane. ``work_dir`` (machine-local) hosts calib.json + the job queue;
    the durable ledger/checkpoints stay under out_dir. ``calib``/``stats`` are
    injectable for testing without a GPU."""
    gpus = list(dict.fromkeys(int(g) for g in gpus)) or [0]
    multi = len(gpus) > 1
    work_base = Path(work_dir) if work_dir else Path(config.out_dir)

    def qdir_for(g):
        base = protocol.default_qdir(work_base)
        return base / f"gpu{g}" if multi else base

    common = dict(config_path=config_path, logout_address=logout_address,
                  h5_override=h5_override, poll=poll, ready_timeout=ready_timeout,
                  reclaim_timeout=reclaim_timeout, no_calib=True, ram_divisor=len(gpus),
                  spawn_worker=spawn_worker, free_vram_fn=free_vram_fn, work_dir=work_dir,
                  vm_name=vm_name, liveness=liveness)

    primary = Supervisor(config, gpu=gpus[0], qdir=qdir_for(gpus[0]),
                         retry_failed=retry_failed, calib=calib, stats=stats,
                         shard=shard, **common)
    primary._prepare(gpus)                    # auto-clear GPUs + calib + stats + coordinator
    lanes = [primary]
    for g in gpus[1:]:
        lane = Supervisor(config, gpu=g, qdir=qdir_for(g), retry_failed=False, **common)
        lane._share_from(primary)
        lanes.append(lane)

    print(f"supervisor: driving {primary.total} combos as vm={primary.vm_name} across gpus={gpus} "
          f"| {jobspec.effective_cpu_count()} cores -> {primary._data_workers} data-workers/lane", flush=True)
    try:
        if len(lanes) == 1:
            lanes[0]._lane_loop()
        else:
            threads = [threading.Thread(target=ln._lane_loop, name=f"lane-gpu{ln.gpu}", daemon=True)
                       for ln in lanes]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    finally:
        primary.close()                        # stop the lease refresher

    summary = primary.ledger.summary()
    print(f"supervisor: sweep complete [{primary.total}/{primary.total}] "
          f"gpus={gpus} vm={primary.vm_name} -> {summary}", flush=True)
    return summary


def _parse_gpus(spec: str) -> list[int]:
    return [int(x) for x in str(spec).replace(",", " ").split()]


def _parse_shard(spec: str) -> tuple[int, int]:
    i, _, n = str(spec).partition("/")
    i, n = int(i), int(n)
    if n < 1 or not (0 <= i < n):
        raise SystemExit(f"--shard must be 'i/N' with 0 <= i < N (got {spec!r})")
    return i, n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--h5", default=None,
                   help="Override config.h5 (e.g. a fast local-disk copy of the embedding H5). "
                        "Only the H5 input is redirected; checkpoints/ledger stay at out_dir.")
    p.add_argument("--out-dir", default=None,
                   help="Override config.out_dir: the DURABLE state (ledger + checkpoints + "
                        "probes). Give each machine its own out_dir when sharding one sweep "
                        "across VMs, so their ledgers never contend on the network FS.")
    p.add_argument("--work-dir", default=None,
                   help="Machine-LOCAL scratch dir (e.g. RunPod /tmp or /root) for calib.json "
                        "(measured on this GPU) + the job queue. Keeps machine-specific + "
                        "transient files off the shared network FS. Defaults to out_dir.")
    p.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    p.add_argument("--retry-failed", action="store_true",
                   help="Reset 'failed' combos (attempts exhausted) back to pending so they "
                        "are retried -- e.g. after a code fix that should make them fit.")
    p.add_argument("--gpu", type=int, default=0,
                   help="Single physical GPU to run on (default 0). Ignored if --gpus is given.")
    p.add_argument("--gpus", default=None,
                   help="Comma/space-separated physical GPU ids to run across concurrently, "
                        "one worker per GPU (e.g. '0,1,2,3'). Combos are split across GPUs via a "
                        "shared in-process grid; each GPU trains a different combo at the same time.")
    p.add_argument("--shard", default=None,
                   help="(Legacy static split) run only combos whose index %% N == i ('i/N'). "
                        "With the default cross-VM coordinator you usually DON'T need this -- all "
                        "VMs share one out_dir and claim combos dynamically. Use only to hard-carve "
                        "the grid across machines that must not share an out_dir.")
    p.add_argument("--vm-name", default=None,
                   help="This machine's name in the shared VM_parallel/ registry (default: "
                        "hostname). Duplicates get a _N suffix. VMs coordinate through the shared "
                        "out_dir, claiming combos atomically so none is trained twice.")
    return p.parse_args(argv)


def run_main(args) -> None:
    config = SweepConfig.load(args.config)
    if args.h5:
        config.h5 = str(args.h5)
    if args.out_dir:
        config.out_dir = str(args.out_dir)
    gpus = _parse_gpus(args.gpus) if args.gpus else [args.gpu]
    shard = _parse_shard(args.shard) if args.shard else (0, 1)
    summary = run_sweep(config, args.config, gpus, logout_address=args.logout_address,
                        h5_override=args.h5, retry_failed=args.retry_failed, shard=shard,
                        work_dir=args.work_dir, vm_name=args.vm_name)
    print(f"sweep done: {summary}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_with_optional_tee(args.logout_address, run_main, args)


if __name__ == "__main__":
    main()
