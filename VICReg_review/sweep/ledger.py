"""Append-only JSONL sweep ledger -- the single source of truth for resume.

State machine per (combo_id):

    pending --dispatch--> running --result ok------> done
                             |--- oom/err ---------> failed   (attempts exhausted)
                             |                    \-> pending  (retry, downgraded)
                             \--- worker died -----> interrupted --(reconcile)--> ...

Design rules baked in (from hard-won lessons):

* 'running' records the **worker PID** + a heartbeat timestamp. A 'running' entry
  whose PID is no longer alive is a lie; ``reconcile_running`` reclassifies it as
  'interrupted' on supervisor startup.
* The ledger is **never authoritative for completion**. The supervisor re-verifies
  every non-'done' (and re-checks 'done') against the on-disk checkpoint + the
  trainer's own manifest before skipping a combo. The ledger is bookkeeping +
  crash forensics.
* Writes are **append-only + fsync'd**; a torn last line (mid-write crash) is
  skipped on read, so state transitions never corrupt the file.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

VALID_STATUS = {"pending", "running", "done", "failed", "interrupted"}
_SETTING_KEYS = ("backward_mode", "paired", "stem_chunk_size")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def pid_alive(pid) -> bool:
    if not pid:
        return False
    pid = int(pid)
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        pass
    try:
        os.kill(pid, 0)          # signal 0 = liveness probe (POSIX)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True              # exists but owned by another user
    except OSError:
        return False


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- low level ---------------------------------------------------------
    def _append(self, record: dict) -> dict:
        record = {**record, "ts": _now()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def load(self) -> dict:
        """Latest record per combo_id (last line wins)."""
        latest: dict[str, dict] = {}
        if not self.path.exists():
            return latest
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn final line from a crash mid-write
            cid = rec.get("combo_id")
            if cid:
                latest[cid] = rec
        return latest

    def get(self, combo_id: str) -> dict | None:
        return self.load().get(combo_id)

    def update(self, combo_id: str, **fields) -> dict:
        prior = self.get(combo_id) or {"combo_id": combo_id}
        return self._append({**prior, **fields, "combo_id": combo_id})

    # --- transitions -------------------------------------------------------
    def mark_running(self, combo_id: str, config_hash: str, worker_pid: int, settings: dict) -> dict:
        prior = self.get(combo_id) or {}
        extra = {k: settings[k] for k in _SETTING_KEYS if k in settings}
        return self.update(
            combo_id,
            status="running",
            config_hash=config_hash,
            worker_pid=int(worker_pid),
            heartbeat=_now(),
            attempts=int(prior.get("attempts", 0)) + 1,
            error=None,
            **extra,
        )

    def heartbeat(self, combo_id: str, worker_pid: int) -> dict:
        return self.update(combo_id, status="running", worker_pid=int(worker_pid), heartbeat=_now())

    def mark_done(self, combo_id: str, peak_mem_gib=None, ckpt=None) -> dict:
        return self.update(combo_id, status="done", peak_mem_gib=peak_mem_gib, ckpt=ckpt, error=None)

    def mark_failed(self, combo_id: str, error) -> dict:
        return self.update(combo_id, status="failed", error=str(error))

    def mark_interrupted(self, combo_id: str, reason: str = "") -> dict:
        return self.update(combo_id, status="interrupted", error=reason or None)

    # --- resume helpers ----------------------------------------------------
    def reconcile_running(self) -> list[str]:
        """Reclassify 'running' entries whose worker PID is dead as 'interrupted'.
        Returns the combo_ids that were reconciled."""
        touched = []
        for cid, rec in self.load().items():
            if rec.get("status") == "running" and not pid_alive(rec.get("worker_pid")):
                self.mark_interrupted(cid, reason=f"worker pid {rec.get('worker_pid')} not alive at reconcile")
                touched.append(cid)
        return touched

    def is_done(self, combo_id: str, config_hash: str) -> bool:
        """Ledger-level done check. The supervisor still re-verifies against the
        checkpoint/manifest before trusting this."""
        rec = self.get(combo_id)
        return bool(rec and rec.get("status") == "done" and rec.get("config_hash") == config_hash)

    def summary(self) -> dict:
        return dict(Counter(r.get("status", "?") for r in self.load().values()))
