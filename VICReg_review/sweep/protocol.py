"""File-based job/result protocol between supervisor and worker.

Mirrors the probe_queue idiom (on-disk, restart-safe). The supervisor writes one
``<combo_id>.job.json`` and waits for ``<combo_id>.result.json``; the worker polls
for pending jobs, trains, and writes the result. A ``worker.heartbeat`` carries
the worker PID + current combo; ``worker.ready`` signals startup (calibration)
is done; ``STOP`` tells the worker to exit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_JOB_SUFFIX = ".job.json"


def jobs_dir(out_dir) -> Path:
    return Path(out_dir) / "sweep_jobs"


def job_path(out_dir, combo_id: str) -> Path:
    return jobs_dir(out_dir) / f"{combo_id}{_JOB_SUFFIX}"


def result_path(out_dir, combo_id: str) -> Path:
    return jobs_dir(out_dir) / f"{combo_id}.result.json"


def heartbeat_path(out_dir) -> Path:
    return jobs_dir(out_dir) / "worker.heartbeat"


def ready_path(out_dir) -> Path:
    return jobs_dir(out_dir) / "worker.ready"


def stop_path(out_dir) -> Path:
    return jobs_dir(out_dir) / "STOP"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomic_write(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_job(out_dir, job: dict) -> None:
    atomic_write(job_path(out_dir, job["combo_id"]), job)


def write_result(out_dir, result: dict) -> None:
    atomic_write(result_path(out_dir, result["combo_id"]), result)


def read_result(out_dir, combo_id: str) -> dict | None:
    return read_json(result_path(out_dir, combo_id))


def clear_combo(out_dir, combo_id: str) -> None:
    jd = jobs_dir(out_dir)
    Path(job_path(out_dir, combo_id)).unlink(missing_ok=True)
    Path(result_path(out_dir, combo_id)).unlink(missing_ok=True)
    (jd / f"{combo_id}{_JOB_SUFFIX}.done").unlink(missing_ok=True)


def pending_jobs(out_dir) -> list[Path]:
    jd = jobs_dir(out_dir)
    if not jd.exists():
        return []
    out = []
    for p in sorted(jd.glob(f"*{_JOB_SUFFIX}")):
        combo_id = p.name[: -len(_JOB_SUFFIX)]
        if not result_path(out_dir, combo_id).exists():
            out.append(p)
    return out


def mark_job_consumed(job_file) -> None:
    # Tolerant: the supervisor may have already cleared the job (e.g. it reacted
    # to the result and moved on) -- renaming a gone file must not crash the worker.
    try:
        Path(job_file).rename(str(job_file) + ".done")
    except OSError:
        pass


def write_heartbeat(out_dir, pid: int, combo_id: str | None = None) -> None:
    atomic_write(heartbeat_path(out_dir), {"pid": int(pid), "ts": _now(), "combo_id": combo_id})


def write_ready(out_dir, pid: int) -> None:
    atomic_write(ready_path(out_dir), {"pid": int(pid), "ts": _now()})


def write_stop(out_dir) -> None:
    atomic_write(stop_path(out_dir), {"ts": _now()})


def clear_signals(out_dir) -> None:
    for p in (ready_path(out_dir), stop_path(out_dir)):
        Path(p).unlink(missing_ok=True)
