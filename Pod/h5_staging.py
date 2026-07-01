"""Helpers for staging the large embedding H5 onto local RunPod disk.

The local copy is only reused when both the byte size and lightweight H5
metadata checks match the workspace source. A stale partial copy is removed and
rebuilt before training uses it.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


GIB = 1024 ** 3
REQUIRED_DATASETS = ("vectors", "review_offsets", "game_review_offsets", "game_names")


def _free_bytes(path: str | os.PathLike) -> int:
    st = os.statvfs(path)
    return int(st.f_bavail * st.f_frsize)


def _gib(num_bytes: int | float) -> float:
    return float(num_bytes) / GIB


def validate_training_h5_quick(path: str | os.PathLike, *, expected_size: int | None = None) -> dict:
    """Fast validation for the training H5.

    This opens metadata and reads a few scalar offsets. It does not scan the
    150GiB vector matrix.
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_size is not None and size != int(expected_size):
        raise ValueError(f"size mismatch: actual={size} expected={int(expected_size)}")

    with h5py.File(path, "r") as h5:
        missing = [name for name in REQUIRED_DATASETS if name not in h5]
        if missing:
            raise ValueError(f"missing datasets: {missing}")
        if "input_dim" not in h5.attrs:
            raise ValueError("missing root attr: input_dim")

        vectors = h5["vectors"]
        review_offsets = h5["review_offsets"]
        game_review_offsets = h5["game_review_offsets"]
        game_names = h5["game_names"]
        input_dim = int(h5.attrs["input_dim"])

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")
        if int(vectors.shape[1]) != input_dim:
            raise ValueError(f"input_dim={input_dim} but vectors.shape[1]={int(vectors.shape[1])}")
        if review_offsets.ndim != 1 or game_review_offsets.ndim != 1:
            raise ValueError("review_offsets and game_review_offsets must be 1D")
        if int(game_review_offsets.shape[0]) != int(game_names.shape[0]) + 1:
            raise ValueError("game_review_offsets length does not match game_names")
        if int(review_offsets[-1]) != int(vectors.shape[0]):
            raise ValueError("review_offsets[-1] does not match vectors rows")
        if int(game_review_offsets[-1]) != int(review_offsets.shape[0]) - 1:
            raise ValueError("game_review_offsets[-1] does not match review count")

        return {
            "size": size,
            "vectors": tuple(int(x) for x in vectors.shape),
            "input_dim": input_dim,
            "games": int(game_names.shape[0]),
            "reviews": int(review_offsets.shape[0]) - 1,
        }


def parallel_copy(src: str | os.PathLike, dst: str | os.PathLike, *, workers: int = 8, chunk: int = 64 << 20) -> None:
    """Thread-parallel byte-range copy for large H5 files."""
    src = str(src)
    dst = str(dst)
    size = os.path.getsize(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.truncate(size)

    bounds = [(i * size // workers, (i + 1) * size // workers) for i in range(workers)]
    errors = []

    def copy_range(start: int, end: int) -> None:
        sfd = os.open(src, os.O_RDONLY)
        dfd = os.open(dst, os.O_WRONLY)
        try:
            off = start
            while off < end:
                data = os.pread(sfd, min(chunk, end - off), off)
                if not data:
                    break
                os.pwrite(dfd, data, off)
                off += len(data)
        except BaseException as exc:
            errors.append(exc)
        finally:
            os.close(sfd)
            os.close(dfd)

    threads = [threading.Thread(target=copy_range, args=bounds_i) for bounds_i in bounds]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    if os.path.getsize(dst) != size:
        raise OSError(f"size mismatch after copy: {os.path.getsize(dst)} != {size}")


def stage_training_h5(
    workspace_h5: str | os.PathLike,
    local_h5: str | os.PathLike,
    *,
    workers: int = 8,
    headroom_gib: float = 5.0,
) -> str:
    """Return the H5 path training should use, staging to local disk if safe."""
    workspace_h5 = Path(workspace_h5)
    local_h5 = Path(local_h5)

    src_info = validate_training_h5_quick(workspace_h5)
    src_size = int(src_info["size"])
    print(
        f"workspace H5 OK: {workspace_h5} "
        f"size={_gib(src_size):.1f}GiB vectors={src_info['vectors']} input_dim={src_info['input_dim']}",
        flush=True,
    )

    if local_h5.exists():
        local_size = local_h5.stat().st_size
        if local_size != src_size:
            print(
                f"local H5 size mismatch: {local_h5} "
                f"local={_gib(local_size):.1f}GiB workspace={_gib(src_size):.1f}GiB; removing stale copy",
                flush=True,
            )
            local_h5.unlink()
        else:
            try:
                local_info = validate_training_h5_quick(local_h5, expected_size=src_size)
                print(
                    f"local H5 already staged and validated: {local_h5} "
                    f"({ _gib(src_size):.1f} GiB, vectors={local_info['vectors']})",
                    flush=True,
                )
                return str(local_h5)
            except Exception as exc:
                print(f"local H5 failed validation ({type(exc).__name__}: {exc}); removing stale copy", flush=True)
                local_h5.unlink(missing_ok=True)

    free = _free_bytes("/")
    need = src_size + int(float(headroom_gib) * GIB)
    print(f"H5={_gib(src_size):.1f}GiB  local('/') free={_gib(free):.1f}GiB  need~{_gib(need):.1f}GiB", flush=True)
    if free < need:
        print("not enough local space -> using /workspace H5 (expand Container Disk to stage locally)", flush=True)
        return str(workspace_h5)

    tmp = local_h5.with_name(local_h5.name + ".copying")
    try:
        tmp.unlink(missing_ok=True)
        t0 = time.time()
        parallel_copy(workspace_h5, tmp, workers=workers)
        validate_training_h5_quick(tmp, expected_size=src_size)
        tmp.replace(local_h5)
        dt = max(time.time() - t0, 1e-6)
        print(f"staged -> {local_h5} in {dt:.0f}s ({_gib(src_size) / dt:.2f} GiB/s)", flush=True)
        return str(local_h5)
    except BaseException as exc:
        print(f"staging failed ({type(exc).__name__}: {exc}) -> using /workspace H5", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return str(workspace_h5)
