"""Out-of-core training cache for per-game embedded JSON files.

The embedded corpus is one JSON file per game (the ``embedded/`` output of the
game_review_data build). For a big corpus those files live on slow cloud storage
(Google Drive), and the whole thing does not fit on local disk. Instead of
merging everything into one giant H5, we keep the per-game files separate and
stream them through a small local working set:

  1. ``build_sequence`` pre-computes a *seeded random order* over the game files
     and writes it to ``train_sequence.json``. Because the order is fixed up
     front, any consumer can prefetch the next files without re-deciding the
     shuffle, and the order is reproducible across runs/machines.

  2. ``PrefetchCache`` walks that order in blocks (file sizes summing to
     ``block_bytes``, e.g. 100 GB). For each block it copies the files from the
     source dir into a local cache dir, hands them to the trainer, then deletes
     them and moves on. With ``prefetch_ahead >= 1`` the next block is copied in
     a background thread while the GPU trains on the current one, so I/O overlaps
     compute (the "cache subprogram").

This module is pure I/O — it has no torch/model dependency, so it stays cheap to
import and easy to test. The trainer (train_vicreg_review.py) builds the actual
DataLoader over each block's local directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from pathlib import Path

import numpy as np

SEQUENCE_VERSION = 1
GIB = 1 << 30


# --------------------------------------------------------------------------- sequence
def build_sequence(source_dir, out_path, seed=42, pattern="*.json"):
    """Write a seeded random ordering of the game files under ``source_dir``.

    The manifest records each file's name and size so a consumer can group the
    order into fixed-size blocks without stat-ing the (slow) source again.
    """
    source_dir = Path(source_dir)
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files matching {pattern!r} in {source_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    entries = [
        {"name": files[int(i)].name, "bytes": int(files[int(i)].stat().st_size)}
        for i in order
    ]
    payload = {
        "version": SEQUENCE_VERSION,
        "seed": int(seed),
        "pattern": pattern,
        "source_dir": str(source_dir.resolve()),
        "count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    print(
        f"build_sequence: {payload['count']} files, "
        f"{payload['total_bytes'] / GIB:.1f} GiB, seed={seed} -> {out_path}",
        flush=True,
    )
    return payload


def load_sequence(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != SEQUENCE_VERSION:
        print(
            f"[warn] {path}: sequence version {payload.get('version')} "
            f"!= expected {SEQUENCE_VERSION}; attempting to use it anyway.",
            flush=True,
        )
    return payload


def group_blocks(entries, block_bytes):
    """Greedily pack ordered entries into blocks summing to <= block_bytes.

    A single file larger than block_bytes still becomes its own block (we never
    split a game file). Order within and across blocks is preserved.
    """
    block_bytes = int(block_bytes)
    blocks = []
    current = []
    current_bytes = 0
    for entry in entries:
        size = int(entry["bytes"])
        if current and current_bytes + size > block_bytes:
            blocks.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += size
    if current:
        blocks.append(current)
    return blocks


# --------------------------------------------------------------------------- cache
class PrefetchCache:
    """Stream blocks of files from ``source_dir`` through ``cache_dir``.

    Usage::

        cache = PrefetchCache(source_dir, cache_dir, blocks, prefetch_ahead=1)
        for block_index, block_dir, block_entries in cache.iter_blocks():
            train_on(block_dir)        # block_dir holds this block's files
            # block is deleted automatically after the loop body advances

    Each block is copied into ``cache_dir/block_<i>/``. Copies run in daemon
    threads; ``prefetch_ahead`` controls how many upcoming blocks are copied
    ahead of the one being consumed (1 = double-buffered). Setting it to 0 makes
    fetch/train/delete strictly sequential (minimal disk, no overlap).
    """

    def __init__(self, source_dir, cache_dir, blocks, prefetch_ahead=1, verbose=True):
        self.source_dir = Path(source_dir)
        self.cache_dir = Path(cache_dir)
        self.blocks = list(blocks)
        self.prefetch_ahead = max(0, int(prefetch_ahead))
        self.verbose = verbose
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._ready = {}    # block_index -> threading.Event (set when copy done)
        self._threads = {}  # block_index -> Thread

    def _block_dir(self, index):
        return self.cache_dir / f"block_{index:05d}"

    def _copy_block(self, index):
        block_dir = self._block_dir(index)
        block_dir.mkdir(parents=True, exist_ok=True)
        copied = skipped = missing = 0
        for entry in self.blocks[index]:
            name = entry["name"]
            src = self.source_dir / name
            dst = block_dir / name
            if dst.exists() and dst.stat().st_size == int(entry["bytes"]):
                skipped += 1
                continue
            if not src.exists():
                print(f"[warn] cache: source file missing, skipping: {src}", flush=True)
                missing += 1
                continue
            tmp = dst.with_name(dst.name + ".tmp")
            try:
                shutil.copyfile(src, tmp)
                tmp.replace(dst)
                copied += 1
            except BaseException as exc:
                tmp.unlink(missing_ok=True)
                print(f"[warn] cache: copy failed for {src}: {exc}", flush=True)
                missing += 1
        if self.verbose:
            block_bytes = sum(int(e["bytes"]) for e in self.blocks[index])
            print(
                f"cache: block {index} ready ({len(self.blocks[index])} files, "
                f"{block_bytes / GIB:.1f} GiB, copied={copied} cached={skipped} "
                f"missing={missing}) -> {block_dir}",
                flush=True,
            )
        self._ready[index].set()

    def _ensure_started(self, index):
        if index < 0 or index >= len(self.blocks):
            return
        with self._lock:
            if index in self._threads:
                return
            self._ready[index] = threading.Event()
            thread = threading.Thread(target=self._copy_block, args=(index,), daemon=True)
            self._threads[index] = thread
            thread.start()

    def iter_blocks(self):
        n = len(self.blocks)
        # Warm up the current block + prefetch window.
        for i in range(min(self.prefetch_ahead + 1, n)):
            self._ensure_started(i)

        for index in range(n):
            self._ensure_started(index)
            self._ready[index].wait()
            # Kick off prefetch of upcoming blocks while this one is consumed.
            for j in range(index + 1, min(index + 1 + self.prefetch_ahead, n)):
                self._ensure_started(j)
            try:
                yield index, self._block_dir(index), self.blocks[index]
            finally:
                self.release(index)

    def release(self, index):
        block_dir = self._block_dir(index)
        if block_dir.exists():
            shutil.rmtree(block_dir, ignore_errors=True)
        with self._lock:
            self._threads.pop(index, None)
            self._ready.pop(index, None)

    def cleanup(self):
        """Remove this cache's block_* dirs (call on shutdown to reclaim disk).

        Only block_* subdirectories are removed, so pointing --cache-dir at a
        directory that holds other files is safe.
        """
        for child in self.cache_dir.glob("block_*"):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)


# --------------------------------------------------------------------------- cli
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Write a seeded random training sequence.")
    build.add_argument("--source-dir", required=True, type=Path,
                       help="Directory of per-game embedded JSON files (on Drive).")
    build.add_argument("--out", required=True, type=Path,
                       help="Output train_sequence.json path.")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--pattern", default="*.npz",
                       help="Glob for per-game corpus files (default *.npz; use *.json for legacy).")

    inspect = sub.add_parser("inspect", help="Summarize a sequence and its block layout.")
    inspect.add_argument("--sequence-file", required=True, type=Path)
    inspect.add_argument("--block-gb", type=float, default=100.0)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "build":
        build_sequence(args.source_dir, args.out, seed=args.seed, pattern=args.pattern)
    elif args.command == "inspect":
        seq = load_sequence(args.sequence_file)
        blocks = group_blocks(seq["files"], int(args.block_gb * GIB))
        print(
            f"sequence: {seq['count']} files, {seq['total_bytes'] / GIB:.1f} GiB, "
            f"seed={seq.get('seed')}, source={seq.get('source_dir')}"
        )
        print(f"blocks at {args.block_gb} GiB each: {len(blocks)}")
        for i, block in enumerate(blocks):
            block_bytes = sum(int(e["bytes"]) for e in block)
            print(f"  block {i:3d}: {len(block):5d} files, {block_bytes / GIB:6.1f} GiB")


if __name__ == "__main__":
    main()
