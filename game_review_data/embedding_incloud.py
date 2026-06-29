"""In-RAM embedding of the unified text H5 into an embedding H5.

Two modes are supported:

    1. one-file output: current monolith behavior for a big local GPU box.
    2. cloud stream output: shard locally, upload shards to Drive, resume from
       a Drive-backed manifest, and let a later server reassemble the final H5.

The original single-file path is still the default. Pass ``--cloud-stream-dir``
to switch to the Drive-backed stream workflow.

Strategy for the one-file path (tuned for one big GPU + lots of host RAM, e.g.
80GB A100 / 200GB RAM):

    1. Load *all* sentence texts from ``text_h5.h5`` into host RAM at once.
    2. Sort sentence indices by length (descending) so each batch packs
       similar-length texts -> dynamic padding wastes almost nothing.
    3. Tokenize batches on a background thread pool (HF fast tokenizers run in
       Rust and release the GIL) while the main thread keeps the GPU busy with
       bf16 forward passes -> CPU tokenization overlaps GPU compute.
    4. Scatter each batch's vectors back to their *original* positions in one
       big RAM matrix, then write that matrix to ``embedding_h5.h5`` in a single
       sequential pass (no random HDF5 chunk writes).

The output keeps every text/review/game dataset from ``text_h5.h5`` and adds the
streamable ``vectors`` dataset, identical in layout to ``embedding_data.py`` so
it is a drop-in ``embedding_h5.h5`` for ``train_vicreg_review_h5.py``.

This is the "single fat GPU" sibling of ``embedding_data.py`` (which streams
disk->disk and targets the cloud TEI endpoint). Use this one on the A100 box.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

# h5_corpus.py / cloud_embedding.py live alongside / one level up from this file.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.logging_tee import run_with_optional_tee

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from game_review_data.h5_corpus import (
        DEFAULT_EMBEDDING_H5,
        DEFAULT_TEXT_H5,
        EMBEDDING_H5_SCHEMA,
        atomic_json_write,
        best_effort_unlink,
        compression_kwargs,
        copy_text_h5,
        decode_text,
        replace_with_retry,
        sync_embedding_release_date,
    )
except ImportError:  # pragma: no cover - direct script execution
    from h5_corpus import (
        DEFAULT_EMBEDDING_H5,
        DEFAULT_TEXT_H5,
        EMBEDDING_H5_SCHEMA,
        atomic_json_write,
        best_effort_unlink,
        compression_kwargs,
        copy_text_h5,
        decode_text,
        replace_with_retry,
        sync_embedding_release_date,
    )

DEFAULT_INPUT_H5 = DEFAULT_TEXT_H5
DEFAULT_OUTPUT_H5 = DEFAULT_EMBEDDING_H5
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BACKEND = "local_incloud"
DEFAULT_SHARD_SIZE = 2_000_000
DEFAULT_TEXT_LOAD_CHUNK_SIZE = 250_000
CPU_DEFAULT_TOKEN_BUDGET = 131072
VRAM_TOKEN_BUDGET_TIERS = (
    (16, 65536),
    (32, 131072),
    (48, 196608),
    (80, 327680),
    (100, 393216),
    (120, 458752),
)
MANIFEST_SCHEMA = "embedding_incloud.manifest.v1"
STREAM_MANIFEST_SCHEMA = "embedding_stream.manifest.v1"
STREAM_STATUS_IN_PROGRESS = "in_progress"
STREAM_STATUS_COMPLETE = "complete"
STREAM_EMBED_PENDING = "pending"
STREAM_EMBED_DONE = "done"
STREAM_UPLOAD_WAIT = "wait"
STREAM_UPLOAD_UPLOADING = "uploading"
STREAM_UPLOAD_FAILED = "failed"
STREAM_UPLOAD_DONE = "done"
STREAM_SOURCE_MONOLITH = "monolith"
STREAM_SOURCE_STREAM = "stream"


def _read_int_file(path: str) -> int | None:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
        return int(raw)
    except (OSError, ValueError):
        return None


def effective_cpu_count() -> int:
    """Return CPUs usable by this process, respecting affinity/cgroup limits."""
    counts: list[int] = []
    for name in ("EMBED_INCLOUD_CPU_COUNT", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(name)
        if value:
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed > 0:
                counts.append(parsed)

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            counts.append(len(affinity(0)))
        except OSError:
            pass

    # cgroup v2: "quota period", or "max period" when unlimited.
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip().split()
        if len(raw) >= 2 and raw[0] != "max":
            quota = int(raw[0])
            period = int(raw[1])
            if quota > 0 and period > 0:
                counts.append(max(1, quota // period))
    except (OSError, ValueError):
        pass

    # cgroup v1 fallback.
    quota = _read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and period and quota > 0:
        counts.append(max(1, quota // period))

    counts.append(os.cpu_count() or 1)
    return max(1, min(counts))


class FatGpuEmbedder:
    """Single-GPU Qwen3-Embedding with bf16 compute and last-token pooling.

    Tokenization is dynamic-padded per batch (no global max_length padding);
    callers feed length-sorted batches so padding stays minimal.
    """

    def __init__(self, model_name, device=None, max_length=2048, attn_impl="auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Left padding => the last real token is always at column -1, so pooling
        # is a single slice with no per-row gather.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model_name = model_name
        self.model = self._load_model(AutoModel, model_name, dtype, attn_impl)
        self.model = self.model.to(self.device).eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    def _load_model(self, AutoModel, model_name, dtype, attn_impl):
        torch = self.torch
        candidates = [attn_impl] if attn_impl != "auto" else ["flash_attention_2", "sdpa", "eager"]
        last_exc = None
        for impl in candidates:
            try:
                return AutoModel.from_pretrained(
                    model_name, torch_dtype=dtype, attn_implementation=impl
                )
            except (ImportError, ValueError, RuntimeError) as exc:
                last_exc = exc
                print(f"embed_incloud: attn_implementation={impl!r} unavailable ({exc}); trying next", flush=True)
        # Last resort: let transformers pick its own default.
        if last_exc is not None:
            print("embed_incloud: falling back to default attention implementation", flush=True)
        return AutoModel.from_pretrained(model_name, torch_dtype=dtype)

    def tokenize(self, texts, pin_memory=False):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if pin_memory and self.device.startswith("cuda"):
            enc = {key: value.pin_memory() for key, value in enc.items()}
        return enc

    def tokenize_ids(self, texts):
        """Ragged, truncated token ids (no padding) -> list of int32 arrays.

        Used to learn each sentence's *true* token length up front, so batches
        can be packed by real token count (robust to CJK/multilingual text where
        chars-per-token varies wildly).
        """
        enc = self.tokenizer(texts, padding=False, truncation=True, max_length=self.max_length)
        return [np.asarray(ids, dtype=np.int32) for ids in enc["input_ids"]]

    def pad_ids(self, ids_list, pin_memory=False):
        """Left-pad a batch of cached id arrays into model-ready tensors."""
        torch = self.torch
        maxlen = max((len(ids) for ids in ids_list), default=1)
        maxlen = max(int(maxlen), 1)
        rows = len(ids_list)
        input_ids = np.full((rows, maxlen), self.pad_id, dtype=np.int64)
        attention = np.zeros((rows, maxlen), dtype=np.int64)
        for row, ids in enumerate(ids_list):
            length = len(ids)
            if length:
                input_ids[row, maxlen - length:] = ids  # left pad
                attention[row, maxlen - length:] = 1
        enc = {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
        }
        if pin_memory and self.device.startswith("cuda"):
            enc = {key: value.pin_memory() for key, value in enc.items()}
        return enc

    def embed_tokens(self, enc, normalize=False, out_dtype=None):
        torch = self.torch
        out_dtype = out_dtype or torch.float16
        gpu = {key: value.to(self.device, non_blocking=True) for key, value in enc.items()}
        with torch.no_grad():
            hidden = self.model(**gpu).last_hidden_state
            pooled = hidden[:, -1]  # left padding -> last token at column -1
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.to(out_dtype).cpu().numpy()


def default_text_load_workers() -> int:
    """Default parallel H5 text-load workers: all visible CPU cores minus one."""
    return max(1, effective_cpu_count() - 1)


def default_pretok_workers() -> int:
    """Default pre-tokenize workers: oversubscribe single-thread tokenizers."""
    return max(1, effective_cpu_count() * 2)


# Per-worker tokenizer, built once via the pool initializer (spawn-safe: each
# worker re-imports transformers and loads its own fast tokenizer rather than
# inheriting the parent's CUDA-initialized state).
_PRETOK_STATE: dict = {}


def _pretok_init(model_name: str, max_length: int) -> None:
    # Many tokenizing processes => keep each tokenizer single-threaded by
    # default so N processes do not each spawn N rayon threads. Set
    # EMBED_INCLOUD_TOKENIZERS_PARALLELISM=true to benchmark nested parallelism.
    parallelism = os.environ.get(
        "EMBED_INCLOUD_TOKENIZERS_PARALLELISM",
        os.environ.get("TOKENIZERS_PARALLELISM", "false"),
    )
    os.environ["TOKENIZERS_PARALLELISM"] = parallelism
    from transformers import AutoTokenizer

    _PRETOK_STATE["tok"] = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    _PRETOK_STATE["max_length"] = int(max_length)


def _pretok_chunk(payload):
    """Tokenize one chunk of texts -> (offset, list[int32 ids]) (no padding)."""
    offset, texts_chunk = payload
    tok = _PRETOK_STATE["tok"]
    max_length = _PRETOK_STATE["max_length"]
    enc = tok(texts_chunk, padding=False, truncation=True, max_length=max_length)
    ids = [np.asarray(item, dtype=np.int32) for item in enc["input_ids"]]
    return offset, ids


def _pretokenize_serial(texts, start, n, embedder, *, pre_chunk, log_prefix, pre_started):
    """Single-thread pre-tokenization (relies on HF fast-tokenizer parallelism)."""
    ids_cache: list = [None] * n
    for c in range(0, n, pre_chunk):
        upper = min(c + pre_chunk, n)
        chunk_ids = embedder.tokenize_ids([texts[start + j] for j in range(c, upper)])
        ids_cache[c:upper] = chunk_ids
        if upper == n or (c // pre_chunk) % 5 == 0:
            elapsed = time.time() - pre_started
            rate = upper / elapsed if elapsed > 0 else 0.0
            print(f"{log_prefix}  pre-tokenized {upper}/{n} ({rate:.0f}/s)", flush=True)
    return ids_cache


def _pretokenize_parallel(
    texts, start, n, *, model_name, max_length, workers, pre_chunk, log_prefix, pre_started
):
    """Fan pre-tokenization across a spawn-based process pool (CPU-bound work)."""
    import multiprocessing as mp

    ids_cache: list = [None] * n
    ranges = [(c, min(c + pre_chunk, n)) for c in range(0, n, pre_chunk)]
    ctx = mp.get_context("spawn")
    done = 0
    last_log = 0
    log_stride = max(pre_chunk * 4, 200_000)
    max_in_flight = max(workers * 2, workers)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_pretok_init,
        initargs=(model_name, int(max_length)),
    ) as executor:
        in_flight = {}
        next_submit = 0

        def fill():
            nonlocal next_submit
            while next_submit < len(ranges) and len(in_flight) < max_in_flight:
                c, upper = ranges[next_submit]
                payload = (c, [texts[start + j] for j in range(c, upper)])
                future = executor.submit(_pretok_chunk, payload)
                in_flight[future] = (c, upper)
                next_submit += 1

        fill()
        while in_flight:
            ready, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in ready:
                in_flight.pop(future)
                offset, ids = future.result()
                ids_cache[offset : offset + len(ids)] = ids
                done += len(ids)
            fill()
            if done == n or done - last_log >= log_stride:
                last_log = done
                elapsed = time.time() - pre_started
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"{log_prefix}  pre-tokenized {done}/{n} ({rate:.0f}/s, {workers} procs)",
                    flush=True,
                )
    return ids_cache


def _load_text_chunk(input_h5: Path, start: int, end: int) -> tuple[int, list[str]]:
    import h5py

    with h5py.File(input_h5, "r") as h5:
        values = h5["texts"].asstr()[int(start) : int(end)].tolist()
    return int(start), values


def load_all_texts(
    input_h5: Path,
    *,
    workers: int | None = None,
    chunk_size: int = DEFAULT_TEXT_LOAD_CHUNK_SIZE,
    method: str = "auto",
    log_prefix: str = "embed_incloud: ",
) -> list[str]:
    """Read every sentence text into a Python list (held in host RAM)."""
    import h5py

    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as h5:
        if "texts" not in h5:
            raise ValueError(f"{input_h5} has no 'texts' dataset")
        total = int(h5["texts"].shape[0])

    if total <= 0:
        return []

    worker_count = int(workers if workers is not None and workers > 0 else default_text_load_workers())
    chunk_size = max(1, int(chunk_size))
    if method == "auto":
        method = "serial"
    if method == "serial":
        worker_count = 1
    if method not in {"thread", "process", "serial"}:
        raise ValueError(f"unknown text preload method: {method}")
    ranges = [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]
    texts: list[str | None] = [None] * total
    started = time.time()

    print(
        f"{log_prefix}parallel text preload: workers={worker_count} "
        f"chunk_size={chunk_size} chunks={len(ranges)} method={method}",
        flush=True,
    )

    done = 0
    last_log = 0
    log_stride = max(chunk_size, 1_000_000)

    def record_chunk(chunk_start: int, chunk: list[str]) -> None:
        nonlocal done, last_log
        texts[chunk_start : chunk_start + len(chunk)] = chunk
        done += len(chunk)
        if done == total or done - last_log >= log_stride:
            last_log = done
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"{log_prefix}  loaded texts {done}/{total} ({rate:.0f}/s)", flush=True)

    if worker_count == 1 or len(ranges) == 1:
        for start, end in ranges:
            chunk_start, chunk = _load_text_chunk(input_h5, start, end)
            record_chunk(chunk_start, chunk)
    else:
        executor_cls = ProcessPoolExecutor if method == "process" else ThreadPoolExecutor
        max_in_flight = min(len(ranges), max(worker_count * 2, worker_count))
        with executor_cls(max_workers=worker_count) as executor:
            in_flight = {}
            next_submit = 0

            def fill():
                nonlocal next_submit
                while next_submit < len(ranges) and len(in_flight) < max_in_flight:
                    start, end = ranges[next_submit]
                    future = executor.submit(_load_text_chunk, input_h5, start, end)
                    in_flight[future] = (start, end)
                    next_submit += 1

            fill()
            while in_flight:
                ready, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in ready:
                    in_flight.pop(future)
                    chunk_start, chunk = future.result()
                    record_chunk(chunk_start, chunk)
                fill()

    missing = sum(1 for text in texts if text is None)
    if missing:
        raise RuntimeError(f"text preload left {missing} rows empty")
    return texts


def manifest_path_for(output_h5: Path) -> Path:
    output_h5 = Path(output_h5)
    return output_h5.with_name(output_h5.name + ".incloud_manifest.json")


def working_path_for(output_h5: Path, output_dir: Path | None = None) -> Path:
    """Stable (non-PID) working-file name so it survives restarts for resume."""
    output_h5 = Path(output_h5)
    base_dir = Path(output_dir) if output_dir is not None else output_h5.parent
    return base_dir / (output_h5.name + ".incloud.partial.h5")


def plan_shards(total: int, shard_size: int) -> list[dict]:
    """Tile [0, total) into contiguous, non-overlapping shards."""
    shards: list[dict] = []
    start = 0
    shard_id = 0
    while start < total:
        end = min(start + shard_size, total)
        shards.append({"id": shard_id, "start": start, "end": end, "rows": end - start, "status": "pending"})
        start = end
        shard_id += 1
    return shards


def stream_manifest_path(cloud_stream_dir: Path) -> Path:
    return Path(cloud_stream_dir) / "stream_manifest.json"


def stream_manifest_bak_path(cloud_stream_dir: Path) -> Path:
    return Path(cloud_stream_dir) / "stream_manifest.json.bak"


def stream_remote_text_h5_path(cloud_stream_dir: Path) -> Path:
    return Path(cloud_stream_dir) / "text_h5.h5"


def stream_remote_shard_path(cloud_stream_dir: Path, shard_id: int) -> Path:
    return Path(cloud_stream_dir) / f"shard_{int(shard_id):05d}.h5"


def stream_config(
    input_h5: Path,
    total_sentences: int,
    dim: int,
    vector_dtype,
    normalize: bool,
    local_model: str,
    shard_size: int,
) -> dict:
    return {
        "text_h5": str(Path(input_h5).resolve()),
        "total_sentences": int(total_sentences),
        "dim": int(dim),
        "dtype": str(np.dtype(vector_dtype)),
        "normalize": bool(normalize),
        "model": str(local_model),
        "shard_size": int(shard_size),
    }


def config_matches(expected: dict, recorded: dict | None) -> bool:
    if not isinstance(recorded, dict):
        return False
    alias_map = {"text_h5": "input_h5", "input_h5": "text_h5"}
    for key, value in expected.items():
        candidate = recorded.get(key)
        if candidate is None and key in alias_map:
            candidate = recorded.get(alias_map[key])
        if candidate != value:
            return False
    return True


def load_json_with_backup(path: Path, backup_path: Path | None = None) -> dict | None:
    path = Path(path)
    candidates = [path]
    if backup_path is not None:
        candidates.append(Path(backup_path))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def write_json_via_move(payload: dict, path: Path, *, stage_dir: Path, backup_path: Path | None = None) -> None:
    path = Path(path)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stage_dir / (path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if backup_path is not None and path.exists():
            shutil.copy2(path, backup_path)
        if path.exists():
            path.unlink()
        shutil.move(str(tmp_path), str(path))
    except BaseException:
        best_effort_unlink(tmp_path)
        raise


def ensure_local_copy(source: Path, target: Path) -> Path:
    source = Path(source)
    target = Path(target)
    if target.exists():
        return target
    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def ensure_text_h5_pair(local_path: Path, remote_path: Path) -> Path:
    local_path = Path(local_path)
    remote_path = Path(remote_path)
    local_exists = local_path.exists()
    remote_exists = remote_path.exists()
    if local_exists and not remote_exists:
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, remote_path)
    elif remote_exists:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)
    else:
        raise FileNotFoundError(f"Neither local nor remote text H5 exists: {local_path} / {remote_path}")
    return local_path if local_path.exists() else remote_path


def move_overwrite(source: Path, target: Path) -> Path:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))
    return target


def load_text_range(input_h5: Path, start: int, end: int) -> list[str]:
    import h5py

    with h5py.File(input_h5, "r") as h5:
        if "texts" not in h5:
            raise ValueError(f"{input_h5} has no 'texts' dataset")
        raw = h5["texts"][int(start) : int(end)]
    return [decode_text(value) for value in raw]


def load_vectors_only_shard(
    path: Path,
    rows: int,
    dim: int,
    vector_dtype,
    *,
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    """Read a shard or monolith slice into a vectors-only ndarray."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "vectors" not in h5:
            raise ValueError(f"{path} has no 'vectors' dataset")
        ds = h5["vectors"]
        if start is None and end is None:
            raw = ds[:]
        else:
            raw = ds[int(start) : int(end)]
        vectors = np.asarray(raw, dtype=np.dtype(vector_dtype))
    if vectors.ndim != 2 or int(vectors.shape[0]) != int(rows) or int(vectors.shape[1]) != int(dim):
        raise ValueError(f"{path}: vectors shape {vectors.shape} does not match rows={rows}, dim={dim}")
    return vectors


def shard_file_payload(
    vectors: np.ndarray,
    shard: dict,
    *,
    dim: int,
    vector_dtype,
    normalize: bool,
    source_text_h5: Path,
    local_model: str,
    compression: str,
    gzip_level: int,
    rows_per_chunk: int,
):
    import h5py

    vectors = np.asarray(vectors, dtype=np.dtype(vector_dtype))
    if vectors.ndim != 2:
        raise ValueError(f"Shard vectors must be 2D, got {vectors.shape}")
    rows, actual_dim = map(int, vectors.shape)
    if rows != int(shard["rows"]) or actual_dim != int(dim):
        raise ValueError(
            f"Shard payload shape {vectors.shape} does not match rows={shard['rows']} dim={dim}"
        )
    tmp = h5py.File  # keep local import usage obvious to linters
    del tmp
    return {
        "vectors": vectors,
        "attrs": {
            "global_start": int(shard["start"]),
            "global_end": int(shard["end"]),
            "shard_id": int(shard["id"]),
            "dim": int(dim),
            "dtype": str(np.dtype(vector_dtype)),
            "normalize": bool(normalize),
            "source_text_h5": str(Path(source_text_h5).resolve()),
            "embedding_backend": EMBEDDING_BACKEND,
            "embedding_model": str(local_model),
            "compression": str(compression),
            "gzip_level": int(gzip_level),
            "rows_per_chunk": int(rows_per_chunk),
        },
    }


def write_vectors_shard(
    shard_path: Path,
    vectors: np.ndarray,
    shard: dict,
    *,
    dim: int,
    vector_dtype,
    normalize: bool,
    source_text_h5: Path,
    local_model: str,
    compression: str,
    gzip_level: int,
    stage_dir: Path,
    rows_per_chunk: int,
) -> Path:
    import h5py

    shard_path = Path(shard_path)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stage_dir / (shard_path.name + ".tmp")
    payload = shard_file_payload(
        vectors,
        shard,
        dim=dim,
        vector_dtype=vector_dtype,
        normalize=normalize,
        source_text_h5=source_text_h5,
        local_model=local_model,
        compression=compression,
        gzip_level=gzip_level,
        rows_per_chunk=rows_per_chunk,
    )
    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset(
                "vectors",
                data=payload["vectors"],
                dtype=np.dtype(vector_dtype),
                chunks=(max(1, min(int(rows_per_chunk), int(vectors.shape[0]))), int(dim)),
                **compression_kwargs(compression, gzip_level),
            )
            for key, value in payload["attrs"].items():
                h5.attrs[key] = value
        move_overwrite(tmp_path, shard_path)
    except BaseException:
        best_effort_unlink(tmp_path)
        raise
    return shard_path


def validate_shard_file(path: Path, shard: dict, dim: int, vector_dtype) -> tuple[bool, str, dict]:
    import h5py

    path = Path(path)
    if not path.exists():
        return False, "missing file", {}
    try:
        with h5py.File(path, "r") as h5:
            if "vectors" not in h5:
                return False, "missing vectors dataset", {}
            vectors = h5["vectors"]
            rows, actual_dim = map(int, vectors.shape)
            actual_dtype = str(vectors.dtype)
            verify = {"rows": rows, "dim": actual_dim, "dtype": actual_dtype}
            if rows != int(shard["rows"]):
                return False, f"rows mismatch: actual={rows} expected={int(shard['rows'])}", verify
            if actual_dim != int(dim):
                return False, f"dim mismatch: actual={actual_dim} expected={int(dim)}", verify
            if actual_dtype != str(np.dtype(vector_dtype)):
                return False, f"dtype mismatch: actual={actual_dtype} expected={str(np.dtype(vector_dtype))}", verify
            if int(h5.attrs.get("global_start", -1)) != int(shard["start"]):
                return False, "global_start attr mismatch", verify
            if int(h5.attrs.get("global_end", -1)) != int(shard["end"]):
                return False, "global_end attr mismatch", verify
            if int(h5.attrs.get("shard_id", -1)) != int(shard["id"]):
                return False, "shard_id attr mismatch", verify
            return True, "ok", verify
    except Exception as exc:
        return False, f"cannot read H5: {exc}", {}


def build_batches(order, lengths, batch_size, token_budget, max_batch):
    """Group length-sorted local indices into batches.

    ``order`` is local positions sorted by *true* token length (descending) and
    ``lengths`` are those true token counts. With ``token_budget > 0`` each batch
    packs ``k`` items so that ``k * head_len <= token_budget`` (head = the batch's
    longest, which sets the padded width). Because lengths are real token counts,
    the padded token total is a hard bound regardless of language — short
    sentences get many per batch, long ones get few, and it never OOMs from a
    bad chars-per-token guess.
    """
    if not token_budget or token_budget <= 0:
        return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]

    budget = max(1, int(token_budget))
    batches = []
    i = 0
    n = len(order)
    while i < n:
        head = max(int(lengths[order[i]]), 1)  # longest in this run -> padded width
        k = max(1, min(int(max_batch), budget // head))
        batches.append(order[i : i + k])
        i += k
    return batches


def detect_available_gib(device: str | None) -> float | None:
    """Return currently available CUDA memory in GiB, or None for non-CUDA."""
    if not device or not str(device).startswith("cuda"):
        return None
    import torch

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device=None)
    except TypeError:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return float(free_bytes) / (1024**3)


def auto_token_budget(
    *,
    device: str | None,
    requested: int,
) -> tuple[int, str]:
    """Pick a token budget tier from the available GPU memory."""
    requested = int(requested)
    if requested > 0:
        return requested, "fixed"
    free_gib = detect_available_gib(device)
    if free_gib is None:
        return int(CPU_DEFAULT_TOKEN_BUDGET), "cpu-default"
    for limit_gib, budget in VRAM_TOKEN_BUDGET_TIERS:
        if free_gib <= limit_gib:
            return int(budget), f"tier<= {limit_gib}GiB ({free_gib:.1f} GiB free)"
    return 524288, f"tier>120GiB ({free_gib:.1f} GiB free)"


def embed_index_range(
    texts,
    start,
    end,
    embedder,
    *,
    dim,
    batch_size,
    sort,
    normalize,
    tok_workers,
    prefetch,
    vector_dtype,
    torch_out_dtype,
    token_budget=0,
    max_batch=8192,
    max_length=2048,
    pre_tok_workers=0,
    pre_tok_chunk=50_000,
    log_prefix="",
):
    """Embed the contiguous original-order range [start, end) -> (end-start, dim).

    Tokenizes the shard once up front (to learn true token lengths and cache the
    ids), packs batches by real token count, then runs the GPU forward from the
    cached ids. ``buf`` row ``j`` always holds sentence ``start + j``, so the
    in-shard length sort never affects output order.
    """
    n = end - start
    buf = np.empty((n, dim), dtype=vector_dtype)

    # --- pre-tokenize the whole shard (CPU-bound; GPU is idle until it ends) ---
    # This is the slow serial spot on fat CPU boxes, so fan it across processes
    # when workers > 1. Each worker holds its own single-threaded tokenizer.
    pre_started = time.time()
    pre_chunk = max(1, int(pre_tok_chunk))
    workers = int(pre_tok_workers) if pre_tok_workers and pre_tok_workers > 0 else default_pretok_workers()
    use_parallel = workers > 1 and n >= 2 * pre_chunk
    print(
        f"{log_prefix}pre-tokenizing {n} sentences "
        f"({'%d procs' % workers if use_parallel else 'serial'}; GPU idle until this finishes) ...",
        flush=True,
    )
    if use_parallel:
        ids_cache = _pretokenize_parallel(
            texts,
            start,
            n,
            model_name=embedder.model_name,
            max_length=max_length,
            workers=workers,
            pre_chunk=pre_chunk,
            log_prefix=log_prefix,
            pre_started=pre_started,
        )
    else:
        ids_cache = _pretokenize_serial(
            texts, start, n, embedder, pre_chunk=pre_chunk, log_prefix=log_prefix, pre_started=pre_started
        )
    lengths = [int(len(ids)) for ids in ids_cache]  # true token lengths
    order = list(range(n))  # local positions; row j == sentence start+j
    if sort:
        order.sort(key=lambda j: lengths[j], reverse=True)
    batches = build_batches(order, lengths, batch_size, token_budget, max_batch)
    n_batches = len(batches)
    print(
        f"{log_prefix}pre-tokenized {n} sentences in {time.time() - pre_started:.1f}s "
        f"-> {n_batches} batches (max token len={max(lengths) if lengths else 0})",
        flush=True,
    )

    max_in_flight = max(1, prefetch)
    started = time.time()
    done_sentences = 0
    done_batches = 0
    last_log_batches = 0
    log_every = 50

    def pad_job(local_batch):
        return local_batch, embedder.pad_ids([ids_cache[j] for j in local_batch], pin_memory=True)

    with ThreadPoolExecutor(max_workers=max(1, tok_workers)) as executor:
        in_flight = {}
        next_submit = 0

        def fill():
            nonlocal next_submit
            while next_submit < n_batches and len(in_flight) < max_in_flight:
                future = executor.submit(pad_job, batches[next_submit])
                in_flight[future] = next_submit
                next_submit += 1

        fill()
        while in_flight:
            ready, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in ready:
                in_flight.pop(future)
                local_batch, enc = future.result()
                vectors = embedder.embed_tokens(enc, normalize=normalize, out_dtype=torch_out_dtype)
                buf[np.asarray(local_batch, dtype=np.int64)] = vectors  # row j == sentence start+j
                done_sentences += len(local_batch)
                done_batches += 1
            fill()
            if done_batches == 1 or done_batches == n_batches or done_batches - last_log_batches >= log_every:
                last_log_batches = done_batches
                elapsed = time.time() - started
                rate = done_sentences / elapsed if elapsed > 0 else 0.0
                print(
                    f"{log_prefix}batch {done_batches}/{n_batches} "
                    f"{done_sentences}/{n} sentences elapsed={elapsed:.1f}s rate={rate:.0f}/s",
                    flush=True,
                )
    return buf


def _build_config(input_h5, total_sentences, dim, vector_dtype, normalize, local_model, shard_size, chunk_rows, compression):
    """Identity of a run; resume is only allowed when this matches the manifest."""
    return {
        "input_h5": str(Path(input_h5).resolve()),
        "total_sentences": int(total_sentences),
        "dim": int(dim),
        "dtype": str(vector_dtype),
        "normalize": bool(normalize),
        "model": str(local_model),
        "shard_size": int(shard_size),
        "chunk_rows": int(chunk_rows),
        "compression": str(compression),
    }


def _create_working_h5(
    working, input_h5, total_sentences, dim, vector_dtype, normalize, local_model, chunk_rows, compression, gzip_level
):
    """Create the working H5 with all text metadata + an empty vectors dataset."""
    import h5py

    Path(working).parent.mkdir(parents=True, exist_ok=True)
    best_effort_unlink(Path(working))
    with h5py.File(input_h5, "r") as source, h5py.File(working, "w") as out:
        copy_text_h5(source, out)
        out.attrs["schema"] = EMBEDDING_H5_SCHEMA
        out.attrs["text_h5"] = str(Path(input_h5).resolve())
        out.attrs["embedding_backend"] = EMBEDDING_BACKEND
        out.attrs["embedding_model"] = str(local_model)
        out.attrs["embedding_dtype"] = str(vector_dtype)
        out.attrs["embedding_normalize"] = bool(normalize)
        out.attrs["embedding_created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out.attrs["input_dim"] = dim
        out.attrs["embedding_dim"] = dim
        out.attrs["dtype"] = str(vector_dtype)
        out.attrs["sentences"] = total_sentences
        rows_per_chunk = max(1, min(int(chunk_rows), total_sentences))
        out.create_dataset(
            "vectors",
            shape=(total_sentences, dim),
            chunks=(rows_per_chunk, dim),
            dtype=vector_dtype,
            **compression_kwargs(compression, gzip_level),
        )


def _load_resumable_manifest(manifest_file, working, config, dim, vector_dtype, total_sentences):
    """Return a usable manifest for resume, or None if a fresh start is required."""
    import h5py

    if not manifest_file.exists() or not Path(working).exists():
        return None
    try:
        manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("config") != config:
        print("embed_incloud: manifest config differs from current run; starting fresh", flush=True)
        return None
    try:
        with h5py.File(working, "r") as h5:
            if "vectors" not in h5 or tuple(h5["vectors"].shape) != (total_sentences, dim):
                return None
            if str(h5["vectors"].dtype) != str(vector_dtype):
                return None
    except Exception as exc:
        print(f"embed_incloud: working file unreadable ({exc}); starting fresh", flush=True)
        return None
    return manifest


def one_file_output(
    input_h5: Path = DEFAULT_INPUT_H5,
    output_h5: Path = DEFAULT_OUTPUT_H5,
    local_model: str = DEFAULT_LOCAL_MODEL,
    device: str | None = None,
    batch_size: int = 256,
    max_length: int = 2048,
    attn_impl: str = "auto",
    normalize: bool = False,
    sort: bool = True,
    tok_workers: int = 4,
    prefetch: int = 8,
    dtype: str = "float16",
    chunk_rows: int = 2048,
    compression: str = "none",
    gzip_level: int = 1,
    shard_size: int = DEFAULT_SHARD_SIZE,
    token_budget: int = 0,
    max_batch: int = 8192,
    pre_tok_workers: int = 0,
    pre_tok_chunk: int = 50_000,
    text_load_workers: int | None = None,
    text_load_chunk_size: int = DEFAULT_TEXT_LOAD_CHUNK_SIZE,
    text_load_method: str = "auto",
    overwrite: bool = False,
) -> Path:
    """Old monolith behavior: embed everything in RAM, then write one H5."""
    import gc

    import h5py
    import torch

    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    if output_h5.exists() and not overwrite:
        changed, reason = sync_embedding_release_date(input_h5, output_h5)
        if changed:
            print(
                f"embed_incloud: {reason} in existing {output_h5}; skip re-embedding",
                flush=True,
            )
            return output_h5
        if reason != "release_date already up to date":
            print(f"embed_incloud: release_date compat sync skipped ({reason})", flush=True)
        print(f"embed_incloud: skip existing {output_h5}", flush=True)
        return output_h5
    if not input_h5.exists():
        raise FileNotFoundError(f"Input text H5 not found: {input_h5}")

    manifest_file = manifest_path_for(output_h5)
    working = working_path_for(output_h5)
    if overwrite:
        best_effort_unlink(working)
        best_effort_unlink(manifest_file)

    started = time.time()
    print(f"embed_incloud: loading all texts from {input_h5} into RAM ...", flush=True)
    texts = load_all_texts(
        input_h5,
        workers=text_load_workers,
        chunk_size=text_load_chunk_size,
        method=text_load_method,
    )
    total_sentences = len(texts)
    if total_sentences == 0:
        raise ValueError(f"{input_h5} has no sentence texts")
    print(f"embed_incloud: loaded {total_sentences} sentences in {time.time() - started:.1f}s", flush=True)

    embedder = FatGpuEmbedder(local_model, device=device, max_length=max_length, attn_impl=attn_impl)
    dim = embedder.embedding_dim
    vector_dtype = np.dtype(dtype)
    torch_out_dtype = torch.float16 if vector_dtype == np.float16 else torch.float32
    token_budget, token_budget_mode = auto_token_budget(
        device=embedder.device,
        requested=token_budget,
    )

    rows_per_chunk = max(1, min(int(chunk_rows), total_sentences))
    shard_size = max(rows_per_chunk, (max(1, int(shard_size)) // rows_per_chunk) * rows_per_chunk)

    config = _build_config(
        input_h5, total_sentences, dim, vector_dtype, normalize, local_model, shard_size, chunk_rows, compression
    )
    manifest = _load_resumable_manifest(manifest_file, working, config, dim, vector_dtype, total_sentences)
    if manifest is None:
        print(
            f"embed_incloud: fresh start; device={embedder.device} dim={dim} dtype={vector_dtype} "
            f"shard_size={shard_size} -> final ~{total_sentences * dim * vector_dtype.itemsize / 1e9:.1f}GB on disk",
            flush=True,
        )
        _create_working_h5(
            working,
            input_h5,
            total_sentences,
            dim,
            vector_dtype,
            normalize,
            local_model,
            chunk_rows,
            compression,
            gzip_level,
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "output_h5": str(output_h5.resolve()),
            "working_h5": str(Path(working).resolve()),
            "config": config,
            "shards": plan_shards(total_sentences, shard_size),
        }
        atomic_json_write(manifest, manifest_file)
    else:
        done = sum(1 for s in manifest["shards"] if s["status"] == "done")
        print(
            f"embed_incloud: resuming {working}; {done}/{len(manifest['shards'])} shards already done",
            flush=True,
        )

    shards = manifest["shards"]
    pending = [s for s in shards if s["status"] != "done"]
    batching = (
        f"token_budget≈{token_budget} ({token_budget_mode}) max_batch={max_batch}" if token_budget and token_budget > 0
        else f"fixed batch_size={batch_size}"
    )
    print(
        f"embed_incloud: {len(shards)} shards total, {len(pending)} to embed "
        f"(shard_size={shard_size}, batching: {batching})",
        flush=True,
    )

    run_started = time.time()
    for shard in shards:
        if shard["status"] == "done":
            continue
        start, end = int(shard["start"]), int(shard["end"])
        prefix = f"[shard {shard['id']}/{len(shards) - 1} {start}:{end}] "
        print(f"embed_incloud: embedding {prefix.strip()}", flush=True)
        buf = embed_index_range(
            texts,
            start,
            end,
            embedder,
            dim=dim,
            batch_size=batch_size,
            sort=sort,
            normalize=normalize,
            tok_workers=tok_workers,
            prefetch=prefetch,
            vector_dtype=vector_dtype,
            torch_out_dtype=torch_out_dtype,
            token_budget=token_budget,
            max_batch=max_batch,
            max_length=max_length,
            pre_tok_workers=pre_tok_workers,
            pre_tok_chunk=pre_tok_chunk,
            log_prefix=prefix,
        )
        with h5py.File(working, "a") as out:
            out["vectors"][start:end] = buf
            out.flush()
        del buf
        if str(embedder.device).startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()
        shard["status"] = "done"
        atomic_json_write(manifest, manifest_file)
        elapsed = time.time() - run_started
        print(f"embed_incloud: shard {shard['id']} written + checkpointed (run elapsed={elapsed:.1f}s)", flush=True)

    replace_with_retry(Path(working), output_h5)
    best_effort_unlink(manifest_file)
    print(
        f"embed_incloud: wrote {output_h5} sentences={total_sentences} dim={dim} "
        f"total={time.time() - started:.1f}s",
        flush=True,
    )
    return output_h5


def stream_manifest_initial_state(
    *,
    input_h5: Path,
    output_h5: Path,
    cloud_stream_dir: Path,
    working_dir: Path,
    total_sentences: int,
    dim: int,
    vector_dtype,
    normalize: bool,
    local_model: str,
    shard_size: int,
    use_existing_monolith: bool,
) -> dict:
    stream_manifest_file = stream_manifest_path(cloud_stream_dir)
    stream_manifest_backup = stream_manifest_bak_path(cloud_stream_dir)
    legacy_manifest_file = manifest_path_for(output_h5)
    expected_config = stream_config(input_h5, total_sentences, dim, vector_dtype, normalize, local_model, shard_size)
    manifest = load_json_with_backup(stream_manifest_file, stream_manifest_backup)
    if manifest is not None and (
        manifest.get("schema") != STREAM_MANIFEST_SCHEMA or not config_matches(expected_config, manifest.get("config"))
    ):
        print("embed_incloud: stream config differs from existing manifest; starting fresh", flush=True)
        manifest = None

    if manifest is None:
        legacy_manifest = load_json_with_backup(legacy_manifest_file)
        if legacy_manifest is not None and config_matches(expected_config, legacy_manifest.get("config")):
            shards = []
            for shard in legacy_manifest.get("shards", []):
                if not isinstance(shard, dict):
                    continue
                shard_id = int(shard.get("id", len(shards)))
                start = int(shard.get("start", 0))
                end = int(shard.get("end", start))
                rows = int(shard.get("rows", max(0, end - start)))
                legacy_status = str(shard.get("status", "pending"))
                source = STREAM_SOURCE_MONOLITH if legacy_status == "done" else STREAM_SOURCE_STREAM
                embed_status = STREAM_EMBED_DONE if legacy_status == "done" else STREAM_EMBED_PENDING
                shards.append(
                    {
                        "id": shard_id,
                        "start": start,
                        "end": end,
                        "rows": rows,
                        "embed_status": embed_status,
                        "upload_status": STREAM_UPLOAD_WAIT,
                        "source": source,
                        "remote_path": str(stream_remote_shard_path(cloud_stream_dir, shard_id)),
                        "bytes": 0,
                        "verify": {},
                    }
                )
            manifest = {
                "schema": STREAM_MANIFEST_SCHEMA,
                "config": expected_config,
                "cloud_stream_dir": str(Path(cloud_stream_dir).resolve()),
                "created_at": legacy_manifest.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": STREAM_STATUS_IN_PROGRESS,
                "shards": shards,
            }
        else:
            manifest = {
                "schema": STREAM_MANIFEST_SCHEMA,
                "config": expected_config,
                "cloud_stream_dir": str(Path(cloud_stream_dir).resolve()),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": STREAM_STATUS_IN_PROGRESS,
                "shards": [],
            }
    if not manifest.get("shards"):
        shards = plan_shards(total_sentences, shard_size)
        for shard in shards:
            shard["embed_status"] = STREAM_EMBED_PENDING
            shard["upload_status"] = STREAM_UPLOAD_WAIT
            shard["source"] = STREAM_SOURCE_STREAM
            shard["remote_path"] = str(stream_remote_shard_path(cloud_stream_dir, shard["id"]))
            shard["bytes"] = 0
            shard["verify"] = {}
        manifest["shards"] = shards
    else:
        manifest["shards"] = list(manifest["shards"])

    monolith_path = working_path_for(output_h5, output_dir=working_dir)
    if use_existing_monolith and monolith_path.exists():
        for shard in manifest["shards"]:
            if shard.get("source") == STREAM_SOURCE_MONOLITH and shard.get("upload_status") != STREAM_UPLOAD_DONE:
                shard["upload_status"] = STREAM_UPLOAD_WAIT
                shard["embed_status"] = STREAM_EMBED_DONE
                shard["remote_path"] = str(stream_remote_shard_path(cloud_stream_dir, shard["id"]))
    elif not monolith_path.exists():
        for shard in manifest["shards"]:
            if shard.get("source") == STREAM_SOURCE_MONOLITH and shard.get("upload_status") != STREAM_UPLOAD_DONE:
                shard["embed_status"] = STREAM_EMBED_PENDING
                shard["source"] = STREAM_SOURCE_STREAM
    elif not use_existing_monolith:
        for shard in manifest["shards"]:
            if shard.get("source") == STREAM_SOURCE_MONOLITH and shard.get("upload_status") != STREAM_UPLOAD_DONE:
                shard["embed_status"] = STREAM_EMBED_PENDING
                shard["source"] = STREAM_SOURCE_STREAM
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return manifest


def copy_monolith_shard_to_drive(
    *,
    monolith_path: Path,
    shard: dict,
    cloud_stream_dir: Path,
    output_dir: Path,
    vector_dtype,
    dim: int,
    normalize: bool,
    local_model: str,
    free_floor_bytes: int,
    compression: str,
    gzip_level: int,
    stage_dir: Path,
) -> tuple[bool, str]:
    free_bytes = shutil.disk_usage("/content").free if Path("/content").exists() else shutil.disk_usage(output_dir).free
    if int(free_bytes) <= int(free_floor_bytes):
        return False, f"free space below floor: {free_bytes} <= {free_floor_bytes}"
    if not Path(monolith_path).exists():
        return False, f"missing monolith {monolith_path}"

    rows = int(shard["rows"])
    start = int(shard["start"])
    end = int(shard["end"])
    vectors = load_vectors_only_shard(
        Path(monolith_path),
        rows=rows,
        dim=dim,
        vector_dtype=vector_dtype,
        start=start,
        end=end,
    )
    shard_path = Path(output_dir) / f"shard_{int(shard['id']):05d}.h5"
    write_vectors_shard(
        shard_path,
        vectors,
        shard,
        dim=dim,
        vector_dtype=vector_dtype,
        normalize=normalize,
        source_text_h5=Path(monolith_path),
        local_model=local_model,
        compression=compression,
        gzip_level=gzip_level,
        stage_dir=stage_dir,
        rows_per_chunk=max(1, min(rows, rows)),
    )
    remote_path = stream_remote_shard_path(cloud_stream_dir, shard["id"])
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    move_overwrite(shard_path, remote_path)
    ok, reason, verify = validate_shard_file(remote_path, shard, dim, vector_dtype)
    if not ok:
        return False, f"verify failed: {reason}"
    shard["upload_status"] = STREAM_UPLOAD_DONE
    shard["bytes"] = int(remote_path.stat().st_size)
    shard["verify"] = verify
    return True, str(remote_path)


def stream_cloud_output(
    input_h5: Path = DEFAULT_INPUT_H5,
    output_h5: Path = DEFAULT_OUTPUT_H5,
    local_model: str = DEFAULT_LOCAL_MODEL,
    device: str | None = None,
    batch_size: int = 256,
    max_length: int = 2048,
    attn_impl: str = "auto",
    normalize: bool = False,
    sort: bool = True,
    tok_workers: int = 4,
    prefetch: int = 8,
    dtype: str = "float16",
    chunk_rows: int = 2048,
    compression: str = "none",
    gzip_level: int = 1,
    shard_size: int = DEFAULT_SHARD_SIZE,
    token_budget: int = 0,
    max_batch: int = 8192,
    pre_tok_workers: int = 0,
    pre_tok_chunk: int = 50_000,
    text_load_workers: int | None = None,
    text_load_chunk_size: int = DEFAULT_TEXT_LOAD_CHUNK_SIZE,
    text_load_method: str = "auto",
    overwrite: bool = False,
    output_dir: Path | None = None,
    cloud_stream_dir: Path | None = None,
    free_floor_bytes: int = 8 * 1024**3,
) -> Path:
    """Drive-backed streaming workflow for Colab-style limited local storage."""
    import gc
    import h5py
    import torch

    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    output_dir = Path(output_dir) if output_dir is not None else output_h5.parent
    cloud_stream_dir = Path(cloud_stream_dir) if cloud_stream_dir is not None else None
    if cloud_stream_dir is None:
        raise ValueError("cloud_stream_dir is required for stream output")

    cloud_stream_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    working = working_path_for(output_h5, output_dir=output_dir)
    if overwrite:
        best_effort_unlink(working)
        best_effort_unlink(output_h5)
        best_effort_unlink(stream_manifest_path(cloud_stream_dir))
        best_effort_unlink(stream_manifest_bak_path(cloud_stream_dir))
        best_effort_unlink(manifest_path_for(output_h5))

    remote_text_h5 = stream_remote_text_h5_path(cloud_stream_dir)
    input_h5 = ensure_text_h5_pair(input_h5, remote_text_h5)

    started = time.time()
    print(f"embed_incloud: loading all texts from {input_h5} into RAM ...", flush=True)
    texts = load_all_texts(
        input_h5,
        workers=text_load_workers,
        chunk_size=text_load_chunk_size,
        method=text_load_method,
    )
    total_sentences = len(texts)
    if total_sentences == 0:
        raise ValueError(f"{input_h5} has no sentence texts")
    print(f"embed_incloud: loaded {total_sentences} sentences in {time.time() - started:.1f}s", flush=True)

    embedder = FatGpuEmbedder(local_model, device=device, max_length=max_length, attn_impl=attn_impl)
    dim = embedder.embedding_dim
    vector_dtype = np.dtype(dtype)
    torch_out_dtype = torch.float16 if vector_dtype == np.float16 else torch.float32
    token_budget, token_budget_mode = auto_token_budget(
        device=embedder.device,
        requested=token_budget,
    )
    rows_per_chunk = max(1, min(int(chunk_rows), total_sentences))
    shard_size = max(rows_per_chunk, (max(1, int(shard_size)) // rows_per_chunk) * rows_per_chunk)
    manifest = stream_manifest_initial_state(
        input_h5=input_h5,
        output_h5=output_h5,
        cloud_stream_dir=cloud_stream_dir,
        working_dir=output_dir,
        total_sentences=total_sentences,
        dim=dim,
        vector_dtype=vector_dtype,
        normalize=normalize,
        local_model=local_model,
        shard_size=shard_size,
        use_existing_monolith=Path(working).exists(),
    )

    stream_manifest_file = stream_manifest_path(cloud_stream_dir)
    stream_manifest_backup = stream_manifest_bak_path(cloud_stream_dir)
    if not stream_manifest_file.exists():
        write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)

    if not Path(working).exists() and manifest and any(
        shard.get("source") == STREAM_SOURCE_MONOLITH and shard.get("upload_status") != STREAM_UPLOAD_DONE
        for shard in manifest.get("shards", [])
    ):
        print("embed_incloud: monolith missing; pending monolith shards will be re-embedded from text_h5", flush=True)

    shards = manifest["shards"]
    print(
        f"embed_incloud: stream mode -> {len(shards)} shards total "
        f"(shard_size={shard_size}, free_floor={free_floor_bytes}, token_budget={token_budget} {token_budget_mode})",
        flush=True,
    )

    # Phase 1: if monolith exists, copy shards out first.
    monolith_path = Path(working)
    if monolith_path.exists():
        print(f"embed_incloud: phase 1 copy from monolith {monolith_path}", flush=True)
        for shard in shards:
            if shard.get("upload_status") == STREAM_UPLOAD_DONE:
                continue
            if shard.get("source") != STREAM_SOURCE_MONOLITH:
                continue
            while shutil.disk_usage(output_dir).free <= int(free_floor_bytes):
                time.sleep(2.0)
            shard["upload_status"] = STREAM_UPLOAD_UPLOADING
            shard["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
            start = int(shard["start"])
            end = int(shard["end"])
            buf = load_vectors_only_shard(
                monolith_path,
                rows=int(shard["rows"]),
                dim=dim,
                vector_dtype=vector_dtype,
                start=start,
                end=end,
            )
            shard["source"] = STREAM_SOURCE_MONOLITH
            shard_path = write_vectors_shard(
                output_dir / f"shard_{int(shard['id']):05d}.h5",
                buf,
                shard,
                dim=dim,
                vector_dtype=vector_dtype,
                normalize=normalize,
                source_text_h5=input_h5,
                local_model=local_model,
                compression=compression,
                gzip_level=gzip_level,
                stage_dir=output_dir,
                rows_per_chunk=rows_per_chunk,
            )
            remote_path = stream_remote_shard_path(cloud_stream_dir, shard["id"])
            move_overwrite(shard_path, remote_path)
            ok, reason, verify = validate_shard_file(remote_path, shard, dim, vector_dtype)
            if not ok:
                shard["upload_status"] = STREAM_UPLOAD_FAILED
                shard["verify"] = {"error": reason}
                write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
                print(f"embed_incloud: shard {shard['id']} verify failed: {reason}", flush=True)
                continue
            shard["upload_status"] = STREAM_UPLOAD_DONE
            shard["bytes"] = int(remote_path.stat().st_size)
            shard["verify"] = verify
            shard["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
            print(f"embed_incloud: shard {shard['id']} uploaded from monolith", flush=True)
        if all(shard.get("upload_status") == STREAM_UPLOAD_DONE for shard in shards if shard.get("source") == STREAM_SOURCE_MONOLITH):
            best_effort_unlink(monolith_path)
            print(f"embed_incloud: removed monolith {monolith_path}", flush=True)

    # Phase 2: embed any remaining pending shards.
    for shard in shards:
        if shard.get("upload_status") == STREAM_UPLOAD_DONE:
            continue
        while shutil.disk_usage(output_dir).free <= int(free_floor_bytes):
            time.sleep(2.0)
        start = int(shard["start"])
        end = int(shard["end"])
        shard["embed_status"] = STREAM_EMBED_PENDING
        shard["upload_status"] = STREAM_UPLOAD_UPLOADING
        shard["source"] = STREAM_SOURCE_STREAM
        shard["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
        print(f"embed_incloud: embedding shard {shard['id']} [{start}:{end}]", flush=True)
        buf = embed_index_range(
            texts,
            start,
            end,
            embedder,
            dim=dim,
            batch_size=batch_size,
            sort=sort,
            normalize=normalize,
            tok_workers=tok_workers,
            prefetch=prefetch,
            vector_dtype=vector_dtype,
            torch_out_dtype=torch_out_dtype,
            token_budget=token_budget,
            max_batch=max_batch,
            max_length=max_length,
            pre_tok_workers=pre_tok_workers,
            pre_tok_chunk=pre_tok_chunk,
            log_prefix=f"[stream {shard['id']}] ",
        )
        shard_path = write_vectors_shard(
            output_dir / f"shard_{int(shard['id']):05d}.h5",
            buf,
            shard,
            dim=dim,
            vector_dtype=vector_dtype,
            normalize=normalize,
            source_text_h5=input_h5,
            local_model=local_model,
            compression=compression,
            gzip_level=gzip_level,
            stage_dir=output_dir,
            rows_per_chunk=rows_per_chunk,
        )
        shard["embed_status"] = STREAM_EMBED_DONE
        shard["source"] = STREAM_SOURCE_STREAM
        write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
        remote_path = stream_remote_shard_path(cloud_stream_dir, shard["id"])
        move_overwrite(shard_path, remote_path)
        ok, reason, verify = validate_shard_file(remote_path, shard, dim, vector_dtype)
        if not ok:
            shard["upload_status"] = STREAM_UPLOAD_FAILED
            shard["verify"] = {"error": reason}
            write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
            print(f"embed_incloud: shard {shard['id']} verify failed: {reason}", flush=True)
            continue
        shard["upload_status"] = STREAM_UPLOAD_DONE
        shard["bytes"] = int(remote_path.stat().st_size)
        shard["verify"] = verify
        shard["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
        del buf
        if str(embedder.device).startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()
        print(f"embed_incloud: shard {shard['id']} streamed to Drive", flush=True)

    if all(shard.get("upload_status") == STREAM_UPLOAD_DONE for shard in shards):
        manifest["status"] = STREAM_STATUS_COMPLETE
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json_via_move(manifest, stream_manifest_file, stage_dir=output_dir, backup_path=stream_manifest_backup)
    print(f"embed_incloud: stream run complete -> {stream_manifest_file}", flush=True)
    return stream_manifest_file


def embed_incloud(
    input_h5: Path = DEFAULT_INPUT_H5,
    output_h5: Path = DEFAULT_OUTPUT_H5,
    local_model: str = DEFAULT_LOCAL_MODEL,
    device: str | None = None,
    batch_size: int = 256,
    max_length: int = 2048,
    attn_impl: str = "auto",
    normalize: bool = False,
    sort: bool = True,
    tok_workers: int = 4,
    prefetch: int = 8,
    dtype: str = "float16",
    chunk_rows: int = 2048,
    compression: str = "none",
    gzip_level: int = 1,
    shard_size: int = DEFAULT_SHARD_SIZE,
    token_budget: int = 0,
    max_batch: int = 8192,
    pre_tok_workers: int = 0,
    pre_tok_chunk: int = 50_000,
    text_load_workers: int | None = None,
    text_load_chunk_size: int = DEFAULT_TEXT_LOAD_CHUNK_SIZE,
    text_load_method: str = "auto",
    overwrite: bool = False,
    output_dir: Path | None = None,
    cloud_stream_dir: Path | None = None,
    free_floor_bytes: int = 8 * 1024**3,
) -> Path:
    """Public entry point: one-file default, cloud stream when requested."""
    if cloud_stream_dir is not None:
        return stream_cloud_output(
            input_h5=input_h5,
            output_h5=output_h5,
            local_model=local_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            attn_impl=attn_impl,
            normalize=normalize,
            sort=sort,
            tok_workers=tok_workers,
            prefetch=prefetch,
            dtype=dtype,
            chunk_rows=chunk_rows,
            compression=compression,
            gzip_level=gzip_level,
            shard_size=shard_size,
            token_budget=token_budget,
            max_batch=max_batch,
            pre_tok_workers=pre_tok_workers,
            pre_tok_chunk=pre_tok_chunk,
            text_load_workers=text_load_workers,
            text_load_chunk_size=text_load_chunk_size,
            text_load_method=text_load_method,
            overwrite=overwrite,
            output_dir=output_dir,
            cloud_stream_dir=cloud_stream_dir,
            free_floor_bytes=free_floor_bytes,
        )
    return one_file_output(
        input_h5=input_h5,
        output_h5=output_h5,
        local_model=local_model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        attn_impl=attn_impl,
        normalize=normalize,
        sort=sort,
        tok_workers=tok_workers,
        prefetch=prefetch,
        dtype=dtype,
        chunk_rows=chunk_rows,
        compression=compression,
        gzip_level=gzip_level,
        shard_size=shard_size,
        token_budget=token_budget,
        max_batch=max_batch,
        pre_tok_workers=pre_tok_workers,
        pre_tok_chunk=pre_tok_chunk,
        text_load_workers=text_load_workers,
        text_load_chunk_size=text_load_chunk_size,
        text_load_method=text_load_method,
        overwrite=overwrite,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--output-dir", type=Path, default=None, help="local temporary output directory")
    parser.add_argument("--cloud-stream-dir", type=Path, default=None, help="Drive directory for streamed shards")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="fixed sentences/batch; only used when --token-budget 0",
    )
    parser.add_argument(
        "--token-budget",
        default=0,
        type=int,
        help="tokens/batch; 0 auto-picks a VRAM tier after the model loads, non-zero locks the value",
    )
    parser.add_argument(
        "--max-batch",
        default=8192,
        type=int,
        help="cap on sentences/batch under --token-budget (guards short-sentence batches)",
    )
    parser.add_argument("--max-length", default=2048, type=int)
    parser.add_argument(
        "--attn-impl",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="auto tries flash_attention_2 -> sdpa -> eager",
    )
    parser.add_argument("--normalize", action="store_true", help="L2-normalize output vectors")
    parser.add_argument("--no-sort", dest="sort", action="store_false", help="disable length-sorted batching")
    parser.add_argument("--tok-workers", default=4, type=int, help="CPU pad/collate threads during GPU forward")
    parser.add_argument(
        "--pre-tok-workers",
        default=0,
        type=int,
        help="pre-tokenize processes (CPU-bound up-front pass); 0 uses cpu_count()-1, 1 forces serial",
    )
    parser.add_argument(
        "--pre-tok-chunk",
        default=50_000,
        type=int,
        help="sentences per pre-tokenize task",
    )
    parser.add_argument("--prefetch", default=8, type=int, help="max tokenized batches kept in flight")
    parser.add_argument(
        "--text-load-workers",
        default=0,
        type=int,
        help="parallel H5 text preload workers; 0 uses cpu_count()-1",
    )
    parser.add_argument(
        "--text-load-chunk-size",
        default=DEFAULT_TEXT_LOAD_CHUNK_SIZE,
        type=int,
        help="sentences per H5 text preload task",
    )
    parser.add_argument(
        "--text-load-method",
        choices=["auto", "thread", "process", "serial"],
        default="auto",
        help="H5 text preload executor; auto uses optimized serial HDF5 reads",
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--chunk-rows", default=2048, type=int)
    parser.add_argument("--compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--gzip-level", default=1, type=int)
    parser.add_argument(
        "--shard-size",
        default=DEFAULT_SHARD_SIZE,
        type=int,
        help="sentences per shard/checkpoint (aligned down to a multiple of --chunk-rows)",
    )
    parser.add_argument("--free-floor-bytes", default=8 * 1024**3, type=int, help="minimum free local bytes before work continues")
    parser.add_argument("--overwrite", action="store_true", help="discard any resume state and recompute")
    parser.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_with_optional_tee(args.logout_address, run_main, args)


def run_main(args):
    embed_incloud(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        local_model=args.local_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        attn_impl=args.attn_impl,
        normalize=args.normalize,
        sort=args.sort,
        tok_workers=args.tok_workers,
        prefetch=args.prefetch,
        dtype=args.dtype,
        chunk_rows=args.chunk_rows,
        compression=args.compression,
        gzip_level=args.gzip_level,
        shard_size=args.shard_size,
        token_budget=args.token_budget,
        max_batch=args.max_batch,
        pre_tok_workers=args.pre_tok_workers,
        pre_tok_chunk=args.pre_tok_chunk,
        text_load_workers=args.text_load_workers,
        text_load_chunk_size=args.text_load_chunk_size,
        text_load_method=args.text_load_method,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
        cloud_stream_dir=args.cloud_stream_dir,
        free_floor_bytes=args.free_floor_bytes,
    )


if __name__ == "__main__":
    main()
