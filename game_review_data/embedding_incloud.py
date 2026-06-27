"""In-RAM, single-A100 embedding of the unified text H5 into an embedding H5.

Strategy (tuned for one big GPU + lots of host RAM, e.g. 80GB A100 / 200GB RAM):

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
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

# h5_corpus.py / cloud_embedding.py live alongside / one level up from this file.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
    )

DEFAULT_INPUT_H5 = DEFAULT_TEXT_H5
DEFAULT_OUTPUT_H5 = DEFAULT_EMBEDDING_H5
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BACKEND = "local_incloud"
DEFAULT_SHARD_SIZE = 2_000_000
MANIFEST_SCHEMA = "embedding_incloud.manifest.v1"


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


def load_all_texts(input_h5: Path):
    """Read every sentence text into a Python list (held in host RAM)."""
    import h5py

    with h5py.File(input_h5, "r") as h5:
        if "texts" not in h5:
            raise ValueError(f"{input_h5} has no 'texts' dataset")
        raw = h5["texts"][:]
    texts = [decode_text(value) for value in raw]
    return texts


def manifest_path_for(output_h5: Path) -> Path:
    output_h5 = Path(output_h5)
    return output_h5.with_name(output_h5.name + ".incloud_manifest.json")


def working_path_for(output_h5: Path) -> Path:
    """Stable (non-PID) working-file name so it survives restarts for resume."""
    output_h5 = Path(output_h5)
    return output_h5.with_name(output_h5.name + ".incloud.partial.h5")


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

    # --- pre-tokenize the whole shard (bulk, fast-tokenizer Rust-parallel) ---
    # GPU is idle during this CPU pass, so log progress or it looks frozen.
    pre_started = time.time()
    ids_cache: list = [None] * n
    pre_chunk = 50_000
    print(f"{log_prefix}pre-tokenizing {n} sentences (GPU idle until this finishes) ...", flush=True)
    for c in range(0, n, pre_chunk):
        upper = min(c + pre_chunk, n)
        chunk_ids = embedder.tokenize_ids([texts[start + j] for j in range(c, upper)])
        ids_cache[c:upper] = chunk_ids
        if upper == n or (c // pre_chunk) % 5 == 0:
            elapsed = time.time() - pre_started
            rate = upper / elapsed if elapsed > 0 else 0.0
            print(f"{log_prefix}  pre-tokenized {upper}/{n} ({rate:.0f}/s)", flush=True)
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
    token_budget: int = 131072,
    max_batch: int = 8192,
    overwrite: bool = False,
) -> Path:
    """Shard the corpus into contiguous ranges, embed each into the working H5,
    and resume from a manifest if interrupted.

    Each shard is a contiguous slice ``[start, end)`` of the original order and
    is written straight to ``vectors[start:end]``, flushed, then marked done in
    the manifest. Order is therefore identical to ``text_h5.h5`` regardless of
    the in-shard length sort, and a killed run resumes at the last finished shard.
    """
    import gc

    import h5py
    import torch

    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    if output_h5.exists() and not overwrite:
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
    texts = load_all_texts(input_h5)
    total_sentences = len(texts)
    if total_sentences == 0:
        raise ValueError(f"{input_h5} has no sentence texts")
    print(f"embed_incloud: loaded {total_sentences} sentences in {time.time() - started:.1f}s", flush=True)

    embedder = FatGpuEmbedder(local_model, device=device, max_length=max_length, attn_impl=attn_impl)
    dim = embedder.embedding_dim
    vector_dtype = np.dtype(dtype)
    torch_out_dtype = torch.float16 if vector_dtype == np.float16 else torch.float32

    # Align shard size to a whole number of chunks so a chunk never straddles two
    # shards (keeps each shard's write self-contained).
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
            working, input_h5, total_sentences, dim, vector_dtype, normalize, local_model,
            chunk_rows, compression, gzip_level,
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
        f"token_budget≈{token_budget} max_batch={max_batch}" if token_budget and token_budget > 0
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
            texts, start, end, embedder,
            dim=dim, batch_size=batch_size, sort=sort, normalize=normalize,
            tok_workers=tok_workers, prefetch=prefetch,
            vector_dtype=vector_dtype, torch_out_dtype=torch_out_dtype,
            token_budget=token_budget, max_batch=max_batch, max_length=max_length,
            log_prefix=prefix,
        )
        with h5py.File(working, "a") as out:
            out["vectors"][start:end] = buf
            out.flush()
        del buf
        # Release this shard's cached GPU blocks back to the allocator so batch-shape
        # variation across shards can't fragment into an eventual OOM. The model stays
        # resident; only the freed activation cache is returned.
        if str(embedder.device).startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()
        shard["status"] = "done"
        atomic_json_write(manifest, manifest_file)
        elapsed = time.time() - run_started
        print(f"embed_incloud: shard {shard['id']} written + checkpointed (run elapsed={elapsed:.1f}s)", flush=True)

    # All shards present in the working file: promote it to the final output.
    replace_with_retry(Path(working), output_h5)
    best_effort_unlink(manifest_file)
    print(
        f"embed_incloud: wrote {output_h5} sentences={total_sentences} dim={dim} "
        f"total={time.time() - started:.1f}s",
        flush=True,
    )
    return output_h5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
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
        default=131072,
        type=int,
        help="hard cap on real tokens/batch for dynamic batching (0 disables -> fixed "
        "--batch-size); raise it until GPU memory is ~70%% full",
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
    parser.add_argument("--tok-workers", default=4, type=int, help="CPU tokenization threads")
    parser.add_argument("--prefetch", default=8, type=int, help="max tokenized batches kept in flight")
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
    parser.add_argument("--overwrite", action="store_true", help="discard any resume state and recompute")
    return parser.parse_args()


def main():
    args = parse_args()
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
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
