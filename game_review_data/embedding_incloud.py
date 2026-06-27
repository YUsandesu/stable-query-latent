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
        atomic_h5_path,
        best_effort_unlink,
        compression_kwargs,
        copy_text_h5,
        decode_text,
        replace_with_retry,
        unlink_with_retry,
    )
except ImportError:  # pragma: no cover - direct script execution
    from h5_corpus import (
        DEFAULT_EMBEDDING_H5,
        DEFAULT_TEXT_H5,
        EMBEDDING_H5_SCHEMA,
        atomic_h5_path,
        best_effort_unlink,
        compression_kwargs,
        copy_text_h5,
        decode_text,
        replace_with_retry,
        unlink_with_retry,
    )

DEFAULT_INPUT_H5 = DEFAULT_TEXT_H5
DEFAULT_OUTPUT_H5 = DEFAULT_EMBEDDING_H5
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BACKEND = "local_incloud"


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


def length_sorted_batches(texts, batch_size, sort=True):
    """Yield lists of original indices, batched and (optionally) length-sorted.

    Sorting descending by character length keeps each batch's texts close in
    length, so dynamic padding adds almost no wasted tokens.
    """
    order = list(range(len(texts)))
    if sort:
        order.sort(key=lambda i: len(texts[i]), reverse=True)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


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
    overwrite: bool = False,
) -> Path:
    import h5py
    import torch

    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    if output_h5.exists() and not overwrite:
        print(f"embed_incloud: skip existing {output_h5}", flush=True)
        return output_h5
    if not input_h5.exists():
        raise FileNotFoundError(f"Input text H5 not found: {input_h5}")

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

    est_gb = total_sentences * dim * vector_dtype.itemsize / 1e9
    print(
        f"embed_incloud: device={embedder.device} dim={dim} dtype={vector_dtype} "
        f"-> RAM vectors buffer ~{est_gb:.1f}GB",
        flush=True,
    )
    # Original-order buffer: batch outputs are scattered back by index, so the
    # final write to H5 is a single sequential pass.
    vectors_ram = np.empty((total_sentences, dim), dtype=vector_dtype)

    torch_out_dtype = torch.float16 if vector_dtype == np.float16 else torch.float32
    print(
        f"embed_incloud: sorting {total_sentences} sentences by length "
        f"(sort={sort}) and building batches ...",
        flush=True,
    )
    sort_started = time.time()
    batches = list(length_sorted_batches(texts, batch_size, sort=sort))
    n_batches = len(batches)
    max_in_flight = max(1, prefetch)
    print(
        f"embed_incloud: {n_batches} batches ready in {time.time() - sort_started:.1f}s; "
        f"starting GPU embedding (longest sentences first)",
        flush=True,
    )
    embed_started = time.time()
    done_sentences = 0
    done_batches = 0
    last_log_batches = 0
    log_every = 50

    def tokenize_job(batch_indices):
        batch_texts = [texts[i] for i in batch_indices]
        enc = embedder.tokenize(batch_texts, pin_memory=True)
        return batch_indices, enc

    with ThreadPoolExecutor(max_workers=max(1, tok_workers)) as executor:
        in_flight = {}
        next_submit = 0

        def fill():
            nonlocal next_submit
            while next_submit < n_batches and len(in_flight) < max_in_flight:
                future = executor.submit(tokenize_job, batches[next_submit])
                in_flight[future] = next_submit
                next_submit += 1

        fill()
        while in_flight:
            ready, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in ready:
                in_flight.pop(future)
                batch_indices, enc = future.result()
                vectors = embedder.embed_tokens(enc, normalize=normalize, out_dtype=torch_out_dtype)
                vectors_ram[batch_indices] = vectors
                done_sentences += len(batch_indices)
                done_batches += 1
            fill()
            # Log the first batch (immediate sign of life), then every log_every
            # batches, then the final one.
            if done_batches == 1 or done_batches == n_batches or done_batches - last_log_batches >= log_every:
                last_log_batches = done_batches
                elapsed = time.time() - embed_started
                rate = done_sentences / elapsed if elapsed > 0 else 0.0
                print(
                    f"[embed-incloud] batch {done_batches}/{n_batches} "
                    f"{done_sentences}/{total_sentences} sentences "
                    f"elapsed={elapsed:.1f}s rate={rate:.0f}/s",
                    flush=True,
                )

    print(
        f"embed_incloud: embedding done in {time.time() - embed_started:.1f}s; writing {output_h5}",
        flush=True,
    )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = atomic_h5_path(output_h5)
    unlink_with_retry(tmp_path)
    write_started = time.time()
    try:
        with h5py.File(input_h5, "r") as source, h5py.File(tmp_path, "w") as out:
            copy_text_h5(source, out)
            out.attrs["schema"] = EMBEDDING_H5_SCHEMA
            out.attrs["text_h5"] = str(input_h5.resolve())
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
            vectors_ds = out.create_dataset(
                "vectors",
                shape=(total_sentences, dim),
                chunks=(rows_per_chunk, dim),
                dtype=vector_dtype,
                **compression_kwargs(compression, gzip_level),
            )
            # Sequential, chunk-aligned write of the original-order buffer.
            write_block = max(rows_per_chunk, 1) * 64
            for start in range(0, total_sentences, write_block):
                end = min(start + write_block, total_sentences)
                vectors_ds[start:end] = vectors_ram[start:end]

        replace_with_retry(tmp_path, output_h5)
    except BaseException:
        best_effort_unlink(tmp_path)
        raise

    print(
        f"embed_incloud: wrote {output_h5} sentences={total_sentences} dim={dim} "
        f"write={time.time() - write_started:.1f}s total={time.time() - started:.1f}s",
        flush=True,
    )
    return output_h5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", default=256, type=int, help="GPU forward batch size")
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
    parser.add_argument("--overwrite", action="store_true")
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
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
