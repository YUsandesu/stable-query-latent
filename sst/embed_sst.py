"""Embed the SST `default_*` splits produced by sst_clean.py, keeping each
sentence's label next to its embedding.

For every ``clean/default_<split>.csv`` (split = test/dev/train) this reads the
``sentence`` and ``label`` columns, embeds the sentences, and writes one JSON file
per split to ``clean/sentence_embeddings/default_<split>.json`` as a list of
records:

    {"sentence": "...", "label": 0.5, "embedding": [ ... 1024 floats ... ]}

Backend (user choice):
    --backend cloud   remote TEI endpoint (token from tokenAPI.txt)
    --backend local   local Qwen3-Embedding via transformers (GPU/CPU)

Both backends are the same ``LocalEmbedder`` / ``CloudEmbedder`` classes used by
``embedding_data.py``, so embedding semantics are identical to the game-review
pipeline. Records are streamed to a .tmp file in outer chunks so memory stays
bounded for large splits, then atomically renamed on success.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# embedding_data.py (and the cloud_embedding.py it imports) live in
# ../game_review_data/. Put that on sys.path so they import cleanly from any cwd.
_GAME_REVIEW_DATA = Path(__file__).resolve().parent.parent / "game_review_data"
if str(_GAME_REVIEW_DATA) not in sys.path:
    sys.path.insert(0, str(_GAME_REVIEW_DATA))

from embedding_data import DEFAULT_LOCAL_MODEL, CloudEmbedder, LocalEmbedder  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = "clean"
DEFAULT_OUTPUT_DIR = "clean/sentence_embeddings"
DEFAULT_SPLITS = ["test", "dev", "train"]
# How many records the writer holds in memory between flushes. SST splits are
# small (<=8.5k), so the default is fine; bump it down for tighter memory.
DEFAULT_OUTER_CHUNK = 1024


def resolve_script_relative(path):
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def load_rows(csv_path):
    """Return ordered [{'sentence', 'label'}], skipping rows with empty text."""
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            sentence = (row.get("sentence") or "").strip()
            if not sentence:
                continue
            rows.append({"sentence": sentence, "label": float(row["label"])})
    return rows


def embed_split_streaming(records, embedder, out_path, outer_chunk):
    """Embed records in outer chunks and stream {sentence,label,embedding} JSON
    objects to a .tmp file in input order. On any error the .tmp is removed so a
    failed run never leaves a half-written output that resume logic would skip.
    Returns (written_count, embedding_dim)."""
    tmp_path = out_path.with_suffix(".json.tmp")
    count = 0
    dim = 0
    separator = ""
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            file.write("[")
            for start in range(0, len(records), outer_chunk):
                chunk = records[start : start + outer_chunk]
                texts = [record["sentence"] for record in chunk]
                vectors = embedder.embed(texts)
                for record, vector in zip(chunk, vectors):
                    file.write(separator)
                    file.write(
                        json.dumps(
                            {"sentence": record["sentence"], "label": record["label"], "embedding": vector},
                            ensure_ascii=False,
                        )
                    )
                    separator = ","
                    count += 1
                    if dim == 0 and vector:
                        dim = len(vector)
            file.write("]")
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return count, dim


def build_embedder(args):
    if args.backend == "local":
        embedder = LocalEmbedder(args.local_model, device=args.device, batch_size=args.batch_size)
        return embedder, None
    embedder = CloudEmbedder(
        base_url=args.base_url,
        token_file=args.token_file,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_in_flight=args.max_in_flight,
        normalize=args.normalize,
    )
    return embedder, embedder.close


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path,
                        help="Dir containing default_<split>.csv (sst_clean.py output).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                        choices=["test", "dev", "train"])
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud",
                        help="Embedding backend: local Qwen via transformers, or remote TEI endpoint.")
    parser.add_argument("--outer-chunk", default=DEFAULT_OUTER_CHUNK, type=int,
                        help="Records per writer chunk; bounds in-memory vector buffer.")
    parser.add_argument("--overwrite", action="store_true")
    # local backend
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None, help="e.g. 'cuda' or 'cpu' (local backend).")
    # cloud backend
    parser.add_argument("--base-url", default=None, help="Cloud endpoint base URL.")
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", default=4, type=int,
                        help="Cloud: concurrent HTTP requests in flight.")
    parser.add_argument("--batch-size", default=32, type=int,
                        help="Sentences per request (cloud) or per local forward pass.")
    parser.add_argument("--max-in-flight", default=None, type=int,
                        help="Cloud: cap on in-flight futures (default: =concurrency).")
    parser.add_argument("--normalize", action="store_true",
                        help="Cloud: ask the endpoint to L2-normalize embeddings.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = resolve_script_relative(args.input_dir).resolve()
    output_dir = resolve_script_relative(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder, closer = build_embedder(args)
    print(f"embed_sst: backend={args.backend} splits={args.splits} "
          f"input={input_dir} output={output_dir}")
    try:
        for split in args.splits:
            csv_path = input_dir / f"default_{split}.csv"
            out_path = output_dir / f"default_{split}.json"
            if not csv_path.exists():
                print(f"[{split}] SKIP: {csv_path} not found", file=sys.stderr)
                continue
            if out_path.exists() and not args.overwrite:
                print(f"[{split}] skip (already embedded): {out_path.name}")
                continue

            records = load_rows(csv_path)
            if not records:
                out_path.write_text("[]", encoding="utf-8")
                print(f"[{split}] empty, wrote []")
                continue

            started = time.time()
            count, dim = embed_split_streaming(records, embedder, out_path, args.outer_chunk)
            print(
                f"[{split}] {len(records)} sentences -> {count} records "
                f"(label+embedding, dim {dim}) in {time.time() - started:.1f}s -> {out_path.name}"
            )
    finally:
        if closer:
            closer()
    print(f"Done. SST embeddings written to {output_dir}")


if __name__ == "__main__":
    main()
