"""Small-workdir orchestrator for the unified game-review H5 pipeline.

Outputs under ``--workdir``:

    metadata/         cleaned per-game review arrays
    sentences/        split review_id/sentence_id text JSON
    text_h5.h5        unified text + metadata corpus
    embedding_h5.h5   unified text + metadata + vectors corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_metadata import DEFAULT_GAMES_JSON, DEFAULT_REVIEWS_DIR, build_metadata
from embedding_data import DEFAULT_LOCAL_MODEL, embed_data
from h5_corpus import DEFAULT_TAG_MAPPING, build_text_h5
from split_data import split_data

SCRIPT_DIR = Path(__file__).resolve().parent
STAGES = ("metadata", "split", "text-h5", "embed-h5")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=SCRIPT_DIR / "gamedata", type=Path)
    parser.add_argument("--only", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--reviews-dir", default=None)
    parser.add_argument("--games-json", default=None)
    parser.add_argument("--min-length", default=300, type=int)
    parser.add_argument("--min-count", default=500, type=int)
    parser.add_argument("--no-meta", dest="with_meta", action="store_false")

    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-budget", default=0, type=int)

    parser.add_argument("--text-h5", type=Path, default=None)
    parser.add_argument("--embedding-h5", type=Path, default=None)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--text-chunk-rows", type=int, default=8192)
    parser.add_argument("--tag-mapping", type=Path, default=DEFAULT_TAG_MAPPING)
    parser.add_argument("--no-tag-labels", action="store_true")

    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-in-flight", default=None, type=int)
    parser.add_argument("--read-batch-size", default=4096, type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--embedding-chunk-rows", default=2048, type=int)
    parser.add_argument("--embedding-compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--gzip-level", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    run = set(args.only) if args.only else set(STAGES)
    run -= set(args.skip)

    workdir = args.workdir
    metadata_dir = workdir / "metadata"
    sentences_dir = workdir / "sentences"
    text_h5 = args.text_h5 or (workdir / "text_h5.h5")
    embedding_h5 = args.embedding_h5 or (workdir / "embedding_h5.h5")
    games_json = Path(args.games_json or DEFAULT_GAMES_JSON)
    reviews_dir = Path(args.reviews_dir or DEFAULT_REVIEWS_DIR)

    print(
        f"=== pipeline workdir={workdir} | stages={sorted(run)} | backend={args.backend} ===\n"
        f"text-h5={text_h5}\n"
        f"embedding-h5={embedding_h5}"
    )

    if "metadata" in run:
        print("\n--- stage 1/4: build_metadata ---")
        build_metadata(
            reviews_dir=reviews_dir,
            games_json=games_json,
            output_dir=metadata_dir,
            min_length=args.min_length,
            min_count=args.min_count,
            with_meta=args.with_meta,
            overwrite=args.overwrite,
        )

    if "split" in run:
        print("\n--- stage 2/4: split_data ---")
        split_data(
            input_dir=metadata_dir,
            output_dir=sentences_dir,
            model=args.split_model,
            device=args.split_device,
            chunk_budget=args.chunk_budget,
            overwrite=args.overwrite,
        )

    if "text-h5" in run:
        print("\n--- stage 3/4: build_text_h5 ---")
        build_text_h5(
            sentences_dir=sentences_dir,
            games_json=games_json,
            output_h5=text_h5,
            overwrite=args.overwrite,
            limit_files=args.limit_files,
            chunk_rows=args.text_chunk_rows,
            tag_mapping=args.tag_mapping,
            no_tag_labels=args.no_tag_labels,
            reviews_dirs=[reviews_dir],
            label_min_length=args.min_length,
        )

    if "embed-h5" in run:
        print("\n--- stage 4/4: embedding_data ---")
        embed_data(
            input_h5=text_h5,
            output_h5=embedding_h5,
            backend=args.backend,
            overwrite=args.overwrite,
            local_model=args.local_model,
            device=args.embed_device,
            base_url=args.base_url,
            token_file=args.token_file,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_in_flight=args.max_in_flight,
            read_batch_size=args.read_batch_size,
            normalize=args.normalize,
            dtype=args.embedding_dtype,
            chunk_rows=args.embedding_chunk_rows,
            compression=args.embedding_compression,
            gzip_level=args.gzip_level,
        )

    print(f"\n=== pipeline done. text={text_h5} embedding={embedding_h5} ===")


if __name__ == "__main__":
    main()
