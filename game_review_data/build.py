"""Orchestrator for the game-review embedding pipeline. Runs the three stages in
sequence inside a single working directory (handy for experiments):

    <workdir>/metadata    build_metadata.py : raw review CSVs -> per-game JSON arrays
    <workdir>/sentences   split_data.py     : -> nested review_id/sentence_id text
    <workdir>/embedded    embedding_data.py : -> same nested structure + vectors

Each stage can be skipped with --skip so you can re-run just part of the pipeline.
The embedding backend (local Qwen vs cloud TEI) is selected with --backend.

Example (experiment on a subset, clean step only):
    python build_gamedata.py --workdir experiments/run1 --reviews-dir <dir> --only metadata
"""

import argparse
from pathlib import Path

from build_metadata import DEFAULT_GAMES_JSON, DEFAULT_REVIEWS_DIR, build_metadata
from split_data import split_data
from embedding_data import embed_data

STAGES = ("metadata", "split", "embed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="gamedata", type=Path,
                        help="Directory holding the metadata/sentences/embedded subdirs.")
    parser.add_argument("--only", nargs="+", choices=STAGES, default=None,
                        help="Run only these stages (default: all).")
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[],
                        help="Skip these stages.")
    parser.add_argument("--overwrite", action="store_true")

    # stage 1: metadata
    parser.add_argument("--reviews-dir", default=None, help="Raw review CSV dir (build_metadata default if unset).")
    parser.add_argument("--games-json", default=None)
    parser.add_argument("--min-length", default=300, type=int)
    parser.add_argument("--min-count", default=500, type=int)
    parser.add_argument("--no-meta", dest="with_meta", action="store_false")

    # stage 2: split
    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-size", default=2000, type=int)

    # stage 3: embed
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run = set(args.only) if args.only else set(STAGES)
    run -= set(args.skip)

    workdir = args.workdir
    metadata_dir = workdir / "metadata"
    sentences_dir = workdir / "sentences"
    embedded_dir = workdir / "embedded"

    print(f"=== pipeline workdir={workdir} | stages={sorted(run)} | backend={args.backend} ===")

    if "metadata" in run:
        print("\n--- stage 1/3: build_metadata ---")
        build_metadata(
            reviews_dir=args.reviews_dir or DEFAULT_REVIEWS_DIR,
            games_json=args.games_json or DEFAULT_GAMES_JSON,
            output_dir=metadata_dir,
            min_length=args.min_length,
            min_count=args.min_count,
            with_meta=args.with_meta,
            overwrite=args.overwrite,
        )

    if "split" in run:
        print("\n--- stage 2/3: split_data ---")
        split_data(
            input_dir=metadata_dir,
            output_dir=sentences_dir,
            model=args.split_model,
            device=args.split_device,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )

    if "embed" in run:
        print("\n--- stage 3/3: embedding_data ---")
        embed_data(
            input_dir=sentences_dir,
            output_dir=embedded_dir,
            backend=args.backend,
            overwrite=args.overwrite,
            local_model=args.local_model,
            device=args.embed_device,
            base_url=args.base_url,
            token_file=args.token_file,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )

    print(f"\n=== pipeline done. outputs under {workdir} ===")


if __name__ == "__main__":
    main()
