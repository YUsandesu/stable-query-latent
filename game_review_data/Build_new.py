"""Single-source Kaggle build for the game-review H5 pipeline.

This wrapper downloads one Kaggle dataset and then either:

1. uses prepared per-game CSVs directly, or
2. converts a raw table into per-game CSVs with ``prepare_kaggle_steam_reviews.py``.

After that it runs the existing metadata -> split -> text H5 -> embedding H5
pipeline, keeping the same sentence splitting and H5 layout as the original
build.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = "andrewmvd/steam-reviews"
DEFAULT_PREPARED_DIR = SCRIPT_DIR / "kaggle_prepared"
DEFAULT_WORKDIR = SCRIPT_DIR / "build_new_gamedata"
DEFAULT_KAGGLE_CACHE = SCRIPT_DIR / "kagglehub_cache"
STAGES = ("metadata", "split", "text-h5", "embed-h5")

PREPARED_STEM = re.compile(r"^\d+_\d+$")


def run_command(cmd: list[str], cwd: Path = SCRIPT_DIR) -> None:
    print("RUN " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def download_kaggle_dataset(dataset: str, cache_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required. Install it with:\n"
            f"  {sys.executable} -m pip install kagglehub"
        ) from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    old_cache = os.environ.get("KAGGLEHUB_CACHE")
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
    try:
        path = Path(kagglehub.dataset_download(dataset))
    finally:
        if old_cache is None:
            os.environ.pop("KAGGLEHUB_CACHE", None)
        else:
            os.environ["KAGGLEHUB_CACHE"] = old_cache

    print(f"Kaggle dataset path: {path}", flush=True)
    return path


def prepared_done(prepared_dir: Path) -> bool:
    return (
        (prepared_dir / "prepare_manifest.json").exists()
        and (prepared_dir / "games.json").exists()
        and any((prepared_dir / "reviews").glob("*.csv"))
    )


def looks_like_prepared_review_dir(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        return False
    if (path / "prepare_manifest.json").exists():
        return True
    matches = sum(1 for csv_path in csv_files if PREPARED_STEM.match(csv_path.stem))
    return matches == len(csv_files) or (len(csv_files) >= 2 and matches / len(csv_files) >= 0.8)


def find_prepared_review_dirs(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        path = Path(path)
        if path not in seen and looks_like_prepared_review_dir(path):
            seen.add(path)
            candidates.append(path)

    reviews_dir = root / "reviews"
    if reviews_dir.is_dir():
        add(reviews_dir)

    add(root)

    csv_dirs = sorted({csv_path.parent for csv_path in root.rglob("*.csv")}, key=lambda p: (len(p.parts), str(p)))
    for directory in csv_dirs:
        add(directory)

    return candidates


def find_games_json(root: Path) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    direct = root / "games.json"
    if direct.exists():
        return direct
    candidates = sorted(root.rglob("games.json"), key=lambda path: (len(path.parts), str(path)))
    return candidates[0] if candidates else None


def discover_raw_input(root: Path) -> Path:
    from prepare_kaggle_steam_reviews import discover_input

    return discover_input(root)


def prepare_raw_dataset(args, kaggle_input: Path) -> Path:
    prepared_dir = Path(args.prepared_dir)
    if prepared_done(prepared_dir) and not args.overwrite:
        print(f"prepare: existing prepared data at {prepared_dir}", flush=True)
        return prepared_dir

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_kaggle_steam_reviews.py"),
        "--input",
        str(kaggle_input),
        "--output-dir",
        str(prepared_dir),
        "--min-length",
        str(args.min_length),
        "--min-count",
        str(args.min_count),
        "--chunksize",
        str(args.prepare_chunksize),
    ]
    cmd.append("--strict-length" if args.strict_length else "--no-strict-length")
    cmd.append("--strict-count" if args.strict_count else "--no-strict-count")
    if args.overwrite:
        cmd.append("--overwrite")
    run_command(cmd)
    return prepared_dir


def optional_arg(cmd: list[str], name: str, value) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def split_data_parallel(args, metadata_dir: Path, sentences_dir: Path) -> None:
    workers = max(1, int(args.split_workers))
    if workers <= 1:
        from split_data import split_data

        split_data(
            input_dir=metadata_dir,
            output_dir=sentences_dir,
            model=args.split_model,
            device=args.split_device,
            chunk_budget=args.chunk_budget,
            overwrite=args.overwrite,
            batch_size=args.split_batch_size,
            outer_batch_size=args.split_outer_batch_size,
            prefetch_ram_target=args.split_prefetch_ram_target,
            prefetch_max_files=args.split_prefetch_max_files,
            prefetch_workers=args.split_prefetch_workers,
        )
        return

    print(f"split_data: launching {workers} shard worker processes", flush=True)
    processes = []
    for shard_index in range(workers):
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "split_data.py"),
            "--input-dir",
            str(metadata_dir),
            "--output-dir",
            str(sentences_dir),
            "--model",
            args.split_model,
            "--chunk-budget",
            str(args.chunk_budget),
            "--shard-count",
            str(workers),
            "--shard-index",
            str(shard_index),
        ]
        optional_arg(cmd, "--device", args.split_device)
        optional_arg(cmd, "--batch-size", args.split_batch_size)
        optional_arg(cmd, "--outer-batch-size", args.split_outer_batch_size)
        optional_arg(cmd, "--prefetch-ram-target", args.split_prefetch_ram_target)
        optional_arg(
            cmd,
            "--prefetch-max-files",
            1 if args.split_prefetch_max_files is None else args.split_prefetch_max_files,
        )
        optional_arg(
            cmd,
            "--prefetch-workers",
            1 if args.split_prefetch_workers is None else args.split_prefetch_workers,
        )
        if args.overwrite:
            cmd.append("--overwrite")
        print("RUN " + " ".join(cmd), flush=True)
        processes.append(subprocess.Popen(cmd, cwd=str(SCRIPT_DIR)))

    failures = []
    for shard_index, process in enumerate(processes):
        returncode = process.wait()
        if returncode:
            failures.append((shard_index, returncode))
    if failures:
        formatted = ", ".join(f"shard {idx} rc={rc}" for idx, rc in failures)
        raise RuntimeError(f"split shard worker(s) failed: {formatted}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--kaggle-cache", type=Path, default=DEFAULT_KAGGLE_CACHE)
    parser.add_argument(
        "--kaggle-input",
        type=Path,
        default=None,
        help="Existing Kaggle CSV/dir/file. If omitted, kagglehub downloads the dataset.",
    )
    parser.add_argument("--input-mode", choices=["auto", "prepared", "raw"], default="auto")
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")

    parser.add_argument("--prepare-chunksize", type=int, default=200_000)
    parser.add_argument("--strict-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-count", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)
    parser.add_argument("--metadata-workers", type=int, default=0,
                        help="Metadata CSV filter workers (0 -> up to 16 CPU cores, 1 -> single process).")
    parser.add_argument("--only", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[])
    parser.add_argument("--no-meta", action="store_true")

    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-budget", type=int, default=0)
    parser.add_argument("--split-batch-size", type=int, default=None)
    parser.add_argument("--split-outer-batch-size", type=int, default=None)
    parser.add_argument("--split-prefetch-ram-target", type=float, default=None)
    parser.add_argument("--split-prefetch-max-files", type=int, default=None)
    parser.add_argument("--split-prefetch-workers", type=int, default=None)
    parser.add_argument("--split-workers", type=int, default=1,
                        help="Parallel split shard processes (1 -> in-process).")

    parser.add_argument("--text-h5", type=Path, default=None)
    parser.add_argument("--embedding-h5", type=Path, default=None)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--text-chunk-rows", type=int, default=8192)
    parser.add_argument("--tag-mapping", type=Path, default=None)
    parser.add_argument("--no-tag-labels", action="store_true")

    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--embedding-chunk-rows", type=int, default=2048)
    parser.add_argument("--embedding-compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--gzip-level", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.prepared_dir = Path(args.prepared_dir)
    args.workdir = Path(args.workdir)
    if args.kaggle_input is not None:
        args.kaggle_input = Path(args.kaggle_input)
    if args.tag_mapping is None:
        from h5_corpus import DEFAULT_TAG_MAPPING

        args.tag_mapping = DEFAULT_TAG_MAPPING

    run = set(args.only) if args.only else set(STAGES)
    run -= set(args.skip)

    kaggle_root = args.kaggle_input
    if kaggle_root is None and not args.skip_download:
        kaggle_root = download_kaggle_dataset(args.dataset, args.kaggle_cache)
    elif kaggle_root is not None:
        kaggle_root = Path(kaggle_root)

    review_dirs: list[Path] = []
    games_json: Path | None = None
    meta_available = False

    if kaggle_root is not None and kaggle_root.exists():
        if args.input_mode == "raw":
            if args.skip_prepare:
                raise SystemExit(
                    "--input-mode raw requires preparation; drop --skip-prepare or point "
                    "--kaggle-input at already prepared per-game CSVs."
                )
            raw_input = discover_raw_input(kaggle_root)
            prepared_dir = prepare_raw_dataset(args, raw_input)
            review_dirs = [prepared_dir / "reviews"]
            games_json = prepared_dir / "games.json"
            meta_available = games_json.exists()
        else:
            review_dirs = find_prepared_review_dirs(kaggle_root)
            games_json = find_games_json(kaggle_root)
            meta_available = games_json is not None and games_json.exists()
            if args.input_mode == "prepared" and not review_dirs:
                raise SystemExit(
                    f"--input-mode prepared but no prepared review CSVs were found under {kaggle_root}"
                )
            if args.input_mode == "auto" and not review_dirs:
                if args.skip_prepare:
                    raise SystemExit(
                        "No prepared per-game CSVs were found under --kaggle-input, and "
                        "--skip-prepare prevents converting a raw Kaggle table."
                    )
                raw_input = discover_raw_input(kaggle_root)
                prepared_dir = prepare_raw_dataset(args, raw_input)
                review_dirs = [prepared_dir / "reviews"]
                games_json = prepared_dir / "games.json"
                meta_available = games_json.exists()
    elif args.skip_download:
        if prepared_done(args.prepared_dir):
            review_dirs = [args.prepared_dir / "reviews"]
            games_json = args.prepared_dir / "games.json"
            meta_available = games_json.exists()
        elif not args.skip_prepare:
            raise SystemExit(
                f"prepared data not found at {args.prepared_dir}. "
                "Run without --skip-prepare, or point --kaggle-input at a downloaded dataset."
            )
    else:
        raise SystemExit("No Kaggle input available. Use --kaggle-input or let the script download it.")

    if not review_dirs:
        raise SystemExit(
            "No review CSV directories found. The dataset may not contain prepared per-game CSVs."
        )

    games_json_for_pipeline = games_json or (args.workdir / "games.json")
    with_meta = meta_available and not args.no_meta

    metadata_dir = args.workdir / "metadata"
    sentences_dir = args.workdir / "sentences"
    text_h5 = args.text_h5 or (args.workdir / "text_h5.h5")
    embedding_h5 = args.embedding_h5 or (args.workdir / "embedding_h5.h5")

    print(
        f"=== Build_new ===\n"
        f"dataset     : {args.dataset}\n"
        f"input-mode  : {args.input_mode}\n"
        f"reviews-dirs: {[str(path) for path in review_dirs]}\n"
        f"games-json  : {games_json_for_pipeline if games_json else '(missing -> empty fallback)'}\n"
        f"workdir     : {args.workdir}\n"
        f"stages      : {sorted(run)}\n"
        f"backend     : {args.backend}",
        flush=True,
    )

    if "metadata" in run:
        print("\n--- stage 1/4: metadata (clean + filter + prepend descriptions) ---", flush=True)
        from build_metadata import build_metadata

        build_metadata(
            reviews_dirs=review_dirs,
            games_json=games_json_for_pipeline,
            output_dir=metadata_dir,
            min_length=args.min_length,
            min_count=args.min_count,
            with_meta=with_meta,
            overwrite=args.overwrite,
            workers=args.metadata_workers,
        )

    if "split" in run:
        print("\n--- stage 2/4: split (SaT sentence splitter) ---", flush=True)
        split_data_parallel(args, metadata_dir, sentences_dir)

    if "text-h5" in run:
        print("\n--- stage 3/4: text-h5 (unified text corpus) ---", flush=True)
        from h5_corpus import build_text_h5

        build_text_h5(
            sentences_dir=sentences_dir,
            games_json=games_json_for_pipeline,
            output_h5=text_h5,
            overwrite=args.overwrite,
            limit_files=args.limit_files,
            chunk_rows=args.text_chunk_rows,
            tag_mapping=args.tag_mapping,
            no_tag_labels=args.no_tag_labels,
            reviews_dirs=review_dirs,
            label_min_length=args.min_length,
        )

    if "embed-h5" in run:
        print("\n--- stage 4/4: embed-h5 (stream text H5 -> embedding H5) ---", flush=True)
        from embedding_data import embed_data

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

    print(
        f"\n=== build_new done. text={text_h5} embedding={embedding_h5} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
