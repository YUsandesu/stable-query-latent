"""Build the Kaggle Steam Reviews branch of the game-review dataset.

This is the second source pipeline used by the top-level ``build.py``:

1. download Kaggle ``andrewmvd/steam-reviews`` when needed;
2. filter it into per-game CSVs with ``prepare_kaggle_steam_reviews.py``;
3. enrich ``games.json`` with Steam store-page descriptions/tags;
4. run ``build_1.py`` on the prepared per-game CSVs to produce text/embedding H5.

The script is resumable by default. Existing prepared CSVs, metadata JSONs,
sentence JSONs, text H5, and embedding H5 are skipped unless ``--overwrite`` is
passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET = "andrewmvd/steam-reviews"
DEFAULT_PREPARED_DIR = SCRIPT_DIR / "kaggle_steam_reviews_prepared"
DEFAULT_KAGGLE_GAMES_JSON = SCRIPT_DIR / "kaggle_storepage_data" / "games.json"
DEFAULT_WORKDIR = SCRIPT_DIR / "build_2_gamedata"
DEFAULT_KAGGLE_CACHE = SCRIPT_DIR / "kagglehub_cache"
STAGES = ("metadata", "split", "text-h5", "embed-h5")


def run_command(cmd: list[str], cwd: Path = SCRIPT_DIR, env: dict[str, str] | None = None) -> None:
    print("RUN " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def download_kaggle_dataset(dataset: str, cache_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required for build_2.py. Install it with:\n"
            f"  {sys.executable} -m pip install kagglehub"
        ) from exc

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


def build_pipeline(args) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_1.py"),
        "--workdir",
        str(args.workdir),
        "--reviews-dir",
        str(args.prepared_dir / "reviews"),
        "--games-json",
        str(args.prepared_dir / "games.json"),
        "--min-length",
        str(args.min_length),
        "--min-count",
        str(args.min_count),
        "--split-model",
        args.split_model,
        "--chunk-budget",
        str(args.chunk_budget),
        "--backend",
        args.backend,
        "--local-model",
        args.local_model,
        "--concurrency",
        str(args.concurrency),
        "--batch-size",
        str(args.batch_size),
        "--read-batch-size",
        str(args.read_batch_size),
        "--embedding-dtype",
        args.embedding_dtype,
        "--embedding-compression",
        args.embedding_compression,
    ]
    if args.only:
        cmd.extend(["--only", *args.only])
    if args.skip:
        cmd.extend(["--skip", *args.skip])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_meta:
        cmd.append("--no-meta")
    if args.split_device:
        cmd.extend(["--split-device", args.split_device])
    if args.embed_device:
        cmd.extend(["--embed-device", args.embed_device])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.token_file:
        cmd.extend(["--token-file", args.token_file])
    if args.normalize:
        cmd.append("--normalize")
    if args.max_in_flight:
        cmd.extend(["--max-in-flight", str(args.max_in_flight)])
    if args.no_tag_labels:
        cmd.append("--no-tag-labels")
    run_command(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--kaggle-cache", type=Path, default=DEFAULT_KAGGLE_CACHE)
    parser.add_argument("--kaggle-input", type=Path, default=None,
                        help="Existing Kaggle CSV/dir. If omitted, kagglehub downloads the dataset.")
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--kaggle-games-json", type=Path, default=DEFAULT_KAGGLE_GAMES_JSON)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")

    # prepare_kaggle_steam_reviews.py
    parser.add_argument("--prepare-chunksize", type=int, default=200_000)
    parser.add_argument("--strict-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-count", action=argparse.BooleanOptionalAction, default=True)

    # shared filtering / build_1.py
    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)
    parser.add_argument("--no-meta", action="store_true")
    parser.add_argument("--only", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[])

    # enrich_steam_store_metadata.py
    parser.add_argument("--enrich-batch-size", type=int, default=1)
    parser.add_argument("--enrich-sleep", type=float, default=2.0)
    parser.add_argument("--enrich-retry-sleep", type=float, default=10.0)
    parser.add_argument("--enrich-retries", type=int, default=5)

    # split/embed H5
    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-budget", type=int, default=0)
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud")
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
    parser.add_argument("--embedding-compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--no-tag-labels", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.prepared_dir = Path(args.prepared_dir)
    args.kaggle_games_json = Path(args.kaggle_games_json)
    args.workdir = Path(args.workdir)

    kaggle_input = args.kaggle_input
    if not args.skip_download and kaggle_input is None:
        kaggle_input = download_kaggle_dataset(args.dataset, args.kaggle_cache)

    if not args.skip_prepare:
        if prepared_done(args.prepared_dir) and not args.overwrite:
            print(f"skip prepare: existing prepared data at {args.prepared_dir}", flush=True)
        else:
            if kaggle_input is None:
                raise SystemExit("--kaggle-input is required when --skip-download is used and prepared data is absent.")
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "prepare_kaggle_steam_reviews.py"),
                "--input",
                str(kaggle_input),
                "--output-dir",
                str(args.prepared_dir),
                "--min-length",
                str(args.min_length),
                "--min-count",
                str(args.min_count),
                "--chunksize",
                str(args.prepare_chunksize),
            ]
            if args.strict_length:
                cmd.append("--strict-length")
            else:
                cmd.append("--no-strict-length")
            if args.strict_count:
                cmd.append("--strict-count")
            else:
                cmd.append("--no-strict-count")
            if args.overwrite:
                cmd.append("--overwrite")
            run_command(cmd)

    if not args.skip_enrich:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "enrich_steam_store_metadata.py"),
            "--games-json",
            str(args.prepared_dir / "games.json"),
            "--output-json",
            str(args.kaggle_games_json),
            "--batch-size",
            str(args.enrich_batch_size),
            "--sleep",
            str(args.enrich_sleep),
            "--retry-sleep",
            str(args.enrich_retry_sleep),
            "--retries",
            str(args.enrich_retries),
        ]
        if args.overwrite:
            cmd.append("--overwrite-cache")
        run_command(cmd)

    if not args.skip_pipeline:
        build_pipeline(args)

    print(f"\n=== build_2 done. prepared={args.prepared_dir} workdir={args.workdir} ===", flush=True)


if __name__ == "__main__":
    main()
