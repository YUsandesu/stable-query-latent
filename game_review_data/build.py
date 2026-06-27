"""Unified build: download → merge → clean → split → embed → H5.

note.txt cleaning spec (applied to both sources jointly):
  cleaned   : keep reviews with length > --min-length (default 300)
  cleaned_1 : keep games with >= --min-count reviews remaining (default 500)
  cleaned_2 : all reviews for a game concatenated into one list
  cleaned_3 : game descriptions (detailed / about / short) prepended as first
              entries so the encoder always sees metadata context

Sources:

  source1  Steam Games Metadata and Player Reviews (2020–2024)/
           └─ Game Reviews/*.csv     (23 k games; appid_count.csv format)
           └─ games.json             (rich metadata: positive/negative/tags/…)
           Downloaded automatically from Mendeley (see note.txt).
           Use --skip-source1 to omit.

  source2  Kaggle andrewmvd/steam-reviews
           └─ reviews/*.csv          (656 games after filtering)
           └─ games.json             (enriched via Steam store API)
           Downloaded automatically via kagglehub unless --skip-source2 is set.
           Requires: pip install kagglehub  and  ~/.kaggle/kaggle.json credentials.

When the same appid appears in both sources, source1 wins.

Pipeline stages inside --workdir (default: combined_gamedata/):

  0. download  source2 via kagglehub + prepare_kaggle_steam_reviews.py (auto)
  1. games.json merge  → workdir/games.json         (source1 priority)
  2. metadata          → workdir/metadata/           (clean+filter+prepend)
  3. sentences         → workdir/sentences/          (SaT sentence split)
  4. embedded          → workdir/embedded/           (Qwen3 embed, local or cloud)
  5. h5 (optional)     → VICReg_review/h5/           (shard+merge into one HDF5)

Every stage is resumable: existing output files are skipped unless --overwrite.
Pass --only or --skip to run a subset of stages 2-4; use --build-h5 for stage 5.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# --------------------------------------------------------------------------- source paths
SOURCE1_DIR = SCRIPT_DIR / "Steam Games Metadata and Player Reviews (2020–2024"
SOURCE1_REVIEWS = SOURCE1_DIR / "Game Reviews"
SOURCE1_GAMES_JSON = SOURCE1_DIR / "games.json"
SOURCE1_DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/jxy85cr3th/download/2"
SOURCE1_ZIP_CACHE = SCRIPT_DIR / "mendeley_steam_reviews.zip"

KAGGLE_DATASET = "andrewmvd/steam-reviews"
DEFAULT_KAGGLE_CACHE = SCRIPT_DIR / "kagglehub_cache"
DEFAULT_PREPARED_DIR = SCRIPT_DIR / "kaggle_steam_reviews_prepared"
SOURCE2_REVIEWS = DEFAULT_PREPARED_DIR / "reviews"
SOURCE2_GAMES_JSON = DEFAULT_PREPARED_DIR / "games.json"

DEFAULT_WORKDIR = SCRIPT_DIR / "combined_gamedata"
H5_SCRIPT = PROJECT_ROOT / "VICReg_review" / "build_review_h5.py"

PIPELINE_STAGES = ("metadata", "split", "embed")


# --------------------------------------------------------------------------- helpers
def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def source1_done(source1_dir: Path) -> bool:
    reviews_dir = source1_dir / "Game Reviews"
    return (
        (source1_dir / "games.json").exists()
        and reviews_dir.exists()
        and any(reviews_dir.glob("*.csv"))
    )


def download_source1(source1_dir: Path, zip_cache: Path) -> bool:
    """Download and extract the Mendeley 2020-2024 Steam dataset.

    Returns True if source1 is ready afterwards.
    """
    if source1_done(source1_dir):
        print(f"source1: already present at {source1_dir}", flush=True)
        return True

    # Download zip if not cached
    if not zip_cache.exists():
        print(f"source1: downloading from Mendeley ...\n  {SOURCE1_DOWNLOAD_URL}", flush=True)
        zip_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = zip_cache.with_suffix(".zip.tmp")
        try:
            def _progress(block_count, block_size, total_size):
                if total_size > 0 and block_count % 500 == 0:
                    pct = min(100, block_count * block_size * 100 // total_size)
                    print(f"  ... {pct}%", flush=True)

            urllib.request.urlretrieve(SOURCE1_DOWNLOAD_URL, str(tmp_zip), _progress)
            tmp_zip.replace(zip_cache)
        except Exception as exc:
            tmp_zip.unlink(missing_ok=True)
            print(f"[warn] Mendeley download failed: {exc}\nsource1 will be skipped.", flush=True)
            return False
        print(f"source1: downloaded -> {zip_cache}", flush=True)
    else:
        print(f"source1: using cached zip {zip_cache}", flush=True)

    # Extract
    print(f"source1: extracting to {source1_dir.parent} ...", flush=True)
    try:
        with zipfile.ZipFile(zip_cache, "r") as zf:
            zf.extractall(source1_dir.parent)
    except Exception as exc:
        print(f"[warn] extraction failed: {exc}\nsource1 will be skipped.", flush=True)
        return False

    if not source1_done(source1_dir):
        print(
            f"[warn] extraction finished but expected layout not found under {source1_dir}.\n"
            "       The zip may have a different internal structure; inspect and pass "
            "--skip-source1-download with --source1-reviews pointing to the correct path.",
            flush=True,
        )
        return False

    print(f"source1: ready ({source1_dir})", flush=True)
    return True


def prepared_done(prepared_dir: Path) -> bool:
    return (
        (prepared_dir / "prepare_manifest.json").exists()
        and (prepared_dir / "games.json").exists()
        and any((prepared_dir / "reviews").glob("*.csv"))
    )


def download_and_prepare_kaggle(args) -> bool:
    """Download Kaggle dataset and run prepare_kaggle_steam_reviews.py.

    Returns True if source2 is ready to use afterwards.
    """
    prepared_dir = Path(args.source2_prepared_dir)

    if prepared_done(prepared_dir) and not args.overwrite:
        print(f"source2: prepared data already exists at {prepared_dir}", flush=True)
        return True

    # Download
    kaggle_input = getattr(args, "kaggle_input", None)
    if kaggle_input is None:
        try:
            import kagglehub
        except ImportError:
            print(
                "[warn] kagglehub not installed. Install with:\n"
                f"  {sys.executable} -m pip install kagglehub\n"
                "  Then set up ~/.kaggle/kaggle.json credentials.\n"
                "source2 will be skipped.",
                flush=True,
            )
            return False

        cache_dir = Path(args.kaggle_cache)
        old_cache = os.environ.get("KAGGLEHUB_CACHE")
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
        try:
            print(f"source2: downloading {KAGGLE_DATASET} ...", flush=True)
            kaggle_input = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        except Exception as exc:
            print(f"[warn] Kaggle download failed: {exc}\nsource2 will be skipped.", flush=True)
            return False
        finally:
            if old_cache is None:
                os.environ.pop("KAGGLEHUB_CACHE", None)
            else:
                os.environ["KAGGLEHUB_CACHE"] = old_cache
        print(f"source2: downloaded to {kaggle_input}", flush=True)

    # Prepare (filter into per-game CSVs + minimal games.json)
    cmd = [
        str(args.python),
        str(SCRIPT_DIR / "prepare_kaggle_steam_reviews.py"),
        "--input", str(kaggle_input),
        "--output-dir", str(prepared_dir),
        "--min-length", str(args.min_length),
        "--min-count", str(args.min_count),
        "--chunksize", str(args.prepare_chunksize),
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
    print("RUN " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)

    # Enrich with Steam store descriptions/tags
    if not args.skip_enrich:
        enrich_cmd = [
            str(args.python),
            str(SCRIPT_DIR / "enrich_steam_store_metadata.py"),
            "--games-json", str(prepared_dir / "games.json"),
            "--batch-size", str(args.enrich_batch_size),
            "--sleep", str(args.enrich_sleep),
            "--retry-sleep", str(args.enrich_retry_sleep),
            "--retries", str(args.enrich_retries),
        ]
        if args.overwrite:
            enrich_cmd.append("--overwrite-cache")
        print("RUN " + " ".join(str(c) for c in enrich_cmd), flush=True)
        subprocess.run(enrich_cmd, cwd=str(SCRIPT_DIR), check=True)

    return prepared_done(prepared_dir)


def merge_games_json(sources: list[Path], output_path: Path, overwrite: bool) -> dict:
    """Merge games.json files from multiple sources; first source wins per appid."""
    if output_path.exists() and not overwrite:
        print(f"merge_games_json: skip existing {output_path}", flush=True)
        return json.loads(output_path.read_text(encoding="utf-8"))

    merged: dict = {}
    for path in sources:
        path = Path(path)
        if not path.exists():
            print(f"  [warn] games.json not found: {path}", flush=True)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print(f"  [warn] {path}: not a JSON object, skipping", flush=True)
            continue
        added = sum(
            1 for appid, record in data.items()
            if str(appid) not in merged and not merged.update({str(appid): record})  # type: ignore[func-returns-value]
        )
        print(f"  {path}: {len(data)} records, {added} new", flush=True)

    atomic_json_write(merged, output_path)
    print(f"merge_games_json: {len(merged)} total appids -> {output_path}", flush=True)
    return merged


def run_h5_build(args, workdir: Path) -> None:
    cmd = [
        str(args.python),
        str(H5_SCRIPT),
        "--input-dir", str(workdir / "embedded"),
        "--games-json", str(workdir / "games.json"),
        "--workers", str(args.h5_workers),
        "--shards", str(args.h5_shards),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print("RUN " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


# --------------------------------------------------------------------------- main
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Source 1 (2020-2024, auto-downloaded from Mendeley)
    parser.add_argument("--source1-dir", type=Path, default=SOURCE1_DIR,
                        help="Root directory for the 2020-2024 dataset after extraction.")
    parser.add_argument("--source1-reviews", type=Path, default=SOURCE1_REVIEWS)
    parser.add_argument("--source1-games-json", type=Path, default=SOURCE1_GAMES_JSON)
    parser.add_argument("--source1-zip-cache", type=Path, default=SOURCE1_ZIP_CACHE,
                        help="Where to cache the Mendeley zip file.")
    parser.add_argument("--skip-source1", action="store_true",
                        help="Skip source1 entirely.")
    parser.add_argument("--skip-source1-download", action="store_true",
                        help="Skip downloading source1; use existing files if present.")

    # Source 2 (Kaggle, auto download)
    parser.add_argument("--skip-source2", action="store_true",
                        help="Skip source2 (Kaggle dataset) entirely.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download; use existing prepared data.")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip Steam store enrichment for source2.")
    parser.add_argument("--kaggle-input", type=Path, default=None,
                        help="Path to already-downloaded Kaggle CSV/dir (skips download).")
    parser.add_argument("--kaggle-cache", type=Path, default=DEFAULT_KAGGLE_CACHE)
    parser.add_argument("--source2-prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--source2-reviews", type=Path, default=SOURCE2_REVIEWS)
    parser.add_argument("--source2-games-json", type=Path, default=SOURCE2_GAMES_JSON)
    parser.add_argument("--prepare-chunksize", type=int, default=200_000)
    parser.add_argument("--strict-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-count", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enrich-batch-size", type=int, default=1)
    parser.add_argument("--enrich-sleep", type=float, default=2.0)
    parser.add_argument("--enrich-retry-sleep", type=float, default=10.0)
    parser.add_argument("--enrich-retries", type=int, default=5)

    # Output
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)

    # Stage control
    parser.add_argument("--only", nargs="+", choices=PIPELINE_STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=PIPELINE_STAGES, default=[])
    parser.add_argument("--skip-merge-games-json", action="store_true")
    parser.add_argument("--build-h5", action="store_true",
                        help="After embedding, build VICReg_review/h5/*.h5.")
    parser.add_argument("--overwrite", action="store_true")

    # Cleaning / filtering (note.txt spec)
    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)

    # Splitting
    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)

    # Embedding
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud",
                        help="'local': local Qwen3-Embedding-0.6B; 'cloud': HF TEI endpoint.")
    parser.add_argument("--local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true")

    # H5
    parser.add_argument("--h5-workers", type=int, default=2)
    parser.add_argument("--h5-shards", type=int, default=8)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))

    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    run = set(args.only) if args.only else set(PIPELINE_STAGES)
    run -= set(args.skip)

    # ------------------------------------------------------------------ source1
    review_dirs: list[Path] = []
    games_json_sources: list[Path] = []

    if not args.skip_source1:
        s1_ready = source1_done(args.source1_dir)
        if not s1_ready and not args.skip_source1_download:
            s1_ready = download_source1(args.source1_dir, args.source1_zip_cache)
        if s1_ready:
            s1 = args.source1_reviews
            if s1.exists() and any(s1.glob("*.csv")):
                review_dirs.append(s1)
                if args.source1_games_json.exists():
                    games_json_sources.append(args.source1_games_json)
        else:
            print(f"[warn] source1 not available, continuing without it.", flush=True)

    # ------------------------------------------------------------------ source2
    if not args.skip_source2:
        s2_ready = False
        if not args.skip_download:
            s2_ready = download_and_prepare_kaggle(args)
        elif prepared_done(args.source2_prepared_dir):
            s2_ready = True
        else:
            print(
                f"[warn] source2 prepared data not found at {args.source2_prepared_dir}. "
                "Run without --skip-download to fetch it.",
                flush=True,
            )

        if s2_ready:
            s2 = args.source2_reviews
            if s2.exists() and any(s2.glob("*.csv")):
                review_dirs.append(s2)
                if args.source2_games_json.exists():
                    games_json_sources.append(args.source2_games_json)

    if not review_dirs:
        raise SystemExit(
            "No review CSV directories found. Both sources failed to download.\n"
            f"  source1: {SOURCE1_DOWNLOAD_URL}\n"
            "  source2: install kagglehub + configure ~/.kaggle/kaggle.json, then re-run."
        )

    print(
        f"=== unified game-review build ===\n"
        f"workdir  : {workdir}\n"
        f"sources  : {[str(p) for p in review_dirs]}\n"
        f"stages   : {sorted(run)}\n"
        f"backend  : {args.backend}\n",
        flush=True,
    )

    # ------------------------------------------------------------------ games.json
    if not args.skip_merge_games_json:
        print("\n--- merge games.json ---", flush=True)
        merge_games_json(games_json_sources, workdir / "games.json", args.overwrite)

    games_json = workdir / "games.json"
    if not games_json.exists():
        raise SystemExit(
            f"games.json not found at {games_json}. "
            "Run without --skip-merge-games-json or supply it manually."
        )

    # ------------------------------------------------------------------ stage 1: metadata
    if "metadata" in run:
        print("\n--- stage 1/3: metadata (clean + filter + prepend descriptions) ---", flush=True)
        from build_metadata import build_metadata
        build_metadata(
            reviews_dirs=review_dirs,
            games_json=games_json,
            output_dir=workdir / "metadata",
            min_length=args.min_length,
            min_count=args.min_count,
            with_meta=True,
            overwrite=args.overwrite,
        )

    # ------------------------------------------------------------------ stage 2: split
    if "split" in run:
        print("\n--- stage 2/3: split (SaT sentence splitter) ---", flush=True)
        from split_data import split_data
        split_data(
            input_dir=workdir / "metadata",
            output_dir=workdir / "sentences",
            model=args.split_model,
            device=args.split_device,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )

    # ------------------------------------------------------------------ stage 3: embed
    if "embed" in run:
        print("\n--- stage 3/3: embed (Qwen3 vectors) ---", flush=True)
        from embedding_data import embed_data
        embed_data(
            input_dir=workdir / "sentences",
            output_dir=workdir / "embedded",
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

    # ------------------------------------------------------------------ H5 build
    if args.build_h5:
        print("\n--- H5: shard + merge -> VICReg_review/h5/ ---", flush=True)
        run_h5_build(args, workdir)

    elapsed = time.time() - started
    print(f"\n=== build done in {elapsed:.0f}s. workdir={workdir} ===", flush=True)


if __name__ == "__main__":
    main()
