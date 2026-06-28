"""Unified game-review build: download -> clean -> split -> text H5 -> embedding H5.

Sources:

  source1  Steam Games Metadata and Player Reviews (2020-2024)
           Game Reviews/*.csv + games.json, downloaded from Mendeley.

  source2  Kaggle najzeko/steam-reviews-2021
           prepared into reviews/*.csv + enriched games.json.

When the same appid appears in both sources, source1 wins. The clean stage keeps
long reviews, drops games below the review-count threshold, and prepends the
three Steam description fields. The split stage preserves review_id/sentence_id.

Durable corpus artifacts:

  <data-dir>/text_h5.h5       all sentence text + review/game metadata
  <data-dir>/embedding_h5.h5  same metadata + Qwen sentence vectors

No per-game embedded JSON, no NPZ corpus, and no separate conversion step.
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


def _download_stream(url: str, out_file: Path, total_hint: int = 0) -> None:
    """Single-connection sequential download (used as the fallback path)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total = total_hint or int(resp.headers.get("content-length") or 0)
        downloaded = 0
        chunk = 1 << 20
        with out_file.open("wb") as file:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                file.write(block)
                downloaded += len(block)
                if total > 0 and downloaded % (500 << 20) < chunk:
                    pct = downloaded * 100 // total
                    print(f"  ... {pct}% ({downloaded >> 20} MB)", flush=True)


def parallel_download(url: str, out_file: Path, workers: int = 8) -> None:
    """Download url into out_file using HTTP Range requests across `workers`
    connections, resuming from a prior interrupted attempt when possible.

    A single big file over one TCP connection is usually capped by per-stream
    throughput, not link bandwidth; several parallel ranges saturate the pipe.

    Resume: each worker owns a fixed byte range and records how many of its
    bytes have landed in a ``<out_file>.progress.json`` sidecar (flushed
    periodically). On a re-run the preallocated ``out_file`` is reopened in
    place (never truncated) and every worker continues from its recorded
    offset, so an interrupted download is never restarted from zero. Falls back
    to a single sequential stream when the server does not support byte ranges.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    out_file = Path(out_file)
    progress_path = out_file.with_name(out_file.name + ".progress.json")

    # Probe length + range support with a tiny ranged GET (HEAD is often blocked
    # by the S3 redirect Mendeley hands out).
    probe = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"}
    )
    total = 0
    ranges_ok = False
    try:
        with urllib.request.urlopen(probe) as resp:
            ranges_ok = resp.status == 206
            cr = resp.headers.get("content-range")  # "bytes 0-0/12345"
            if cr and "/" in cr:
                total = int(cr.rsplit("/", 1)[1])
    except Exception:
        ranges_ok = False

    if not ranges_ok or total <= 0 or workers <= 1:
        # No ranges -> cannot resume; always restart the single stream cleanly.
        print("  (range not supported, single-stream download)", flush=True)
        progress_path.unlink(missing_ok=True)
        _download_stream(url, out_file, total)
        return

    part = -(-total // workers)  # ceil division
    bounds = []
    for i in range(workers):
        start = i * part
        end = min(start + part, total) - 1
        if start <= end:
            bounds.append((start, end))

    # Per-worker bytes already on disk (index aligned with bounds).
    done_bytes = [0] * len(bounds)
    resumed = False
    if out_file.exists() and progress_path.exists():
        try:
            prev = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                int(prev.get("total", -1)) == total
                and len(prev.get("done", [])) == len(bounds)
                and out_file.stat().st_size == total
            ):
                done_bytes = [int(v) for v in prev["done"]]
                resumed = True
        except Exception:
            resumed = False

    if not resumed:
        # Fresh attempt: preallocate so each worker can seek+write independently.
        with out_file.open("wb") as f:
            f.truncate(total)
        done_bytes = [0] * len(bounds)

    already = sum(done_bytes)
    if resumed:
        print(
            f"  resuming download: {already >> 20}/{total >> 20} MB already on disk, "
            f"{len(bounds)} connections",
            flush=True,
        )
    else:
        print(f"  parallel download: {total >> 20} MB over {len(bounds)} connections", flush=True)

    done_total = [already]
    lock = threading.Lock()

    def save_progress():
        atomic_json_write({"total": total, "done": done_bytes}, progress_path)

    def fetch(index):
        start, end = bounds[index]
        resume_at = start + done_bytes[index]
        if resume_at > end:
            return  # this slice is already complete
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={resume_at}-{end}"},
        )
        with urllib.request.urlopen(req) as resp, open(out_file, "r+b") as f:
            f.seek(resume_at)
            chunk = 1 << 20
            since_save = 0
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                with lock:
                    done_bytes[index] += len(block)
                    done_total[0] += len(block)
                    since_save += len(block)
                    if done_total[0] % (500 << 20) < chunk:
                        print(
                            f"  ... {done_total[0] * 100 // total}% ({done_total[0] >> 20} MB)",
                            flush=True,
                        )
                    if since_save >= (100 << 20):  # checkpoint every ~100 MB
                        save_progress()
                        since_save = 0

    try:
        with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
            list(pool.map(fetch, range(len(bounds))))
    finally:
        # Persist whatever progress we have so an interruption can resume.
        save_progress()

    if sum(done_bytes) < total:
        raise IOError(
            f"download incomplete: {sum(done_bytes)}/{total} bytes; rerun to resume"
        )
    progress_path.unlink(missing_ok=True)


def parallel_extractall(zip_path: Path, dest: Path, workers: int = 0) -> None:
    """Extract every file member of zip_path into dest using a thread pool.

    zlib releases the GIL while decompressing, so threads give a near-linear
    speed-up over zipfile.extractall() for archives with many members (e.g. the
    inner Game Reviews.zip with its hundreds of per-game CSVs). ZipFile handles
    are not thread-safe, so each worker opens its own handle and processes an
    interleaved slice of the member list.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
    if not infos:
        return
    if workers <= 0:
        workers = min(16, os.cpu_count() or 4)
    workers = max(1, min(workers, len(infos)))

    # Pre-create all target directories so worker extract() calls never race.
    for parent in {(dest / info.filename).parent for info in infos}:
        parent.mkdir(parents=True, exist_ok=True)

    def worker(chunk):
        with zipfile.ZipFile(zip_path) as zf:
            for info in chunk:
                zf.extract(info, dest)

    chunks = [infos[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, chunks))

SOURCE1_DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/jxy85cr3th/download/2"
KAGGLE_DATASET = "najzeko/steam-reviews-2021"

DEFAULT_DATA_DIR = SCRIPT_DIR
DEFAULT_TEXT_H5 = DEFAULT_DATA_DIR / "text_h5.h5"
DEFAULT_EMBEDDING_H5 = DEFAULT_DATA_DIR / "embedding_h5.h5"
DEFAULT_TAP_MAPPING = PROJECT_ROOT / "VICReg_review" / "tags" / "tap_mapping.json"

PIPELINE_STAGES = ("metadata", "split", "text-h5", "embed-h5")


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
    if source1_done(source1_dir):
        print(f"source1: already present at {source1_dir}", flush=True)
        return True

    if not zip_cache.exists():
        print(f"source1: downloading from Mendeley ...\n  {SOURCE1_DOWNLOAD_URL}", flush=True)
        zip_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = zip_cache.with_suffix(".zip.tmp")
        try:
            parallel_download(SOURCE1_DOWNLOAD_URL, tmp_zip, workers=8)
            tmp_zip.replace(zip_cache)
        except Exception as exc:
            # Keep the partial file + its .progress.json so a re-run resumes
            # instead of restarting the multi-GB download from zero.
            print(
                f"[warn] Mendeley download incomplete: {exc}\n"
                f"       partial kept at {tmp_zip}; re-run to resume. source1 skipped for now.",
                flush=True,
            )
            return False
        print(f"source1: downloaded -> {zip_cache}", flush=True)
    else:
        print(f"source1: using cached zip {zip_cache}", flush=True)

    print(f"source1: extracting to {source1_dir.parent} ...", flush=True)
    try:
        parallel_extractall(zip_cache, source1_dir.parent)
    except Exception as exc:
        print(f"[warn] extraction failed: {exc}\nsource1 will be skipped.", flush=True)
        return False

    extracted = source1_dir
    if not extracted.is_dir():
        for candidate in source1_dir.parent.iterdir():
            if candidate.is_dir() and (candidate / "games.json").exists():
                extracted = candidate
                break

    if not extracted.is_dir():
        print(
            f"[warn] outer zip extracted but no folder with games.json found under {source1_dir.parent}.",
            flush=True,
        )
        return False

    inner_zip = extracted / "Game Reviews.zip"
    reviews_dir = extracted / "Game Reviews"
    if inner_zip.exists() and not (reviews_dir.exists() and any(reviews_dir.glob("*.csv"))):
        print(f"source1: extracting inner zip {inner_zip.name} ...", flush=True)
        try:
            parallel_extractall(inner_zip, extracted)
        except Exception as exc:
            print(f"[warn] inner zip extraction failed: {exc}\nsource1 will be skipped.", flush=True)
            return False

    if extracted != source1_dir:
        print(f"source1: renaming {extracted.name!r} -> {source1_dir.name!r}", flush=True)
        extracted.rename(source1_dir)

    if not source1_done(source1_dir):
        print(
            f"[warn] extraction finished but Game Reviews/*.csv still not found under {source1_dir}.",
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


def appids_in_reviews_dir(reviews_dir: Path) -> set[str]:
    appids: set[str] = set()
    reviews_dir = Path(reviews_dir)
    if not reviews_dir.exists():
        return appids
    for csv_path in reviews_dir.glob("*.csv"):
        appid = csv_path.stem.split("_", 1)[0]
        if appid:
            appids.add(appid)
    return appids


def report_source_overlap(review_dirs: list[Path]) -> None:
    if len(review_dirs) < 2:
        return

    seen: set[str] = set()
    unique_total = 0
    lines = []
    for index, reviews_dir in enumerate(review_dirs, start=1):
        appids = appids_in_reviews_dir(reviews_dir)
        overlap = seen & appids
        added = appids - seen
        unique_total += len(added)
        lines.append(
            f"  src{index}: files_appids={len(appids)} "
            f"new={len(added)} duplicate_lower_priority={len(overlap)} dir={reviews_dir}"
        )
        seen.update(appids)

    print(
        "source appid dedup priority: earlier sources win per appid "
        "(source1/Mendeley 2020-2024 before source2/Kaggle).\n"
        + "\n".join(lines)
        + f"\n  final_unique_appids={unique_total}",
        flush=True,
    )


def download_and_prepare_kaggle(args, source2_dir: Path, kaggle_cache: Path) -> bool:
    prepared_dir = source2_dir
    if prepared_done(prepared_dir) and not args.overwrite:
        print(f"source2: prepared data already exists at {prepared_dir}", flush=True)
        return True

    kaggle_input = getattr(args, "kaggle_input", None)
    if kaggle_input is None:
        try:
            import kagglehub
        except ImportError:
            print(
                "[warn] kagglehub not installed. Install with:\n"
                f"  {sys.executable} -m pip install kagglehub\n"
                "source2 will be skipped.",
                flush=True,
            )
            return False

        old_cache = os.environ.get("KAGGLEHUB_CACHE")
        os.environ["KAGGLEHUB_CACHE"] = str(kaggle_cache)
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

    cmd = [
        str(args.python),
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
        "--workers",
        str(args.prepare_workers),
    ]
    cmd.append("--strict-length" if args.strict_length else "--no-strict-length")
    cmd.append("--strict-count" if args.strict_count else "--no-strict-count")
    if args.overwrite:
        cmd.append("--overwrite")
    print("RUN " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)

    if not args.skip_enrich:
        enrich_cmd = [
            str(args.python),
            str(SCRIPT_DIR / "enrich_steam_store_metadata.py"),
            "--games-json",
            str(prepared_dir / "games.json"),
            "--batch-size",
            str(args.enrich_batch_size),
            "--sleep",
            str(args.enrich_sleep),
            "--retry-sleep",
            str(args.enrich_retry_sleep),
            "--retries",
            str(args.enrich_retries),
            "--cache-dir",
            str(args.enrich_cache_dir),
        ]
        if args.overwrite:
            enrich_cmd.append("--overwrite-cache")
        print("RUN " + " ".join(str(c) for c in enrich_cmd), flush=True)
        subprocess.run(enrich_cmd, cwd=str(SCRIPT_DIR), check=True)

    return prepared_done(prepared_dir)


def merge_games_json(sources: list[Path], output_path: Path, overwrite: bool) -> dict:
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
        added = 0
        for appid, record in data.items():
            key = str(appid)
            if key not in merged:
                merged[key] = record
                added += 1
        print(f"  {path}: {len(data)} records, {added} new", flush=True)

    atomic_json_write(merged, output_path)
    print(f"merge_games_json: {len(merged)} total appids -> {output_path}", flush=True)
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Root directory for downloads and outputs: games.json, metadata/, "
            "sentences/, text_h5.h5, embedding_h5.h5."
        ),
    )

    parser.add_argument("--skip-source1", action="store_true")
    parser.add_argument("--skip-source1-download", action="store_true")
    parser.add_argument("--skip-source2", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download and use existing prepared source2 data.")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--kaggle-input", type=Path, default=None)

    parser.add_argument("--prepare-chunksize", type=int, default=200_000)
    parser.add_argument("--prepare-workers", type=int, default=0,
                        help="Kaggle prepare filter workers (0 -> all CPU cores, 1 -> single process).")
    parser.add_argument("--strict-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-count", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enrich-batch-size", type=int, default=1)
    parser.add_argument(
        "--enrich-cache-dir",
        type=Path,
        default=SCRIPT_DIR / "_steam_appdetails_cache",
        help="Persistent Steam appdetails cache, kept OUTSIDE --data-dir so "
             "wiping the data dir does not force re-fetching from the API.",
    )
    parser.add_argument("--enrich-sleep", type=float, default=2.0)
    parser.add_argument("--enrich-retry-sleep", type=float, default=10.0)
    parser.add_argument("--enrich-retries", type=int, default=5)

    parser.add_argument("--only", nargs="+", choices=PIPELINE_STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=PIPELINE_STAGES, default=[])
    parser.add_argument("--skip-merge-games-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)

    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)

    parser.add_argument("--text-h5", type=Path, default=None)
    parser.add_argument("--embedding-h5", type=Path, default=None)
    parser.add_argument("--limit-files", type=int, default=0,
                        help="Debug limit for sentence JSON files when building text H5.")
    parser.add_argument("--text-chunk-rows", type=int, default=8192)
    parser.add_argument("--tap-mapping", type=Path, default=DEFAULT_TAP_MAPPING)
    parser.add_argument("--no-tap-labels", action="store_true")

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
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def resolve_args_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.enrich_cache_dir = Path(args.enrich_cache_dir).expanduser().resolve()
    if args.kaggle_input is not None:
        args.kaggle_input = Path(args.kaggle_input).expanduser().resolve()
    if args.text_h5 is not None:
        args.text_h5 = Path(args.text_h5).expanduser().resolve()
    if args.embedding_h5 is not None:
        args.embedding_h5 = Path(args.embedding_h5).expanduser().resolve()
    if args.tap_mapping is not None:
        args.tap_mapping = Path(args.tap_mapping).expanduser().resolve()
    if args.token_file is not None:
        args.token_file = str(Path(args.token_file).expanduser().resolve())
    args.python = Path(args.python).expanduser().resolve()
    return args


def main():
    args = resolve_args_paths(parse_args())
    started = time.time()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    source1_dir = data_dir / "Steam Games Metadata and Player Reviews (2020–2024"
    source1_zip = data_dir / "mendeley_steam_reviews.zip"
    source2_dir = data_dir / "kaggle_steam_reviews_prepared"
    kaggle_cache = data_dir / "kagglehub_cache"
    metadata_dir = data_dir / "metadata"
    sentences_dir = data_dir / "sentences"
    games_json_path = data_dir / "games.json"
    text_h5 = args.text_h5 or (data_dir / "text_h5.h5")
    embedding_h5 = args.embedding_h5 or (data_dir / "embedding_h5.h5")

    run = set(args.only) if args.only else set(PIPELINE_STAGES)
    run -= set(args.skip)

    review_dirs: list[Path] = []
    games_json_sources: list[Path] = []

    if not args.skip_source1:
        s1_ready = source1_done(source1_dir)
        if not s1_ready and not args.skip_source1_download:
            s1_ready = download_source1(source1_dir, source1_zip)
        if s1_ready:
            s1_reviews = source1_dir / "Game Reviews"
            if s1_reviews.exists() and any(s1_reviews.glob("*.csv")):
                review_dirs.append(s1_reviews)
            s1_games_json = source1_dir / "games.json"
            if s1_games_json.exists():
                games_json_sources.append(s1_games_json)
        else:
            print("[warn] source1 not available, continuing without it.", flush=True)

    if not args.skip_source2:
        s2_ready = False
        if not args.skip_download:
            s2_ready = download_and_prepare_kaggle(args, source2_dir, kaggle_cache)
        elif prepared_done(source2_dir):
            s2_ready = True
        else:
            print(
                f"[warn] source2 prepared data not found at {source2_dir}. "
                "Run without --skip-download to fetch it.",
                flush=True,
            )

        if s2_ready:
            s2_reviews = source2_dir / "reviews"
            if s2_reviews.exists() and any(s2_reviews.glob("*.csv")):
                review_dirs.append(s2_reviews)
            s2_games_json = source2_dir / "games.json"
            if s2_games_json.exists():
                games_json_sources.append(s2_games_json)

    if not review_dirs:
        raise SystemExit(
            "No review CSV directories found. Both sources failed or were skipped.\n"
            f"  source1: {SOURCE1_DOWNLOAD_URL}\n"
            "  source2: install kagglehub + configure Kaggle credentials, then re-run."
        )

    report_source_overlap(review_dirs)

    print(
        f"=== unified game-review build ===\n"
        f"data-dir    : {data_dir}\n"
        f"text-h5     : {text_h5}\n"
        f"embedding-h5: {embedding_h5}\n"
        f"sources     : {[str(p) for p in review_dirs]}\n"
        f"stages      : {sorted(run)}\n"
        f"backend     : {args.backend}\n",
        flush=True,
    )

    if not args.skip_merge_games_json:
        print("\n--- merge games.json ---", flush=True)
        merge_games_json(games_json_sources, games_json_path, args.overwrite)

    if not games_json_path.exists():
        raise SystemExit(
            f"games.json not found at {games_json_path}. "
            "Run without --skip-merge-games-json or supply it manually."
        )

    if "metadata" in run:
        print("\n--- stage 1/4: metadata (clean + filter + prepend descriptions) ---", flush=True)
        from build_metadata import build_metadata

        build_metadata(
            reviews_dirs=review_dirs,
            games_json=games_json_path,
            output_dir=metadata_dir,
            min_length=args.min_length,
            min_count=args.min_count,
            with_meta=True,
            overwrite=args.overwrite,
        )

    if "split" in run:
        print("\n--- stage 2/4: split (SaT sentence splitter) ---", flush=True)
        from split_data import split_data

        split_data(
            input_dir=metadata_dir,
            output_dir=sentences_dir,
            model=args.split_model,
            device=args.split_device,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )

    if "text-h5" in run:
        print("\n--- stage 3/4: text-h5 (unified text corpus) ---", flush=True)
        from h5_corpus import build_text_h5

        build_text_h5(
            sentences_dir=sentences_dir,
            games_json=games_json_path,
            output_h5=text_h5,
            overwrite=args.overwrite,
            limit_files=args.limit_files,
            chunk_rows=args.text_chunk_rows,
            tap_mapping=args.tap_mapping,
            no_tap_labels=args.no_tap_labels,
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

    elapsed = time.time() - started
    print(
        f"\n=== build done in {elapsed:.0f}s ===\n"
        f"text-h5     : {text_h5}\n"
        f"embedding-h5: {embedding_h5}",
        flush=True,
    )


if __name__ == "__main__":
    main()
