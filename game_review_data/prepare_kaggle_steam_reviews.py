"""Prepare Kaggle's ``andrewmvd/steam-reviews`` table for the local pipeline.

The downstream pipeline consumes one CSV per game plus a generated ``games.json``
metadata file:

    <prepared>/reviews/<appid>_<kept_review_count>.csv
    <prepared>/games.json

This script converts Kaggle's large, usually single-file review table into that
shape while preserving the project filters:

* keep only reviews with text length > 300 by default;
* keep only games with more than 500 kept reviews by default;
* preserve the recommendation label for the downstream positive-rate probe;
* create a minimal ``games.json`` from the Kaggle table, then let the optional
  enrichment stage fill Steam store-page descriptions/tags.

The top-level ``game_review_data/build.py`` runs this stage automatically and
uses the generated prepared ``games.json`` as an intermediate output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "kaggle_steam_reviews_prepared"

POSITIVE_VALUE = "Recommended"
NEGATIVE_VALUE = "Not Recommended"
CSV_FIELDS = ("user", "playtime", "post_date", "helpfulness", "review", "recommend", "early_access_review")

APPID_COLUMNS = (
    "app_id",
    "appid",
    "app id",
    "game_id",
    "game id",
    "id",
)
NAME_COLUMNS = (
    "app_name",
    "app name",
    "game_name",
    "game name",
    "name",
    "title",
)
REVIEW_COLUMNS = (
    "review_text",
    "review text",
    "review",
    "text",
    "content",
)
RECOMMEND_COLUMNS = (
    "recommend",
    "recommended",
    "voted_up",
    "review_score",
    "review score",
    "sentiment",
    "score",
)
HELPFUL_COLUMNS = (
    "review_votes",
    "review votes",
    "votes",
    "helpful",
    "helpfulness",
)
USER_COLUMNS = ("user", "author", "username", "steamid", "steam_id")
PLAYTIME_COLUMNS = ("playtime", "hours", "playtime_forever", "author_playtime_forever")
DATE_COLUMNS = ("post_date", "date", "timestamp_created", "created_at", "posted")


def normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def infer_column(columns: Iterable[str], explicit: str | None, candidates: Iterable[str], label: str,
                 required: bool = True) -> str | None:
    if explicit:
        if explicit not in columns:
            raise KeyError(f"--{label}-column={explicit!r} is not present. Columns: {list(columns)}")
        return explicit
    by_norm = {normalize_column(column): column for column in columns}
    for candidate in candidates:
        hit = by_norm.get(normalize_column(candidate))
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"Could not infer {label} column. Columns: {list(columns)}")
    return None


def discover_input(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    patterns = ("*.csv", "*.csv.gz", "*.tsv", "*.zip", "*.parquet")
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(path.rglob(pattern)))
    if not matches:
        raise ValueError(f"No supported Kaggle table file found under {path}")
    # Prefer the largest file; Kaggle datasets often include small metadata files.
    return max(matches, key=lambda item: item.stat().st_size)


def read_head(path: Path, encoding: str | None) -> pd.DataFrame:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".parquet" in suffixes:
        return pd.read_parquet(path).head(8)
    kwargs = {"nrows": 8}
    if encoding:
        kwargs["encoding"] = encoding
    if path.suffix.lower() == ".tsv":
        kwargs["sep"] = "\t"
    return pd.read_csv(path, **kwargs)


def read_chunks(path: Path, chunksize: int, encoding: str | None):
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if ".parquet" in suffixes:
        yield pd.read_parquet(path)
        return
    kwargs = {"chunksize": chunksize, "low_memory": False}
    if encoding:
        kwargs["encoding"] = encoding
    if path.suffix.lower() == ".tsv":
        kwargs["sep"] = "\t"
    yield from pd.read_csv(path, **kwargs)


def clean_appid(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def is_kept_length(text: str, min_length: int, strict: bool) -> bool:
    length = len(text)
    return length > min_length if strict else length >= min_length


def keep_count(count: int, min_count: int, strict: bool) -> bool:
    return count > min_count if strict else count >= min_count


def to_recommend(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return POSITIVE_VALUE if value else NEGATIVE_VALUE
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    positive = {"1", "1.0", "true", "yes", "y", "positive", "pos", "recommended", "recommend"}
    negative = {"0", "0.0", "-1", "-1.0", "false", "no", "n", "negative", "neg", "not recommended", "not_recommended"}
    if lowered in positive:
        return POSITIVE_VALUE
    if lowered in negative:
        return NEGATIVE_VALUE
    try:
        return POSITIVE_VALUE if float(lowered) > 0 else NEGATIVE_VALUE
    except ValueError:
        return ""


def safe_filename_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._") or "unknown"


def replace_with_retry(src: Path, dst: Path, retries: int = 20, delay: float = 0.25) -> None:
    """Windows may briefly keep a just-closed CSV handle busy; retry the rename."""
    for attempt in range(retries):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt + 1 >= retries:
                raise
            time.sleep(delay)


class LruCsvWriters:
    def __init__(self, reviews_dir: Path, counts: dict[str, int], max_open: int):
        self.reviews_dir = reviews_dir
        self.counts = counts
        self.max_open = max_open
        self.open: OrderedDict[str, tuple[object, csv.DictWriter, Path]] = OrderedDict()
        self.tmp_paths: dict[str, Path] = {}

    def _path_for(self, appid: str) -> Path:
        stem = f"{safe_filename_part(appid)}_{self.counts[appid]}"
        return self.reviews_dir / f"{stem}.csv.tmp"

    def writer_for(self, appid: str) -> csv.DictWriter:
        if appid in self.open:
            handle, writer, path = self.open.pop(appid)
            self.open[appid] = (handle, writer, path)
            return writer
        if len(self.open) >= self.max_open:
            _, (old_handle, _, _) = self.open.popitem(last=False)
            old_handle.close()
        tmp_path = self._path_for(appid)
        is_new = not tmp_path.exists()
        handle = tmp_path.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        self.open[appid] = (handle, writer, tmp_path)
        self.tmp_paths[appid] = tmp_path
        return writer

    def close_all(self) -> None:
        while self.open:
            _, (handle, _, _) = self.open.popitem(last=False)
            handle.close()

    def finalize(self) -> None:
        self.close_all()
        for tmp_path in self.tmp_paths.values():
            final_path = tmp_path.with_suffix("")
            replace_with_retry(tmp_path, final_path)


def infer_schema(args, input_path: Path) -> dict[str, str | None]:
    head = read_head(input_path, args.encoding)
    columns = list(head.columns)
    try:
        schema = {
            "appid": infer_column(columns, args.appid_column, APPID_COLUMNS, "appid"),
            "name": infer_column(columns, args.name_column, NAME_COLUMNS, "name", required=False),
            "review": infer_column(columns, args.review_column, REVIEW_COLUMNS, "review"),
            "recommend": infer_column(columns, args.recommend_column, RECOMMEND_COLUMNS, "recommend", required=False),
            "helpfulness": infer_column(columns, args.helpful_column, HELPFUL_COLUMNS, "helpful", required=False),
            "user": infer_column(columns, args.user_column, USER_COLUMNS, "user", required=False),
            "playtime": infer_column(columns, args.playtime_column, PLAYTIME_COLUMNS, "playtime", required=False),
            "post_date": infer_column(columns, args.date_column, DATE_COLUMNS, "date", required=False),
        }
    except KeyError as exc:
        raise SystemExit(
            f"{exc}\n"
            f"Input table: {input_path}\n"
            "This pipeline needs raw review text columns such as 'review_text', "
            "'review', 'text', or 'content', plus an app id column. Datasets that "
            "only contain Steam recommendation metadata cannot be converted into "
            "the text embedding corpus. Use a raw text dataset such as "
            "andrewmvd/steam-reviews, or pass an already prepared reviews/ directory."
        ) from exc
    print("inferred schema:", json.dumps(schema, ensure_ascii=False, indent=2), flush=True)
    return schema


def iter_filtered_rows(chunk: pd.DataFrame, schema: dict[str, str | None], args):
    if len(chunk) == 0:
        return

    # Clean + length-filter the review text for the WHOLE chunk at once. This is
    # the only work that touches every row, so doing it vectorized in pandas/C
    # (instead of a per-row Python loop) is the main speed-up. The whitespace
    # collapse + strip mirrors clean_text() exactly.
    reviews = (
        chunk[schema["review"]]
        .astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    lengths = reviews.str.len().to_numpy()
    keep = lengths > args.min_length if args.strict_length else lengths >= args.min_length
    survivors = np.flatnonzero(keep)
    if survivors.size == 0:
        return

    # Only the surviving rows get per-row scalar cleaning, and we index by
    # position into numpy arrays (no slow per-row .loc lookups).
    reviews_arr = reviews.to_numpy()
    appids_arr = chunk[schema["appid"]].to_numpy()

    def column(key):
        name = schema.get(key)
        return chunk[name].to_numpy() if name else None

    names_arr = column("name")
    recommends_arr = column("recommend")
    helpful_arr = column("helpfulness")
    users_arr = column("user")
    playtimes_arr = column("playtime")
    dates_arr = column("post_date")

    for i in survivors:
        appid = clean_appid(appids_arr[i])
        if not appid:
            continue
        yield {
            "appid": appid,
            "name": clean_text(names_arr[i]) if names_arr is not None else "",
            "review": reviews_arr[i],
            "recommend": to_recommend(recommends_arr[i]) if recommends_arr is not None else "",
            "helpfulness": clean_text(helpful_arr[i]) if helpful_arr is not None else "",
            "user": clean_text(users_arr[i]) if users_arr is not None else "",
            "playtime": clean_text(playtimes_arr[i]) if playtimes_arr is not None else "",
            "post_date": clean_text(dates_arr[i]) if dates_arr is not None else "",
        }


def _filter_chunk_records(chunk: pd.DataFrame, schema: dict[str, str | None],
                          min_length: int, strict_length: bool) -> list[dict]:
    """Process-pool worker: filter one chunk, return surviving row dicts.

    Lives at module scope so it is picklable. The expensive CSV parse already
    happened in the reader process; here each worker runs the (still single-core)
    vectorized clean + filter on its own chunk, so N workers give ~N-way speed-up
    on the part that was pinning one core.
    """
    import types
    args = types.SimpleNamespace(min_length=min_length, strict_length=strict_length)
    return list(iter_filtered_rows(chunk, schema, args))


def iter_chunk_records(input_path: Path, schema: dict[str, str | None], args):
    """Yield (chunk_index, n_rows, rows) for every chunk, filtered.

    With ``args.workers > 1`` the per-chunk filtering runs on a process pool while
    the main process keeps reading ahead, bounded to ~2x workers chunks in flight
    so memory stays flat on multi-GB inputs. Results are yielded in input order so
    downstream counting/writing is deterministic. Falls back to in-process work
    when workers <= 1.
    """
    import collections

    workers = int(getattr(args, "workers", 1) or 1)
    chunks = read_chunks(input_path, args.chunksize, args.encoding)

    if workers <= 1:
        for chunk_index, chunk in enumerate(chunks, start=1):
            yield chunk_index, len(chunk), list(iter_filtered_rows(chunk, schema, args))
        return

    from concurrent.futures import ProcessPoolExecutor

    min_length, strict = args.min_length, args.strict_length
    # Keep only a few chunks beyond the worker count in flight: enough to keep
    # the pool fed, small enough that buffered chunks don't blow up memory.
    max_inflight = workers + 2
    chunk_index = 0
    pending: collections.deque = collections.deque()
    exhausted = False

    def submit_next(executor):
        nonlocal chunk_index, exhausted
        try:
            chunk = next(chunks)
        except StopIteration:
            exhausted = True
            return
        chunk_index += 1
        pending.append((
            chunk_index,
            len(chunk),
            executor.submit(_filter_chunk_records, chunk, schema, min_length, strict),
        ))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        while not exhausted and len(pending) < max_inflight:
            submit_next(executor)
        while pending:
            idx, n_rows, future = pending.popleft()
            rows = future.result()
            yield idx, n_rows, rows
            if not exhausted:
                submit_next(executor)


def count_games(input_path: Path, schema: dict[str, str | None], args) -> tuple[dict[str, int], dict[str, str], int]:
    counts: defaultdict[str, int] = defaultdict(int)
    names: dict[str, str] = {}
    total_kept_reviews = 0
    for chunk_index, n_rows, rows in iter_chunk_records(input_path, schema, args):
        chunk_kept = 0
        for row in rows:
            appid = row["appid"]
            counts[appid] += 1
            chunk_kept += 1
            if row["name"] and not names.get(appid):
                names[appid] = row["name"]
        total_kept_reviews += chunk_kept
        print(
            f"pass1 chunk={chunk_index} rows={n_rows} kept_reviews={chunk_kept} "
            f"candidate_games={len(counts)}",
            flush=True,
        )
    return dict(counts), names, total_kept_reviews


def write_reviews(input_path: Path, schema: dict[str, str | None], keep_appids: set[str],
                  counts: dict[str, int], args) -> int:
    reviews_dir = args.output_dir / "reviews"
    writers = LruCsvWriters(reviews_dir, counts, max_open=args.max_open_files)
    written = 0
    try:
        for chunk_index, n_rows, rows in iter_chunk_records(input_path, schema, args):
            chunk_written = 0
            for row in rows:
                appid = row["appid"]
                if appid not in keep_appids:
                    continue
                writers.writer_for(appid).writerow({
                    "user": row["user"],
                    "playtime": row["playtime"],
                    "post_date": row["post_date"],
                    "helpfulness": row["helpfulness"],
                    "review": row["review"],
                    "recommend": row["recommend"],
                    "early_access_review": "",
                })
                written += 1
                chunk_written += 1
            print(f"pass2 chunk={chunk_index} rows={n_rows} written_reviews={chunk_written}", flush=True)
        writers.finalize()
    except BaseException:
        writers.close_all()
        raise
    return written


def write_games_json(output_path: Path, keep_appids: set[str], counts: dict[str, int],
                     names: dict[str, str], base_games: dict) -> dict:
    output = {}
    missing_metadata = []
    for appid in sorted(keep_appids, key=lambda value: int(value) if value.isdigit() else value):
        record = dict(base_games.get(appid) or {})
        if not record:
            missing_metadata.append(appid)
        if names.get(appid) and not record.get("name"):
            record["name"] = names[appid]
        record.setdefault("name", appid)
        record.setdefault("detailed_description", "")
        record.setdefault("about_the_game", "")
        record.setdefault("short_description", "")
        record.setdefault("tags", {})
        record["kaggle_kept_reviews"] = counts[appid]
        output[appid] = record

    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return {"games": len(output), "missing_metadata": missing_metadata}


def parse_prepared_filename(path: Path) -> tuple[str, int]:
    stem = path.stem
    if stem.endswith(".csv"):
        stem = stem[:-4]
    appid, _, count_text = stem.rpartition("_")
    if not appid or not count_text.isdigit():
        raise ValueError(f"Cannot parse prepared filename: {path.name}")
    return appid, int(count_text)


def finalize_existing_outputs(args) -> None:
    reviews_dir = args.output_dir / "reviews"
    if not reviews_dir.exists():
        raise FileNotFoundError(reviews_dir)
    tmp_files = sorted(reviews_dir.glob("*.csv.tmp"))
    for tmp_path in tmp_files:
        final_path = tmp_path.with_suffix("")
        replace_with_retry(tmp_path, final_path)

    csv_files = sorted(reviews_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No prepared CSV files found in {reviews_dir}")

    counts = {}
    for path in csv_files:
        appid, count = parse_prepared_filename(path)
        counts[appid] = count
    keep_appids = set(counts)
    games_info = write_games_json(
        args.output_dir / "games.json",
        keep_appids,
        counts,
        names={},
        base_games={},
    )
    manifest = {
        "status": "finalized_existing",
        "output_dir": str(args.output_dir.resolve()),
        "min_length": args.min_length,
        "strict_length": args.strict_length,
        "min_count": args.min_count,
        "strict_count": args.strict_count,
        "games_after_count_filter": len(keep_appids),
        "written_reviews": int(sum(counts.values())),
        "missing_metadata_appids": games_info["missing_metadata"],
    }
    tmp_manifest = args.output_dir / "prepare_manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(tmp_manifest, args.output_dir / "prepare_manifest.json")
    print(
        f"finalized existing prepared reviews: games={len(keep_appids)} "
        f"reviews={sum(counts.values())} tmp_finalized={len(tmp_files)} -> {args.output_dir}",
        flush=True,
    )


def maybe_prepare_output(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir = args.output_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    existing = list(reviews_dir.glob("*.csv")) + list(reviews_dir.glob("*.csv.tmp"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{reviews_dir} already contains CSV outputs. Pass --overwrite or choose a new --output-dir."
        )
    if args.overwrite:
        for path in existing:
            path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="Kaggle CSV/TSV/ZIP/PARQUET file, or a directory containing one.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)
    parser.add_argument("--strict-length", action=argparse.BooleanOptionalAction, default=True,
                        help="Default keeps len(review) > min_length; disable for >=.")
    parser.add_argument("--strict-count", action=argparse.BooleanOptionalAction, default=True,
                        help="Default keeps games with count > min_count; disable for >=.")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process-pool workers for chunk filtering (0 -> all CPU cores, 1 -> in-process).",
    )
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--max-open-files", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Finalize an interrupted output-dir without re-reading the Kaggle source table.")
    parser.add_argument("--appid-column", default=None)
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--review-column", default=None)
    parser.add_argument("--recommend-column", default=None)
    parser.add_argument("--helpful-column", default=None)
    parser.add_argument("--user-column", default=None)
    parser.add_argument("--playtime-column", default=None)
    parser.add_argument("--date-column", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    if args.workers <= 0:
        import os
        # The single-threaded CSV reader in the main process is the throughput
        # ceiling, so a modest pool is plenty; capping avoids hundreds of forked
        # workers (and their in-flight chunk buffers) on high-core cloud boxes.
        args.workers = min(os.cpu_count() or 1, 16)
    print(f"prepare: using {args.workers} filter worker(s)", flush=True)
    if args.finalize_existing:
        finalize_existing_outputs(args)
        return
    if args.input is None:
        raise SystemExit("--input is required unless --finalize-existing is used.")
    input_path = discover_input(args.input)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            print(f"zip members: {zf.namelist()[:20]}", flush=True)
    maybe_prepare_output(args)

    schema = infer_schema(args, input_path)
    counts, names, total_kept_reviews = count_games(input_path, schema, args)
    keep_appids = {
        appid for appid, count in counts.items()
        if keep_count(count, args.min_count, args.strict_count)
    }
    kept_counts = {appid: counts[appid] for appid in keep_appids}
    print(
        f"filter summary: kept_reviews_after_length={total_kept_reviews} "
        f"games_before_count={len(counts)} games_after_count={len(keep_appids)}",
        flush=True,
    )

    written = write_reviews(input_path, schema, keep_appids, kept_counts, args)
    games_info = write_games_json(
        args.output_dir / "games.json",
        keep_appids,
        kept_counts,
        names,
        {},
    )
    manifest = {
        "input": str(input_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "schema": schema,
        "min_length": args.min_length,
        "strict_length": args.strict_length,
        "min_count": args.min_count,
        "strict_count": args.strict_count,
        "total_kept_reviews_after_length": total_kept_reviews,
        "games_before_count_filter": len(counts),
        "games_after_count_filter": len(keep_appids),
        "written_reviews": written,
        "missing_metadata_appids": games_info["missing_metadata"],
    }
    (args.output_dir / "prepare_manifest.json.tmp").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "prepare_manifest.json.tmp").replace(args.output_dir / "prepare_manifest.json")
    print(
        f"prepared Kaggle reviews: games={len(keep_appids)} reviews={written} -> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
