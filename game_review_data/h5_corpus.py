"""Unified H5 corpus helpers for the Steam game-review pipeline.

The pipeline intentionally has only two durable corpus artifacts after sentence
splitting:

    text_h5.h5       sentence text + review/game metadata
    embedding_h5.h5  the same metadata plus sentence vectors

This keeps new builds out of the old per-game embedded JSON / NPZ formats while
preserving the offset layout expected by ``train_vicreg_review_h5.py``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SENTENCES_DIR = SCRIPT_DIR / "sentences"
DEFAULT_GAMES_JSON = SCRIPT_DIR / "games.json"
DEFAULT_TEXT_H5 = SCRIPT_DIR / "text_h5.h5"
DEFAULT_EMBEDDING_H5 = SCRIPT_DIR / "embedding_h5.h5"
DEFAULT_TAG_MAPPING = PROJECT_ROOT / "VICReg_review" / "tags" / "tag_mapping.json"
TEXT_H5_SCHEMA = "game_review_text_h5.v1"
EMBEDDING_H5_SCHEMA = "game_review_embedding_h5.v1"
POSITIVE_VALUE = "recommended"
NEGATIVE_VALUE = "not recommended"


def atomic_h5_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def unlink_with_retry(path: Path, attempts: int = 120, delay: float = 1.0) -> None:
    path = Path(path)
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def replace_with_retry(source: Path, target: Path, attempts: int = 120, delay: float = 1.0) -> None:
    source = Path(source)
    target = Path(target)
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def best_effort_unlink(path: Path) -> None:
    try:
        unlink_with_retry(path, attempts=5, delay=0.2)
    except PermissionError:
        pass


def atomic_json_write(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        replace_with_retry(tmp_path, path)
    except BaseException:
        best_effort_unlink(tmp_path)
        raise


def text_h5_manifest_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".manifest.json")


def string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def decode_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def numeric_suffix(value, prefix: str):
    text = str(value)
    if text.startswith(prefix):
        text = text[len(prefix):]
    try:
        return int(text)
    except ValueError:
        return text


def ordered_sentence_items(sentence_map: dict) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for sentence_id, payload in sorted(
        sentence_map.items(),
        key=lambda kv: numeric_suffix(kv[0], "sentence_"),
    ):
        if not isinstance(payload, dict):
            continue
        text = payload.get("sentence_text")
        if text is None:
            continue
        text = str(text).strip()
        if text:
            items.append((str(sentence_id), text))
    return items


def iter_ordered_reviews(raw: dict) -> Iterable[tuple[str, list[tuple[str, str]]]]:
    for review_id, sentence_map in sorted(raw.items(), key=lambda kv: numeric_suffix(kv[0], "")):
        if not isinstance(sentence_map, dict):
            continue
        sentences = ordered_sentence_items(sentence_map)
        if sentences:
            yield str(review_id), sentences


def load_sentence_mapping(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a review_id -> sentence mapping.")
    return raw


def text_worker_count(workers: int, units: int) -> int:
    if units <= 0:
        return 1
    if workers is None or int(workers) <= 0:
        return max(1, min(os.cpu_count() or 4, units, 16))
    return max(1, min(int(workers), units))


def _scan_sentence_file(task: tuple[int, str]) -> dict | None:
    index, path_text = task
    path = Path(path_text)
    raw = load_sentence_mapping(path)
    review_lengths: list[int] = []
    for _, sentences in iter_ordered_reviews(raw):
        review_lengths.append(len(sentences))
    sentence_count = int(sum(review_lengths))
    if sentence_count == 0:
        return {
            "index": index,
            "path": str(path),
            "game_name": path.stem,
            "reviews": 0,
            "sentences": 0,
            "empty": True,
        }
    return {
        "index": index,
        "path": str(path),
        "game_name": path.stem,
        "reviews": len(review_lengths),
        "sentences": sentence_count,
        "empty": False,
    }


def scan_sentence_files(
    sentences_dir: Path,
    limit_files: int = 0,
    workers: int = 1,
) -> tuple[list[dict], int, int]:
    files = sorted(Path(sentences_dir).glob("*.json"))
    if limit_files > 0:
        files = files[:limit_files]
    if not files:
        raise ValueError(f"No sentence JSON files found in {sentences_dir}")

    plans: list[dict] = []
    total_reviews = 0
    total_sentences = 0
    worker_count = text_worker_count(workers, len(files))
    if worker_count <= 1:
        iterator = (_scan_sentence_file((index, str(path))) for index, path in enumerate(files, start=1))
        for result in iterator:
            if result is None:
                continue
            index = int(result["index"])
            path = Path(str(result["path"]))
            if result.get("empty"):
                print(f"[scan {index}/{len(files)}] {path.name}: skip (no sentences)", flush=True)
            else:
                result["path"] = path
                plans.append(result)
                total_reviews += int(result["reviews"])
                total_sentences += int(result["sentences"])
            if index % 100 == 0 or index == len(files):
                print(
                    f"[scan {index}/{len(files)}] games={len(plans)} "
                    f"reviews={total_reviews} sentences={total_sentences}",
                    flush=True,
                )
    else:
        print(f"scan_sentence_files: using {worker_count} worker processes", flush=True)
        results: list[dict] = []
        completed = 0
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(_scan_sentence_file, (index, str(path)))
                for index, path in enumerate(files, start=1)
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                completed += 1
                if completed % 100 == 0 or completed == len(files):
                    non_empty = [item for item in results if not item.get("empty")]
                    print(
                        f"[scan {completed}/{len(files)}] scanned, "
                        f"non_empty={len(non_empty)}",
                        flush=True,
                    )

        for result in sorted(results, key=lambda item: int(item["index"])):
            path = Path(str(result["path"]))
            if result.get("empty"):
                print(f"[scan {result['index']}/{len(files)}] {path.name}: skip (no sentences)", flush=True)
                continue
            result["path"] = path
            plans.append(result)
            total_reviews += int(result["reviews"])
            total_sentences += int(result["sentences"])
        print(
            f"[scan done] games={len(plans)} reviews={total_reviews} "
            f"sentences={total_sentences}",
            flush=True,
        )

    if not plans:
        raise ValueError(f"No non-empty sentence mappings found in {sentences_dir}")
    return plans, total_reviews, total_sentences


def text_h5_expected_counts(plans: list[dict], total_reviews: int, total_sentences: int) -> dict:
    return {
        "schema": TEXT_H5_SCHEMA,
        "games": int(len(plans)),
        "reviews": int(total_reviews),
        "sentences": int(total_sentences),
        "source_sentence_files": [str(Path(plan["path"]).resolve()) for plan in plans],
    }


def read_text_h5_counts(path: Path) -> dict:
    """Read the cheap count summary from a text H5 without loading text rows."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        counts = {
            "schema": decode_text(h5.attrs.get("schema", "")),
            "games": int(h5.attrs.get("games", -1)),
            "reviews": int(h5.attrs.get("reviews", -1)),
            "sentences": int(h5.attrs.get("sentences", -1)),
            "datasets": {key: tuple(int(v) for v in h5[key].shape) for key in h5.keys()},
        }
        if "source_sentence_files" in h5:
            counts["source_sentence_files"] = [
                decode_text(value) for value in h5["source_sentence_files"][:]
            ]
        return counts


def validate_text_h5(path: Path, expected: dict) -> tuple[bool, str, dict]:
    """Return whether ``path`` is a complete text H5 matching expected counts."""
    path = Path(path)
    if not path.exists():
        return False, "missing file", {}

    try:
        actual = read_text_h5_counts(path)
    except Exception as exc:
        return False, f"cannot read H5: {exc}", {}

    for key in ("schema", "games", "reviews", "sentences"):
        if actual.get(key) != expected.get(key):
            return False, f"{key} mismatch: actual={actual.get(key)!r} expected={expected.get(key)!r}", actual

    games = int(expected["games"])
    reviews = int(expected["reviews"])
    sentences = int(expected["sentences"])
    required_shapes = {
        "texts": (sentences,),
        "sentence_ids": (sentences,),
        "review_ids": (reviews,),
        "review_offsets": (reviews + 1,),
        "game_review_offsets": (games + 1,),
        "game_names": (games,),
        "appids": (games,),
        "game_titles": (games,),
        "tags_json": (games,),
        "positive": (games,),
        "negative": (games,),
        "positive_rate": (games,),
        "recommendation_label_source": (games,),
        "source_sentence_files": (games,),
    }
    datasets = actual.get("datasets", {})
    for name, shape in required_shapes.items():
        if datasets.get(name) != shape:
            return False, f"dataset {name!r} shape mismatch: actual={datasets.get(name)} expected={shape}", actual

    try:
        with h5py.File(path, "r") as h5:
            if int(h5["review_offsets"][0]) != 0:
                return False, "review_offsets[0] is not 0", actual
            if int(h5["review_offsets"][-1]) != sentences:
                return False, "review_offsets[-1] does not equal sentence count", actual
            if int(h5["game_review_offsets"][0]) != 0:
                return False, "game_review_offsets[0] is not 0", actual
            if int(h5["game_review_offsets"][-1]) != reviews:
                return False, "game_review_offsets[-1] does not equal review count", actual
    except Exception as exc:
        return False, f"offset validation failed: {exc}", actual

    return True, "counts and required datasets match", actual


def write_text_h5_manifest(
    output_h5: Path,
    expected: dict,
    sentences_dir: Path,
    games_json: Path,
    reviews_dirs: list[Path],
    label_min_length: int,
    limit_files: int,
    status: str,
) -> None:
    manifest = {
        "schema": TEXT_H5_SCHEMA,
        "text_h5": str(Path(output_h5).resolve()),
        "status": status,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sentences_dir": str(Path(sentences_dir).resolve()),
        "games_json": str(Path(games_json).resolve()),
        "review_dirs": [str(Path(path).resolve()) for path in reviews_dirs],
        "recommendation_label_min_length": int(label_min_length),
        "limit_files": int(limit_files),
        "games": int(expected["games"]),
        "reviews": int(expected["reviews"]),
        "sentences": int(expected["sentences"]),
        "source_sentence_file_count": len(expected.get("source_sentence_files", [])),
        "source_sentence_files": expected.get("source_sentence_files", []),
    }
    atomic_json_write(manifest, text_h5_manifest_path(output_h5))


def load_games_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object keyed by appid.")
    return {str(key): value for key, value in data.items()}


def parse_number(value, *, as_int: bool = False):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if as_int else float(number)


def review_csv_path(reviews_dirs: list[Path], game_name: str) -> Path:
    appid = str(game_name).split("_", 1)[0]
    for reviews_dir in reviews_dirs:
        exact = reviews_dir / f"{game_name}.csv"
        if exact.exists():
            return exact
        matches = sorted(reviews_dir.glob(f"{appid}_*.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No review CSV found for {game_name}")


def count_recommendations(csv_path: Path, label_min_length: int = 0) -> tuple[int, int]:
    positive = 0
    negative = 0
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "recommend" not in (reader.fieldnames or []):
            raise KeyError(f"{csv_path} has no 'recommend' column")
        for row in reader:
            if label_min_length > 0 and len(row.get("review") or "") < label_min_length:
                continue
            value = (row.get("recommend") or "").strip().lower()
            if value == POSITIVE_VALUE:
                positive += 1
            elif value == NEGATIVE_VALUE:
                negative += 1
    return positive, negative


def game_metadata(
    game_name: str,
    games: dict,
    reviews_dirs: list[Path] | None = None,
    label_min_length: int = 0,
) -> dict:
    appid = str(game_name).split("_", 1)[0]
    record = games.get(appid) or {}
    if not isinstance(record, dict):
        record = {}

    positive = parse_number(record.get("positive"), as_int=True)
    negative = parse_number(record.get("negative"), as_int=True)
    positive_rate = parse_number(record.get("positive_rate"))
    if positive_rate is None and positive is not None and negative is not None and positive + negative > 0:
        positive_rate = positive / (positive + negative)
    label_source = "games_json"
    if reviews_dirs:
        try:
            csv_positive, csv_negative = count_recommendations(
                review_csv_path(reviews_dirs, game_name),
                label_min_length=label_min_length,
            )
            if csv_positive + csv_negative > 0:
                positive = csv_positive
                negative = csv_negative
                positive_rate = positive / (positive + negative)
                label_source = "review_csv"
        except Exception:
            label_source = "games_json"

    tags = record.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}
    return {
        "appid": appid,
        "title": str(record.get("name") or record.get("title") or appid),
        "positive": -1 if positive is None else positive,
        "negative": -1 if negative is None else negative,
        "positive_rate": np.nan if positive_rate is None else positive_rate,
        "tags_json": json.dumps(tags, ensure_ascii=False, sort_keys=True),
        "label_source": label_source,
    }


def validate_text_shard(path: Path, plan: dict) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as h5:
            return (
                decode_text(h5.attrs.get("schema", "")) == TEXT_H5_SCHEMA
                and decode_text(h5.attrs.get("source_sentence_file", "")) == str(Path(plan["path"]).resolve())
                and decode_text(h5.attrs.get("game_name", "")) == str(plan["game_name"])
                and int(h5.attrs.get("reviews", -1)) == int(plan["reviews"])
                and int(h5.attrs.get("sentences", -1)) == int(plan["sentences"])
                and tuple(h5["texts"].shape) == (int(plan["sentences"]),)
                and tuple(h5["sentence_ids"].shape) == (int(plan["sentences"]),)
                and tuple(h5["review_ids"].shape) == (int(plan["reviews"]),)
                and tuple(h5["review_offsets"].shape) == (int(plan["reviews"]) + 1,)
                and int(h5["review_offsets"][0]) == 0
                and int(h5["review_offsets"][-1]) == int(plan["sentences"])
            )
    except Exception:
        return False


def _write_text_shard(task: dict) -> dict:
    plan = dict(task["plan"])
    path = Path(plan["path"])
    shard_path = Path(task["shard_path"])
    games = task["games"]
    review_dirs = [Path(value) for value in task["review_dirs"]]
    label_min_length = int(task["label_min_length"])
    chunk_rows = max(1, int(task["chunk_rows"]))
    overwrite = bool(task["overwrite"])

    if not overwrite and validate_text_shard(shard_path, plan):
        return {
            "index": int(plan["index"]),
            "path": str(path),
            "shard_path": str(shard_path),
            "reviews": int(plan["reviews"]),
            "sentences": int(plan["sentences"]),
            "reused": True,
        }

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = atomic_h5_path(shard_path)
    unlink_with_retry(tmp_path)
    meta = game_metadata(
        str(plan["game_name"]),
        games,
        reviews_dirs=review_dirs,
        label_min_length=label_min_length,
    )
    raw = load_sentence_mapping(path)
    review_count = int(plan["reviews"])
    sentence_count = int(plan["sentences"])
    sentence_chunk = max(1, min(chunk_rows, sentence_count))
    review_chunk = max(1, min(chunk_rows, review_count))

    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset(
                "texts", shape=(sentence_count,), dtype=string_dtype(),
                chunks=(sentence_chunk,),
            )
            h5.create_dataset(
                "sentence_ids", shape=(sentence_count,), dtype=string_dtype(),
                chunks=(sentence_chunk,),
            )
            h5.create_dataset(
                "review_ids", shape=(review_count,), dtype=string_dtype(),
                chunks=(review_chunk,),
            )
            h5.create_dataset("review_offsets", shape=(review_count + 1,), dtype=np.int64)
            h5["review_offsets"][0] = 0

            sentence_cursor = 0
            review_cursor = 0
            for review_id, sentences in iter_ordered_reviews(raw):
                h5["review_ids"][review_cursor] = review_id
                local_sentence_ids = [sentence_id for sentence_id, _ in sentences]
                local_texts = [text for _, text in sentences]
                count = len(local_texts)
                end = sentence_cursor + count
                h5["texts"][sentence_cursor:end] = local_texts
                h5["sentence_ids"][sentence_cursor:end] = local_sentence_ids
                sentence_cursor = end
                review_cursor += 1
                h5["review_offsets"][review_cursor] = sentence_cursor

            if review_cursor != review_count or sentence_cursor != sentence_count:
                raise RuntimeError(
                    f"{path.name}: shard count mismatch "
                    f"reviews {review_cursor}/{review_count}, "
                    f"sentences {sentence_cursor}/{sentence_count}"
                )

            h5.attrs["schema"] = TEXT_H5_SCHEMA
            h5.attrs["source_sentence_file"] = str(path.resolve())
            h5.attrs["game_name"] = str(plan["game_name"])
            h5.attrs["reviews"] = review_count
            h5.attrs["sentences"] = sentence_count
            for key, value in meta.items():
                h5.attrs[f"meta_{key}"] = value

        replace_with_retry(tmp_path, shard_path)
    except BaseException:
        best_effort_unlink(tmp_path)
        raise

    return {
        "index": int(plan["index"]),
        "path": str(path),
        "shard_path": str(shard_path),
        "reviews": review_count,
        "sentences": sentence_count,
        "reused": False,
    }


def write_text_shards(
    plans: list[dict],
    shard_dir: Path,
    games: dict,
    review_dirs: list[Path],
    label_min_length: int,
    chunk_rows: int,
    workers: int,
    overwrite: bool,
) -> list[Path]:
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    worker_count = text_worker_count(workers, len(plans))
    tasks: list[dict] = []
    for row, plan in enumerate(plans):
        shard_path = shard_dir / f"{row:06d}_{Path(plan['path']).stem}.h5"
        plan["shard_path"] = shard_path
        appid = str(plan["game_name"]).split("_", 1)[0]
        tasks.append(
            {
                "plan": {
                    "index": int(plan["index"]),
                    "path": str(Path(plan["path"])),
                    "game_name": str(plan["game_name"]),
                    "reviews": int(plan["reviews"]),
                    "sentences": int(plan["sentences"]),
                },
                "shard_path": str(shard_path),
                "games": {appid: games.get(appid)},
                "review_dirs": [str(Path(path)) for path in review_dirs],
                "label_min_length": int(label_min_length),
                "chunk_rows": int(chunk_rows),
                "overwrite": bool(overwrite),
            }
        )

    started = time.time()
    if worker_count <= 1:
        print("write_text_shards: using 1 worker process", flush=True)
        for position, task in enumerate(tasks, start=1):
            result = _write_text_shard(task)
            if position % 25 == 0 or position == len(tasks):
                action = "reused" if result["reused"] else "wrote"
                print(
                    f"[text-shard {position}/{len(tasks)}] {action} "
                    f"{Path(result['path']).name} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    else:
        print(f"write_text_shards: using {worker_count} worker processes", flush=True)
        completed = 0
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_write_text_shard, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    action = "reused" if result["reused"] else "wrote"
                    print(
                        f"[text-shard {completed}/{len(tasks)}] last={action} "
                        f"{Path(result['path']).name} elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )

    return [Path(plan["shard_path"]) for plan in plans]


def write_string_dataset(h5: h5py.File, name: str, values: list[str] | np.ndarray) -> None:
    if name in h5:
        del h5[name]
    h5.create_dataset(name, data=np.asarray(values, dtype=string_dtype()))


def write_tag_metadata(h5: h5py.File, game_names: list[str], games: dict, mapping_path: Path) -> None:
    try:
        from VICReg_review.tag_mapping import load_tag_mapping, map_tag_dict, vectorize_tags
    except ImportError:  # pragma: no cover - direct script fallback
        from tag_mapping import load_tag_mapping, map_tag_dict, vectorize_tags

    spec = load_tag_mapping(mapping_path)
    tag_names = spec["tag_names"]
    labels = np.zeros((len(game_names), len(tag_names)), dtype=np.uint8)
    raw_counts = np.zeros((len(game_names), len(tag_names)), dtype=np.float32)
    missing: list[str] = []
    for row, game_name in enumerate(game_names):
        appid = str(game_name).split("_", 1)[0]
        record = games.get(appid) or {}
        if not record:
            missing.append(appid)
            continue
        mapped = map_tag_dict(record.get("tags") or {}, spec)
        labels[row], raw_counts[row] = vectorize_tags(mapped, tag_names)

    write_string_dataset(h5, "tag_names", tag_names)
    h5.create_dataset("tag_labels", data=labels, dtype=np.uint8)
    h5.create_dataset("tag_raw_counts", data=raw_counts, dtype=np.float32)
    h5.attrs["tag_mapping_json"] = json.dumps(spec["raw"], ensure_ascii=False, sort_keys=True)
    h5.attrs["tag_mapping_path"] = str(Path(mapping_path).resolve())
    h5.attrs["tag_missing_appids"] = json.dumps(sorted(set(missing)), ensure_ascii=False)
    h5.attrs["tag_count"] = len(tag_names)


def build_text_h5(
    sentences_dir: Path = DEFAULT_SENTENCES_DIR,
    games_json: Path = DEFAULT_GAMES_JSON,
    output_h5: Path = DEFAULT_TEXT_H5,
    overwrite: bool = False,
    limit_files: int = 0,
    chunk_rows: int = 8192,
    workers: int = 0,
    tag_mapping: Path = DEFAULT_TAG_MAPPING,
    no_tag_labels: bool = False,
    reviews_dirs=None,
    label_min_length: int = 0,
) -> Path:
    """Build ``text_h5.h5`` from split sentence JSON files."""
    sentences_dir = Path(sentences_dir)
    games_json = Path(games_json)
    output_h5 = Path(output_h5)
    if reviews_dirs is None:
        review_dirs: list[Path] = []
    elif isinstance(reviews_dirs, (str, Path)):
        review_dirs = [Path(reviews_dirs)]
    else:
        review_dirs = [Path(path) for path in reviews_dirs]

    worker_count = text_worker_count(workers, len(list(sentences_dir.glob("*.json"))))
    plans, total_reviews, total_sentences = scan_sentence_files(
        sentences_dir,
        limit_files,
        workers=worker_count,
    )
    review_cursor = 0
    sentence_cursor = 0
    for row, plan in enumerate(plans):
        plan["row"] = row
        plan["review_start"] = review_cursor
        plan["sentence_start"] = sentence_cursor
        review_cursor += int(plan["reviews"])
        sentence_cursor += int(plan["sentences"])

    expected = text_h5_expected_counts(plans, total_reviews, total_sentences)
    if output_h5.exists() and not overwrite:
        valid, reason, _ = validate_text_h5(output_h5, expected)
        if valid:
            write_text_h5_manifest(
                output_h5,
                expected,
                sentences_dir,
                games_json,
                review_dirs,
                label_min_length,
                limit_files,
                status="validated_existing",
            )
            print(
                f"build_text_h5: validated existing {output_h5} "
                f"games={expected['games']} reviews={expected['reviews']} "
                f"sentences={expected['sentences']} -> skip rebuild",
                flush=True,
            )
            return output_h5
        print(f"build_text_h5: existing {output_h5} is stale/incomplete ({reason}); rebuilding", flush=True)

    games = load_games_json(games_json)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    chunk_sentences = max(1, min(int(chunk_rows), total_sentences))
    chunk_reviews = max(1, min(int(chunk_rows), total_reviews))
    shard_dir = output_h5.with_name(output_h5.name + ".shards")
    partial_path = output_h5.with_name(output_h5.name + ".partial")
    shard_paths = write_text_shards(
        plans=plans,
        shard_dir=shard_dir,
        games=games,
        review_dirs=review_dirs,
        label_min_length=label_min_length,
        chunk_rows=chunk_rows,
        workers=worker_count,
        overwrite=overwrite,
    )

    try:
        with h5py.File(partial_path, "w") as h5:
            h5.create_dataset(
                "texts", shape=(total_sentences,), dtype=string_dtype(),
                chunks=(chunk_sentences,),
            )
            h5.create_dataset(
                "sentence_ids", shape=(total_sentences,), dtype=string_dtype(),
                chunks=(chunk_sentences,),
            )
            h5.create_dataset(
                "review_ids", shape=(total_reviews,), dtype=string_dtype(),
                chunks=(chunk_reviews,),
            )
            h5.create_dataset("review_offsets", shape=(total_reviews + 1,), dtype=np.int64)
            h5.create_dataset("game_review_offsets", shape=(len(plans) + 1,), dtype=np.int64)
            for name in (
                "game_names", "appids", "game_titles", "tags_json",
                "recommendation_label_source", "source_sentence_files",
            ):
                h5.create_dataset(name, shape=(len(plans),), dtype=string_dtype())
            h5.create_dataset("positive", shape=(len(plans),), dtype=np.int64)
            h5.create_dataset("negative", shape=(len(plans),), dtype=np.int64)
            h5.create_dataset("positive_rate", shape=(len(plans),), dtype=np.float32)

            texts = h5["texts"]
            sentence_ids = h5["sentence_ids"]
            review_ids = h5["review_ids"]
            review_offsets = h5["review_offsets"]
            game_review_offsets = h5["game_review_offsets"]
            review_offsets[0] = 0
            game_review_offsets[0] = 0

            for game_index, (plan, shard_path) in enumerate(zip(plans, shard_paths), start=1):
                row = game_index - 1
                review_start = int(plan["review_start"])
                sentence_start = int(plan["sentence_start"])
                review_end = review_start + int(plan["reviews"])
                sentence_end = sentence_start + int(plan["sentences"])
                path = Path(plan["path"])

                if not validate_text_shard(shard_path, plan):
                    raise RuntimeError(f"text shard validation failed: {shard_path}")
                with h5py.File(shard_path, "r") as shard:
                    for local_start in range(0, int(plan["sentences"]), chunk_sentences):
                        local_end = min(local_start + chunk_sentences, int(plan["sentences"]))
                        global_start = sentence_start + local_start
                        global_end = sentence_start + local_end
                        texts[global_start:global_end] = shard["texts"][local_start:local_end]
                        sentence_ids[global_start:global_end] = shard["sentence_ids"][local_start:local_end]

                    for local_start in range(0, int(plan["reviews"]), chunk_reviews):
                        local_end = min(local_start + chunk_reviews, int(plan["reviews"]))
                        global_start = review_start + local_start
                        global_end = review_start + local_end
                        review_ids[global_start:global_end] = shard["review_ids"][local_start:local_end]

                    review_offsets[review_start:review_end + 1] = (
                        shard["review_offsets"][:] + sentence_start
                    )
                    game_review_offsets[game_index] = review_end
                    h5["game_names"][row] = str(plan["game_name"])
                    h5["appids"][row] = decode_text(shard.attrs["meta_appid"])
                    h5["game_titles"][row] = decode_text(shard.attrs["meta_title"])
                    h5["tags_json"][row] = decode_text(shard.attrs["meta_tags_json"])
                    h5["recommendation_label_source"][row] = decode_text(shard.attrs["meta_label_source"])
                    h5["source_sentence_files"][row] = str(path.resolve())
                    h5["positive"][row] = int(shard.attrs["meta_positive"])
                    h5["negative"][row] = int(shard.attrs["meta_negative"])
                    h5["positive_rate"][row] = float(shard.attrs["meta_positive_rate"])

                if game_index % 25 == 0 or game_index == len(plans):
                    elapsed = time.time() - started
                    print(
                        f"[text-h5 merge {game_index}/{len(plans)}] {path.name}: "
                        f"reviews={plan['reviews']} sentences={plan['sentences']} "
                        f"total_sentences={sentence_end} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    h5.flush()

            if int(review_offsets[-1]) != total_sentences or int(game_review_offsets[-1]) != total_reviews:
                raise RuntimeError(
                    "text_h5 offset mismatch: "
                    f"review_offsets[-1]={int(review_offsets[-1])}/{total_sentences}, "
                    f"game_review_offsets[-1]={int(game_review_offsets[-1])}/{total_reviews}"
                )

            # --- finalize: tag labels + public attrs, then drop resume markers ---
            if not no_tag_labels:
                for key in ("tag_names", "tag_labels", "tag_raw_counts"):
                    if key in h5:
                        del h5[key]
                game_names = [decode_text(value) for value in h5["game_names"][:]]
                write_tag_metadata(h5, game_names, games, Path(tag_mapping))

            h5.attrs["schema"] = TEXT_H5_SCHEMA
            h5.attrs["source"] = "game_review_cleaned_3_sentences_text"
            h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            h5.attrs["games"] = len(plans)
            h5.attrs["reviews"] = total_reviews
            h5.attrs["sentences"] = total_sentences
            h5.attrs["games_json"] = str(Path(games_json).resolve())
            h5.attrs["sentences_dir"] = str(Path(sentences_dir).resolve())
            h5.attrs["recommendation_label_min_length"] = int(label_min_length)
            h5.attrs["recommendation_label_review_dirs"] = json.dumps(
                [str(path.resolve()) for path in review_dirs],
                ensure_ascii=False,
            )

        replace_with_retry(partial_path, output_h5)
        write_text_h5_manifest(
            output_h5,
            expected,
            sentences_dir,
            games_json,
            review_dirs,
            label_min_length,
            limit_files,
            status="written",
        )
    except BaseException:
        # Keep completed shards; the next run reuses matching shard files and
        # rebuilds this partial H5 merge deterministically.
        raise

    print(
        f"build_text_h5: wrote {output_h5} games={len(plans)} reviews={total_reviews} "
        f"sentences={total_sentences} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return output_h5


def copy_text_h5(source: h5py.File, target: h5py.File) -> None:
    for key in source.keys():
        if key == "vectors":
            continue
        source.copy(key, target)
    for key, value in source.attrs.items():
        target.attrs[key] = value


def compression_kwargs(name: str, level: int) -> dict:
    if name == "none":
        return {}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": level}
    if name == "lzf":
        return {"compression": "lzf"}
    raise ValueError(f"Unknown compression: {name}")


def embed_text_h5(
    input_h5: Path = DEFAULT_TEXT_H5,
    output_h5: Path = DEFAULT_EMBEDDING_H5,
    embedder=None,
    backend: str | None = None,
    embedding_model: str | None = None,
    overwrite: bool = False,
    read_batch_size: int = 4096,
    dtype: str = "float16",
    chunk_rows: int = 2048,
    compression: str = "none",
    gzip_level: int = 1,
) -> Path:
    """Stream texts from ``text_h5.h5`` and write ``embedding_h5.h5``."""
    if embedder is None:
        raise ValueError("embedder is required")

    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    if output_h5.exists() and not overwrite:
        print(f"embed_text_h5: skip existing {output_h5}", flush=True)
        return output_h5
    if not input_h5.exists():
        raise FileNotFoundError(f"Input text H5 not found: {input_h5}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = atomic_h5_path(output_h5)
    unlink_with_retry(tmp_path)
    vector_dtype = np.dtype(dtype)
    started = time.time()

    try:
        with h5py.File(input_h5, "r") as source, h5py.File(tmp_path, "w") as out:
            if "texts" not in source:
                raise ValueError(f"{input_h5} has no 'texts' dataset")
            total_sentences = int(source["texts"].shape[0])
            if total_sentences <= 0:
                raise ValueError(f"{input_h5} has no sentence texts")

            copy_text_h5(source, out)
            out.attrs["schema"] = EMBEDDING_H5_SCHEMA
            out.attrs["text_h5"] = str(input_h5.resolve())
            out.attrs["embedding_backend"] = "" if backend is None else str(backend)
            out.attrs["embedding_model"] = "" if embedding_model is None else str(embedding_model)
            out.attrs["embedding_dtype"] = str(vector_dtype)
            out.attrs["embedding_created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

            vectors_ds = None
            texts = source["texts"]
            for start in range(0, total_sentences, int(read_batch_size)):
                end = min(start + int(read_batch_size), total_sentences)
                batch_texts = [decode_text(value) for value in texts[start:end]]
                embedded = np.asarray(embedder.embed(batch_texts), dtype=vector_dtype)
                if embedded.ndim != 2 or embedded.shape[0] != len(batch_texts):
                    raise ValueError(
                        f"Embedder returned shape {embedded.shape}; expected "
                        f"({len(batch_texts)}, dim)"
                    )
                if vectors_ds is None:
                    dim = int(embedded.shape[1])
                    rows_per_chunk = max(1, min(int(chunk_rows), total_sentences))
                    vectors_ds = out.create_dataset(
                        "vectors",
                        shape=(total_sentences, dim),
                        chunks=(rows_per_chunk, dim),
                        dtype=vector_dtype,
                        **compression_kwargs(compression, gzip_level),
                    )
                    out.attrs["input_dim"] = dim
                    out.attrs["embedding_dim"] = dim
                elif int(embedded.shape[1]) != int(out.attrs["input_dim"]):
                    raise ValueError(
                        f"Embedding dimension changed from {out.attrs['input_dim']} "
                        f"to {embedded.shape[1]}"
                    )

                vectors_ds[start:end] = embedded
                if end == total_sentences or end % max(int(read_batch_size) * 10, 1) == 0:
                    elapsed = time.time() - started
                    print(
                        f"[embed-h5] {end}/{total_sentences} sentences "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                out.flush()

            out.attrs["dtype"] = str(vector_dtype)
            out.attrs["sentences"] = total_sentences

        replace_with_retry(tmp_path, output_h5)
    except BaseException:
        best_effort_unlink(tmp_path)
        raise

    print(
        f"embed_text_h5: wrote {output_h5} sentences={total_sentences} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return output_h5
