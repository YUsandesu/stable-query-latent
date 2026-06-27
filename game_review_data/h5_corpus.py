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
DEFAULT_TAP_MAPPING = PROJECT_ROOT / "VICReg_review" / "tags" / "tap_mapping.json"
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
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a review_id -> sentence mapping.")
    return raw


def scan_sentence_files(sentences_dir: Path, limit_files: int = 0) -> tuple[list[dict], int, int]:
    files = sorted(Path(sentences_dir).glob("*.json"))
    if limit_files > 0:
        files = files[:limit_files]
    if not files:
        raise ValueError(f"No sentence JSON files found in {sentences_dir}")

    plans: list[dict] = []
    total_reviews = 0
    total_sentences = 0
    for index, path in enumerate(files, start=1):
        raw = load_sentence_mapping(path)
        review_lengths: list[int] = []
        for _, sentences in iter_ordered_reviews(raw):
            review_lengths.append(len(sentences))
        sentence_count = int(sum(review_lengths))
        if sentence_count == 0:
            print(f"[scan {index}/{len(files)}] {path.name}: skip (no sentences)", flush=True)
            continue
        plans.append(
            {
                "path": path,
                "game_name": path.stem,
                "reviews": len(review_lengths),
                "sentences": sentence_count,
            }
        )
        total_reviews += len(review_lengths)
        total_sentences += sentence_count
        if index % 100 == 0 or index == len(files):
            print(
                f"[scan {index}/{len(files)}] games={len(plans)} "
                f"reviews={total_reviews} sentences={total_sentences}",
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
    data = json.loads(path.read_text(encoding="utf-8"))
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


def write_string_dataset(h5: h5py.File, name: str, values: list[str] | np.ndarray) -> None:
    if name in h5:
        del h5[name]
    h5.create_dataset(name, data=np.asarray(values, dtype=string_dtype()))


def write_tap_metadata(h5: h5py.File, game_names: list[str], games: dict, mapping_path: Path) -> None:
    try:
        from VICReg_review.tap_mapping import load_tap_mapping, map_tag_dict, vectorize_taps
    except ImportError:  # pragma: no cover - direct script fallback
        from tap_mapping import load_tap_mapping, map_tag_dict, vectorize_taps

    spec = load_tap_mapping(mapping_path)
    tap_names = spec["tap_names"]
    labels = np.zeros((len(game_names), len(tap_names)), dtype=np.uint8)
    raw_counts = np.zeros((len(game_names), len(tap_names)), dtype=np.float32)
    missing: list[str] = []
    for row, game_name in enumerate(game_names):
        appid = str(game_name).split("_", 1)[0]
        record = games.get(appid) or {}
        if not record:
            missing.append(appid)
            continue
        mapped = map_tag_dict(record.get("tags") or {}, spec)
        labels[row], raw_counts[row] = vectorize_taps(mapped, tap_names)

    write_string_dataset(h5, "tap_names", tap_names)
    h5.create_dataset("tap_labels", data=labels, dtype=np.uint8)
    h5.create_dataset("tap_raw_counts", data=raw_counts, dtype=np.float32)
    h5.attrs["tap_mapping_json"] = json.dumps(spec["raw"], ensure_ascii=False, sort_keys=True)
    h5.attrs["tap_mapping_path"] = str(Path(mapping_path).resolve())
    h5.attrs["tap_missing_appids"] = json.dumps(sorted(set(missing)), ensure_ascii=False)
    h5.attrs["tap_count"] = len(tap_names)


def build_text_h5(
    sentences_dir: Path = DEFAULT_SENTENCES_DIR,
    games_json: Path = DEFAULT_GAMES_JSON,
    output_h5: Path = DEFAULT_TEXT_H5,
    overwrite: bool = False,
    limit_files: int = 0,
    chunk_rows: int = 8192,
    tap_mapping: Path = DEFAULT_TAP_MAPPING,
    no_tap_labels: bool = False,
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

    plans, total_reviews, total_sentences = scan_sentence_files(sentences_dir, limit_files)
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
    tmp_path = atomic_h5_path(output_h5)
    unlink_with_retry(tmp_path)
    started = time.time()
    chunk_sentences = max(1, min(int(chunk_rows), total_sentences))
    chunk_reviews = max(1, min(int(chunk_rows), total_reviews))

    try:
        with h5py.File(tmp_path, "w") as h5:
            texts = h5.create_dataset(
                "texts",
                shape=(total_sentences,),
                dtype=string_dtype(),
                chunks=(chunk_sentences,),
            )
            sentence_ids = h5.create_dataset(
                "sentence_ids",
                shape=(total_sentences,),
                dtype=string_dtype(),
                chunks=(chunk_sentences,),
            )
            review_ids = h5.create_dataset(
                "review_ids",
                shape=(total_reviews,),
                dtype=string_dtype(),
                chunks=(chunk_reviews,),
            )
            review_offsets = h5.create_dataset("review_offsets", shape=(total_reviews + 1,), dtype=np.int64)
            game_review_offsets = h5.create_dataset(
                "game_review_offsets",
                shape=(len(plans) + 1,),
                dtype=np.int64,
            )

            game_names: list[str] = []
            appids: list[str] = []
            game_titles: list[str] = []
            tags_json: list[str] = []
            positives: list[int] = []
            negatives: list[int] = []
            positive_rates: list[float] = []
            label_sources: list[str] = []
            source_sentence_files: list[str] = []

            sentence_cursor = 0
            review_cursor = 0
            review_offsets[0] = 0
            game_review_offsets[0] = 0

            for game_index, plan in enumerate(plans, start=1):
                path = Path(plan["path"])
                raw = load_sentence_mapping(path)
                meta = game_metadata(
                    str(plan["game_name"]),
                    games,
                    reviews_dirs=review_dirs,
                    label_min_length=label_min_length,
                )

                game_names.append(str(plan["game_name"]))
                appids.append(meta["appid"])
                game_titles.append(meta["title"])
                positives.append(int(meta["positive"]))
                negatives.append(int(meta["negative"]))
                positive_rates.append(float(meta["positive_rate"]))
                tags_json.append(meta["tags_json"])
                label_sources.append(meta["label_source"])
                source_sentence_files.append(str(path.resolve()))

                written_reviews = 0
                written_sentences = 0
                for review_id, sentences in iter_ordered_reviews(raw):
                    review_ids[review_cursor] = review_id
                    local_sentence_ids = [sentence_id for sentence_id, _ in sentences]
                    local_texts = [text for _, text in sentences]
                    count = len(local_texts)
                    end = sentence_cursor + count
                    texts[sentence_cursor:end] = local_texts
                    sentence_ids[sentence_cursor:end] = local_sentence_ids
                    sentence_cursor = end
                    review_cursor += 1
                    review_offsets[review_cursor] = sentence_cursor
                    written_reviews += 1
                    written_sentences += count

                game_review_offsets[game_index] = review_cursor
                if game_index % 25 == 0 or game_index == len(plans):
                    elapsed = time.time() - started
                    print(
                        f"[text-h5 {game_index}/{len(plans)}] {path.name}: "
                        f"reviews={written_reviews} sentences={written_sentences} "
                        f"total_sentences={sentence_cursor} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            if review_cursor != total_reviews or sentence_cursor != total_sentences:
                raise RuntimeError(
                    "text_h5 write count mismatch: "
                    f"reviews {review_cursor}/{total_reviews}, "
                    f"sentences {sentence_cursor}/{total_sentences}"
                )

            write_string_dataset(h5, "game_names", game_names)
            write_string_dataset(h5, "appids", appids)
            write_string_dataset(h5, "game_titles", game_titles)
            write_string_dataset(h5, "tags_json", tags_json)
            write_string_dataset(h5, "recommendation_label_source", label_sources)
            write_string_dataset(h5, "source_sentence_files", source_sentence_files)
            h5.create_dataset("positive", data=np.asarray(positives, dtype=np.int64))
            h5.create_dataset("negative", data=np.asarray(negatives, dtype=np.int64))
            h5.create_dataset("positive_rate", data=np.asarray(positive_rates, dtype=np.float32))
            if not no_tap_labels:
                write_tap_metadata(h5, game_names, games, Path(tap_mapping))

            h5.attrs["schema"] = TEXT_H5_SCHEMA
            h5.attrs["source"] = "game_review_cleaned_3_sentences_text"
            h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            h5.attrs["games"] = len(game_names)
            h5.attrs["reviews"] = total_reviews
            h5.attrs["sentences"] = total_sentences
            h5.attrs["games_json"] = str(Path(games_json).resolve())
            h5.attrs["sentences_dir"] = str(Path(sentences_dir).resolve())
            h5.attrs["recommendation_label_min_length"] = int(label_min_length)
            h5.attrs["recommendation_label_review_dirs"] = json.dumps(
                [str(path.resolve()) for path in review_dirs],
                ensure_ascii=False,
            )

        replace_with_retry(tmp_path, output_h5)
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
        best_effort_unlink(tmp_path)
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
