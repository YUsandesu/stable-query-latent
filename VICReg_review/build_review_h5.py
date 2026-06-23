"""Build HDF5 training data for VICReg game-review training.

The JSON files are expensive to parse during training. This script converts
them into several shard H5 files in parallel, then merges the shards into one
streamable H5 layout:

    vectors              (total_sentences, 1024)
    review_offsets       (total_reviews + 1) offsets into vectors
    game_review_offsets  (num_games + 1) offsets into review_offsets
    game_names           (num_games)

All vectors for a game are contiguous, and all sentences for a review are
contiguous. Training can therefore sample reviews and load full games with
large sequential reads instead of reparsing JSON every step.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "game_review_data" / "game_review_cleaned_3_sentences"
DEFAULT_H5_DIR = SCRIPT_DIR / "h5"
DEFAULT_SHARD_DIR = DEFAULT_H5_DIR / "shards"
DEFAULT_OUTPUT_H5 = DEFAULT_H5_DIR / "game_review_cleaned_3_sentences.h5"


def numeric_suffix(value, prefix):
    text = str(value)
    if text.startswith(prefix):
        text = text[len(prefix):]
    try:
        return int(text)
    except ValueError:
        return text


def compression_kwargs(name, level):
    if name == "none":
        return {}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": level}
    if name == "lzf":
        return {"compression": "lzf"}
    raise ValueError(f"Unknown compression: {name}")


def atomic_h5_path(path):
    path = Path(path)
    return path.with_name(path.name + ".tmp")


def load_game_as_arrays(path, dtype, input_dim):
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a review mapping.")

    reviews = []
    review_items = sorted(raw.items(), key=lambda item: numeric_suffix(item[0], ""))
    for _, sentence_map in review_items:
        if not isinstance(sentence_map, dict):
            continue
        vectors = []
        sentence_items = sorted(
            sentence_map.items(),
            key=lambda item: numeric_suffix(item[0], "sentence_"),
        )
        for _, payload in sentence_items:
            if not isinstance(payload, dict):
                continue
            vector = payload.get("vector")
            if not vector:
                continue
            if len(vector) != input_dim:
                raise ValueError(f"{path}: vector dim {len(vector)} != expected {input_dim}")
            vectors.append(vector)
        if vectors:
            reviews.append(np.asarray(vectors, dtype=dtype))

    if not reviews:
        raise ValueError(f"{path} contains no vectors.")

    sentence_count = sum(review.shape[0] for review in reviews)
    flat = np.empty((sentence_count, input_dim), dtype=dtype)
    lengths = np.empty((len(reviews),), dtype=np.int64)
    cursor = 0
    for index, review in enumerate(reviews):
        length = review.shape[0]
        flat[cursor : cursor + length] = review
        lengths[index] = length
        cursor += length
    return flat, lengths


def resize_append(dataset, values):
    values = np.asarray(values)
    start = dataset.shape[0]
    dataset.resize((start + values.shape[0],) + dataset.shape[1:])
    dataset[start : start + values.shape[0]] = values
    return start


def write_shard(task):
    shard_index = task["shard_index"]
    files = [Path(value) for value in task["files"]]
    output_path = Path(task["output_path"])
    dtype = np.dtype(task["dtype"])
    input_dim = int(task["input_dim"])
    chunk_rows = int(task["chunk_rows"])
    compression = task["compression"]
    gzip_level = int(task["gzip_level"])
    overwrite = bool(task["overwrite"])

    if output_path.exists() and not overwrite:
        return {"path": str(output_path), "skipped": True, "games": None}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = atomic_h5_path(output_path)
    tmp_path.unlink(missing_ok=True)

    vector_count = 0
    review_count = 0
    game_names = []
    review_offsets = [0]
    game_review_offsets = [0]
    started = time.time()

    try:
        with h5py.File(tmp_path, "w") as h5:
            vectors = h5.create_dataset(
                "vectors",
                shape=(0, input_dim),
                maxshape=(None, input_dim),
                chunks=(chunk_rows, input_dim),
                dtype=dtype,
                **compression_kwargs(compression, gzip_level),
            )
            for file_index, path in enumerate(files, start=1):
                flat, lengths = load_game_as_arrays(path, dtype, input_dim)
                resize_append(vectors, flat)

                cumulative = int(review_offsets[-1])
                review_offsets.extend((cumulative + np.cumsum(lengths)).astype(np.int64).tolist())
                review_count += len(lengths)
                vector_count += flat.shape[0]
                game_review_offsets.append(review_count)
                game_names.append(path.stem)

                elapsed = time.time() - started
                print(
                    f"[shard {shard_index}] {file_index}/{len(files)} {path.name}: "
                    f"reviews={len(lengths)} sentences={flat.shape[0]} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            h5.create_dataset("review_offsets", data=np.asarray(review_offsets, dtype=np.int64))
            h5.create_dataset(
                "game_review_offsets",
                data=np.asarray(game_review_offsets, dtype=np.int64),
            )
            h5.create_dataset(
                "game_names",
                data=np.asarray(game_names, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            h5.attrs["input_dim"] = input_dim
            h5.attrs["dtype"] = str(dtype)
            h5.attrs["source"] = "game_review_cleaned_3_sentences"
            h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            h5.attrs["games"] = len(game_names)
            h5.attrs["reviews"] = review_count
            h5.attrs["sentences"] = vector_count

        tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return {
        "path": str(output_path),
        "skipped": False,
        "games": len(game_names),
        "reviews": review_count,
        "sentences": vector_count,
        "elapsed_seconds": round(time.time() - started, 1),
    }


def partition_files(files, shard_count):
    shard_count = max(1, min(shard_count, len(files)))
    buckets = [{"size": 0, "files": []} for _ in range(shard_count)]
    for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True):
        bucket = min(buckets, key=lambda item: item["size"])
        bucket["files"].append(path)
        bucket["size"] += path.stat().st_size
    return [bucket["files"] for bucket in buckets if bucket["files"]]


def build_shards(args):
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.json"))
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise ValueError(f"No JSON files found in {input_dir}")

    shard_count = args.shards or args.workers
    partitions = partition_files(files, shard_count)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, partition in enumerate(partitions):
        tasks.append(
            {
                "shard_index": index,
                "files": [str(path) for path in partition],
                "output_path": str(args.shard_dir / f"review_vectors_shard_{index:04d}.h5"),
                "dtype": args.dtype,
                "input_dim": args.input_dim,
                "chunk_rows": args.chunk_rows,
                "compression": args.compression,
                "gzip_level": args.gzip_level,
                "overwrite": args.overwrite,
            }
        )

    print(
        f"build_shards: files={len(files)} shards={len(tasks)} workers={args.workers} "
        f"dtype={args.dtype} compression={args.compression}",
        flush=True,
    )
    if args.workers == 1:
        results = [write_shard(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = list(pool.imap_unordered(write_shard, tasks))
    return sorted(Path(result["path"]) for result in results)


def read_string_dataset(dataset):
    values = dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def merge_shards(shard_paths, output_path, args):
    shard_paths = [Path(path) for path in shard_paths]
    output_path = Path(output_path)
    if output_path.exists() and not args.overwrite:
        print(f"merge_shards: skip existing {output_path}", flush=True)
        return output_path

    totals = {"games": 0, "reviews": 0, "sentences": 0}
    input_dim = None
    for path in shard_paths:
        with h5py.File(path, "r") as h5:
            if input_dim is None:
                input_dim = int(h5.attrs["input_dim"])
            elif input_dim != int(h5.attrs["input_dim"]):
                raise ValueError(f"{path}: input_dim mismatch")
            totals["games"] += int(h5["game_names"].shape[0])
            totals["reviews"] += int(h5["review_offsets"].shape[0] - 1)
            totals["sentences"] += int(h5["vectors"].shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = atomic_h5_path(output_path)
    tmp_path.unlink(missing_ok=True)
    started = time.time()
    try:
        with h5py.File(tmp_path, "w") as out:
            vectors = out.create_dataset(
                "vectors",
                shape=(totals["sentences"], input_dim),
                chunks=(args.chunk_rows, input_dim),
                dtype=np.dtype(args.dtype),
                **compression_kwargs(args.compression, args.gzip_level),
            )
            review_offsets = out.create_dataset(
                "review_offsets",
                shape=(totals["reviews"] + 1,),
                dtype=np.int64,
            )
            game_review_offsets = out.create_dataset(
                "game_review_offsets",
                shape=(totals["games"] + 1,),
                dtype=np.int64,
            )

            review_offsets[0] = 0
            game_review_offsets[0] = 0
            game_names = []
            sentence_cursor = 0
            review_cursor = 0
            game_cursor = 0

            for shard_index, path in enumerate(shard_paths, start=1):
                with h5py.File(path, "r") as h5:
                    shard_sentences = int(h5["vectors"].shape[0])
                    shard_reviews = int(h5["review_offsets"].shape[0] - 1)
                    shard_games = int(h5["game_names"].shape[0])

                    for start in range(0, shard_sentences, args.copy_rows):
                        end = min(start + args.copy_rows, shard_sentences)
                        vectors[sentence_cursor + start : sentence_cursor + end] = h5["vectors"][start:end]

                    shard_review_offsets = h5["review_offsets"][1:] + sentence_cursor
                    review_offsets[review_cursor + 1 : review_cursor + 1 + shard_reviews] = shard_review_offsets

                    shard_game_offsets = h5["game_review_offsets"][1:] + review_cursor
                    game_review_offsets[game_cursor + 1 : game_cursor + 1 + shard_games] = shard_game_offsets
                    game_names.extend(read_string_dataset(h5["game_names"]))

                    sentence_cursor += shard_sentences
                    review_cursor += shard_reviews
                    game_cursor += shard_games
                    print(
                        f"merge {shard_index}/{len(shard_paths)} {path.name}: "
                        f"games={shard_games} reviews={shard_reviews} sentences={shard_sentences}",
                        flush=True,
                    )

            out.create_dataset(
                "game_names",
                data=np.asarray(game_names, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            out.attrs["input_dim"] = input_dim
            out.attrs["dtype"] = args.dtype
            out.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            out.attrs["games"] = totals["games"]
            out.attrs["reviews"] = totals["reviews"]
            out.attrs["sentences"] = totals["sentences"]
            out.attrs["source_shards"] = len(shard_paths)

        tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    print(
        f"merged: {output_path} games={totals['games']} reviews={totals['reviews']} "
        f"sentences={totals['sentences']} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--copy-rows", type=int, default=16384)
    parser.add_argument("--compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-merge", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    if args.only_merge:
        shard_paths = sorted(args.shard_dir.glob("*.h5"))
        if not shard_paths:
            raise ValueError(f"No shard H5 files found in {args.shard_dir}")
    else:
        shard_paths = build_shards(args)

    if not args.skip_merge:
        merge_shards(shard_paths, args.output_h5, args)
        if not args.keep_shards:
            for shard_path in shard_paths:
                shard_path.unlink(missing_ok=True)

    print(f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
