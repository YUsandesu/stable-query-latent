"""Reassemble streamed embedding shards into a final embedding H5.

Usage:
    python game_review_data/combine_shard.py --shard-dir <drive-dir> --text-h5 <text_h5.h5> --output <embedding_h5.h5>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from game_review_data.h5_corpus import (
        EMBEDDING_H5_SCHEMA,
        atomic_h5_path,
        best_effort_unlink,
        compression_kwargs,
        copy_text_h5,
        replace_with_retry,
    )
except ImportError:  # pragma: no cover - direct script execution
    from h5_corpus import (
        EMBEDDING_H5_SCHEMA,
        atomic_h5_path,
        best_effort_unlink,
        compression_kwargs,
        copy_text_h5,
        replace_with_retry,
    )

from game_review_data.embedding_incloud import STREAM_MANIFEST_SCHEMA, stream_manifest_path


def decode_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_text_h5_counts(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return {
            "schema": decode_text(h5.attrs.get("schema", "")),
            "games": int(h5.attrs.get("games", -1)),
            "reviews": int(h5.attrs.get("reviews", -1)),
            "sentences": int(h5.attrs.get("sentences", -1)),
            "review_offsets": int(h5["review_offsets"].shape[0]) if "review_offsets" in h5 else -1,
            "game_review_offsets": int(h5["game_review_offsets"].shape[0]) if "game_review_offsets" in h5 else -1,
        }


def load_manifest(shard_dir: Path) -> dict:
    manifest_path = stream_manifest_path(shard_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != STREAM_MANIFEST_SCHEMA:
        raise ValueError(f"Unexpected manifest schema: {payload.get('schema')}")
    return payload


def validate_manifest(manifest: dict) -> tuple[list[dict], dict]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Manifest has no shards")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Manifest has no config")
    if manifest.get("status") != "complete":
        raise ValueError(f"Manifest status must be complete, got {manifest.get('status')!r}")
    ordered = []
    total_rows = 0
    last_end = None
    expected_id = 0
    for shard in sorted(shards, key=lambda item: int(item["id"])):
        if int(shard["id"]) != expected_id:
            raise ValueError(f"Shard ids must be continuous from 0; expected {expected_id}, got {shard['id']}")
        if shard.get("upload_status") != "done":
            raise ValueError(f"Shard {shard.get('id')} is not done")
        start = int(shard["start"])
        end = int(shard["end"])
        rows = int(shard["rows"])
        if end <= start or rows != end - start:
            raise ValueError(f"Shard {shard.get('id')} has invalid range")
        if last_end is not None and start != last_end:
            raise ValueError(f"Shard {shard.get('id')} breaks continuity at {start}")
        last_end = end
        total_rows += rows
        ordered.append(shard)
        expected_id += 1
    if total_rows != int(config["total_sentences"]):
        raise ValueError(f"Total shard rows {total_rows} != total_sentences {config['total_sentences']}")
    return ordered, config


def build_output(
    text_h5: Path,
    output: Path,
    total_sentences: int,
    dim: int,
    dtype: np.dtype,
    model_name: str,
    compression: str,
    gzip_level: int,
):
    tmp_path = atomic_h5_path(output)
    best_effort_unlink(tmp_path)
    with h5py.File(text_h5, "r") as source, h5py.File(tmp_path, "w") as out:
        copy_text_h5(source, out)
        out.attrs["schema"] = EMBEDDING_H5_SCHEMA
        out.attrs["text_h5"] = str(Path(text_h5).resolve())
        out.attrs["embedding_backend"] = "stream_recombine"
        out.attrs["embedding_model"] = str(model_name)
        out.attrs["embedding_dtype"] = str(dtype)
        out.attrs["embedding_created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out.attrs["input_dim"] = int(dim)
        out.attrs["embedding_dim"] = int(dim)
        out.attrs["dtype"] = str(dtype)
        out.attrs["sentences"] = int(total_sentences)
        rows_per_chunk = max(1, min(int(2048), int(total_sentences)))
        out.create_dataset(
            "vectors",
            shape=(int(total_sentences), int(dim)),
            chunks=(rows_per_chunk, int(dim)),
            dtype=dtype,
            **compression_kwargs(compression, gzip_level),
        )
    return tmp_path


def combine_shards(shard_dir: Path, text_h5: Path, output: Path, *, compression: str, gzip_level: int) -> Path:
    manifest = load_manifest(shard_dir)
    shards, config = validate_manifest(manifest)
    dtype = np.dtype(config["dtype"])
    total_sentences = int(config["total_sentences"])
    dim = int(config["dim"])
    model_name = str(config.get("model") or "")
    counts = read_text_h5_counts(text_h5)
    if counts["sentences"] != total_sentences:
        raise ValueError(f"text_h5 sentences={counts['sentences']} but manifest total_sentences={total_sentences}")
    tmp_path = build_output(text_h5, output, total_sentences, dim, dtype, model_name, compression, gzip_level)

    try:
        with h5py.File(tmp_path, "a") as out:
            vectors = out["vectors"]
            for shard in shards:
                shard_path = Path(shard.get("remote_path") or (Path(shard_dir) / f"shard_{int(shard['id']):05d}.h5"))
                if not shard_path.exists():
                    raise FileNotFoundError(f"Missing shard file: {shard_path}")
                with h5py.File(shard_path, "r") as h5:
                    if "vectors" not in h5:
                        raise ValueError(f"{shard_path} has no vectors dataset")
                    shard_vectors = h5["vectors"]
                    if tuple(shard_vectors.shape) != (int(shard["rows"]), dim):
                        raise ValueError(f"{shard_path} shape mismatch: {shard_vectors.shape}")
                    if str(shard_vectors.dtype) != str(dtype):
                        raise ValueError(f"{shard_path} dtype mismatch: {shard_vectors.dtype}")
                    vectors[int(shard["start"]):int(shard["end"])] = shard_vectors[:]
            review_offsets = out["review_offsets"][:]
            if int(review_offsets[-1]) != int(vectors.shape[0]):
                raise ValueError(
                    f"review_offsets[-1]={int(review_offsets[-1])} but vectors rows={int(vectors.shape[0])}"
                )
            if int(out["game_review_offsets"][-1]) != int(out["review_offsets"].shape[0] - 1):
                raise ValueError(
                    f"game_review_offsets[-1]={int(out['game_review_offsets'][-1])} "
                    f"but reviews={int(out['review_offsets'].shape[0] - 1)}"
                )
            out.attrs["schema"] = EMBEDDING_H5_SCHEMA
            out.attrs["embedding_backend"] = "stream_recombine"
            out.attrs["embedding_model"] = str(model_name)
            out.attrs["embedding_dtype"] = str(dtype)
            out.attrs["embedding_created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        replace_with_retry(tmp_path, output)
    except BaseException:
        best_effort_unlink(tmp_path)
        raise
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--text-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--gzip-level", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    combine_shards(
        args.shard_dir,
        args.text_h5,
        args.output,
        compression=args.compression,
        gzip_level=args.gzip_level,
    )


if __name__ == "__main__":
    main()
