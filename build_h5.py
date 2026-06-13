import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_SCORE_COLUMNS = [
    "psychological_meaning",
    "psychological_mastery",
    "psychological_curiosity",
    "psychological_autonomy",
    "psychological_immersion",
    "functional_progress_feedback",
    "functional_ease_of_control",
    "functional_audiovisual_appeal",
    "functional_goals_and_rules",
    "functional_challenge",
]

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_script_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def write_string_dataset(h5, name, values):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.asarray(values, dtype=object), dtype=string_dtype)


def create_numeric_dataset(h5, name, data, compression):
    kwargs = {}
    if compression:
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = 4
    h5.create_dataset(name, data=data, **kwargs)


def build_h5(
    target_csv,
    sentence_metadata_csv,
    embeddings_npy,
    output_h5,
    score_columns,
    compression,
):
    target_csv = resolve_script_path(target_csv)
    sentence_metadata_csv = resolve_script_path(sentence_metadata_csv)
    embeddings_npy = resolve_script_path(embeddings_npy)
    output_h5 = resolve_script_path(output_h5)

    target_rows = pd.read_csv(
        target_csv,
        dtype={"game_id": str, "genre_id": str, "game_name": str, "genre_name": str},
    )
    sentence_metadata = pd.read_csv(
        sentence_metadata_csv,
        dtype={"game_id": str, "genre_id": str, "game_name": str, "genre_name": str},
    )
    embeddings = np.load(embeddings_npy).astype(np.float32)

    missing_scores = [column for column in score_columns if column not in target_rows.columns]
    if missing_scores:
        raise ValueError(f"Missing score columns in target CSV: {missing_scores}")

    if len(sentence_metadata) != len(embeddings):
        raise ValueError(
            "sentence_metadata row count does not match sentence_embeddings.npy: "
            f"{len(sentence_metadata)} != {len(embeddings)}"
        )

    if "row_index" not in sentence_metadata.columns:
        raise ValueError("sentence_metadata CSV must contain a row_index column.")

    row_indices = pd.to_numeric(sentence_metadata["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    expected_row_indices = set(range(len(target_rows)))
    available_row_indices = set(row_indices.tolist())
    missing_row_indices = sorted(expected_row_indices - available_row_indices)
    if missing_row_indices:
        raise ValueError(
            "Some target CSV rows do not appear in sentence metadata row_index values: "
            f"{missing_row_indices[:20]}"
        )
    extra_row_indices = sorted(available_row_indices - expected_row_indices)
    if extra_row_indices:
        raise ValueError(
            "sentence metadata contains row_index values outside the target CSV range: "
            f"{extra_row_indices[:20]}"
        )

    row_to_embedding_indices = {
        int(row_index): group.sort_values("sentence_index")["embedding_index"].to_numpy(dtype=np.int64)
        for row_index, group in sentence_metadata.groupby(row_indices, sort=False)
    }
    sequence_lengths = np.asarray(
        [len(row_to_embedding_indices[row_index]) for row_index in range(len(target_rows))],
        dtype=np.int64,
    )

    sample_count = len(target_rows)
    max_sequence_length = int(sequence_lengths.max())
    embedding_dim = int(embeddings.shape[1])
    inputs = np.zeros((sample_count, max_sequence_length, embedding_dim), dtype=np.float32)
    key_padding_mask = np.ones((sample_count, max_sequence_length), dtype=np.bool_)

    for row_index in range(len(target_rows)):
        embedding_indices = row_to_embedding_indices[row_index]
        length = len(embedding_indices)
        inputs[row_index, :length] = embeddings[embedding_indices]
        key_padding_mask[row_index, :length] = False

    targets = target_rows[score_columns].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as h5:
        h5.attrs["source_target_csv"] = str(target_csv.resolve())
        h5.attrs["source_sentence_metadata_csv"] = str(sentence_metadata_csv.resolve())
        h5.attrs["source_embeddings_npy"] = str(embeddings_npy.resolve())
        h5.attrs["score_columns_json"] = json.dumps(score_columns)
        h5.attrs["input_layout"] = "text_variant_sample, sentence_token, embedding_dim"
        h5.attrs["target_layout"] = "mapped_game_score_repeated_per_text_variant"

        h5.attrs["numeric_dataset_compression"] = compression or "none"

        create_numeric_dataset(h5, "inputs", inputs, compression)
        create_numeric_dataset(h5, "key_padding_mask", key_padding_mask, compression)
        create_numeric_dataset(h5, "targets", targets, compression)
        h5.create_dataset("sequence_lengths", data=sequence_lengths)
        h5.create_dataset(
            "benchmark_row_index",
            data=np.arange(sample_count, dtype=np.int64),
        )

        write_string_dataset(h5, "score_columns", score_columns)
        write_string_dataset(h5, "benchmark_game_id", target_rows["game_id"].tolist())
        write_string_dataset(h5, "benchmark_game_name", target_rows["game_name"].astype(str).tolist())
        write_string_dataset(h5, "benchmark_genre_id", target_rows["genre_id"].astype(str).tolist())
        write_string_dataset(h5, "benchmark_genre_name", target_rows["genre_name"].astype(str).tolist())
        h5.create_dataset(
            "benchmark_sample_index",
            data=np.arange(sample_count, dtype=np.int64),
        )
        h5.create_dataset(
            "benchmark_score_sample_count",
            data=target_rows["score_sample_count"].to_numpy(dtype=np.int64),
        )
        if "text_variant_index" in target_rows.columns:
            h5.create_dataset(
                "text_variant_index",
                data=pd.to_numeric(target_rows["text_variant_index"], errors="raise").to_numpy(
                    dtype=np.int64
                ),
            )

        metadata_group = h5.create_group("sentence_metadata")
        for column in sentence_metadata.columns:
            values = sentence_metadata[column]
            numeric_values = pd.to_numeric(values, errors="coerce")
            if numeric_values.notna().all():
                metadata_group.create_dataset(column, data=numeric_values.to_numpy())
            else:
                write_string_dataset(metadata_group, column, values.astype(str).tolist())

    return {
        "path": str(output_h5),
        "samples": sample_count,
        "max_sequence_length": max_sequence_length,
        "embedding_dim": embedding_dim,
        "output_dim": len(score_columns),
        "compression": compression or "none",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Package mapped pseudo-text scores, sentence metadata, and sentence embeddings into HDF5."
    )
    parser.add_argument("--target-csv", default="pseudo_text_data_one_per_game.csv")
    parser.add_argument("--benchmark-csv", dest="target_csv", help=argparse.SUPPRESS)
    parser.add_argument(
        "--sentence-metadata-csv",
        default="pseudo_text_sentence_embeddings/sentence_metadata.csv",
    )
    parser.add_argument(
        "--embeddings-npy",
        default="pseudo_text_sentence_embeddings/sentence_embeddings.npy",
    )
    parser.add_argument("--output-h5", default="benchmark_sentence_latent_query.h5")
    parser.add_argument("--score-columns", nargs="*", default=DEFAULT_SCORE_COLUMNS)
    parser.add_argument(
        "--compression",
        choices=["none", "gzip", "lzf"],
        default="none",
        help="Compression for numeric training arrays. Default none is faster for random training reads.",
    )
    args = parser.parse_args()
    compression = None if args.compression == "none" else args.compression

    summary = build_h5(
        args.target_csv,
        args.sentence_metadata_csv,
        args.embeddings_npy,
        args.output_h5,
        args.score_columns,
        compression,
    )
    print(
        "wrote {path}: samples={samples}, max_tokens={max_sequence_length}, "
        "embedding_dim={embedding_dim}, output_dim={output_dim}, compression={compression}".format(**summary)
    )


if __name__ == "__main__":
    main()
