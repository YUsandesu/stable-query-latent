import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INPUT_CSV = str(SCRIPT_DIR.parent / "pesudo_data" / "pseudo_text_data_one_per_game.csv")
DEFAULT_TEXT_COLUMN = "generated_text"
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR.parent / "pesudo_data" / "pseudo_text_sentence_embeddings")
DEFAULT_SENTENCE_MODEL_NAME = "sat-3l-sm"


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def resolve_script_relative(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_sentence_splitter(args):
    try:
        from wtpsplit import SaT
    except ImportError as exc:
        raise ImportError("wtpsplit is required for sentence splitting. Install it with: pip install wtpsplit") from exc

    splitter = SaT(args.sentence_model_name)
    if args.sentence_device:
        splitter.to(args.sentence_device)
        if args.sentence_device.startswith("cuda"):
            splitter.half()
    return splitter


def load_source_rows(input_csv, text_column):
    rows = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if text_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"Column '{text_column}' was not found. Available columns: {available}")

        for row_index, row in enumerate(reader):
            rows.append((row_index, row, normalize_text(row[text_column])))

    return rows


def build_sentence_records(source_rows, splitter):
    records = []
    texts = [text for _, _, text in source_rows]
    split_texts = list(splitter.split(texts))

    for (row_index, row, _), sentences in zip(source_rows, split_texts):
        for sentence_index, sentence in enumerate(sentences):
            cleaned_sentence = sentence.strip()
            if not cleaned_sentence:
                continue
            records.append(
                {
                    "row_index": row_index,
                    "sentence_index": sentence_index,
                    "game_id": row.get("game_id", ""),
                    "game_name": row.get("game_name", ""),
                    "genre_id": row.get("genre_id", ""),
                    "genre_name": row.get("genre_name", ""),
                    "sentence": cleaned_sentence,
                }
            )

    return records


def write_metadata(records, output_path):
    fieldnames = [
        "embedding_index",
        "row_index",
        "sentence_index",
        "game_id",
        "game_name",
        "genre_id",
        "genre_name",
        "sentence",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for embedding_index, record in enumerate(records):
            writer.writerow({"embedding_index": embedding_index, **record})


def encode_with_sentence_transformers(sentences, args):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for --backend sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(args.model_name, device=args.device)
    return model.encode(
        sentences,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize_embeddings,
    ).astype(np.float32)


def last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def encode_with_transformers(sentences, args):
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for --backend transformers. "
            "Install them with: pip install transformers torch"
        ) from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    embeddings = []
    for start in range(0, len(sentences), args.batch_size):
        batch_sentences = sentences[start : start + args.batch_size]
        batch = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            batch_embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            if args.normalize_embeddings:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        embeddings.append(batch_embeddings.cpu().float().numpy())
        print(f"Encoded {min(start + args.batch_size, len(sentences))}/{len(sentences)} sentences")

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split pseudo_text_data_one_per_game descriptions into sentences and embed each sentence with Qwen."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, type=Path)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--sentence-model-name",
        default=DEFAULT_SENTENCE_MODEL_NAME,
        help="wtpsplit SaT model used for sentence segmentation.",
    )
    parser.add_argument(
        "--sentence-device",
        default=None,
        help="Optional device for wtpsplit sentence segmentation, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-length", default=8192, type=int)
    parser.add_argument(
        "--backend",
        choices=["transformers", "sentence-transformers"],
        default="transformers",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = resolve_script_relative(args.input_csv).resolve()
    output_dir = resolve_script_relative(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_source_rows(input_csv, args.text_column)
    sentence_splitter = load_sentence_splitter(args)
    records = build_sentence_records(source_rows, sentence_splitter)
    sentences = [record["sentence"] for record in records]

    if not sentences:
        raise ValueError(f"No sentences were found in {input_csv}.")

    if args.backend == "sentence-transformers":
        embeddings = encode_with_sentence_transformers(sentences, args)
    else:
        embeddings = encode_with_transformers(sentences, args)

    np.save(output_dir / "sentence_embeddings.npy", embeddings)
    write_metadata(records, output_dir / "sentence_metadata.csv")

    manifest = {
        "input_csv": str(input_csv),
        "text_column": args.text_column,
        "model_name": args.model_name,
        "sentence_splitter": "wtpsplit.SaT",
        "sentence_model_name": args.sentence_model_name,
        "sentence_device": args.sentence_device,
        "backend": args.backend,
        "max_length": args.max_length,
        "sentence_count": len(sentences),
        "embedding_shape": list(embeddings.shape),
        "normalize_embeddings": args.normalize_embeddings,
        "embeddings_file": "sentence_embeddings.npy",
        "metadata_file": "sentence_metadata.csv",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(sentences)} sentence embeddings to {output_dir}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
