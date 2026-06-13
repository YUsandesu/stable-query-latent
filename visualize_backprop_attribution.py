import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from embed_pseudo_text_sentences import (
    encode_with_sentence_transformers,
    encode_with_transformers,
    load_sentence_splitter,
    normalize_text,
)
from latent_query_model import LatentQueryFlatRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "latent_query_benchmark_multi_classifier.pt"
DEFAULT_OUTPUT_HTML = "backprop_attribution.html"


def resolve_script_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_manifest(path):
    if not path:
        return {}
    path = resolve_script_path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_embed_args(args, manifest):
    return SimpleNamespace(
        model_name=args.embedding_model or manifest.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
        sentence_model_name=args.sentence_model_name
        or manifest.get("sentence_model_name", "sat-3l-sm"),
        sentence_device=args.sentence_device,
        backend=args.embedding_backend or manifest.get("backend", "transformers"),
        device=args.embedding_device or args.device,
        batch_size=args.embedding_batch_size,
        max_length=args.max_length or int(manifest.get("max_length", 8192)),
        normalize_embeddings=args.normalize_embeddings
        if args.normalize_embeddings
        else bool(manifest.get("normalize_embeddings", False)),
    )


def split_text(text, embed_args):
    text = normalize_text(text)
    if not text:
        raise ValueError("Input text is empty.")

    splitter = load_sentence_splitter(embed_args)
    split_result = list(splitter.split([text]))
    if split_result and isinstance(split_result[0], str):
        raw_sentences = split_result
    else:
        raw_sentences = split_result[0] if split_result else []

    sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]
    if not sentences:
        return [text]
    return sentences


def embed_sentences(sentences, embed_args):
    if embed_args.backend == "sentence-transformers":
        return encode_with_sentence_transformers(sentences, embed_args)
    return encode_with_transformers(sentences, embed_args)


def load_model(checkpoint_path, device):
    checkpoint_path = resolve_script_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    score_dim = int(checkpoint["score_dim"])
    score_class_count = int(checkpoint.get("score_class_count", 5))
    model = LatentQueryFlatRegressor(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint.get("output_dim", score_dim * score_class_count)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        flat_dim=int(checkpoint["flat_dim"]),
        query_sizes=tuple(checkpoint["query_sizes"]),
        num_heads=int(checkpoint["num_heads"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def get_score_columns(checkpoint):
    score_dim = int(checkpoint["score_dim"])
    return list(
        checkpoint.get("score_columns")
        or [f"score_{index + 1}" for index in range(score_dim)]
    )


def print_score_columns(checkpoint_path):
    checkpoint = torch.load(
        resolve_script_path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    for index, column in enumerate(get_score_columns(checkpoint)):
        print(f"{index}\t{column}")


def resolve_score_index(args, score_columns):
    if args.score_column is not None:
        if args.score_column not in score_columns:
            available = ", ".join(score_columns)
            raise ValueError(f"Unknown --score-column '{args.score_column}'. Available: {available}")
        return score_columns.index(args.score_column)
    if args.score_index is None:
        raise ValueError("Pass either --score-column or --score-index.")
    if args.score_index < 0 or args.score_index >= len(score_columns):
        raise ValueError(f"--score-index must be in [0, {len(score_columns) - 1}].")
    return args.score_index


def resolve_class_index(args, logits_for_score):
    predicted_class = int(logits_for_score.argmax().item())
    if args.class_index is not None:
        class_index = args.class_index
    elif args.score_value is not None:
        class_index = args.score_value - 1
    else:
        class_index = predicted_class

    if class_index < 0 or class_index >= logits_for_score.numel():
        raise ValueError(f"Class index must be in [0, {logits_for_score.numel() - 1}].")
    return class_index, predicted_class


def compute_attribution(model, embeddings, score_index, class_index, score_dim, score_class_count, device, method):
    input_tensor = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor).view(1, score_dim, score_class_count)
    selected_logit = logits[0, score_index, class_index]
    selected_logit.backward()

    gradients = input_tensor.grad.detach()[0]
    inputs = input_tensor.detach()[0]
    if method == "grad-norm":
        signed = gradients.norm(dim=-1)
        importance = signed
    else:
        signed = (gradients * inputs).sum(dim=-1)
        importance = (gradients * inputs).abs().sum(dim=-1)

    probabilities = logits.softmax(dim=-1).detach()[0]
    return {
        "logits": logits.detach()[0].cpu(),
        "probabilities": probabilities.cpu(),
        "signed": signed.cpu(),
        "importance": importance.cpu(),
    }


def normalize_scores(scores):
    max_score = float(scores.max().item()) if scores.numel() else 0.0
    if max_score <= 0:
        return [0.0 for _ in scores]
    return [float(value.item()) / max_score for value in scores]


def write_html(output_path, sentences, rows, metadata):
    output_path = resolve_script_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for row in rows:
        alpha = 0.12 + 0.78 * row["normalized"]
        direction = "support" if row["signed"] >= 0 else "oppose"
        color = f"rgba(219, 70, 38, {alpha:.3f})" if row["signed"] >= 0 else f"rgba(37, 99, 235, {alpha:.3f})"
        blocks.append(
            "<section class='sentence {direction}' style='background:{color}'>"
            "<div class='meta'>#{rank} importance={importance:.6g} signed={signed:.6g}</div>"
            "<p>{sentence}</p>"
            "</section>".format(
                direction=direction,
                color=color,
                rank=row["rank"],
                importance=row["importance"],
                signed=row["signed"],
                sentence=html.escape(sentences[row["index"]]),
            )
        )

    probability_rows = "\n".join(
        "<tr><td>{score}</td><td>{probability:.6f}</td></tr>".format(
            score=score,
            probability=probability,
        )
        for score, probability in metadata["probabilities"]
    )

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Backprop Attribution</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 32px;
      color: #161616;
      background: #fafafa;
    }}
    .summary {{
      margin-bottom: 24px;
      max-width: 960px;
    }}
    .summary code {{
      background: #eeeeee;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    table {{
      border-collapse: collapse;
      margin: 12px 0 20px;
    }}
    td, th {{
      border: 1px solid #d8d8d8;
      padding: 5px 8px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    .sentence {{
      border-left: 5px solid #db4626;
      margin: 10px 0;
      padding: 10px 14px;
      max-width: 960px;
      border-radius: 6px;
    }}
    .sentence.oppose {{
      border-left-color: #2563eb;
    }}
    .sentence p {{
      margin: 4px 0 0;
    }}
    .meta {{
      color: #4c4c4c;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="summary">
    <h1>Backprop Attribution</h1>
    <p>Score dimension: <code>{score_column}</code>; target score: <code>{target_score}</code>; predicted score: <code>{predicted_score}</code>; method: <code>{method}</code>.</p>
    <table>
      <thead><tr><th>Score</th><th>Probability</th></tr></thead>
      <tbody>{probability_rows}</tbody>
    </table>
    <p>Red sentences push the selected target logit up; blue sentences push it down. Darker background means larger absolute attribution.</p>
  </div>
  {blocks}
</body>
</html>
""".format(
        score_column=html.escape(metadata["score_column"]),
        target_score=metadata["target_score"],
        predicted_score=metadata["predicted_score"],
        method=html.escape(metadata["method"]),
        probability_rows=probability_rows,
        blocks="\n  ".join(blocks),
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize which input sentences influence one latent-query classifier dimension."
    )
    parser.add_argument("--text", help="Raw text to explain.")
    parser.add_argument("--text-file", type=Path, help="UTF-8 text file to explain.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default="pseudo_text_sentence_embeddings_multi/manifest.json")
    parser.add_argument(
        "--list-score-columns",
        action="store_true",
        help="Print available score dimensions from the checkpoint and exit.",
    )
    parser.add_argument("--score-column")
    parser.add_argument("--score-index", type=int)
    parser.add_argument("--score-value", type=int, choices=range(1, 6), metavar="{1,2,3,4,5}")
    parser.add_argument("--class-index", type=int, help="Zero-based class index. Overrides --score-value.")
    parser.add_argument(
        "--method",
        choices=["grad-times-input", "grad-norm"],
        default="grad-times-input",
    )
    parser.add_argument("--output-html", default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--top-k", type=int, default=0, help="Print only top K rows. Default prints all.")
    parser.add_argument("--device", default=None, help="Model device, for example cuda or cpu.")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-backend", choices=["transformers", "sentence-transformers"])
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--sentence-model-name", default=None)
    parser.add_argument("--sentence-device", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--normalize-embeddings", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_score_columns:
        print_score_columns(args.checkpoint)
        return

    if bool(args.text) == bool(args.text_file):
        raise ValueError("Pass exactly one of --text or --text-file.")

    text = args.text if args.text else resolve_script_path(args.text_file).read_text(encoding="utf-8")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest = load_manifest(args.manifest)
    embed_args = make_embed_args(args, manifest)

    sentences = split_text(text, embed_args)
    embeddings = embed_sentences(sentences, embed_args)
    model, checkpoint = load_model(args.checkpoint, device)

    score_columns = get_score_columns(checkpoint)
    score_index = resolve_score_index(args, score_columns)
    score_dim = int(checkpoint["score_dim"])
    score_class_count = int(checkpoint.get("score_class_count", 5))

    with torch.no_grad():
        preview_inputs = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
        preview_logits = model(preview_inputs).view(1, score_dim, score_class_count)[0, score_index]
    class_index, predicted_class = resolve_class_index(args, preview_logits)

    attribution = compute_attribution(
        model,
        embeddings,
        score_index,
        class_index,
        score_dim,
        score_class_count,
        device,
        args.method,
    )
    normalized = normalize_scores(attribution["importance"])
    rows = []
    for index, (importance, signed, norm) in enumerate(
        zip(attribution["importance"], attribution["signed"], normalized)
    ):
        rows.append(
            {
                "index": index,
                "importance": float(importance.item()),
                "signed": float(signed.item()),
                "normalized": norm,
            }
        )
    rows.sort(key=lambda row: row["importance"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    metadata = {
        "score_column": score_columns[score_index],
        "target_score": class_index + 1,
        "predicted_score": predicted_class + 1,
        "method": args.method,
        "probabilities": [
            (index + 1, float(probability.item()))
            for index, probability in enumerate(attribution["probabilities"][score_index])
        ],
    }
    output_path = write_html(args.output_html, sentences, rows, metadata)

    print(
        f"score_column={metadata['score_column']} "
        f"target_score={metadata['target_score']} "
        f"predicted_score={metadata['predicted_score']} "
        f"sentences={len(sentences)}"
    )
    print(f"html={output_path}")
    print(
        "probabilities="
        + ",".join(
            f"{score}:{probability:.6f}"
            for score, probability in metadata["probabilities"]
        )
    )
    print("rank\timportance\tsigned\tsentence")
    rows_to_print = rows[: args.top_k] if args.top_k > 0 else rows
    for row in rows_to_print:
        sentence = sentences[row["index"]].replace("\t", " ").replace("\n", " ")
        print(f"{row['rank']}\t{row['importance']:.8g}\t{row['signed']:.8g}\t{sentence}")


if __name__ == "__main__":
    main()
