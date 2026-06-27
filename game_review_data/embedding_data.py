"""Stage 3 of the game-review embedding pipeline: embed every sentence and attach
its vector, keeping the review_id / sentence_id structure.

Reads the nested sentence files produced by split_data.py and writes one JSON object
per game where each sentence gains a ``vector`` field:

    { "<review_id>": { "sentence_1": {"sentence_text": ..., "vector": [...]}, ... }, ... }

Two backends (user choice):
    --backend local   local Qwen3-Embedding via transformers (GPU/CPU, last-token pool)
    --backend cloud   remote TEI endpoint (token from tokenAPI.txt, concurrent requests)

Vectors are inserted at their exact review_id/sentence_id position, so the
review <-> sentence <-> vector correspondence is preserved by construction.
"""

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# cloud_embedding.py lives at the project root, one level up from this file.
# Put that on sys.path so the deferred ``from cloud_embedding import ...`` inside
# CloudEmbedder works no matter the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = SCRIPT_DIR / "game_review_sentences"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "game_review_embedded"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def flatten_positions(nested):
    return [(rid, skey) for rid, sentences in nested.items() for skey in sentences]


def _numeric_suffix(value, prefix):
    text = str(value)
    if text.startswith(prefix):
        text = text[len(prefix):]
    try:
        return int(text)
    except ValueError:
        return text


def ordered_review_texts(nested):
    """Flatten a sentences mapping into (texts, review_lengths) in a stable order.

    Reviews are sorted by numeric id, sentences by their ``sentence_<n>`` suffix —
    matching the order the trainer / H5 builder reconstruct, so the npz layout is
    deterministic. ``review_lengths`` lets us split the flat embedding output back
    into per-review groups for save_game_npz.
    """
    texts = []
    review_lengths = []
    for _, sentence_map in sorted(nested.items(), key=lambda kv: _numeric_suffix(kv[0], "")):
        if not isinstance(sentence_map, dict):
            continue
        count = 0
        for _, payload in sorted(
            sentence_map.items(), key=lambda kv: _numeric_suffix(kv[0], "sentence_")
        ):
            text = payload.get("sentence_text") if isinstance(payload, dict) else None
            if text is None:
                continue
            texts.append(text)
            count += 1
        if count:
            review_lengths.append(count)
    return texts, review_lengths


def split_into_reviews(vectors, review_lengths):
    """Split a flat list of sentence vectors back into per-review groups."""
    reviews = []
    cursor = 0
    for length in review_lengths:
        reviews.append(vectors[cursor : cursor + length])
        cursor += length
    return reviews


# --------------------------------------------------------------------------- local
class LocalEmbedder:
    """Local Qwen3-Embedding via transformers with last-token pooling."""

    def __init__(self, model_name, device=None, max_length=2048, batch_size=32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @staticmethod
    def _last_token_pool(last_hidden, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        import torch

        return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]

    def embed(self, texts):
        torch = self.torch
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
                vectors = self._last_token_pool(hidden, tokens["attention_mask"])
            out.extend(vectors.cpu().float().tolist())
        return out


# --------------------------------------------------------------------------- cloud
class CloudEmbedder:
    """Remote TEI endpoint via the robust client from cloud_embedding.py."""

    def __init__(self, base_url=None, token_file=None, concurrency=256, batch_size=32,
                 max_in_flight=None, normalize=False):
        from cloud_embedding import (
            DEFAULT_TOKEN_FILE,
            EmbeddingClient,
            chunked,
            load_credentials,
        )

        self.chunked = chunked
        creds = load_credentials(token_file or DEFAULT_TOKEN_FILE)
        # URL and token both come from the credentials file by default; ``base_url``
        # only overrides the URL key when explicitly passed.
        self.client = EmbeddingClient(base_url or creds["url"], creds["token"], normalize=normalize)
        self.batch_size = batch_size
        self.max_in_flight = max_in_flight or concurrency
        self.executor = ThreadPoolExecutor(max_workers=concurrency)

    def embed(self, texts):
        batches = [batch for _, batch in self.chunked(texts, self.batch_size)]
        n = len(batches)
        results = [None] * n
        in_flight = {}
        next_submit = 0

        def fill():
            nonlocal next_submit
            while next_submit < n and len(in_flight) < self.max_in_flight:
                future = self.executor.submit(self.client.embed_batch, batches[next_submit])
                in_flight[future] = next_submit
                next_submit += 1

        fill()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                results[in_flight.pop(future)] = future.result()
            fill()

        vectors = []
        for batch_vectors in results:
            vectors.extend(batch_vectors)
        return vectors

    def close(self):
        self.executor.shutdown(wait=True)


def embed_data(input_dir, output_dir, backend, overwrite=False, output_format="npz", **kwargs):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format not in ("npz", "json"):
        raise ValueError(f"output_format must be 'npz' or 'json', got {output_format!r}")
    out_suffix = ".npz" if output_format == "npz" else ".json"
    if output_format == "npz":
        sys.path.insert(0, str(_PROJECT_ROOT))
        from game_npz import save_game_npz

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    if backend == "local":
        embedder = LocalEmbedder(
            kwargs.get("local_model", DEFAULT_LOCAL_MODEL),
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 32),
        )
        closer = None
    else:
        embedder = CloudEmbedder(
            base_url=kwargs.get("base_url"),
            token_file=kwargs.get("token_file"),
            concurrency=kwargs.get("concurrency", 256),
            batch_size=kwargs.get("batch_size", 32),
            max_in_flight=kwargs.get("max_in_flight"),
            normalize=kwargs.get("normalize", False),
        )
        closer = embedder.close

    print(f"embed_data: {len(input_files)} files, backend={backend} format={output_format} -> {output_dir}")
    try:
        for file_index, input_path in enumerate(input_files, start=1):
            output_path = (output_dir / input_path.name).with_suffix(out_suffix)
            if output_path.exists() and not overwrite:
                print(f"[{file_index}/{len(input_files)}] {input_path.name}: skip (exists)")
                continue

            with input_path.open("r", encoding="utf-8") as file:
                nested = json.load(file)

            if output_format == "npz":
                texts, review_lengths = ordered_review_texts(nested)
                if not texts:
                    save_game_npz(output_path, [])
                    continue
                started = time.time()
                vectors = embedder.embed(texts)
                reviews = split_into_reviews(vectors, review_lengths)
                save_game_npz(output_path, reviews)
                dim = len(vectors[0]) if vectors else 0
                print(
                    f"[{file_index}/{len(input_files)}] {input_path.name}: "
                    f"{len(texts)} sentences, {len(review_lengths)} reviews, dim {dim} "
                    f"in {time.time() - started:.1f}s -> {output_path.name}"
                )
                continue

            # output_format == "json": keep the nested {rid:{skey:{text,vector}}} layout.
            positions = flatten_positions(nested)
            if not positions:
                output_path.write_text("{}", encoding="utf-8")
                continue
            texts = [nested[rid][skey]["sentence_text"] for rid, skey in positions]
            started = time.time()
            vectors = embedder.embed(texts)
            for (rid, skey), vector in zip(positions, vectors):
                nested[rid][skey]["vector"] = vector
            tmp_path = output_path.with_suffix(".json.tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as file:
                    json.dump(nested, file, ensure_ascii=False)
                tmp_path.replace(output_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
            dim = len(vectors[0]) if vectors else 0
            print(
                f"[{file_index}/{len(input_files)}] {input_path.name}: "
                f"{len(texts)} sentences, dim {dim} in {time.time() - started:.1f}s"
            )
    finally:
        if closer:
            closer()
    print(f"Done. Embedded {output_format} written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-format", choices=["npz", "json"], default="npz",
                        help="npz: compact fp16 vectors + review_offsets (default, ~10x smaller). "
                             "json: legacy nested {review:{sentence:{text,vector}}}.")
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-in-flight", default=None, type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    embed_data(
        args.input_dir,
        args.output_dir,
        args.backend,
        overwrite=args.overwrite,
        output_format=args.output_format,
        local_model=args.local_model,
        device=args.device,
        base_url=args.base_url,
        token_file=args.token_file,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_in_flight=args.max_in_flight,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
