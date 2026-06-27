"""Stage 2 of the game-review embedding pipeline: split each review into sentences,
keeping the review_id / sentence_id structure.

Reads the per-game JSON arrays produced by build_metadata.py and writes one JSON
object per game:

    { "<review_id>": { "sentence_1": {"sentence_text": ...},
                        "sentence_2": {"sentence_text": ...}, ... }, ... }

review_id is the 0-based index of the review in the source array (so the original
text is recoverable as source[int(review_id)]); reviews that yield no sentence are
omitted. Reviews are fed to the wtpsplit SaT splitter in fixed-size chunks so GPU
memory stays bounded on games with very many reviews.
"""

import argparse
import json
import re
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = SCRIPT_DIR / "game_review_metadata"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "game_review_sentences"
DEFAULT_MODEL = "sat-3l-sm"
DEFAULT_CHUNK_SIZE = 2000


def replace_with_retry(tmp_path: Path, output_path: Path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            tmp_path.replace(output_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_splitter(model_name, device):
    from wtpsplit import SaT

    splitter = SaT(model_name)
    if device:
        splitter.to(device)
        if str(device).startswith("cuda"):
            splitter.half()
    return splitter


def _clear_cuda_cache(device):
    if device and str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def split_reviews_into_mapping(reviews, splitter, chunk_size, device):
    """Return (mapping, sentence_count). mapping keeps review_id -> sentence_id ->
    {sentence_text}; review_id is the source-array index."""
    normalized = [normalize_text(review) for review in reviews]

    mapping = {}
    sentence_count = 0
    for start in range(0, len(normalized), chunk_size):
        chunk = normalized[start : start + chunk_size]
        for local_index, review_sentences in enumerate(
            split_chunk_with_fallback(chunk, splitter, start, device)
        ):
            review_id = start + local_index
            sentences = {}
            sid = 0
            for sentence in review_sentences:
                cleaned = sentence.strip()
                if cleaned:
                    sid += 1
                    sentences[f"sentence_{sid}"] = {"sentence_text": cleaned}
            if sentences:
                mapping[str(review_id)] = sentences
                sentence_count += sid
        _clear_cuda_cache(device)
    return mapping, sentence_count


def regex_sentence_fallback(text):
    parts = re.split(r"(?<=[.!?。！？])\s+", normalize_text(text))
    return [part.strip() for part in parts if part.strip()]


def split_chunk_with_fallback(chunk, splitter, start_index, device):
    try:
        return list(splitter.split(chunk))
    except AssertionError as exc:
        reason = f"SaT assertion: {exc}"
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        reason = f"CUDA OOM: {exc}"

    _clear_cuda_cache(device)
    if len(chunk) > 1:
        mid = len(chunk) // 2
        print(
            f"  [warn] split chunk starting review {start_index} failed ({reason}); "
            f"retrying as {mid}+{len(chunk) - mid}",
            flush=True,
        )
        return (
            split_chunk_with_fallback(chunk[:mid], splitter, start_index, device)
            + split_chunk_with_fallback(chunk[mid:], splitter, start_index + mid, device)
        )

    print(
        f"  [warn] review {start_index} failed SaT split ({reason}); using regex fallback",
        flush=True,
    )
    return [regex_sentence_fallback(chunk[0])]


def split_data(input_dir, output_dir, model=DEFAULT_MODEL, device=None,
               chunk_size=DEFAULT_CHUNK_SIZE, overwrite=False, splitter=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    if splitter is None:
        splitter = load_splitter(model, device)

    print(f"split_data: {len(input_files)} files, model={model}, device={device} -> {output_dir}", flush=True)
    skipped_existing = 0
    for file_index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        if skipped_existing:
            print(f"skip existing sentence JSON files: {skipped_existing}", flush=True)
            skipped_existing = 0

        with input_path.open("r", encoding="utf-8") as file:
            reviews = json.load(file)
        if not isinstance(reviews, list):
            print(f"[{file_index}/{len(input_files)}] {input_path.name}: skip (not a list)", flush=True)
            continue

        mapping, sentence_count = split_reviews_into_mapping(reviews, splitter, chunk_size, device)

        tmp_path = output_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(mapping, file, ensure_ascii=False)
        replace_with_retry(tmp_path, output_path)

        print(
            f"[{file_index}/{len(input_files)}] {input_path.name}: "
            f"{len(reviews)} reviews -> {len(mapping)} non-empty, {sentence_count} sentences",
            flush=True,
        )
    if skipped_existing:
        print(f"skip existing sentence JSON files: {skipped_existing}", flush=True)
    print(f"Done. Sentence JSON written to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="wtpsplit SaT model name.")
    parser.add_argument("--device", default=None, help="e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    split_data(
        args.input_dir,
        args.output_dir,
        model=args.model,
        device=args.device,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
