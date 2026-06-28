"""Stage 2 of the game-review embedding pipeline: split each review into sentences,
keeping the review_id / sentence_id structure.

Reads the per-game JSON arrays produced by build_metadata.py and writes one JSON
object per game:

    { "<review_id>": { "sentence_1": {"sentence_text": ...},
                        "sentence_2": {"sentence_text": ...}, ... }, ... }

review_id is the 0-based index of the review in the source array (so the original
text is recoverable as source[int(review_id)]); reviews that yield no sentence are
omitted. Reviews are fed to the wtpsplit SaT splitter in fixed-size chunks so GPU
memory stays bounded on games with very many reviews. When ``--chunk-budget 0`` is
used, the script inspects total system RAM and selects one of the internal budget
tiers automatically.
"""

import argparse
import ctypes
import json
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = SCRIPT_DIR / "game_review_metadata"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "game_review_sentences"
DEFAULT_MODEL = "sat-3l-sm"
DEFAULT_CHUNK_BUDGET = 0

AUTO_BUDGET_POINTS = (
    (32, 320_000),
    (64, 1_280_000),
    (125, 6_250_000),
    (250, 12_500_000),
)


def replace_with_retry(tmp_path: Path, output_path: Path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            tmp_path.replace(output_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def read_reviews_normalized(input_path: Path):
    with input_path.open("r", encoding="utf-8-sig") as file:
        reviews = json.load(file)
    if isinstance(reviews, list):
        return [normalize_text(review) for review in reviews]
    return reviews


def write_sentence_mapping(output_path: Path, mapping) -> None:
    tmp_path = output_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, ensure_ascii=False)
    replace_with_retry(tmp_path, output_path)


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


def get_total_system_ram_gib() -> float | None:
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.ullTotalPhys / (1024 ** 3)

        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024 ** 3)
    except Exception:
        return None
    return None


def choose_auto_chunk_budget(total_ram_gib: float | None) -> tuple[int, int | None]:
    if total_ram_gib is None:
        return AUTO_BUDGET_POINTS[0][1], None

    ram_gib = max(float(total_ram_gib), float(AUTO_BUDGET_POINTS[0][0]))
    for (left_ram, left_budget), (right_ram, right_budget) in zip(
        AUTO_BUDGET_POINTS, AUTO_BUDGET_POINTS[1:]
    ):
        if ram_gib <= right_ram:
            fraction = (ram_gib - left_ram) / (right_ram - left_ram)
            budget = left_budget + fraction * (right_budget - left_budget)
            return int(round(budget)), int(round(ram_gib))

    left_ram, left_budget = AUTO_BUDGET_POINTS[-2]
    right_ram, right_budget = AUTO_BUDGET_POINTS[-1]
    slope = (right_budget - left_budget) / (right_ram - left_ram)
    budget = right_budget + (ram_gib - right_ram) * slope
    return int(round(budget)), int(round(ram_gib))


def resolve_chunk_budget(chunk_budget):
    if chunk_budget and chunk_budget > 0:
        return int(chunk_budget), None, None

    total_ram_gib = get_total_system_ram_gib()
    resolved_budget, tier_gib = choose_auto_chunk_budget(total_ram_gib)
    return resolved_budget, tier_gib, total_ram_gib


def iter_review_chunks(normalized_reviews, chunk_budget):
    chunk = []
    used = 0
    for review in normalized_reviews:
        cost = max(1, len(review))
        if chunk and used + cost > chunk_budget:
            yield chunk
            chunk = []
            used = 0
        chunk.append(review)
        used += cost
    if chunk:
        yield chunk


def split_reviews_into_mapping(reviews, splitter, device, chunk_budget):
    """Return (mapping, sentence_count). mapping keeps review_id -> sentence_id ->
    {sentence_text}; review_id is the source-array index."""
    normalized = [normalize_text(review) for review in reviews]
    return split_normalized_reviews_into_mapping(
        normalized,
        splitter,
        device,
        chunk_budget,
    )


def split_normalized_reviews_into_mapping(normalized, splitter, device, chunk_budget):
    """Return (mapping, sentence_count) for reviews already normalized."""
    mapping = {}
    sentence_count = 0
    start = 0
    for chunk in iter_review_chunks(normalized, chunk_budget):
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
        start += len(chunk)
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
               chunk_budget=DEFAULT_CHUNK_BUDGET, overwrite=False, splitter=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    if splitter is None:
        splitter = load_splitter(model, device)

    resolved_budget, tier_gib, total_ram_gib = resolve_chunk_budget(chunk_budget)
    if tier_gib is None:
        budget_note = f"chunk_budget={resolved_budget}"
    else:
        budget_note = (
            f"chunk_budget=auto({int(total_ram_gib or 0)}GiB->tier{tier_gib}G,"
            f"budget={resolved_budget})"
        )
    print(
        f"split_data: {len(input_files)} files, model={model}, device={device}, "
        f"{budget_note} -> {output_dir}",
        flush=True,
    )

    skipped_existing = 0
    process_files = []
    for file_index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        process_files.append((file_index, input_path, output_path))

    if skipped_existing:
        print(f"skip existing sentence JSON files: {skipped_existing}", flush=True)

    pending_writes = []

    def finish_oldest_write():
        future, message = pending_writes.pop(0)
        future.result()
        print(message, flush=True)

    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="split-read") as read_pool,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="split-write") as write_pool,
    ):
        read_future = (
            read_pool.submit(read_reviews_normalized, process_files[0][1])
            if process_files
            else None
        )

        for process_index, (file_index, input_path, output_path) in enumerate(process_files):
            reviews = read_future.result()
            next_index = process_index + 1
            read_future = (
                read_pool.submit(read_reviews_normalized, process_files[next_index][1])
                if next_index < len(process_files)
                else None
            )

            if not isinstance(reviews, list):
                print(
                    f"[{file_index}/{len(input_files)}] {input_path.name}: skip (not a list)",
                    flush=True,
                )
                continue

            mapping, sentence_count = split_normalized_reviews_into_mapping(
                reviews,
                splitter,
                device,
                chunk_budget=resolved_budget,
            )

            pending_writes.append(
                (
                    write_pool.submit(write_sentence_mapping, output_path, mapping),
                    (
                        f"[{file_index}/{len(input_files)}] {input_path.name}: "
                        f"{len(reviews)} reviews -> {len(mapping)} non-empty, "
                        f"{sentence_count} sentences"
                    ),
                )
            )

            while len(pending_writes) >= 2:
                finish_oldest_write()

        while pending_writes:
            finish_oldest_write()

    print(f"Done. Sentence JSON written to {output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="wtpsplit SaT model name.")
    parser.add_argument("--device", default=None, help="e.g. 'cuda' or 'cpu'.")
    parser.add_argument(
        "--chunk-budget",
        default=DEFAULT_CHUNK_BUDGET,
        type=int,
        help="Approximate character budget per SaT chunk; 0 auto-selects a tier from total system RAM.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    split_data(
        args.input_dir,
        args.output_dir,
        model=args.model,
        device=args.device,
        chunk_budget=args.chunk_budget,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
