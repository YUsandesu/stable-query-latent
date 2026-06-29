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
from collections import deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = SCRIPT_DIR / "game_review_metadata"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "game_review_sentences"
DEFAULT_MODEL = "sat-3l-sm"
DEFAULT_CHUNK_BUDGET = 0
DEFAULT_SPLIT_BATCH_SIZE = 32
DEFAULT_SPLIT_OUTER_BATCH_SIZE = 1000
DEFAULT_PREFETCH_RAM_TARGET = 0.50
DEFAULT_PREFETCH_MIN_RAM_GIB = 100
DEFAULT_PREFETCH_MAX_FILES = 0
DEFAULT_PREFETCH_WORKERS = 2

AUTO_BUDGET_POINTS = (
    (32, 320_000),
    (64, 1_280_000),
    (125, 6_250_000),
    (250, 12_500_000),
)

AUTO_BATCH_POINTS = (
    (4, 32),
    (8, 64),
    (16, 128),
    (24, 192),
    (48, 384),
)

AUTO_OUTER_BATCH_POINTS = (
    (64, 1000),
    (125, 4000),
    (250, 8000),
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


def get_cuda_total_memory_gib(device) -> float | None:
    if not device or not str(device).startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device_text = str(device)
        if ":" in device_text:
            device_index = int(device_text.split(":", 1)[1])
        else:
            device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return None


def _read_cgroup_int(path: str) -> int | None:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text or text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def get_cgroup_memory() -> tuple[int, int] | None:
    limit = _read_cgroup_int("/sys/fs/cgroup/memory.max")
    used = _read_cgroup_int("/sys/fs/cgroup/memory.current")
    if limit is not None and used is not None:
        return used, limit

    limit = _read_cgroup_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    used = _read_cgroup_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit is not None and used is not None:
        return used, limit

    return None


def get_host_total_memory_bytes() -> int | None:
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
            return int(status.ullTotalPhys)

        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages * page_size)
    except Exception:
        return None
    return None


def get_total_system_ram_gib() -> float | None:
    cgroup = get_cgroup_memory()
    host_total = get_host_total_memory_bytes()
    if cgroup is not None:
        _used, limit = cgroup
        if host_total is None or limit < host_total:
            return limit / (1024 ** 3)

    if host_total is not None:
        return host_total / (1024 ** 3)

    return None


def get_system_memory_usage() -> tuple[float, float, float] | None:
    cgroup = get_cgroup_memory()
    host_total = get_host_total_memory_bytes()
    if cgroup is not None:
        used, limit = cgroup
        if host_total is None or limit < host_total:
            used = max(0, min(used, limit))
            return used / limit, used / (1024 ** 3), limit / (1024 ** 3)

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
            total = status.ullTotalPhys
            available = status.ullAvailPhys
            used = max(0, total - available)
            return used / total, used / (1024 ** 3), total / (1024 ** 3)

        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and available_pages >= 0 and page_size > 0:
                total = pages * page_size
                available = available_pages * page_size
                used = max(0, total - available)
                return used / total, used / (1024 ** 3), total / (1024 ** 3)
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


def resolve_prefetch_settings(
    total_ram_gib: float | None,
    prefetch_ram_target: float | None,
    prefetch_max_files: int | None,
    prefetch_workers: int | None,
):
    enabled_by_ram = total_ram_gib is not None and total_ram_gib >= DEFAULT_PREFETCH_MIN_RAM_GIB

    if prefetch_ram_target is None:
        prefetch_ram_target = DEFAULT_PREFETCH_RAM_TARGET if enabled_by_ram else 0.0
    if prefetch_max_files is None:
        prefetch_max_files = DEFAULT_PREFETCH_MAX_FILES if enabled_by_ram else 1
    if prefetch_workers is None:
        prefetch_workers = DEFAULT_PREFETCH_WORKERS if enabled_by_ram else 1

    prefetch_ram_target = max(0.0, min(0.95, float(prefetch_ram_target)))
    prefetch_max_files = max(0, int(prefetch_max_files))
    prefetch_workers = max(1, int(prefetch_workers))
    return prefetch_ram_target, prefetch_max_files, prefetch_workers, enabled_by_ram


def choose_auto_batch_size(total_vram_gib: float | None) -> tuple[int, int | None]:
    if total_vram_gib is None:
        return DEFAULT_SPLIT_BATCH_SIZE, None

    vram_gib = max(float(total_vram_gib), float(AUTO_BATCH_POINTS[0][0]))
    for (left_vram, left_batch), (right_vram, right_batch) in zip(
        AUTO_BATCH_POINTS, AUTO_BATCH_POINTS[1:]
    ):
        if vram_gib <= right_vram:
            fraction = (vram_gib - left_vram) / (right_vram - left_vram)
            batch = left_batch + fraction * (right_batch - left_batch)
            return max(1, int(round(batch))), int(round(vram_gib))

    left_vram, left_batch = AUTO_BATCH_POINTS[-2]
    right_vram, right_batch = AUTO_BATCH_POINTS[-1]
    slope = (right_batch - left_batch) / (right_vram - left_vram)
    batch = right_batch + (vram_gib - right_vram) * slope
    return max(1, int(round(batch))), int(round(vram_gib))


def choose_auto_outer_batch_size(total_ram_gib: float | None) -> tuple[int, int | None]:
    if total_ram_gib is None:
        return DEFAULT_SPLIT_OUTER_BATCH_SIZE, None

    ram_gib = max(float(total_ram_gib), float(AUTO_OUTER_BATCH_POINTS[0][0]))
    for (left_ram, left_batch), (right_ram, right_batch) in zip(
        AUTO_OUTER_BATCH_POINTS, AUTO_OUTER_BATCH_POINTS[1:]
    ):
        if ram_gib <= right_ram:
            fraction = (ram_gib - left_ram) / (right_ram - left_ram)
            batch = left_batch + fraction * (right_batch - left_batch)
            return max(1, int(round(batch))), int(round(ram_gib))

    left_ram, left_batch = AUTO_OUTER_BATCH_POINTS[-2]
    right_ram, right_batch = AUTO_OUTER_BATCH_POINTS[-1]
    slope = (right_batch - left_batch) / (right_ram - left_ram)
    batch = right_batch + (ram_gib - right_ram) * slope
    return max(1, int(round(batch))), int(round(ram_gib))


def resolve_batch_size(batch_size: int | None, device) -> tuple[int, int | None]:
    if batch_size and batch_size > 0:
        return int(batch_size), None

    total_vram_gib = get_cuda_total_memory_gib(device)
    resolved_batch_size, vram_tier_gib = choose_auto_batch_size(total_vram_gib)
    return resolved_batch_size, vram_tier_gib


def resolve_outer_batch_size(
    outer_batch_size: int | None,
    total_ram_gib: float | None,
) -> tuple[int, int | None]:
    if outer_batch_size and outer_batch_size > 0:
        return int(outer_batch_size), None

    resolved_outer_batch_size, ram_tier_gib = choose_auto_outer_batch_size(total_ram_gib)
    return resolved_outer_batch_size, ram_tier_gib


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
        batch_size=DEFAULT_SPLIT_BATCH_SIZE,
        outer_batch_size=DEFAULT_SPLIT_OUTER_BATCH_SIZE,
    )


def split_normalized_reviews_into_mapping(
    normalized,
    splitter,
    device,
    chunk_budget,
    batch_size=DEFAULT_SPLIT_BATCH_SIZE,
    outer_batch_size=DEFAULT_SPLIT_OUTER_BATCH_SIZE,
):
    """Return (mapping, sentence_count) for reviews already normalized."""
    mapping = {}
    sentence_count = 0
    start = 0
    for chunk in iter_review_chunks(normalized, chunk_budget):
        for local_index, review_sentences in enumerate(
            split_chunk_with_fallback(
                chunk,
                splitter,
                start,
                device,
                batch_size=batch_size,
                outer_batch_size=outer_batch_size,
            )
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


def split_chunk_with_fallback(
    chunk,
    splitter,
    start_index,
    device,
    batch_size=DEFAULT_SPLIT_BATCH_SIZE,
    outer_batch_size=DEFAULT_SPLIT_OUTER_BATCH_SIZE,
):
    oom_reason = None
    try:
        return list(
            splitter.split(
                chunk,
                batch_size=batch_size,
                outer_batch_size=outer_batch_size,
            )
        )
    except AssertionError as exc:
        reason = f"SaT assertion: {exc}"
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        reason = f"CUDA OOM: {exc}"
        oom_reason = reason

    _clear_cuda_cache(device)
    if oom_reason and batch_size > DEFAULT_SPLIT_BATCH_SIZE:
        reduced_batch_size = max(DEFAULT_SPLIT_BATCH_SIZE, batch_size // 2)
        print(
            f"  [warn] split chunk starting review {start_index} failed ({oom_reason}); "
            f"retrying with batch_size={reduced_batch_size}",
            flush=True,
        )
        return split_chunk_with_fallback(
            chunk,
            splitter,
            start_index,
            device,
            batch_size=reduced_batch_size,
            outer_batch_size=outer_batch_size,
        )

    if len(chunk) > 1:
        mid = len(chunk) // 2
        print(
            f"  [warn] split chunk starting review {start_index} failed ({reason}); "
            f"retrying as {mid}+{len(chunk) - mid}",
            flush=True,
        )
        return (
            split_chunk_with_fallback(
                chunk[:mid],
                splitter,
                start_index,
                device,
                batch_size=batch_size,
                outer_batch_size=outer_batch_size,
            )
            + split_chunk_with_fallback(
                chunk[mid:],
                splitter,
                start_index + mid,
                device,
                batch_size=batch_size,
                outer_batch_size=outer_batch_size,
            )
        )

    print(
        f"  [warn] review {start_index} failed SaT split ({reason}); using regex fallback",
        flush=True,
    )
    return [regex_sentence_fallback(chunk[0])]


def split_data(
    input_dir,
    output_dir,
    model=DEFAULT_MODEL,
    device=None,
    chunk_budget=DEFAULT_CHUNK_BUDGET,
    overwrite=False,
    splitter=None,
    batch_size=None,
    outer_batch_size=None,
    prefetch_ram_target=None,
    prefetch_max_files=None,
    prefetch_workers=None,
    shard_count=1,
    shard_index=0,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    total_files = len(input_files)
    if shard_count > 1:
        input_files = [
            path for index, path in enumerate(input_files) if index % shard_count == shard_index
        ]

    if splitter is None:
        splitter = load_splitter(model, device)

    resolved_budget, tier_gib, total_ram_gib = resolve_chunk_budget(chunk_budget)
    resolved_batch_size, vram_tier_gib = resolve_batch_size(batch_size, device)
    resolved_outer_batch_size, outer_ram_tier_gib = resolve_outer_batch_size(
        outer_batch_size,
        total_ram_gib,
    )
    (
        prefetch_ram_target,
        prefetch_max_files,
        prefetch_workers,
        prefetch_enabled_by_ram,
    ) = resolve_prefetch_settings(
        total_ram_gib,
        prefetch_ram_target,
        prefetch_max_files,
        prefetch_workers,
    )
    if tier_gib is None:
        budget_note = f"chunk_budget={resolved_budget}"
    else:
        budget_note = (
            f"chunk_budget=auto({int(total_ram_gib or 0)}GiB->tier{tier_gib}G,"
            f"budget={resolved_budget})"
        )
    if batch_size and batch_size > 0:
        batch_note = f"batch_size={resolved_batch_size}"
    elif vram_tier_gib is None:
        batch_note = f"batch_size=auto(default {resolved_batch_size})"
    else:
        batch_note = f"batch_size=auto({vram_tier_gib}GiB VRAM->{resolved_batch_size})"
    if outer_batch_size and outer_batch_size > 0:
        outer_batch_note = f"outer_batch_size={resolved_outer_batch_size}"
    elif outer_ram_tier_gib is None:
        outer_batch_note = f"outer_batch_size=auto(default {resolved_outer_batch_size})"
    else:
        outer_batch_note = (
            f"outer_batch_size=auto({outer_ram_tier_gib}GiB RAM->"
            f"{resolved_outer_batch_size})"
        )
    print(
        f"split_data: {len(input_files)}/{total_files} files, "
        f"shard={shard_index + 1}/{shard_count}, model={model}, device={device}, "
        f"{budget_note}, {batch_note}, "
        f"{outer_batch_note}, "
        f"prefetch_ram_target={prefetch_ram_target:.0%}, "
        f"prefetch_max_files={prefetch_max_files or 'unlimited'}, "
        f"prefetch_workers={prefetch_workers}, "
        f"prefetch_auto_ram={'on' if prefetch_enabled_by_ram else 'off'} "
        f"-> {output_dir}",
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
        ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="split-read") as read_pool,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="split-write") as write_pool,
    ):
        pending_reads = {}
        ready_reads = deque()
        next_prefetch_index = 0

        def below_prefetch_ram_target() -> bool:
            usage = get_system_memory_usage()
            return usage is None or usage[0] < prefetch_ram_target

        def pump_reads(force_one=False):
            nonlocal next_prefetch_index
            while (
                next_prefetch_index < len(process_files)
                and len(pending_reads) < prefetch_workers
                and (
                    prefetch_max_files <= 0
                    or len(pending_reads) + len(ready_reads) < prefetch_max_files
                )
                and (force_one or below_prefetch_ram_target())
            ):
                item = process_files[next_prefetch_index]
                pending_reads[read_pool.submit(read_reviews_normalized, item[1])] = item
                next_prefetch_index += 1
                force_one = False

        def collect_ready_reads():
            for future, item in list(pending_reads.items()):
                if future.done():
                    ready_reads.append((item, future))
                    del pending_reads[future]

        def next_read_result():
            while not ready_reads:
                collect_ready_reads()
                pump_reads(force_one=not pending_reads)
                if not ready_reads and pending_reads:
                    time.sleep(0.01)
            item, future = ready_reads.popleft()
            return item, future.result()

        pump_reads(force_one=True)

        for _ in range(len(process_files)):
            (file_index, input_path, output_path), reviews = next_read_result()
            collect_ready_reads()
            pump_reads()

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
                batch_size=resolved_batch_size,
                outer_batch_size=resolved_outer_batch_size,
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
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Internal wtpsplit SaT inference batch size. Default: auto from CUDA VRAM.",
    )
    parser.add_argument(
        "--outer-batch-size",
        default=None,
        type=int,
        help="Internal wtpsplit SaT outer text batch size. Default: auto from system RAM.",
    )
    parser.add_argument(
        "--prefetch-ram-target",
        default=None,
        type=float,
        help=(
            "Keep pre-reading metadata JSON files while system RAM usage is below "
            "this fraction. Default: auto, enabled only on >=100GiB RAM."
        ),
    )
    parser.add_argument(
        "--prefetch-max-files",
        default=None,
        type=int,
        help="Maximum normalized metadata JSON files queued ahead of splitting. Default: auto.",
    )
    parser.add_argument(
        "--prefetch-workers",
        default=None,
        type=int,
        help="Reader threads for split-stage metadata prefetch. Default: auto.",
    )
    parser.add_argument("--shard-count", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
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
        batch_size=args.batch_size,
        outer_batch_size=args.outer_batch_size,
        prefetch_ram_target=args.prefetch_ram_target,
        prefetch_max_files=args.prefetch_max_files,
        prefetch_workers=args.prefetch_workers,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    main()
