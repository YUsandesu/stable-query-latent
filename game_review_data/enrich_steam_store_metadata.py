"""Fill prepared ``games.json`` records with Steam store-page metadata.

Kaggle's ``andrewmvd/steam-reviews`` table contains reviews plus app ids/names,
but not the store descriptions used by the local review pipeline. This script
updates a prepared ``games.json`` in place using Steam's public appdetails API,
with a JSON cache so interrupted runs can resume without re-fetching.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GAMES_JSON = SCRIPT_DIR / "kaggle_steam_reviews_prepared" / "games.json"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "kaggle_steam_reviews_prepared" / "_steam_appdetails_cache"

META_FIELDS = ("detailed_description", "about_the_game", "short_description")
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"

CATEGORY_ALIASES = {
    "Single-player": "Singleplayer",
    "Multi-player": "Multiplayer",
    "Online PvP": "PvP",
    "LAN PvP": "PvP",
    "Shared/Split Screen PvP": "PvP",
    "Online Co-op": "Online Co-Op",
    "LAN Co-op": "Online Co-Op",
    "Shared/Split Screen Co-op": "Local Co-Op",
    "Cross-Platform Multiplayer": "Multiplayer",
}


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def clean_store_text(value) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"(?is)<\s*(br|p|div|li|h[1-6])\b[^>]*>", ". ", text)
    text = re.sub(r"(?is)<\s*/\s*(p|div|li|h[1-6]|ul|ol)\s*>", ". ", text)
    text = re.sub(r"(?is)<(script|style|video|source|img|span)\b.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\[/?[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.!?]){3,}", r"\1", text)
    return text.strip(" .\t\r\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_namespace(filters: str | None) -> str:
    text = (filters or "full").strip() or "full"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def cache_payload_path(cache_dir: Path, appid: str, language: str, filters: str | None) -> Path:
    return cache_dir / language / cache_namespace(filters) / f"{appid}.json"


def fetch_cached_appdetails(appid: str, args) -> dict | None:
    cache_path = cache_payload_path(args.cache_dir, appid, args.language, args.filters)
    if cache_path.exists() and not args.overwrite_cache:
        return load_json(cache_path)
    return None


def write_cached_appdetails(appid: str, entry: dict, args) -> None:
    cache_path = cache_payload_path(args.cache_dir, appid, args.language, args.filters)
    atomic_json_write({str(appid): entry}, cache_path)


def fetch_appdetails_many(appids: list[str], args) -> tuple[dict[str, dict], int]:
    payloads: dict[str, dict] = {}
    missing = []
    for appid in appids:
        cached = fetch_cached_appdetails(appid, args)
        if cached is None:
            missing.append(appid)
        else:
            payloads[appid] = cached
    if not missing:
        return payloads, 0

    params = {
        "appids": ",".join(missing),
        "l": args.language,
        "cc": args.country,
    }
    if args.filters:
        params["filters"] = args.filters
    last_error = None
    for attempt in range(args.retries):
        try:
            response = requests.get(APPDETAILS_URL, params=params, timeout=args.timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else args.retry_sleep * (attempt + 1) * 3
                time.sleep(wait_seconds)
            response.raise_for_status()
            payload = response.json()
            for appid in missing:
                entry = payload.get(str(appid)) or {"success": False}
                write_cached_appdetails(appid, entry, args)
                payloads[appid] = {str(appid): entry}
            if args.sleep > 0:
                time.sleep(args.sleep)
            return payloads, len(missing)
        except Exception as exc:  # noqa: BLE001 - preserve API error context.
            last_error = exc
            time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(f"Steam appdetails failed for appids={missing[:5]}...: {last_error}")


def extract_app_data(payload: dict, appid: str) -> dict:
    entry = payload.get(str(appid)) or {}
    if not entry.get("success"):
        return {}
    data = entry.get("data")
    return data if isinstance(data, dict) else {}


def tag_names_from_appdetails(data: dict) -> dict[str, float]:
    tags: dict[str, float] = {}
    for item in data.get("genres") or []:
        if isinstance(item, dict) and item.get("description"):
            tags[str(item["description"])] = 1.0
    for item in data.get("categories") or []:
        if not isinstance(item, dict) or not item.get("description"):
            continue
        name = str(item["description"])
        mapped = CATEGORY_ALIASES.get(name)
        if mapped:
            tags[mapped] = 1.0
    return tags


def has_description(record: dict) -> bool:
    return any(str(record.get(field) or "").strip() for field in META_FIELDS)


def enrich_record(appid: str, record: dict, data: dict, overwrite_existing: bool) -> tuple[dict, bool]:
    changed = False
    if data.get("name") and (overwrite_existing or not record.get("name")):
        record["name"] = str(data["name"])
        changed = True
    for field in META_FIELDS:
        cleaned = clean_store_text(data.get(field))
        if cleaned and (overwrite_existing or not str(record.get(field) or "").strip()):
            record[field] = cleaned
            changed = True
    if data.get("steam_appid"):
        record["steam_appid"] = int(data["steam_appid"])
    app_tags = tag_names_from_appdetails(data)
    if app_tags:
        tags = dict(record.get("tags") or {})
        for tag, weight in app_tags.items():
            tags.setdefault(tag, weight)
        record["tags"] = tags
        changed = True
    record.setdefault("name", appid)
    record.setdefault("detailed_description", "")
    record.setdefault("about_the_game", "")
    record.setdefault("short_description", "")
    record.setdefault("tags", {})
    return record, changed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-json", type=Path, default=DEFAULT_GAMES_JSON)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--language", default="english")
    parser.add_argument("--country", default="US")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--filters",
        default="",
        help="Steam appdetails filters. Empty string fetches full payload, including genres/categories.",
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    games = load_json(args.games_json)
    output_json = args.output_json or args.games_json
    appids = sorted(games, key=lambda value: int(value) if str(value).isdigit() else str(value))
    if args.limit > 0:
        appids = appids[: args.limit]

    pending = []
    skipped = 0
    for appid in appids:
        record = games[str(appid)]
        if has_description(record) and record.get("tags") and not args.overwrite_existing:
            skipped += 1
        else:
            pending.append(str(appid))

    # Up-front scan of the local cache: anything already on disk is served
    # without touching the Steam API. Lets you confirm a persisted cache is
    # being reused instead of silently re-fetching.
    cached_hits = sum(
        1 for appid in pending
        if cache_payload_path(args.cache_dir, appid, args.language, args.filters).exists()
        and not args.overwrite_cache
    )
    print(
        f"enrich: games={len(appids)} already_complete={skipped} pending={len(pending)} "
        f"| cache_dir={args.cache_dir} cached={cached_hits} need_api={len(pending) - cached_hits}",
        flush=True,
    )

    fetched = changed = failed = processed = 0
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            payloads, fetched_count = fetch_appdetails_many(batch, args)
            fetched += fetched_count
        except Exception as exc:  # noqa: BLE001 - keep going across delisted/rate-limited batches.
            failed += len(batch)
            print(
                f"[{start + 1}-{start + len(batch)}/{len(pending)}] batch ERROR "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        for appid in batch:
            processed += 1
            payload = payloads.get(appid) or {}
            data = extract_app_data(payload, appid)
            if not data:
                failed += 1
                print(f"[{processed}/{len(pending)}] {appid}: no appdetails data", flush=True)
                continue
            games[appid], did_change = enrich_record(
                appid,
                games[appid],
                data,
                overwrite_existing=args.overwrite_existing,
            )
            changed += int(did_change)
            status = "updated" if did_change else "unchanged"
            print(f"[{processed}/{len(pending)}] {appid}: {status}", flush=True)

        if processed % 25 == 0:
            atomic_json_write(games, output_json)

    atomic_json_write(games, output_json)
    missing_desc = sum(1 for record in games.values() if not has_description(record))
    missing_tags = sum(1 for record in games.values() if not record.get("tags"))
    print(
        f"enriched metadata -> {output_json} fetched={fetched} changed={changed} "
        f"skipped={skipped} failed={failed} missing_desc={missing_desc} missing_tags={missing_tags}",
        flush=True,
    )


if __name__ == "__main__":
    main()
