#!/usr/bin/env python3
"""Download per-game benchmark data from the PXI Dash app.

The site is a Dash application. It does not expose static download links; the
benchmark values are returned by Dash callbacks that populate the two box plots.
This script calls those callbacks directly and saves one JSON file per game,
plus CSV files that are convenient for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://protected-ridge-17548.herokuapp.com/"
DEFAULT_GENDERS = ["m", "f", "b", "n", "o"]
# Output goes to ../PXIbenchmark_data/ alongside this build/ dir.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = str(SCRIPT_DIR.parent / "PXIbenchmark_data")


def post_json(url: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "pxi-benchmark-downloader/1.0",
    }
    request = Request(url, data=data, headers=headers, method="POST")

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"POST {url} failed after {retries} attempts: {exc}") from exc
            time.sleep(1.5 * attempt)

    raise RuntimeError("unreachable")


def get_json(url: str, retries: int = 3) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "pxi-benchmark-downloader/1.0",
        },
    )
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"GET {url} failed after {retries} attempts: {exc}") from exc
            time.sleep(1.5 * attempt)

    raise RuntimeError("unreachable")


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s.-]", "", text, flags=re.UNICODE)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:90] or "game"


def column_key(*parts: str) -> str:
    raw = "_".join(part for part in parts if part)
    key = re.sub(r"[^\w]+", "_", raw.lower(), flags=re.UNICODE)
    key = re.sub(r"_+", "_", key).strip("_")
    return key or "value"


def walk_components(node: Any):
    if isinstance(node, dict):
        yield node
        props = node.get("props", {})
        children = props.get("children")
        if isinstance(children, list):
            for child in children:
                yield from walk_components(child)
        elif children is not None:
            yield from walk_components(children)


def extract_options(layout: dict[str, Any], component_id: str) -> list[dict[str, str]]:
    for component in walk_components(layout):
        props = component.get("props", {})
        if props.get("id") == component_id:
            return [
                {"label": str(option["label"]), "value": str(option["value"])}
                for option in props.get("options", [])
            ]
    raise RuntimeError(f"Could not find component {component_id!r} in Dash layout")


def normalize_mapping(options: list[dict[str, str]], id_field: str, name_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: dict[str, int] = {}

    for index, option in enumerate(options, start=1):
        item_id = option["value"].strip()
        item_name = option["label"].strip()
        seen_names[item_name] = seen_names.get(item_name, 0) + 1
        rows.append(
            {
                "source_order": index,
                id_field: item_id,
                name_field: item_name,
                "is_duplicate_id": item_id in seen_ids,
                "is_duplicate_name": False,
                "is_numeric_name": item_name.isdigit(),
            }
        )
        seen_ids.add(item_id)

    for row in rows:
        row["is_duplicate_name"] = seen_names[row[name_field]] > 1

    return rows


def dash_payload(output: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "output": output,
        "inputs": inputs,
        "changedPropIds": [f"{item['id']}.{item['property']}" for item in inputs],
    }


def call_dash(base_url: str, output: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    update_url = urljoin(base_url, "/_dash-update-component")
    return post_json(update_url, dash_payload(output, inputs))


def build_game_genre_mapping(
    base_url: str,
    game_mapping: list[dict[str, Any]],
    genre_mapping: list[dict[str, Any]],
    genders: list[str],
    age_range: list[int],
) -> list[dict[str, Any]]:
    games_by_id = {row["game_id"]: row for row in game_mapping}
    rows: list[dict[str, Any]] = []

    for genre in genre_mapping:
        response = call_dash(
            base_url,
            "game_list.value",
            [
                {"id": "genre_list", "property": "value", "value": [genre["genre_id"]]},
                {"id": "gender_selector", "property": "value", "value": genders},
                {"id": "year_slider", "property": "value", "value": age_range},
                {"id": "game_selector", "property": "value", "value": "custom"},
            ],
        )
        game_ids = response["response"]["game_list"]["value"]
        for order_in_genre, game_id in enumerate(game_ids, start=1):
            game = games_by_id.get(game_id, {})
            rows.append(
                {
                    "genre_id": genre["genre_id"],
                    "genre_name": genre["genre_name"],
                    "order_in_genre": order_in_genre,
                    "game_id": game_id,
                    "game_name": game.get("game_name", ""),
                    "game_source_order": game.get("source_order", ""),
                }
            )

    return rows


def build_game_catalog(
    game_mapping: list[dict[str, Any]],
    game_genre_mapping: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    genres_by_game: dict[str, list[dict[str, Any]]] = {}
    for row in game_genre_mapping:
        genres_by_game.setdefault(row["game_id"], []).append(row)

    catalog: list[dict[str, Any]] = []
    for game in game_mapping:
        genre_rows = sorted(
            genres_by_game.get(game["game_id"], []),
            key=lambda item: (item["genre_id"], item["order_in_genre"]),
        )
        genre_ids = [row["genre_id"] for row in genre_rows]
        genre_names = [row["genre_name"] for row in genre_rows]
        catalog.append(
            {
                "source_order": game["source_order"],
                "game_id": game["game_id"],
                "game_name": game["game_name"],
                "genre_count": len(genre_rows),
                "genre_ids": "|".join(genre_ids),
                "genre_names": "|".join(genre_names),
                "is_duplicate_id": game["is_duplicate_id"],
                "is_duplicate_name": game["is_duplicate_name"],
                "is_numeric_name": game["is_numeric_name"],
            }
        )

    return catalog


def metric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
            "stdev": "",
        }

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def extract_metric_rows(
    game: dict[str, str],
    catalog_by_game_id: dict[str, dict[str, Any]],
    figure_kind: str,
    figure: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    catalog = catalog_by_game_id.get(game["value"], {})
    mapped_game_name = catalog.get("game_name", game["label"])
    genre_ids = catalog.get("genre_ids", "")
    genre_names = catalog.get("genre_names", "")

    for trace in figure.get("data", []):
        metric = trace.get("name", "")
        values = [float(value) for value in trace.get("y", []) if value is not None]
        stats = metric_stats(values)
        summary_rows.append(
            {
                "game_id": game["value"],
                "game_name": mapped_game_name,
                "genre_ids": genre_ids,
                "genre_names": genre_names,
                "benchmark_group": figure_kind,
                "metric": metric,
                **stats,
                "values_json": json.dumps(values, ensure_ascii=False),
            }
        )
        for index, value in enumerate(values, start=1):
            value_rows.append(
                {
                    "game_id": game["value"],
                    "game_name": mapped_game_name,
                    "genre_ids": genre_ids,
                    "genre_names": genre_names,
                    "benchmark_group": figure_kind,
                    "metric": metric,
                    "sample_index": index,
                    "value": value,
                }
            )

    return summary_rows, value_rows


def aggregate_fields(record: dict[str, Any]) -> dict[str, Any]:
    values = record.get("aggregate", [])
    return {
        "participant_count": values[0] if len(values) > 0 else "",
        "selected_game_count": values[1] if len(values) > 1 else "",
        "selected_genre_count": values[2] if len(values) > 2 else "",
    }


def trace_values_by_metric(figure_kind: str, figure: dict[str, Any]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for trace in figure.get("data", []):
        metric = str(trace.get("name", ""))
        key = column_key(figure_kind, metric)
        metrics[key] = [float(value) for value in trace.get("y", []) if value is not None]
    return metrics


def build_wide_value_rows(
    game: dict[str, str],
    catalog_by_game_id: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    catalog = catalog_by_game_id.get(game["value"], {})
    metric_values = {
        **trace_values_by_metric("psychological", record["psychological"]),
        **trace_values_by_metric("functional", record["functional"]),
    }
    max_samples = max((len(values) for values in metric_values.values()), default=0)

    rows: list[dict[str, Any]] = []
    for sample_index in range(max_samples):
        row = {
            "game_id": game["value"],
            "game_name": catalog.get("game_name", game["label"]),
            "game_title": catalog.get("game_name", game["label"]),
            "game_source_order": catalog.get("source_order", game.get("source_order", "")),
            "is_numeric_game_title": catalog.get("is_numeric_name", ""),
            "genre_ids": catalog.get("genre_ids", ""),
            "genre_names": catalog.get("genre_names", ""),
            "sample_index": sample_index + 1,
        }
        for metric, values in metric_values.items():
            row[metric] = values[sample_index] if sample_index < len(values) else ""
        rows.append(row)

    return rows


def build_final_value_rows(
    game: dict[str, str],
    game_genre_mapping: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    genre_rows = [row for row in game_genre_mapping if row["game_id"] == game["value"]]
    if not genre_rows:
        genre_rows = [
            {
                "genre_id": "",
                "genre_name": "",
                "order_in_genre": "",
                "game_id": game["value"],
                "game_name": game["label"],
                "game_source_order": game.get("source_order", ""),
            }
        ]

    metric_values = {
        **trace_values_by_metric("psychological", record["psychological"]),
        **trace_values_by_metric("functional", record["functional"]),
    }
    max_samples = max((len(values) for values in metric_values.values()), default=0)

    rows: list[dict[str, Any]] = []
    for genre in genre_rows:
        for sample_index in range(max_samples):
            row = {
                "game_id": game["value"],
                "game_name": genre.get("game_name", game["label"]),
                "game_title": genre.get("game_name", game["label"]),
                "game_source_order": genre.get("game_source_order", game.get("source_order", "")),
                "is_numeric_game_title": str(genre.get("game_name", game["label"])).isdigit(),
                "genre_id": genre.get("genre_id", ""),
                "genre_name": genre.get("genre_name", ""),
                "sample_index": sample_index + 1,
            }
            for metric, values in metric_values.items():
                row[metric] = values[sample_index] if sample_index < len(values) else ""
            rows.append(row)

    return rows


def build_wide_summary_row(
    game: dict[str, str],
    catalog_by_game_id: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    catalog = catalog_by_game_id.get(game["value"], {})
    row = {
        "game_id": game["value"],
        "game_name": catalog.get("game_name", game["label"]),
        "game_title": catalog.get("game_name", game["label"]),
        "game_source_order": catalog.get("source_order", game.get("source_order", "")),
        "is_numeric_game_title": catalog.get("is_numeric_name", ""),
        "genre_ids": catalog.get("genre_ids", ""),
        "genre_names": catalog.get("genre_names", ""),
        **aggregate_fields(record),
    }

    metric_values = {
        **trace_values_by_metric("psychological", record["psychological"]),
        **trace_values_by_metric("functional", record["functional"]),
    }
    for metric, values in metric_values.items():
        stats = metric_stats(values)
        for stat_name in ["n", "mean", "median", "min", "max", "stdev"]:
            row[f"{metric}_{stat_name}"] = stats[stat_name]
        row[f"{metric}_values"] = json.dumps(values, ensure_ascii=False)

    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def ordered_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for field in preferred:
        if field not in seen:
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def download_game(
    base_url: str,
    game: dict[str, str],
    genres: list[str],
    genders: list[str],
    age_range: list[int],
) -> dict[str, Any]:
    aggregate_inputs = [
        {"id": "genre_list", "property": "value", "value": genres},
        {"id": "gender_selector", "property": "value", "value": genders},
        {"id": "year_slider", "property": "value", "value": age_range},
        {"id": "game_list", "property": "value", "value": [game["value"]]},
    ]
    figure_inputs = [
        {"id": "gender_selector", "property": "value", "value": genders},
        {"id": "game_list", "property": "value", "value": [game["value"]]},
        {"id": "year_slider", "property": "value", "value": age_range},
        {"id": "genre_list", "property": "value", "value": genres},
    ]

    aggregate = call_dash(base_url, "aggregate_data.data", aggregate_inputs)
    psychological = call_dash(base_url, "psyc_graph.figure", figure_inputs)
    functional = call_dash(base_url, "func_graph.figure", figure_inputs)

    return {
        "game": game,
        "filters": {
            "genres": genres,
            "genders": genders,
            "age_range": age_range,
        },
        "aggregate": aggregate["response"]["aggregate_data"]["data"],
        "psychological": psychological["response"]["psyc_graph"]["figure"],
        "functional": functional["response"]["func_graph"]["figure"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download benchmark data for every game in the PXI Dash app."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--limit", type=int, help="Only download the first N games")
    parser.add_argument("--game-id", action="append", help="Download only this game id; repeatable")
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Only write game/genre mapping files; do not download benchmarks",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing JSON files")
    parser.add_argument("--sleep", type=float, default=0.15, help="Pause between games in seconds")
    parser.add_argument("--age-min", type=int, default=15)
    parser.add_argument("--age-max", type=int, default=79)
    parser.add_argument(
        "--genders",
        nargs="+",
        default=DEFAULT_GENDERS,
        help="Gender codes to include, default: m f b n o",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/") + "/"
    out_dir = Path(args.out)
    raw_dir = out_dir / "json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    layout = get_json(urljoin(base_url, "/_dash-layout"))
    game_mapping = normalize_mapping(extract_options(layout, "game_list"), "game_id", "game_name")
    genre_mapping = normalize_mapping(extract_options(layout, "genre_list"), "genre_id", "genre_name")
    games = [
        {"value": row["game_id"], "label": row["game_name"], "source_order": row["source_order"]}
        for row in game_mapping
    ]
    genres = [row["genre_id"] for row in genre_mapping]

    write_json(out_dir / "game_mapping.json", game_mapping)
    write_csv(
        out_dir / "game_mapping.csv",
        game_mapping,
        [
            "source_order",
            "game_id",
            "game_name",
            "is_duplicate_id",
            "is_duplicate_name",
            "is_numeric_name",
        ],
    )
    write_json(out_dir / "genre_mapping.json", genre_mapping)
    write_csv(
        out_dir / "genre_mapping.csv",
        genre_mapping,
        [
            "source_order",
            "genre_id",
            "genre_name",
            "is_duplicate_id",
            "is_duplicate_name",
            "is_numeric_name",
        ],
    )
    game_genre_mapping = build_game_genre_mapping(
        base_url=base_url,
        game_mapping=game_mapping,
        genre_mapping=genre_mapping,
        genders=args.genders,
        age_range=[args.age_min, args.age_max],
    )
    write_json(out_dir / "game_genre_mapping.json", game_genre_mapping)
    write_csv(
        out_dir / "game_genre_mapping.csv",
        game_genre_mapping,
        [
            "genre_id",
            "genre_name",
            "order_in_genre",
            "game_id",
            "game_name",
            "game_source_order",
        ],
    )
    game_catalog = build_game_catalog(game_mapping, game_genre_mapping)
    catalog_by_game_id = {row["game_id"]: row for row in game_catalog}
    write_json(out_dir / "game_catalog.json", game_catalog)
    write_csv(
        out_dir / "game_catalog.csv",
        game_catalog,
        [
            "source_order",
            "game_id",
            "game_name",
            "genre_count",
            "genre_ids",
            "genre_names",
            "is_duplicate_id",
            "is_duplicate_name",
            "is_numeric_name",
        ],
    )

    if args.game_id:
        selected_ids = set(args.game_id)
        games = [game for game in games if game["value"] in selected_ids]
    if args.limit:
        games = games[: args.limit]

    if not games:
        raise SystemExit("No games matched the requested filters.")

    if args.mapping_only:
        print(f"Done. Saved mappings to {out_dir.resolve()}")
        return

    summary_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    wide_summary_rows: list[dict[str, Any]] = []
    wide_value_rows: list[dict[str, Any]] = []
    final_value_rows: list[dict[str, Any]] = []

    manifest = {
        "base_url": base_url,
        "game_count": len(games),
        "game_mapping": str(out_dir / "game_mapping.csv"),
        "genre_mapping": str(out_dir / "genre_mapping.csv"),
        "game_genre_mapping": str(out_dir / "game_genre_mapping.csv"),
        "game_catalog": str(out_dir / "game_catalog.csv"),
        "files": [],
    }

    for index, game in enumerate(games, start=1):
        filename = f"{game['value']}_{slugify(game['label'])}.json"
        output_path = raw_dir / filename

        if output_path.exists() and not args.overwrite:
            with output_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            status = "cached"
        else:
            print(f"[{index}/{len(games)}] downloading {game['value']} {game['label']}")
            record = download_game(
                base_url=base_url,
                game=game,
                genres=genres,
                genders=args.genders,
                age_range=[args.age_min, args.age_max],
            )
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
            status = "downloaded"
            time.sleep(args.sleep)

        manifest["files"].append(
            {
                "game_id": game["value"],
                "game_name": game["label"],
                "path": str(output_path),
                "status": status,
            }
        )

        psyc_summary, psyc_values = extract_metric_rows(
            game, catalog_by_game_id, "psychological", record["psychological"]
        )
        func_summary, func_values = extract_metric_rows(
            game, catalog_by_game_id, "functional", record["functional"]
        )
        summary_rows.extend(psyc_summary)
        summary_rows.extend(func_summary)
        value_rows.extend(psyc_values)
        value_rows.extend(func_values)
        wide_summary_rows.append(build_wide_summary_row(game, catalog_by_game_id, record))
        wide_value_rows.extend(build_wide_value_rows(game, catalog_by_game_id, record))
        final_value_rows.extend(build_final_value_rows(game, game_genre_mapping, record))

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    write_csv(
        out_dir / "benchmark_summary.csv",
        summary_rows,
        [
            "game_id",
            "game_name",
            "genre_ids",
            "genre_names",
            "benchmark_group",
            "metric",
            "n",
            "mean",
            "median",
            "min",
            "max",
            "stdev",
            "values_json",
        ],
    )
    write_csv(
        out_dir / "benchmark_values.csv",
        value_rows,
        [
            "game_id",
            "game_name",
            "genre_ids",
            "genre_names",
            "benchmark_group",
            "metric",
            "sample_index",
            "value",
        ],
    )
    write_csv(
        out_dir / "benchmark_summary_wide.csv",
        wide_summary_rows,
        ordered_fieldnames(
            wide_summary_rows,
            [
                "game_id",
                "game_name",
                "game_title",
                "game_source_order",
                "is_numeric_game_title",
                "genre_ids",
                "genre_names",
                "participant_count",
                "selected_game_count",
                "selected_genre_count",
            ],
        ),
    )
    write_csv(
        out_dir / "benchmark_values_wide.csv",
        wide_value_rows,
        ordered_fieldnames(
            wide_value_rows,
            [
                "game_id",
                "game_name",
                "game_title",
                "game_source_order",
                "is_numeric_game_title",
                "genre_ids",
                "genre_names",
                "sample_index",
            ],
        ),
    )
    write_csv(
        out_dir / "final_benchmark_values_wide.csv",
        final_value_rows,
        ordered_fieldnames(
            final_value_rows,
            [
                "game_id",
                "game_name",
                "game_title",
                "game_source_order",
                "is_numeric_game_title",
                "genre_id",
                "genre_name",
                "sample_index",
            ],
        ),
    )

    print(f"Done. Saved {len(games)} games to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
