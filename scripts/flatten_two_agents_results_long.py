#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def j(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False) if x is not None else ""


def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def make_row(
    *,
    result_json_path: str,
    result_data: Dict[str, Any],
    row_kind: str,
    base_tile: Dict[str, Any],
    variant_tile: Dict[str, Any],
    eyes: Dict[str, Any],
    brains: Dict[str, Any],
    zoom_index: Any = "",
) -> Dict[str, Any]:
    """
    One CSV row = one analyzed image variant.

    row_kind:
      - parent
      - zoom
    """

    return {
        # provenance
        "result_json_path": result_json_path,
        "row_kind": row_kind,
        "zoom_index": zoom_index,

        # run-level / final base-tile decision
        "timestamp": result_data.get("timestamp", ""),
        "base_tile_id": base_tile.get("tile_id", ""),
        "base_region_name": base_tile.get("region_name", ""),
        "base_center_lat": base_tile.get("center_lat", ""),
        "base_center_lon": base_tile.get("center_lon", ""),
        "base_tile_km": base_tile.get("tile_km", ""),
        "base_manifest_row": base_tile.get("manifest_row", ""),
        "base_manifest_col": base_tile.get("manifest_col", ""),
        "parent_score": result_data.get("parent_score", ""),
        "parent_initial_label": result_data.get("parent_initial_label", ""),
        "zoom_triggered": result_data.get("zoom_triggered", ""),
        "final_label": result_data.get("final_label", ""),
        "aggregation_reason": result_data.get("aggregation_reason", ""),

        # variant-level image metadata
        "variant_tile_id": variant_tile.get("tile_id", ""),
        "variant_region_name": variant_tile.get("region_name", ""),
        "variant_center_lat": variant_tile.get("center_lat", ""),
        "variant_center_lon": variant_tile.get("center_lon", ""),
        "variant_tile_km": variant_tile.get("tile_km", ""),
        "variant_image_path": variant_tile.get("image_path", ""),
        "variant_raw_path": variant_tile.get("raw_path", ""),
        "variant_manifest_row": variant_tile.get("manifest_row", ""),
        "variant_manifest_col": variant_tile.get("manifest_col", ""),
        "variant_source_kind": variant_tile.get("source_kind", ""),
        "variant_derived_from_tile_id": variant_tile.get("derived_from_tile_id", ""),
        "variant_kind": variant_tile.get("variant_kind", ""),
        "variant_label": variant_tile.get("variant_label", ""),

        # model outputs for this exact row/image
        "analysis": eyes.get("analysis", ""),
        "score_-10_10": brains.get("score_-10_10", ""),
        "navigate": brains.get("navigate", ""),
        "eyes_json": j(eyes),
        "brains_json": j(brains),

        # useful raw blobs
        "base_tile_json": j(base_tile),
        "variant_tile_json": j(variant_tile),
    }


def flatten_result_long(data: Dict[str, Any], result_json_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    base = safe_dict(data.get("base_tile"))
    parent_eyes = safe_dict(data.get("parent_eyes"))
    parent_brains = safe_dict(data.get("parent_brains"))

    # Parent/base tile row
    rows.append(
        make_row(
            result_json_path=result_json_path,
            result_data=data,
            row_kind="parent",
            base_tile=base,
            variant_tile=base,
            eyes=parent_eyes,
            brains=parent_brains,
            zoom_index="",
        )
    )

    # One row per zoom variant
    zoom_records = data.get("zoom_records", [])
    if isinstance(zoom_records, list):
        for idx, rec in enumerate(zoom_records, start=1):
            if not isinstance(rec, dict):
                continue

            ztile = safe_dict(rec.get("tile"))
            zeyes = safe_dict(rec.get("eyes"))
            zbrains = safe_dict(rec.get("brains"))

            rows.append(
                make_row(
                    result_json_path=result_json_path,
                    result_data=data,
                    row_kind="zoom",
                    base_tile=base,
                    variant_tile=ztile,
                    eyes=zeyes,
                    brains=zbrains,
                    zoom_index=idx,
                )
            )

    return rows


def find_result_jsons(run_dir: str) -> List[str]:
    paths: List[str] = []
    per_tile = Path(run_dir) / "per_tile"

    if per_tile.is_dir():
        for p in per_tile.glob("*/result.json"):
            paths.append(str(p))
    else:
        for p in Path(run_dir).rglob("result.json"):
            paths.append(str(p))

    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten two_agent per_tile/result.json files into long-format CSV: one row per parent or zoom variant."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to one run output directory that contains per_tile/",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV path. Default: <run_dir>/all_results_long.csv",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_csv = (
        os.path.abspath(args.output_csv)
        if args.output_csv
        else os.path.join(run_dir, "all_results_long.csv")
    )

    result_paths = find_result_jsons(run_dir)
    if not result_paths:
        raise SystemExit(f"No result.json files found under: {run_dir}")

    rows: List[Dict[str, Any]] = []
    failed = 0

    for path in result_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.extend(flatten_result_long(data, path))
        except Exception as e:
            failed += 1
            print(f"Warning: failed to parse {path}: {e}")

    if not rows:
        raise SystemExit("No rows parsed successfully.")

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    parent_rows = sum(1 for r in rows if r.get("row_kind") == "parent")
    zoom_rows = sum(1 for r in rows if r.get("row_kind") == "zoom")

    print(f"Wrote {len(rows)} rows to: {output_csv}")
    print(f"  parent rows: {parent_rows}")
    print(f"  zoom rows:   {zoom_rows}")
    print(f"  failed JSON files: {failed}")


if __name__ == "__main__":
    main()