#!/usr/bin/env python3
"""
two_agent_cached_tiles.py

Run a simple 2-agent (Mr. Eyes + Mr. Brains) pipeline on cached satellite tiles
downloaded by the existing Google Static Maps tiling script.

Design goals:
- Work directly on local cached PNG tiles (no Google Static Maps API needed)
- Be compatible with manifest.csv produced by the existing tile download script
  that stores `tile_path`, `raw_path`, `row`, `col`, and `status`
- Support sequential processing of:
    * a whole cache directory
    * a single tile selected by tile_id
    * a single tile selected by row/col
    * a single tile selected by nearest lat/lon
    * a single local image path
- Implement one round of navigation:
    * zoom_in  -> split the current tile into 4 resized child tiles
    * zoom_out -> prefer raw_path (overscan image); otherwise stitch a 3x3 neighbor mosaic

Outputs:
- results.jsonl / results.csv with full intermediate and final decisions
- progress.json for resumable runs
- optional saved zoom artifacts for inspection
- optional copies of final yes / unresolved tiles into dedicated folders

This script intentionally keeps aggregation simple and code-based, matching the MVP
direction discussed in the email chain:
1) Mr. Eyes produces structured reasoning on a tile
2) Mr. Brains converts that reasoning into a signed score and navigation suggestion
3) If the score is ambiguous, perform one round of zoom and rerun Eyes + Brains
4) Combine results with a simple deterministic rule
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps

from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------
# Prompts
# ----------------------------


EYES_PROMPT = """
You are an expert satellite imagery analyst.

Explain why you think this satellite image contains or does not contain a secure facility with layered fencing. In your response, describe the indicators you see, why these indicators point to the presence of a secure facility with layered fencing or why not, and where each indicator is in the image. Then, be a contrarian and offer a good-faith, though briefer, argument in favor of the opposite view.

Then explain whether it would be helpful to zoom in to examine the same area in higher resolution and more detail or zoom out for more context.

- zoom in = need more detail on a local feature
- zoom out = need broader context/site layout
- neither = current view is sufficient, or no useful next step

Focus on observable image evidence only. Keep your response within 8-10 sentences but reason fully.

Return ONLY valid JSON:
{{
  "analysis": "your 8-10 sentence analysis"
}}
""".strip()

BRAINS_PROMPT_TEMPLATE = """
A satellite imagery analyst has analyzed a satellite image and reasoned as follows about whether the image contains a secure facility with layered fencing:

{eyes_analysis}

Based on the above:
Does the analyst believe that the image contains a secure facility with layered fencing? Rate on a scale from -10 to 10, with -10 indicating that the analyst confidently rules out the presence of a secure facility with layered fencing in the image, and 10 indicating that the analyst is certain that a secure facility with layered fencing is present in the image. Accordingly, a score closer to 0 would indicate ambivalence.

Did the analyst indicate it would be helpful to zoom in for more detail, zoom out for more context, or neither?

Return ONLY valid JSON:
{{
  "score_-10_10": integer from -10 to 10,
  "navigate": "zoom_in" or "zoom_out" or "neither"
}}
""".strip()

# ----------------------------
# Schemas
# ----------------------------

# EYES_SCHEMA: Dict[str, Any] = {
#     "name": "eyes_analysis",
#     "schema": {
#         "type": "object",
#         "properties": {
#             "supports_presence": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "indicator": {"type": "string"},
#                         "where_in_image": {"type": "string"},
#                         "why_it_matters": {"type": "string"},
#                     },
#                     "required": ["indicator", "where_in_image", "why_it_matters"],
#                     "additionalProperties": False,
#                 },
#             },
#             "supports_absence": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "indicator": {"type": "string"},
#                         "where_in_image": {"type": "string"},
#                         "why_it_matters": {"type": "string"},
#                     },
#                     "required": ["indicator", "where_in_image", "why_it_matters"],
#                     "additionalProperties": False,
#                 },
#             },
#             "contrarian_argument": {"type": "string"},
#             "most_convincing_side": {
#                 "type": "string",
#                 "enum": ["present", "absent", "tie"],
#             },
#             "navigate": {
#                 "type": "string",
#                 "enum": ["zoom_in", "zoom_out", "neither"],
#             },
#             "ambiguity_reason": {"type": "string"},
#             "conclusion": {"type": "string"},
#         },
#         "required": [
#             "supports_presence",
#             "supports_absence",
#             "contrarian_argument",
#             "most_convincing_side",
#             "navigate",
#             "ambiguity_reason",
#             "conclusion",
#         ],
#         "additionalProperties": False,
#     },
#     "strict": True,
# }

EYES_SCHEMA: Dict[str, Any] = {
    "name": "eyes_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
        },
        "required": ["analysis"],
        "additionalProperties": False,
    },
    "strict": True,
}

# BRAINS_SCHEMA: Dict[str, Any] = {
#     "name": "brains_score",
#     "schema": {
#         "type": "object",
#         "properties": {
#             "score_-10_10": {"type": "integer", "minimum": -10, "maximum": 10},
#             "navigate": {
#                 "type": "string",
#                 "enum": ["zoom_in", "zoom_out", "neither"],
#             },
#             "confidence_band": {
#                 "type": "string",
#                 "enum": ["low", "medium", "high"],
#             },
#             "reason": {"type": "string"},
#         },
#         "required": ["score_-10_10", "navigate", "confidence_band", "reason"],
#         "additionalProperties": False,
#     },
#     "strict": True,
# }

BRAINS_SCHEMA: Dict[str, Any] = {
    "name": "brains_score",
    "schema": {
        "type": "object",
        "properties": {
            "score_-10_10": {
                "type": "integer",
                "minimum": -10,
                "maximum": 10,
            },
            "navigate": {
                "type": "string",
                "enum": ["zoom_in", "zoom_out", "neither"],
            },
        },
        "required": ["score_-10_10", "navigate"],
        "additionalProperties": False,
    },
    "strict": True,
}

# ----------------------------
# Data classes
# ----------------------------

@dataclass
class TileRecord:
    idx: int
    tile_id: str
    region_name: str
    center_lat: float
    center_lon: float
    tile_km: float
    image_path: str
    raw_path: Optional[str] = None
    overscan_km_requested: Optional[float] = None
    manifest_row: Optional[int] = None
    manifest_col: Optional[int] = None
    source_kind: str = "manifest"
    derived_from_tile_id: Optional[str] = None
    variant_kind: str = "base"
    variant_label: str = "base"

    def as_short_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "tile_id": self.tile_id,
            "region_name": self.region_name,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "tile_km": self.tile_km,
            "image_path": self.image_path,
            "raw_path": self.raw_path,
            "overscan_km_requested": self.overscan_km_requested,
            "manifest_row": self.manifest_row,
            "manifest_col": self.manifest_col,
            "source_kind": self.source_kind,
            "derived_from_tile_id": self.derived_from_tile_id,
            "variant_kind": self.variant_kind,
            "variant_label": self.variant_label,
        }


@dataclass
class Thresholds:
    ambiguous_abs: int = 4
    positive_abs: int = 5
    negative_abs: int = 5


@dataclass
class AgentOutputs:
    eyes: Dict[str, Any]
    brains: Dict[str, Any]


# ----------------------------
# Utility helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("._")


def safe_grid_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_tile_km_from_image_path(image_path: str, default: float = 2.0) -> float:
    base = os.path.basename(image_path)
    patterns = [
        r"view([\d.]+)km",
        r"tile([\d.]+)km",
        r"_([\d.]+)km(?:\.|_)",
    ]
    for pat in patterns:
        m = re.search(pat, base, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return default


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
    return r * c


def ensure_dir(path: str | Path) -> str:
    p = os.path.abspath(str(path))
    os.makedirs(p, exist_ok=True)
    return p


def read_image_as_data_url(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_csv(path: str, rec: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = {k: ("" if rec.get(k) is None else rec.get(k)) for k in fieldnames}
        writer.writerow(row)


def clamp_score(val: Any, default: int = 0, lo: int = -10, hi: int = 10) -> int:
    try:
        v = int(val)
    except Exception:
        v = default
    return max(lo, min(hi, v))

def should_omit_temperature(model: str) -> bool:
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ----------------------------
# Manifest loading / selection
# ----------------------------

def load_manifest_rows(cache_dir: str) -> List[TileRecord]:
    """
    Load tiles from manifest.csv, supporting both:
    - existing cached local classifier style (`image_path`)
    - existing download script style (`tile_path`, `raw_path`, `row`, `col`, `status`)
    """
    manifest_path = os.path.join(cache_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

    rows: List[TileRecord] = []
    region_default = os.path.basename(os.path.abspath(cache_dir).rstrip(os.sep))

    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()

            image_path = str(row.get("image_path", "")).strip()
            if not image_path:
                image_path = str(row.get("tile_path", "")).strip()

            if not image_path:
                continue

            raw_path = str(row.get("raw_path", "")).strip() or None

            if not os.path.isfile(image_path):
                base = os.path.basename(image_path)
                fallback = os.path.join(cache_dir, "tiles", base)
                if os.path.isfile(fallback):
                    image_path = fallback
                else:
                    continue

            if raw_path and not os.path.isfile(raw_path):
                raw_base = os.path.basename(raw_path)
                raw_fallback = os.path.join(cache_dir, "raw", raw_base)
                if os.path.isfile(raw_fallback):
                    raw_path = raw_fallback
                else:
                    raw_path = None

            if status not in {"", "ok", "downloaded", "cached"}:
                continue

            try:
                center_lat = float(row["center_lat"])
                center_lon = float(row["center_lon"])
            except Exception:
                continue

            tile_km_s = str(row.get("tile_km", "")).strip()
            if tile_km_s:
                try:
                    tile_km = float(tile_km_s)
                except ValueError:
                    tile_km = parse_tile_km_from_image_path(image_path, default=2.0)
            else:
                tile_km = parse_tile_km_from_image_path(image_path, default=2.0)

            overscan_km_requested = None
            if str(row.get("overscan_km_requested", "")).strip():
                try:
                    overscan_km_requested = float(row["overscan_km_requested"])
                except ValueError:
                    overscan_km_requested = None

            tile_id = str(row.get("tile_id", "")).strip()
            if not tile_id:
                r = row.get("row")
                c = row.get("col")
                if r not in (None, "") and c not in (None, ""):
                    tile_id = f"r{r}_c{c}"
                else:
                    tile_id = Path(image_path).stem

            region_name = str(row.get("region_name", "")).strip() or region_default

            rows.append(
                TileRecord(
                    idx=len(rows) + 1,
                    tile_id=tile_id,
                    region_name=region_name,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    tile_km=tile_km,
                    image_path=image_path,
                    raw_path=raw_path,
                    overscan_km_requested=overscan_km_requested,
                    manifest_row=safe_grid_int(row.get("row")),
                    manifest_col=safe_grid_int(row.get("col")),
                    source_kind="manifest",
                    derived_from_tile_id=None,
                    variant_kind="base",
                    variant_label="base",
                )
            )

    return rows


def build_row_col_index(rows: Sequence[TileRecord]) -> Dict[Tuple[int, int], TileRecord]:
    out: Dict[Tuple[int, int], TileRecord] = {}
    for r in rows:
        if r.manifest_row is not None and r.manifest_col is not None:
            out[(r.manifest_row, r.manifest_col)] = r
    return out

def select_rows(
    rows: List[TileRecord],
    tile_id: Optional[str],
    row: Optional[int],
    col: Optional[int],
    lat: Optional[float],
    lon: Optional[float],
    scan_radius_km: Optional[float],
    limit: Optional[int],
) -> List[TileRecord]:
    if tile_id:
        filtered = [r for r in rows if r.tile_id == tile_id]
        if not filtered:
            raise ValueError(f"No tile found with tile_id={tile_id!r}")
        return filtered[:1]

    if row is not None or col is not None:
        if row is None or col is None:
            raise ValueError("Both --row and --col are required together.")
        filtered = [r for r in rows if r.manifest_row == row and r.manifest_col == col]
        if not filtered:
            raise ValueError(f"No tile found with row={row}, col={col}")
        return filtered[:1]

    if lat is not None or lon is not None:
        if lat is None or lon is None:
            raise ValueError("Both --lat and --lon are required together.")

        if not rows:
            return []

        # 新模式：按半径选所有 tile
        if scan_radius_km is not None:
            filtered = [
                r for r in rows
                if haversine_km(lat, lon, r.center_lat, r.center_lon) <= scan_radius_km
            ]
            filtered.sort(key=lambda r: haversine_km(lat, lon, r.center_lat, r.center_lon))

            if not filtered:
                raise ValueError(
                    f"No cached tiles found within {scan_radius_km} km of ({lat}, {lon})."
                )

            if limit is not None:
                filtered = filtered[: int(limit)]

            return filtered

        # 旧模式：只选最近一张
        nearest = min(rows, key=lambda r: haversine_km(lat, lon, r.center_lat, r.center_lon))
        return [nearest]

    if limit is not None:
        return rows[: int(limit)]

    return rows


# def select_rows(
#     rows: List[TileRecord],
#     tile_id: Optional[str],
#     row: Optional[int],
#     col: Optional[int],
#     lat: Optional[float],
#     lon: Optional[float],
#     limit: Optional[int],
# ) -> List[TileRecord]:
#     if tile_id:
#         filtered = [r for r in rows if r.tile_id == tile_id]
#         if not filtered:
#             raise ValueError(f"No tile found with tile_id={tile_id!r}")
#         return filtered[:1]

#     if row is not None or col is not None:
#         if row is None or col is None:
#             raise ValueError("Both --row and --col are required together.")
#         filtered = [r for r in rows if r.manifest_row == row and r.manifest_col == col]
#         if not filtered:
#             raise ValueError(f"No tile found with row={row}, col={col}")
#         return filtered[:1]

#     if lat is not None or lon is not None:
#         if lat is None or lon is None:
#             raise ValueError("Both --lat and --lon are required together.")
#         if not rows:
#             return []
#         nearest = min(rows, key=lambda r: haversine_km(lat, lon, r.center_lat, r.center_lon))
#         return [nearest]

#     if limit is not None:
#         return rows[: int(limit)]

#     return rows


# ----------------------------
# Model calls
# ----------------------------

def call_structured_vision_json(
    client: OpenAI,
    model: str,
    prompt_text: str,
    image_data_url: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    kwargs = {
        "model": model,
        "response_format": {
            "type": "json_schema",
            "json_schema": schema,
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }

    if not should_omit_temperature(model):
        kwargs["temperature"] = 0

    resp = client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip()
    return json.loads(raw)


def call_structured_text_json(
    client: OpenAI,
    model: str,
    prompt_text: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        # temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": schema,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    return json.loads(raw)


def run_mr_eyes(
    client: OpenAI,
    model: str,
    tile: TileRecord,
) -> Dict[str, Any]:
    image_data_url = read_image_as_data_url(tile.image_path)
    try:
        data = call_structured_vision_json(
            client=client,
            model=model,
            prompt_text=EYES_PROMPT,
            image_data_url=image_data_url,
            schema=EYES_SCHEMA,
        )
    # except Exception as e:
    #     data = {
    #         "supports_presence": [],
    #         "supports_absence": [],
    #         "contrarian_argument": f"Eyes error: {e}",
    #         "most_convincing_side": "tie",
    #         "navigate": "neither",
    #         "ambiguity_reason": f"Eyes error: {e}",
    #         "conclusion": f"Eyes error: {e}",
    #     }
    except Exception as e:
        data = {
            "analysis": f"Eyes error: {e}"
        }
    return data

def run_mr_brains(
    client: OpenAI,
    model: str,
    eyes_output: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = BRAINS_PROMPT_TEMPLATE.format(
        eyes_analysis=eyes_output.get("analysis", "")
    )

    try:
        data = call_structured_text_json(
            client=client,
            model=model,
            prompt_text=prompt,
            schema=BRAINS_SCHEMA,
        )
        data["score_-10_10"] = clamp_score(data.get("score_-10_10", 0))

        if data.get("navigate") not in {"zoom_in", "zoom_out", "neither"}:
            data["navigate"] = "neither"

    except Exception as e:
        data = {
            "score_-10_10": 0,
            "navigate": "neither",
        }

    return data
# def run_mr_brains(
#     client: OpenAI,
#     model: str,
#     eyes_output: Dict[str, Any],
# ) -> Dict[str, Any]:
#     prompt = BRAINS_PROMPT_TEMPLATE.format(
#         eyes_json=json.dumps(eyes_output, ensure_ascii=False, indent=2)
#     )
#     try:
#         data = call_structured_text_json(
#             client=client,
#             model=model,
#             prompt_text=prompt,
#             schema=BRAINS_SCHEMA,
#         )
#         data["score_-10_10"] = clamp_score(data.get("score_-10_10", 0))
#         if data.get("navigate") not in {"zoom_in", "zoom_out", "neither"}:
#             data["navigate"] = "neither"
#         if data.get("confidence_band") not in {"low", "medium", "high"}:
#             data["confidence_band"] = "low"
#     except Exception as e:
#         data = {
#             "score_-10_10": 0,
#             "navigate": "neither",
#             "confidence_band": "low",
#             "reason": f"Brains error: {e}",
#         }
#     return data


def run_agents_for_tile(
    client: OpenAI,
    eyes_model: str,
    brains_model: str,
    tile: TileRecord,
) -> AgentOutputs:
    eyes = run_mr_eyes(client, eyes_model, tile)
    brains = run_mr_brains(client, brains_model, eyes)
    return AgentOutputs(eyes=eyes, brains=brains)


# ----------------------------
# Zoom generation
# ----------------------------

def save_resized(img: Image.Image, path: str, size: Tuple[int, int]) -> None:
    out = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)
    if out.size != size:
        out = ImageOps.fit(out, size, method=Image.Resampling.LANCZOS)
    out.save(path, format="PNG")


def generate_zoom_in_variants(
    tile: TileRecord,
    zoom_dir: str,
) -> List[TileRecord]:
    """
    Split the tile into 4 quadrants and resize each back to original size.
    Ground extent becomes tile_km / 2.
    """
    img = open_rgb(tile.image_path)
    w, h = img.size
    half_w, half_h = w // 2, h // 2

    specs = [
        ("NW", (0, 0, half_w, half_h)),
        ("NE", (half_w, 0, w, half_h)),
        ("SW", (0, half_h, half_w, h)),
        ("SE", (half_w, half_h, w, h)),
    ]

    out_records: List[TileRecord] = []
    for i, (label, box) in enumerate(specs, start=1):
        crop = img.crop(box)
        fname = f"{sanitize_tag(tile.tile_id)}__zoom_in__{label}.png"
        fpath = os.path.join(zoom_dir, fname)
        save_resized(crop, fpath, img.size)

        delta_deg_lat = (tile.tile_km / 4.0) / 110.574
        delta_deg_lon = (tile.tile_km / 4.0) / max(
            1e-9, 111.320 * math.cos(math.radians(tile.center_lat))
        )
        child_lat = tile.center_lat + (1 if label.startswith("N") else -1) * delta_deg_lat
        child_lon = tile.center_lon + (1 if label.endswith("E") else -1) * delta_deg_lon

        out_records.append(
            TileRecord(
                idx=i,
                tile_id=f"{tile.tile_id}__zin__{label}",
                region_name=tile.region_name,
                center_lat=child_lat,
                center_lon=child_lon,
                tile_km=max(tile.tile_km / 2.0, 1e-6),
                image_path=fpath,
                raw_path=None,
                overscan_km_requested=None,
                manifest_row=None,
                manifest_col=None,
                source_kind="derived_zoom_in",
                derived_from_tile_id=tile.tile_id,
                variant_kind="zoom_in",
                variant_label=label,
            )
        )

    return out_records


def try_zoom_out_from_raw(
    tile: TileRecord,
    zoom_dir: str,
) -> Optional[TileRecord]:
    """
    Prefer raw_path if available: this is the original overscan image from the download script.
    Resize it to the current tile size so the model sees a broader-context version at the same
    pixel dimensions.
    """
    if not tile.raw_path or not os.path.isfile(tile.raw_path):
        return None

    base_img = open_rgb(tile.image_path)
    raw_img = open_rgb(tile.raw_path)
    if raw_img.size[0] < base_img.size[0] or raw_img.size[1] < base_img.size[1]:
        return None

    fname = f"{sanitize_tag(tile.tile_id)}__zoom_out__raw.png"
    fpath = os.path.join(zoom_dir, fname)
    save_resized(raw_img, fpath, base_img.size)

    ground_extent = tile.overscan_km_requested or (tile.tile_km * 1.2)

    return TileRecord(
        idx=1,
        tile_id=f"{tile.tile_id}__zout__raw",
        region_name=tile.region_name,
        center_lat=tile.center_lat,
        center_lon=tile.center_lon,
        tile_km=ground_extent,
        image_path=fpath,
        raw_path=None,
        overscan_km_requested=None,
        manifest_row=None,
        manifest_col=None,
        source_kind="derived_zoom_out",
        derived_from_tile_id=tile.tile_id,
        variant_kind="zoom_out",
        variant_label="raw",
    )


def try_zoom_out_from_neighbors(
    tile: TileRecord,
    row_col_index: Dict[Tuple[int, int], TileRecord],
    zoom_dir: str,
) -> List[TileRecord]:
    """
    Build all complete 2x2 mosaics that include the current tile.

    Possible 2x2 corner blocks:
    - NW_block: current tile is bottom-right
    - NE_block: current tile is bottom-left
    - SW_block: current tile is top-right
    - SE_block: current tile is top-left

    Return every complete block. If a block is incomplete, skip it.
    """
    if tile.manifest_row is None or tile.manifest_col is None:
        return []

    r0 = tile.manifest_row
    c0 = tile.manifest_col

    block_options = [
        ("NW_block", [(r0 - 1, c0 - 1), (r0 - 1, c0),
                      (r0,     c0 - 1), (r0,     c0)]),

        ("NE_block", [(r0 - 1, c0),     (r0 - 1, c0 + 1),
                      (r0,     c0),     (r0,     c0 + 1)]),

        ("SW_block", [(r0,     c0 - 1), (r0,     c0),
                      (r0 + 1, c0 - 1), (r0 + 1, c0)]),

        ("SE_block", [(r0,     c0),     (r0,     c0 + 1),
                      (r0 + 1, c0),     (r0 + 1, c0 + 1)]),
    ]

    center_img = open_rgb(tile.image_path)
    w, h = center_img.size

    out_records: List[TileRecord] = []

    for label, coords in block_options:
        neighbors = [row_col_index.get(rc) for rc in coords]

        # 某个角不完整：直接跳过
        if any(n is None for n in neighbors):
            continue
        if any(not os.path.isfile(n.image_path) for n in neighbors if n is not None):
            continue

        canvas = Image.new("RGB", (w * 2, h * 2))

        idx = 0
        for rr in range(2):
            for cc in range(2):
                neighbor = neighbors[idx]
                idx += 1
                assert neighbor is not None

                img = open_rgb(neighbor.image_path)
                if img.size != (w, h):
                    img = ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS)

                canvas.paste(img, (cc * w, rr * h))

        fname = f"{sanitize_tag(tile.tile_id)}__zoom_out__mosaic2x2__{label}.png"
        fpath = os.path.join(zoom_dir, fname)

        # 压回原始输入尺寸
        save_resized(canvas, fpath, center_img.size)

        valid_neighbors = [n for n in neighbors if n is not None]
        mosaic_center_lat = sum(n.center_lat for n in valid_neighbors) / len(valid_neighbors)
        mosaic_center_lon = sum(n.center_lon for n in valid_neighbors) / len(valid_neighbors)

        out_records.append(
            TileRecord(
                idx=len(out_records) + 1,
                tile_id=f"{tile.tile_id}__zout__mosaic2x2__{label}",
                region_name=tile.region_name,
                center_lat=mosaic_center_lat,
                center_lon=mosaic_center_lon,
                tile_km=tile.tile_km * 2.0,
                image_path=fpath,
                raw_path=None,
                overscan_km_requested=None,
                manifest_row=None,
                manifest_col=None,
                source_kind="derived_zoom_out",
                derived_from_tile_id=tile.tile_id,
                variant_kind="zoom_out",
                variant_label=f"mosaic2x2_{label}",
            )
        )

    return out_records


def generate_zoom_tiles(
    tile: TileRecord,
    navigate: str,
    row_col_index: Dict[Tuple[int, int], TileRecord],
    zoom_dir: str,
) -> List[TileRecord]:
    ensure_dir(zoom_dir)

    if navigate == "zoom_in":
        return generate_zoom_in_variants(tile, zoom_dir)

    if navigate == "zoom_out":
        return try_zoom_out_from_neighbors(tile, row_col_index, zoom_dir)

    return []


# ----------------------------
# Aggregation
# ----------------------------

def score_to_label(score: int, thresholds: Thresholds) -> str:
    if score >= thresholds.positive_abs:
        return "yes"
    if score <= -thresholds.negative_abs:
        return "no"
    return "unresolved"


def should_zoom(parent_score: int, navigate: str, thresholds: Thresholds) -> bool:
    return abs(parent_score) <= thresholds.ambiguous_abs and navigate in {"zoom_in", "zoom_out"}


def aggregate_after_zoom(
    parent_score: int,
    navigate: str,
    zoom_scores: Sequence[int],
    thresholds: Thresholds,
) -> Tuple[str, str]:
    """
    Simple MVP aggregation rule:
    - If parent is already decisive, caller should not invoke this.
    - zoom_in:
        * if any child is decisively positive -> yes
        * elif all children are decisively negative -> no
        * else unresolved
    - zoom_out:
        * use the single broader-context score decisively if possible
        * otherwise unresolved
    """
    if not zoom_scores:
        return ("unresolved", "ambiguous_parent_but_no_zoom_tiles_available")

    if navigate == "zoom_in":
        if max(zoom_scores) >= thresholds.positive_abs:
            return ("yes", "zoom_in_found_decisive_positive_child")
        if all(s <= -thresholds.negative_abs for s in zoom_scores):
            return ("no", "zoom_in_all_children_decisively_negative")
        return ("unresolved", "zoom_in_children_still_mixed_or_ambiguous")

    if navigate == "zoom_out":
        if max(zoom_scores) >= thresholds.positive_abs:
            return ("yes", "zoom_out_found_decisive_positive_variant")
        if all(s <= -thresholds.negative_abs for s in zoom_scores):
            return ("no", "zoom_out_all_variants_decisively_negative")
        return ("unresolved", "zoom_out_variants_mixed_or_ambiguous")

    return (score_to_label(parent_score, thresholds), "no_navigation_requested")


# ----------------------------
# Per-tile pipeline
# ----------------------------

def maybe_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def process_one_base_tile(
    client: OpenAI,
    base_tile: TileRecord,
    eyes_model: str,
    brains_model: str,
    row_col_index: Dict[Tuple[int, int], TileRecord],
    thresholds: Thresholds,
    out_dir: str,
    save_zoom_artifacts: bool,
) -> Dict[str, Any]:
    tile_out_dir = ensure_dir(os.path.join(out_dir, "per_tile", sanitize_tag(base_tile.tile_id)))
    zoom_dir = ensure_dir(os.path.join(tile_out_dir, "zoom_variants"))

    # Save a convenient local copy of the source tile for inspection
    source_tile_copy = os.path.join(
        tile_out_dir,
        os.path.basename(base_tile.image_path)
    )
    maybe_copy(base_tile.image_path, source_tile_copy)

    parent_outputs = run_agents_for_tile(client, eyes_model, brains_model, base_tile)
    parent_score = clamp_score(parent_outputs.brains.get("score_-10_10", 0))
    parent_nav = parent_outputs.brains.get("navigate", "neither")
    parent_label = score_to_label(parent_score, thresholds)

    zoom_records: List[Dict[str, Any]] = []
    final_label = parent_label
    aggregation_reason = "parent_decision_direct"

    if should_zoom(parent_score, parent_nav, thresholds):
        zoom_tiles = generate_zoom_tiles(base_tile, parent_nav, row_col_index, zoom_dir)
        zoom_scores: List[int] = []

        for ztile in zoom_tiles:
            z_outputs = run_agents_for_tile(client, eyes_model, brains_model, ztile)
            z_score = clamp_score(z_outputs.brains.get("score_-10_10", 0))
            zoom_scores.append(z_score)

            zoom_records.append(
                {
                    "tile": ztile.as_short_dict(),
                    "eyes": z_outputs.eyes,
                    "brains": z_outputs.brains,
                }
            )

        final_label, aggregation_reason = aggregate_after_zoom(
            parent_score=parent_score,
            navigate=parent_nav,
            zoom_scores=zoom_scores,
            thresholds=thresholds,
        )

        if not save_zoom_artifacts and os.path.isdir(zoom_dir):
            shutil.rmtree(zoom_dir, ignore_errors=True)

    result = {
        "timestamp": utc_now_iso(),
        "base_tile": base_tile.as_short_dict(),
        "source_tile_copy": source_tile_copy,
        "parent_eyes": parent_outputs.eyes,
        "parent_brains": parent_outputs.brains,
        "parent_score": parent_score,
        "parent_initial_label": parent_label,
        "zoom_triggered": should_zoom(parent_score, parent_nav, thresholds),
        "zoom_records": zoom_records,
        "final_label": final_label,
        "aggregation_reason": aggregation_reason,
    }

    with open(os.path.join(tile_out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a 2-agent cached-tile pipeline (Mr. Eyes + Mr. Brains) with one optional zoom round."
    )

    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--cache_dir",
        type=str,
        help="Directory containing manifest.csv and tiles/ from the previous download script.",
    )
    input_group.add_argument(
        "--image_path",
        type=str,
        help="Single local image to test. For this mode, zoom_out may be unavailable unless --raw_path is also provided.",
    )

    p.add_argument("--raw_path", type=str, default=None, help="Optional raw/overscan image for single-image mode.")
    p.add_argument("--tile_km", type=float, default=2.0, help="Tile ground extent in km for single-image mode.")
    p.add_argument("--region_name", type=str, default="single_image", help="Region name for single-image mode.")
    p.add_argument("--tile_id", type=str, default=None, help="Select one tile from cache_dir by tile_id.")
    p.add_argument("--row", type=int, default=None, help="Select one tile from cache_dir by manifest row.")
    p.add_argument("--col", type=int, default=None, help="Select one tile from cache_dir by manifest col.")
    p.add_argument("--lat", type=float, default=None, help="Select nearest tile from cache_dir by latitude.")
    p.add_argument("--lon", type=float, default=None, help="Select nearest tile from cache_dir by longitude.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N tiles.")

    p.add_argument("--model", type=str, default=None, help="Default model for both agents.")
    p.add_argument("--eyes_model", type=str, default=None, help="Optional override model for Mr. Eyes.")
    p.add_argument("--brains_model", type=str, default=None, help="Optional override model for Mr. Brains.")

    p.add_argument("--ambiguous_abs", type=int, default=4, help="|score| <= this triggers zoom if navigation suggests it.")
    p.add_argument("--positive_abs", type=int, default=5, help="score >= this means decisive yes.")
    p.add_argument("--negative_abs", type=int, default=5, help="score <= -this means decisive no.")

    p.add_argument("--output", type=str, default="./two_agent_outputs", help="Base output directory.")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between tiles.")
    p.add_argument("--overwrite", action="store_true", help="Reprocess tiles even if already present in progress.")
    p.add_argument("--save_zoom_artifacts", action="store_true", help="Keep derived zoom images on disk.")
    p.add_argument("--copy_final_yes", action="store_true", help="Copy final yes base tiles into final_yes/.")
    p.add_argument("--copy_final_unresolved", action="store_true", help="Copy final unresolved base tiles into final_unresolved/.")

    p.add_argument(
    "--scan_radius_km",
    type=float,
    default=None,
    help="When used with --lat and --lon, process all cached tiles whose centers fall within this radius (km).",
)
    p.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of tiles to process in parallel. Default 1 keeps the original sequential behavior.",
)

    return p


def resolve_models(args: argparse.Namespace) -> Tuple[str, str]:
    default_model = args.model
    eyes_model = args.eyes_model or default_model
    brains_model = args.brains_model or default_model
    if not eyes_model or not brains_model:
        raise ValueError("Provide --model or both --eyes_model and --brains_model.")
    return eyes_model, brains_model


def build_single_image_record(args: argparse.Namespace) -> TileRecord:
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"image_path does not exist: {args.image_path}")
    raw_path = None
    if args.raw_path:
        if not os.path.isfile(args.raw_path):
            raise FileNotFoundError(f"raw_path does not exist: {args.raw_path}")
        raw_path = args.raw_path
    tile_id = Path(args.image_path).stem
    return TileRecord(
        idx=1,
        tile_id=tile_id,
        region_name=args.region_name,
        center_lat=0.0,
        center_lon=0.0,
        tile_km=float(args.tile_km),
        image_path=os.path.abspath(args.image_path),
        raw_path=os.path.abspath(raw_path) if raw_path else None,
        overscan_km_requested=None,
        manifest_row=None,
        manifest_col=None,
        source_kind="single_image",
        derived_from_tile_id=None,
        variant_kind="base",
        variant_label="base",
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    eyes_model, brains_model = resolve_models(args)

    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not set.")
        raise SystemExit(1)

    client = OpenAI(api_key=openai_key)
    thresholds = Thresholds(
        ambiguous_abs=args.ambiguous_abs,
        positive_abs=args.positive_abs,
        negative_abs=args.negative_abs,
    )

    if args.cache_dir:
        cache_dir = os.path.abspath(args.cache_dir)
        rows = load_manifest_rows(cache_dir)
        # selected_rows = select_rows(
        #     rows=rows,
        #     tile_id=args.tile_id,
        #     row=args.row,
        #     col=args.col,
        #     lat=args.lat,
        #     lon=args.lon,
        #     limit=args.limit,
        # )
        selected_rows = select_rows(
            rows=rows,
            tile_id=args.tile_id,
            row=args.row,
            col=args.col,
            lat=args.lat,
            lon=args.lon,
            scan_radius_km=args.scan_radius_km,
            limit=args.limit,
        )
        row_col_index = build_row_col_index(rows)
        run_name = f"{os.path.basename(cache_dir.rstrip(os.sep))}__two_agent__{sanitize_tag(eyes_model)}"
        # if args.lat is not None and args.lon is not None and selected_rows:
        #     picked = selected_rows[0]
        #     print(
        #         "Nearest tile selected for "
        #         f"({args.lat:.6f}, {args.lon:.6f}) -> tile_id={picked.tile_id} "
        #         f"at ({picked.center_lat:.6f}, {picked.center_lon:.6f})"
        #     )
        if args.lat is not None and args.lon is not None and selected_rows:
            if args.scan_radius_km is not None:
                print(
                    f"Selected {len(selected_rows)} cached tiles within "
                    f"{args.scan_radius_km:.3f} km of ({args.lat:.6f}, {args.lon:.6f})"
                )
                nearest = min(
                    selected_rows,
                    key=lambda r: haversine_km(args.lat, args.lon, r.center_lat, r.center_lon)
                )
                print(
                    f"Nearest selected tile: tile_id={nearest.tile_id} "
                    f"at ({nearest.center_lat:.6f}, {nearest.center_lon:.6f})"
                )
            else:
                picked = selected_rows[0]
                print(
                    "Nearest tile selected for "
                    f"({args.lat:.6f}, {args.lon:.6f}) -> tile_id={picked.tile_id} "
                    f"at ({picked.center_lat:.6f}, {picked.center_lon:.6f})"
                )
    else:
        selected_rows = [build_single_image_record(args)]
        row_col_index = {}
        run_name = f"{sanitize_tag(selected_rows[0].tile_id)}__two_agent__{sanitize_tag(eyes_model)}"

    date_str = datetime.now().strftime("%b-%d-%Y")
    out_dir = ensure_dir(os.path.join(os.path.abspath(args.output), f"{run_name}__{date_str}"))
    progress_path = os.path.join(out_dir, "progress.json")
    results_jsonl = os.path.join(out_dir, "results.jsonl")
    results_csv = os.path.join(out_dir, "results.csv")

    csv_fields = [
        "timestamp",
        "tile_id",
        "region_name",
        "center_lat",
        "center_lon",
        "tile_km",
        "manifest_row",
        "manifest_col",
        "parent_score",
        "parent_navigate",
        "zoom_triggered",
        "final_label",
        "aggregation_reason",
        "image_path",
    ]

    processed_tile_ids = set()
    prior_results: List[Dict[str, Any]] = []
    if os.path.isfile(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
            processed_tile_ids = set(prog.get("processed_tile_ids", []))
            prior_results = prog.get("summary", [])
        except Exception as e:
            print(f"Warning: could not read progress.json: {e}. Starting fresh.")

    print("Configuration:")
    print(f"  Eyes model:   {eyes_model}")
    print(f"  Brains model: {brains_model}")
    print(f"  Selected tiles: {len(selected_rows)}")
    print(f"  Output: {out_dir}")
    print(
        "  Thresholds: "
        f"ambiguous_abs={thresholds.ambiguous_abs}, "
        f"positive_abs={thresholds.positive_abs}, "
        f"negative_abs={thresholds.negative_abs}"
    )

    summary_rows: List[Dict[str, Any]] = list(prior_results)

    tiles_to_process: List[Tuple[int, TileRecord]] = []

    for i, tile in enumerate(selected_rows, start=1):
        if (tile.tile_id in processed_tile_ids) and (not args.overwrite):
            print(f"[{i}/{len(selected_rows)}] skip already processed: {tile.tile_id}")
            continue

        tiles_to_process.append((i, tile))

    def process_tile_job(item: Tuple[int, TileRecord]) -> Tuple[int, TileRecord, Dict[str, Any]]:
        i, tile = item

        # Do not share one OpenAI client across threads.
        worker_client = OpenAI(api_key=openai_key)

        print(
            f"[{i}/{len(selected_rows)}] processing tile_id={tile.tile_id} "
            f"({tile.center_lat:.6f}, {tile.center_lon:.6f})"
        )

        result = process_one_base_tile(
            client=worker_client,
            base_tile=tile,
            eyes_model=eyes_model,
            brains_model=brains_model,
            row_col_index=row_col_index,
            thresholds=thresholds,
            out_dir=out_dir,
            save_zoom_artifacts=args.save_zoom_artifacts,
        )

        if args.sleep > 0:
            time.sleep(max(0.0, args.sleep))

        return i, tile, result

    def save_completed_result(i: int, tile: TileRecord, result: Dict[str, Any]) -> None:
        parent_nav = result["parent_brains"].get("navigate", "neither")

        summary_rec = {
            "timestamp": result["timestamp"],
            "tile_id": tile.tile_id,
            "region_name": tile.region_name,
            "center_lat": tile.center_lat,
            "center_lon": tile.center_lon,
            "tile_km": tile.tile_km,
            "manifest_row": tile.manifest_row,
            "manifest_col": tile.manifest_col,
            "parent_score": result["parent_score"],
            "parent_navigate": parent_nav,
            "zoom_triggered": result["zoom_triggered"],
            "final_label": result["final_label"],
            "aggregation_reason": result["aggregation_reason"],
            "image_path": tile.image_path,
        }

        append_jsonl(results_jsonl, summary_rec)
        append_csv(results_csv, summary_rec, csv_fields)

        summary_rows.append(summary_rec)
        processed_tile_ids.add(tile.tile_id)

        if args.copy_final_yes and result["final_label"] == "yes":
            dst = os.path.join(out_dir, "final_yes", os.path.basename(tile.image_path))
            maybe_copy(tile.image_path, dst)

        if args.copy_final_unresolved and result["final_label"] == "unresolved":
            dst = os.path.join(out_dir, "final_unresolved", os.path.basename(tile.image_path))
            maybe_copy(tile.image_path, dst)

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed_tile_ids": sorted(processed_tile_ids),
                    "summary": summary_rows,
                    "timestamp": utc_now_iso(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"  -> [{i}/{len(selected_rows)}] tile_id={tile.tile_id}, "
            f"parent_score={result['parent_score']}, "
            f"zoom_triggered={result['zoom_triggered']}, "
            f"final_label={result['final_label']} "
            f"({result['aggregation_reason']})"
        )

    max_workers = max(1, int(args.workers))

    if max_workers == 1:
        for item in tiles_to_process:
            i, tile, result = process_tile_job(item)
            save_completed_result(i, tile, result)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_tile_job, item)
                for item in tiles_to_process
            ]

            for future in as_completed(futures):
                i, tile, result = future.result()
                save_completed_result(i, tile, result)

    print("\nDone.")
    print(f"Results JSONL: {results_jsonl}")
    print(f"Results CSV:   {results_csv}")
    print(f"Progress:      {progress_path}")


if __name__ == "__main__":
    main()
