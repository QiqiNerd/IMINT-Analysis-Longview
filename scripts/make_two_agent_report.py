#!/usr/bin/env python3
"""
make_two_agent_static_report.py

Generate an offline, static HTML review report for the cached-tile two-agent pipeline.

What it reads:
  REQUIRED: A two-agent run directory created by two_agent_cached_tiles.py
     - results.csv
     - per_tile/<tile_id>/result.json

  OPTIONAL: A scan/cache directory created by scan_tiles_overscan_crop.py
     - manifest.csv
     - tiles/
     - optionally raw/

Important:
  The report scope is always the two-agent run_dir results.csv.
  If scan_dir is supplied, it is used only as fallback metadata / optional neighbor thumbnails;
  it does NOT cause the report to include every cached tile from the larger scan.

What it writes:
  output_dir/
    index.html
    assets/
      images/*.jpg              # small review thumbnails for processed tiles and optional context neighbors

The report is static and offline-friendly: your boss can open index.html directly.
It includes:
  - run summary
  - tile score / label grid
  - final YES cards first
  - all tile cards with images, metadata, Mr. Eyes analysis, Mr. Brains output
  - zoom records when available, without rendering missing zoom images
  - optional 3x3 neighbor context grid, controlled by --neighbors

Example:
  python scripts/make_two_agent_report.py \
    --run_dir "../IMINT-Outputs/example_run" \
    --output_dir "../IMINT-Outputs/example_report"

Optional, recommended if you want 3x3 neighbor context at the edge of a smaller two-agent run:
  python scripts/make_two_agent_report.py \
    --run_dir "../IMINT-Outputs/example_run" \
    --scan_dir "../IMINT-Images/example_z16" \
    --output_dir "../IMINT-Outputs/example_report"
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageOps
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with: python -m pip install pillow"
    ) from exc


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def sanitize_tag(value: Any) -> str:
    """Match the spirit of two_agent_cached_tiles.py sanitize_tag()."""
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return s or "untitled"


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def format_float(value: Any, digits: int = 6, blank: str = "") -> str:
    try:
        v = float(value)
        if math.isnan(v):
            return blank
        return f"{v:.{digits}f}"
    except Exception:
        return blank


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def normalize_path(value: Any, base_dir: Optional[Path] = None) -> Optional[Path]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------


@dataclass
class ManifestTile:
    tile_id: str
    row: Optional[int]
    col: Optional[int]
    center_lat: Optional[float]
    center_lon: Optional[float]
    tile_path: Optional[Path]
    raw_path: Optional[Path]
    status: str = ""
    tile_km: Optional[float] = None
    zoom: Optional[int] = None
    actual_returned_width_km: Optional[float] = None
    source_row: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewTile:
    tile_id: str
    safe_id: str
    row: Optional[int]
    col: Optional[int]
    center_lat: Optional[float]
    center_lon: Optional[float]
    image_path: Optional[Path]
    final_label: str
    parent_score: Optional[int]
    parent_navigate: str
    zoom_triggered: bool
    aggregation_reason: str
    result_json_path: Optional[Path] = None
    result_json: Optional[Dict[str, Any]] = None
    manifest: Optional[ManifestTile] = None
    thumbnail_rel: Optional[str] = None
    preview_rel: Optional[str] = None
    result_json_rel: Optional[str] = None
    cluster_id: Optional[int] = None
    source_row: Dict[str, Any] = field(default_factory=dict)

    @property
    def sort_score(self) -> int:
        return self.parent_score if self.parent_score is not None else -999


# -----------------------------------------------------------------------------
# Loading source data
# -----------------------------------------------------------------------------


def build_tile_id_from_manifest_row(row: Dict[str, Any], tile_path: Optional[Path]) -> str:
    explicit = str(row.get("tile_id", "")).strip()
    if explicit:
        return explicit
    r = row.get("row")
    c = row.get("col")
    if str(r).strip() and str(c).strip():
        return f"r{r}_c{c}"
    if tile_path is not None:
        return tile_path.stem
    return "unknown_tile"


def load_manifest(scan_dir: Path) -> Tuple[Dict[str, ManifestTile], Dict[Tuple[int, int], ManifestTile]]:
    manifest_path = scan_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.csv not found in scan_dir: {manifest_path}")

    rows = read_csv_dicts(manifest_path)
    by_id: Dict[str, ManifestTile] = {}
    by_rc: Dict[Tuple[int, int], ManifestTile] = {}

    for row in rows:
        tile_path = normalize_path(row.get("tile_path") or row.get("image_path"), base_dir=scan_dir)
        raw_path = normalize_path(row.get("raw_path"), base_dir=scan_dir)

        # Support moved cache dirs: fall back by basename.
        if tile_path and not tile_path.is_file():
            fallback = scan_dir / "tiles" / tile_path.name
            if fallback.is_file():
                tile_path = fallback

        if raw_path and not raw_path.is_file():
            fallback = scan_dir / "raw" / raw_path.name
            if fallback.is_file():
                raw_path = fallback

        tile_id = build_tile_id_from_manifest_row(row, tile_path)
        r = safe_int(row.get("row"))
        c = safe_int(row.get("col"))

        item = ManifestTile(
            tile_id=tile_id,
            row=r,
            col=c,
            center_lat=safe_float(row.get("center_lat")),
            center_lon=safe_float(row.get("center_lon")),
            tile_path=tile_path,
            raw_path=raw_path,
            status=str(row.get("status", "")).strip(),
            tile_km=safe_float(row.get("tile_km")),
            zoom=safe_int(row.get("zoom")),
            actual_returned_width_km=safe_float(row.get("actual_returned_width_km")),
            source_row=row,
        )
        by_id[tile_id] = item
        if r is not None and c is not None:
            by_rc[(r, c)] = item

    return by_id, by_rc


def index_result_jsons(run_dir: Path) -> Dict[str, Path]:
    """Map base tile_id -> per_tile result.json path."""
    out: Dict[str, Path] = {}
    per_tile = run_dir / "per_tile"
    if not per_tile.is_dir():
        return out

    for p in per_tile.glob("*/result.json"):
        data = load_json(p)
        if not data:
            continue
        tile_id = (
            data.get("base_tile", {}).get("tile_id")
            or data.get("tile", {}).get("tile_id")
            or p.parent.name
        )
        out[str(tile_id)] = p

    return out


def find_result_json_for_tile(run_dir: Path, tile_id: str, json_index: Dict[str, Path]) -> Optional[Path]:
    if tile_id in json_index:
        return json_index[tile_id]
    candidate = run_dir / "per_tile" / sanitize_tag(tile_id) / "result.json"
    if candidate.is_file():
        return candidate
    return None


def infer_image_path(
    result_row: Dict[str, Any],
    detail: Optional[Dict[str, Any]],
    manifest: Optional[ManifestTile],
    run_dir: Path,
) -> Optional[Path]:
    # Prefer the copied source tile created by two_agent, because it is inside run_dir.
    if detail:
        p = normalize_path(detail.get("source_tile_copy"), base_dir=run_dir)
        if p and p.is_file():
            return p
        p = normalize_path(detail.get("base_tile", {}).get("image_path"), base_dir=run_dir)
        if p and p.is_file():
            return p

    p = normalize_path(result_row.get("image_path"), base_dir=run_dir)
    if p and p.is_file():
        return p

    if manifest and manifest.tile_path and manifest.tile_path.is_file():
        return manifest.tile_path

    return p


def load_review_tiles(run_dir: Path, manifest_by_id: Dict[str, ManifestTile], manifest_by_rc: Dict[Tuple[int, int], ManifestTile]) -> List[ReviewTile]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"results.csv not found in run_dir: {results_path}")

    json_index = index_result_jsons(run_dir)
    rows = read_csv_dicts(results_path)
    tiles: List[ReviewTile] = []

    seen_ids: set[str] = set()
    for row in rows:
        tile_id = str(row.get("tile_id", "")).strip()
        r = safe_int(row.get("manifest_row"))
        c = safe_int(row.get("manifest_col"))

        if not tile_id and r is not None and c is not None:
            tile_id = f"r{r}_c{c}"
        if not tile_id:
            # Last-resort identity from path.
            image_path_tmp = normalize_path(row.get("image_path"), base_dir=run_dir)
            tile_id = image_path_tmp.stem if image_path_tmp else f"tile_{len(tiles)+1}"

        # If results.csv has duplicate rows from repeated/continued runs, keep the latest occurrence.
        # We handle this by removing prior item with same tile_id and appending the new one.
        if tile_id in seen_ids:
            tiles = [t for t in tiles if t.tile_id != tile_id]
        seen_ids.add(tile_id)

        manifest = manifest_by_id.get(tile_id)
        if manifest is None and r is not None and c is not None:
            manifest = manifest_by_rc.get((r, c))

        result_json_path = find_result_json_for_tile(run_dir, tile_id, json_index)
        detail = load_json(result_json_path) if result_json_path else None

        # Prefer detailed result fields if present.
        base_tile = detail.get("base_tile", {}) if detail else {}
        if r is None:
            r = safe_int(base_tile.get("manifest_row"))
        if c is None:
            c = safe_int(base_tile.get("manifest_col"))
        if r is None and manifest:
            r = manifest.row
        if c is None and manifest:
            c = manifest.col

        lat = safe_float(row.get("center_lat"))
        lon = safe_float(row.get("center_lon"))
        if lat is None:
            lat = safe_float(base_tile.get("center_lat"))
        if lon is None:
            lon = safe_float(base_tile.get("center_lon"))
        if lat is None and manifest:
            lat = manifest.center_lat
        if lon is None and manifest:
            lon = manifest.center_lon

        score = safe_int(row.get("parent_score"))
        if detail and score is None:
            score = safe_int(detail.get("parent_score"))
        if score is None and detail:
            score = safe_int(detail.get("parent_brains", {}).get("score_-10_10"))

        final_label = str(row.get("final_label") or (detail or {}).get("final_label") or "unknown").strip().lower()
        parent_nav = str(row.get("parent_navigate") or (detail or {}).get("parent_brains", {}).get("navigate") or "").strip()
        zoom_triggered = safe_bool(row.get("zoom_triggered") or (detail or {}).get("zoom_triggered"))
        aggregation_reason = str(row.get("aggregation_reason") or (detail or {}).get("aggregation_reason") or "").strip()

        image_path = infer_image_path(row, detail, manifest, run_dir)

        tiles.append(
            ReviewTile(
                tile_id=tile_id,
                safe_id=sanitize_tag(tile_id),
                row=r,
                col=c,
                center_lat=lat,
                center_lon=lon,
                image_path=image_path,
                final_label=final_label,
                parent_score=score,
                parent_navigate=parent_nav,
                zoom_triggered=zoom_triggered,
                aggregation_reason=aggregation_reason,
                result_json_path=result_json_path,
                result_json=detail,
                manifest=manifest,
                source_row=row,
            )
        )

    return tiles


# -----------------------------------------------------------------------------
# Assets
# -----------------------------------------------------------------------------


def create_jpeg_variant(src: Optional[Path], dst: Path, max_size: int, quality: int = 85) -> bool:
    """Create a compressed preview while preserving aspect ratio. Never copies the source full-res image."""
    if src is None or not src.is_file():
        return False
    if dst.is_file() and dst.stat().st_size > 0:
        return True
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            ensure_dir(dst.parent)
            im.save(dst, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception as exc:
        print(f"Warning: could not create image asset for {src}: {exc}", file=sys.stderr)
        return False


def create_jpeg_square(src: Optional[Path], dst: Path, size: int, quality: int = 82) -> bool:
    """Create a compressed square thumbnail. This is used for tile mosaics / neighbors."""
    if src is None or not src.is_file():
        return False
    if dst.is_file() and dst.stat().st_size > 0:
        return True
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            ensure_dir(dst.parent)
            im.save(dst, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception as exc:
        print(f"Warning: could not create square thumbnail for {src}: {exc}", file=sys.stderr)
        return False


def copy_result_json(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.is_file():
        return False
    ensure_dir(dst.parent)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        print(f"Warning: could not copy result JSON {src}: {exc}", file=sys.stderr)
        return False


def prepare_tile_assets(
    tiles: List[ReviewTile],
    output_dir: Path,
    thumbnail_size: int,
    preview_size: int,
) -> None:
    """Create only small review images in one folder. Do not copy full-size source tiles."""
    images_dir = ensure_dir(output_dir / "assets" / "images")

    for tile in tiles:
        image_path = images_dir / f"tile_{tile.safe_id}.jpg"
        if create_jpeg_square(tile.image_path, image_path, thumbnail_size, quality=84):
            tile.thumbnail_rel = make_rel(image_path, output_dir)
            tile.preview_rel = tile.thumbnail_rel


def should_render_neighbors_for_tile(tile: ReviewTile, neighbors_mode: str) -> bool:
    if neighbors_mode == "all":
        return True
    if neighbors_mode == "priority":
        return tile.final_label in {"yes", "unresolved"}
    return False


def prepare_context_neighbor_assets(
    tiles: List[ReviewTile],
    manifest_by_rc: Dict[Tuple[int, int], ManifestTile],
    review_by_rc: Dict[Tuple[int, int], ReviewTile],
    output_dir: Path,
    thumbnail_size: int,
    neighbors_mode: str,
) -> Dict[Tuple[int, int], str]:
    """
    Create thumbnails only for context neighbors that are needed in rendered 3x3 grids
    and were not processed by two-agent. This does NOT copy the full 40km cache.
    """
    if neighbors_mode == "none":
        return {}

    needed: set[Tuple[int, int]] = set()
    for tile in tiles:
        if not should_render_neighbors_for_tile(tile, neighbors_mode):
            continue
        if tile.row is None or tile.col is None:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rc = (tile.row + dr, tile.col + dc)
                if rc not in review_by_rc and rc in manifest_by_rc:
                    needed.add(rc)

    out: Dict[Tuple[int, int], str] = {}
    dst_dir = ensure_dir(output_dir / "assets" / "images")
    for rc in sorted(needed):
        mt = manifest_by_rc[rc]
        safe = f"context_r{rc[0]}_c{rc[1]}"
        dst = dst_dir / f"{safe}.jpg"
        if create_jpeg_square(mt.tile_path, dst, thumbnail_size, quality=82):
            out[rc] = make_rel(dst, output_dir)
    return out


def prepare_zoom_assets(tile: ReviewTile, output_dir: Path, preview_size: int) -> None:
    """Mutate tile.result_json zoom_records with HTML-ready relative preview path if image exists."""
    if not tile.result_json:
        return
    records = tile.result_json.get("zoom_records") or []
    if not isinstance(records, list):
        return
    zoom_dir = ensure_dir(output_dir / "assets" / "zoom_previews" / tile.safe_id)
    for i, rec in enumerate(records, start=1):
        ztile = rec.get("tile", {}) if isinstance(rec, dict) else {}
        zpath = normalize_path(ztile.get("image_path"), base_dir=tile.result_json_path.parent if tile.result_json_path else None)
        label = sanitize_tag(ztile.get("variant_label") or ztile.get("tile_id") or f"zoom_{i}")
        dst = zoom_dir / f"{i:02d}_{label}.jpg"
        if create_jpeg_variant(zpath, dst, preview_size, quality=88):
            rec["_preview_rel"] = make_rel(dst, output_dir)
        else:
            rec["_preview_rel"] = None
            rec["_image_missing"] = str(zpath) if zpath else ""


# -----------------------------------------------------------------------------
# Clustering and ordering
# -----------------------------------------------------------------------------


def assign_simple_clusters(tiles: List[ReviewTile], include_labels: Sequence[str] = ("yes", "unresolved")) -> None:
    """Assign connected-component cluster ids by row/col adjacency for selected labels."""
    candidates = {
        (t.row, t.col): t
        for t in tiles
        if t.row is not None and t.col is not None and t.final_label in include_labels
    }
    visited: set[Tuple[int, int]] = set()
    cluster_id = 0

    # 8-neighbor connected components.
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for rc in sorted(candidates):
        if rc in visited:
            continue
        cluster_id += 1
        queue: deque[Tuple[int, int]] = deque([rc])
        visited.add(rc)
        while queue:
            cur = queue.popleft()
            candidates[cur].cluster_id = cluster_id
            r, c = cur
            for dr, dc in neighbor_offsets:
                nxt = (r + dr, c + dc)
                if nxt in candidates and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)


def label_priority(label: str) -> int:
    return {"yes": 0, "unresolved": 1, "no": 2}.get(str(label).lower(), 3)


def sort_tiles_for_review(tiles: List[ReviewTile]) -> List[ReviewTile]:
    def key(t: ReviewTile) -> Tuple[int, int, int, int, str]:
        # YES: high positive score first.
        # UNRESOLVED: scores closest to 0 first, then higher absolute ambiguity.
        # NO: least negative / closest to decision boundary first.
        label = t.final_label
        score = t.parent_score if t.parent_score is not None else -999
        if label == "yes":
            secondary = -score
        elif label == "unresolved":
            secondary = abs(score)
        else:
            secondary = -score
        return (
            label_priority(label),
            secondary,
            t.row if t.row is not None else 999999,
            t.col if t.col is not None else 999999,
            t.tile_id,
        )
    return sorted(tiles, key=key)


def cluster_summaries(tiles: List[ReviewTile]) -> List[Dict[str, Any]]:
    groups: Dict[int, List[ReviewTile]] = defaultdict(list)
    for t in tiles:
        if t.cluster_id is not None:
            groups[t.cluster_id].append(t)

    summaries: List[Dict[str, Any]] = []
    for cid, members in groups.items():
        labels = Counter(t.final_label for t in members)
        scores = [t.parent_score for t in members if t.parent_score is not None]
        lats = [t.center_lat for t in members if t.center_lat is not None]
        lons = [t.center_lon for t in members if t.center_lon is not None]
        rows = [t.row for t in members if t.row is not None]
        cols = [t.col for t in members if t.col is not None]
        best = max(members, key=lambda t: t.parent_score if t.parent_score is not None else -999)
        summaries.append({
            "cluster_id": cid,
            "count": len(members),
            "yes": labels.get("yes", 0),
            "unresolved": labels.get("unresolved", 0),
            "max_score": max(scores) if scores else None,
            "mean_score": statistics.mean(scores) if scores else None,
            "center_lat": statistics.mean(lats) if lats else None,
            "center_lon": statistics.mean(lons) if lons else None,
            "row_min": min(rows) if rows else None,
            "row_max": max(rows) if rows else None,
            "col_min": min(cols) if cols else None,
            "col_max": max(cols) if cols else None,
            "best_tile": best,
        })

    return sorted(summaries, key=lambda d: (-d["yes"], -d["count"], -(d["max_score"] or -999)))


# -----------------------------------------------------------------------------
# HTML rendering helpers
# -----------------------------------------------------------------------------


def score_color(score: Optional[int]) -> str:
    """Map every integer score from -10 to 10 onto a continuous blue→near-white→red ramp.

    Negative scores become progressively deeper blue, 0 is near-white, and positive
    scores become progressively deeper red. Zoomed tiles call this with display_score,
    which is max(child_scores) when zoom child scores exist.
    """
    if score is None:
        return "#d1d5db"

    s = max(-10, min(10, int(round(float(score)))))

    def lerp(a: int, b: int, t: float) -> int:
        return int(round(a + (b - a) * t))

    # Deep but readable endpoints. Neutral is deliberately not pure white so empty
    # cells remain visually distinct from score=0 tiles.
    neg = (37, 99, 235)      # blue-600
    zero = (249, 250, 251)   # gray-50
    pos = (220, 38, 38)      # red-600

    if s < 0:
        t = (s + 10) / 10.0  # -10 -> 0, 0 -> 1
        r = lerp(neg[0], zero[0], t)
        g = lerp(neg[1], zero[1], t)
        b = lerp(neg[2], zero[2], t)
    else:
        t = s / 10.0         # 0 -> 0, 10 -> 1
        r = lerp(zero[0], pos[0], t)
        g = lerp(zero[1], pos[1], t)
        b = lerp(zero[2], pos[2], t)

    return f"#{r:02x}{g:02x}{b:02x}"


def label_class(label: str) -> str:
    label = str(label).lower()
    if label == "yes":
        return "label-yes"
    if label == "unresolved":
        return "label-unresolved"
    if label == "no":
        return "label-no"
    return "label-unknown"


def render_img(rel: Optional[str], alt: str, css_class: str = "tile-img", lazy: bool = True) -> str:
    if not rel:
        return f"<div class='missing-image'>Missing image<br><span>{esc(alt)}</span></div>"
    loading = " loading='lazy'" if lazy else ""
    return f"<img class='{css_class}' src='{esc(rel)}' alt='{esc(alt)}'{loading}>"


def google_maps_url(lat: Optional[float], lon: Optional[float], zoom: int = 16) -> Optional[str]:
    """Create a Google Maps satellite URL centered on the tile. No API key required."""
    if lat is None or lon is None:
        return None
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except Exception:
        return None
    return (
        "https://www.google.com/maps/@?api=1"
        f"&map_action=map&center={lat_f:.8f},{lon_f:.8f}"
        f"&zoom={int(zoom)}&basemap=satellite"
    )


def zoom_child_scores(tile: ReviewTile) -> List[int]:
    records = (tile.result_json or {}).get("zoom_records") or []
    scores: List[int] = []
    if not isinstance(records, list):
        return scores
    for rec in records:
        if not isinstance(rec, dict):
            continue
        brains = rec.get("brains") or {}
        if not isinstance(brains, dict):
            continue
        val = brains.get("score_-10_10")
        try:
            scores.append(int(val))
        except Exception:
            pass
    return scores


def display_score(tile: ReviewTile) -> Optional[int]:
    child_scores = zoom_child_scores(tile)
    if tile.zoom_triggered and child_scores:
        return max(child_scores)
    return tile.parent_score


def zoom_reason_class(tile: ReviewTile) -> str:
    reason = str(tile.aggregation_reason or "").lower()
    if "decisive_positive_child" in reason:
        return " zoom-decisive-positive-child"
    if "children_still_mixed_or_ambiguous" in reason:
        return " zoom-children-mixed"
    if tile.zoom_triggered:
        return " zoom-other"
    return ""


def render_summary_cards(tiles: List[ReviewTile]) -> str:
    counts = Counter(t.final_label for t in tiles)
    scores = [t.parent_score for t in tiles if t.parent_score is not None]
    zoom_count = sum(1 for t in tiles if t.zoom_triggered)
    avg_score = statistics.mean(scores) if scores else None
    max_score = max(scores) if scores else None
    min_score = min(scores) if scores else None

    cards = [
        ("Total reviewed", len(tiles), ""),
        ("Final YES", counts.get("yes", 0), "yes"),
        ("Unresolved", counts.get("unresolved", 0), "unresolved"),
        ("Final NO", counts.get("no", 0), "no"),
        ("Zoom triggered", zoom_count, ""),
        ("Avg score", f"{avg_score:.2f}" if avg_score is not None else "", ""),
        ("Max score", max_score if max_score is not None else "", ""),
        ("Min score", min_score if min_score is not None else "", ""),
    ]

    out = ["<section class='summary-cards'>"]
    for title, value, tag in cards:
        tag_cls = f" card-{tag}" if tag else ""
        out.append(
            f"<div class='summary-card{tag_cls}'><div class='summary-value'>{esc(value)}</div>"
            f"<div class='summary-title'>{esc(title)}</div></div>"
        )
    out.append("</section>")
    return "\n".join(out)


def render_controls() -> str:
    return """
<section class="controls sticky-panel">
  <div class="control-row">
    <label><input type="checkbox" class="label-filter" value="yes" checked> YES</label>
    <label><input type="checkbox" class="label-filter" value="unresolved" checked> Unresolved</label>
    <label><input type="checkbox" class="label-filter" value="no" checked> NO</label>
    <label><input type="checkbox" id="zoomOnly"> Zoom-triggered only</label>
  </div>
  <div class="control-row">
    <label>Search tile/analysis: <input id="searchBox" type="search" placeholder="tile_id, score, text..." /></label>
    <label>Min score: <input id="minScore" type="number" min="-10" max="10" value="-10" /></label>
    <label>Max score: <input id="maxScore" type="number" min="-10" max="10" value="10" /></label>
    <label>Review order:
      <select id="sortMode">
        <option value="default" selected>Label + parent score</option>
        <option value="rowcol">Row → Column</option>
        <option value="colrow">Column → Row</option>
        <option value="display_score_desc">Display score high → low</option>
        <option value="parent_score_desc">Parent score high → low</option>
      </select>
    </label>
    <button type="button" onclick="collapseAllCards()">Collapse all cards</button>
    <button type="button" onclick="expandVisibleYesCards()">Expand visible YES</button>
    <label>Map mode:
      <select id="mapMode">
        <option value="score" selected>Score heat over thumbnails</option>
        <option value="label">Final label over thumbnails</option>
        <option value="image">Image only</option>
      </select>
    </label>
  </div>
  <div class="control-row map-visual-controls">
    <label>Map tile size:
      <input id="mapTileSize" type="range" min="28" max="96" step="2" value="48" />
      <span id="mapTileSizeValue">48</span>px
    </label>
    <label>Score overlay opacity:
      <input id="scoreOpacity" type="range" min="0" max="0.85" step="0.01" value="0.38" />
      <span id="scoreOpacityValue">0.38</span>
    </label>
    <label>Label overlay opacity:
      <input id="labelOpacity" type="range" min="0" max="0.85" step="0.01" value="0.30" />
      <span id="labelOpacityValue">0.30</span>
    </label>
  </div>
  <div class="control-row small-note">
    The map now uses the existing assets/images tile thumbnails. Score heat mode overlays the -10→+10 blue→white→red score color while still letting the tile imagery show through.
  </div>
</section>
"""


def render_tile_map(tiles: List[ReviewTile]) -> str:
    tiles_with_rc = [t for t in tiles if t.row is not None and t.col is not None]
    if not tiles_with_rc:
        return "<section><h2>Tile Map</h2><p>No row/col data found.</p></section>"

    by_rc = {(t.row, t.col): t for t in tiles_with_rc}
    rows = sorted({t.row for t in tiles_with_rc if t.row is not None})
    cols = sorted({t.col for t in tiles_with_rc if t.col is not None})
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    n_cols = max_c - min_c + 1
    out = ["<section class='tile-map-section' id='tile-map'>"]
    out.append("<h2>Tile Score / Label Map</h2>")
    out.append("<div class='legend'>")
    out.append("<span class='score-legend'><span style='background:#2563eb'></span><b>-10</b><span style='background:#f9fafb'></span><b>0</b><span style='background:#dc2626'></span><b>+10</b></span>")
    out.append("<span class='legend-chip label-yes'>YES</span>")
    out.append("<span class='legend-chip label-unresolved'>Unresolved</span>")
    out.append("<span class='legend-chip label-no'>NO</span>")
    out.append("<span class='legend-note'>Each grid cell shows the existing tile thumbnail. Score heat mode draws a transparent blue→white→red score overlay on top. Zoomed tiles use max child score when available. Solid black border = decisive positive child; dashed black border = children still mixed/ambiguous. Solid grey border = all other zoom-triggered.</span>")
    out.append("</div>")
    # out.append(f"<div class='tile-grid tile-thumb-grid' style='grid-template-columns: repeat({n_cols}, var(--map-tile-size));'>")
    out.append(
        f"<div class='tile-grid tile-thumb-grid tile-axis-grid' "
        f"style='grid-template-columns: 44px repeat({n_cols}, var(--map-tile-size));'>"
    )

    # Top-left corner cell
    out.append("<div class='axis-cell axis-corner'>r\\c</div>")

    # Column labels
    for c in range(min_c, max_c + 1):
        out.append(f"<div class='axis-cell axis-col'>c{esc(c)}</div>")

    for r in range(min_r, max_r + 1):
        out.append(f"<div class='axis-cell axis-row'>r{esc(r)}</div>")
        for c in range(min_c, max_c + 1):
            t = by_rc.get((r, c))
            if not t:
                out.append("<div class='map-cell empty-cell' title='No processed tile'></div>")
                continue

            dscore = display_score(t)
            zoom_cls = zoom_reason_class(t)
            label_cls = label_class(t.final_label)
            cls = f"map-cell {label_cls}{zoom_cls}"
            title = (
                f"{t.tile_id} | r{t.row} c{t.col} | "
                f"label={t.final_label} | parent_score={t.parent_score} | display_score={dscore} | nav={t.parent_navigate} | "
                f"{t.aggregation_reason}"
            )
            search_blob = f"{t.tile_id} {t.final_label} {t.parent_score} {dscore} {t.parent_navigate} {t.aggregation_reason}"

            if t.thumbnail_rel:
                thumb_html = (
                    f"<img class='map-thumb' src='{esc(t.thumbnail_rel)}' "
                    f"alt='{esc(t.tile_id)}' loading='lazy'>"
                )
            else:
                thumb_html = "<div class='map-thumb missing-map-thumb'></div>"

            out.append(
                f"<button class='{cls}' id='map-{esc(t.safe_id)}' "
                f"style='--score-color:{score_color(dscore)}' "
                f"data-tile-id='{esc(t.tile_id)}' data-safe-id='{esc(t.safe_id)}' "
                f"data-label='{esc(t.final_label)}' data-score='{esc(dscore if dscore is not None else '')}' "
                f"data-zoom='{str(t.zoom_triggered).lower()}' data-search='{esc(search_blob)}' "
                f"title='{esc(title)}' onclick=\"scrollToTile('{esc(t.safe_id)}')\">"
                f"{thumb_html}"
                f"<span class='map-overlay map-score-overlay'></span>"
                f"<span class='map-overlay map-label-overlay {label_cls}'></span>"
                f"</button>"
            )

    out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def render_cluster_summary(tiles: List[ReviewTile]) -> str:
    summaries = cluster_summaries(tiles)
    if not summaries:
        return ""

    out = ["<details class='cluster-section' id='clusters'>"]
    out.append("<summary><h2>▶ Hotspot Clusters: YES + Unresolved, 8-neighbor adjacency</h2></summary>")
    out.append("<div class='cluster-grid'>")
    for s in summaries:
        best: ReviewTile = s["best_tile"]
        out.append("<div class='cluster-card'>")
        out.append(f"<h3>Cluster {esc(s['cluster_id'])}</h3>")
        out.append("<div class='cluster-body'>")
        out.append(render_img(best.thumbnail_rel, best.tile_id, css_class="cluster-thumb"))
        out.append("<div>")
        out.append(f"<p><b>Tiles:</b> {esc(s['count'])} &nbsp; <b>YES:</b> {esc(s['yes'])} &nbsp; <b>Unresolved:</b> {esc(s['unresolved'])}</p>")
        out.append(f"<p><b>Max score:</b> {esc(s['max_score'])} &nbsp; <b>Mean score:</b> {format_float(s['mean_score'], 2)}</p>")
        out.append(f"<p><b>Center:</b> {format_float(s['center_lat'], 6)}, {format_float(s['center_lon'], 6)}</p>")
        out.append(f"<p><b>Grid bbox:</b> rows {esc(s['row_min'])}–{esc(s['row_max'])}, cols {esc(s['col_min'])}–{esc(s['col_max'])}</p>")
        out.append(f"<p><button onclick=\"scrollToTile('{esc(best.safe_id)}')\">Open highest-score tile: {esc(best.tile_id)}</button></p>")
        out.append("</div></div></div>")
    out.append("</div></details>")
    return "\n".join(out)


def get_eyes_analysis(tile: ReviewTile) -> str:
    data = tile.result_json or {}
    val = data.get("parent_eyes", {}).get("analysis")
    if val is None:
        # Old schema compatibility.
        val = json.dumps(data.get("parent_eyes", {}), ensure_ascii=False, indent=2)
    return str(val or "")


def get_brains(tile: ReviewTile) -> Dict[str, Any]:
    data = tile.result_json or {}
    brains = data.get("parent_brains") or {}
    if not isinstance(brains, dict):
        return {}
    return brains


def render_metadata_table(tile: ReviewTile) -> str:
    dscore = display_score(tile)
    child_scores = zoom_child_scores(tile)
    rows = [
        ("tile_id", tile.tile_id),
        ("final_label", tile.final_label),
        ("parent_score", tile.parent_score),
        ("display_score", dscore if dscore is not None else ""),
        ("max_child_score", max(child_scores) if child_scores else ""),
        ("parent_navigate", tile.parent_navigate),
        ("zoom_triggered", tile.zoom_triggered),
        ("aggregation_reason", tile.aggregation_reason),
        ("cluster_id", tile.cluster_id if tile.cluster_id is not None else ""),
        ("row / col", f"{tile.row} / {tile.col}"),
        ("center_lat / center_lon", f"{format_float(tile.center_lat, 8)} / {format_float(tile.center_lon, 8)}"),
    ]
    if tile.manifest:
        rows.extend([
            ("manifest zoom", tile.manifest.zoom),
            ("manifest actual_width_km", tile.manifest.actual_returned_width_km),
        ])

    out = ["<table class='meta-table'>"]
    for k, v in rows:
        out.append(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def render_brains_block(tile: ReviewTile) -> str:
    brains = get_brains(tile)
    return (
        "<div class='agent-block brains-block'>"
        "<h4>Mr. Brains</h4>"
        f"<pre>{esc(json.dumps(brains, ensure_ascii=False, indent=2))}</pre>"
        "</div>"
    )


def render_eyes_block(tile: ReviewTile) -> str:
    analysis = get_eyes_analysis(tile)
    return (
        "<div class='agent-block eyes-block'>"
        "<h4>Mr. Eyes Analysis</h4>"
        f"<p class='analysis-text'>{esc(analysis)}</p>"
        "</div>"
    )


def render_zoom_records(tile: ReviewTile) -> str:
    records = []
    if tile.result_json:
        records = tile.result_json.get("zoom_records") or []
    if not records:
        return "<div class='zoom-section'><h4>Zoom Records</h4><p>No zoom records.</p></div>"

    out = ["<div class='zoom-section'><h4>Zoom Records</h4>"]
    for i, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue
        ztile = rec.get("tile") or {}
        eyes = rec.get("eyes") or {}
        brains = rec.get("brains") or {}
        out.append("<details class='zoom-card' open>")
        out.append(
            f"<summary>Zoom {i}: {esc(ztile.get('variant_kind', ''))} / "
            f"{esc(ztile.get('variant_label', ''))} — score {esc(brains.get('score_-10_10', ''))}, "
            f"navigate {esc(brains.get('navigate', ''))}</summary>"
        )
        out.append("<div class='zoom-content'>")
        out.append("<div class='zoom-text'>")
        out.append("<table class='meta-table compact'>")
        for k in ["tile_id", "variant_kind", "variant_label", "center_lat", "center_lon"]:
            out.append(f"<tr><th>{esc(k)}</th><td>{esc(ztile.get(k, ''))}</td></tr>")
        out.append("</table>")
        out.append("<h5>Zoom Eyes Analysis</h5>")
        out.append(f"<p class='analysis-text'>{esc(eyes.get('analysis') or json.dumps(eyes, ensure_ascii=False, indent=2))}</p>")
        out.append("<h5>Zoom Brains</h5>")
        out.append(f"<pre>{esc(json.dumps(brains, ensure_ascii=False, indent=2))}</pre>")
        out.append("</div></div></details>")
    out.append("</div>")
    return "\n".join(out)


def render_neighbor_grid(
    tile: ReviewTile,
    review_by_rc: Dict[Tuple[int, int], ReviewTile],
    manifest_by_rc: Dict[Tuple[int, int], ManifestTile],
    context_thumb_by_rc: Dict[Tuple[int, int], str],
) -> str:
    if tile.row is None or tile.col is None:
        return "<div class='neighbors'><h4>3x3 Neighbors</h4><p>No row/col data for this tile.</p></div>"

    offsets = [
        (-1, -1, "NW"), (-1, 0, "N"), (-1, 1, "NE"),
        (0, -1, "W"),   (0, 0, "C"),  (0, 1, "E"),
        (1, -1, "SW"),  (1, 0, "S"),  (1, 1, "SE"),
    ]
    out = ["<div class='neighbors'><h4>3x3 Neighbor Context</h4><div class='neighbor-grid'>"]
    for dr, dc, label in offsets:
        rc = (tile.row + dr, tile.col + dc)
        rt = review_by_rc.get(rc)
        mt = manifest_by_rc.get(rc)
        if rt:
            cls = f"neighbor-cell {label_class(rt.final_label)}"
            if rt.tile_id == tile.tile_id:
                cls += " current-neighbor"
            score = rt.parent_score if rt.parent_score is not None else ""
            out.append(f"<button class='{cls}' onclick=\"scrollToTile('{esc(rt.safe_id)}')\" title='{esc(rt.tile_id)}'>")
            out.append(render_img(rt.thumbnail_rel, rt.tile_id, css_class="neighbor-img"))
            out.append(f"<div class='neighbor-overlay'>{esc(label)} | {esc(rt.final_label)} | {esc(score)}</div>")
            out.append("</button>")
        elif mt:
            thumb = context_thumb_by_rc.get(rc)
            out.append("<div class='neighbor-cell not-run' title='Cached in the larger scan cache, but not processed by this two-agent run'>")
            out.append(render_img(thumb, mt.tile_id, css_class="neighbor-img"))
            out.append(f"<div class='neighbor-overlay'>{esc(label)} | context only</div>")
            out.append("</div>")
        else:
            out.append(f"<div class='neighbor-cell missing-neighbor'><div>{esc(label)}<br>missing</div></div>")
    out.append("</div></div>")
    return "\n".join(out)


def render_tile_card(
    tile: ReviewTile,
    review_by_rc: Dict[Tuple[int, int], ReviewTile],
    manifest_by_rc: Dict[Tuple[int, int], ManifestTile],
    context_thumb_by_rc: Dict[Tuple[int, int], str],
    open_by_default: bool,
    include_neighbors: bool,
) -> str:
    label_cls = label_class(tile.final_label)
    search_text = " ".join([
        tile.tile_id,
        tile.final_label,
        str(tile.parent_score or ""),
        tile.parent_navigate,
        tile.aggregation_reason,
        get_eyes_analysis(tile),
    ])
    open_attr = " open" if open_by_default else ""
    score = tile.parent_score if tile.parent_score is not None else ""
    dscore = display_score(tile)
    dscore_str = dscore if dscore is not None else ""
    zoom_cls = " zoomed" if tile.zoom_triggered else ""
    zoom_reason_cls = zoom_reason_class(tile)

    out = [
        f"<details class='tile-card {label_cls}{zoom_cls}{zoom_reason_cls}' id='tile-{esc(tile.safe_id)}'{open_attr} "
        f"data-tile-id='{esc(tile.tile_id)}' "
        f"data-label='{esc(tile.final_label)}' "
        f"data-score='{esc(dscore_str)}' "
        f"data-parent-score='{esc(tile.parent_score if tile.parent_score is not None else '')}' "
        f"data-display-score='{esc(dscore_str)}' "
        f"data-row='{esc(tile.row if tile.row is not None else '')}' "
        f"data-col='{esc(tile.col if tile.col is not None else '')}' "
        f"data-zoom='{str(tile.zoom_triggered).lower()}' "
        f"data-search='{esc(search_text)}'>"
    ]
    out.append(
        "<summary>"
        f"<span class='pill {label_cls}'>{esc(tile.final_label.upper())}</span> "
        f"<b>{esc(tile.tile_id)}</b> "
        f"<span class='muted'>parent_score={esc(score)} | display_score={esc(dscore_str)} | nav={esc(tile.parent_navigate)} | "
        f"r{esc(tile.row)} c{esc(tile.col)} | cluster={esc(tile.cluster_id if tile.cluster_id is not None else '')}</span>"
        "</summary>"
    )
    out.append("<div class='tile-card-body'>")
    out.append("<div class='tile-main-layout'>")
    out.append("<div class='image-panel'>")
    out.append(render_img(tile.thumbnail_rel, tile.tile_id, css_class="preview-img"))
    gmap = google_maps_url(tile.center_lat, tile.center_lon, zoom=16)
    if gmap:
        out.append(f"<p><a class='map-link-button' href='{esc(gmap)}' target='_blank' rel='noopener'>Open in Google Maps</a></p>")
    out.append("</div>")
    out.append("<div class='detail-panel'>")
    out.append(render_metadata_table(tile))
    out.append(render_eyes_block(tile))
    out.append(render_brains_block(tile))
    out.append("</div>")
    out.append("</div>")

    if include_neighbors:
        out.append(render_neighbor_grid(tile, review_by_rc, manifest_by_rc, context_thumb_by_rc))

    out.append(render_zoom_records(tile))
    out.append("</div></details>")
    return "\n".join(out)


def render_review_cards(
    tiles: List[ReviewTile],
    manifest_by_rc: Dict[Tuple[int, int], ManifestTile],
    output_dir: Path,
    thumbnail_size: int,
    neighbors_mode: str,
    context_thumb_by_rc: Dict[Tuple[int, int], str],
) -> str:
    review_by_rc = {
        (t.row, t.col): t
        for t in tiles
        if t.row is not None and t.col is not None
    }
    ordered = sort_tiles_for_review(tiles)

    out = ["<section class='review-section' id='review-cards'>"]
    out.append("<h2>Per-Tile Review Cards</h2>")
    out.append("<p class='small-note'>YES tiles are placed first. Each card contains the image, Mr. Eyes analysis, Mr. Brains output, result metadata, zoom records, and optionally 3x3 neighbors.</p>")

    for t in ordered:
        open_by_default = t.final_label == "yes"
        if neighbors_mode == "all":
            include_neighbors = True
        elif neighbors_mode == "priority":
            include_neighbors = t.final_label in {"yes", "unresolved"}
        else:
            include_neighbors = False
        out.append(render_tile_card(t, review_by_rc, manifest_by_rc, context_thumb_by_rc, open_by_default, include_neighbors))

    out.append("</section>")
    return "\n".join(out)


def render_all_tiles_table(tiles: List[ReviewTile]) -> str:
    ordered = sort_tiles_for_review(tiles)
    out = ["<section class='table-section'><h2>All Tiles Table</h2>"]
    out.append("<p><button type='button' onclick='downloadAllTilesTableCsv()'>Download All Tiles Table CSV</button></p>")
    out.append("<div class='table-wrap'><table class='all-table'>")
    out.append("<thead><tr><th>Open</th><th>tile_id</th><th>label</th><th>score</th><th>navigate</th><th>zoom?</th><th>cluster</th><th>row</th><th>col</th><th>lat</th><th>lon</th><th>aggregation_reason</th></tr></thead><tbody>")
    for t in ordered:
        score = t.parent_score if t.parent_score is not None else ""
        dscore = display_score(t)
        search_text = " ".join([
            t.tile_id,
            t.final_label,
            str(t.parent_score or ""),
            str(dscore if dscore is not None else ""),
            t.parent_navigate,
            t.aggregation_reason,
        ])
        out.append(
            f"<tr class='{label_class(t.final_label)}' "
            f"data-tile-id='{esc(t.tile_id)}' "
            f"data-label='{esc(t.final_label)}' "
            f"data-score='{esc(dscore if dscore is not None else score)}' "
            f"data-parent-score='{esc(t.parent_score if t.parent_score is not None else '')}' "
            f"data-display-score='{esc(dscore if dscore is not None else '')}' "
            f"data-row='{esc(t.row if t.row is not None else '')}' "
            f"data-col='{esc(t.col if t.col is not None else '')}' "
            f"data-zoom='{str(t.zoom_triggered).lower()}' "
            f"data-search='{esc(search_text)}'>"
            f"<td><button onclick=\"scrollToTile('{esc(t.safe_id)}')\">Open</button></td>"
            f"<td>{esc(t.tile_id)}</td>"
            f"<td>{esc(t.final_label)}</td>"
            f"<td>{esc(score)}</td>"
            f"<td>{esc(t.parent_navigate)}</td>"
            f"<td>{esc(t.zoom_triggered)}</td>"
            f"<td>{esc(t.cluster_id if t.cluster_id is not None else '')}</td>"
            f"<td>{esc(t.row)}</td>"
            f"<td>{esc(t.col)}</td>"
            f"<td>{format_float(t.center_lat, 6)}</td>"
            f"<td>{format_float(t.center_lon, 6)}</td>"
            f"<td>{esc(t.aggregation_reason)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div></section>")
    return "\n".join(out)


def render_css() -> str:
    return """
:root {
  --bg: #f7f7f8;
  --panel: #ffffff;
  --text: #111827;
  --muted: #6b7280;
  --border: #d1d5db;
  --yes: #dc2626;
  --yes-soft: #fee2e2;
  --unresolved: #d97706;
  --unresolved-soft: #fef3c7;
  --no: #4b5563;
  --no-soft: #f3f4f6;
  --unknown: #7c3aed;

  --map-tile-size: 48px;
  --map-score-opacity: 0.38;
  --map-label-opacity: 0.30;
}

* { box-sizing: border-box; }
body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--text); }
header { padding: 24px 32px; background: #111827; color: white; }
header h1 { margin: 0 0 6px 0; font-size: 28px; }
header p { margin: 0; color: #d1d5db; }
main { padding: 20px 32px 60px; }
h2 { margin-top: 28px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
h3, h4, h5 { margin-bottom: 8px; }
a { color: #2563eb; }
button { cursor: pointer; }
.summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 12px; margin: 18px 0; }
.summary-card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.summary-card.card-yes { border-color: #fecaca; background: #fff7f7; }
.summary-card.card-unresolved { border-color: #fde68a; background: #fffaf0; }
.summary-card.card-no { border-color: #e5e7eb; background: #fafafa; }
.summary-value { font-size: 26px; font-weight: 800; }
.summary-title { color: var(--muted); font-size: 13px; }
.controls { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; margin: 16px 0; box-shadow: 0 2px 6px rgba(0,0,0,.05); }
.sticky-panel { position: sticky; top: 0; z-index: 20; }
.control-row { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; margin: 6px 0; }
.control-row input[type="search"] { min-width: 280px; }
.control-row input, .control-row select { padding: 6px 8px; border: 1px solid var(--border); border-radius: 8px; }
.small-note { color: var(--muted); font-size: 13px; }
.legend { display: flex; align-items: center; gap: 10px; margin: 10px 0 14px; flex-wrap: wrap; }
.legend-chip { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; border: 1px solid rgba(0,0,0,.12); }
.legend-note { color: var(--muted); font-size: 13px; }
.score-legend { display: inline-flex; align-items: center; gap: 5px; border: 1px solid rgba(0,0,0,.12); border-radius: 999px; padding: 4px 8px; background: white; font-size: 12px; }
.score-legend span { display: inline-block; width: 22px; height: 12px; border-radius: 4px; border: 1px solid rgba(0,0,0,.12); }
.tile-grid {
  display: grid;
  gap: 1px;
  align-items: start;
  background: #111827;
  border: 1px solid var(--border);
  padding: 8px;
  width: max-content;
  max-width: 100%;
  overflow: auto;
  border-radius: 10px;
}
.tile-axis-grid {
  align-items: stretch;
}

.axis-cell {
  height: var(--map-tile-size);
  min-width: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f9fafb;
  color: #111827;
  border: 1px solid #374151;
  font-size: 11px;
  font-weight: 800;
  line-height: normal;
  z-index: 2;
}

.axis-corner {
  position: sticky;
  left: 0;
  top: 0;
  background: #e5e7eb;
  z-index: 4;
}

.axis-col {
  position: sticky;
  top: 0;
  background: #e5e7eb;
  z-index: 3;
}

.axis-row {
  position: sticky;
  left: 0;
  background: #e5e7eb;
  z-index: 3;
}
.tile-thumb-grid {
  line-height: 0;
}
.map-cell {
  position: relative;
  width: var(--map-tile-size);
  height: var(--map-tile-size);
  border: 1px solid rgba(0,0,0,.22);
  border-radius: 0;
  padding: 0;
  margin: 0;
  overflow: hidden;
  background: #000;
  appearance: none;
  -webkit-appearance: none;
}
.map-cell.empty-cell {
  width: var(--map-tile-size);
  height: var(--map-tile-size);
  background: #ffffff;
  opacity: .22;
  border: 1px solid #f3f4f6;
}
.map-cell.zoom-decisive-positive-child { border: 2px solid #111827; }
.map-cell.zoom-children-mixed { border: 2px dashed #111827; }
.map-cell.zoom-other { border: 2px solid #374151; }
.map-thumb {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  image-rendering: auto;
}
.missing-map-thumb {
  width: 100%;
  height: 100%;
  background: #f3f4f6;
}
.map-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.map-score-overlay {
  display: block;
  background: var(--score-color);
  opacity: var(--map-score-opacity);
}
.map-label-overlay {
  display: none;
  opacity: var(--map-label-opacity);
}
.map-label-overlay.label-yes { background: var(--yes); }
.map-label-overlay.label-unresolved { background: #f59e0b; }
.map-label-overlay.label-no { background: #9ca3af; }
.map-label-overlay.label-unknown { background: var(--unknown); }
body.map-label-mode .map-score-overlay { display: none; }
body.map-label-mode .map-label-overlay { display: block; }
body.map-image-mode .map-score-overlay,
body.map-image-mode .map-label-overlay { display: none; }
.label-yes { background: var(--yes-soft); }
.label-unresolved { background: var(--unresolved-soft); }
.label-no { background: var(--no-soft); }
.label-unknown { background: #ede9fe; }
.pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 800; border: 1px solid rgba(0,0,0,.16); }
.pill.label-yes { color: #7f1d1d; background: #fecaca; }
.pill.label-unresolved { color: #78350f; background: #fde68a; }
.pill.label-no { color: #374151; background: #e5e7eb; }
.cluster-section > summary {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.cluster-section > summary h2 {
  margin: 28px 0 0 0;
  flex: 1;
}
.cluster-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 12px; }
.cluster-card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }
.cluster-body { display: grid; grid-template-columns: 120px 1fr; gap: 12px; align-items: start; }
.cluster-thumb { width: 120px; height: 120px; object-fit: cover; border-radius: 10px; border: 1px solid var(--border); }
.tile-card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; margin: 12px 0; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
.tile-card[open] { box-shadow: 0 4px 14px rgba(0,0,0,.08); }
.tile-card > summary { padding: 12px 14px; background: #fafafa; border-bottom: 1px solid var(--border); list-style: none; }
.tile-card > summary::-webkit-details-marker { display: none; }
.tile-card.label-yes { border-left: 8px solid var(--yes); }
.tile-card.label-unresolved { border-left: 8px solid var(--unresolved); }
.tile-card.label-no { border-left: 8px solid #9ca3af; }
.tile-card.zoomed > summary { background-image: linear-gradient(90deg, rgba(17,24,39,.06), transparent); }
.muted { color: var(--muted); font-size: 13px; }
.tile-card-body { padding: 14px; }
.tile-main-layout { display: grid; grid-template-columns: minmax(220px, 340px) 1fr; gap: 18px; align-items: start; }
.image-panel { position: sticky; top: 112px; }
.preview-img { width: 100%; max-width: 320px; max-height: 320px; object-fit: cover; background: #111827; border-radius: 12px; border: 1px solid var(--border); display:block; }
.tile-img { max-width: 100%; border-radius: 10px; border: 1px solid var(--border); }
.missing-image { width: 100%; min-height: 160px; display: grid; place-items: center; text-align: center; color: var(--muted); border: 1px dashed var(--border); border-radius: 10px; background: #fafafa; padding: 12px; }
.missing-image span { font-size: 11px; word-break: break-word; }
.meta-table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 13px; }
.meta-table th { width: 190px; text-align: left; color: #374151; background: #f9fafb; }
.meta-table th, .meta-table td { border: 1px solid #e5e7eb; padding: 6px 8px; vertical-align: top; word-break: break-word; }
.meta-table.compact th { width: 140px; }
.agent-block { margin: 12px 0; }
.agent-block h4 { margin-top: 0; }
.analysis-text { white-space: pre-wrap; line-height: 1.45; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; }
pre { white-space: pre-wrap; word-break: break-word; background: #111827; color: #e5e7eb; border-radius: 10px; padding: 10px; font-size: 12px; }
.neighbors { margin-top: 18px; }
.neighbor-grid {
  --neighbor-size: 168px;
  display: grid;
  grid-template-columns: repeat(3, var(--neighbor-size));
  grid-auto-rows: var(--neighbor-size);
  gap: 0;
  width: max-content;
  max-width: 100%;
  overflow: auto;
  border: 2px solid #111827;
  background: #111827;
}
.neighbor-cell {
  position: relative;
  width: var(--neighbor-size);
  height: var(--neighbor-size);
  border: 1px solid rgba(17,24,39,.55);
  border-radius: 0;
  overflow: hidden;
  padding: 0;
  margin: 0;
  background: #f3f4f6;
  display: block;
  appearance: none;
}
.neighbor-cell.label-yes { box-shadow: inset 0 0 0 4px var(--yes); }
.neighbor-cell.label-unresolved { box-shadow: inset 0 0 0 4px var(--unresolved); }
.neighbor-cell.label-no { box-shadow: inset 0 0 0 3px #6b7280; }
.current-neighbor { box-shadow: inset 0 0 0 5px #111827, inset 0 0 0 9px #ffffff !important; }
.not-run { box-shadow: inset 0 0 0 4px #6366f1; }
.missing-neighbor { display: grid; place-items: center; color: var(--muted); font-size: 14px; text-align: center; }
.neighbor-img { width: 100%; height: 100%; object-fit: cover; display: block; }
.neighbor-overlay { position: absolute; left: 0; right: 0; bottom: 0; background: rgba(17,24,39,.82); color: white; padding: 4px 6px; font-size: 12px; text-align: center; }
.zoom-section { margin-top: 18px; }
.tile-card.zoom-decisive-positive-child { border-color: #111827; box-shadow: inset 0 0 0 2px #111827; }
.tile-card.zoom-children-mixed { border-style: dashed; border-color: #111827; }
.tile-card.zoom-other { border-color: #374151; }
.map-link-button { display:inline-block; margin-top:8px; padding:8px 12px; border-radius:999px; background:#111827; color:white; text-decoration:none; font-weight:800; font-size:13px; }
.map-link-button:hover { background:#374151; }
.zoom-card { border: 1px solid var(--border); border-radius: 12px; margin: 8px 0; background: #fff; }
.zoom-card summary { padding: 10px 12px; background: #f9fafb; border-radius: 12px; }
.zoom-content { padding: 12px; }
.zoom-img { width: 100%; max-height: 500px; object-fit: contain; background: #111827; border-radius: 10px; border: 1px solid var(--border); }
.warning { color: #b45309; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 8px; }
.table-wrap { max-height: 520px; overflow: auto; border: 1px solid var(--border); border-radius: 12px; background: white; }
.all-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.all-table th, .all-table td { border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; white-space: nowrap; }
.all-table th { position: sticky; top: 0; background: #f9fafb; z-index: 5; }
.hidden-by-filter { display: none !important; }
footer { padding: 24px 32px; color: var(--muted); border-top: 1px solid var(--border); }
@media (max-width: 980px) {
  main { padding: 14px; }
  .tile-main-layout, .zoom-content { grid-template-columns: 1fr; }
  .image-panel { position: static; }
}

#backToTop {
  position: fixed;
  right: 24px;
  bottom: 24px;
  z-index: 9999;
  padding: 10px 14px;
  border: 1px solid rgba(17,24,39,.22);
  border-radius: 999px;
  background: #111827;
  color: white;
  font-weight: 800;
  font-size: 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,.22);
  opacity: 0;
  pointer-events: none;
  transform: translateY(8px);
  transition: opacity .18s ease, transform .18s ease, background .18s ease;
}
#backToTop.visible {
  opacity: 1;
  pointer-events: auto;
  transform: translateY(0);
}
#backToTop:hover {
  background: #374151;
}

@media print {
  #backToTop { display: none; }
  .controls { display: none; }
  .image-panel { position: static; }
  .tile-card:not(.label-yes) { break-inside: avoid; }
}
"""


def render_js() -> str:
    return """
function labelRank(label) {
  if (label === 'yes') return 0;
  if (label === 'unresolved') return 1;
  if (label === 'no') return 2;
  return 3;
}

function numData(el, key, fallback) {
  const raw = el.dataset[key];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function textData(el, key) {
  const raw = el.dataset[key];
  return raw === undefined || raw === null ? '' : String(raw);
}

function defaultSecondary(el) {
  const label = el.dataset.label || '';
  const score = numData(el, 'parentScore', -999);
  if (label === 'yes') return -score;
  if (label === 'unresolved') return Math.abs(score);
  return -score;
}

function compareElements(a, b, mode) {
  const ar = numData(a, 'row', 999999);
  const ac = numData(a, 'col', 999999);
  const br = numData(b, 'row', 999999);
  const bc = numData(b, 'col', 999999);

  const ad = numData(a, 'displayScore', -999999);
  const bd = numData(b, 'displayScore', -999999);

  const ap = numData(a, 'parentScore', -999999);
  const bp = numData(b, 'parentScore', -999999);

  const al = a.dataset.label || '';
  const bl = b.dataset.label || '';

  const aid = textData(a, 'tileId');
  const bid = textData(b, 'tileId');

  if (mode === 'rowcol') {
    return (ar - br) || (ac - bc) || labelRank(al) - labelRank(bl) || (bd - ad) || aid.localeCompare(bid);
  }

  if (mode === 'colrow') {
    return (ac - bc) || (ar - br) || labelRank(al) - labelRank(bl) || (bd - ad) || aid.localeCompare(bid);
  }

  if (mode === 'display_score_desc') {
    return (bd - ad) || labelRank(al) - labelRank(bl) || (ar - br) || (ac - bc) || aid.localeCompare(bid);
  }

  if (mode === 'parent_score_desc') {
    return (bp - ap) || labelRank(al) - labelRank(bl) || (ar - br) || (ac - bc) || aid.localeCompare(bid);
  }

  return labelRank(al) - labelRank(bl) || defaultSecondary(a) - defaultSecondary(b) || (ar - br) || (ac - bc) || aid.localeCompare(bid);
}

function sortChildren(container, selector, mode) {
  if (!container) return;
  const items = Array.from(container.querySelectorAll(selector));
  items.sort((a, b) => compareElements(a, b, mode));
  items.forEach(item => container.appendChild(item));
}

function applySort() {
  const mode = document.getElementById('sortMode')?.value || 'default';

  const cardsContainer = document.getElementById('review-cards');
  sortChildren(cardsContainer, '.tile-card', mode);

  const tableBody = document.querySelector('.all-table tbody');
  sortChildren(tableBody, 'tr', mode);
}

function collapseAllCards() {
  document.querySelectorAll('.tile-card').forEach(el => {
    el.open = false;
  });
}

function expandVisibleYesCards() {
  document.querySelectorAll('.tile-card').forEach(el => {
    if (!el.classList.contains('hidden-by-filter') && el.dataset.label === 'yes') {
      el.open = true;
    }
  });
}

function scrollToTile(safeId) {
  const el = document.getElementById('tile-' + safeId);
  if (!el) return;
  el.open = true;
  el.scrollIntoView({behavior: 'smooth', block: 'start'});
  el.classList.add('flash');
  setTimeout(() => el.classList.remove('flash'), 1200);
}

function csvEscape(value) {
  const s = String(value === undefined || value === null ? '' : value);
  const LF = String.fromCharCode(10);
  const CR = String.fromCharCode(13);
  const needsQuotes = s.includes(',') || s.includes('"') || s.includes(LF) || s.includes(CR);

  if (needsQuotes) {
    return '"' + s.split('"').join('""') + '"';
  }

  return s;
}

function downloadAllTilesTableCsv() {
  const table = document.querySelector('.all-table');
  if (!table) return;

  const rows = Array.from(table.querySelectorAll('tr')).filter(row => {
    return !row.classList.contains('hidden-by-filter');
  });

  const csv = rows.map(row => {
    const cells = Array.from(row.querySelectorAll('th, td')).slice(1);
    return cells.map(cell => csvEscape(cell.innerText.trim())).join(',');
  }).join(String.fromCharCode(10));

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = 'filtered_all_tiles_table.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  URL.revokeObjectURL(url);
}

function selectedLabels() {
  return new Set(Array.from(document.querySelectorAll('.label-filter:checked')).map(x => x.value));
}

function passesFilters(el) {
  const labels = selectedLabels();
  const label = el.dataset.label || '';
  if (!labels.has(label)) return false;

  const zoomOnly = document.getElementById('zoomOnly').checked;
  if (zoomOnly && el.dataset.zoom !== 'true') return false;

  const minScore = parseFloat(document.getElementById('minScore').value || '-10');
  const maxScore = parseFloat(document.getElementById('maxScore').value || '10');
  const scoreRaw = el.dataset.score;
  if (scoreRaw !== undefined && scoreRaw !== '') {
    const score = parseFloat(scoreRaw);
    if (score < minScore || score > maxScore) return false;
  }

  const q = (document.getElementById('searchBox').value || '').toLowerCase().trim();
  if (q) {
    const blob = (el.dataset.search || el.textContent || '').toLowerCase();
    if (!blob.includes(q)) return false;
  }
  return true;
}

function applyFilters() {
  document.querySelectorAll('.tile-card').forEach(el => {
    el.classList.toggle('hidden-by-filter', !passesFilters(el));
  });
  document.querySelectorAll('.map-cell:not(.empty-cell)').forEach(el => {
    el.classList.toggle('hidden-by-filter', !passesFilters(el));
  });
  document.querySelectorAll('.all-table tbody tr').forEach(el => {
    el.classList.toggle('hidden-by-filter', !passesFilters(el));
  });
}

function updateMapMode() {
  const mode = document.getElementById('mapMode').value;
  document.body.classList.toggle('map-score-mode', mode === 'score');
  document.body.classList.toggle('map-label-mode', mode === 'label');
  document.body.classList.toggle('map-image-mode', mode === 'image');
}

function setRootVar(name, value) {
  document.documentElement.style.setProperty(name, value);
}

function updateMapVisualControls() {
  const tileSize = document.getElementById('mapTileSize').value;
  const scoreOpacity = document.getElementById('scoreOpacity').value;
  const labelOpacity = document.getElementById('labelOpacity').value;

  setRootVar('--map-tile-size', tileSize + 'px');
  setRootVar('--map-score-opacity', scoreOpacity);
  setRootVar('--map-label-opacity', labelOpacity);

  document.getElementById('mapTileSizeValue').textContent = tileSize;
  document.getElementById('scoreOpacityValue').textContent = Number(scoreOpacity).toFixed(2);
  document.getElementById('labelOpacityValue').textContent = Number(labelOpacity).toFixed(2);
}

function updateBackToTopButton() {
  const btn = document.getElementById('backToTop');
  if (!btn) return;
  btn.classList.toggle('visible', window.scrollY > 500);
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.label-filter, #zoomOnly, #minScore, #maxScore, #searchBox').forEach(el => {
    el.addEventListener('input', applyFilters);
    el.addEventListener('change', applyFilters);
  });

  document.getElementById('sortMode').addEventListener('change', () => {
    applySort();
    applyFilters();
  });

  document.getElementById('mapMode').addEventListener('change', updateMapMode);

  document.querySelectorAll('#mapTileSize, #scoreOpacity, #labelOpacity').forEach(el => {
    el.addEventListener('input', updateMapVisualControls);
    el.addEventListener('change', updateMapVisualControls);
  });

  const backToTop = document.getElementById('backToTop');
  if (backToTop) {
    backToTop.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  window.addEventListener('scroll', updateBackToTopButton);

  updateMapMode();
  updateMapVisualControls();
  applySort();
  applyFilters();
  updateBackToTopButton();
});
"""


def render_html(
    tiles: List[ReviewTile],
    manifest_by_rc: Dict[Tuple[int, int], ManifestTile],
    output_dir: Path,
    scan_dir: Optional[Path],
    run_dir: Path,
    thumbnail_size: int,
    neighbors_mode: str,
    context_thumb_by_rc: Dict[Tuple[int, int], str],
) -> str:
    title = f"Two-Agent Tile Review Report — {run_dir.name}"
    scan_text = str(scan_dir) if scan_dir is not None else "not supplied; report uses run_dir results only"
    body = []
    body.append(f"<header><h1>{esc(title)}</h1><p>Static offline report generated from run_dir: {esc(run_dir)}<br>scan_dir: {esc(scan_text)}<br><b>Scope:</b> only tiles present in this two-agent run results.csv are included.</p></header>")
    body.append("<main>")
    body.append(render_summary_cards(tiles))
    body.append(render_controls())
    body.append(render_tile_map(tiles))
    body.append(render_cluster_summary(tiles))
    body.append(render_all_tiles_table(tiles))
    body.append(render_review_cards(tiles, manifest_by_rc, output_dir, thumbnail_size, neighbors_mode, context_thumb_by_rc))
    body.append("</main>")
    body.append("<footer>Generated by make_two_agent_static_report.py. This report is static; all review images are stored in the assets folder.</footer>")
    body.append("<button id='backToTop' type='button' title='Back to top'>↑ Top</button>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>{render_css()}</style>
</head>
<body class="map-score-mode">
{''.join(body)}
<script>{render_js()}</script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a static offline HTML review report for two-agent cached-tile results."
    )
    p.add_argument("--scan_dir", required=False, default=None, type=Path, help="Optional directory containing manifest.csv and tiles/ from scan_tiles_overscan_crop.py. Used only as fallback metadata / not-run neighbor thumbnails; report scope remains run_dir results.csv.")
    p.add_argument("--run_dir", required=True, type=Path, help="Directory containing results.csv and per_tile/ from two_agent_cached_tiles.py. This defines the report scope.")
    p.add_argument("--output_dir", required=True, type=Path, help="Output report directory")
    p.add_argument("--thumbnail_size", type=int, default=320, help="Pixel size for per-tile and optional context thumbnails stored in assets/images")
    p.add_argument("--preview_size", type=int, default=1100, help="Deprecated; kept for compatibility. Card images use --thumbnail_size.")
    p.add_argument(
        "--neighbors",
        choices=["all", "priority", "none"],
        default="priority",
        help="Render 3x3 neighbors for all tiles, only yes/unresolved tiles, or none. Default priority keeps the report lighter.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite index.html if it already exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scan_dir = args.scan_dir.expanduser().resolve() if args.scan_dir is not None else None
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if scan_dir is not None and not scan_dir.is_dir():
        print(f"Error: scan_dir does not exist: {scan_dir}", file=sys.stderr)
        return 2
    if not run_dir.is_dir():
        print(f"Error: run_dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    index_path = output_dir / "index.html"
    if index_path.exists() and not args.overwrite:
        print(f"Error: {index_path} already exists. Use --overwrite to replace it.", file=sys.stderr)
        return 2

    # Preserve existing assets/images so already-generated thumbnails can be reused.
    # With --overwrite, only replace index.html; use a fresh output_dir if you want to rebuild assets.
    ensure_dir(output_dir)
    if index_path.exists() and args.overwrite:
        index_path.unlink()

    if scan_dir is not None:
        print("Loading optional scan manifest for fallback metadata / neighbor thumbnails...")
        manifest_by_id, manifest_by_rc = load_manifest(scan_dir)
        print(f"  Manifest tiles available as optional context: {len(manifest_by_id)}")
    else:
        manifest_by_id, manifest_by_rc = {}, {}
        print("No scan_dir supplied. Report will use only this two-agent run_dir results.")

    print("Loading two-agent results. This defines the report scope...")
    tiles = load_review_tiles(run_dir, manifest_by_id, manifest_by_rc)
    print(f"  Review tiles: {len(tiles)}")

    if not tiles:
        print("Error: no review tiles found in results.csv", file=sys.stderr)
        return 2

    print("Assigning simple YES/UNRESOLVED clusters...")
    assign_simple_clusters(tiles)
    n_clusters = len({t.cluster_id for t in tiles if t.cluster_id is not None})
    print(f"  Clusters: {n_clusters}")

    print("Reusing existing thumbnails when present; creating only missing thumbnails in assets/images...")
    prepare_tile_assets(tiles, output_dir, args.thumbnail_size, args.preview_size)

    # Zoom image files are not copied or rendered by default. Zoom analysis and scores are shown from result.json.

    review_by_rc = {
        (t.row, t.col): t
        for t in tiles
        if t.row is not None and t.col is not None
    }
    print("Preparing optional context-only neighbor thumbnails...")
    context_thumb_by_rc = prepare_context_neighbor_assets(
        tiles=tiles,
        manifest_by_rc=manifest_by_rc,
        review_by_rc=review_by_rc,
        output_dir=output_dir,
        thumbnail_size=args.thumbnail_size,
        neighbors_mode=args.neighbors,
    )
    print(f"  Context-only neighbor thumbnails: {len(context_thumb_by_rc)}")

    print("Rendering HTML...")
    html_text = render_html(
        tiles=tiles,
        manifest_by_rc=manifest_by_rc,
        output_dir=output_dir,
        scan_dir=scan_dir,
        run_dir=run_dir,
        thumbnail_size=args.thumbnail_size,
        neighbors_mode=args.neighbors,
        context_thumb_by_rc=context_thumb_by_rc,
    )
    index_path.write_text(html_text, encoding="utf-8")

    counts = Counter(t.final_label for t in tiles)
    print("\nDone.")
    print(f"  Output: {index_path}")
    print(f"  Counts: {dict(counts)}")
    print("  Open index.html in a browser. If sharing, zip the entire output_dir folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
