#!/usr/bin/env python3
"""
Scan a circular area around an input coordinate using center points on a local meter grid.

Modes:
1) Default overscan+crop mode:
   - Download one Google Static Maps image per tile center.
   - Overscan raw image.
   - Center-crop to tile_km.

2) Legacy --mosaic2x2 mode:
   - Fetch four Static Maps images around each tile center.
   - Stitch into a 2x2 mosaic.
   - Save the full mosaic without crop.
   - This preserves the behavior of the previous script, but it can create
     geographic overlap/gap if the mosaic geographic width != tile_km.

3) Recommended --auto_mosaic mode:
   - Dynamically choose zoom from latitude and desired final GSD.
   - Dynamically choose mosaic grid size, e.g. 2x2, 3x3, 4x4, etc.
   - Stitch enough source imagery to cover tile_km.
   - Center-crop exactly tile_km x tile_km in native pixels.
   - Resize to output_px x output_px.
   - This keeps final geographic size, pixel size, and effective GSD consistent
     across different latitudes.

Example recommended command:

python scripts/scan_tiles_overscan_crop.py \
  --lat 34.88 \
  --lon 51.00 \
  --radius_km 3 \
  --tile_km 1.9 \
  --size 640 \
  --scale 2 \
  --maptype satellite \
  --auto_mosaic \
  --output_px 2560 \
  --skip_raw \
  --output ../IMINT-Images/fordow_30km_tiles_auto_mosaic
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image


EARTH_RADIUS_M = 6371008.8
GOOGLE_BASE_RESOLUTION_M_PER_PX = 156543.03392


@dataclass(frozen=True)
class TileCenter:
    row: int
    col: int
    x_m: float
    y_m: float
    lat: float
    lon: float
    distance_m: float


@dataclass(frozen=True)
class AutoMosaicPlan:
    target_km: float
    output_px: int
    target_mpp: float
    zoom: int
    native_mpp: float
    crop_px: int
    single_px: int
    single_width_m: float
    grid_n: int
    mosaic_width_km: float


def parse_coord(value: str) -> float:
    s = str(value).strip().upper().replace(" ", "").replace("°", "")
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([NSEW]?)", s)
    if not m:
        raise ValueError(f"Could not parse coordinate: {value}")
    num = float(m.group(1))
    hemi = m.group(2)
    if hemi in ("S", "W"):
        num = -abs(num)
    elif hemi in ("N", "E"):
        num = abs(num)
    return num


def normalize_longitude(lon: float) -> float:
    normalized = ((lon + 180.0) % 360.0) - 180.0
    if math.isclose(normalized, -180.0):
        return 180.0
    return normalized


def local_xy_to_latlon(
    lat0_deg: float,
    lon0_deg: float,
    x_m: float,
    y_m: float,
) -> tuple[float, float]:
    """
    Local tangent-plane approximation.

    For ~1-3 km tiles this is usually accurate enough for sampling grids.
    If you later need survey-grade bounds, replace this with pyproj/geodesic logic.
    """
    lat0_rad = math.radians(lat0_deg)
    dlat = (y_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (x_m / (EARTH_RADIUS_M * max(1e-12, math.cos(lat0_rad)))) * (180.0 / math.pi)
    lat = lat0_deg + dlat
    lon = normalize_longitude(lon0_deg + dlon)
    return lat, lon


def generate_tile_centers(
    lat0: float,
    lon0: float,
    radius_km: float,
    tile_km: float,
) -> list[TileCenter]:
    """
    Generate tile centers with center-to-center spacing equal to tile_km.

    If the final saved tile geographic width is also tile_km, neighboring
    saved tiles should be logically edge-to-edge: no intentional gap/overlap.
    """
    radius_m = radius_km * 1000.0
    tile_m = tile_km * 1000.0
    n = math.ceil(radius_m / tile_m)

    centers: list[TileCenter] = []

    for row_offset in range(n, -n - 1, -1):
        y_m = row_offset * tile_m
        row = n - row_offset

        for col_offset in range(-n, n + 1):
            x_m = col_offset * tile_m
            dist_m = math.hypot(x_m, y_m)

            if dist_m > radius_m + 1e-9:
                continue

            col = col_offset + n
            lat, lon = local_xy_to_latlon(lat0, lon0, x_m, y_m)

            centers.append(
                TileCenter(
                    row=row,
                    col=col,
                    x_m=x_m,
                    y_m=y_m,
                    lat=lat,
                    lon=lon,
                    distance_m=dist_m,
                )
            )

    centers.sort(key=lambda c: (c.row, c.col))
    return centers


def returned_m_per_px(lat_deg: float, zoom: int, scale: int = 2) -> float:
    """
    Returned-image meter per pixel.

    For Google Static Maps:
      geographic_mpp_at_zoom = base * cos(lat) / 2**zoom
      returned pixel mpp = geographic_mpp_at_zoom / scale

    If size=640 and scale=2, Google commonly returns 1280 px,
    and each returned pixel is half the meter size of scale=1.
    """
    return (
        GOOGLE_BASE_RESOLUTION_M_PER_PX
        * math.cos(math.radians(lat_deg))
        / (2 ** zoom)
        / scale
    )


def actual_returned_width_km(
    lat_deg: float,
    zoom: int,
    returned_px: int,
    scale: int = 2,
) -> float:
    return returned_m_per_px(lat_deg, zoom, scale=scale) * returned_px / 1000.0


def calculate_zoom(
    view_km: float,
    size_px: int,
    lat_deg: float,
    scale: int = 2,
    strategy: str = "floor",
) -> int:
    """
    Legacy helper for the original overscan+crop mode.
    """
    if view_km <= 0:
        raise ValueError("view_km must be positive")
    if size_px <= 0:
        raise ValueError("size_px must be positive")
    if scale not in (1, 2):
        raise ValueError("scale must be 1 or 2")

    cos_lat = math.cos(math.radians(lat_deg))
    if abs(cos_lat) < 1e-8:
        raise ValueError("Latitude too close to the poles")

    returned_px = size_px * scale
    target_m_per_returned_px = (view_km * 1000.0) / returned_px

    raw_zoom = math.log2(
        (GOOGLE_BASE_RESOLUTION_M_PER_PX * cos_lat)
        / (target_m_per_returned_px * scale)
    )

    if strategy == "floor":
        zoom = math.floor(raw_zoom)
    elif strategy == "ceil":
        zoom = math.ceil(raw_zoom)
    elif strategy == "round":
        zoom = round(raw_zoom)
    else:
        raise ValueError("zoom strategy must be one of floor/ceil/round")

    return int(max(0, min(21, zoom)))


def choose_zoom_for_target_mpp(
    *,
    lat_deg: float,
    target_mpp: float,
    scale: int,
    min_zoom: int = 0,
    max_zoom: int = 21,
    mpp_tolerance: float = 0.0,
) -> int:
    """
    Choose the lowest zoom whose native returned-pixel mpp is <= target_mpp.

    This avoids using source imagery that is coarser than the requested final
    effective GSD. If mpp_tolerance > 0, a small amount of coarser native mpp
    is allowed to preserve older behavior or reduce requests.

    Example:
      target_mpp = 1900 / 2560 ~= 0.742 m/px

    At lat ~= 51.5, z16 is approximately this value.
    At lat ~= 34.88, z16 is too coarse, so this should choose z17.
    """
    if target_mpp <= 0:
        raise ValueError("target_mpp must be positive")
    if scale not in (1, 2):
        raise ValueError("scale must be 1 or 2")
    if min_zoom < 0 or max_zoom > 21 or min_zoom > max_zoom:
        raise ValueError("Invalid zoom range")

    allowed_mpp = target_mpp * (1.0 + mpp_tolerance)

    for zoom in range(min_zoom, max_zoom + 1):
        mpp = returned_m_per_px(lat_deg, zoom, scale=scale)
        if mpp <= allowed_mpp:
            return zoom

    raise ValueError(
        f"Could not find zoom with mpp <= {allowed_mpp:.6f} "
        f"for lat={lat_deg:.6f}, scale={scale}"
    )


def choose_grid_n_for_crop(
    *,
    target_width_m: float,
    single_width_m: float,
    overlap_m: float,
    min_grid: int = 2,
    max_grid: int = 8,
) -> tuple[int, float]:
    """
    Choose the smallest grid_n such that geographic mosaic width covers target_width_m.

    If overlap_m=0:
      mosaic_width_m = single_width_m * grid_n

    More generally:
      spacing_m = single_width_m - overlap_m
      mosaic_width_m = single_width_m + spacing_m * (grid_n - 1)
    """
    if target_width_m <= 0:
        raise ValueError("target_width_m must be positive")
    if single_width_m <= 0:
        raise ValueError("single_width_m must be positive")
    if min_grid <= 0:
        raise ValueError("min_grid must be positive")
    if max_grid < min_grid:
        raise ValueError("max_grid must be >= min_grid")

    spacing_m = max(1.0, single_width_m - overlap_m)

    for grid_n in range(min_grid, max_grid + 1):
        mosaic_width_m = single_width_m + spacing_m * (grid_n - 1)
        if mosaic_width_m >= target_width_m:
            return grid_n, mosaic_width_m

    raise ValueError(
        f"Need mosaic wider than max_grid={max_grid}. "
        f"target_width_m={target_width_m:.3f}, "
        f"single_width_m={single_width_m:.3f}, overlap_m={overlap_m:.3f}"
    )


def build_static_map_url(
    *,
    center_lat: float,
    center_lon: float,
    zoom: int,
    size_px: int,
    scale: int,
    maptype: str,
    api_key: str,
    timestamp: Optional[str] = None,
) -> str:
    params = {
        "center": f"{center_lat:.8f},{center_lon:.8f}",
        "zoom": str(zoom),
        "size": f"{size_px}x{size_px}",
        "scale": str(scale),
        "maptype": maptype,
        "key": api_key,
    }

    if timestamp:
        params["timestamp"] = timestamp

    return "https://maps.googleapis.com/maps/api/staticmap?" + urlencode(params)


def google_returned_no_imagery(content: bytes) -> bool:
    text = content.decode("utf-8", errors="ignore")
    return "Sorry, we have no imagery here" in text


def fetch_static_map_image(
    *,
    session: requests.Session,
    api_key: str,
    lat_deg: float,
    lon_deg: float,
    zoom: int,
    size_px: int,
    scale: int,
    maptype: str,
    timestamp: Optional[str],
    timeout: int,
) -> tuple[Image.Image, str]:
    url = build_static_map_url(
        center_lat=lat_deg,
        center_lon=lon_deg,
        zoom=zoom,
        size_px=size_px,
        scale=scale,
        maptype=maptype,
        api_key=api_key,
        timestamp=timestamp,
    )

    resp = session.get(url, timeout=timeout)
    content_type = resp.headers.get("Content-Type", "")
    resp.raise_for_status()

    if google_returned_no_imagery(resp.content):
        raise RuntimeError("Google returned no imagery")

    try:
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Image decode failed. Content-Type={content_type}. "
            f"First 200 bytes={resp.content[:200]!r}. Error={exc}"
        )

    if img.size[0] != img.size[1]:
        raise RuntimeError(f"Returned image is not square: {img.size}")

    return img, url


def choose_workable_zoom_by_fetch(
    *,
    session: requests.Session,
    api_key: str,
    lat_deg: float,
    lon_deg: float,
    tile_km: float,
    overscan_km: float,
    size_px: int,
    scale: int,
    maptype: str,
    timestamp: Optional[str],
    strategy: str,
    timeout: int,
) -> tuple[int, float, float, int, Image.Image, str]:
    """
    Original overscan+crop zoom chooser.

    It actually fetches a candidate image and checks whether the returned image
    can be center-cropped to tile_km.
    """
    zoom = calculate_zoom(
        view_km=overscan_km,
        size_px=size_px,
        lat_deg=lat_deg,
        scale=scale,
        strategy=strategy,
    )

    while zoom >= 0:
        img, url = fetch_static_map_image(
            session=session,
            api_key=api_key,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            zoom=zoom,
            size_px=size_px,
            scale=scale,
            maptype=maptype,
            timestamp=timestamp,
            timeout=timeout,
        )

        actual_w_px, actual_h_px = img.size
        if actual_w_px != actual_h_px:
            raise RuntimeError(f"Returned image is not square: {img.size}")

        mpp = returned_m_per_px(lat_deg, zoom, scale=scale)
        actual_width_km = actual_returned_width_km(
            lat_deg=lat_deg,
            zoom=zoom,
            returned_px=actual_w_px,
            scale=scale,
        )
        crop_px = int(round((tile_km * 1000.0) / mpp))

        if crop_px <= actual_w_px and actual_width_km >= tile_km:
            return zoom, mpp, actual_width_km, crop_px, img, url

        zoom -= 1

    raise ValueError("Could not find a workable zoom after real fetch checks.")


def crop_center_square(img: Image.Image, crop_px: int) -> Image.Image:
    w, h = img.size

    if crop_px <= 0:
        raise ValueError(f"crop_px must be positive, got {crop_px}")

    if crop_px > min(w, h):
        raise ValueError(f"crop_px={crop_px} exceeds image size {w}x{h}")

    left = (w - crop_px) // 2
    top = (h - crop_px) // 2
    right = left + crop_px
    bottom = top + crop_px

    return img.crop((left, top, right, bottom))


def resize_square(img: Image.Image, output_px: int) -> Image.Image:
    if output_px <= 0:
        raise ValueError(f"output_px must be positive, got {output_px}")

    if img.size == (output_px, output_px):
        return img

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return img.resize((output_px, output_px), resampling)


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(s)).strip("._")


def fetch_mosaic_grid_tile(
    *,
    session: requests.Session,
    api_key: str,
    center_lat: float,
    center_lon: float,
    zoom: int,
    grid_n: int,
    size_px: int,
    scale: int,
    maptype: str,
    timestamp: Optional[str],
    timeout: int,
    overlap_m: float = 0.0,
    expected_single_px: Optional[int] = None,
) -> tuple[Image.Image, float, float, int, int, str]:
    """
    Fetch an NxN mosaic centered around one logical tile center.

    For grid_n=2, subimage center offsets are:
      -0.5 spacing, +0.5 spacing

    For grid_n=4:
      -1.5 spacing, -0.5 spacing, +0.5 spacing, +1.5 spacing

    This keeps the whole mosaic centered on the requested tile center.

    Returns:
      mosaic_img,
      mpp,
      mosaic_width_km,
      single_image_px,
      mosaic_image_px,
      representative_url
    """
    if grid_n <= 0:
        raise ValueError("grid_n must be positive")

    # Probe one image at the parent center to determine actual returned size.
    probe_img, representative_url = fetch_static_map_image(
        session=session,
        api_key=api_key,
        lat_deg=center_lat,
        lon_deg=center_lon,
        zoom=zoom,
        size_px=size_px,
        scale=scale,
        maptype=maptype,
        timestamp=timestamp,
        timeout=timeout,
    )

    single_px = probe_img.size[0]

    if expected_single_px is not None and single_px != expected_single_px:
        # Not fatal; Google may clamp or otherwise return a different size.
        # We record actual size and continue.
        pass

    mpp = returned_m_per_px(center_lat, zoom, scale=scale)
    single_width_m = mpp * single_px
    spacing_m = max(1.0, single_width_m - overlap_m)

    mosaic = Image.new("RGB", (single_px * grid_n, single_px * grid_n))

    # Centered coordinate indices.
    # grid_n=2 -> -0.5, +0.5
    # grid_n=3 -> -1, 0, +1
    start = -(grid_n - 1) / 2.0

    for row in range(grid_n):
        for col in range(grid_n):
            x_index = start + col
            y_index = -start - row

            x_m = x_index * spacing_m
            y_m = y_index * spacing_m

            lat, lon = local_xy_to_latlon(center_lat, center_lon, x_m, y_m)

            img, _url = fetch_static_map_image(
                session=session,
                api_key=api_key,
                lat_deg=lat,
                lon_deg=lon,
                zoom=zoom,
                size_px=size_px,
                scale=scale,
                maptype=maptype,
                timestamp=timestamp,
                timeout=timeout,
            )

            if img.size != (single_px, single_px):
                raise RuntimeError(
                    f"Mosaic subimage returned different size: "
                    f"expected {(single_px, single_px)}, got {img.size}"
                )

            mosaic.paste(img, (col * single_px, row * single_px))

    mosaic_width_km = (
        single_width_m + spacing_m * (grid_n - 1)
    ) / 1000.0

    return (
        mosaic,
        mpp,
        mosaic_width_km,
        single_px,
        single_px * grid_n,
        representative_url,
    )


def build_auto_mosaic_plan(
    *,
    session: requests.Session,
    api_key: str,
    center_lat: float,
    center_lon: float,
    target_km: float,
    output_px: int,
    size_px: int,
    scale: int,
    maptype: str,
    timestamp: Optional[str],
    timeout: int,
    overlap_m: float,
    min_zoom: int,
    max_zoom: int,
    force_zoom: Optional[int],
    max_grid: int,
    mpp_tolerance: float,
) -> AutoMosaicPlan:
    """
    Decide zoom and grid size for one tile center.

    This uses one real fetch to get the actual returned single-image pixel size,
    because Static Maps behavior can differ from the naive size_px * scale assumption.
    """
    if target_km <= 0:
        raise ValueError("target_km must be positive")
    if output_px <= 0:
        raise ValueError("output_px must be positive")

    target_m = target_km * 1000.0
    target_mpp = target_m / output_px

    if force_zoom is None:
        zoom = choose_zoom_for_target_mpp(
            lat_deg=center_lat,
            target_mpp=target_mpp,
            scale=scale,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            mpp_tolerance=mpp_tolerance,
        )
    else:
        if not (0 <= force_zoom <= 21):
            raise ValueError("--force_zoom must be between 0 and 21")
        zoom = force_zoom

    # Probe actual image size.
    probe_img, _url = fetch_static_map_image(
        session=session,
        api_key=api_key,
        lat_deg=center_lat,
        lon_deg=center_lon,
        zoom=zoom,
        size_px=size_px,
        scale=scale,
        maptype=maptype,
        timestamp=timestamp,
        timeout=timeout,
    )

    single_px = probe_img.size[0]
    native_mpp = returned_m_per_px(center_lat, zoom, scale=scale)
    single_width_m = native_mpp * single_px
    crop_px = int(round(target_m / native_mpp))

    grid_n, mosaic_width_m = choose_grid_n_for_crop(
        target_width_m=target_m,
        single_width_m=single_width_m,
        overlap_m=overlap_m,
        min_grid=2,
        max_grid=max_grid,
    )

    # Pixel coverage check. This protects against rounding edge cases.
    if crop_px > single_px * grid_n:
        while crop_px > single_px * grid_n and grid_n < max_grid:
            grid_n += 1

        if crop_px > single_px * grid_n:
            raise ValueError(
                f"crop_px={crop_px} exceeds mosaic pixels "
                f"{single_px * grid_n}; increase --max_mosaic_grid"
            )

    return AutoMosaicPlan(
        target_km=target_km,
        output_px=output_px,
        target_mpp=target_mpp,
        zoom=zoom,
        native_mpp=native_mpp,
        crop_px=crop_px,
        single_px=single_px,
        single_width_m=single_width_m,
        grid_n=grid_n,
        mosaic_width_km=mosaic_width_m / 1000.0,
    )


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "row",
        "col",
        "x_m",
        "y_m",
        "center_lat",
        "center_lon",
        "distance_from_origin_m",

        "mode",
        "tile_km",
        "overscan_km_requested",

        "zoom",
        "mosaic_grid_n",
        "mosaic_overlap_m",

        "target_mpp",
        "native_mpp",
        "returned_m_per_px",

        "actual_returned_width_km",
        "mosaic_width_km",

        "crop_px",
        "tile_image_px",
        "raw_image_px",
        "single_image_px",
        "output_px",

        "tile_path",
        "raw_path",
        "url",
        "status",
        "message",
        "timestamp_utc",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_preview_html(path: Path, tile_records: list[dict], title: str) -> None:
    ok_records = [r for r in tile_records if r["status"] in ("downloaded", "cached")]

    if not ok_records:
        path.write_text(
            "<html><body><p>No tiles available.</p></body></html>",
            encoding="utf-8",
        )
        return

    rows = sorted({int(r["row"]) for r in ok_records})
    cols = sorted({int(r["col"]) for r in ok_records})
    rec_by_rc = {(int(r["row"]), int(r["col"])): r for r in ok_records}

    css = """
    body { font-family: Arial, sans-serif; margin: 20px; }
    h2 { margin-bottom: 12px; }
    .grid { display: grid; gap: 0; align-items: start; }
    .cell { width: 160px; border: none; font-size: 10px; background: #fafafa; }
    .cell img { width: 160px; display: block; }
    .empty { width: 160px; height: 160px; background: #eee; }
    .label { padding: 2px; word-break: break-word; }
    """

    html = ["<html><head><meta charset='utf-8'><title>Tile Preview</title>"]
    html.append(f"<style>{css}</style></head><body>")
    html.append(f"<h2>{title}</h2>")
    html.append(f"<div class='grid' style='grid-template-columns: repeat({len(cols)}, 160px);'>")

    for row in rows:
        for col in cols:
            rec = rec_by_rc.get((row, col))

            if rec is None:
                html.append("<div class='empty'></div>")
                continue

            try:
                rel_path = Path(rec["tile_path"]).relative_to(path.parent).as_posix()
            except ValueError:
                rel_path = Path(rec["tile_path"]).as_posix()

            html.append("<div class='cell'>")
            html.append(f"<img src='{rel_path}' alt='tile r{row} c{col}'>")
            html.append(
                f"<div class='label'>"
                f"r{row} c{col}<br>"
                f"{rec['center_lat']}, {rec['center_lon']}<br>"
                f"mode={rec.get('mode', '')}<br>"
                f"z={rec.get('zoom', '')}, grid={rec.get('mosaic_grid_n', '')}"
                f"</div>"
            )
            html.append("</div>")

    html.append("</div></body></html>")
    path.write_text("\n".join(html), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a circular area with stable tile centers. Supports original "
            "overscan+crop, legacy mosaic2x2, and dynamic auto_mosaic."
        )
    )

    parser.add_argument("--lat", required=True, help="Center latitude, e.g. 51.52 or 51.52N")
    parser.add_argument("--lon", required=True, help="Center longitude, e.g. 46.00 or 46.00E")
    parser.add_argument("--radius_km", type=float, required=True, help="Scan radius in km")
    parser.add_argument("--tile_km", type=float, required=True, help="Final logical tile size and center spacing in km")

    parser.add_argument("--overscan_km", type=float, default=None, help="Raw fetch width in km before crop for legacy overscan mode")
    parser.add_argument("--size", type=int, default=640, help="Google Static Maps size parameter")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2], help="Google Static Maps scale")
    parser.add_argument("--maptype", default="satellite", choices=["roadmap", "satellite", "hybrid", "terrain"])
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--zoom_strategy", default="floor", choices=["floor", "round", "ceil"])

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region_name", default="region")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_raw", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # Legacy mode.
    parser.add_argument(
        "--mosaic2x2",
        action="store_true",
        help="Legacy behavior: fetch four Static Maps images and save full 2x2 mosaic without crop.",
    )
    parser.add_argument(
        "--mosaic_zoom",
        type=int,
        default=16,
        help="Zoom level for legacy --mosaic2x2 mode.",
    )

    # New recommended mode.
    parser.add_argument(
        "--auto_mosaic",
        action="store_true",
        help=(
            "Recommended: dynamically choose zoom and mosaic grid by latitude, "
            "center-crop to tile_km, then resize to output_px."
        ),
    )
    parser.add_argument(
        "--output_px",
        type=int,
        default=2560,
        help="Final saved image size in pixels for --auto_mosaic.",
    )
    parser.add_argument(
        "--force_zoom",
        type=int,
        default=None,
        help="Optional fixed zoom override for --auto_mosaic. Usually leave unset.",
    )
    parser.add_argument(
        "--min_zoom",
        type=int,
        default=0,
        help="Minimum zoom allowed for --auto_mosaic.",
    )
    parser.add_argument(
        "--max_zoom",
        type=int,
        default=21,
        help="Maximum zoom allowed for --auto_mosaic.",
    )
    parser.add_argument(
        "--max_mosaic_grid",
        type=int,
        default=8,
        help="Maximum NxN grid allowed for --auto_mosaic.",
    )
    parser.add_argument(
        "--mpp_tolerance",
        type=float,
        default=0.0,
        help=(
            "Allowed fractional tolerance when choosing auto zoom. "
            "0.0 means source mpp must be <= target mpp. "
            "0.005 allows 0.5%% coarser native mpp."
        ),
    )
    parser.add_argument(
        "--mosaic_overlap_m",
        type=float,
        default=0.0,
        help=(
            "Optional overlap between adjacent mosaic subimages, in meters. "
            "This is subimage overlap, not final tile overlap."
        ),
    )

    args = parser.parse_args()

    if args.auto_mosaic and args.mosaic2x2:
        parser.error("Use either --auto_mosaic or --mosaic2x2, not both.")

    return args


def main() -> int:
    args = parse_args()

    lat0 = parse_coord(args.lat)
    lon0 = parse_coord(args.lon)
    radius_km = args.radius_km
    tile_km = args.tile_km
    overscan_km = args.overscan_km if args.overscan_km is not None else tile_km * 1.2

    if radius_km <= 0:
        print("Error: radius_km must be positive.", file=sys.stderr)
        return 2

    if tile_km <= 0:
        print("Error: tile_km must be positive.", file=sys.stderr)
        return 2

    if not args.auto_mosaic and not args.mosaic2x2:
        if overscan_km <= tile_km:
            print("Error: overscan_km must be larger than tile_km in overscan mode.", file=sys.stderr)
            return 2

    load_dotenv()

    api_key = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not api_key:
        print("Error: GOOGLE_MAPS_STATIC_API_KEY not found in environment or .env", file=sys.stderr)
        return 2

    output_dir = args.output.resolve()
    tiles_dir = output_dir / "tiles"
    raw_dir = output_dir / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    centers = generate_tile_centers(
        lat0=lat0,
        lon0=lon0,
        radius_km=radius_km,
        tile_km=tile_km,
    )

    if args.auto_mosaic:
        mode_name = "auto_mosaic"
    elif args.mosaic2x2:
        mode_name = "legacy_mosaic2x2"
    else:
        mode_name = "overscan_crop"

    print("Configuration")
    print(f"  mode: {mode_name}")
    print(f"  center: ({lat0:.6f}, {lon0:.6f})")
    print(f"  radius_km: {radius_km}")
    print(f"  tile_km: {tile_km}")
    print(f"  size: {args.size}")
    print(f"  scale: {args.scale}")
    print(f"  maptype: {args.maptype}")
    print(f"  total centers: {len(centers)}")
    print(f"  output: {output_dir}")

    if args.auto_mosaic:
        target_mpp = tile_km * 1000.0 / args.output_px
        print(f"  output_px: {args.output_px}")
        print(f"  target_mpp: {target_mpp:.6f} m/px")
        print(f"  max_mosaic_grid: {args.max_mosaic_grid}")
        print(f"  force_zoom: {args.force_zoom}")
    elif args.mosaic2x2:
        print(f"  mosaic_zoom: {args.mosaic_zoom}")
        print(f"  mosaic_overlap_m: {args.mosaic_overlap_m}")
    else:
        print(f"  overscan_km: {overscan_km}")
        print(f"  zoom_strategy: {args.zoom_strategy}")

    session = requests.Session()
    manifest_rows: list[dict] = []

    try:
        for idx, c in enumerate(centers, start=1):
            status = "pending"
            message = ""

            img: Optional[Image.Image] = None
            final_img: Optional[Image.Image] = None

            url = ""
            zoom: int | str = ""
            mpp: float | str = ""
            native_mpp: float | str = ""
            target_mpp: float | str = ""
            actual_width_km: float | str = ""
            mosaic_width_km: float | str = ""
            crop_px: int | str = ""
            returned_px: int | str = ""
            single_px: int | str = ""
            mosaic_grid_n: int | str = ""
            output_px: int | str = ""

            # ------------------------------------------------------------
            # Build output paths BEFORE any network request.
            # This is the key fix: cache check must happen before fetching.
            # ------------------------------------------------------------
            region_tag = safe_filename(args.region_name)
            base_name = (
                f"{region_tag}_r{c.row:03d}_c{c.col:03d}_"
                f"{c.lat:.6f}_{c.lon:.6f}_tile{tile_km:.2f}km"
            )

            tile_path = tiles_dir / f"{base_name}.png"
            raw_path = raw_dir / f"{base_name}_raw.png"

            cached_tile_path = tile_path

            if not args.overwrite:
                if tile_path.exists():
                    cached_tile_path = tile_path
                else:
                    # Fallback: if row/col changed, still match by coordinate + tile size.
                    coord_pattern = f"*_{c.lat:.6f}_{c.lon:.6f}_tile{tile_km:.2f}km.png"
                    matches = sorted(tiles_dir.glob(coord_pattern))
                    if matches:
                        cached_tile_path = matches[0]

            if cached_tile_path.exists() and not args.overwrite:
                status = "cached"
                message = "tile already exists"
                tile_path = cached_tile_path

                print(
                    f"[{idx}/{len(centers)}] cached "
                    f"r{c.row:03d} c{c.col:03d} {c.lat:.6f}, {c.lon:.6f} "
                    f"-> {tile_path.name}"
                )

                manifest_rows.append(
                    {
                        "row": c.row,
                        "col": c.col,
                        "x_m": f"{c.x_m:.3f}",
                        "y_m": f"{c.y_m:.3f}",
                        "center_lat": f"{c.lat:.8f}",
                        "center_lon": f"{c.lon:.8f}",
                        "distance_from_origin_m": f"{c.distance_m:.3f}",

                        "mode": mode_name,
                        "tile_km": f"{tile_km:.6f}",
                        "overscan_km_requested": f"{overscan_km:.6f}" if overscan_km else "",

                        "zoom": "",
                        "mosaic_grid_n": "",
                        "mosaic_overlap_m": f"{args.mosaic_overlap_m:.3f}",

                        "target_mpp": "",
                        "native_mpp": "",
                        "returned_m_per_px": "",

                        "actual_returned_width_km": "",
                        "mosaic_width_km": "",

                        "crop_px": "",
                        "tile_image_px": "",
                        "raw_image_px": "",
                        "single_image_px": "",
                        "output_px": "",

                        "tile_path": str(tile_path),
                        "raw_path": "" if args.skip_raw else str(raw_path),
                        "url": "",
                        "status": status,
                        "message": message,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )

                continue

            try:
                if args.auto_mosaic:
                    plan = build_auto_mosaic_plan(
                        session=session,
                        api_key=api_key,
                        center_lat=c.lat,
                        center_lon=c.lon,
                        target_km=tile_km,
                        output_px=args.output_px,
                        size_px=args.size,
                        scale=args.scale,
                        maptype=args.maptype,
                        timestamp=args.timestamp,
                        timeout=args.timeout,
                        overlap_m=args.mosaic_overlap_m,
                        min_zoom=args.min_zoom,
                        max_zoom=args.max_zoom,
                        force_zoom=args.force_zoom,
                        max_grid=args.max_mosaic_grid,
                        mpp_tolerance=args.mpp_tolerance,
                    )

                    zoom = plan.zoom
                    mpp = plan.native_mpp
                    native_mpp = plan.native_mpp
                    target_mpp = plan.target_mpp
                    crop_px = plan.crop_px
                    single_px = plan.single_px
                    mosaic_grid_n = plan.grid_n
                    mosaic_width_km = plan.mosaic_width_km
                    output_px = args.output_px

                    img, mpp, actual_mosaic_width_km, single_px_actual, mosaic_px, url = fetch_mosaic_grid_tile(
                        session=session,
                        api_key=api_key,
                        center_lat=c.lat,
                        center_lon=c.lon,
                        zoom=plan.zoom,
                        grid_n=plan.grid_n,
                        size_px=args.size,
                        scale=args.scale,
                        maptype=args.maptype,
                        timestamp=args.timestamp,
                        timeout=args.timeout,
                        overlap_m=args.mosaic_overlap_m,
                        expected_single_px=plan.single_px,
                    )

                    single_px = single_px_actual
                    returned_px = mosaic_px
                    mosaic_width_km = actual_mosaic_width_km
                    actual_width_km = actual_mosaic_width_km

                    cropped = crop_center_square(img, int(crop_px))
                    final_img = resize_square(cropped, args.output_px)

                elif args.mosaic2x2:
                    zoom = args.mosaic_zoom

                    img, mpp, actual_width_km, single_px, mosaic_px, url = fetch_mosaic_grid_tile(
                        session=session,
                        api_key=api_key,
                        center_lat=c.lat,
                        center_lon=c.lon,
                        zoom=zoom,
                        grid_n=2,
                        size_px=args.size,
                        scale=args.scale,
                        maptype=args.maptype,
                        timestamp=args.timestamp,
                        timeout=args.timeout,
                        overlap_m=args.mosaic_overlap_m,
                    )

                    # Legacy behavior: no crop.
                    crop_px = mosaic_px
                    returned_px = mosaic_px
                    mosaic_grid_n = 2
                    mosaic_width_km = actual_width_km
                    output_px = mosaic_px
                    final_img = img

                else:
                    zoom, mpp, actual_width_km, crop_px, img, url = choose_workable_zoom_by_fetch(
                        session=session,
                        api_key=api_key,
                        lat_deg=c.lat,
                        lon_deg=c.lon,
                        tile_km=tile_km,
                        overscan_km=overscan_km,
                        size_px=args.size,
                        scale=args.scale,
                        maptype=args.maptype,
                        timestamp=args.timestamp,
                        strategy=args.zoom_strategy,
                        timeout=args.timeout,
                    )

                    returned_px = img.size[0]
                    single_px = img.size[0]
                    output_px = crop_px
                    final_img = crop_center_square(img, int(crop_px))

            except Exception as exc:
                status = "error"
                message = f"fetch/planning error: {exc}"

            if status == "error":
                print(f"[{idx}/{len(centers)}] ERROR r{c.row:03d} c{c.col:03d}: {message}")

            elif args.dry_run:
                status = "dry_run"
                message = "not downloaded"
                print(
                    f"[{idx}/{len(centers)}] dry_run "
                    f"r{c.row:03d} c{c.col:03d} {c.lat:.6f}, {c.lon:.6f} "
                    f"zoom={zoom} grid={mosaic_grid_n} crop_px={crop_px}"
                )

            else:
                if final_img is None:
                    status = "error"
                    message = "internal error: final_img is None"
                    print(f"[{idx}/{len(centers)}] ERROR r{c.row:03d} c{c.col:03d}: {message}")

                else:
                    print(
                        f"[{idx}/{len(centers)}] saving "
                        f"r{c.row:03d} c{c.col:03d} "
                        f"{c.lat:.6f}, {c.lon:.6f} "
                        f"mode={mode_name} zoom={zoom} grid={mosaic_grid_n} "
                        f"mpp={float(mpp):.6f} "
                        f"crop_px={crop_px} "
                        f"final={final_img.size[0]}x{final_img.size[1]}"
                    )

                    try:
                        final_img.save(tile_path, format="PNG")

                        if not args.skip_raw and img is not None:
                            img.save(raw_path, format="PNG")

                        status = "downloaded"
                        message = "ok"
                        print(f"  -> saved {tile_path}")

                    except Exception as exc:
                        status = "error"
                        message = f"image saving error: {exc}"
                        print(f"  -> ERROR {message}")

                    if args.sleep > 0:
                        time.sleep(args.sleep)

            manifest_rows.append(
                {
                    "row": c.row,
                    "col": c.col,
                    "x_m": f"{c.x_m:.3f}",
                    "y_m": f"{c.y_m:.3f}",
                    "center_lat": f"{c.lat:.8f}",
                    "center_lon": f"{c.lon:.8f}",
                    "distance_from_origin_m": f"{c.distance_m:.3f}",

                    "mode": mode_name,
                    "tile_km": f"{tile_km:.6f}",
                    "overscan_km_requested": f"{overscan_km:.6f}" if overscan_km else "",

                    "zoom": zoom,
                    "mosaic_grid_n": mosaic_grid_n,
                    "mosaic_overlap_m": f"{args.mosaic_overlap_m:.3f}",

                    "target_mpp": f"{float(target_mpp):.9f}" if target_mpp != "" else "",
                    "native_mpp": f"{float(native_mpp):.9f}" if native_mpp != "" else "",
                    "returned_m_per_px": f"{float(mpp):.9f}" if mpp != "" else "",

                    "actual_returned_width_km": (
                        f"{float(actual_width_km):.9f}" if actual_width_km != "" else ""
                    ),
                    "mosaic_width_km": (
                        f"{float(mosaic_width_km):.9f}" if mosaic_width_km != "" else ""
                    ),

                    "crop_px": crop_px,
                    "tile_image_px": (
                        final_img.size[0]
                        if status in ("downloaded", "cached") and final_img is not None
                        else ""
                    ),
                    "raw_image_px": returned_px,
                    "single_image_px": single_px,
                    "output_px": output_px,

                    "tile_path": str(tile_path),
                    "raw_path": "" if args.skip_raw else str(raw_path),
                    "url": url,
                    "status": status,
                    "message": message,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )

    finally:
        session.close()

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    preview_path = output_dir / "preview.html"
    write_preview_html(
        preview_path,
        manifest_rows,
        title=(
            f"{args.region_name} | mode={mode_name} | "
            f"radius={radius_km}km | tile={tile_km}km"
        ),
    )

    statuses: dict[str, int] = {}
    for r in manifest_rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    print("\nDone")
    print(f"  status summary: {statuses}")
    print(f"  manifest: {manifest_path}")
    print(f"  preview:  {preview_path}")

    return 1 if statuses.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())