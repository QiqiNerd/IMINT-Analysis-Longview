#!/usr/bin/env python3
"""
Script to fetch Google Maps images for clusters and plot detection markers.

Reads cluster_analysis_results.json, fetches Google Maps images covering each cluster,
and draws markers at each detection coordinate.
"""

import os
import sys
import json
import argparse
import requests
import math
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from google_maps_collection import (
    build_gmaps_static_url,
    calculate_zoom_level,
)


def latlon_to_pixel(lat: float, lon: float, center_lat: float, center_lon: float,
                    zoom: int, image_size: int, scale: int = 1) -> Tuple[int, int]:
    """
    Convert lat/lon coordinates to pixel coordinates on a Google Maps static image.

    Uses the exact Web Mercator projection (EPSG:3857) that Google Maps uses.

    Args:
        lat: Latitude of point to convert
        lon: Longitude of point to convert
        center_lat: Latitude of image center
        center_lon: Longitude of image center
        zoom: Zoom level of the image
        image_size: Size of the image in pixels (the ACTUAL size, e.g. 4096 if scale=2)
        scale: Scale factor used when fetching the image (1 or 2)

    Returns:
        Tuple of (x, y) pixel coordinates
    """
    # Google Maps tile size and world size at this zoom
    # Multiply by scale because scale=2 doubles the pixels for the same geographic area
    TILE_SIZE = 256
    world_size = TILE_SIZE * (2 ** zoom) * scale

    # Convert lat/lon to Web Mercator pixel coordinates
    # This is the exact formula Google Maps uses
    def lat_to_y(lat_deg):
        lat_rad = math.radians(lat_deg)
        # Clamp latitude to avoid infinity at poles
        lat_rad = max(min(lat_rad, math.radians(85.051129)), math.radians(-85.051129))
        y = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2
        return y * world_size

    def lon_to_x(lon_deg):
        return (lon_deg + 180) / 360 * world_size

    # World pixel coordinates
    x_world = lon_to_x(lon)
    y_world = lat_to_y(lat)

    center_x_world = lon_to_x(center_lon)
    center_y_world = lat_to_y(center_lat)

    # Pixel coordinates relative to image center
    x_pixel = x_world - center_x_world + image_size / 2
    y_pixel = y_world - center_y_world + image_size / 2

    return (int(round(x_pixel)), int(round(y_pixel)))


def pixel_to_latlon(x: int, y: int, center_lat: float, center_lon: float,
                    zoom: int, image_size: int, scale: int = 1) -> Tuple[float, float]:
    """
    Convert pixel coordinates back to lat/lon coordinates on a Google Maps static image.

    Inverse of latlon_to_pixel. Uses exact Web Mercator projection (EPSG:3857).

    Args:
        x: X pixel coordinate
        y: Y pixel coordinate
        center_lat: Latitude of image center
        center_lon: Longitude of image center
        zoom: Zoom level of the image
        image_size: Size of the image in pixels (the ACTUAL size)
        scale: Scale factor used when fetching the image (1 or 2)

    Returns:
        Tuple of (lat, lon)
    """
    TILE_SIZE = 256
    world_size = TILE_SIZE * (2 ** zoom) * scale

    # Convert center to world pixel coordinates
    def lat_to_y(lat_deg):
        lat_rad = math.radians(lat_deg)
        lat_rad = max(min(lat_rad, math.radians(85.051129)), math.radians(-85.051129))
        y = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2
        return y * world_size

    def lon_to_x(lon_deg):
        return (lon_deg + 180) / 360 * world_size

    center_x_world = lon_to_x(center_lon)
    center_y_world = lat_to_y(center_lat)

    # Convert pixel coordinates to world coordinates
    x_world = x - image_size / 2 + center_x_world
    y_world = y - image_size / 2 + center_y_world

    # Convert world coordinates back to lat/lon
    def y_to_lat(y_world):
        y_norm = y_world / world_size
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_norm)))
        return math.degrees(lat_rad)

    def x_to_lon(x_world):
        return x_world / world_size * 360 - 180

    lon = x_to_lon(x_world)
    lat = y_to_lat(y_world)

    return (lat, lon)


def calculate_bounding_box(clusters: List[Dict]) -> Tuple[float, float, float, float, float, float, float]:
    """
    Calculate bounding box that encompasses all clusters.

    Args:
        clusters: List of cluster data dicts

    Returns:
        Tuple of (center_lat, center_lon, ground_distance_km, min_lat, max_lat, min_lon, max_lon)
    """
    if not clusters:
        return None

    all_lats = []
    all_lons = []
    max_radius = 0.0

    for cluster in clusters:
        center_lat = cluster['center_lat']
        center_lon = cluster['center_lon']
        radius_km = cluster['radius_km']

        all_lats.append(center_lat)
        all_lons.append(center_lon)
        max_radius = max(max_radius, radius_km)

        # Also include all detection points
        for detection in cluster.get('detections', []):
            all_lats.append(detection['lat'])
            all_lons.append(detection['lon'])

    min_lat = min(all_lats)
    max_lat = max(all_lats)
    min_lon = min(all_lons)
    max_lon = max(all_lons)

    # Calculate center of bounding box
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Calculate distance from center to corners to ensure we cover everything
    def haversine_km(lat1, lon1, lat2, lon2):
        """Great-circle distance in km."""
        R = 6371.0088
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # Calculate distance to each corner
    corners = [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, min_lon),
        (max_lat, max_lon)
    ]

    max_distance = 0.0
    for corner_lat, corner_lon in corners:
        dist = haversine_km(center_lat, center_lon, corner_lat, corner_lon)
        max_distance = max(max_distance, dist)

    # Add buffer (40% + max cluster radius + extra for ellipse padding)
    # The extra buffer ensures ellipses drawn around clusters aren't cut off at edges
    # Add 5% of max_distance as additional buffer for ellipse visibility
    ellipse_buffer_km = max(max_distance * 0.05, 1.0)  # At least 1km, or 5% of distance
    ground_distance_km = max_distance * 1.4 + max_radius + ellipse_buffer_km

    return (center_lat, center_lon, ground_distance_km, min_lat, max_lat, min_lon, max_lon)


def draw_markers_on_image(image: Image.Image, detections: List[Dict],
                          center_lat: float, center_lon: float,
                          zoom: int, image_size: int = None,
                          marker_color: str = "red",
                          marker_size: int = 8,
                          scale: int = 1) -> Image.Image:
    """
    Draw markers on an image at detection coordinates.

    Args:
        image: PIL Image object
        detections: List of detection dicts with 'lat' and 'lon' keys
        center_lat: Latitude of image center
        center_lon: Longitude of image center
        zoom: Zoom level of the image
        image_size: Size of image in pixels (if None, uses image.size[0])
        marker_color: Color of markers (default: "red")
        marker_size: Size of markers in pixels (default: 8)
        scale: Scale factor used when fetching the image (1 or 2)

    Returns:
        Image with markers drawn
    """
    draw = ImageDraw.Draw(image)
    if image_size is None:
        image_size = image.size[0]  # Assuming square image

    for detection in detections:
        lat = detection['lat']
        lon = detection['lon']

        # Convert to pixel coordinates
        x, y = latlon_to_pixel(lat, lon, center_lat, center_lon, zoom, image_size, scale=scale)

        # Draw marker (circle)
        # Draw a filled circle with a border
        bbox = [
            x - marker_size, y - marker_size,
            x + marker_size, y + marker_size
        ]
        # Convert color name to RGB tuple if needed
        if isinstance(marker_color, str):
            # Convert common color names to RGB
            color_map = {
                "red": (255, 0, 0),
                "blue": (0, 0, 255),
                "green": (0, 255, 0),
                "yellow": (255, 255, 0),
                "orange": (255, 165, 0),
                "purple": (128, 0, 128),
                "cyan": (0, 255, 255),
                "magenta": (255, 0, 255),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "gray": (128, 128, 128),
                "grey": (128, 128, 128),
            }
            fill_color = color_map.get(marker_color.lower(), (255, 0, 0))  # Default to red
        else:
            fill_color = marker_color
        draw.ellipse(bbox, fill=fill_color, outline=(255, 255, 255), width=2)

    return image


def process_cluster(cluster_data: Dict, api_key: str, output_dir: str,
                   image_size: int = 2048, maptype: str = "satellite",
                   timestamp: str = None, marker_color: str = "red",
                   marker_size: int = 8) -> str:
    """
    Process a single cluster: fetch image and draw markers.

    Args:
        cluster_data: Cluster data dict from JSON
        api_key: Google Maps API key
        output_dir: Output directory for images
        image_size: Size of image in pixels (default: 2048)
        maptype: Map type (default: "satellite")
        timestamp: Optional timestamp for historical imagery
        marker_color: Color of markers (default: "red")
        marker_size: Size of markers in pixels (default: 8)

    Returns:
        Path to saved image, or None if failed
    """
    cluster_id = cluster_data['cluster_id']
    center_lat = cluster_data['center_lat']
    center_lon = cluster_data['center_lon']
    radius_km = cluster_data['radius_km']
    detections = cluster_data['detections']

    # Calculate ground distance to cover entire cluster (radius * 2 + buffer)
    ground_distance_km = radius_km * 2.2  # Add 10% buffer on each side

    # Calculate zoom level
    zoom = calculate_zoom_level(ground_distance_km, image_size)

    print(f"Processing cluster {cluster_id}:")
    print(f"  Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  Radius: {radius_km:.2f} km")
    print(f"  Ground distance: {ground_distance_km:.2f} km")
    print(f"  Zoom level: {zoom}")

    # Warn if zoom level is very low (may result in poor detail for satellite imagery)
    if zoom < 6:
        print(f"  Warning: Zoom level {zoom} is very low. Image may lack detail for large area ({ground_distance_km:.2f} km).")

    print(f"  Detections: {len(detections)}")

    # Build Google Maps URL
    image_url = build_gmaps_static_url(
        center_lat,
        center_lon,
        api_key,
        ground_distance_km=ground_distance_km,
        image_size=image_size,
        maptype=maptype,
        timestamp=timestamp,
        scale=2,
    )

    # Fetch image
    print(f"  Fetching image...")
    response = requests.get(image_url)

    if response.status_code != 200:
        print(f"  Error: Failed to fetch image (status {response.status_code})")
        return None

    if b"Sorry, we have no imagery here" in response.content:
        print(f"  Error: No imagery available at this location")
        return None

    # Load image
    from io import BytesIO
    image = Image.open(BytesIO(response.content))

    # Convert to RGB mode if needed (to avoid palette mode color limitations)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get actual image size (may be 2x if scale=2 was used)
    actual_image_size = image.size[0]  # Assuming square image

    # Determine marker color based on compound_present
    compound_present = cluster_data.get('compound_present', 'no')
    if compound_present.lower() == 'yes':
        marker_color = 'green'
    else:
        marker_color = 'gray'

    # Draw markers
    print(f"  Drawing {len(detections)} markers in {marker_color} (compound_present: {compound_present})...")
    image = draw_markers_on_image(
        image, detections, center_lat, center_lon, zoom, actual_image_size,
        marker_color=marker_color, marker_size=marker_size, scale=2
    )

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    filename = f"cluster_{cluster_id}_with_markers_{center_lat:.6f}_{center_lon:.6f}_{ground_distance_km:.2f}km.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    print(f"  Saved: {filepath}")

    return filepath


def create_combined_image(clusters: List[Dict], api_key: str, output_dir: str,
                         image_size: int = 2048, maptype: str = "satellite",
                         timestamp: str = None, marker_size: int = 8,
                         zoom_out: float = 1.0, single_color: str = None) -> str:
    """
    Create one large image showing all clusters with different colors.

    Args:
        clusters: List of cluster data dicts
        api_key: Google Maps API key
        output_dir: Output directory for images
        image_size: Size of image in pixels (max 2048 for premium, 640 for standard) (default: 2048)
        maptype: Map type (default: "satellite")
        timestamp: Optional timestamp for historical imagery
        marker_size: Size of markers in pixels (default: 8)
        zoom_out: Multiplier for ground distance (1.0 = fit clusters, 2.0 = double area) (default: 1.0)
        single_color: If set, use this color for all markers instead of green/gray (default: None)

    Returns:
        Path to saved image, or None if failed
    """
    if not clusters:
        print("No clusters to combine")
        return None

    # Google Maps Static API limits:
    # - Standard: 640x640 pixels max
    # - Premium: 2048x2048 pixels max
    # Cap at 2048 and warn if larger requested
    MAX_IMAGE_SIZE = 2048
    if image_size > MAX_IMAGE_SIZE:
        print(f"Warning: Requested image size {image_size} exceeds Google Maps API limit of {MAX_IMAGE_SIZE} pixels.")
        print(f"  Capping to {MAX_IMAGE_SIZE} pixels. For larger images, consider using tiling.")
        image_size = MAX_IMAGE_SIZE

    print(f"\n=== Creating combined image for {len(clusters)} clusters ===")

    # Step 1: Calculate initial bounding box from detections
    center_lat, center_lon, ground_distance_km, min_lat, max_lat, min_lon, max_lon = calculate_bounding_box(clusters)

    # Step 2: Calculate initial zoom level
    zoom = calculate_zoom_level(ground_distance_km, image_size)

    # Step 3: Convert all detections to pixel coordinates
    # Use a temporary center for pixel conversion (we'll recalculate center later)
    temp_center_lat = center_lat
    temp_center_lon = center_lon

    # Collect all detection pixel coordinates by cluster
    cluster_pixel_bounds = []
    ellipse_padding = 10  # Same as used when drawing ellipses

    for cluster in clusters:
        detections = cluster.get('detections', [])
        if len(detections) < 2:
            continue

        pixel_coords = []
        for detection in detections:
            lat = detection['lat']
            lon = detection['lon']
            x, y = latlon_to_pixel(lat, lon, temp_center_lat, temp_center_lon, zoom, image_size)
            pixel_coords.append((x, y))

        # Find bounding box of this cluster's detections
        x_coords = [x for x, y in pixel_coords]
        y_coords = [y for x, y in pixel_coords]

        # Add ellipse padding
        min_x = min(x_coords) - ellipse_padding
        max_x = max(x_coords) + ellipse_padding
        min_y = min(y_coords) - ellipse_padding
        max_y = max(y_coords) + ellipse_padding

        cluster_pixel_bounds.append((min_x, min_y, max_x, max_y))

    # Step 4: Find overall bounding box of all ellipses in pixel space
    ellipse_min_lat = min_lat
    ellipse_max_lat = max_lat
    ellipse_min_lon = min_lon
    ellipse_max_lon = max_lon

    if cluster_pixel_bounds:
        all_min_x = min(bounds[0] for bounds in cluster_pixel_bounds)
        all_min_y = min(bounds[1] for bounds in cluster_pixel_bounds)
        all_max_x = max(bounds[2] for bounds in cluster_pixel_bounds)
        all_max_y = max(bounds[3] for bounds in cluster_pixel_bounds)

        # Step 5: Convert ellipse corners back to lat/lon
        top_left_lat, top_left_lon = pixel_to_latlon(all_min_x, all_min_y, temp_center_lat, temp_center_lon, zoom, image_size)
        top_right_lat, top_right_lon = pixel_to_latlon(all_max_x, all_min_y, temp_center_lat, temp_center_lon, zoom, image_size)
        bottom_left_lat, bottom_left_lon = pixel_to_latlon(all_min_x, all_max_y, temp_center_lat, temp_center_lon, zoom, image_size)
        bottom_right_lat, bottom_right_lon = pixel_to_latlon(all_max_x, all_max_y, temp_center_lat, temp_center_lon, zoom, image_size)

        # Step 6: Recalculate center and ground distance from ellipse corners
        ellipse_min_lat = min(top_left_lat, top_right_lat, bottom_left_lat, bottom_right_lat)
        ellipse_max_lat = max(top_left_lat, top_right_lat, bottom_left_lat, bottom_right_lat)
        ellipse_min_lon = min(top_left_lon, top_right_lon, bottom_left_lon, bottom_right_lon)
        ellipse_max_lon = max(top_left_lon, top_right_lon, bottom_left_lon, bottom_right_lon)

        center_lat = (ellipse_min_lat + ellipse_max_lat) / 2
        center_lon = (ellipse_min_lon + ellipse_max_lon) / 2

        # Calculate distance from center to corners
        def haversine_km(lat1, lon1, lat2, lon2):
            """Great-circle distance in km."""
            R = 6371.0088
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c

        corners = [
            (ellipse_min_lat, ellipse_min_lon),
            (ellipse_min_lat, ellipse_max_lon),
            (ellipse_max_lat, ellipse_min_lon),
            (ellipse_max_lat, ellipse_max_lon)
        ]

        max_distance = 0.0
        for corner_lat, corner_lon in corners:
            dist = haversine_km(center_lat, center_lon, corner_lat, corner_lon)
            max_distance = max(max_distance, dist)

        # Ground distance should cover from center to corners
        # Apply zoom_out factor to show more area
        ground_distance_km = max_distance * 2.0 * zoom_out  # Diameter = 2 * radius * zoom_out

        # Recalculate zoom with new ground distance
        zoom = calculate_zoom_level(ground_distance_km, image_size)

    print(f"Bounding box (based on ellipse edges):")
    print(f"  Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  Range: Lat [{ellipse_min_lat:.6f}, {ellipse_max_lat:.6f}], Lon [{ellipse_min_lon:.6f}, {ellipse_max_lon:.6f}]")
    print(f"  Ground distance: {ground_distance_km:.2f} km")
    print(f"  Zoom level: {zoom}")

    # Warn if zoom level is very low (may result in poor detail for satellite imagery)
    if zoom < 6:
        print(f"  Warning: Zoom level {zoom} is very low. Image may lack detail for large area ({ground_distance_km:.2f} km).")
        print(f"    Consider using tiling for better detail over large areas.")

    # Build Google Maps URL
    image_url = build_gmaps_static_url(
        center_lat,
        center_lon,
        api_key,
        ground_distance_km=ground_distance_km,
        image_size=image_size,
        maptype=maptype,
        timestamp=timestamp,
        scale=2,
    )

    # Fetch image
    print(f"  Fetching combined image...")
    response = requests.get(image_url)

    if response.status_code != 200:
        print(f"  Error: Failed to fetch image (status {response.status_code})")
        return None

    if b"Sorry, we have no imagery here" in response.content:
        print(f"  Error: No imagery available at this location")
        return None

    # Load image
    from io import BytesIO
    image = Image.open(BytesIO(response.content))
    actual_image_size = image.size[0]  # Assuming square image

    # Convert to RGB mode if needed (to avoid palette mode color limitations)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Define RGB colors for each cluster (one color per cluster)
    # Using RGB tuples for better compatibility
    colors = [
        (255, 0, 0),      # red
        (0, 0, 255),      # blue
        (0, 255, 0),      # green
        (255, 255, 0),    # yellow
        (255, 165, 0),    # orange
        (128, 0, 128),    # purple
        (0, 255, 255),    # cyan
        (255, 0, 255),   # magenta
        (50, 205, 50),    # lime green
        (255, 192, 203),  # pink
        (255, 215, 0),    # gold
        (255, 127, 80),   # coral
        (0, 255, 255),    # aqua
        (0, 0, 128),      # navy
        (128, 0, 0),      # maroon
        (128, 128, 0),   # olive
    ]

    # Draw markers for each cluster in different colors (one color per cluster)
    draw = ImageDraw.Draw(image)
    print(f"  Drawing markers for {len(clusters)} clusters...")

    # Store pixel coordinates for each cluster to draw ellipses later
    cluster_pixel_coords = []

    # Color map for single_color option
    color_map = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }

    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = cluster['cluster_id']
        detections = cluster.get('detections', [])

        # Determine marker color
        if single_color:
            marker_color = color_map.get(single_color.lower(), (0, 0, 255))  # default to blue
            color_name = single_color
        else:
            # Color based on compound_present
            compound_present = cluster.get('compound_present', 'no')
            if compound_present.lower() == 'yes':
                marker_color = (0, 255, 0)  # green
                color_name = "green"
            else:
                marker_color = (128, 128, 128)  # gray
                color_name = "gray"

        print(f"    Cluster {cluster_id}: {len(detections)} detections in {color_name}")

        # Store pixel coordinates for this cluster
        pixel_coords = []

        for detection in detections:
            lat = detection['lat']
            lon = detection['lon']

            # Convert to pixel coordinates (scale=2 because image is fetched with scale=2)
            x, y = latlon_to_pixel(lat, lon, center_lat, center_lon, zoom, actual_image_size, scale=2)
            pixel_coords.append((x, y))

            # Draw marker (circle) - all points in same cluster use same color
            bbox = [
                x - marker_size, y - marker_size,
                x + marker_size, y + marker_size
            ]
            draw.ellipse(bbox, fill=marker_color, outline=(255, 255, 255), width=2)

        cluster_pixel_coords.append((cluster_id, marker_color, pixel_coords))

    # Draw ellipses around each cluster
    print(f"  Drawing ellipses around clusters...")
    for cluster_id, marker_color, pixel_coords in cluster_pixel_coords:
        if len(pixel_coords) < 2:
            # Skip clusters with too few points to draw a meaningful ellipse
            continue

        # Find bounding box of all points in this cluster
        x_coords = [x for x, y in pixel_coords]
        y_coords = [y for x, y in pixel_coords]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Add padding around the cluster
        padding = 10
        ellipse_bbox = [
            min_x - padding,
            min_y - padding,
            max_x + padding,
            max_y + padding
        ]

        # Draw ellipse outline in cluster color
        # Use a thicker outline to make it visible
        draw.ellipse(ellipse_bbox, outline=marker_color, width=3)

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    filename = f"clusters_combined_{center_lat:.6f}_{center_lon:.6f}_{ground_distance_km:.2f}km.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    print(f"  Saved combined image: {filepath}")

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Google Maps images for clusters and plot detection markers"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to cluster_analysis_results.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for annotated images (default: same directory as JSON file)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=2048,
        help="Image size in pixels (default: 2048)"
    )
    parser.add_argument(
        "--maptype",
        default="satellite",
        choices=["roadmap", "satellite", "hybrid", "terrain"],
        help="Map type for Google Maps (default: satellite)"
    )
    parser.add_argument(
        "--date",
        help="Date for historical imagery (YYYY-MM or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--marker_color",
        default="red",
        help="Color of markers (default: red)"
    )
    parser.add_argument(
        "--marker_size",
        type=int,
        default=8,
        help="Size of markers in pixels (default: 8)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between API calls (default: 0.2)"
    )
    parser.add_argument(
        "--combined_image_size",
        type=int,
        default=2048,
        help="Image size in pixels for combined image (max 2048 for premium, 640 for standard) (default: 2048)"
    )
    parser.add_argument(
        "--no_combined",
        action="store_true",
        help="Skip creating combined image with all clusters"
    )
    parser.add_argument(
        "--combined_only",
        action="store_true",
        help="Only create combined image, skip individual cluster images"
    )
    parser.add_argument(
        "--zoom_out",
        type=float,
        default=1.0,
        help="Zoom out factor for combined image (e.g., 1.5 = 50%% more area, 2.0 = double the area) (default: 1.0)"
    )
    parser.add_argument(
        "--single_color",
        type=str,
        default=None,
        help="Use a single color for all markers (e.g., 'blue', 'red', 'green') instead of green/gray based on compound_present"
    )

    args = parser.parse_args()

    # Validate and cap image sizes to Google Maps API limits
    # Standard: 640x640 max, Premium: 2048x2048 max
    MAX_IMAGE_SIZE = 2048
    if args.image_size > MAX_IMAGE_SIZE:
        print(f"Warning: --image_size {args.image_size} exceeds Google Maps API limit of {MAX_IMAGE_SIZE} pixels.")
        print(f"  Capping to {MAX_IMAGE_SIZE} pixels.")
        args.image_size = MAX_IMAGE_SIZE

    if args.combined_image_size > MAX_IMAGE_SIZE:
        print(f"Warning: --combined_image_size {args.combined_image_size} exceeds Google Maps API limit of {MAX_IMAGE_SIZE} pixels.")
        print(f"  Capping to {MAX_IMAGE_SIZE} pixels.")
        args.combined_image_size = MAX_IMAGE_SIZE

    # Set output directory to same folder as JSON input
    json_path = Path(args.json).resolve()
    if args.output is None:
        args.output = str(json_path.parent)
    else:
        # Ensure output is an absolute path
        args.output = str(Path(args.output).resolve())

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not api_key:
        print("Error: GOOGLE_MAPS_STATIC_API_KEY not set.")
        raise SystemExit(1)

    # Load JSON file
    print(f"Loading cluster results from: {args.json}")
    with open(args.json, 'r') as f:
        data = json.load(f)

    clusters = data.get('results', [])
    print(f"Found {len(clusters)} clusters to process")
    print(f"Output directory: {args.output}\n")

    # Process each cluster (unless --combined_only)
    successful = 0
    if not args.combined_only:
        for cluster_data in clusters:
            try:
                result = process_cluster(
                    cluster_data,
                    api_key,
                    args.output,
                    image_size=args.image_size,
                    maptype=args.maptype,
                    timestamp=args.date,
                    marker_color=args.marker_color,
                    marker_size=args.marker_size
                )
                if result:
                    successful += 1

                # Sleep to avoid rate limits
                import time
                time.sleep(args.sleep)
            except Exception as e:
                print(f"  Error processing cluster {cluster_data.get('cluster_id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Finished processing {successful}/{len(clusters)} clusters")
        print(f"Images saved to: {args.output}")

    # Create combined image showing all clusters
    if (not args.no_combined or args.combined_only) and len(clusters) > 0:
        try:
            import time
            time.sleep(args.sleep)  # Sleep before combined image request
            create_combined_image(
                clusters,
                api_key,
                args.output,
                image_size=args.combined_image_size,
                maptype=args.maptype,
                timestamp=args.date,
                marker_size=args.marker_size,
                zoom_out=args.zoom_out,
                single_color=args.single_color
            )
        except Exception as e:
            print(f"  Error creating combined image: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
