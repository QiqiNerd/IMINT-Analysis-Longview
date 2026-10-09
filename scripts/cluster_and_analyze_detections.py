#!/usr/bin/env python3
"""
Script to cluster detection coordinates from image filenames and analyze with GPT-5.

Takes a folder of detection images, extracts coordinates from filenames,
clusters them, finds central coordinates and radii, fetches Google Maps images,
and prompts GPT-5 to identify enclosed compounds.
"""

import os
import re
import sys
import json
import argparse
import time
import requests
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from google_maps_collection import (
    save_static_map_image,
    build_gmaps_static_url,
)


def extract_coordinates_from_filename(filename: str) -> Tuple[float, float, float, str]:
    """
    Extract coordinates and distance from image filename.

    Expected format: {name}_{lat}_{lon}_{distance}km.png
    Example: Belgorod-22-1_50.56329_35.74938_0.5km.png

    Returns:
        Tuple of (lat, lon, distance_km, name) or None if parsing fails
    """
    # Remove extension
    base = os.path.splitext(filename)[0]

    # Pattern: name_lat_lon_distancekm
    # Try to match the pattern with coordinates and distance
    pattern = r'^(.+?)_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(\d+\.?\d*)km$'
    match = re.match(pattern, base)

    if match:
        name = match.group(1)
        lat = float(match.group(2))
        lon = float(match.group(3))
        distance = float(match.group(4))
        return (lat, lon, distance, name)

    # Try alternative pattern without distance
    pattern_alt = r'^(.+?)_(-?\d+\.?\d*)_(-?\d+\.?\d*)$'
    match_alt = re.match(pattern_alt, base)

    if match_alt:
        name = match_alt.group(1)
        lat = float(match_alt.group(2))
        lon = float(match_alt.group(3))
        return (lat, lon, 0.5, name)  # Default distance

    return None


def extract_coordinates_from_folder(folder_path: str) -> List[Dict]:
    """
    Extract coordinates from all image files in a folder.

    Returns:
        List of dicts with keys: 'lat', 'lon', 'distance', 'name', 'filename'
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    coordinates = []
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

    for file_path in folder.iterdir():
        if file_path.suffix in image_extensions:
            result = extract_coordinates_from_filename(file_path.name)
            if result:
                lat, lon, distance, name = result
                coordinates.append({
                    'lat': lat,
                    'lon': lon,
                    'distance': distance,
                    'name': name,
                    'filename': file_path.name
                })
            else:
                print(f"Warning: Could not parse coordinates from {file_path.name}")

    print(f"Extracted {len(coordinates)} coordinates from {folder_path}")
    return coordinates


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers.
    """
    R = 6371.0088  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def cluster_coordinates(coordinates: List[Dict], max_radius_km: float = 5.0) -> List[List[Dict]]:
    """
    Cluster coordinates using Agglomerative Clustering with distance threshold.
    This ensures all points in a cluster are within max_radius_km of each other.

    Args:
        coordinates: List of coordinate dicts
        max_radius_km: Maximum radius in km for a cluster (all points must be within this distance)

    Returns:
        List of clusters, where each cluster is a list of coordinate dicts
    """
    if len(coordinates) == 0:
        return []

    n = len(coordinates)

    # Compute pairwise distance matrix using haversine distance
    print("Computing distance matrix...")
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance_km(
                coordinates[i]['lat'], coordinates[i]['lon'],
                coordinates[j]['lat'], coordinates[j]['lon']
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Perform Agglomerative Clustering with precomputed distance matrix
    # distance_threshold: maximum distance between points in the same cluster
    # linkage='complete': ensures all points in cluster are within threshold
    # n_clusters=None: let algorithm determine number of clusters based on threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=max_radius_km,
        metric='precomputed',
        linkage='complete'
    )

    labels = clustering.fit_predict(distance_matrix)

    # Group coordinates by cluster label
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(coordinates[idx])

    # Return all clusters (no noise points with Agglomerative Clustering)
    valid_clusters = [clusters[label] for label in clusters.keys()]

    print(f"Found {len(valid_clusters)} clusters (max radius: {max_radius_km} km)")
    return valid_clusters


def calculate_cluster_center_and_radius(cluster: List[Dict]) -> Tuple[float, float, float]:
    """
    Calculate the center coordinate and radius for a cluster.

    Args:
        cluster: List of coordinate dicts in the cluster

    Returns:
        Tuple of (center_lat, center_lon, radius_km)
    """
    if len(cluster) == 0:
        return None

    # Calculate center as mean of coordinates
    center_lat = np.mean([c['lat'] for c in cluster])
    center_lon = np.mean([c['lon'] for c in cluster])

    # Calculate radius as maximum distance from center to any point in cluster
    def haversine_km(lat1, lon1, lat2, lon2):
        """Great-circle distance in km."""
        R = 6371.0088
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    max_radius = 0.0
    for coord in cluster:
        dist = haversine_km(center_lat, center_lon, coord['lat'], coord['lon'])
        max_radius = max(max_radius, dist)

    # Add a small buffer (10% or minimum 0.5km)
    radius_km = max(max_radius * 1.1, max_radius + 0.5)

    return (center_lat, center_lon, radius_km)


def classify_compound_image(client: OpenAI, model: str, image_url: str) -> Tuple[str, str]:
    """
    Classify if an image shows an enclosed compound using GPT.

    Returns:
        Tuple of (answer: 'yes' or 'no', explanation)
    """
    prompt = "Do you see indications of an enclosed compound in this image?"

    # GPT-5 doesn't support temperature parameter, only default (1) is allowed
    # For other models, use temperature=0 for deterministic outputs
    request_params = {
        "model": model,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "compound_presence",
                "schema": {
                    "type": "object",
                    "properties": {
                        "compound_present": {"type": "string", "enum": ["yes", "no"]},
                        "explanation": {"type": "string"}
                    },
                    "required": ["compound_present", "explanation"],
                    "additionalProperties": False
                },
                "strict": True,
            },
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }

    # Only set temperature for models that support it (not GPT-5)
    if "gpt-5" not in model.lower():
        request_params["temperature"] = 0

    try:
        resp = client.chat.completions.create(**request_params)
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        answer = str(data.get("compound_present", "no")).strip().lower()
        explanation = str(data.get("explanation", "")).strip()
        label = "yes" if answer == "yes" else "no"
        return (label, explanation)
    except Exception as e:
        print(f"Classification error: {e}")
        return ("no", f"Classification error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster detection coordinates and analyze with GPT-5"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing detection images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for fetched images and results (default: input_folder/cluster_analysis)"
    )
    parser.add_argument(
        "--max_radius_km",
        type=float,
        default=5.0,
        help="Maximum radius in km for clusters (all points in a cluster must be within this distance) (default: 5.0)"
    )
    parser.add_argument(
        "--ground_distance_km",
        type=float,
        default=2.0,
        help="Ground distance in km for fetched images (default: 2.0)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size in pixels (default: 1024)"
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
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o, use 'gpt-5' if available)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between API calls (default: 0.2)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="After clustering, generate annotated satellite images with detection markers and a combined overview"
    )
    parser.add_argument(
        "--marker_color",
        default="red",
        help="Marker color for visualization (default: red)"
    )
    parser.add_argument(
        "--marker_size",
        type=int,
        default=8,
        help="Marker size in pixels for visualization (default: 8)"
    )

    args = parser.parse_args()

    # Set output directory to be under the input folder if not specified
    if args.output is None:
        input_folder = Path(args.folder)
        args.output = str(input_folder / "cluster_analysis")

    # Load environment variables
    load_dotenv()

    gmaps_key = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not gmaps_key:
        print("Error: GOOGLE_MAPS_STATIC_API_KEY not set.")
        raise SystemExit(1)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not set.")
        raise SystemExit(1)

    client = OpenAI(api_key=openai_key)

    # Extract coordinates from folder
    print(f"Extracting coordinates from: {args.folder}")
    coordinates = extract_coordinates_from_folder(args.folder)

    if len(coordinates) == 0:
        print("No coordinates found. Exiting.")
        return

    # Cluster coordinates
    print(f"\nClustering coordinates (max_radius={args.max_radius_km}km)...")
    clusters = cluster_coordinates(coordinates, max_radius_km=args.max_radius_km)

    if len(clusters) == 0:
        print("No clusters found. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process each cluster
    results = []
    for cluster_idx, cluster in enumerate(clusters, 1):
        print(f"\n--- Processing Cluster {cluster_idx}/{len(clusters)} ---")
        print(f"Cluster size: {len(cluster)} detections")

        # Calculate center and radius
        center_lat, center_lon, radius_km = calculate_cluster_center_and_radius(cluster)
        print(f"Center: ({center_lat:.6f}, {center_lon:.6f})")
        print(f"Radius: {radius_km:.2f} km")

        # Use cluster radius * 2 for image size to capture full cluster with context
        # But cap at max_radius_km (max cluster radius) and ensure minimum from args.ground_distance_km
        image_ground_distance = min(max(radius_km * 2, args.ground_distance_km), args.max_radius_km)

        # Build Google Maps URL
        image_url = build_gmaps_static_url(
            center_lat,
            center_lon,
            gmaps_key,
            ground_distance_km=image_ground_distance,
            image_size=args.image_size,
            maptype=args.maptype,
            timestamp=args.date,
            scale=2,
        )

        # Classify with GPT
        print("Classifying with GPT...")
        answer, explanation = classify_compound_image(client, args.model, image_url)
        print(f"Result: {answer}")
        print(f"Explanation: {explanation[:200]}..." if len(explanation) > 200 else f"Explanation: {explanation}")

        # Save image
        image_name = f"cluster_{cluster_idx}"
        image_path = os.path.join(
            args.output,
            f"{image_name}_{center_lat:.6f}_{center_lon:.6f}_{image_ground_distance:.2f}km.png"
        )

        # Download and save image
        response = requests.get(image_url)
        if response.status_code == 200:
            if b"Sorry, we have no imagery here" not in response.content:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Saved image: {image_path}")
            else:
                print("No imagery available at this location")
                image_path = None
        else:
            print(f"Failed to fetch image: {response.status_code}")
            image_path = None

        # Store result
        results.append({
            "cluster_id": cluster_idx,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km,
            "num_detections": len(cluster),
            "detections": cluster,
            "compound_present": answer,
            "explanation": explanation,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat()
        })

        # Sleep to avoid rate limits
        time.sleep(args.sleep)

    # Save results to JSON
    results_path = os.path.join(args.output, "cluster_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "folder": args.folder,
            "total_detections": len(coordinates),
            "num_clusters": len(clusters),
            "clustering_params": {
                "max_radius_km": args.max_radius_km
            },
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✅ Finished processing {len(clusters)} clusters")
    print(f"Results saved to: {results_path}")
    print(f"Images saved to: {args.output}")

    if args.visualize:
        from plot_cluster_detections import process_cluster, create_combined_image

        print(f"\n=== Generating annotated cluster images ===")
        vis_ok = 0
        for cluster_data in results:
            try:
                path = process_cluster(
                    cluster_data,
                    gmaps_key,
                    args.output,
                    image_size=args.image_size,
                    maptype=args.maptype,
                    timestamp=args.date,
                    marker_color=args.marker_color,
                    marker_size=args.marker_size,
                )
                if path:
                    vis_ok += 1
            except Exception as e:
                print(f"  Error visualizing cluster {cluster_data.get('cluster_id')}: {e}")
            time.sleep(args.sleep)

        if results:
            try:
                time.sleep(args.sleep)
                create_combined_image(
                    results,
                    gmaps_key,
                    args.output,
                    image_size=args.image_size,
                    maptype=args.maptype,
                    timestamp=args.date,
                    marker_size=args.marker_size,
                )
            except Exception as e:
                print(f"  Error creating combined image: {e}")

        print(f"\nVisualization complete. {vis_ok}/{len(results)} cluster images generated.")


if __name__ == "__main__":
    main()
