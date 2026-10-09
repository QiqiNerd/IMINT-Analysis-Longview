#!/usr/bin/env python3
"""
Generate random negative coordinates around Russian cities from warhead storage locations.
Excludes actual warhead facility coordinates.
"""

import csv
import random
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Set

# Minimum distance from actual facilities (in degrees, roughly 0.01 deg ≈ 1.1 km)
MIN_DISTANCE_DEG = 0.01  # ~1.1 km buffer

def extract_base_name(name: str) -> str:
    """Extract base city/region name from facility name (e.g., 'Tver-1' -> 'Tver')."""
    # Remove numbers and dashes at the end
    parts = name.split('-')
    if len(parts) > 1 and parts[-1].isdigit():
        return '-'.join(parts[:-1])
    return name

def load_warhead_facilities(csv_path: str) -> Tuple[List[Tuple[float, float]], dict]:
    """Load warhead facilities and group by city/region."""
    df = pd.read_csv(csv_path)

    facilities = []
    city_groups = defaultdict(list)

    for _, row in df.iterrows():
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        name = row['Name']

        facilities.append((lat, lon))
        base_name = extract_base_name(name)
        city_groups[base_name].append((lat, lon, name))

    return facilities, city_groups

def calculate_city_bbox(coords: List[Tuple[float, float, str]]) -> Tuple[float, float, float, float]:
    """Calculate bounding box for a city's facilities with buffer."""
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    min_lat = min(lats) - 0.1  # ~11 km buffer
    max_lat = max(lats) + 0.1
    min_lon = min(lons) - 0.1
    max_lon = max(lons) + 0.1

    return (min_lat, max_lat, min_lon, max_lon)

def distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate approximate distance in degrees."""
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

def is_too_close(lat: float, lon: float, facilities: List[Tuple[float, float]],
                 min_distance: float) -> bool:
    """Check if coordinate is too close to any facility."""
    for fac_lat, fac_lon in facilities:
        if distance_deg(lat, lon, fac_lat, fac_lon) < min_distance:
            return True
    return False

def generate_negative_coordinates(n: int, warhead_csv_path: str, output_path: str,
                                min_distance_deg: float = MIN_DISTANCE_DEG,
                                max_attempts: int = 1000) -> None:
    """Generate n random negative coordinates around Russian cities."""

    # Load warhead facilities
    facilities, city_groups = load_warhead_facilities(warhead_csv_path)

    print(f"Loaded {len(facilities)} warhead facilities from {len(city_groups)} cities/regions")

    # Create bounding boxes for each city
    city_bboxes = {}
    for city, coords in city_groups.items():
        city_bboxes[city] = calculate_city_bbox(coords)
        print(f"  {city}: {len(coords)} facilities, bbox {city_bboxes[city]}")

    # Generate random coordinates
    coordinates = []
    cities = list(city_bboxes.keys())

    print(f"\nGenerating {n} negative coordinates...")

    for i in range(n):
        attempts = 0
        while attempts < max_attempts:
            # Pick a random city
            city = random.choice(cities)
            min_lat, max_lat, min_lon, max_lon = city_bboxes[city]

            # Generate random coordinate in city's bounding box
            lat = round(random.uniform(min_lat, max_lat), 6)
            lon = round(random.uniform(min_lon, max_lon), 6)

            # Check if it's far enough from facilities
            if not is_too_close(lat, lon, facilities, min_distance_deg):
                coordinates.append({
                    'name': f'Negative_{city}_{i+1}',
                    'lat': lat,
                    'lon': lon,
                    'city': city
                })
                break

            attempts += 1

        if attempts >= max_attempts:
            print(f"Warning: Could not generate coordinate {i+1} after {max_attempts} attempts")
            # Fall back to a coordinate further from facilities
            city = random.choice(cities)
            min_lat, max_lat, min_lon, max_lon = city_bboxes[city]
            # Use larger buffer
            lat = round(random.uniform(min_lat - 0.2, max_lat + 0.2), 6)
            lon = round(random.uniform(min_lon - 0.2, max_lon + 0.2), 6)
            coordinates.append({
                'name': f'Negative_{city}_{i+1}',
                'lat': lat,
                'lon': lon,
                'city': city
            })

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{n} coordinates...")

    # Write to CSV in Russian warhead storage format
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Latitude', 'Longitude', 'Altitude', 'Description'])

        for coord in coordinates:
            writer.writerow([
                coord['name'],
                coord['lat'],
                coord['lon'],
                0,
                f"Negative sample near {coord['city']} (not a warhead storage facility)"
            ])

    print(f"\n✅ Generated {len(coordinates)} negative coordinates and saved to '{output_path}'")
    print(f"   Format: Russian warhead storage CSV (compatible with google_maps_collection.py)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate random negative coordinates around Russian cities from warhead storage locations"
    )
    parser.add_argument('--n', type=int, default=103,
                       help='Number of negative coordinates to generate (default: 103)')
    parser.add_argument('--warhead-csv',
                       default='coordinates/russian_warhead_storage.csv',
                       help='Path to warhead storage CSV file')
    parser.add_argument('--output',
                       default='coordinates/russian_warhead_negatives.csv',
                       help='Output CSV filename')
    parser.add_argument('--min-distance', type=float, default=MIN_DISTANCE_DEG,
                       help=f'Minimum distance from facilities in degrees (default: {MIN_DISTANCE_DEG})')

    args = parser.parse_args()

    generate_negative_coordinates(
        n=args.n,
        warhead_csv_path=args.warhead_csv,
        output_path=args.output,
        min_distance_deg=args.min_distance
    )
