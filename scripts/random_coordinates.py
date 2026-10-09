import csv
import random

CITY_BOUNDS = {
    # name: (min_lat, max_lat, min_lon, max_lon)
    "beijing": (39.4, 40.3, 115.9, 117.0),
    "tianjin": (38.6, 39.5, 116.6, 118.1),
    "shanghai": (30.9, 31.6, 120.8, 122.0),
    "shenzhen": (22.4, 22.9, 113.7, 114.3),
    "guangzhou": (22.8, 23.6, 112.9, 113.9),
    "chengdu": (30.3, 30.9, 103.6, 104.3),
    "chongqing": (29.2, 30.1, 106.2, 107.0),
    "wuhan": (30.3, 30.9, 114.0, 114.7),
    "xian": (34.1, 34.5, 108.7, 109.1),
    "hangzhou": (30.0, 30.6, 119.6, 120.6),
}

def _sample_in_bbox(min_lat, max_lat, min_lon, max_lon):
    lat = round(random.uniform(min_lat, max_lat), 6)
    lon = round(random.uniform(min_lon, max_lon), 6)
    return lat, lon

def generate_random_coordinates(n, output_path, region="NK", city=None):
    region = (region or "NK").upper()
    city = city.lower() if city else None

    # North Korea national bounding box (approx)
    NK_BBOX = (37.5, 43.0, 124.0, 130.7)

    header = [
        '', 'Y (North-South direction)', 'X (East-West direction)',
        'Z', 'Name', 'Y/N', 'Correction', 'IMDATE Visible'
    ]

    rows = []
    if region == "NK":
        min_lat, max_lat, min_lon, max_lon = NK_BBOX
        for _ in range(n):
            lat, lon = _sample_in_bbox(min_lat, max_lat, min_lon, max_lon)
            rows.append(['', lat, lon, 0, 'AAA', 'Y', '', ''])
    elif region == "CH":
        cities = [city] if (city and city in CITY_BOUNDS) else list(CITY_BOUNDS.keys())
        for i in range(n):
            name = cities[i % len(cities)]
            min_lat, max_lat, min_lon, max_lon = CITY_BOUNDS[name]
            lat, lon = _sample_in_bbox(min_lat, max_lat, min_lon, max_lon)
            # Use city name in the Name column
            rows.append(['', lat, lon, 0, name.upper(), 'Y', '', ''])
    else:
        raise ValueError("region must be 'NK' or 'CH'")

    # Write to CSV
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✅ Generated {n} coordinates and saved to '{output_path}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate random coordinates for NK or Chinese cities")
    parser.add_argument('--n', type=int, default=100, help='Number of coordinates to generate')
    parser.add_argument('--region', choices=['NK', 'CH'], default='NK', help='Region: NK (North Korea) or CH (China cities)')
    parser.add_argument('--city', help='For CH: pick a specific city (e.g., beijing, shanghai). If omitted, cycle across preset cities.')
    parser.add_argument('--out', help='Output CSV filename (default: coords_<REGION>.csv)')

    args = parser.parse_args()
    default_out = args.out or f"coords_{args.region}.csv"
    generate_random_coordinates(args.n, default_out, region=args.region, city=args.city)