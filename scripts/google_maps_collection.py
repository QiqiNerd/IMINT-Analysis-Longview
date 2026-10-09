import pandas as pd
from dotenv import load_dotenv
import os
import requests

def dms_to_decimal(dms_string):
    """Converts DMS (Degrees, Minutes, Seconds) coordinate string to decimal degrees.

    Args:
        dms_string: String like "39° 5'27.29\"N" or "125°37'2.83\"E"

    Returns:
        float: Decimal degrees (negative for S/W directions)
    """
    import re

    # Handle case where coordinates might be combined (lat lon in one cell)
    if ' ' in dms_string and ('N' in dms_string or 'S' in dms_string) and ('E' in dms_string or 'W' in dms_string):
        # Split combined coordinates like "39° 5'27.29"N 125°37'2.83"E"
        parts = dms_string.split()
        lat_part = None
        lon_part = None

        for part in parts:
            if 'N' in part or 'S' in part:
                lat_part = part
            elif 'E' in part or 'W' in part:
                lon_part = part

        if lat_part and lon_part:
            return dms_to_decimal(lat_part), dms_to_decimal(lon_part)

    # Clean up the string
    dms_string = str(dms_string).strip()

    # Check if it's already a decimal number
    try:
        return float(dms_string)
    except ValueError:
        pass

    # Extract degrees, minutes, seconds, and direction
    pattern = r'(\d+)°?\s*(\d+)\'?\s*([0-9.]+)\"?\s*([NSEW])?'
    match = re.search(pattern, dms_string)

    if not match:
        # Try simpler pattern for just degrees
        pattern = r'(\d+\.?\d*)°?\s*([NSEW])?'
        match = re.search(pattern, dms_string)
        if match:
            degrees = float(match.group(1))
            direction = match.group(2)
            if direction in ['S', 'W']:
                degrees = -degrees
            return degrees
        else:
            raise ValueError(f"Could not parse coordinate: {dms_string}")

    degrees = float(match.group(1))
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)

    # Convert to decimal degrees
    decimal = degrees + minutes/60 + seconds/3600

    # Make negative for South/West
    if direction in ['S', 'W']:
        decimal = -decimal

    return decimal

def _sanitize_name(name: str) -> str:
    """Sanitize a name for safe filename usage (no path separators or odd chars)."""
    name = str(name)
    name = name.replace('/', '_').replace('\\', '_')
    return __import__('re').sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("._")

def parse_russian_warhead_csv(csv_file_path):
    """Parses a Russian warhead storage CSV file and returns a list of coordinate dicts with name, lat, lon.
    Expects columns: Name, Latitude, Longitude, Altitude, Description.
    Coordinates are already in decimal format."""

    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    coordinate_data = []
    for _, row in df.iterrows():
        # Use Name column, or create a generic name if empty
        name = row['Name'] if pd.notna(row['Name']) and row['Name'].strip() else f"Location_{len(coordinate_data)+1}"

        # Handle coordinate parsing (already in decimal format)
        try:
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])

            coordinate_data.append({
                'name': name.strip(),
                'lat': lat,
                'lon': lon
            })
        except Exception as e:
            print(f"Warning: Could not parse coordinates for {name}: {e}")
            print(f"  Lat: {row.get('Latitude', 'N/A')}")
            print(f"  Lon: {row.get('Longitude', 'N/A')}")
            continue

    return coordinate_data

def parse_csv_coordinates(csv_file_path):
    """Parses a CSV file and returns a list of coordinate dicts with name, lat, lon.
    Only includes rows where Y/N column is 'Y' and IMDATE Visible is empty."""

    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Filter rows where Y/N is 'Y' and IMDATE Visible is empty/NaN
    filtered_df = df[
        (df['Y/N'] == 'Y') &
        (df['IMDATE Visible'].isna() | (df['IMDATE Visible'] == ''))
    ]

    coordinate_data = []
    for _, row in filtered_df.iterrows():
        # Use Name column, or create a generic name if empty
        name = row['Name'] if pd.notna(row['Name']) and row['Name'].strip() else f"Location_{len(coordinate_data)+1}"

        # Handle coordinate conversion
        try:
            lat_raw = row['Y (North-South direction)']
            lon_raw = row['X (East-West direction)']

            # Check if coordinates are combined in one cell
            if pd.isna(lon_raw) or lon_raw == '' or str(lon_raw).strip() == '':
                # Try to extract both lat and lon from the lat column
                coords = dms_to_decimal(str(lat_raw))
                if isinstance(coords, tuple):
                    lat, lon = coords
                else:
                    raise ValueError("Could not extract both coordinates")
            else:
                # Convert each coordinate separately
                lat = dms_to_decimal(str(lat_raw))
                lon = dms_to_decimal(str(lon_raw))

            coordinate_data.append({
                'name': name.strip(),
                'lat': lat,
                'lon': lon
            })
        except Exception as e:
            print(f"Warning: Could not parse coordinates for {name}: {e}")
            print(f"  Lat: {lat_raw}")
            print(f"  Lon: {lon_raw}")
            continue

    return coordinate_data

def parse_csv_coordinates_auto(csv_file_path):
    """Automatically detects CSV format and parses accordingly.
    Returns a list of coordinate dicts with name, lat, lon."""

    # Read CSV file to detect format
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()

    # Check if it's Russian warhead storage format (has 'Latitude' and 'Longitude' columns)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        print("Detected Russian warhead storage CSV format")
        return parse_russian_warhead_csv(csv_file_path)
    # Check if it's the AAA format (has 'Y/N' and 'Y (North-South direction)' columns)
    elif 'Y/N' in df.columns and 'Y (North-South direction)' in df.columns:
        print("Detected AAA/NK coordinates CSV format")
        return parse_csv_coordinates(csv_file_path)
    else:
        raise ValueError(f"Unknown CSV format. Expected either:\n"
                        f"  - Russian warhead format: Name, Latitude, Longitude, ...\n"
                        f"  - AAA/NK format: Name, Y/N, Y (North-South direction), X (East-West direction), ...")

def calculate_zoom_level(ground_distance_km, image_size_px=640):
    """Calculate the appropriate zoom level based on ground distance in kilometers.

    Args:
        ground_distance_km: Desired ground distance in kilometers (width/height of the image)
        image_size_px: Size of the image in pixels (default: 640x640)

    Returns:
        int: Zoom level (0-21)
    """
    import math
    # Earth's circumference in kilometers
    EQUATOR_CIRCUMFERENCE = 40075.0

    # Calculate the ground resolution at the equator for each zoom level
    zoom_levels = {}
    for zoom in range(0, 22):
        # At zoom 0, the whole world is 256x256 pixels
        # Each zoom level doubles the resolution
        pixels = 256 * (2 ** zoom)
        # Ground resolution at the equator (km/pixel)
        resolution = EQUATOR_CIRCUMFERENCE / pixels
        # Ground distance covered by the image (km)
        ground_distance = resolution * image_size_px
        zoom_levels[zoom] = ground_distance

    # Find the zoom level that most closely matches the desired ground distance
    best_zoom = 0
    min_diff = float('inf')
    for zoom, distance in zoom_levels.items():
        diff = abs(distance - ground_distance_km)
        if diff < min_diff:
            min_diff = diff
            best_zoom = zoom

    return best_zoom

def build_gmaps_static_url(lat, lon, api_key, ground_distance_km=1.0, image_size=640, maptype='satellite', timestamp=None, scale=2):
    """Constructs a Google Maps Static API URL for a given location.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        api_key: Google Maps API key
        ground_distance_km: Desired ground distance in kilometers (width/height of the image)
        image_size: Size of the image in pixels (square image)
        maptype: Map type ('roadmap', 'satellite', 'hybrid', 'terrain')
        timestamp: Optional timestamp for historical imagery (format: 'YYYY-MM-DD', 'YYYY-MM', or Unix timestamp)
    """
    zoom = calculate_zoom_level(ground_distance_km, image_size)

    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        'center': f"{lat},{lon}",
        'zoom': str(zoom),
        'size': f"{image_size}x{image_size}",
        'scale': str(scale),
        'maptype': maptype,
        'key': api_key
    }

    # Add timestamp parameter for historical imagery if provided
    if timestamp:
        # Convert date string to Unix timestamp if needed
        if isinstance(timestamp, str) and ('-' in timestamp):
            import datetime
            import time
            try:
                # Try parsing as YYYY-MM-DD
                if len(timestamp.split('-')) == 3:
                    dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d')
                # Try parsing as YYYY-MM
                elif len(timestamp.split('-')) == 2:
                    dt = datetime.datetime.strptime(timestamp, '%Y-%m')
                unix_timestamp = int(time.mktime(dt.timetuple()))
                params['timestamp'] = str(unix_timestamp)
            except ValueError:
                print(f"Warning: Invalid timestamp format '{timestamp}'. Using current date.")
        else:
            # Assume it's already a Unix timestamp
            params['timestamp'] = str(timestamp)
    param_str = '&'.join(f"{k}={v}" for k, v in params.items())
    return f"{base_url}?{param_str}"

def save_static_map_image(lat, lon, name, api_key, out_dir, ground_distance_km=1.0, image_size=640, maptype='satellite', timestamp=None, current=None, total=None):
    """Downloads and saves a Google Maps Static image for the location if valid imagery is available."""
    url = build_gmaps_static_url(lat, lon, api_key, ground_distance_km, image_size, maptype, timestamp)
    response = requests.get(url)

    if response.status_code == 200:
        # Check if response is the "no imagery" placeholder
        if b"Sorry, we have no imagery here" in response.content:
            print(f"No imagery at ({lat}, {lon}) — skipping.")
            return False  # indicates no image saved

        # Ensure output directory exists and filename is safe
        os.makedirs(out_dir, exist_ok=True)
        safe_name = _sanitize_name(name)
        filename = f"{safe_name}_{lat:.5f}_{lon:.5f}_{ground_distance_km}km.png"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Saved ({current}/{total}): {filepath}")
        return True  # image saved
    else:
        print(f"Failed to fetch image for {name} ({lat}, {lon}): {response.status_code} - {response.text}")
        return False  # image not saved


import glob

def screenshot_all_csv_locations(csv_path, api_key, base_out_dir, ground_distance_km=1.0, image_size=640, maptype='satellite', timestamp=None):
    out_dir = base_out_dir
    os.makedirs(out_dir, exist_ok=True)

    coordinates = parse_csv_coordinates_auto(csv_path)
    total = len(coordinates)
    print(f"Total coordinates to process: {total}")
    print(f"Output folder: {out_dir}")

    saved_count = 0
    for idx, coord in enumerate(coordinates, 1):
        # Build expected filename to check for existence
        safe_name = _sanitize_name(coord['name'])
        expected_filename = f"{safe_name}_{coord['lat']:.5f}_{coord['lon']:.5f}_{ground_distance_km}km.png"
        expected_path = os.path.join(out_dir, expected_filename)

        # Skip if file already exists
        if os.path.exists(expected_path):
            print(f"Skipping ({saved_count+1}/{total}): {expected_filename} already exists")
            saved_count += 1
            continue

        # Attempt to fetch/save image
        success = save_static_map_image(
            coord['lat'], coord['lon'], coord['name'], api_key, out_dir,
            ground_distance_km=ground_distance_km,
            image_size=image_size,
            maptype=maptype,
            timestamp=timestamp,
            current=saved_count + 1,
            total=total
        )
        if success:
            saved_count += 1

    print(f"\n✅ Finished. {saved_count} valid images saved out of {len(coordinates)} locations.")



if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Collect satellite images of CSV coordinate locations using Google Maps Static API')
    parser.add_argument('--csv', default="coordinates/non_aaa_nk_coords_500.csv", help='Path to CSV file')
    parser.add_argument('--output', default="../IMINT-Images/Non_AAA_training_images", help='Output directory for screenshots')
    parser.add_argument('--distance', type=float, default=0.5,
                        help='Ground distance in kilometers (width/height of the image)')
    parser.add_argument('--size', type=int, default=640,
                        help='Image size in pixels (square image)')
    parser.add_argument('--maptype', default="satellite",
                        choices=['roadmap', 'satellite', 'hybrid', 'terrain'],
                        help='Map type')
    parser.add_argument('--date', help='Date for historical imagery (format: YYYY-MM-DD or YYYY-MM)')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Configuration
    CSV_PATH = args.csv
    API_KEY = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not API_KEY:
        print("Error: GOOGLE_MAPS_STATIC_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment variables.")
        exit(1)

    OUT_DIR = args.output
    GROUND_DISTANCE_KM = args.distance
    IMAGE_SIZE = args.size
    MAPTYPE = args.maptype
    TIMESTAMP = args.date

    # Print configuration
    print(f"Configuration:")
    print(f"  CSV file: {CSV_PATH}")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Ground distance: {GROUND_DISTANCE_KM} km")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
    print(f"  Map type: {MAPTYPE}")
    print(f"  Date: {TIMESTAMP if TIMESTAMP else 'Current (latest available)'}")

    # Check if CSV file exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file '{CSV_PATH}' not found.")
        exit(1)

    # Take screenshots
    screenshot_all_csv_locations(
        CSV_PATH, API_KEY, OUT_DIR,
        ground_distance_km=GROUND_DISTANCE_KM,
        image_size=IMAGE_SIZE,
        maptype=MAPTYPE,
        timestamp=TIMESTAMP
    )