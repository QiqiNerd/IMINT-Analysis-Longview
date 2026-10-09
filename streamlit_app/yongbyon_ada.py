import zipfile
import os
import io
from xml.etree import ElementTree as ET
import math
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import requests
from math import cos, radians

def extract_kmz(kmz_path, extract_to):
    """Extracts a KMZ file to the specified directory."""
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kmz.extractall(extract_to)

def parse_kml_placemarks(kml_file_path):
    """Parses a KML file and returns a list of placemark dicts with name, lat, lon."""
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_file_path)
    root = tree.getroot()
    placemarks = root.findall('.//kml:Placemark', namespace)
    placemark_data = []
    for placemark in placemarks:
        name_elem = placemark.find('kml:name', namespace)
        coord_elem = placemark.find('.//kml:coordinates', namespace)
        if name_elem is not None and coord_elem is not None:
            name = name_elem.text.strip()
            coords_text = coord_elem.text.strip()
            lon, lat, *_ = map(float, coords_text.split(','))
            placemark_data.append({
                'name': name,
                'lat': lat,
                'lon': lon
            })
    return placemark_data

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth (specified in decimal degrees). Returns miles."""

    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 3958.8  # Radius of earth (mi)
    return c * r

def count_nearby_sites(placemark_data, target_coord, keywords, radius_miles):
    """Counts placemarks matching keywords within radius_miles of target_coord."""
    count = 0
    matching_sites = []
    for site in placemark_data:
        if any(keyword in site['name'].lower() for keyword in keywords):
            distance = haversine_distance(site['lat'], site['lon'], target_coord[0], target_coord[1])
            if distance <= radius_miles:
                count += 1
                matching_sites.append(site)
    return count, matching_sites

def analyze_kmz_sites(kmz_path, kml_extract_path, kml_filename, target_sites, radius_miles, keywords):
    """
    Extracts KMZ, parses KML, and summarizes nearby sites.
    Returns a DataFrame summary.
    """
    if target_sites is None:
        target_sites = {
            "Yongbyon Nuclear Science and Weapons center": (39.7961, 125.7558),
            "Kangson U enrichment plant": (38.9513, 125.6123),
            "Pyongsan uranium mine and mill": (38.200, 126.4365),
            "Yongbyon U enrichment": (39.7679, 125.7511),
            "Yongbyon Pu reprocessing": (39.7802, 125.7535)
        }
    if keywords is None:
        keywords = ['aaa']

    extract_kmz(kmz_path, kml_extract_path)
    kml_file_path = os.path.join(kml_extract_path, kml_filename)
    placemark_data = parse_kml_placemarks(kml_file_path)

    results = {}
    for name, coord in target_sites.items():
        count, matches = count_nearby_sites(placemark_data, coord, keywords, radius_miles)
        results[name] = {
            'count': count,
            'matches': matches
        }

    keywords_label = ', '.join(keywords)
    col_name = f"Nearby {keywords_label} Sites (within {radius_miles} miles)"
    summary_df = pd.DataFrame([
        {"Location": name, col_name: data['count']}
        for name, data in results.items()
    ]).sort_values(by=col_name, ascending=False)

    return summary_df, results

def get_mapbox_satellite_image(lat, lon, zoom=10.63, width=600, height=400):
    """
    Get a satellite image from Mapbox API for the specified coordinates.
    Returns the image as a PIL Image object.
    """
    # Try to get token from Streamlit secrets first, then fall back to environment variables
    try:
        access_token = st.secrets['MAPBOX_TOKEN']
    except KeyError:
        import os
        access_token = os.environ.get("MAPBOX_TOKEN")

    if not access_token:
        st.warning("Mapbox token not found in secrets or environment variables. Satellite imagery won't be available.")
        return None

    base_url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
    bearing = 0

    url = f"{base_url}/{lon},{lat},{zoom},{bearing}/{width}x{height}"

    params = {
        "access_token": access_token
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            st.error(f"Mapbox API request failed with status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching satellite image: {str(e)}")
        return None

def draw_on_satellite_image(image, matching_sites, lat, lon, radius_miles, zoom_level):
    """Draw markers and radius on the satellite image using PIL."""
    if image is None:
        return None

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Image dimensions
    width, height = image.size
    center_x, center_y = width // 2, height // 2

    # Draw search radius (blue circle)
    # Formula: degrees_per_pixel_lon = 360 / (256 * 2^zoom_level)
    degrees_per_pixel_lon = 360.0 / (256 * (2 ** zoom_level))
    delta_lon = width * degrees_per_pixel_lon  # width is image width in pixels
    center_lat = lat
    miles_per_deg_lon = 69.172 * cos(radians(center_lat))
    map_width_miles = delta_lon * miles_per_deg_lon
    miles_per_pixel = map_width_miles / width
    radius_px = int(radius_miles / miles_per_pixel)

    draw.ellipse(
        [(center_x - radius_px, center_y - radius_px),
         (center_x + radius_px, center_y + radius_px)],
        outline="blue", width=2
    )

    # Use accurate degrees-per-pixel for both longitude and latitude
    degrees_per_pixel_lon = 360.0 / (256 * (2 ** zoom_level))
    # Latitude: world map goes from -85.0511 to +85.0511 (Mercator)
    lat_min = -85.0511
    lat_max = 85.0511
    degrees_per_pixel_lat = (lat_max - lat_min) / (256 * (2 ** zoom_level))

    for site in matching_sites:
        dx = (site['lon'] - lon) / degrees_per_pixel_lon
        dy = -(site['lat'] - lat) / degrees_per_pixel_lat  # Negative because y increases downward
        x, y = center_x + dx, center_y + dy

        # Draw green circle for the site
        draw.ellipse([(x-6, y-6), (x+6, y+6)], fill="green", outline="white", width=1)

        # Alternative approach for text that's more visible
        # Create small black rectangle behind text for better visibility
        text = site['name']
        text_width = len(text) * 7  # Rough estimate of text width
        text_height = 15
        draw.rectangle([(x+10, y-5), (x+10+text_width, y+text_height)], fill=(0, 0, 0, 128))
        draw.text((x+12, y-3), text, fill="white")

    # Draw target location (red dot in center)
    draw.ellipse([(center_x-8, center_y-8), (center_x+8, center_y+8)],
                 fill="red", outline="white", width=2)

    # Add legend
    legend_x = 20
    legend_y = height - 80

    # Target legend
    draw.ellipse([(legend_x, legend_y), (legend_x+16, legend_y+16)], fill="red", outline="white", width=2)
    draw.text((legend_x+25, legend_y+3), "Target Location", fill="white")

    # Sites legend
    draw.ellipse([(legend_x, legend_y+30), (legend_x+16, legend_y+30+16)], fill="green", outline="white", width=1)
    draw.text((legend_x+25, legend_y+30+3), "Matching Sites", fill="white")

    return image

def main():
    st.title("Yongbyon ADA Analysis")

    kmz_file = st.file_uploader("Upload KMZ file", type=["kmz"])
    if not kmz_file:
        st.info("Please upload a KMZ file.")
        return

    # Save uploaded file
    kmz_path = "uploaded.kmz"
    kml_extract_path = "uploaded_kml"
    kml_filename = "doc.kml"
    with open(kmz_path, "wb") as f:
        f.write(kmz_file.read())

    # User input for target location and parameters
    st.sidebar.header("Target Site Parameters")
    lat_str = st.sidebar.text_input("Target Latitude", value="39.7961")
    lon_str = st.sidebar.text_input("Target Longitude", value="125.7558")
    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except ValueError:
        st.sidebar.error("Please enter valid numbers for latitude and longitude.")
        return
    radius_miles = st.sidebar.number_input("Radius (miles)", value=5.0, min_value=0.1, step=0.1, format="%f")
    keywords_str = st.sidebar.text_input("Keywords (comma separated)", value="aaa")
    keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]

    # Map settings
    st.sidebar.header("Map Settings")
    zoom_level = st.sidebar.slider("Zoom Level", min_value=9.0, max_value=16.0, value=12.0, step=0.5)
    map_width = st.sidebar.slider("Map Width (px)", min_value=400, max_value=1200, value=800, step=100)
    map_height = st.sidebar.slider("Map Height (px)", min_value=300, max_value=800, value=600, step=100)

    # Extract and parse KMZ/KML
    extract_kmz(kmz_path, kml_extract_path)
    kml_file_path = os.path.join(kml_extract_path, kml_filename)
    placemark_data = parse_kml_placemarks(kml_file_path)

    # Find matching sites
    count, matching_sites = count_nearby_sites(placemark_data, (lat, lon), keywords, radius_miles)
    st.write(f"Found {count} matching sites within {radius_miles} miles of ({lat}, {lon}) for keywords: {', '.join(keywords)}")

    # Prepare DataFrame for display and mapping
    if matching_sites:
        df = pd.DataFrame(matching_sites)

        # --- Download KMZ for found sites ---
        def generate_kml(sites):
            kml = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<kml xmlns="http://www.opengis.net/kml/2.2">',
                '  <Document>'
            ]
            for site in sites:
                kml.append(f'''    <Placemark>\n      <n>{site['name']}</n>\n      <Point><coordinates>{site['lon']},{site['lat']},0</coordinates></Point>\n    </Placemark>''')
            kml.append('  </Document>')
            kml.append('</kml>')
            return '\n'.join(kml)

        kml_str = generate_kml(matching_sites)
        kml_bytes = kml_str.encode("utf-8")
        kmz_buf = io.BytesIO()
        with zipfile.ZipFile(kmz_buf, 'w', zipfile.ZIP_DEFLATED) as kmz:
            kmz.writestr('doc.kml', kml_bytes)
        kmz_buf.seek(0)

        # Create a satellite map with sites marked
        with st.spinner("Fetching satellite imagery..."):
            satellite_image = get_mapbox_satellite_image(lat, lon, zoom=zoom_level, width=map_width, height=map_height)

            if satellite_image:
                # Draw markers and radius on the image
                satellite_image_with_markers = draw_on_satellite_image(
                    satellite_image.copy(), matching_sites, lat, lon, radius_miles, zoom_level
                )

                # Display the image
                st.write("### Satellite Map")
                st.image(satellite_image_with_markers, caption=f"Target: ({lat}, {lon}) | Radius: {radius_miles} miles | Found: {count} sites")

                # Save the images for download
                img_with_markers_path = "satellite_with_markers.png"
                img_original_path = "satellite_original.png"
                satellite_image_with_markers.save(img_with_markers_path)
                satellite_image.save(img_original_path)

                # Add download button for the original satellite image without markers
                with open(img_original_path, "rb") as file:
                    st.download_button(
                        label="Download Satellite Image",
                        data=file,
                        file_name=f"satellite_{lat}_{lon}_zoom{zoom_level}_original.png",
                        mime="image/png"
                    )

                # Add download button for the satellite image with markers
                with open(img_with_markers_path, "rb") as file:
                    st.download_button(
                        label="Download Satellite Image with Markers",
                        data=file,
                        file_name=f"satellite_{lat}_{lon}_zoom{zoom_level}_with_markers.png",
                        mime="image/png"
                    )
            else:
                st.error("Could not fetch satellite imagery. Please check your MAPBOX_TOKEN environment variable.")

                st.write("### Fallback Map")
                fig = px.scatter_mapbox(
                    df,
                    lat="lat",
                    lon="lon",
                    hover_name="name",
                    zoom=zoom_level,
                    height=map_height,
                    width=map_width,
                )

                # Add target location as a marker
                fig.add_trace(go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode="markers",
                    marker=dict(size=15, color="red"),
                    name="Target Site",
                ))

                # Setup the map layout
                fig.update_layout(
                    mapbox_style="open-street-map",  # Fallback to OSM when no token
                    mapbox_center_lat=lat,
                    mapbox_center_lon=lon,
                    margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox_zoom=zoom_level,
                )

                st.plotly_chart(fig)

        # Display tabular data
        st.write("### Found Sites")
        st.dataframe(df)

        # Add a download button for the current map data
        with io.StringIO() as buffer:
            df.to_csv(buffer, index=False)
            st.download_button(
                label="Download Sites Data (CSV)",
                data=buffer.getvalue(),
                file_name=f"sites_data_{lat}_{lon}_{radius_miles}miles.csv",
                mime="text/csv"
            )
    else:
        st.warning("No matching sites found.")

if __name__ == "__main__":
    main()