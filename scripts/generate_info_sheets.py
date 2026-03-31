"""Generate HTML info sheets for each 5km control cell.

Each sheet includes:
  - Cell metadata (ID, ward, district, sample status)
  - Table of selected 500m sub-cells with Google Maps links
  - Embedded overview and detail map images (if present)

Outputs one HTML file per cell into the same folder structure
used by generate_all_maps.py.

Usage:
    python scripts/generate_info_sheets.py
    python scripts/generate_info_sheets.py --output-dir "G:/path/to/output"
    python scripts/generate_info_sheets.py --limit 3
    python scripts/generate_info_sheets.py --single 13682
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
from tqdm import tqdm

from src.utils.config_loader import load_config, get_data_dir, get_output_dir
from src.data_processing.load_boundaries import (
    load_control_grid,
    load_selected_subcells,
    validate_crs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HTML info sheets for control cells."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: from config).",
    )
    parser.add_argument(
        "--single", default=None,
        help="Generate for a single 5km cell ID only.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N cells (for testing).",
    )
    parser.add_argument(
        "--status", choices=["sampled", "replacement"], default=None,
        help="Generate only for cells with this sample_status.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', str(name)).strip('_')


def get_ward_name(grid_id, selected_subcells: gpd.GeoDataFrame) -> str:
    cell_subcells = selected_subcells[selected_subcells["5km_id"] == grid_id]
    if "ward_name" in cell_subcells.columns and len(cell_subcells) > 0:
        ward = cell_subcells.iloc[0]["ward_name"]
        if pd.notna(ward) and str(ward).strip():
            return sanitize_name(str(ward).strip())
    return "unknown_ward"


def get_district(grid_id, selected_subcells: gpd.GeoDataFrame) -> str:
    cell_subcells = selected_subcells[selected_subcells["5km_id"] == grid_id]
    if "district" in cell_subcells.columns and len(cell_subcells) > 0:
        district = cell_subcells.iloc[0]["district"]
        if pd.notna(district) and str(district).strip():
            return str(district).strip()
    return "—"


def google_maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def find_images(cell_folder: Path) -> dict:
    """Find overview and detail PNGs in the cell folder."""
    images = {"overview": None, "primary": [], "reserve": []}

    overview = cell_folder / "overview_5km.png"
    if overview.exists():
        images["overview"] = "overview_5km.png"

    for role in ["primary", "reserve"]:
        role_dir = cell_folder / role
        if role_dir.exists():
            for png in sorted(role_dir.glob("*.png")):
                images[role].append(f"{role}/{png.name}")

    return images


def subcell_to_geojson(subcell_row, crs) -> dict:
    """Convert a sub-cell geometry to WGS84 GeoJSON."""
    geom = subcell_row.geometry
    # Reproject to WGS84 if needed
    if crs and crs.to_epsg() != 4326:
        gs = gpd.GeoSeries([geom], crs=crs).to_crs(epsg=4326)
        geom = gs.iloc[0]
    return mapping(geom)


def generate_html(
    grid_id: int,
    sample_status: str,
    ward_name: str,
    district: str,
    subcells: gpd.GeoDataFrame,
    images: dict,
) -> str:
    """Build the HTML content for one cell's info sheet."""

    # Sort: primary first
    subcells_sorted = subcells.sort_values("selection_role", ascending=True)

    # Sub-cell table rows
    rows_html = ""
    for _, sc in subcells_sorted.iterrows():
        sc_id = sc.get("grid_id", "—")
        role = sc["selection_role"]
        buildings = sc.get("building_count", "—")
        lat = sc.get("latitude", None)
        lon = sc.get("longitude", None)

        if lat is not None and lon is not None:
            maps_link = f'<a href="{google_maps_url(lat, lon)}" target="_blank">Open in Google Maps</a>'
            coords = f"{lat:.5f}, {lon:.5f}"
        else:
            maps_link = "—"
            coords = "—"

        role_badge = (
            f'<span class="badge primary">{role}</span>'
            if role == "primary"
            else f'<span class="badge reserve">{role}</span>'
        )

        rows_html += f"""
        <tr>
            <td>{sc_id}</td>
            <td>{role_badge}</td>
            <td>{buildings}</td>
            <td>{coords}</td>
            <td>{maps_link}</td>
        </tr>"""

    # Build Leaflet mini-maps for each sub-cell
    leaflet_maps_html = ""
    for idx, (_, sc) in enumerate(subcells_sorted.iterrows()):
        sc_id = sc.get("grid_id", f"subcell_{idx}")
        role = sc["selection_role"]
        buildings = sc.get("building_count", "?")
        lat = sc.get("latitude", None)
        lon = sc.get("longitude", None)
        geojson = subcell_to_geojson(sc, subcells.crs)
        geojson_str = json.dumps(geojson)

        color = "#22c55e" if role == "primary" else "#eab308"
        map_id = f"map_{idx}"

        gmaps_btn = ""
        if lat is not None and lon is not None:
            gmaps_btn = (
                f'<a href="{google_maps_url(lat, lon)}" target="_blank" '
                f'class="gmaps-btn">Open in Google Maps</a>'
            )

        leaflet_maps_html += f"""
    <div class="subcell-card">
        <h3>Sub-cell {sc_id} — <span class="badge {role}">{role}</span> — {buildings} buildings</h3>
        <div id="{map_id}" class="leaflet-map"></div>
        {gmaps_btn}
        <script>
            (function() {{
                var geojson = {geojson_str};
                var map = L.map('{map_id}', {{zoomControl: true}});
                L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}', {{
                    maxZoom: 20,
                    attribution: 'Google Hybrid'
                }}).addTo(map);
                var layer = L.geoJSON(geojson, {{
                    style: {{color: '{color}', weight: 3, fillOpacity: 0.15, fillColor: '{color}'}}
                }}).addTo(map);
                map.fitBounds(layer.getBounds().pad(0.3));
            }})();
        </script>
    </div>
"""

    # Image sections
    overview_html = ""
    if images["overview"]:
        overview_html = f"""
    <h2>Overview Map (5km)</h2>
    <img src="{images['overview']}" alt="Overview map" class="map-img">
    """

    detail_html = ""
    for role in ["primary", "reserve"]:
        if images[role]:
            detail_html += f'<h2>Detail Maps — {role.title()}</h2>\n'
            for img_path in images[role]:
                detail_html += f'    <img src="{img_path}" alt="{role} detail" class="map-img">\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cell {grid_id} — {ward_name}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            border-bottom: 3px solid #2563eb;
            padding-bottom: 10px;
        }}
        .meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin: 20px 0;
        }}
        .meta-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .meta-card .label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .meta-card .value {{
            font-size: 1.3em;
            font-weight: 600;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2563eb;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8fafc;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge.primary {{
            background: #d1fae5;
            color: #065f46;
        }}
        .badge.reserve {{
            background: #fef3c7;
            color: #92400e;
        }}
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .map-img {{
            width: 100%;
            max-width: 1100px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            margin: 10px 0 30px 0;
        }}
        h2 {{
            margin-top: 40px;
            color: #1e40af;
        }}
        .subcell-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .subcell-card h3 {{
            margin-top: 0;
        }}
        .leaflet-map {{
            width: 100%;
            height: 400px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .gmaps-btn {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background: #2563eb;
            color: white !important;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .gmaps-btn:hover {{
            background: #1d4ed8;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <h1>Cell {grid_id} — {ward_name}</h1>

    <div class="meta">
        <div class="meta-card">
            <div class="label">5km Cell ID</div>
            <div class="value">{grid_id}</div>
        </div>
        <div class="meta-card">
            <div class="label">Ward</div>
            <div class="value">{ward_name}</div>
        </div>
        <div class="meta-card">
            <div class="label">District</div>
            <div class="value">{district}</div>
        </div>
        <div class="meta-card">
            <div class="label">Sample Status</div>
            <div class="value">{sample_status}</div>
        </div>
        <div class="meta-card">
            <div class="label">Selected Sub-cells</div>
            <div class="value">{len(subcells_sorted)}</div>
        </div>
    </div>

    <h2>Selected 500m Sub-cells</h2>
    <table>
        <thead>
            <tr>
                <th>Sub-cell ID</th>
                <th>Role</th>
                <th>Buildings</th>
                <th>Centroid (lat, lon)</th>
                <th>Navigate</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <h2>Interactive Maps</h2>
    {leaflet_maps_html}

    {overview_html}
    {detail_html}
</body>
</html>"""


def main():
    args = parse_args()

    config = load_config()
    data_dir = get_data_dir(config)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_output_dir(config) / "listing_maps"

    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    grid_5km = load_control_grid(data_dir)
    grid_5km = validate_crs(grid_5km)

    selected = load_selected_subcells(data_dir)
    if selected is None:
        print("Error: No selected sub-cells found. Run build_data.py first.")
        sys.exit(1)
    selected = validate_crs(selected)

    # Exclude dropped sparse cells
    grid_5km = grid_5km[grid_5km["sample_status"] != "dropped_sparse"].copy()
    selected = selected[selected["sample_status"] != "dropped_sparse"].copy()

    # Filters
    if args.status:
        grid_5km = grid_5km[grid_5km["sample_status"] == args.status].copy()
        selected = selected[selected["sample_status"] == args.status].copy()
        print(f"Filtered to '{args.status}': {len(grid_5km)} cells")

    if args.single:
        grid_5km = grid_5km[grid_5km["id"].astype(str) == str(args.single)].copy()
        selected = selected[selected["5km_id"].astype(str) == str(args.single)].copy()
        if len(grid_5km) == 0:
            print(f"Error: Cell '{args.single}' not found.")
            sys.exit(1)

    if args.limit:
        grid_5km = grid_5km.head(args.limit).copy()
        cell_ids = set(grid_5km["id"])
        selected = selected[selected["5km_id"].isin(cell_ids)].copy()
        print(f"Limited to {args.limit} cells")

    # Generate info sheets
    total = len(grid_5km)
    print(f"\nGenerating info sheets for {total} cells...")

    for _, row in tqdm(grid_5km.iterrows(), total=total, desc="Info sheets"):
        grid_id = row["id"]
        sample_status = row["sample_status"]
        cell_selected = selected[selected["5km_id"] == grid_id]

        if len(cell_selected) == 0:
            continue

        ward_name = get_ward_name(grid_id, cell_selected)
        district = get_district(grid_id, cell_selected)
        cell_folder = output_dir / sample_status / f"{ward_name}_{int(grid_id)}"
        cell_folder.mkdir(parents=True, exist_ok=True)

        # Find existing map images
        images = find_images(cell_folder)

        html = generate_html(
            grid_id=int(grid_id),
            sample_status=sample_status,
            ward_name=ward_name,
            district=district,
            subcells=cell_selected,
            images=images,
        )

        html_path = cell_folder / "info_sheet.html"
        html_path.write_text(html, encoding="utf-8")

    print(f"\nDone! Generated {total} info sheets in {output_dir}")


if __name__ == "__main__":
    main()
