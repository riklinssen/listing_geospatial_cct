# Listing Geospatial CCT

Generate high-resolution field maps and info sheets for household listing in Tanzania. Maps are tablet-optimized for use via SurveyCTO.

## Sampling Approach

The study area (Morogoro/Tanga, Tanzania) is divided into a 5km grid (905 cells), of which 48 are control cells (33 sampled + 15 replacement). The pipeline:

1. **5km level**: Sparse sampled cells (no 500m sub-cell with >=20 buildings) are replaced by viable replacement cells. Activated replacements become primary ("sampled").
2. **500m level**: Within each active 5km cell, 2 primary + 2 reserve 500m sub-cells are drawn using **PPS sampling** (probability proportional to building count).
3. **Listing**: Every 3rd household within each selected 500m sub-cell.

Building counts come from Google Open Buildings (342k footprints via Overture Maps).

## Quick Start

```bash
# Setup
git clone <repo-url>
cd listing_geospatial_cct
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Configure local paths (if Google Drive is mounted differently)
# Edit config/config.local.yaml (gitignored)

# Verify setup
python scripts/test_setup.py
```

## Pipeline

Run the full pipeline with one command:

```bash
python scripts/run_all.py --output-dir "G:\Shared drives\...\0_Listing\1_Input"
```

Or step by step:

```bash
# 1. Build grids + PPS sub-cell selection
python scripts/build_data.py --force

# 2. Generate maps + GeoJSON polygons
python scripts/generate_all_maps.py --output-dir "G:\...\1_Input"

# 3. Generate HTML info sheets (run after maps)
python scripts/generate_info_sheets.py --output-dir "G:\...\1_Input"
```

Test with a few cells first: add `--limit 3` to any command.

### Other scripts

```bash
# Download building footprints + compute counts per grid
python scripts/download_google_buildings.py --grid-size 500 --visualize
```

## Output Structure

```
<output_dir>/
├── sampled/
│   └── <ward_name>_<5km_id>/
│       ├── info_sheet.html              # metadata, interactive Leaflet maps, Google Maps links
│       ├── overview_5km.png             # 5km cell with sub-cells highlighted
│       ├── primary/
│       │   ├── subcell_<id>_primary.png     # detail map with buildings + START marker
│       │   └── subcell_<id>_primary.geojson # polygon for SurveyCTO
│       └── reserve/
│           ├── subcell_<id>_reserve.png
│           └── subcell_<id>_reserve.geojson
└── replacement/
    └── ...
```

### Map features

- **Overview (5km)**: Google Hybrid basemap (satellite + village labels), selected sub-cells highlighted (green=primary, yellow=reserve), centroid markers, sub-cell IDs
- **Detail (500m)**: Google Hybrid basemap, building footprints (yellow/orange), red START crosshair at centroid
- **Info sheet**: Cell metadata, sub-cell table, Google Maps links, interactive Leaflet maps with polygon overlay
- **GeoJSON**: WGS84 polygons per sub-cell for SurveyCTO integration

## Project Structure

```
listing_geospatial_cct/
├── src/
│   ├── data_processing/        # load_boundaries.py: grid loading, sub-cell selection (PPS)
│   ├── mapping/                # map_generator.py: overview + detail map generation
│   └── utils/                  # config_loader.py
├── scripts/
│   ├── run_all.py              # Full pipeline orchestrator
│   ├── build_data.py           # Build grids + PPS selection + 5km replacement
│   ├── generate_all_maps.py    # Maps + GeoJSON export
│   ├── generate_info_sheets.py # HTML info sheets with Leaflet maps
│   ├── download_google_buildings.py  # Building data acquisition + counts
│   └── test_setup.py           # Environment verification
├── notebooks/                  # Exploration and prototyping
│   ├── 01_explore_data.ipynb
│   ├── 02_test_map_generation.ipynb
│   ├── 03_batch_processing.ipynb
│   ├── 03_debug_open_buildings.ipynb
│   ├── 04_building_counts.ipynb
│   ├── 05_test_map_styling.ipynb
│   ├── 06_subcell_selection.ipynb
│   ├── 07_osm_landmarks.ipynb
│   └── 08_surveycto_test.ipynb
├── config/
│   ├── config.yaml             # Shared config (tracked)
│   └── config.local.yaml       # Local path overrides (gitignored)
├── requirements.txt
└── PROJECT_INFO.md             # Detailed project documentation
```

## Data Location

All geospatial data lives on Google Drive (not in this repo):

```
0.4_listing_geospatial/
├── 01_input_data/
│   ├── boundaries/              # 5km grid, sub-grids, selected sub-cells
│   └── base_layers/             # overture_buildings.parquet, roads
└── 02_outputs/
    └── building_counts/         # Grid files with building counts + choropleths
```

## Configuration

Default config points to the shared Google Drive. Override locally:

```yaml
# config/config.local.yaml (gitignored)
paths:
  data_dir: "D:/My Drive/0.4_listing_geospatial"
```

## Tech Stack

Python (geopandas, matplotlib, contextily, folium), Google Drive for data, GitHub for code, SurveyCTO for field deployment.
