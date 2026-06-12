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
# PNG maps + info sheets only
python scripts/run_all.py --output-dir "G:\Shared drives\...\0_Listing\1_Input"

# PNG + MBTiles (for SurveyCTO offline basemaps) + info sheets
python scripts/run_all.py --mbtiles --output-dir "G:\Shared drives\...\0_Listing\1_Input"
```

Or step by step, in order:

```bash
# 1. Build grids + PPS sub-cell selection + VCSL village flags
#    Writes:  01_input_data/boundaries/control_grid_5km_flagged.gpkg
#             01_input_data/boundaries/selected_subcells_500m.gpkg
#    Both get a `vcsl_village` column via an attribute merge with the CCT-provided
#    Sampled_control_pixels.gpkg (see Data Location — it lives outside data_dir).
python scripts/build_data.py --force

# 2. Generate maps + GeoJSON polygons (+ optionally MBTiles)
#    Reads:   the two .gpkg files above + overture_buildings.parquet
#    Writes:  overview_5km.png, subcell_<id>_<role>.png, subcell_<id>_<role>.geojson
#             overview_5km.mbtiles                          (per cell folder, with --mbtiles)
#             layers/<ward>_<5km_id>_<status>_<subcell_id>/...mbtiles + samplegrids_*.xlsx (with --mbtiles)
python scripts/generate_all_maps.py --output-dir "G:\...\1_Input"
python scripts/generate_all_maps.py --mbtiles --output-dir "G:\...\1_Input"

# 3. Generate HTML info sheets
#    Reads:   the PNGs written by step 2 (embedded as <img> in the HTML)
#    Writes:  info_sheet.html per cell folder
python scripts/generate_info_sheets.py --output-dir "G:\...\1_Input"
```

**Important**: step 3 must run *after* step 2 — the info sheet embeds the PNG maps by filename. MBTiles files are not referenced by the info sheet: the 5km overview tile is dropped into the cell folder, and the 500m sub-cell tiles go to `layers/` (see Output Structure) for SurveyCTO to pick up directly.

Test with a few cells first: add `--limit 3` to any command, or `--single <5km_id>` to target one cell.

### MBTiles-specific flags

`generate_all_maps.py` (and `run_all.py`) also accept:

```bash
--mbtiles                    # emit .mbtiles alongside PNGs
--mbtiles-only               # emit only .mbtiles, skip PNGs
--mbtiles-detail-zoom 19     # max zoom for 500m detail tiles (default 19)
--mbtiles-overview-zoom 15   # max zoom for 5km overview tiles (default 15)
```

See **MBTiles** below for the full story.

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
│       ├── info_sheet.html                   # metadata, interactive Leaflet maps, Google Maps links
│       ├── overview_5km.png                  # 5km cell with sub-cells highlighted
│       ├── overview_5km.mbtiles              # (optional) 5km overview tiles — stays in the cell folder
│       ├── primary/
│       │   ├── subcell_<id>_primary.png      # detail map with buildings + centroid crosshair
│       │   └── subcell_<id>_primary.geojson  # polygon for SurveyCTO
│       └── reserve/
│           ├── subcell_<id>_reserve.png
│           └── subcell_<id>_reserve.geojson
├── replacement/
│   └── ...
└── layers/                                   # ⚠️ 500m sub-cell MBTiles live HERE, not in the cell folders
    ├── <ward_name>_<5km_id>_<status>_<subcell_id>/
    │   └── <ward_name>_<5km_id>_<status>_<subcell_id>.mbtiles
    └── samplegrids_primary_replacement.xlsx  # index of every layer (ward, 5km_id, status, role,
                                              #   building count, VCSL village, lat/lon, mbtiles path)
```

> **⚠️ MBTiles are split across two places.** The 5km **overview** tiles (`overview_5km.mbtiles`) sit
> inside each cell folder under `sampled/`/`replacement/`, next to the PNGs. The 500m **sub-cell**
> tiles are *not* in the cell folders — they are written as flat, self-named layers under
> `layers/<ward>_<5km_id>_<status>_<subcell_id>/`, each with a matching `samplegrids_primary_replacement.xlsx`
> index. This keeps the per-sub-cell tiles in one easy-to-load place for SurveyCTO.

The `.mbtiles` files (and the `layers/` folder + xlsx) are only written when `--mbtiles`
(or `--mbtiles-only`) is passed.

### Map features

- **Overview (5km)**: Google Hybrid basemap (satellite + village labels), 5km cell boundary, selected sub-cells outlined (green=primary, goldenrod=reserve), red crosshair at each sub-cell centroid
- **Detail (500m)**: Google Hybrid basemap, building footprints (yellow/orange), sub-cell boundary in green/goldenrod, red crosshair at the centroid
- **Info sheet**: Cell metadata, sub-cell table, Google Maps links, interactive Leaflet maps with polygon overlay
- **GeoJSON**: WGS84 polygons per sub-cell for SurveyCTO integration (includes the `vcsl_village` property)
- **MBTiles**: Georeferenced raster tiles (same overlays as the PNG burned into the pixels) for SurveyCTO offline basemaps — see below
- **VCSL village flag**: cells overlapping a CCT-flagged VCSL village carry a `vcsl_village` label, surfaced on the info sheet, in the GeoJSON, in the MBTiles title banner, and in `layers/samplegrids_primary_replacement.xlsx`

## MBTiles (offline basemaps for SurveyCTO)

SurveyCTO can load a local MBTiles file as an offline basemap inside a form. The same overlays you see in the PNG (sub-cell boundary, building footprints, centroid crosshair) are burned into the raster pixels so field staff see them without a network connection.

Enable with `--mbtiles` on any of `run_all.py` / `generate_all_maps.py`:

```bash
# Emit both PNG and MBTiles (recommended — info sheets need the PNGs)
python scripts/generate_all_maps.py --mbtiles --single 13151

# MBTiles only (PNGs skipped — use if you already generated PNGs earlier)
python scripts/generate_all_maps.py --mbtiles-only
```

Each `.mbtiles` file has a title banner burned *above* the geographic bbox (so it never obscures map content) showing the cell ID, sub-cell ID, role, building count, and the VCSL village (when the cell is flagged).

The 500m sub-cell tiles are written to `layers/` as self-named files
(`<ward>_<5km_id>_<status>_<subcell_id>.mbtiles`), each in its own folder, with a
`samplegrids_primary_replacement.xlsx` listing every layer. The 5km overview tiles stay in the
per-cell folder. See **Output Structure** above.

**File size / zoom trade-off.** Defaults are `detail=19`, `overview=15`. Typical sizes:

| File                                            | Default zoom | Typical size |
| ----------------------------------------------- | ------------ | ------------ |
| `layers/.../<...>_<subcell_id>.mbtiles` (500m)  | 19           | 1–3 MB       |
| `overview_5km.mbtiles` (5km)                    | 15           | 2–8 MB       |

Drop zoom by 1 to roughly halve the file size:

```bash
python scripts/generate_all_maps.py --mbtiles \
    --mbtiles-detail-zoom 18 --mbtiles-overview-zoom 14
```

**Prototyping**: [`notebooks/09_mbtiles_conversion_test.ipynb`](notebooks/09_mbtiles_conversion_test.ipynb) walks through the conversion pipeline (download Google Hybrid tiles → burn overlays → MBTiles + overviews) on a single test cell. Useful for tuning zoom/size before running on all cells.

**SurveyCTO test cells**: [`notebooks/08_surveycto_test.ipynb`](notebooks/08_surveycto_test.ipynb) generates test MBTiles (+ GeoJSON + PNG + HTML) for office locations in Dar es Salaam / Kampala / Amsterdam — one square and one chevron shape per office — for end-to-end SurveyCTO form testing before a real field deployment.

## Project Structure

```
listing_geospatial_cct/
├── src/
│   ├── data_processing/        # load_boundaries.py: grid loading, sub-cell selection (PPS)
│   ├── mapping/                # map_generator.py: PNG maps
│   │                           # mbtiles_export.py: styled MBTiles for SurveyCTO
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
│   ├── 08_surveycto_test.ipynb          # Test GeoJSON + MBTiles + PNG for office cells
│   └── 09_mbtiles_conversion_test.ipynb # MBTiles pipeline prototyping
├── config/
│   ├── config.yaml             # Shared config (tracked)
│   └── config.local.yaml       # Local path overrides (gitignored)
├── requirements.txt
└── PROJECT_INFO.md             # Detailed project documentation
```

## Data Location

All geospatial data lives on Google Drive (not in this repo). Most of it sits under the
`data_dir` (`0.4_listing_geospatial`), **but the VCSL flags file is in a different tree** — the
listing input folder, not `data_dir`:

```
1_Baseline/0.4_listing_geospatial/         # <- data_dir (config paths.data_dir)
├── 01_input_data/
│   ├── boundaries/              # 5km grid, sub-grids, selected sub-cells (now incl. vcsl_village)
│   └── base_layers/             # overture_buildings.parquet, roads
└── 02_outputs/
    └── building_counts/         # Grid files with building counts + choropleths

0_Listing/1_Input/grid_info/_temp/         # <- DIFFERENT tree, outside data_dir
└── Sampled_control_pixels.gpkg # CCT-provided VCSL village flags (5km-pixel level)
```

The VCSL flags path is configured separately as `paths.vcsl_flags` in `config/config.yaml`
(override it in `config.local.yaml` if your mount differs). `build_data.py` reads it to add the
`vcsl_village` column to the control grid and selected sub-cells.

## Configuration

Default config points to the shared Google Drive. Override locally:

```yaml
# config/config.local.yaml (gitignored)
paths:
  data_dir: "D:/My Drive/0.4_listing_geospatial"
```

## Tech Stack

Python (geopandas, matplotlib, contextily, folium), Google Drive for data, GitHub for code, SurveyCTO for field deployment.
