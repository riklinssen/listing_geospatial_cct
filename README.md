# Listing Geospatial CCT

Generate high-resolution static maps for enumeration area grid cells, optimized for tablet display via SurveyCTO.

See [PROJECT_INFO.md](PROJECT_INFO.md) for full project goals, data structure, and design decisions.

## Quick Start

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd listing_geospatial_cct
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure paths

The default config points to the shared Google Drive:

```
G:\Shared drives\TZ-CCT_RUBEV-0825\Data\Data\0.4_listing_geospatial
```

If your Google Drive is mounted at a different path, create a local override:

```bash
copy config\config.local.yaml config\config.local.yaml
```

Edit `config/config.local.yaml` and uncomment/set your paths:

```yaml
paths:
  data_dir: "D:/My Drive/0.4_listing_geospatial"
  output_dir: "C:/projects/listing_geospatial_cct/output"
```

The `.local.yaml` file is gitignored so it won't affect other team members.

### 5. Verify setup

```bash
python scripts/test_setup.py
```

This checks that all packages are installed, config loads correctly, and data files are accessible.

## Usage

### Notebooks (interactive exploration)

Open notebooks in VS Code with the Jupyter extension:

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_explore_data.ipynb` | Load and inspect boundary files, quick visualization |
| `notebooks/02_test_map_generation.ipynb` | Generate a single test map, tweak styling |
| `notebooks/03_batch_processing.ipynb` | Generate maps for all grid cells with progress tracking |

Make sure to select the `venv` Python interpreter as the notebook kernel.

### Scripts (production / automation)

Generate maps for all grid cells:

```bash
python scripts/generate_all_maps.py
```

Options:

```bash
# Specify which column contains grid IDs
python scripts/generate_all_maps.py --grid-id-col CELL_ID

# Generate without online basemap tiles (offline mode)
python scripts/generate_all_maps.py --no-basemap

# Generate a single grid cell for testing
python scripts/generate_all_maps.py --single 001
```

## Project Structure

```
listing_geospatial_cct/
├── src/
│   ├── data_processing/     # Load boundaries, validate CRS
│   ├── mapping/             # MapGenerator class for PNG output
│   └── utils/               # Config loader, helpers
├── scripts/
│   ├── generate_all_maps.py # CLI for batch map generation
│   └── test_setup.py        # Environment verification
├── config/
│   ├── config.yaml          # Tracked config (shared paths)
│   └── config.local.yaml    # Local overrides (gitignored)
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # pytest tests
├── docs/                    # Documentation
├── PROJECT_INFO.md          # Project goals and data structure
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Data Location

All geospatial data lives on Google Drive (not in this repo):

```
0.4_listing_geospatial/
├── 01_input_data/
│   ├── boundaries/          # enumeration_areas.geojson, control_grid.geojson
│   ├── base_layers/         # roads.geojson, buildings.geojson
│   └── reference_data/
└── 02_outputs/
    └── generated_maps/      # Output PNG maps (one folder per grid cell)
```

## Output

Each grid cell gets a folder with its map:

```
generated_maps/
├── grid_cell_001/
│   └── grid_cell_001_map.png
├── grid_cell_002/
│   └── grid_cell_002_map.png
└── ...
```

Maps are 1920x1080 pixels (tablet-optimized) with:
- Highlighted grid cell boundary in red
- Grid ID label at the center
- Context layers (roads, buildings) when available
- OpenStreetMap basemap when online
