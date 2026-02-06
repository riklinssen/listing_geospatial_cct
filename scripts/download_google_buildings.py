"""Download building footprints for enumeration areas.

Uses the Overture Maps CLI to fetch building footprints for the study area.
The Overture Maps dataset includes Google Open Buildings data.

Usage:
    python scripts/download_google_buildings.py
    python scripts/download_google_buildings.py --grid-size 500
    python scripts/download_google_buildings.py --status sampled
    python scripts/download_google_buildings.py --single 42
    python scripts/download_google_buildings.py --skip-counts

Prerequisites:
    pip install overturemaps
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd

from src.utils.config_loader import load_config, get_data_dir
from src.data_processing.load_boundaries import (
    load_control_grid,
    load_subgrid,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download building footprints from Overture Maps for enumeration areas."
    )
    parser.add_argument(
        "--grid-size",
        choices=["5000", "1000", "500"],
        default="5000",
        help="Grid cell size in metres: 5000 (5km), 1000 (1km), or 500 (500m). Default: 5000.",
    )
    parser.add_argument(
        "--status",
        choices=["sampled", "replacement"],
        default=None,
        help="Filter to cells with this sample_status (5km grid only).",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Process a single grid cell ID only (for testing).",
    )
    parser.add_argument(
        "--skip-counts",
        action="store_true",
        help="Skip computing per-cell building counts (just download buildings).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if buildings file already exists.",
    )
    return parser.parse_args()


def load_grid(data_dir: Path, grid_size: str, status: str | None, single: str | None) -> gpd.GeoDataFrame:
    """Load the appropriate grid based on size parameter."""
    if grid_size == "5000":
        grid = load_control_grid(data_dir)
        if status:
            grid = grid[grid["sample_status"] == status].copy()
            print(f"Filtered to {len(grid)} cells with status '{status}'")
    else:
        cell_size = int(grid_size)
        grid = load_subgrid(data_dir, cell_size=cell_size)

    if single:
        if "id" in grid.columns:
            mask = grid["id"].astype(str) == str(single)
        elif "grid_id" in grid.columns:
            mask = grid["grid_id"].astype(str) == str(single)
        else:
            raise ValueError("Grid has no 'id' or 'grid_id' column for filtering")
        grid = grid[mask].copy()
        if len(grid) == 0:
            raise ValueError(f"Grid cell '{single}' not found")
        print(f"Filtered to single cell: {single}")

    return grid


def download_buildings(grid: gpd.GeoDataFrame, output_path: Path) -> bool:
    """Download building footprints using Overture Maps CLI.

    Uses the bounding box of all grid cells to download buildings.
    """
    # Ensure WGS84 for bbox
    grid_wgs84 = grid.to_crs(epsg=4326)
    bounds = grid_wgs84.total_bounds  # minx, miny, maxx, maxy
    bbox = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"

    cmd = [
        "overturemaps", "download",
        "--bbox", bbox,
        "-f", "geoparquet",
        "--type", "building",
        "-o", str(output_path)
    ]

    print(f"\nDownloading buildings from Overture Maps...")
    print(f"  Bbox: {bbox}")
    print(f"  Output: {output_path}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if output_path.exists():
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Download complete: {output_path} ({size_mb:.1f} MB)")
            return True
        else:
            print("Error: Output file not created")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error downloading buildings:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: 'overturemaps' command not found.")
        print("Install with: pip install overturemaps")
        return False


def compute_building_counts(
    buildings_path: Path,
    grid: gpd.GeoDataFrame,
    grid_size: str,
) -> gpd.GeoDataFrame:
    """Compute building counts per grid cell via spatial join."""
    print(f"\nComputing building counts for {len(grid)} grid cells...")

    # Load buildings
    buildings = gpd.read_parquet(buildings_path)
    print(f"Loaded {len(buildings)} buildings")

    # Ensure same CRS
    if buildings.crs != grid.crs:
        buildings = buildings.to_crs(grid.crs)

    # Get centroids for faster spatial join
    buildings["centroid"] = buildings.geometry.centroid
    building_points = buildings.set_geometry("centroid")[["centroid"]].copy()
    building_points = building_points.rename(columns={"centroid": "geometry"}).set_geometry("geometry")
    building_points.crs = buildings.crs

    # Spatial join: count buildings per cell
    joined = gpd.sjoin(building_points, grid, how="inner", predicate="within")

    # Determine the grid ID column
    if "grid_id" in grid.columns:
        id_col = "grid_id"
    elif "id" in grid.columns:
        id_col = "id"
    else:
        raise ValueError("Grid has no 'id' or 'grid_id' column")

    # Count per cell
    counts = joined.groupby(f"{id_col}_right" if f"{id_col}_right" in joined.columns else id_col).size()
    counts.name = "building_count"

    # Merge counts back to grid
    grid_with_counts = grid.copy()
    grid_with_counts["building_count"] = grid_with_counts[id_col].map(counts).fillna(0).astype(int)

    # Summary stats
    print(f"\nBuilding count summary:")
    print(f"  Total buildings matched: {grid_with_counts['building_count'].sum()}")
    print(f"  Cells with 0 buildings: {(grid_with_counts['building_count'] == 0).sum()}")
    print(f"  Cells with 1+ buildings: {(grid_with_counts['building_count'] > 0).sum()}")
    print(f"  Max buildings in a cell: {grid_with_counts['building_count'].max()}")
    print(f"  Mean buildings per cell: {grid_with_counts['building_count'].mean():.1f}")

    return grid_with_counts


def main():
    args = parse_args()

    # Load config
    config = load_config()
    data_dir = get_data_dir(config)

    # Output paths
    base_layers_dir = data_dir / "01_input_data" / "base_layers"
    base_layers_dir.mkdir(parents=True, exist_ok=True)

    buildings_path = base_layers_dir / "overture_buildings.parquet"

    print(f"Data directory: {data_dir}")
    print(f"Buildings output: {buildings_path}")

    # Load grid
    grid = load_grid(data_dir, args.grid_size, args.status, args.single)
    print(f"\nLoaded {len(grid)} grid cells ({args.grid_size}m)")

    # Download buildings if needed
    if not buildings_path.exists() or args.force:
        success = download_buildings(grid, buildings_path)
        if not success:
            sys.exit(1)
    else:
        print(f"\nBuildings file already exists: {buildings_path}")
        print("Use --force to re-download.")

    # Compute per-cell counts
    if not args.skip_counts and buildings_path.exists():
        grid_with_counts = compute_building_counts(buildings_path, grid, args.grid_size)

        # Save grid with counts
        grid_size_label = f"{args.grid_size}m" if args.grid_size != "5000" else "5km"
        counts_path = base_layers_dir / f"grid_{grid_size_label}_building_counts.gpkg"
        grid_with_counts.to_file(counts_path, driver="GPKG")
        print(f"\nSaved grid with building counts to: {counts_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
