"""Download building footprints for enumeration areas.

Uses the Overture Maps CLI to fetch building footprints for the study area.
The Overture Maps dataset includes Google Open Buildings data.

Usage:
    python scripts/download_google_buildings.py
    python scripts/download_google_buildings.py --grid-size 500
    python scripts/download_google_buildings.py --grid-size 1000
    python scripts/download_google_buildings.py --status sampled
    python scripts/download_google_buildings.py --single 42
    python scripts/download_google_buildings.py --skip-counts
    python scripts/download_google_buildings.py --visualize
    python scripts/download_google_buildings.py --visualize --n-samples 6

Prerequisites:
    pip install overturemaps matplotlib
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
import matplotlib.pyplot as plt
import numpy as np

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

# Google Maps tile providers
GOOGLE_SATELLITE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
GOOGLE_HYBRID = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"  # satellite + labels
GOOGLE_ROADS = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"

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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of building counts.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of sample cells to visualize (default: 4).",
    )
    parser.add_argument(
        "--generate-tiles",
        action="store_true",
        help="Generate tile maps with Google basemap for each grid cell.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Maximum number of tile maps to generate (default: all inhabited cells).",
    )
    parser.add_argument(
        "--basemap",
        choices=["satellite", "hybrid", "roads"],
        default="hybrid",
        help="Basemap type for tile maps (default: hybrid).",
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


def unpack_buildings_columns(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Unpack nested columns in the buildings dataframe.

    Handles:
    - sources: list of dicts with dataset, confidence, record_id
    - names: dict with language keys (e.g., {'primary': 'Building Name'})
    """
    def extract_source_field(sources, field, default=None):
        """Extract a field from the first source entry."""
        if sources is None or len(sources) == 0:
            return default
        first_source = sources[0]
        if isinstance(first_source, dict):
            return first_source.get(field, default)
        return default

    def extract_primary_name(names, default=None):
        """Extract primary name from names dict."""
        if names is None:
            return default
        if isinstance(names, dict):
            return names.get("primary", names.get("common", default))
        return default

    # Unpack sources column
    if "sources" in buildings.columns:
        print("  Unpacking sources column...")
        buildings["source_dataset"] = buildings["sources"].apply(
            lambda x: extract_source_field(x, "dataset")
        )
        buildings["source_confidence"] = buildings["sources"].apply(
            lambda x: extract_source_field(x, "confidence")
        )
        buildings["source_record_id"] = buildings["sources"].apply(
            lambda x: extract_source_field(x, "record_id")
        )
        buildings = buildings.drop(columns=["sources"])

        # Print summary
        dataset_counts = buildings["source_dataset"].value_counts()
        print(f"    Datasets: {dict(dataset_counts)}")
        conf = buildings["source_confidence"].dropna()
        if len(conf) > 0:
            print(f"    Confidence: min={conf.min():.3f}, mean={conf.mean():.3f}, max={conf.max():.3f}")

    # Unpack names column
    if "names" in buildings.columns:
        print("  Unpacking names column...")
        buildings["building_name"] = buildings["names"].apply(extract_primary_name)
        buildings = buildings.drop(columns=["names"])
        named_count = buildings["building_name"].notna().sum()
        print(f"    Buildings with names: {named_count:,}")

    return buildings


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

    # Unpack nested columns (sources, names)
    buildings = unpack_buildings_columns(buildings)

    # Ensure same CRS
    if buildings.crs != grid.crs:
        buildings = buildings.to_crs(grid.crs)

    # Get centroids for faster spatial join
    buildings["centroid"] = buildings.geometry.centroid
    building_points = buildings.set_geometry("centroid")[["centroid"]].copy()
    building_points = building_points.rename(columns={"centroid": "geometry"}).set_geometry("geometry")
    building_points.crs = buildings.crs

    # Keep confidence for aggregation
    if "source_confidence" in buildings.columns:
        building_points["source_confidence"] = buildings["source_confidence"].values

    # Spatial join: count buildings per cell
    joined = gpd.sjoin(building_points, grid, how="inner", predicate="within")

    # Determine the grid ID column
    if "grid_id" in grid.columns:
        id_col = "grid_id"
    elif "id" in grid.columns:
        id_col = "id"
    else:
        raise ValueError("Grid has no 'id' or 'grid_id' column")

    # Get the join column name
    join_col = f"{id_col}_right" if f"{id_col}_right" in joined.columns else id_col

    # Count per cell
    counts = joined.groupby(join_col).size()
    counts.name = "building_count"

    # Mean confidence per cell
    if "source_confidence" in joined.columns:
        mean_conf = joined.groupby(join_col)["source_confidence"].mean()
        mean_conf.name = "mean_confidence"
    else:
        mean_conf = None

    # Merge counts back to grid
    grid_with_counts = grid.copy()
    grid_with_counts["building_count"] = grid_with_counts[id_col].map(counts).fillna(0).astype(int)
    if mean_conf is not None:
        grid_with_counts["mean_confidence"] = grid_with_counts[id_col].map(mean_conf)

    # Summary stats
    print(f"\nBuilding count summary:")
    print(f"  Total buildings matched: {grid_with_counts['building_count'].sum():,}")
    print(f"  Cells with 0 buildings: {(grid_with_counts['building_count'] == 0).sum():,}")
    print(f"  Cells with 1+ buildings: {(grid_with_counts['building_count'] > 0).sum():,}")
    print(f"  Max buildings in a cell: {grid_with_counts['building_count'].max():,}")
    print(f"  Mean buildings per cell: {grid_with_counts['building_count'].mean():.1f}")
    if "mean_confidence" in grid_with_counts.columns:
        conf = grid_with_counts["mean_confidence"].dropna()
        if len(conf) > 0:
            print(f"  Mean confidence (per cell): min={conf.min():.3f}, mean={conf.mean():.3f}, max={conf.max():.3f}")

    return grid_with_counts


def visualize_building_counts(
    grid_with_counts: gpd.GeoDataFrame,
    buildings_path: Path,
    output_dir: Path,
    grid_size: str,
    n_sample_cells: int = 4,
) -> None:
    """Create visualizations of building counts per grid cell."""
    grid_size_label = f"{grid_size}m" if grid_size != "5000" else "5km"

    # Determine the grid ID column
    if "grid_id" in grid_with_counts.columns:
        id_col = "grid_id"
    elif "id" in grid_with_counts.columns:
        id_col = "id"
    else:
        id_col = grid_with_counts.columns[0]

    # 1. Choropleth map of building counts
    print(f"\nCreating choropleth map of building counts...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Use log scale for better visualization (add 1 to avoid log(0))
    grid_with_counts["log_count"] = np.log10(grid_with_counts["building_count"] + 1)

    grid_with_counts.plot(
        column="log_count",
        cmap="YlOrRd",
        legend=True,
        legend_kwds={"label": "Building Count (log10 scale)", "shrink": 0.6},
        edgecolor="black",
        linewidth=0.2,
        ax=ax,
    )
    ax.set_title(f"Building Counts per {grid_size_label} Grid Cell", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    choropleth_path = output_dir / f"choropleth_{grid_size_label}_building_counts.png"
    fig.savefig(choropleth_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {choropleth_path}")

    # 2. Sample cell visualizations with building footprints
    print(f"\nCreating sample cell visualizations with building footprints...")

    # Select sample cells: top N by building count (most interesting)
    top_cells = grid_with_counts.nlargest(n_sample_cells, "building_count")

    # Load buildings for sample visualization
    buildings = gpd.read_parquet(buildings_path)
    if buildings.crs != grid_with_counts.crs:
        buildings = buildings.to_crs(grid_with_counts.crs)

    for _, cell in top_cells.iterrows():
        cell_id = cell[id_col]
        cell_geom = cell.geometry
        building_count = cell["building_count"]

        # Clip buildings to cell
        cell_buildings = buildings[buildings.geometry.intersects(cell_geom)]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot cell boundary
        gpd.GeoSeries([cell_geom], crs=grid_with_counts.crs).plot(
            ax=ax, facecolor="none", edgecolor="red", linewidth=2
        )

        # Plot buildings
        if len(cell_buildings) > 0:
            cell_buildings.plot(ax=ax, facecolor="steelblue", edgecolor="navy", linewidth=0.3, alpha=0.7)

        ax.set_title(f"Cell {cell_id}: {building_count:,} buildings", fontsize=14)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

        # Set bounds to cell with small buffer
        minx, miny, maxx, maxy = cell_geom.bounds
        buffer = (maxx - minx) * 0.05
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        cell_path = output_dir / f"sample_cell_{grid_size_label}_{cell_id}.png"
        fig.savefig(cell_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {cell_path}")

    print(f"\nVisualization complete. {n_sample_cells + 1} images saved to {output_dir}")


def generate_tile_maps(
    grid: gpd.GeoDataFrame,
    buildings_path: Path,
    output_dir: Path,
    grid_size: str,
    basemap: str = "hybrid",
    max_tiles: int | None = None,
) -> None:
    """Generate a map with Google basemap and building overlay for each grid cell.

    Args:
        grid: GeoDataFrame with grid cells (must have building_count column)
        buildings_path: Path to buildings parquet file
        output_dir: Directory to save maps
        grid_size: Grid size label (e.g., "500", "1000", "5000")
        basemap: "satellite", "hybrid", or "roads"
        max_tiles: Maximum number of tiles to generate (None = all)
    """
    if not HAS_CONTEXTILY:
        print("Error: contextily not installed. Run: pip install contextily")
        return

    # Select basemap
    basemap_urls = {
        "satellite": GOOGLE_SATELLITE,
        "hybrid": GOOGLE_HYBRID,
        "roads": GOOGLE_ROADS,
    }
    tile_url = basemap_urls.get(basemap, GOOGLE_HYBRID)

    # Determine ID column
    if "grid_id" in grid.columns:
        id_col = "grid_id"
    elif "id" in grid.columns:
        id_col = "id"
    else:
        id_col = grid.columns[0]

    # Create output subdirectory
    grid_size_label = f"{grid_size}m" if grid_size != "5000" else "5km"
    maps_dir = output_dir / f"tile_maps_{grid_size_label}"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Load buildings
    print(f"\nLoading buildings...")
    buildings = gpd.read_parquet(buildings_path)
    buildings = unpack_buildings_columns(buildings)

    # Filter to inhabited cells only
    inhabited = grid[grid["building_count"] > 0].copy()
    if max_tiles:
        inhabited = inhabited.head(max_tiles)

    print(f"\nGenerating {len(inhabited)} tile maps to {maps_dir}...")

    for i, (_, cell) in enumerate(inhabited.iterrows()):
        cell_id = cell[id_col]
        cell_geom = cell.geometry
        building_count = cell.get("building_count", 0)

        # Get buildings in this cell
        # Convert to same CRS for intersection
        if buildings.crs != grid.crs:
            buildings_proj = buildings.to_crs(grid.crs)
        else:
            buildings_proj = buildings

        cell_buildings = buildings_proj[buildings_proj.geometry.intersects(cell_geom)]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Convert to Web Mercator for contextily
        cell_gdf = gpd.GeoDataFrame(geometry=[cell_geom], crs=grid.crs).to_crs(epsg=3857)
        cell_buildings_wm = cell_buildings.to_crs(epsg=3857) if len(cell_buildings) > 0 else None

        # Plot cell boundary
        cell_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=3)

        # Plot buildings
        if cell_buildings_wm is not None and len(cell_buildings_wm) > 0:
            cell_buildings_wm.plot(
                ax=ax,
                facecolor="yellow",
                edgecolor="orange",
                linewidth=0.5,
                alpha=0.6,
            )

        # Set bounds with buffer
        minx, miny, maxx, maxy = cell_gdf.total_bounds
        buffer = (maxx - minx) * 0.1
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        # Add basemap
        try:
            cx.add_basemap(ax, source=tile_url, zoom="auto")
        except Exception as e:
            print(f"  Warning: Could not add basemap for cell {cell_id}: {e}")

        # Title and labels
        ax.set_title(f"Cell {cell_id} | {building_count:,} buildings", fontsize=14, fontweight="bold")
        ax.set_axis_off()

        # Save
        out_path = maps_dir / f"tile_{cell_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if (i + 1) % 10 == 0 or (i + 1) == len(inhabited):
            print(f"  Generated {i + 1}/{len(inhabited)} maps")

    print(f"\nDone! Maps saved to {maps_dir}")


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

        # Save grid with counts to outputs folder
        outputs_dir = data_dir / "02_outputs" / "building_counts"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        grid_size_label = f"{args.grid_size}m" if args.grid_size != "5000" else "5km"
        counts_path = outputs_dir / f"grid_{grid_size_label}_building_counts.gpkg"
        grid_with_counts.to_file(counts_path, driver="GPKG")
        print(f"\nSaved grid with building counts to: {counts_path}")

        # Generate visualizations if requested
        if args.visualize:
            visualize_building_counts(
                grid_with_counts,
                buildings_path,
                outputs_dir,
                args.grid_size,
                n_sample_cells=args.n_samples,
            )

        # Generate tile maps if requested
        if args.generate_tiles:
            generate_tile_maps(
                grid_with_counts,
                buildings_path,
                outputs_dir,
                args.grid_size,
                basemap=args.basemap,
                max_tiles=args.max_tiles,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
