"""Generate high-resolution maps for all control grid cells.

Usage:
    python scripts/generate_all_maps.py
    python scripts/generate_all_maps.py --no-basemap
    python scripts/generate_all_maps.py --status sampled
    python scripts/generate_all_maps.py --single 42
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.utils.config_loader import load_config, get_data_dir, get_output_dir
from src.data_processing.load_boundaries import (
    load_control_grid,
    load_subgrid,
    load_layer,
    validate_crs,
)
import contextily as cx

from src.mapping.map_generator import MapGenerator

BASEMAP_PROVIDERS = {
    "esri": cx.providers.Esri.WorldImagery,
    "osm": cx.providers.OpenStreetMap.Mapnik,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate high-resolution maps for all control grid cells."
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Disable online basemap tiles (useful offline).",
    )
    parser.add_argument(
        "--basemap",
        choices=["esri", "osm"],
        default="esri",
        help="Basemap tile provider: 'esri' (Esri WorldImagery) or 'osm' (OpenStreetMap). Default: esri.",
    )
    parser.add_argument(
        "--status",
        choices=["sampled", "replacement"],
        default=None,
        help="Generate maps only for cells with this sample_status.",
    )
    parser.add_argument(
        "--scalebar",
        action="store_true",
        help="Add a distance scale bar to each map.",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Generate map for a single grid cell ID only.",
    )
    parser.add_argument(
        "--subgrid",
        choices=["none", "500", "1000"],
        default="none",
        help="Overlay sub-grid cells: 'none' (default), '500' (500m), or '1000' (1km).",
    )
    return parser.parse_args()


def make_label(grid_id: str, sample_status: str) -> str:
    """Build a map label from grid ID and sample status."""
    return f"{grid_id} ({sample_status})"


def main():
    args = parse_args()

    # Load config
    config = load_config()
    data_dir = get_data_dir(config)
    output_dir = get_output_dir(config)

    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load control grid (auto-builds from source if needed)
    grid_cells = load_control_grid(data_dir)
    grid_cells = validate_crs(grid_cells)

    assert "id" in grid_cells.columns, (
        f"Expected 'id' column in control grid. Found: {list(grid_cells.columns)}"
    )
    assert "sample_status" in grid_cells.columns, (
        f"Expected 'sample_status' column in control grid. Found: {list(grid_cells.columns)}"
    )

    # Optional filter by sample_status
    if args.status:
        grid_cells = grid_cells[grid_cells["sample_status"] == args.status].copy()
        print(f"Filtered to {len(grid_cells)} cells with status '{args.status}'")

    # Load optional layers
    roads = load_layer(data_dir, "roads")
    buildings = load_layer(data_dir, "buildings")

    # Load sub-grid if requested
    subgrid = None
    if args.subgrid != "none":
        cell_size = int(args.subgrid)
        subgrid = load_subgrid(data_dir, cell_size=cell_size)
        print(f"Loaded {len(subgrid)} sub-grid cells ({cell_size}m)")

    # Initialize generator
    map_settings = config.get("map_settings", {})
    basemap_source = BASEMAP_PROVIDERS[args.basemap]
    # Build output subfolder name based on options
    maps_subfolder = "generated_maps"
    if args.basemap != "esri":
        maps_subfolder += f"_{args.basemap}"
    if args.subgrid != "none":
        maps_subfolder += f"_subgrid{args.subgrid}m"

    generator = MapGenerator(
        output_dir=output_dir / maps_subfolder,
        fig_width=map_settings.get("fig_width", 19.2),
        fig_height=map_settings.get("fig_height", 10.8),
        dpi=map_settings.get("dpi", 100),
        add_basemap=not args.no_basemap,
        basemap_source=basemap_source,
        add_scalebar=args.scalebar,
    )

    # Generate maps
    if args.single:
        mask = grid_cells["id"].astype(str) == str(args.single)
        cell = grid_cells[mask]
        if len(cell) == 0:
            print(f"Error: Grid cell '{args.single}' not found.")
            sys.exit(1)
        grid_id = str(args.single)
        label = make_label(grid_id, cell.iloc[0]["sample_status"])
        output_path = generator.generate_map(
            grid_cell=cell,
            grid_id=grid_id,
            label=label,
            all_grid_cells=grid_cells,
            subgrid=subgrid,
            roads=roads,
            buildings=buildings,
        )
        print(f"Generated: {output_path}")
    else:
        total = len(grid_cells)
        print(f"\nGenerating maps for {total} grid cells...")

        for idx, (_, row) in enumerate(tqdm(grid_cells.iterrows(), total=total)):
            grid_id = str(row["id"])
            label = make_label(grid_id, row["sample_status"])

            cell = grid_cells.iloc[[idx]]
            generator.generate_map(
                grid_cell=cell,
                grid_id=grid_id,
                label=label,
                all_grid_cells=grid_cells,
                subgrid=subgrid,
                roads=roads,
                buildings=buildings,
            )

        print(f"\nDone! {total} maps saved to {generator.output_dir}")


if __name__ == "__main__":
    main()
