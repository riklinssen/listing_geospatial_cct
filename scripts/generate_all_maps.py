"""Generate high-resolution maps for all control grid cells.

Usage:
    python scripts/generate_all_maps.py
    python scripts/generate_all_maps.py --grid-id-col CELL_ID
    python scripts/generate_all_maps.py --no-basemap
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
    load_layer,
    validate_crs,
)
from src.mapping.map_generator import MapGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate high-resolution maps for all control grid cells."
    )
    parser.add_argument(
        "--grid-id-col",
        default=None,
        help="Column name containing grid cell IDs. If not set, uses the first "
             "column or falls back to row index.",
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Disable online basemap tiles (useful offline).",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Generate map for a single grid cell ID only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config()
    data_dir = get_data_dir(config)
    output_dir = get_output_dir(config)

    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load grid cells
    grid_cells = load_control_grid(data_dir)
    grid_cells = validate_crs(grid_cells)

    # Determine the grid ID column
    grid_id_col = args.grid_id_col
    if grid_id_col is None:
        # Try common column names
        for candidate in ["CELL_ID", "cell_id", "grid_id", "GRID_ID", "ID", "id", "NAME", "name"]:
            if candidate in grid_cells.columns:
                grid_id_col = candidate
                break

    if grid_id_col and grid_id_col not in grid_cells.columns:
        print(f"Error: Column '{grid_id_col}' not found in grid data.")
        print(f"Available columns: {list(grid_cells.columns)}")
        sys.exit(1)

    # Load optional layers
    roads = load_layer(data_dir, "roads")
    buildings = load_layer(data_dir, "buildings")

    # Initialize generator
    map_settings = config.get("map_settings", {})
    generator = MapGenerator(
        output_dir=output_dir / "generated_maps",
        fig_width=map_settings.get("fig_width", 19.2),
        fig_height=map_settings.get("fig_height", 10.8),
        dpi=map_settings.get("dpi", 100),
        add_basemap=not args.no_basemap,
    )

    # Generate maps
    if args.single:
        # Single grid cell mode
        if grid_id_col:
            mask = grid_cells[grid_id_col].astype(str) == str(args.single)
            cell = grid_cells[mask]
        else:
            cell = grid_cells.iloc[[int(args.single)]]
        if len(cell) == 0:
            print(f"Error: Grid cell '{args.single}' not found.")
            sys.exit(1)
        grid_id = str(args.single)
        output_path = generator.generate_map(
            grid_cell=cell,
            grid_id=grid_id,
            all_grid_cells=grid_cells,
            roads=roads,
            buildings=buildings,
        )
        print(f"Generated: {output_path}")
    else:
        # Batch mode — all grid cells
        total = len(grid_cells)
        print(f"\nGenerating maps for {total} grid cells...")

        for idx, (_, row) in enumerate(tqdm(grid_cells.iterrows(), total=total)):
            if grid_id_col:
                grid_id = str(row[grid_id_col])
            else:
                grid_id = str(idx + 1).zfill(3)

            cell = grid_cells.iloc[[idx]]
            output_path = generator.generate_map(
                grid_cell=cell,
                grid_id=grid_id,
                all_grid_cells=grid_cells,
                roads=roads,
                buildings=buildings,
            )

        print(f"\nDone! {total} maps saved to {generator.output_dir}")


if __name__ == "__main__":
    main()
