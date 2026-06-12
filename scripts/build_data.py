"""Build processed data files from raw inputs.

Loads the 5km study area grid and control sampling flags,
merges them, and saves both the full flagged grid and the
filtered control grid to the boundaries folder.

Also generates 500m and 1km sub-grids from point centroids,
clipped to the parent 5km cell boundaries.

Usage:
    python scripts/build_data.py
    python scripts/build_data.py --force
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_data_dir
from src.data_processing.load_boundaries import (
    build_control_grid,
    save_control_grid,
    load_subgrid_points,
    build_subgrid,
    save_subgrid,
    load_control_grid,
    select_subcells,
    save_selected_subcells,
    load_vcsl_flags,
    flag_vcsl_villages,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build processed data files from raw inputs."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if processed files already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config()
    data_dir = get_data_dir(config)
    boundaries_dir = data_dir / "01_input_data" / "boundaries"

    print(f"Data directory: {data_dir}")

    # Output paths
    control_grid_path = boundaries_dir / "control_grid_5km_flagged.gpkg"
    full_grid_path = boundaries_dir / "study_area_5km_flagged.gpkg"
    subgrid_500m_path = boundaries_dir / "subgrid_500m_control.gpkg"
    subgrid_1km_path = boundaries_dir / "subgrid_1000m_control.gpkg"
    selected_path = boundaries_dir / "selected_subcells_500m.gpkg"
    counts_path = data_dir / "02_outputs" / "building_counts" / "grid_500m_building_counts.gpkg"

    # CCT VCSL village flags (5km-pixel level); path from config, falls back to
    # a copy in the boundaries folder if the configured path is unset/missing.
    vcsl_flags_path = Path(
        config["paths"].get("vcsl_flags")
        or (boundaries_dir / "Sampled_control_pixels.gpkg")
    )

    all_exist = (
        control_grid_path.exists()
        and full_grid_path.exists()
        and subgrid_500m_path.exists()
        and subgrid_1km_path.exists()
        and selected_path.exists()
    )

    if not args.force and all_exist:
        print("Processed files already exist:")
        print(f"  {control_grid_path}")
        print(f"  {full_grid_path}")
        print(f"  {subgrid_500m_path}")
        print(f"  {subgrid_1km_path}")
        print("Use --force to rebuild.")
        return

    # Build 5km control grid (also saves the full study grid)
    print("\n--- Building 5km control grid ---")
    control_grid = build_control_grid(data_dir)
    save_control_grid(control_grid, data_dir)

    # Build sub-grids from 500m point centroids
    print("\n--- Building sub-grids ---")
    subgrid_points = load_subgrid_points(data_dir)

    subgrid_500m = build_subgrid(subgrid_points, control_grid, cell_size=500)
    save_subgrid(subgrid_500m, data_dir, cell_size=500)

    subgrid_1km = build_subgrid(subgrid_points, control_grid, cell_size=1000)
    save_subgrid(subgrid_1km, data_dir, cell_size=1000)

    # Select sub-cells (requires building counts from download_google_buildings.py)
    print("\n--- Selecting 500m sub-cells ---")
    if not counts_path.exists():
        print(f"  Skipping: building counts not found at {counts_path}")
        print("  Run first: python scripts/download_google_buildings.py --grid-size 500")
        selected = None
    else:
        grid_500m_counts = gpd.read_file(counts_path)
        print(f"  Loaded {len(grid_500m_counts)} sub-cells with building counts")
        selected, updated_grid = select_subcells(control_grid, grid_500m_counts)

        # Attach CCT VCSL village flags via attribute merge on the 5km id
        print("\n--- Flagging VCSL villages ---")
        if vcsl_flags_path.exists():
            vcsl_flags = load_vcsl_flags(vcsl_flags_path)
            selected = flag_vcsl_villages(selected, vcsl_flags, id_col="5km_id")
            updated_grid = flag_vcsl_villages(updated_grid, vcsl_flags, id_col="id")
        else:
            print(f"  VCSL flags file not found at {vcsl_flags_path} — skipping.")

        save_selected_subcells(selected, data_dir)
        # Save updated 5km grid (with replacement activations + VCSL flags)
        save_control_grid(updated_grid, data_dir)
        control_grid = updated_grid

    # Summary
    print("\n" + "=" * 60)
    print("Build complete:")
    print("=" * 60)
    print(f"  Full study grid:   {full_grid_path}")
    print(f"  Control grid:      {control_grid_path}")
    print(f"  Sub-grid (500m):   {subgrid_500m_path}")
    print(f"  Sub-grid (1km):    {subgrid_1km_path}")
    print(f"  Selected sub-cells: {selected_path}")
    print()
    print(f"  Total 5km controls: {len(control_grid)}")
    print(f"    Sampled:          {(control_grid['sample_status'] == 'sampled').sum()}")
    print(f"    Replacement:      {(control_grid['sample_status'] == 'replacement').sum()}")
    print(f"    Dropped (sparse): {(control_grid['sample_status'] == 'dropped_sparse').sum()}")
    print(f"  Sub-cells (500m):   {len(subgrid_500m)}")
    print(f"  Sub-cells (1km):    {len(subgrid_1km)}")
    if selected is not None:
        print(f"  Selected sub-cells: {len(selected)} "
              f"({(selected['selection_role'] == 'primary').sum()} primary, "
              f"{(selected['selection_role'] == 'reserve').sum()} reserve)")


if __name__ == "__main__":
    main()
