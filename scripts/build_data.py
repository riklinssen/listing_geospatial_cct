"""Build processed data files from raw inputs.

Loads the 5km study area grid and control sampling flags,
merges them, and saves both the full flagged grid and the
filtered control grid to the boundaries folder.

Usage:
    python scripts/build_data.py
    python scripts/build_data.py --force
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_data_dir
from src.data_processing.load_boundaries import (
    build_control_grid,
    save_control_grid,
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

    control_grid_path = boundaries_dir / "control_grid_5km_flagged.gpkg"
    full_grid_path = boundaries_dir / "study_area_5km_flagged.gpkg"

    if not args.force and control_grid_path.exists() and full_grid_path.exists():
        print(f"Processed files already exist:")
        print(f"  {control_grid_path}")
        print(f"  {full_grid_path}")
        print("Use --force to rebuild.")
        return

    # build_control_grid internally saves the full study grid
    control_grid = build_control_grid(data_dir)
    save_control_grid(control_grid, data_dir)

    print("\nBuild complete:")
    print(f"  Full study grid:  {full_grid_path}")
    print(f"  Control grid:     {control_grid_path}")
    print(f"  Total controls:   {len(control_grid)}")
    print(f"    Sampled:        {(control_grid['sample_status'] == 'sampled').sum()}")
    print(f"    Replacement:    {(control_grid['sample_status'] == 'replacement').sum()}")


if __name__ == "__main__":
    main()
