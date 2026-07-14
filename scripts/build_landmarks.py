"""Build and cache the per-sub-cell landmark table (OSM via Overpass).

For every selected 500m sub-cell, finds recognisable landmarks (villages,
schools, churches, dispensaries, markets, water points) and ranks them
nearest-first with a compass bearing. Writes a long-format CSV that the info
sheets read to render a "How to find this cell" section — no network needed at
info-sheet time.

The cache is **resumable**: sub-cells already present are skipped, so if the
public Overpass servers time out mid-run (they often do), just run it again to
fill in the rest. Use --force to re-query everything.

Writes:
    <data_dir>/01_input_data/boundaries/subcell_landmarks.csv

Usage:
    python scripts/build_landmarks.py
    python scripts/build_landmarks.py --single G_0588_0987
    python scripts/build_landmarks.py --limit 5
    python scripts/build_landmarks.py --force
    python scripts/build_landmarks.py --max-landmarks 10 --pause 1.5
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_data_dir
from src.data_processing.load_boundaries import load_selected_subcells, validate_crs
from src.data_processing.landmarks import build_subcell_landmarks


def parse_args():
    p = argparse.ArgumentParser(description="Build the per-sub-cell OSM landmark cache.")
    p.add_argument("--single", default=None,
                   help="Only process one sub-cell by its grid_id (e.g. G_0588_0987).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N sub-cells (for testing).")
    p.add_argument("--force", action="store_true",
                   help="Re-query every sub-cell, ignoring the existing cache.")
    p.add_argument("--max-landmarks", type=int, default=8,
                   help="Max landmarks kept per sub-cell (default 8).")
    p.add_argument("--pause", type=float, default=1.0,
                   help="Seconds to sleep between sub-cells (Overpass politeness).")
    p.add_argument("--output", default=None,
                   help="Override the cache CSV path.")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config()
    data_dir = get_data_dir(config)

    cache_path = (Path(args.output) if args.output else
                  data_dir / "01_input_data" / "boundaries" / "subcell_landmarks.csv")

    selected = load_selected_subcells(data_dir)
    if selected is None:
        print("Error: No selected sub-cells found. Run build_data.py first.")
        sys.exit(1)
    selected = validate_crs(selected, expected_epsg=4326)  # ensure lat/lon usable
    selected = selected[selected["sample_status"] != "dropped_sparse"].copy()

    if args.single:
        selected = selected[selected["grid_id"].astype(str) == args.single].copy()
        if len(selected) == 0:
            print(f"Error: sub-cell '{args.single}' not found.")
            sys.exit(1)
    if args.limit:
        selected = selected.head(args.limit).copy()

    print(f"Data directory: {data_dir}")
    print(f"Cache path:     {cache_path}")
    print(f"Sub-cells:      {len(selected)}\n")

    build_subcell_landmarks(
        selected, cache_path,
        force=args.force,
        max_landmarks=args.max_landmarks,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
