"""Build and cache the nearest road-access point for every selected 500m sub-cell.

Companion to scripts/build_landmarks.py. For each sub-cell it finds the nearest
point on a (preferably major) road that a vehicle can reach, plus the off-road
walk from that point to the cell centroid. Writes a long-format CSV keyed by
grid_id so it can be merged into the landmark guide later.

Resumable: sub-cells already cached are skipped; re-run to fill any 5km cells
whose Overpass query timed out. Use --force to redo everything.

⚠ Uses the same public Overpass servers as build_landmarks.py — run them one at
a time, not concurrently, or they rate-limit each other.

Writes:
    <data_dir>/01_input_data/boundaries/subcell_road_snaps.csv

Usage:
    python scripts/build_road_snaps.py
    python scripts/build_road_snaps.py --single G_0588_0987
    python scripts/build_road_snaps.py --limit 5
    python scripts/build_road_snaps.py --margin 2000   # widen road search (m)
    python scripts/build_road_snaps.py --force
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_data_dir
from src.data_processing.load_boundaries import load_selected_subcells, validate_crs
from src.data_processing.road_snap import build_subcell_road_snaps


def parse_args():
    p = argparse.ArgumentParser(description="Build the per-sub-cell road-snap cache.")
    p.add_argument("--single", default=None,
                   help="Only process one sub-cell by its grid_id (e.g. G_0588_0987).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N sub-cells (for testing).")
    p.add_argument("--force", action="store_true",
                   help="Re-query every sub-cell, ignoring the existing cache.")
    p.add_argument("--margin", type=int, default=1500,
                   help="Metres to pad the 5km cell bbox when searching for roads (default 1500).")
    p.add_argument("--pause", type=float, default=0.5,
                   help="Seconds to sleep between 5km cells (Overpass politeness).")
    p.add_argument("--output", default=None,
                   help="Override the cache CSV path.")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config()
    data_dir = get_data_dir(config)

    cache_path = (Path(args.output) if args.output else
                  data_dir / "01_input_data" / "boundaries" / "subcell_road_snaps.csv")

    selected = load_selected_subcells(data_dir)
    if selected is None:
        print("Error: No selected sub-cells found. Run build_data.py first.")
        sys.exit(1)
    selected = validate_crs(selected, expected_epsg=4326)
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

    build_subcell_road_snaps(
        selected, cache_path,
        force=args.force, margin_m=args.margin, pause=args.pause,
    )


if __name__ == "__main__":
    main()
