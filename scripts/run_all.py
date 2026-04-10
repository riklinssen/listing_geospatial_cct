"""Run the full pipeline: build data, generate maps, generate info sheets.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --limit 3
    python scripts/run_all.py --output-dir "G:/path/to/output"
    python scripts/run_all.py --skip-build
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(script: str, args: list[str], description: str):
    """Run a script and exit on failure."""
    cmd = [PYTHON, str(SCRIPTS_DIR / script)] + args
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFailed: {script} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full listing maps pipeline.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N cells (for testing).")
    parser.add_argument("--status", choices=["sampled", "replacement"], default=None)
    parser.add_argument("--skip-build", action="store_true", help="Skip data build step.")
    parser.add_argument("--skip-maps", action="store_true", help="Skip map generation.")
    parser.add_argument("--skip-info", action="store_true", help="Skip info sheet generation.")
    parser.add_argument(
        "--mbtiles", action="store_true",
        help="Also emit .mbtiles alongside PNGs (for SurveyCTO offline basemaps).",
    )
    parser.add_argument(
        "--mbtiles-only", action="store_true",
        help="Emit only MBTiles, skip PNG generation. Implies --mbtiles.",
    )
    parser.add_argument("--mbtiles-detail-zoom", type=int, default=None)
    parser.add_argument("--mbtiles-overview-zoom", type=int, default=None)
    args = parser.parse_args()

    # Shared args for map/info scripts
    shared = []
    if args.output_dir:
        shared += ["--output-dir", args.output_dir]
    if args.limit:
        shared += ["--limit", str(args.limit)]
    if args.status:
        shared += ["--status", args.status]

    # Map-only args (MBTiles flags don't apply to info sheets)
    map_args = list(shared)
    if args.mbtiles:
        map_args.append("--mbtiles")
    if args.mbtiles_only:
        map_args.append("--mbtiles-only")
    if args.mbtiles_detail_zoom is not None:
        map_args += ["--mbtiles-detail-zoom", str(args.mbtiles_detail_zoom)]
    if args.mbtiles_overview_zoom is not None:
        map_args += ["--mbtiles-overview-zoom", str(args.mbtiles_overview_zoom)]

    # Step 1: Build data
    if not args.skip_build:
        run("build_data.py", ["--force"], "Step 1/3: Building data (grids + PPS selection)")

    # Step 2: Generate maps
    if not args.skip_maps:
        run("generate_all_maps.py", map_args, "Step 2/3: Generating maps + GeoJSON")

    # Step 3: Generate info sheets
    if not args.skip_info:
        run("generate_info_sheets.py", shared, "Step 3/3: Generating info sheets")

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
