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
    args = parser.parse_args()

    # Shared args for map/info scripts
    shared = []
    if args.output_dir:
        shared += ["--output-dir", args.output_dir]
    if args.limit:
        shared += ["--limit", str(args.limit)]
    if args.status:
        shared += ["--status", args.status]

    # Step 1: Build data
    if not args.skip_build:
        run("build_data.py", ["--force"], "Step 1/3: Building data (grids + PPS selection)")

    # Step 2: Generate maps
    if not args.skip_maps:
        run("generate_all_maps.py", shared, "Step 2/3: Generating maps + GeoJSON")

    # Step 3: Generate info sheets
    if not args.skip_info:
        run("generate_info_sheets.py", shared, "Step 3/3: Generating info sheets")

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
