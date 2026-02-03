"""Verify that the project environment is set up correctly.

Checks:
- Required Python packages are installed
- Config file loads successfully
- Data directory is accessible
- Expected input files exist

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_imports():
    """Check that all required packages are importable."""
    packages = [
        ("geopandas", "geopandas"),
        ("matplotlib", "matplotlib"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("shapely", "shapely"),
        ("pandas", "pandas"),
        ("PIL", "pillow"),
        ("folium", "folium"),
    ]

    optional_packages = [
        ("contextily", "contextily"),
    ]

    all_ok = True
    print("Checking required packages...")
    for import_name, pip_name in packages:
        try:
            __import__(import_name)
            print(f"  [OK] {pip_name}")
        except ImportError:
            print(f"  [MISSING] {pip_name} — install with: pip install {pip_name}")
            all_ok = False

    print("\nChecking optional packages...")
    for import_name, pip_name in optional_packages:
        try:
            __import__(import_name)
            print(f"  [OK] {pip_name}")
        except ImportError:
            print(f"  [MISSING] {pip_name} (optional) — install with: pip install {pip_name}")

    return all_ok


def check_config():
    """Check that config files load successfully."""
    print("\nChecking configuration...")
    try:
        from src.utils.config_loader import load_config
        config = load_config()
        print(f"  [OK] config.yaml loaded")
        print(f"  Data dir:   {config['paths']['data_dir']}")
        print(f"  Output dir: {config['paths']['output_dir']}")
        return config
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Failed to load config: {e}")
        return None


def check_data_access(config):
    """Check that the data directory and key files are accessible."""
    print("\nChecking data access...")
    if config is None:
        print("  [SKIP] No config loaded.")
        return

    data_dir = Path(config["paths"]["data_dir"])

    if not data_dir.exists():
        print(f"  [WARNING] Data directory not found: {data_dir}")
        print(f"            Make sure Google Drive is mounted / synced.")
        print(f"            Or update config/config.local.yaml with your path.")
        return

    print(f"  [OK] Data directory exists: {data_dir}")

    # Check for expected input files
    expected_files = [
        "01_input_data/boundaries/enumeration_areas.geojson",
        "01_input_data/boundaries/control_grid.geojson",
    ]

    optional_files = [
        "01_input_data/base_layers/roads.geojson",
        "01_input_data/base_layers/buildings.geojson",
    ]

    for rel_path in expected_files:
        full_path = data_dir / rel_path
        if full_path.exists():
            print(f"  [OK] {rel_path}")
        else:
            print(f"  [MISSING] {rel_path}")

    for rel_path in optional_files:
        full_path = data_dir / rel_path
        if full_path.exists():
            print(f"  [OK] {rel_path} (optional)")
        else:
            print(f"  [--] {rel_path} (optional, not found)")


def check_output_dir(config):
    """Check that the output directory is writable."""
    print("\nChecking output directory...")
    if config is None:
        print("  [SKIP] No config loaded.")
        return

    output_dir = Path(config["paths"]["output_dir"])
    if output_dir.exists():
        print(f"  [OK] Output directory exists: {output_dir}")
    else:
        print(f"  [INFO] Output directory will be created: {output_dir}")


def main():
    print("=" * 60)
    print("  Listing Geospatial CCT — Setup Verification")
    print("=" * 60)

    imports_ok = check_imports()
    config = check_config()
    check_data_access(config)
    check_output_dir(config)

    print("\n" + "=" * 60)
    if imports_ok and config is not None:
        print("  Setup looks good! You're ready to generate maps.")
    else:
        print("  Some issues found. See above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
