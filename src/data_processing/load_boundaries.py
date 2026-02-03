"""Load and validate geospatial boundary files from Google Drive.

Reads the 5km study area grid and control area sample flags,
merges them into a single GeoDataFrame with a sample_status column,
and provides loaders for optional enhancement layers (roads, buildings).
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_study_area_grid(data_dir: Path) -> gpd.GeoDataFrame:
    """Load the 5km study area grid.

    Args:
        data_dir: Root data directory (e.g. Google Drive 0.4_listing_geospatial/).

    Returns:
        GeoDataFrame with 5km grid cell polygons.

    Raises:
        FileNotFoundError: If the grid file does not exist.
    """
    filepath = data_dir / "01_input_data" / "boundaries" / "Area_study_5km_grid.gpkg"
    if not filepath.exists():
        raise FileNotFoundError(f"5km grid file not found: {filepath}")
    gdf = gpd.read_file(filepath)
    print(f"Loaded {len(gdf)} grid cells from {filepath}")
    return gdf


def load_control_sample_flags(data_dir: Path) -> pd.DataFrame:
    """Load the control area sampling flags CSV.

    Args:
        data_dir: Root data directory.

    Returns:
        DataFrame with grid_id and control_pixel_sampled columns.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    filepath = data_dir / "01_input_data" / "boundaries" / "Rubeho_control_areas_sampled.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Control sample flags not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} control sample records from {filepath}")
    return df


def build_control_grid(data_dir: Path) -> gpd.GeoDataFrame:
    """Build the control grid by merging the 5km grid with sample flags.

    Loads the study area grid and control sampling CSV, then adds a
    'sample_status' column: "sampled" (control_pixel_sampled==1),
    "replacement" (control_pixel_sampled==0), or NaN for non-control cells.

    Only grid cells with sample_status "sampled" or "replacement" are
    returned (i.e. non-control cells are dropped).

    Args:
        data_dir: Root data directory.

    Returns:
        GeoDataFrame with 5km control grid cells and sample_status column.
    """
    grid = load_study_area_grid(data_dir)
    control_sampled = load_control_sample_flags(data_dir)

    # Filter to actual control areas and map flags to labels
    control_ids = control_sampled[control_sampled["control_pixel_sampled"].isin([0, 1])]
    id_to_status = control_ids.set_index("grid_id")["control_pixel_sampled"].map(
        {1: "sampled", 0: "replacement"}
    )

    grid["sample_status"] = grid["id"].map(id_to_status)

    # Verify counts match source data
    expected_sampled = (control_sampled["control_pixel_sampled"] == 1).sum()
    expected_replacement = (control_sampled["control_pixel_sampled"] == 0).sum()
    actual_sampled = (grid["sample_status"] == "sampled").sum()
    actual_replacement = (grid["sample_status"] == "replacement").sum()
    assert actual_sampled == expected_sampled, (
        f"Sampled count mismatch: expected {expected_sampled}, got {actual_sampled}"
    )
    assert actual_replacement == expected_replacement, (
        f"Replacement count mismatch: expected {expected_replacement}, got {actual_replacement}"
    )

    # Save the full grid with flags before filtering
    save_full_study_grid(grid, data_dir)

    # Keep only control cells (sampled + replacement)
    control_grid = grid[grid["sample_status"].notna()].copy()
    print(f"Built control grid: {len(control_grid)} cells "
          f"({actual_sampled} sampled, {actual_replacement} replacement)")
    return control_grid


def save_control_grid(grid: gpd.GeoDataFrame, data_dir: Path) -> Path:
    """Save the processed control grid to Google Drive.

    Args:
        grid: Control grid GeoDataFrame with sample_status column.
        data_dir: Root data directory.

    Returns:
        Path to the saved file.
    """
    output_path = data_dir / "01_input_data" / "boundaries" / "control_grid_5km_flagged.gpkg"
    grid.to_file(output_path, driver="GPKG")
    print(f"Saved control grid ({len(grid)} cells) to {output_path}")
    return output_path


def save_full_study_grid(grid: gpd.GeoDataFrame, data_dir: Path) -> Path:
    """Save the full study area grid (with sample_status flags) to Google Drive.

    Includes all grid cells — sampled, replacement, and unflagged.

    Args:
        grid: Full study area GeoDataFrame with sample_status column.
        data_dir: Root data directory.

    Returns:
        Path to the saved file.
    """
    output_path = data_dir / "01_input_data" / "boundaries" / "study_area_5km_flagged.gpkg"
    grid.to_file(output_path, driver="GPKG")
    print(f"Saved full study grid ({len(grid)} cells) to {output_path}")
    return output_path


def load_control_grid(data_dir: Path) -> gpd.GeoDataFrame:
    """Load the processed control grid (with sample_status flags).

    Looks for the pre-built flagged file first. If it doesn't exist,
    builds it from source files and saves it.

    Args:
        data_dir: Root data directory.

    Returns:
        GeoDataFrame with control grid cells and sample_status column.
    """
    flagged_path = data_dir / "01_input_data" / "boundaries" / "control_grid_5km_flagged.gpkg"

    if flagged_path.exists():
        gdf = gpd.read_file(flagged_path)
        print(f"Loaded {len(gdf)} control grid cells from {flagged_path}")
        return gdf

    print("Flagged control grid not found — building from source files...")
    gdf = build_control_grid(data_dir)
    save_control_grid(gdf, data_dir)
    return gdf


def load_layer(data_dir: Path, layer_name: str) -> gpd.GeoDataFrame | None:
    """Load an optional enhancement layer (roads, buildings, etc.).

    Args:
        data_dir: Root data directory.
        layer_name: Name of the layer file without extension
                    (e.g. 'roads', 'buildings').

    Returns:
        GeoDataFrame if the file exists, None otherwise.
    """
    base_layers_dir = data_dir / "01_input_data" / "base_layers"

    for ext in [".geojson", ".shp", ".gpkg"]:
        filepath = base_layers_dir / f"{layer_name}{ext}"
        if filepath.exists():
            gdf = gpd.read_file(filepath)
            print(f"Loaded layer '{layer_name}' ({len(gdf)} features) from {filepath}")
            return gdf

    print(f"Optional layer '{layer_name}' not found in {base_layers_dir} — skipping.")
    return None


def validate_crs(gdf: gpd.GeoDataFrame, expected_epsg: int = 4326) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to expected CRS if needed.

    Args:
        gdf: Input GeoDataFrame.
        expected_epsg: Target EPSG code (default 4326 / WGS84).

    Returns:
        GeoDataFrame in the expected CRS.
    """
    if gdf.crs is None:
        print(f"Warning: No CRS set. Assuming EPSG:{expected_epsg}.")
        gdf = gdf.set_crs(epsg=expected_epsg)
    elif gdf.crs.to_epsg() != expected_epsg:
        print(f"Reprojecting from {gdf.crs} to EPSG:{expected_epsg}")
        gdf = gdf.to_crs(epsg=expected_epsg)
    return gdf
