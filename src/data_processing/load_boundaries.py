"""Load and validate geospatial boundary files from Google Drive.

Reads enumeration area boundaries, control grid cells, and optional
enhancement layers (roads, buildings) from the shared Google Drive
data directory.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_enumeration_areas(data_dir: Path) -> gpd.GeoDataFrame:
    """Load enumeration area boundaries from GeoJSON.

    Args:
        data_dir: Root data directory (e.g. Google Drive 0.4_listing_geospatial/).

    Returns:
        GeoDataFrame with enumeration area polygons.

    Raises:
        FileNotFoundError: If the enumeration areas file does not exist.
    """
    filepath = data_dir / "01_input_data" / "boundaries" / "enumeration_areas.geojson"
    if not filepath.exists():
        raise FileNotFoundError(f"Enumeration areas file not found: {filepath}")
    gdf = gpd.read_file(filepath)
    print(f"Loaded {len(gdf)} enumeration areas from {filepath}")
    return gdf


def load_control_grid(data_dir: Path) -> gpd.GeoDataFrame:
    """Load control grid cells from GeoJSON or Shapefile.

    Looks for .geojson first, then falls back to .shp.

    Args:
        data_dir: Root data directory.

    Returns:
        GeoDataFrame with control grid cell polygons.

    Raises:
        FileNotFoundError: If no control grid file is found.
    """
    boundaries_dir = data_dir / "01_input_data" / "boundaries"

    geojson_path = boundaries_dir / "control_grid.geojson"
    shp_path = boundaries_dir / "control_grid.shp"

    if geojson_path.exists():
        filepath = geojson_path
    elif shp_path.exists():
        filepath = shp_path
    else:
        raise FileNotFoundError(
            f"Control grid file not found. Looked for:\n"
            f"  {geojson_path}\n"
            f"  {shp_path}"
        )

    gdf = gpd.read_file(filepath)
    print(f"Loaded {len(gdf)} grid cells from {filepath}")
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
