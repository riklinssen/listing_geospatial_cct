"""Load and validate geospatial boundary files from Google Drive.

Reads the 5km study area grid and control area sample flags,
merges them into a single GeoDataFrame with a sample_status column,
and provides loaders for optional enhancement layers (roads, buildings).

Also supports generating sub-grid cells (500m or 1km) from point centroids,
clipped to the parent 5km cell boundaries.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box


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


# ---------------------------------------------------------------------------
# Sub-grid functions (500m / 1km cells)
# ---------------------------------------------------------------------------


def _point_to_square(point, size: int = 500):
    """Convert a point to a square polygon centered on it."""
    half = size / 2
    return shapely_box(point.x - half, point.y - half, point.x + half, point.y + half)


def load_subgrid_points(data_dir: Path) -> gpd.GeoDataFrame:
    """Load the pre-filtered 500m sub-grid point centroids.

    Expects the file to exist at:
        <data_dir>/01_input_data/boundaries/subgrid_500m_control_treatment.gpkg

    Args:
        data_dir: Root data directory.

    Returns:
        GeoDataFrame with 500m point centroids.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = data_dir / "01_input_data" / "boundaries" / "subgrid_500m_control_treatment.gpkg"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Sub-grid points file not found: {filepath}\n"
            "Run the filtering step in notebook 01_explore_data.ipynb first."
        )
    gdf = gpd.read_file(filepath)
    print(f"Loaded {len(gdf)} sub-grid points from {filepath}")
    return gdf


def build_subgrid(
    subgrid_points: gpd.GeoDataFrame,
    control_grid: gpd.GeoDataFrame,
    cell_size: int = 500,
) -> gpd.GeoDataFrame:
    """Generate square sub-cells from point centroids, clipped to parent 5km cells.

    Args:
        subgrid_points: GeoDataFrame of 500m point centroids.
        control_grid: GeoDataFrame of 5km cells (must have 'id' column).
        cell_size: Size of output squares in metres (500 or 1000).

    Returns:
        GeoDataFrame of clipped square polygons with same attributes as input points,
        plus '5km_id' column linking to parent cell.
    """
    # Ensure both are in the same projected CRS (UTM 36S)
    if control_grid.crs.to_epsg() != 32736:
        control_grid = control_grid.to_crs(epsg=32736)

    points = subgrid_points.copy()
    if points.crs != control_grid.crs:
        points = points.to_crs(control_grid.crs)

    # Spatial join to populate 5km_id if missing or all NaN
    if "5km_id" not in points.columns or points["5km_id"].isna().all():
        print("Performing spatial join to assign 500m points to 5km cells...")
        joined = gpd.sjoin(
            points,
            control_grid[["id", "geometry"]],
            how="inner",
            predicate="within",
        )
        joined["5km_id"] = joined["id_right"]
        joined = joined.drop(columns=["id_right", "index_right"], errors="ignore")
        points = joined
        print(f"  Matched {len(points)} points to control cells")

    # Filter to control cells only (those with valid 5km_id)
    control_ids = set(control_grid["id"].astype(int))
    points = points[points["5km_id"].isin(control_ids)].copy()

    # For 1km cells, subsample to every other row/col to avoid overlap
    if cell_size == 1000:
        points = points[
            (points["row"] % 2 == 0) & (points["col"] % 2 == 0)
        ].copy()

    # Generate squares from points
    points["geometry"] = points.geometry.apply(_point_to_square, size=cell_size)

    # Build lookup of parent 5km geometries
    parent_geoms = control_grid.set_index("id")["geometry"].to_dict()

    # Clip each sub-cell to its parent 5km cell
    clipped_rows = []
    for _, row in points.iterrows():
        parent_id = int(row["5km_id"])
        if parent_id in parent_geoms:
            clipped_geom = row.geometry.intersection(parent_geoms[parent_id])
            if not clipped_geom.is_empty:
                new_row = row.copy()
                new_row["geometry"] = clipped_geom
                clipped_rows.append(new_row)

    result = gpd.GeoDataFrame(clipped_rows, crs=points.crs)
    print(f"Generated {len(result)} sub-cells at {cell_size}m resolution")
    return result


def save_subgrid(subgrid: gpd.GeoDataFrame, data_dir: Path, cell_size: int) -> Path:
    """Save a sub-grid to GeoPackage.

    Args:
        subgrid: GeoDataFrame of sub-grid polygons.
        data_dir: Root data directory.
        cell_size: Cell size in metres (for filename).

    Returns:
        Path to the saved file.
    """
    output_path = data_dir / "01_input_data" / "boundaries" / f"subgrid_{cell_size}m_control.gpkg"
    subgrid.to_file(output_path, driver="GPKG")
    print(f"Saved sub-grid ({len(subgrid)} cells) to {output_path}")
    return output_path


def load_subgrid(data_dir: Path, cell_size: int = 500) -> gpd.GeoDataFrame:
    """Load a sub-grid, building it from points if the cached file doesn't exist.

    Args:
        data_dir: Root data directory.
        cell_size: Cell size in metres (500 or 1000).

    Returns:
        GeoDataFrame of sub-grid polygons.
    """
    cached_path = data_dir / "01_input_data" / "boundaries" / f"subgrid_{cell_size}m_control.gpkg"

    if cached_path.exists():
        gdf = gpd.read_file(cached_path)
        print(f"Loaded {len(gdf)} sub-grid cells ({cell_size}m) from {cached_path}")
        return gdf

    print(f"Cached {cell_size}m sub-grid not found — building from points...")
    points = load_subgrid_points(data_dir)
    control_grid = load_control_grid(data_dir)
    subgrid = build_subgrid(points, control_grid, cell_size=cell_size)
    save_subgrid(subgrid, data_dir, cell_size)
    return subgrid


# ---------------------------------------------------------------------------
# Sub-cell selection (500m)
# ---------------------------------------------------------------------------


def select_subcells(
    grid_5km: gpd.GeoDataFrame,
    grid_500m: gpd.GeoDataFrame,
    min_buildings: int = 20,
    n_primary: int = 2,
    n_reserve: int = 2,
    seed: int = 42,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Select 500m sub-cells using PPS sampling, with 5km-level replacement.

    Steps:
      1. Filter 500m sub-cells to those with >= min_buildings.
      2. Identify sparse 5km cells (sampled cells with zero eligible sub-cells).
      3. Replace sparse sampled cells with viable replacement cells.
         Activated replacements become "sampled" (they are now primary cells).
      4. Within each viable 5km cell, draw sub-cells using probability
         proportional to size (PPS), weighted by building_count.

    Args:
        grid_5km: Control grid (must have 'id' and 'sample_status' columns).
        grid_500m: 500m sub-grid with 'building_count' and '5km_id' columns.
        min_buildings: Minimum building count for a sub-cell to be eligible.
        n_primary: Number of primary sub-cells per 5km cell.
        n_reserve: Number of reserve sub-cells per 5km cell.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (selected_subcells, updated_grid_5km) where:
        - selected_subcells: GeoDataFrame with 'selection_role', 'sample_status',
          and centroid columns.
        - updated_grid_5km: GeoDataFrame with updated sample_status reflecting
          replacements (activated replacements become "sampled",
          dropped sparse cells become "dropped_sparse").
    """
    id_col = "5km_id"
    n_required = n_primary + n_reserve
    rng = np.random.default_rng(seed)

    # --- Step 1: Filter eligible 500m sub-cells ---
    eligible = grid_500m[grid_500m["building_count"] >= min_buildings].copy()
    eligible_counts = eligible.groupby(id_col).size()
    print(f"Eligible sub-cells (>={min_buildings} buildings): "
          f"{len(eligible)} / {len(grid_500m)}")

    # --- Step 2: Identify sparse sampled cells and viable replacements ---
    grid = grid_5km.copy()
    grid["n_eligible"] = grid["id"].map(eligible_counts).fillna(0).astype(int)

    sparse_sampled = grid[
        (grid["sample_status"] == "sampled") & (grid["n_eligible"] == 0)
    ]
    viable_replacements = grid[
        (grid["sample_status"] == "replacement") & (grid["n_eligible"] > 0)
    ]

    print(f"\nSparse sampled 5km cells (no eligible sub-cells): {len(sparse_sampled)}")
    print(f"Viable replacement 5km cells: {len(viable_replacements)}")

    # --- Step 3: Activate replacements ---
    n_to_replace = min(len(sparse_sampled), len(viable_replacements))
    if n_to_replace > 0:
        # Pick replacements randomly
        replacement_ids = rng.choice(
            viable_replacements["id"].values,
            size=n_to_replace,
            replace=False,
        )
        # Activate: replacement -> sampled
        grid.loc[grid["id"].isin(replacement_ids), "sample_status"] = "sampled"
        # Mark sparse sampled cells as dropped
        grid.loc[
            (grid["sample_status"] == "sampled") & (grid["n_eligible"] == 0),
            "sample_status",
        ] = "dropped_sparse"

        print(f"\nReplacement summary:")
        print(f"  Activated {n_to_replace} replacement cells -> now 'sampled'")
        for rid in replacement_ids:
            n_elig = eligible_counts.get(rid, 0)
            print(f"    Cell {int(rid)}: {n_elig} eligible sub-cells")
        print(f"  Dropped {len(sparse_sampled)} sparse sampled cells:")
        for _, r in sparse_sampled.iterrows():
            print(f"    Cell {int(r['id'])}: {int(r['n_eligible'])} eligible sub-cells")

    # Also mark sparse replacement cells (no eligible sub-cells, still "replacement")
    grid.loc[
        (grid["sample_status"] == "replacement") & (grid["n_eligible"] == 0),
        "sample_status",
    ] = "dropped_sparse"

    # --- Step 4: PPS sampling within each viable cell ---
    active_cells = grid[grid["sample_status"].isin(["sampled", "replacement"])]
    print(f"\nActive 5km cells for sub-cell selection: {len(active_cells)}")
    print(f"  Sampled: {(active_cells['sample_status'] == 'sampled').sum()}")
    print(f"  Replacement: {(active_cells['sample_status'] == 'replacement').sum()}")

    results = []
    for cell_id in active_cells["id"].values:
        cell_eligible = eligible[eligible[id_col] == cell_id]
        n_avail = len(cell_eligible)
        if n_avail == 0:
            continue

        n_select = min(n_required, n_avail)

        # PPS weights: probability proportional to building_count
        weights = cell_eligible["building_count"].values.astype(float)
        weights = weights / weights.sum()

        selected_idx = rng.choice(
            cell_eligible.index, size=n_select, replace=False, p=weights,
        )

        for i, idx in enumerate(selected_idx):
            role = "primary" if i < n_primary else "reserve"
            row = cell_eligible.loc[idx].copy()
            row["selection_role"] = role
            results.append(row)

    selected = gpd.GeoDataFrame(results, crs=grid_500m.crs)

    # Add sample_status from updated 5km grid
    selected = selected.merge(
        grid[["id", "sample_status"]].rename(columns={"id": id_col}),
        on=id_col, how="left",
    )

    # Drop redundant columns from upstream joins
    drop_cols = ["feature_x", "feature_y", "nearest_x", "nearest_y", "distance", "n"]
    selected = selected.drop(columns=drop_cols, errors="ignore")

    # Add centroids (UTM for analysis, lat/lon for field devices)
    selected["centroid_x"] = selected.geometry.centroid.x
    selected["centroid_y"] = selected.geometry.centroid.y
    centroids_wgs84 = selected.geometry.centroid.to_crs(epsg=4326)
    selected["latitude"] = centroids_wgs84.y
    selected["longitude"] = centroids_wgs84.x

    n_primary_actual = (selected["selection_role"] == "primary").sum()
    n_reserve_actual = (selected["selection_role"] == "reserve").sum()
    n_cells_covered = selected[id_col].nunique()

    print(f"\n{'='*60}")
    print(f"SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Selected {len(selected)} sub-cells "
          f"({n_primary_actual} primary, {n_reserve_actual} reserve)")
    print(f"Covering {n_cells_covered} / {len(grid)} 5km cells")
    print(f"Dropped sparse cells: {(grid['sample_status'] == 'dropped_sparse').sum()}")

    # Drop helper column before returning
    grid = grid.drop(columns=["n_eligible"])

    return selected, grid


def save_selected_subcells(selected: gpd.GeoDataFrame, data_dir: Path) -> Path:
    """Save selected sub-cells to GeoPackage.

    Args:
        selected: GeoDataFrame from select_subcells().
        data_dir: Root data directory.

    Returns:
        Path to the saved file.
    """
    output_path = data_dir / "01_input_data" / "boundaries" / "selected_subcells_500m.gpkg"
    selected.to_file(output_path, driver="GPKG")
    print(f"Saved selected sub-cells ({len(selected)}) to {output_path}")
    return output_path


def load_selected_subcells(data_dir: Path) -> gpd.GeoDataFrame | None:
    """Load previously selected sub-cells.

    Args:
        data_dir: Root data directory.

    Returns:
        GeoDataFrame of selected sub-cells, or None if file doesn't exist.
    """
    path = data_dir / "01_input_data" / "boundaries" / "selected_subcells_500m.gpkg"
    if path.exists():
        gdf = gpd.read_file(path)
        print(f"Loaded {len(gdf)} selected sub-cells from {path}")
        return gdf
    print(f"Selected sub-cells not found: {path}")
    return None


def load_buildings(data_dir: Path) -> gpd.GeoDataFrame | None:
    """Load Google Open Buildings footprints from parquet.

    Args:
        data_dir: Root data directory.

    Returns:
        GeoDataFrame of building footprints, or None if file doesn't exist.
    """
    # Try known filenames in order of preference
    base_layers = data_dir / "01_input_data" / "base_layers"
    for name in ["overture_buildings.parquet", "google_buildings.parquet"]:
        parquet_path = base_layers / name
        if parquet_path.exists():
            gdf = gpd.read_parquet(parquet_path)
            print(f"Loaded {len(gdf)} buildings from {parquet_path}")
            return gdf

    # Fall back to load_layer for other formats (geojson, shp, gpkg)
    return load_layer(data_dir, "buildings")


def load_grid_with_building_counts(data_dir: Path, grid_size: int = 5000) -> gpd.GeoDataFrame | None:
    """Load a grid file that includes building counts.

    Args:
        data_dir: Root data directory.
        grid_size: Grid cell size in metres (5000, 1000, or 500).

    Returns:
        GeoDataFrame with building_count column, or None if file doesn't exist.
    """
    size_label = "5km" if grid_size == 5000 else f"{grid_size}m"
    counts_path = data_dir / "01_input_data" / "base_layers" / f"grid_{size_label}_building_counts.gpkg"

    if counts_path.exists():
        gdf = gpd.read_file(counts_path)
        print(f"Loaded {len(gdf)} grid cells with building counts from {counts_path}")
        return gdf

    print(f"Grid with building counts not found: {counts_path}")
    print("Run: python scripts/download_google_buildings.py --grid-size {grid_size}")
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
