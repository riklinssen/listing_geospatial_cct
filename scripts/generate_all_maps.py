"""Generate listing maps for all selected 500m sub-cells.

For each 5km control cell, generates:
  1. An overview map (5km cell with selected sub-cells highlighted)
  2. A detail map per selected 500m sub-cell (with building footprints)

Output folder structure:
  <output_dir>/<status>/<ward_name>_<5km_id>/
      overview_5km.png
      overview_5km.mbtiles            (with --mbtiles)
      primary/  reserve/  ...PNGs + GeoJSON per sub-cell

  500m sub-cell MBTiles (with --mbtiles) are written as flat layers:
  <output_dir>/layers/<ward_name>_<5km_id>_<status>_<subcell_id>/
      <ward_name>_<5km_id>_<status>_<subcell_id>.mbtiles
  <output_dir>/layers/samplegrids_primary_replacement.xlsx   (index of all layers)

Usage:
    python scripts/generate_all_maps.py
    python scripts/generate_all_maps.py --status sampled
    python scripts/generate_all_maps.py --single 13682
    python scripts/generate_all_maps.py --no-basemap
    python scripts/generate_all_maps.py --output-dir "G:/path/to/output"
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from src.utils.config_loader import load_config, get_data_dir, get_output_dir
from src.data_processing.load_boundaries import (
    load_control_grid,
    load_selected_subcells,
    load_buildings,
    validate_crs,
)
from src.mapping.map_generator import MapGenerator
from src.mapping.mbtiles_export import (
    export_overview_mbtiles,
    export_detail_mbtiles,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate listing maps for selected 500m sub-cells."
    )
    parser.add_argument(
        "--no-basemap", action="store_true",
        help="Disable online basemap tiles (useful offline).",
    )
    parser.add_argument(
        "--status", choices=["sampled", "replacement"], default=None,
        help="Generate maps only for cells with this sample_status.",
    )
    parser.add_argument(
        "--single", default=None,
        help="Generate maps for a single 5km cell ID only.",
    )
    parser.add_argument(
        "--scalebar", action="store_true",
        help="Add a distance scale bar to each map.",
    )
    parser.add_argument(
        "--skip-overview", action="store_true",
        help="Skip generating 5km overview maps.",
    )
    parser.add_argument(
        "--skip-detail", action="store_true",
        help="Skip generating 500m detail maps.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: from config).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N cells (for testing).",
    )
    parser.add_argument(
        "--mbtiles", action="store_true",
        help="Also emit a styled .mbtiles file alongside each PNG (for SurveyCTO offline use).",
    )
    parser.add_argument(
        "--mbtiles-only", action="store_true",
        help="Only emit MBTiles, skip PNG generation. Implies --mbtiles.",
    )
    parser.add_argument(
        "--mbtiles-detail-zoom", type=int, default=19,
        help="Max zoom level for 500m detail MBTiles (default 19).",
    )
    parser.add_argument(
        "--mbtiles-overview-zoom", type=int, default=15,
        help="Max zoom level for 5km overview MBTiles (default 15).",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    """Make a string safe for folder names."""
    return re.sub(r'[^\w\-]', '_', str(name)).strip('_')


def get_ward_name(grid_id, selected_subcells: gpd.GeoDataFrame) -> str:
    """Get ward name for a 5km cell from its sub-cells."""
    cell_subcells = selected_subcells[selected_subcells["5km_id"] == grid_id]
    if "ward_name" in cell_subcells.columns and len(cell_subcells) > 0:
        ward = cell_subcells.iloc[0]["ward_name"]
        if pd.notna(ward) and str(ward).strip():
            return sanitize_name(str(ward).strip())
    return "unknown_ward"


def get_vcsl_village(selected_subcells: gpd.GeoDataFrame) -> str | None:
    """Get the CCT VCSL village label(s) for a cell's sub-cells, or None."""
    if "vcsl_village" in selected_subcells.columns:
        vals = selected_subcells["vcsl_village"].dropna().unique()
        if len(vals) > 0:
            return ", ".join(sorted(str(v) for v in vals))
    return None


def cell_value(cell_subcells: gpd.GeoDataFrame, col: str, default) -> str:
    """Most common non-null value of a column across a 5km cell's sub-cells.

    A 5km cell can span several wards/districts; this returns the dominant one
    for cell-level folder naming. For sample_status (uniform per cell) it just
    returns that status.
    """
    if col in cell_subcells.columns:
        vals = cell_subcells[col].dropna()
        if len(vals) > 0:
            return str(vals.mode().iloc[0])
    return default


def export_subcell_geojson(
    subcell_row, subcells_crs, output_dir: Path,
    grid_id: int, subcell_id, role: str,
) -> Path:
    """Export a single 500m sub-cell polygon as GeoJSON (WGS84)."""
    from shapely.geometry import mapping

    geom = subcell_row.geometry
    # Reproject to WGS84
    if subcells_crs and subcells_crs.to_epsg() != 4326:
        gs = gpd.GeoSeries([geom], crs=subcells_crs).to_crs(epsg=4326)
        geom = gs.iloc[0]

    feature = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "5km_cell_id": grid_id,
                "subcell_id": subcell_id,
                "role": role,
                "building_count": int(subcell_row.get("building_count", 0)),
                "latitude": float(subcell_row.get("latitude", 0)),
                "longitude": float(subcell_row.get("longitude", 0)),
                "vcsl_village": (
                    str(subcell_row["vcsl_village"])
                    if pd.notna(subcell_row.get("vcsl_village"))
                    else None
                ),
            },
            "geometry": mapping(geom),
        }],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"subcell_{subcell_id}_{role}.geojson"
    output_path.write_text(json.dumps(feature, indent=2), encoding="utf-8")
    return output_path


def main():
    args = parse_args()

    # --mbtiles-only implies --mbtiles
    emit_mbtiles = args.mbtiles or args.mbtiles_only
    emit_png = not args.mbtiles_only

    config = load_config()
    data_dir = get_data_dir(config)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_output_dir(config) / "listing_maps"

    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    grid_5km = load_control_grid(data_dir)
    grid_5km = validate_crs(grid_5km)

    selected = load_selected_subcells(data_dir)
    if selected is None:
        print("Error: No selected sub-cells found. Run build_data.py first.")
        sys.exit(1)
    selected = validate_crs(selected)

    # Load buildings for detail maps
    buildings = None
    if not args.skip_detail:
        buildings = load_buildings(data_dir)
        if buildings is not None:
            print(f"Loaded {len(buildings)} building footprints")

    # Exclude dropped sparse cells
    grid_5km = grid_5km[grid_5km["sample_status"] != "dropped_sparse"].copy()
    selected = selected[selected["sample_status"] != "dropped_sparse"].copy()

    # Filter by status
    if args.status:
        grid_5km = grid_5km[grid_5km["sample_status"] == args.status].copy()
        selected = selected[selected["sample_status"] == args.status].copy()
        print(f"Filtered to '{args.status}': {len(grid_5km)} cells, {len(selected)} sub-cells")

    # Filter to single cell
    if args.single:
        single_id = int(args.single)
        grid_5km = grid_5km[grid_5km["id"].astype(int) == single_id].copy()
        # 5km_id is stored as a float (e.g. 13130.0); compare numerically, not as
        # strings ("13130.0" != "13130" would drop every sub-cell).
        selected = selected[selected["5km_id"].fillna(-1).astype(int) == single_id].copy()
        if len(grid_5km) == 0:
            print(f"Error: Cell '{args.single}' not found.")
            sys.exit(1)

    # Limit number of cells (for testing)
    if args.limit:
        grid_5km = grid_5km.head(args.limit).copy()
        cell_ids = set(grid_5km["id"])
        selected = selected[selected["5km_id"].isin(cell_ids)].copy()
        print(f"Limited to first {args.limit} cells: {len(grid_5km)} cells, {len(selected)} sub-cells")

    # Initialize generator (output_dir will be overridden per cell)
    map_settings = config.get("map_settings", {})
    generator = MapGenerator(
        output_dir=output_dir,
        fig_width=map_settings.get("fig_width", 19.2),
        fig_height=map_settings.get("fig_height", 10.8),
        dpi=map_settings.get("dpi", 100),
        add_basemap=not args.no_basemap,
        add_scalebar=args.scalebar,
    )

    # Generate maps
    total_cells = len(grid_5km)
    total_maps = 0
    layer_rows = []  # one entry per sub-cell MBTiles layer (for the index xlsx)
    print(f"\nGenerating maps for {total_cells} cells...")

    for _, row in tqdm(grid_5km.iterrows(), total=total_cells, desc="Cells"):
        grid_id = row["id"]
        cell = grid_5km[grid_5km["id"] == grid_id]
        cell_selected = selected[selected["5km_id"] == grid_id]

        if len(cell_selected) == 0:
            print(f"  Skipping cell {grid_id}: no selected sub-cells")
            continue

        # Folder structure: <status>/<region>_<district>_<5km_id>/<role>/
        # Status comes from the SELECTED sub-cells (post-activation), not the
        # possibly-stale control grid. A 5km cell is one cluster that can span
        # wards/districts, so the cell folder is named by its dominant
        # region/district rather than a single (misleading) ward.
        sample_status = cell_value(cell_selected, "sample_status", row["sample_status"])
        cell_district = cell_value(cell_selected, "district", "unknown")
        cell_region = cell_value(cell_selected, "region", "unknown")
        ward_name = get_ward_name(grid_id, cell_selected)  # representative, for overview labels
        cell_label = sanitize_name(f"{cell_region}_{cell_district}")
        cell_folder = output_dir / sample_status / f"{cell_label}_{int(grid_id)}"

        # Overview map (in the cell folder root)
        if not args.skip_overview:
            cell_folder.mkdir(parents=True, exist_ok=True)
            if emit_png:
                generator.output_dir = cell_folder
                # No single ward here — a 5km cell can span several wards.
                label = f"5km cell {int(grid_id)} ({sample_status}) — {cell_district}, {cell_region}"
                generator.generate_overview(
                    grid_cell=cell,
                    grid_id=str(int(grid_id)),
                    selected_subcells=cell_selected,
                    all_grid_cells=grid_5km,
                    label=label,
                )
                total_maps += 1
            if emit_mbtiles:
                try:
                    overview_title = (
                        f"5km cell {int(grid_id)} — {cell_district}, {cell_region} ({sample_status})"
                    )
                    cell_vcsl = get_vcsl_village(cell_selected)
                    if cell_vcsl:
                        overview_title += f" — VCSL: {cell_vcsl}"
                    export_overview_mbtiles(
                        grid_cell=cell,
                        selected_subcells=cell_selected,
                        output_path=cell_folder / "overview_5km.mbtiles",
                        zoom=args.mbtiles_overview_zoom,
                        title=overview_title,
                    )
                    total_maps += 1
                except Exception as e:
                    print(f"  MBTiles overview failed for {grid_id}: {e}")

        # Detail maps and GeoJSON (in primary/ or reserve/ subfolder)
        if not args.skip_detail:
            cell_selected_sorted = cell_selected.sort_values(
                "selection_role", ascending=True,
            )
            for i, (_, subcell_row) in enumerate(cell_selected_sorted.iterrows(), 1):
                role = subcell_row["selection_role"]
                role_dir = cell_folder / role
                role_dir.mkdir(parents=True, exist_ok=True)
                subcell = cell_selected_sorted[cell_selected_sorted.index == subcell_row.name]
                subcell_id = subcell_row.get("grid_id", f"subcell_{i}")

                if emit_png:
                    generator.output_dir = role_dir
                    generator.generate_detail(
                        subcell=subcell,
                        grid_id=str(int(grid_id)),
                        subcell_index=i,
                        role=role,
                        buildings=buildings,
                    )
                    total_maps += 1

                if emit_mbtiles:
                    try:
                        building_count = subcell_row.get("building_count", "?")
                        sc_vcsl = subcell_row.get("vcsl_village")
                        detail_title = (
                            f"5km: {int(grid_id)} — 500m: {subcell_id} "
                            f"({role}) — {building_count} buildings"
                        )
                        if pd.notna(sc_vcsl):
                            detail_title += f" — VCSL: {sc_vcsl}"

                        # Per-sub-cell admin labels — the ward the sub-cell's
                        # centroid actually sits in (not the cell-level label).
                        sc_ward = subcell_row.get("ward_name")
                        sc_district = subcell_row.get("district")
                        sc_region = subcell_row.get("region")
                        sc_status = subcell_row.get("sample_status", sample_status)
                        ward_token = (sanitize_name(str(sc_ward))
                                      if pd.notna(sc_ward) else ward_name)

                        # Flat per-layer folder under <output_dir>/layers/:
                        #   layers/<ward>_<5km_id>_<5km_status>_<role>_<subcell_id>/<same>.mbtiles
                        # 5km_status = sampled/replacement (the cell); role =
                        # primary/reserve (this 500m sub-cell within the cell).
                        layer_name = (
                            f"{ward_token}_{int(grid_id)}_{sc_status}_{role}_"
                            f"{sanitize_name(str(subcell_id))}"
                        )
                        layer_dir = output_dir / "layers" / layer_name
                        layer_dir.mkdir(parents=True, exist_ok=True)

                        export_detail_mbtiles(
                            subcell=subcell,
                            output_path=layer_dir / f"{layer_name}.mbtiles",
                            buildings=buildings,
                            role=role,
                            zoom=args.mbtiles_detail_zoom,
                            title=detail_title,
                        )
                        total_maps += 1

                        layer_rows.append({
                            "layer": layer_name,
                            "ward_name": sc_ward,
                            "district": sc_district,
                            "region": sc_region,
                            "5km_id": int(grid_id),
                            "subcell_id": subcell_id,
                            "sample_status": sc_status,
                            "selection_role": role,
                            "building_count": (
                                building_count if building_count != "?" else None
                            ),
                            "vcsl_village": sc_vcsl if pd.notna(sc_vcsl) else None,
                            "latitude": subcell_row.get("latitude"),
                            "longitude": subcell_row.get("longitude"),
                            "mbtiles_path": str(
                                Path("layers") / layer_name / f"{layer_name}.mbtiles"
                            ),
                        })
                    except Exception as e:
                        print(f"  MBTiles detail failed for {grid_id}/{subcell_id}: {e}")

                # Export sub-cell polygon as GeoJSON (WGS84 for SurveyCTO)
                export_subcell_geojson(
                    subcell_row, subcells_crs=selected.crs,
                    output_dir=role_dir,
                    grid_id=int(grid_id), subcell_id=subcell_id, role=role,
                )

    # Write the layer index (one row per sub-cell MBTiles layer)
    if emit_mbtiles and layer_rows:
        layers_dir = output_dir / "layers"
        layers_dir.mkdir(parents=True, exist_ok=True)
        index_df = pd.DataFrame(layer_rows)
        index_path = layers_dir / "samplegrids_primary_replacement.xlsx"
        index_df.to_excel(index_path, index=False, sheet_name="sample_grids")
        print(f"Wrote layer index ({len(index_df)} layers) to {index_path}")

    print(f"\nDone! Generated {total_maps} maps across {total_cells} cells")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
