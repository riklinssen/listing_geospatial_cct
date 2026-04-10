"""Generate listing maps for all selected 500m sub-cells.

For each 5km control cell, generates:
  1. An overview map (5km cell with selected sub-cells highlighted)
  2. A detail map per selected 500m sub-cell (with building footprints)

Output folder structure:
  <output_dir>/<ward_name>_<5km_id>/
      overview_5km.png
      subcell_1_primary.png
      subcell_2_primary.png
      subcell_3_reserve.png
      subcell_4_reserve.png

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
        grid_5km = grid_5km[grid_5km["id"].astype(str) == str(args.single)].copy()
        selected = selected[selected["5km_id"].astype(str) == str(args.single)].copy()
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
    print(f"\nGenerating maps for {total_cells} cells...")

    for _, row in tqdm(grid_5km.iterrows(), total=total_cells, desc="Cells"):
        grid_id = row["id"]
        cell = grid_5km[grid_5km["id"] == grid_id]
        cell_selected = selected[selected["5km_id"] == grid_id]

        if len(cell_selected) == 0:
            print(f"  Skipping cell {grid_id}: no selected sub-cells")
            continue

        # Folder structure: <status>/<ward_name>_<5km_id>/<role>/
        sample_status = row["sample_status"]  # "sampled" or "replacement"
        ward_name = get_ward_name(grid_id, cell_selected)
        cell_folder = output_dir / sample_status / f"{ward_name}_{int(grid_id)}"

        # Overview map (in the cell folder root)
        if not args.skip_overview:
            cell_folder.mkdir(parents=True, exist_ok=True)
            if emit_png:
                generator.output_dir = cell_folder
                label = f"{int(grid_id)} ({sample_status}) — {ward_name}"
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
                        f"5km cell {int(grid_id)} — {ward_name} ({sample_status})"
                    )
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
                        detail_title = (
                            f"5km: {int(grid_id)} — 500m: {subcell_id} "
                            f"({role}) — {building_count} buildings"
                        )
                        export_detail_mbtiles(
                            subcell=subcell,
                            output_path=role_dir / f"subcell_{subcell_id}_{role}.mbtiles",
                            buildings=buildings,
                            role=role,
                            zoom=args.mbtiles_detail_zoom,
                            title=detail_title,
                        )
                        total_maps += 1
                    except Exception as e:
                        print(f"  MBTiles detail failed for {grid_id}/{subcell_id}: {e}")

                # Export sub-cell polygon as GeoJSON (WGS84 for SurveyCTO)
                export_subcell_geojson(
                    subcell_row, subcells_crs=selected.crs,
                    output_dir=role_dir,
                    grid_id=int(grid_id), subcell_id=subcell_id, role=role,
                )

    print(f"\nDone! Generated {total_maps} maps across {total_cells} cells")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
