"""Rebuild layers/samplegrids_primary_replacement.xlsx from the on-disk tiles.

Use this after a targeted ``generate_all_maps.py --single`` re-render, which
overwrites the index xlsx with only the re-rendered cell. This scans every
layer ``.mbtiles`` folder, parses its name, and joins per-sub-cell attributes
from selected_subcells_500m.gpkg (+ VCSL flags) to produce a complete index.

The sample_status / ward in each row comes from the on-disk folder NAME (so the
xlsx stays consistent with the actual tile files). Note: that reflects the tile
run's labels, which can be stale for activated-replacement cells — the
authoritative status lives in control_grid_overview / selected_subcells.

Usage:
    python scripts/rebuild_samplegrids.py
    python scripts/rebuild_samplegrids.py "G:/path/to/grid_info"
"""
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_data_dir
from src.data_processing.load_boundaries import load_vcsl_flags, flag_vcsl_villages

DEFAULT_GRID_INFO = Path(
    r"G:\Shared drives\TZ-CCT_RUBEV-0825\Data\0_Listing\1_Input\grid_info"
)


def main():
    grid_info = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_INFO
    layers = grid_info / "layers"
    if not layers.exists():
        sys.exit(f"layers folder not found: {layers}")

    config = load_config()
    data_dir = get_data_dir(config)
    sel = gpd.read_file(data_dir / "01_input_data" / "boundaries" / "selected_subcells_500m.gpkg")
    sel = flag_vcsl_villages(sel, load_vcsl_flags(Path(config["paths"]["vcsl_flags"])), id_col="5km_id")
    by_grid = sel.set_index("grid_id")

    pat = re.compile(r"^(?P<ward>.+)_(?P<id5km>\d+)_(?P<status>sampled|replacement)_(?P<sub>G_\d+_\d+)$")

    rows = []
    for folder in sorted(p for p in layers.iterdir() if p.is_dir()):
        mb = folder / f"{folder.name}.mbtiles"
        if not mb.exists():
            print("  skip (no tile):", folder.name)
            continue
        m = pat.match(folder.name)
        if not m:
            print("  skip (unparseable name):", folder.name)
            continue
        sub = m.group("sub")
        r = by_grid.loc[sub] if sub in by_grid.index else None
        rows.append({
            "layer": folder.name,
            "ward_name": m.group("ward"),
            "5km_id": int(m.group("id5km")),
            "subcell_id": sub,
            "sample_status": m.group("status"),
            "selection_role": (r["selection_role"] if r is not None else None),
            "building_count": (int(r["building_count"]) if r is not None and pd.notna(r["building_count"]) else None),
            "vcsl_village": (r["vcsl_village"] if r is not None and pd.notna(r["vcsl_village"]) else None),
            "latitude": (r["latitude"] if r is not None else None),
            "longitude": (r["longitude"] if r is not None else None),
            "mbtiles_path": str(Path("layers") / folder.name / f"{folder.name}.mbtiles"),
        })

    df = pd.DataFrame(rows).sort_values(["5km_id", "subcell_id"]).reset_index(drop=True)
    out = layers / "samplegrids_primary_replacement.xlsx"
    df.to_excel(out, index=False, sheet_name="sample_grids")
    print(f"rebuilt {out} with {len(df)} tiles")


if __name__ == "__main__":
    main()
