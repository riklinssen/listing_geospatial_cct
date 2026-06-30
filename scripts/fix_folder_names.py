"""Rename grid_info output folders to correct, per-sub-cell names (DRY RUN by default).

Fixes the existing on-disk outputs without re-rendering any tiles:

  * layers/<ward>_<5km>_<5km_status>_<role>_<subcell>
        ward       -> the ward the SUB-CELL's centroid actually sits in
        5km_status -> the correct (post-activation) sampled/replacement of the 5km cell
        role       -> primary/reserve of this 500m sub-cell within the cell
  * <status>/<...>_<5km>   (the sampled/ & replacement/ PNG + info cell folders)
        moved under the correct sampled/ or replacement/, and renamed
        <region>_<district>_<5km>  (a 5km cluster spans wards, so it's named by
        its dominant region/district, not a single ward)

Source of truth: selected_subcells_500m.gpkg (the validated per-sub-cell labels).
The ward is NOT inside the tile image, so renaming is equivalent to re-rendering.

DRY RUN by default — prints every planned change and touches nothing.
Pass --apply to actually rename/move.

Usage:
    python scripts/fix_folder_names.py                      # dry run (default path)
    python scripts/fix_folder_names.py --apply              # execute
    python scripts/fix_folder_names.py "G:/.../grid_info"   # dry run, explicit path
"""
import re
import shutil
import sys
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_data_dir

DEFAULT_GRID_INFO = Path(
    r"G:\Shared drives\TZ-CCT_RUBEV-0825\Data\0_Listing\1_Input\grid_info"
)
TEST_DIRS = {"world", "Dar_office_test"}


def sanitize(name) -> str:
    return re.sub(r"[^\w\-]", "_", str(name)).strip("_")


def _mode(series, default="unknown"):
    vals = series.dropna()
    return str(vals.mode().iloc[0]) if len(vals) else default


def main():
    argv = sys.argv[1:]
    apply = "--apply" in argv
    paths = [a for a in argv if a != "--apply"]
    grid_info = Path(paths[0]) if paths else DEFAULT_GRID_INFO
    layers = grid_info / "layers"

    config = load_config()
    data_dir = get_data_dir(config)
    sel = gpd.read_file(data_dir / "01_input_data" / "boundaries" / "selected_subcells_500m.gpkg")
    by_grid = sel.set_index("grid_id")

    # Per-5km-cell dominant region/district + (uniform) status, for cell-folder names
    g = sel.groupby(sel["5km_id"].astype(int))
    cell = g.agg(
        district=("district", _mode),
        region=("region", _mode),
        status=("sample_status", _mode),
    )

    print(f"=== {'APPLY' if apply else 'DRY RUN'} ===  grid_info: {grid_info}")

    # ---- 1. layers/ : per-sub-cell tile folders ----
    print("\n--- layers/ tile folders (per 500m sub-cell -> ward) ---")
    ren = same = miss = 0
    for p in sorted(x for x in layers.iterdir() if x.is_dir()) if layers.exists() else []:
        if p.name in TEST_DIRS:
            continue
        m = re.search(r"(G_\d+_\d+)$", p.name)
        if not m:
            print("  ? unrecognised, skip:", p.name)
            continue
        sub = m.group(1)
        if sub not in by_grid.index:
            print("  ? not in selected_subcells, skip:", p.name)
            miss += 1
            continue
        r = by_grid.loc[sub]
        # <ward>_<5km_id>_<5km_status>_<role>_<subcell_id>
        new = (f"{sanitize(r['ward_name'])}_{int(r['5km_id'])}_"
               f"{r['sample_status']}_{r['selection_role']}_{sub}")
        has_tile = bool(list(p.glob("*.mbtiles")))
        if new == p.name:
            same += 1
            continue
        tag = "" if has_tile else "   (NO TILE yet - render after)"
        print(f"  RENAME  {p.name}{tag}\n       -> {new}")
        ren += 1
        if apply:
            target = layers / new
            if target.exists():
                print("       !! target exists - SKIPPED")
                continue
            for mb in p.glob("*.mbtiles"):
                mb.rename(p / f"{new}.mbtiles")
            p.rename(target)
    print(f"  layers: {ren} to rename | {same} already correct | {miss} unmatched")

    # ---- 2. sampled/ & replacement/ : per-5km-cell folders ----
    print("\n--- sampled/ & replacement/ cell folders (per 5km cell -> region_district) ---")
    cren = csame = 0
    for status_dir in ["sampled", "replacement"]:
        sd = grid_info / status_dir
        if not sd.exists():
            continue
        for p in sorted(x for x in sd.iterdir() if x.is_dir()):
            m = re.search(r"_(\d+)$", p.name)
            if not m:
                print(f"  ? unrecognised, skip: {status_dir}/{p.name}")
                continue
            id5 = int(m.group(1))
            if id5 not in cell.index:
                print(f"  ? 5km id not in selected, skip: {status_dir}/{p.name}")
                continue
            c = cell.loc[id5]
            new_status = c["status"]
            new_name = f"{sanitize(c['region'] + '_' + c['district'])}_{id5}"
            target = grid_info / new_status / new_name
            if target == p:
                csame += 1
                continue
            moved = "" if new_status == status_dir else f"   (MOVE {status_dir} -> {new_status})"
            print(f"  {status_dir}/{p.name}{moved}\n       -> {new_status}/{new_name}")
            cren += 1
            if apply:
                if target.exists():
                    print("       !! target exists - SKIPPED")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(target))
    print(f"  cell folders: {cren} to move/rename | {csame} already correct")

    if not apply:
        print("\nDRY RUN only - nothing changed. Re-run with --apply to execute.")


if __name__ == "__main__":
    main()
