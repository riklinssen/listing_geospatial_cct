"""Snap each 500m sub-cell to its nearest reachable road access point (Overpass).

Companion to :mod:`landmarks`. Where landmarks give recognisable places to look
for, this gives the nearest point *on a road* that a vehicle/boda can drive to,
plus how far that point is from the 500m cell centroid (the off-road walk). It
is the notebook-12 "road snap" idea, hardened for a batch run:

- **major roads preferred** — trunk/primary/secondary/tertiary/unclassified win
  over minor residential/service/track when both are in reach; the class that
  was actually snapped to is recorded so you can see what it found.
- **one Overpass query per 5km cell** (roads in the cell bbox + margin), reused
  across its sub-cells — same efficiency as the landmark build.
- **mirror rotation + resumable CSV cache** — survives the flaky public servers.

Output is one row per sub-cell, keyed by ``grid_id``, so it merges straight into
the landmark guide later:

    grid_id, 5km_id, centroid_lat, centroid_lon, road_lat, road_lon,
    off_road_m, status, road_class, road_name, road_ref, gmaps_road, osm_road

``status`` is ``crossing`` (a road crosses the 500m square), ``nearest_only``
(snapped to the closest road in the bbox), or ``no_roads`` (none in the bbox).

NOTE: this hits the same public Overpass servers as ``build_landmarks.py`` — do
not run both at once or they rate-limit each other.
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# Reuse the landmark module's Overpass endpoints/UA + link helpers (read-only).
from src.data_processing.landmarks import (
    OVERPASS_MIRRORS, HEADERS, UTM_EPSG, gmaps_pin, osm_map_url,
)

# Highway classes. "Major" = the roads people actually name and drive; minor are
# the tracks/service roads we fall back to when nothing bigger is in reach.
MAJOR_CLASSES = {"trunk", "primary", "secondary", "tertiary", "unclassified"}
MAJOR = "trunk|primary|secondary|tertiary|unclassified"
MINOR = "residential|living_street|service|track"
DRIVABLE = f"{MAJOR}|{MINOR}"

CACHE_COLUMNS = [
    "grid_id", "5km_id", "centroid_lat", "centroid_lon",
    "road_lat", "road_lon", "off_road_m", "status",
    "road_class", "road_name", "road_ref", "gmaps_road", "osm_road",
]


# ---------------------------------------------------------------------------
# Overpass
# ---------------------------------------------------------------------------

def _post_overpass(query: str, timeout: int = 40, max_retries: int = 1) -> list[dict]:
    """POST an Overpass query, rotating across mirrors; raise if all fail."""
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        for url in OVERPASS_MIRRORS:
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=timeout)
                r.raise_for_status()
                return r.json().get("elements", [])
            except Exception as ex:  # noqa: BLE001 - try the next mirror
                last_err = ex
                continue
        time.sleep(0.5)
    raise RuntimeError(f"all Overpass mirrors failed: {last_err}")


def query_roads_bbox(south, west, north, east, highway: str = DRIVABLE,
                     timeout: int = 40) -> list[dict]:
    """Drivable road ways whose geometry falls in the bbox (with full geometry)."""
    q = (f'[out:json][timeout:{timeout}];'
         f'way["highway"~"{highway}"]({south},{west},{north},{east});'
         f'out geom tags;')
    return _post_overpass(q, timeout=timeout)


def roads_to_gdf(elements: list[dict]) -> gpd.GeoDataFrame:
    """Rebuild road ways as LineStrings (EPSG:4326) with class/name/ref tags."""
    rows = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        line = LineString([(p["lon"], p["lat"]) for p in geom])
        tags = el.get("tags", {})
        hw = tags.get("highway")
        rows.append({"geometry": line, "highway": hw,
                     "name": tags.get("name"), "ref": tags.get("ref"),
                     "is_major": hw in MAJOR_CLASSES})
    cols = ["geometry", "highway", "name", "ref", "is_major"]
    if not rows:
        return gpd.GeoDataFrame(columns=cols, crs=4326)
    return gpd.GeoDataFrame(rows, crs=4326)


# ---------------------------------------------------------------------------
# snap one sub-cell
# ---------------------------------------------------------------------------

def _prefer_major(cands: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only major roads if any are present; otherwise return as-is."""
    major = cands[cands["is_major"]]
    return major if len(major) else cands


def snap_one(square_utm, centroid_utm: Point, roads_utm: gpd.GeoDataFrame) -> dict:
    """Snap one 500m sub-cell to the nearest (preferably major) road point.

    Roads that cross the square win; among the candidate set, major roads win.
    The snap is the point on the chosen roads nearest the centroid.
    """
    if len(roads_utm) == 0:
        return {"status": "no_roads"}

    crossing = roads_utm[roads_utm.intersects(square_utm)]
    if len(crossing):
        target, status = _prefer_major(crossing), "crossing"
    else:
        target, status = _prefer_major(roads_utm), "nearest_only"

    access = nearest_points(centroid_utm, target.union_all())[1]
    dists = target.geometry.distance(access)
    road = target.loc[dists.idxmin()]
    return {
        "status": status,
        "access": access,
        "off_road_m": round(centroid_utm.distance(access), 1),
        "road_class": road["highway"],
        "road_name": road.get("name"),
        "road_ref": road.get("ref"),
    }


# ---------------------------------------------------------------------------
# batch build + cache
# ---------------------------------------------------------------------------

def build_subcell_road_snaps(
    selected: gpd.GeoDataFrame,
    cache_path: Path,
    *,
    force: bool = False,
    margin_m: int = 1500,
    pause: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build (and cache) the nearest-road-access point for every selected sub-cell.

    One Overpass query per 5km cell (roads within the cell bbox + ``margin_m``),
    reused across its sub-cells. Resumable: sub-cells already in ``cache_path``
    are skipped unless ``force``. Saved incrementally after each cell.
    """
    sub = selected.copy()
    if sub.crs is None or sub.crs.to_epsg() != UTM_EPSG:
        sub = sub.to_crs(UTM_EPSG)
    sub_wgs = sub.to_crs(4326)

    existing = pd.DataFrame(columns=CACHE_COLUMNS)
    done_ids: set[str] = set()
    if cache_path.exists() and not force:
        existing = pd.read_csv(cache_path)
        done_ids = set(existing["grid_id"].astype(str))
        if verbose:
            print(f"Cache has {len(done_ids)} sub-cells already; resuming")

    margin_deg = margin_m / 111_000.0  # rough deg-per-metre for the bbox pad
    new_rows: list[dict] = []
    groups = [(k, g) for k, g in sub.groupby(sub["5km_id"].fillna(-1).astype(int))]
    for ci, (km5, cell) in enumerate(groups, 1):
        pending = cell[~cell["grid_id"].astype(str).isin(done_ids)]
        if len(pending) == 0:
            continue

        minx, miny, maxx, maxy = sub_wgs.loc[cell.index].total_bounds
        try:
            els = query_roads_bbox(miny - margin_deg, minx - margin_deg,
                                   maxy + margin_deg, maxx + margin_deg)
        except Exception as ex:  # noqa: BLE001
            if verbose:
                print(f"  [{ci}/{len(groups)}] 5km {km5}: ROAD QUERY FAILED ({ex})")
            continue
        roads_utm = roads_to_gdf(els)
        if len(roads_utm):
            roads_utm = roads_utm.to_crs(UTM_EPSG)

        n_snapped = 0
        for _, r in pending.iterrows():
            gid = str(r["grid_id"])
            id5 = int(r["5km_id"]) if pd.notna(r.get("5km_id")) else None
            clat = round(float(r["latitude"]), 6) if pd.notna(r.get("latitude")) else None
            clon = round(float(r["longitude"]), 6) if pd.notna(r.get("longitude")) else None
            cen_utm = r.geometry.centroid
            res = snap_one(r.geometry, cen_utm, roads_utm)

            if "access" not in res:
                new_rows.append({
                    "grid_id": gid, "5km_id": id5,
                    "centroid_lat": clat, "centroid_lon": clon,
                    "road_lat": None, "road_lon": None, "off_road_m": None,
                    "status": res.get("status", "no_roads"),
                    "road_class": None, "road_name": None, "road_ref": None,
                    "gmaps_road": None, "osm_road": None,
                })
            else:
                acc = gpd.GeoSeries([res["access"]], crs=UTM_EPSG).to_crs(4326).iloc[0]
                rla, rlo = round(acc.y, 6), round(acc.x, 6)
                new_rows.append({
                    "grid_id": gid, "5km_id": id5,
                    "centroid_lat": clat, "centroid_lon": clon,
                    "road_lat": rla, "road_lon": rlo, "off_road_m": res["off_road_m"],
                    "status": res["status"], "road_class": res["road_class"],
                    "road_name": res.get("road_name"), "road_ref": res.get("road_ref"),
                    "gmaps_road": gmaps_pin(rla, rlo), "osm_road": osm_map_url(rla, rlo),
                })
                n_snapped += 1

        if verbose:
            print(f"  [{ci}/{len(groups)}] 5km {km5}: {n_snapped}/{len(pending)} snapped "
                  f"({len(roads_utm)} roads in bbox)")
        _save(existing, new_rows, cache_path)
        time.sleep(pause)

    result = _save(existing, new_rows, cache_path)
    if verbose:
        ok = result["road_lat"].notna().sum() if len(result) else 0
        print(f"\nRoad-snap cache: {len(result)} sub-cells ({ok} with a road point) -> {cache_path}")
    return result


def _save(existing: pd.DataFrame, new_rows: list[dict], cache_path: Path) -> pd.DataFrame:
    frames = [existing] if len(existing) else []
    if new_rows:
        frames.append(pd.DataFrame(new_rows, columns=CACHE_COLUMNS))
    combined = (pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame(columns=CACHE_COLUMNS))
    combined = combined.drop_duplicates(subset=["grid_id"], keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    return combined


def load_road_snap_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load the cached road-snap table, or None if it doesn't exist."""
    if cache_path.exists():
        return pd.read_csv(cache_path)
    return None
