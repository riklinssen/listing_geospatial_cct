"""Find recognisable landmarks near each 500m sub-cell, ranked nearest-first.

Field staff in Tanzania navigate by named places — villages, primary schools,
churches, dispensaries, markets — not by the statistical centroid of a 500m cell
(which usually sits in an empty field with no routable road). This module pulls
those landmarks from OpenStreetMap via the Overpass API and ranks them by their
straight-line distance from the *edge* of the sub-cell, with a compass bearing
from the cell, so an enumerator can be told "the school at Isaka village, ~25 m
west of the cell".

Because the public Overpass servers are slow/flaky (frequent 504s and read
timeouts), queries rotate across mirrors with retry/backoff, and results are
**cached to a CSV** so map/info-sheet generation stays offline after the first
run.

Typical use (see scripts/build_landmarks.py):

    from src.data_processing.landmarks import build_subcell_landmarks
    df = build_subcell_landmarks(selected_subcells, cache_path)

Output (long format, one row per sub-cell x landmark, ranked):
    grid_id, 5km_id, rank, name, category, osm_kind, dist_m, direction,
    bearing_deg, inside, latitude, longitude, gmaps_url
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

UTM_EPSG = 32736  # UTM 36S — metric CRS for distances/bearings in the study area

# Public Overpass endpoints, tried in order. The main overpass-api.de instance
# 504s often for this region; the mirrors below are usually more reliable.
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
# A descriptive User-Agent with a contact is required by OSM usage policy;
# the default python-requests UA gets 406'd by overpass-api.de.
HEADERS = {"User-Agent": "listing-geospatial-cct/1.0 (rlinssen@laterite.com)"}

# Landmark taxonomy. Each entry is an Overpass tag filter plus the human-facing
# category and a priority (lower = more useful as a nav anchor) used only to
# break ties between landmarks at (almost) the same distance.
LANDMARK_TAGS = [
    # settlements — the primary anchor in rural TZ
    ('node["place"~"town|village|hamlet|isolated_dwelling|locality"]', "settlement", 1),
    # schools — the strongest man-made rural landmark
    ('node["amenity"="school"]', "school", 2),
    ('way["amenity"="school"]', "school", 2),
    # worship — churches/mosques are highly recognisable
    ('node["amenity"="place_of_worship"]', "worship", 3),
    ('way["amenity"="place_of_worship"]', "worship", 3),
    # health
    ('node["amenity"~"hospital|clinic|health_post|dispensary|doctors"]', "health", 4),
    ('way["amenity"~"hospital|clinic|dispensary"]', "health", 4),
    # markets / shops
    ('node["amenity"="marketplace"]', "market", 5),
    ('way["amenity"="marketplace"]', "market", 5),
    ('node["shop"~"supermarket|general|convenience"]', "shop", 6),
    # civic
    ('node["amenity"~"police|townhall|community_centre|fuel|pharmacy"]', "civic", 7),
    # water points (people orient by boreholes/wells)
    ('node["man_made"~"water_well|water_tower|borehole"]', "water", 8),
    ('node["amenity"="drinking_water"]', "water", 8),
]

_CATEGORY_LABEL = {
    "settlement": "Village / settlement",
    "school": "School",
    "worship": "Church / mosque",
    "health": "Health facility",
    "market": "Market",
    "shop": "Shop",
    "civic": "Civic / fuel",
    "water": "Water point",
}

CACHE_COLUMNS = [
    "grid_id", "5km_id", "rank", "name", "category", "osm_kind",
    "dist_m", "direction", "bearing_deg", "inside",
    "latitude", "longitude", "gmaps_url", "osm_url", "osm_map_url",
    "osm_type", "osm_id",
]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _compass(dx: float, dy: float) -> tuple[str, float]:
    """8-point compass label + bearing (deg) for an east/north offset (metres)."""
    bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((bearing + 22.5) % 360 // 45)], round(bearing, 1)


def gmaps_pin(lat: float, lon: float) -> str:
    """A Google Maps 'drop a pin' link for a lat/lon."""
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def osm_element_url(osm_type: str, osm_id) -> str:
    """Link to the OSM object page (shows its full tags — the 'address' view)."""
    if osm_type and osm_id is not None:
        return f"https://www.openstreetmap.org/{osm_type}/{int(osm_id)}"
    return ""


def osm_map_url(lat: float, lon: float, zoom: int = 17) -> str:
    """Link to the OSM slippy map with a marker at lat/lon (no element needed)."""
    return (f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}"
            f"#map={zoom}/{lat:.6f}/{lon:.6f}")


# ---------------------------------------------------------------------------
# Overpass querying (with mirror rotation + retry)
# ---------------------------------------------------------------------------

def _build_query(lat: float, lon: float, radius_m: int, server_timeout: int) -> str:
    """Assemble one Overpass QL query for all landmark tags around a point."""
    parts = [
        f'  {sel}(around:{radius_m},{lat:.6f},{lon:.6f});'
        for sel, _cat, _prio in LANDMARK_TAGS
    ]
    body = "\n".join(parts)
    return f"[out:json][timeout:{server_timeout}];\n(\n{body}\n);\nout center tags;"


def query_overpass(
    lat: float,
    lon: float,
    radius_m: int,
    *,
    timeout: int = 30,
    max_retries: int = 1,
    pause: float = 0.5,
) -> list[dict]:
    """Run one landmark query, rotating across mirrors on failure.

    Uses a short client timeout so a stuck mirror is abandoned quickly (the
    public servers frequently hang); the first responsive mirror wins. Returns
    the raw Overpass ``elements`` list. Raises RuntimeError only if every mirror
    fails across all retries.
    """
    q = _build_query(lat, lon, radius_m, server_timeout=timeout)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        for url in OVERPASS_MIRRORS:
            try:
                r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=timeout)
                r.raise_for_status()
                return r.json().get("elements", [])
            except Exception as ex:  # noqa: BLE001 - want to try the next mirror
                last_err = ex
                continue
        time.sleep(pause)
    raise RuntimeError(f"all Overpass mirrors failed: {last_err}")


def landmarks_around(lat: float, lon: float, radius_m: int) -> list[dict]:
    """Fetch + classify named landmarks around a point (one Overpass call)."""
    return _elements_to_landmarks(query_overpass(lat, lon, radius_m))


def _elements_to_landmarks(elements: list[dict]) -> list[dict]:
    """Keep named elements, resolve a point (node coord or way center), tag category."""
    # map an element's tags to (category, priority) using the first matching family
    out = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        if el.get("type") == "node":
            plon, plat = el.get("lon"), el.get("lat")
        else:
            c = el.get("center", {})
            plon, plat = c.get("lon"), c.get("lat")
        if plon is None or plat is None:
            continue

        category, osm_kind = _classify(tags)
        out.append({
            "name": str(name).strip(),
            "category": category,
            "osm_kind": osm_kind,
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "lat": plat,
            "lon": plon,
        })
    return out


def _classify(tags: dict) -> tuple[str, str]:
    """Return (category, osm_kind) for an element's tags."""
    if "place" in tags:
        return "settlement", f"place={tags['place']}"
    amen = tags.get("amenity")
    if amen == "school":
        return "school", "amenity=school"
    if amen == "place_of_worship":
        denom = tags.get("religion") or tags.get("denomination")
        return "worship", f"place_of_worship{('/' + denom) if denom else ''}"
    if amen in {"hospital", "clinic", "health_post", "dispensary", "doctors"}:
        return "health", f"amenity={amen}"
    if amen == "marketplace":
        return "market", "amenity=marketplace"
    if amen in {"police", "townhall", "community_centre", "fuel", "pharmacy"}:
        return "civic", f"amenity={amen}"
    if amen == "drinking_water":
        return "water", "amenity=drinking_water"
    if "shop" in tags:
        return "shop", f"shop={tags['shop']}"
    if "man_made" in tags:
        return "water", f"man_made={tags['man_made']}"
    return "other", "?"


# ---------------------------------------------------------------------------
# ranking for one sub-cell
# ---------------------------------------------------------------------------

_PRIORITY = {
    "settlement": 1, "school": 2, "worship": 3, "health": 4,
    "market": 5, "shop": 6, "civic": 7, "water": 8, "other": 9,
}


def rank_from_landmarks(
    square_utm,
    centroid_lonlat: tuple[float, float],
    landmarks: list[dict],
    *,
    max_landmarks: int = 8,
    to_utm=None,
) -> list[dict]:
    """Rank a *pre-fetched* landmark list against one sub-cell, nearest-first.

    Pure geometry — no network. ``landmarks`` is a list of dicts with at least
    name/category/osm_kind/lat/lon (as returned by :func:`landmarks_around`).

    Distances are from the sub-cell **edge** (0 if inside). ``.boundary`` is used
    rather than ``.exterior`` so clipped MultiPolygon sub-cells work too.
    """
    if to_utm is None:
        from pyproj import Transformer
        to_utm = Transformer.from_crs(4326, UTM_EPSG, always_xy=True).transform

    lon, lat = centroid_lonlat
    cx, cy = to_utm(lon, lat)
    boundary = square_utm.boundary  # works for Polygon and MultiPolygon

    ranked = []
    seen = set()
    for lm in landmarks:
        key = (lm["name"].lower(), lm["category"])  # de-dup node+way pairs
        if key in seen:
            continue
        seen.add(key)

        px, py = to_utm(lm["lon"], lm["lat"])
        pt = Point(px, py)
        inside = square_utm.contains(pt)
        dist_edge = 0.0 if inside else boundary.distance(pt)
        direction, bearing = _compass(px - cx, py - cy)
        ranked.append({
            **lm,
            "dist_m": round(dist_edge),
            "direction": direction,
            "bearing_deg": bearing,
            "inside": inside,
            "gmaps_url": gmaps_pin(lm["lat"], lm["lon"]),
            "osm_url": osm_element_url(lm.get("osm_type"), lm.get("osm_id")),
            "osm_map_url": osm_map_url(lm["lat"], lm["lon"]),
        })

    # nearest first; break ties by category priority then name
    ranked.sort(key=lambda r: (r["dist_m"], _PRIORITY.get(r["category"], 9), r["name"]))
    return ranked[:max_landmarks]


def rank_landmarks_for_subcell(
    square_utm,
    centroid_lonlat: tuple[float, float],
    *,
    radius_m: int = 10000,
    max_landmarks: int = 8,
    to_utm=None,
) -> list[dict]:
    """Fetch (one Overpass call) + rank landmarks near one sub-cell.

    Convenience wrapper for single-cell use (e.g. the notebook's spot checks).
    For batches, prefer :func:`build_subcell_landmarks`, which queries once per
    5km cell and reuses the result across its sub-cells.
    """
    lon, lat = centroid_lonlat
    lms = landmarks_around(lat, lon, radius_m)
    return rank_from_landmarks(square_utm, centroid_lonlat, lms,
                               max_landmarks=max_landmarks, to_utm=to_utm)


# ---------------------------------------------------------------------------
# batch build + cache
# ---------------------------------------------------------------------------

def build_subcell_landmarks(
    selected: gpd.GeoDataFrame,
    cache_path: Path,
    *,
    force: bool = False,
    max_landmarks: int = 8,
    radius_m: int = 10000,
    pause: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build (and cache) the ranked landmark table for every selected sub-cell.

    **One Overpass query per 5km cell**, reused across all its sub-cells: the
    sub-cells within a 5km cell sit at most a few km apart, so a single query at
    ``radius_m`` around the cell centre covers them all. This is ~2x fewer calls
    than querying per sub-cell, and the public Overpass servers are the
    bottleneck.

    Resumable: 5km cells whose sub-cells are already in ``cache_path`` are
    skipped unless ``force`` is set, so an interrupted run (Overpass is flaky)
    can be re-run to fill in the rest. Saved incrementally after each cell.

    Args:
        selected: selected 500m sub-cells (needs geometry, ``grid_id``,
            ``5km_id``, ``latitude``, ``longitude``).
        cache_path: CSV to read/write the long-format ranked table.
        force: re-query every cell, ignoring the cache.
        max_landmarks: max landmarks kept per sub-cell.
        radius_m: search radius (m) around each 5km cell centre.
        pause: seconds to sleep between cells (Overpass politeness).

    Returns:
        The full landmark table (all cached sub-cells) as a DataFrame.
    """
    from pyproj import Transformer
    to_utm = Transformer.from_crs(4326, UTM_EPSG, always_xy=True).transform

    sub = selected.copy()
    if sub.crs is None or sub.crs.to_epsg() != UTM_EPSG:
        sub = sub.to_crs(UTM_EPSG)

    # load any existing cache so we can resume
    existing = pd.DataFrame(columns=CACHE_COLUMNS)
    done_ids: set[str] = set()
    if cache_path.exists() and not force:
        existing = pd.read_csv(cache_path)
        done_ids = set(existing["grid_id"].astype(str))
        if verbose:
            print(f"Cache has {existing['grid_id'].nunique()} sub-cells already; resuming")

    new_rows: list[dict] = []
    # group sub-cells by their parent 5km cell → one query per group
    groups = [(k, g) for k, g in sub.groupby(sub["5km_id"].fillna(-1).astype(int))]
    for ci, (km5, cell_sub) in enumerate(groups, 1):
        pending = cell_sub[~cell_sub["grid_id"].astype(str).isin(done_ids)]
        if len(pending) == 0:
            continue

        # query centre = mean of this cell's sub-cell centroids
        qlat = cell_sub["latitude"].astype(float).mean()
        qlon = cell_sub["longitude"].astype(float).mean()
        try:
            cell_lms = landmarks_around(qlat, qlon, radius_m)
        except Exception as ex:  # noqa: BLE001
            if verbose:
                print(f"  [{ci}/{len(groups)}] 5km {km5}: QUERY FAILED ({ex})")
            continue

        for _, row in pending.iterrows():
            gid = str(row.get("grid_id"))
            lon, lat = row.get("longitude"), row.get("latitude")
            if pd.isna(lon) or pd.isna(lat):
                continue
            ranked = rank_from_landmarks(
                row.geometry, (lon, lat), cell_lms,
                max_landmarks=max_landmarks, to_utm=to_utm,
            )
            for rank, lm in enumerate(ranked, 1):
                new_rows.append({
                    "grid_id": gid,
                    "5km_id": int(row["5km_id"]) if pd.notna(row.get("5km_id")) else None,
                    "rank": rank,
                    "name": lm["name"],
                    "category": lm["category"],
                    "osm_kind": lm["osm_kind"],
                    "dist_m": lm["dist_m"],
                    "direction": lm["direction"],
                    "bearing_deg": lm["bearing_deg"],
                    "inside": lm["inside"],
                    "latitude": round(lm["lat"], 6),
                    "longitude": round(lm["lon"], 6),
                    "gmaps_url": lm["gmaps_url"],
                    "osm_url": lm["osm_url"],
                    "osm_map_url": lm["osm_map_url"],
                    "osm_type": lm.get("osm_type"),
                    "osm_id": lm.get("osm_id"),
                })
        if verbose:
            n_named = len({(l["name"], l["category"]) for l in cell_lms})
            print(f"  [{ci}/{len(groups)}] 5km {km5}: {len(pending)} sub-cells · "
                  f"{n_named} named landmarks in {radius_m//1000} km")

        _save(existing, new_rows, cache_path)  # incremental — keep progress if it dies
        time.sleep(pause)

    result = _save(existing, new_rows, cache_path)
    if verbose:
        n_cells = result["grid_id"].nunique() if len(result) else 0
        print(f"\nLandmark cache: {len(result)} rows across {n_cells} sub-cells "
              f"-> {cache_path}")
    return result


def _save(existing: pd.DataFrame, new_rows: list[dict], cache_path: Path) -> pd.DataFrame:
    """Merge new rows into the existing cache and write it out."""
    frames = [existing] if len(existing) else []
    if new_rows:
        frames.append(pd.DataFrame(new_rows, columns=CACHE_COLUMNS))
    combined = (pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame(columns=CACHE_COLUMNS))
    combined = combined.drop_duplicates(subset=["grid_id", "name", "category"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    return combined


def load_landmark_cache(cache_path: Path) -> pd.DataFrame | None:
    """Load the cached landmark table, or None if it doesn't exist."""
    if cache_path.exists():
        return pd.read_csv(cache_path)
    return None


def category_label(category: str) -> str:
    """Human-facing label for a landmark category."""
    return _CATEGORY_LABEL.get(category, category.title())
