"""Generate ONE self-contained HTML dashboard of every cell + its landmarks.

A single browsable file for a coordinator (not the field tablet): each 5km grid
cell expands to its selected 500m sub-cells, and each sub-cell lists its
recognisable landmarks nearest-first (distance from the 500m cell edge + compass
direction) with Google Maps and OpenStreetMap links. Includes a search box and
region/status filters so the whole sample is navigable from one page.

Reads:
    control_grid_5km_flagged.gpkg      (5km cells + sample_status)
    selected_subcells_500m.gpkg        (500m sub-cells + admin + coords)
    subcell_landmarks.csv              (from scripts/build_landmarks.py)

Writes:
    <output_dir>/listing_maps/landmark_dashboard.html   (override with --output)

Only cells present in the landmark cache are shown, so a partial cache still
produces a usable dashboard.

Usage:
    python scripts/generate_landmark_dashboard.py
    python scripts/generate_landmark_dashboard.py --landmarks notebooks/_landmark_cache_test.csv \
        --output notebooks/landmark_dashboard_preview.html
"""

import argparse
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from shapely.geometry import mapping, Point

from src.utils.config_loader import load_config, get_data_dir, get_output_dir
from src.data_processing.load_boundaries import load_control_grid, load_selected_subcells
from src.data_processing.landmarks import category_label

# Marker colours by landmark category (match the notebook map)
CAT_COLOR = {"settlement": "#2563eb", "school": "#c026d3", "worship": "#7c3aed",
             "health": "#dc2626", "market": "#ea580c", "shop": "#ca8a04",
             "civic": "#0891b2", "water": "#0d9488", "other": "#6b7280"}

# Vendored Leaflet (assets/vendor) so the guide is one self-contained file with no
# CDN dependency; only live map tiles need internet. Falls back to CDN if absent.
_VENDOR = PROJECT_ROOT / "assets" / "vendor"
try:
    _LEAFLET_CSS = (_VENDOR / "leaflet-1.9.4.css").read_text(encoding="utf-8")
    _LEAFLET_JS = (_VENDOR / "leaflet-1.9.4.js").read_text(encoding="utf-8")
    LEAFLET_HEAD = f"<style>{_LEAFLET_CSS}</style>\n<script>{_LEAFLET_JS}</script>"
except FileNotFoundError:
    LEAFLET_HEAD = ('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">\n'
                    '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>')


def parse_args():
    p = argparse.ArgumentParser(description="Build the one-file landmark dashboard.")
    p.add_argument("--landmarks", default=None,
                   help="Landmark cache CSV (default: <data_dir>/.../subcell_landmarks.csv).")
    p.add_argument("--road-snaps", default=None,
                   help="Road-snap cache CSV (default: auto-load <data_dir>/.../subcell_road_snaps.csv if present).")
    p.add_argument("--output", default=None,
                   help="Output HTML path (default: <output_dir>/listing_maps/landmark_dashboard.html).")
    return p.parse_args()


def esc(x) -> str:
    return html.escape("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))


def dist_label(m) -> str:
    m = 0 if pd.isna(m) else int(m)
    if m == 0:
        return "in 500m cell"
    return f"{m} m" if m < 1000 else f"{m / 1000:.2f} km"


def gmaps(lat, lon) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def gdir(lat, lon) -> str:
    """Google Maps directions to a destination (origin filled in by the page JS)."""
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}&travelmode=driving"


def odir(lat, lon) -> str:
    """OpenStreetMap directions to a destination (origin filled in by the page JS).

    OSM's routing often has better coverage on rural TZ tracks than Google.
    """
    return (f"https://www.openstreetmap.org/directions?"
            f"to={lat:.6f}%2C{lon:.6f}&engine=fossgis_osrm_car")


def opin(lat, lon) -> str:
    """OpenStreetMap marker link at lat/lon."""
    return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f}"


def sanitize(name) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", str(name)).strip("_")


def road_label(rr) -> str:
    """Human label for the snapped road: class + name/ref when present."""
    cls = str(rr.get("road_class") or "road")
    nm, ref = rr.get("road_name"), rr.get("road_ref")
    label = cls
    if isinstance(nm, str) and nm:
        label += f" · {esc(nm)}"
    if isinstance(ref, str) and ref:
        label += f" ({esc(ref)})"
    return label


def road_access_line(rr, km5) -> str:
    """The 'nearest road access' line for one sub-cell (copyable coord + links)."""
    rlat, rlon = rr.get("road_lat"), rr.get("road_lon")
    if rlat is None or pd.isna(rlat) or rlon is None or pd.isna(rlon):
        return ("<div class='road-row none'><span class='clabel road'>nearest road access</span>"
                "<span class='roadmeta'>no mapped road within reach — use the landmarks above</span></div>")
    rc = f"{rlat:.6f}, {rlon:.6f}"
    off = rr.get("off_road_m")
    offtxt = f"{int(round(off))} m walk to cell" if pd.notna(off) else ""
    status = str(rr.get("status") or "")
    return f"""<div class="road-row">
          <span class="clabel road">nearest road access</span>
          <button class="mapbtn" title="Show on map" onclick="focusRoad('{km5}','{rlat:.6f},{rlon:.6f}')">◎</button>
          <code class="coord" title="click to select">{rc}</code>
          <button class="copy" data-coord="{rc}" onclick="copyCoord(this)">⧉ Copy</button>
          <span class="roadmeta">{esc(offtxt)} · {road_label(rr)}</span>
          <span class="rstatus rs-{esc(status)}">{esc(status)}</span>
          <span class="lbl">open&nbsp;pin:</span>
          <a href="{esc(gmaps(rlat, rlon))}" target="_blank">Google ↗</a>
          <a href="{esc(opin(rlat, rlon))}" target="_blank">OSM ↗</a>
          <span class="lbl">route&nbsp;from&nbsp;start:</span>
          <a class="dir" data-engine="g" data-dest="{rlat:.6f},{rlon:.6f}" href="{esc(gdir(rlat, rlon))}" target="_blank">Google ↗</a>
          <a class="dir" data-engine="o" data-dest="{rlat:.6f},{rlon:.6f}" href="{esc(odir(rlat, rlon))}" target="_blank">OSM ↗</a>
        </div>"""


def narrative(rows: pd.DataFrame) -> str:
    """A one-line summary of the nearest landmarks (NOT a route — distance order)."""
    if len(rows) == 0:
        return "No OSM landmarks nearby — use the satellite map and the VCSL village."
    bits = []
    for _, r in rows.head(3).iterrows():
        where = "inside the 500m cell" if r["dist_m"] == 0 else f"{dist_label(r['dist_m'])} {r['direction']}"
        flag = "" if r.get("in_5km", True) else " <span class='outside'>⚠ outside 5km cell</span>"
        bits.append(f"<b>{esc(r['name'])}</b> ({esc(category_label(r['category']))}, {where}){flag}")
    return "Look for these nearby landmarks (closest first): " + "; ".join(bits) + "."


def landmark_table(rows: pd.DataFrame, km5) -> str:
    if len(rows) == 0:
        return "<div class='empty'>— no OSM landmarks found within 10 km —</div>"
    trs = []
    for _, r in rows.sort_values("rank").iterrows():
        la, lo = r["latitude"], r["longitude"]
        attrs = (f"<a href='{esc(r['osm_url'])}' target='_blank'>attrs&nbsp;↗</a>"
                 if isinstance(r.get("osm_url"), str) and r["osm_url"] else "—")
        inside = " class='inside'" if r["dist_m"] == 0 else ""
        dest = f"{la:.6f},{lo:.6f}"
        in5 = r.get("in_5km", True)
        flag5 = ("<span class='in5 yes'>✓</span>" if in5
                 else "<span class='in5 no'>⚠ outside</span>")
        trs.append(
            f"<tr{inside}>"
            f"<td class='num'>{int(r['rank'])}</td>"
            f"<td class='dist'>{esc(dist_label(r['dist_m']))}</td>"
            f"<td>{esc(r['direction'])}</td>"
            f"<td class='c5'>{flag5}</td>"
            f"<td><span class='cat cat-{esc(r['category'])}'>{esc(category_label(r['category']))}</span></td>"
            f"<td class='name'><button class='mapbtn' title='Show on map' "
            f"onclick=\"focusLandmark('{km5}','{la:.6f},{lo:.6f}')\">◎</button> {esc(r['name'])}</td>"
            f"<td class='mono'>{la:.5f}, {lo:.5f}</td>"
            f"<td class='links'>"
            f"<a class='dir' data-engine='g' data-dest='{dest}' href='{esc(gdir(la, lo))}' target='_blank'>Google&nbsp;↗</a>"
            f"<a class='dir' data-engine='o' data-dest='{dest}' href='{esc(odir(la, lo))}' target='_blank'>OSM&nbsp;↗</a></td>"
            f"<td class='links'>"
            f"<a href='{esc(gmaps(la, lo))}' target='_blank'>Google&nbsp;↗</a>"
            f"<a href='{esc(opin(la, lo))}' target='_blank'>OSM&nbsp;↗</a></td>"
            f"<td>{attrs}</td>"
            f"</tr>"
        )
    return ("<table class='lm'><thead><tr><th>#</th><th>distance</th><th>dir</th>"
            "<th>in&nbsp;5km</th><th>type</th><th>landmark</th><th>coordinates</th>"
            "<th>route&nbsp;from&nbsp;start</th><th>pin</th>"
            "<th>OSM</th></tr></thead><tbody>" + "".join(trs) + "</tbody></table>")


def _cell_map_data(cell_geom, cell_sub, lmc, road_by_gid=None) -> dict:
    """Build the GeoJSON + landmark payload embedded for one 5km cell's Leaflet map.

    Everything is WGS84. ``cell_geom`` is the 5km boundary; ``cell_sub`` the 500m
    squares (with lat/lon + role); ``lmc`` the landmark rows for this cell.
    Landmarks are de-duplicated to unique places, each carrying its distance +
    direction to every sub-cell it was ranked for (straight from the cache).
    """
    # 500m squares as a FeatureCollection (role + a popup with route-to-cell links)
    sq_features = []
    for _, r in cell_sub.iterrows():
        la, lo = r["latitude"], r["longitude"]
        popup = (f"<b>500m sub-cell {esc(r['grid_id'])}</b> ({esc(r['selection_role'])})<br>"
                 f"{int(r.get('building_count') or 0)} buildings<br>"
                 f"route to cell: <a class='dir' data-engine='g' data-dest='{la:.6f},{lo:.6f}' "
                 f"href='{gdir(la, lo)}' target='_blank'>Google</a> · "
                 f"<a class='dir' data-engine='o' data-dest='{la:.6f},{lo:.6f}' "
                 f"href='{odir(la, lo)}' target='_blank'>OSM</a>")
        sq_features.append({"type": "Feature",
                            "properties": {"grid_id": str(r["grid_id"]),
                                           "role": str(r["selection_role"]), "popup": popup},
                            "geometry": mapping(r.geometry)})

    # de-dup landmarks, gather per-sub-cell distance/direction from the cache
    landmarks = []
    seen = {}
    for _, r in lmc.sort_values("dist_m").iterrows():
        key = (str(r["name"]).lower(), round(r["latitude"], 5), round(r["longitude"], 5))
        if key not in seen:
            seen[key] = {"name": str(r["name"]), "category": str(r["category"]),
                         "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                         "osm_url": r["osm_url"] if isinstance(r.get("osm_url"), str) else "",
                         "in_5km": bool(r.get("in_5km", True)), "per_sub": []}
        seen[key]["per_sub"].append((str(r["grid_id"]), int(r["dist_m"]), str(r["direction"])))

    for lmk in seen.values():
        la, lo = lmk["lat"], lmk["lon"]
        outside = not lmk["in_5km"]
        per = "".join(
            f"<div class='pd'>{esc(gid)}: <b>{esc(dist_label(d))}</b> {esc(dr)}</div>"
            for gid, d, dr in lmk["per_sub"])
        attrs = (f" · <a href='{esc(lmk['osm_url'])}' target='_blank'>attrs ↗</a>"
                 if lmk["osm_url"] else "")
        warn = ("<div class='pwarn'>⚠ outside this 5km grid cell</div>" if outside else "")
        popup = (f"<b>{esc(lmk['name'])}</b><br><span class='pc'>{esc(category_label(lmk['category']))}</span>"
                 f"{warn}"
                 f"<div class='pdh'>distance from each 500m cell:</div>{per}"
                 f"<div class='pl'>route: "
                 f"<a class='dir' data-engine='g' data-dest='{la:.6f},{lo:.6f}' href='{gdir(la, lo)}' target='_blank'>Google</a> · "
                 f"<a class='dir' data-engine='o' data-dest='{la:.6f},{lo:.6f}' href='{odir(la, lo)}' target='_blank'>OSM</a><br>"
                 f"pin: <a href='{gmaps(la, lo)}' target='_blank'>Google</a> · "
                 f"<a href='{opin(la, lo)}' target='_blank'>OSM</a>{attrs}</div>")
        landmarks.append({"lat": la, "lon": lo, "name": lmk["name"], "outside": outside,
                          "color": CAT_COLOR.get(lmk["category"], "#6b7280"), "popup": popup})

    # road-access points (one per sub-cell that has a snap) + connector to centroid
    roads = []
    if road_by_gid:
        for _, r in cell_sub.iterrows():
            rr = road_by_gid.get(str(r["grid_id"]))
            if rr is None:
                continue
            rlat, rlon = rr.get("road_lat"), rr.get("road_lon")
            if rlat is None or pd.isna(rlat) or rlon is None or pd.isna(rlon):
                continue
            clat = rr.get("centroid_lat"); clon = rr.get("centroid_lon")
            if pd.isna(clat) or pd.isna(clon):
                clat, clon = r["latitude"], r["longitude"]
            off = rr.get("off_road_m")
            offtxt = f"{int(round(off))} m to cell" if pd.notna(off) else ""
            popup = (f"<b>Road access</b> — {esc(str(r['grid_id']))}<br>"
                     f"<span class='pc'>{esc(offtxt)} · {road_label(rr)}</span>"
                     f"<div class='pl'>route: "
                     f"<a class='dir' data-engine='g' data-dest='{rlat:.6f},{rlon:.6f}' href='{gdir(rlat, rlon)}' target='_blank'>Google</a> · "
                     f"<a class='dir' data-engine='o' data-dest='{rlat:.6f},{rlon:.6f}' href='{odir(rlat, rlon)}' target='_blank'>OSM</a><br>"
                     f"pin: <a href='{gmaps(rlat, rlon)}' target='_blank'>Google</a> · "
                     f"<a href='{opin(rlat, rlon)}' target='_blank'>OSM</a></div>")
            roads.append({"rlat": float(rlat), "rlon": float(rlon),
                          "clat": float(clat), "clon": float(clon),
                          "off": offtxt, "popup": popup})

    return {"cell5k": mapping(cell_geom),
            "squares": {"type": "FeatureCollection", "features": sq_features},
            "landmarks": landmarks, "roads": roads}


def build_html(grid, selected, lm, generated_note: str, base_dir: Path = None,
               roads=None) -> str:
    # cells that actually have landmark data
    cell_ids = sorted(lm["5km_id"].dropna().astype(int).unique())
    # 5km boundary geometry per cell id (WGS84)
    grid_wgs = grid.to_crs(4326)
    cell_geoms = {int(r["id"]): r.geometry for _, r in grid_wgs.iterrows()}
    # optional road-snap rows, keyed by 500m grid_id (auto-loaded when present)
    road_by_gid = {}
    if roads is not None and len(roads):
        road_by_gid = {str(r["grid_id"]): r for _, r in roads.iterrows()}
    cards = []
    map_payload = {}   # km5 (str) -> {cell5k, squares, landmarks} for the Leaflet maps
    for km5 in cell_ids:
        cell_sub = selected[selected["5km_id"].fillna(-1).astype(int) == km5]
        if len(cell_sub) == 0:
            continue
        region = _mode(cell_sub, "region", "—")
        district = _mode(cell_sub, "district", "—")
        ward = _mode(cell_sub, "ward_name", "—")
        status = _mode(cell_sub, "sample_status", "—")

        # Pull in the existing info sheet + overview map, if generated already.
        # Same folder convention as generate_info_sheets.py:
        #   <status>/<region>_<district>_<5km_id>/{info_sheet.html, overview_5km.png}
        rel = f"{status}/{sanitize(region + '_' + district)}_{km5}"
        sheet_link = overview_img = ""
        if base_dir is not None:
            if (base_dir / rel / "info_sheet.html").exists():
                sheet_link = (f"<a class='tag sheet' href='{esc(rel)}/info_sheet.html' "
                              f"target='_blank'>full info sheet ↗</a>")
            if (base_dir / rel / "overview_5km.png").exists():
                overview_img = (
                    "<div class='overview-wrap'>"
                    "<div class='overview-lbl'>5km overview — static map</div>"
                    f"<img class='overview' loading='lazy' "
                    f"src='{esc(rel)}/overview_5km.png' alt='5km overview map'></div>")
        vcsl = "—"
        if "vcsl_village" in cell_sub.columns:
            vcsl = ", ".join(sorted(cell_sub["vcsl_village"].dropna().astype(str).unique())) or "—"
        n_build = int(cell_sub["building_count"].fillna(0).sum()) if "building_count" in cell_sub.columns else 0

        cell_geom = cell_geoms.get(km5)
        lmc = lm[lm["5km_id"].fillna(-1).astype(int) == km5].copy()
        # Flag landmarks that fall OUTSIDE the 5km grid cell boundary (the sampling
        # unit). If the boundary is missing, assume inside (don't false-flag).
        if cell_geom is not None:
            lmc["in_5km"] = [cell_geom.contains(Point(lo, la))
                             for la, lo in zip(lmc["latitude"], lmc["longitude"])]
        else:
            lmc["in_5km"] = True

        subcell_ids = list(cell_sub["grid_id"].astype(str))          # 500m ids, e.g. G_0618_1099
        landmark_names = list(lmc["name"].dropna().astype(str).unique())
        filt = " ".join(str(x) for x in
                        [km5, region, district, ward, status, vcsl]
                        + subcell_ids + landmark_names).lower()

        # map payload for this cell (incl. road-access points when available)
        map_payload[str(km5)] = _cell_map_data(cell_geom, cell_sub, lmc, road_by_gid)

        sub_blocks = []
        for _, sc in cell_sub.sort_values("selection_role").iterrows():
            gid = sc["grid_id"]
            rows = lmc[lmc["grid_id"] == gid]
            role = sc["selection_role"]
            lat, lon = sc["latitude"], sc["longitude"]
            cc = f"{lat:.6f}, {lon:.6f}"   # full-precision centroid for copy-paste
            sub_blocks.append(f"""
      <div class="subcell">
        <div class="sc-head">
          <span class="role role-{esc(role)}">{esc(role)}</span>
          <span class="sc-id">500m sub-cell {esc(gid)}</span>
          <span class="sc-meta">5km grid <code>{km5}</code> · {int(sc.get('building_count') or 0)} buildings</span>
          <button class="focusbtn" onclick="focusSub('{km5}','{esc(gid)}')">◎ Focus on map</button>
        </div>
        <div class="centroid-row">
          <span class="clabel">500m centroid</span>
          <button class="mapbtn" title="Show on map" onclick="focusCentroid('{km5}',{lat:.6f},{lon:.6f})">◎</button>
          <code class="coord" title="click to select">{cc}</code>
          <button class="copy" data-coord="{cc}" onclick="copyCoord(this)">⧉ Copy</button>
          <span class="lbl">open&nbsp;pin:</span>
          <a href="{esc(gmaps(lat, lon))}" target="_blank">Google ↗</a>
          <a href="{esc(opin(lat, lon))}" target="_blank">OSM ↗</a>
          <span class="lbl">route&nbsp;from&nbsp;start:</span>
          <a class="dir" data-engine="g" data-dest="{lat:.6f},{lon:.6f}" href="{esc(gdir(lat, lon))}" target="_blank">Google ↗</a>
          <a class="dir" data-engine="o" data-dest="{lat:.6f},{lon:.6f}" href="{esc(odir(lat, lon))}" target="_blank">OSM ↗</a>
        </div>
        {road_access_line(road_by_gid[str(gid)], km5) if str(gid) in road_by_gid else ""}
        <div class="narrative">{narrative(rows)}</div>
        {landmark_table(rows, km5)}
      </div>""")

        cards.append(f"""
    <details class="cell" data-filter="{esc(filt)}" data-status="{esc(status)}" data-region="{esc(region)}"
             ontoggle="if(this.open)initMap('{km5}')">
      <summary class="cell-head">
        <div>
          <h2>5km grid <code>{km5}</code> — {esc(ward)}</h2>
          <div class="admin">{esc(district)} · {esc(region)}</div>
        </div>
        <div class="tags">
          <span class="status status-{esc(status)}">{esc(status)}</span>
          <span class="tag">{len(cell_sub)} sub-cells</span>
          <span class="tag">{n_build} buildings</span>
          <span class="tag vcsl">VCSL: {esc(vcsl)}</span>
          {sheet_link}
        </div>
      </summary>
      <div class="cell-body">
        {overview_img}
        <div class="map-lbl">Interactive map — zoom, toggle layers, click ◎ to locate points</div>
        <div id="map_{km5}" class="cell-map"></div>
        {''.join(sub_blocks)}
      </div>
    </details>""")

    regions = sorted(selected["region"].dropna().astype(str).unique())
    region_opts = "".join(f"<option value='{esc(r)}'>{esc(r)}</option>" for r in regions)

    cells_json = json.dumps(map_payload).replace("</", "<\\/")  # safe inside <script>
    return _PAGE.replace("{{GENERATED}}", esc(generated_note)) \
                .replace("{{N_CELLS}}", str(len(cell_ids))) \
                .replace("{{N_SUB}}", str(lm['grid_id'].nunique())) \
                .replace("{{REGION_OPTS}}", region_opts) \
                .replace("{{CARDS}}", "".join(cards)) \
                .replace("{{CELLS_JSON}}", cells_json) \
                .replace("{{LEAFLET_HEAD}}", LEAFLET_HEAD)


def _mode(df, col, default):
    if col in df.columns:
        vals = df[col].dropna()
        if len(vals):
            return str(vals.mode().iloc[0])
    return default


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listing — landmark navigation guide</title>
{{LEAFLET_HEAD}}
<style>
:root{--ground:#f5f2ec;--card:#fff;--ink:#1b2a31;--muted:#5f6d74;--accent:#bd5729;
--signal:#2f7a52;--line:#e5e0d6;--chip:#f0ebe1;--mono:ui-monospace,Consolas,monospace;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media(prefers-color-scheme:dark){:root{--ground:#11171b;--card:#1a2228;--ink:#eef2f3;
--muted:#9aabb2;--accent:#e07a48;--signal:#4caf7d;--line:#2a343b;--chip:#222d34;}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);line-height:1.45}
.wrap{max-width:1080px;margin:0 auto;padding:24px 18px 80px}
h1{font-size:1.5rem;margin:0 0 4px}
.sub{color:var(--muted);font-size:.9rem;margin-bottom:18px}
.controls{position:sticky;top:0;background:var(--ground);padding:12px 0;z-index:10;
display:flex;gap:10px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:18px}
.controls input,.controls select{font:inherit;padding:8px 10px;border:1px solid var(--line);
border-radius:8px;background:var(--card);color:var(--ink)}
.controls input{flex:1;min-width:200px}
.count{color:var(--muted);font-size:.85rem;align-self:center}
.cell{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:0 0 18px;
overflow:hidden}
summary.cell-head{cursor:pointer;list-style:none}
summary.cell-head::-webkit-details-marker{display:none}
summary.cell-head::after{content:"▸ open map";margin-left:auto;align-self:center;font-size:.72rem;
font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.04em}
details[open] summary.cell-head::after{content:"▾ close"}
.cell-body{padding:0}
.cell-map{width:100%;height:460px;background:var(--chip)}
.focusbtn{font:inherit;font-size:.75rem;font-weight:600;color:var(--accent);background:transparent;
border:1px solid var(--accent);border-radius:6px;padding:3px 9px;cursor:pointer}
.focusbtn:hover{background:color-mix(in srgb,var(--accent) 12%,transparent)}
.leaflet-popup-content{font:inherit;font-size:.82rem}
.leaflet-popup-content .pc{color:var(--muted);font-size:.78rem}
.leaflet-popup-content .pdh{margin:6px 0 2px;font-size:.72rem;text-transform:uppercase;
letter-spacing:.03em;color:var(--muted)}
.leaflet-popup-content .pd{font-variant-numeric:tabular-nums}
.leaflet-popup-content .pl{margin-top:6px}
.lmname{background:rgba(255,255,255,.82);border:none;box-shadow:none;font-size:.72rem;
font-weight:600;color:#1b2a31;padding:0 3px;white-space:nowrap}
.lmname:before{display:none}
.lmname.out{color:#b91c1c;background:rgba(254,226,226,.92)}
.leaflet-popup-content .pwarn{color:#b91c1c;font-weight:700;font-size:.76rem;margin:3px 0}
.outside{color:#b91c1c;font-weight:700;font-size:.82em}
td.c5{text-align:center}
.in5.yes{color:var(--muted)}
.in5.no{color:#b91c1c;font-weight:700;white-space:nowrap}
.cell-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:16px 18px;
border-bottom:1px solid var(--line);background:linear-gradient(0deg,transparent,color-mix(in srgb,var(--accent) 5%,transparent))}
.cell-head h2{margin:0;font-size:1.1rem}
.cell-head code{font-family:var(--mono);background:var(--chip);padding:1px 7px;border-radius:5px}
.admin{color:var(--muted);font-size:.85rem;margin-top:2px}
.tags{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-start}
.tag,.status{font-size:.72rem;font-weight:700;letter-spacing:.03em;padding:3px 9px;border-radius:999px;
background:var(--chip);color:var(--muted);text-transform:uppercase}
.tag.vcsl{text-transform:none;font-weight:600}
.status-sampled{background:color-mix(in srgb,var(--signal) 18%,transparent);color:var(--signal)}
.status-replacement{background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent)}
.subcell{padding:14px 18px;border-top:1px dashed var(--line)}
.centroid-row{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:8px 0 4px;
padding:8px 12px;background:color-mix(in srgb,var(--accent) 9%,var(--card));
border:1px solid color-mix(in srgb,var(--accent) 30%,transparent);border-radius:8px}
.centroid-row .clabel{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:var(--accent)}
.centroid-row .coord{font-family:var(--mono);font-size:.95rem;font-weight:700;font-variant-numeric:tabular-nums;
user-select:all;-webkit-user-select:all;background:var(--card);padding:2px 8px;border-radius:5px;
border:1px solid var(--line)}
.centroid-row .lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.copy{font:inherit;font-size:.75rem;font-weight:600;cursor:pointer;border:1px solid var(--accent);
color:var(--accent);background:transparent;border-radius:6px;padding:3px 9px}
.copy:hover{background:color-mix(in srgb,var(--accent) 14%,transparent)}
.copy.ok{background:var(--signal);border-color:var(--signal);color:#fff}
.road-row{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:0 0 8px;
padding:8px 12px;background:color-mix(in srgb,#0284c7 8%,var(--card));
border:1px solid color-mix(in srgb,#0284c7 30%,transparent);border-radius:8px}
.road-row .clabel.road{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#0369a1}
.road-row .coord{font-family:var(--mono);font-size:.95rem;font-weight:700;user-select:all;-webkit-user-select:all;
background:var(--card);padding:2px 8px;border-radius:5px;border:1px solid var(--line)}
.road-row .lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.road-row .roadmeta{font-size:.82rem;font-weight:600}
.road-row.none .roadmeta{color:var(--muted);font-weight:400}
.rstatus{font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.03em;
padding:2px 7px;border-radius:999px;background:var(--chip);color:var(--muted)}
.rs-crossing{background:color-mix(in srgb,var(--signal) 18%,transparent);color:var(--signal)}
.rs-nearest_only{background:color-mix(in srgb,#0284c7 16%,transparent);color:#0369a1}
.roadname{background:rgba(224,242,254,.92);border:none;box-shadow:none;font-size:.68rem;
font-weight:600;color:#0369a1;padding:0 3px}
.roadname:before{display:none}
.sc-head{display:flex;gap:8px 14px;flex-wrap:wrap;align-items:baseline}
.sc-id{font-weight:600}
.sc-meta{color:var(--muted);font-size:.82rem}
.sc-meta code{font-family:var(--mono)}
.role{font-size:.68rem;font-weight:700;text-transform:uppercase;padding:2px 8px;border-radius:5px}
.role-primary{background:color-mix(in srgb,var(--signal) 18%,transparent);color:var(--signal)}
.role-reserve{background:color-mix(in srgb,var(--accent) 15%,transparent);color:var(--accent)}
.sc-links{margin-left:auto;display:flex;gap:10px;align-items:baseline}
.gbtn{font-size:.8rem;font-weight:600;color:var(--accent);text-decoration:none}
.gbtn:hover{text-decoration:underline}
.gbtn.ghost{color:var(--muted);font-weight:500}
.startrow{border-bottom:none;padding-top:0;margin-bottom:18px}
.startrow label{align-self:center;font-size:.85rem;color:var(--muted);font-weight:600}
.sheet{background:color-mix(in srgb,var(--accent) 12%,transparent);color:var(--accent);
text-transform:none;text-decoration:none}
.overview-wrap{padding:10px 14px;background:var(--chip);border-bottom:1px solid var(--line)}
.overview-lbl,.map-lbl{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
color:var(--muted);margin-bottom:6px}
.map-lbl{padding:10px 14px 0}
.overview{width:100%;max-height:240px;object-fit:contain;display:block;border-radius:6px;
background:var(--ground)}
a.dir{font-weight:600}
.narrative{margin:10px 0;padding:10px 12px;background:var(--chip);border-radius:8px;font-size:.9rem}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
table.lm{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:4px}
table.lm th{text-align:left;color:var(--muted);font-weight:600;font-size:.72rem;
text-transform:uppercase;letter-spacing:.03em;padding:6px 8px;border-bottom:1px solid var(--line)}
table.lm td{padding:7px 8px;border-bottom:1px solid var(--line)}
table.lm tr.inside td{background:color-mix(in srgb,var(--signal) 8%,transparent)}
td.num,td.dist{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.name{font-weight:600}
.mapbtn{font:inherit;font-size:.9rem;line-height:1;cursor:pointer;border:1px solid var(--accent);
color:var(--accent);background:transparent;border-radius:5px;padding:1px 5px;vertical-align:middle}
.mapbtn:hover{background:color-mix(in srgb,var(--accent) 14%,transparent)}
td.links{white-space:nowrap}
td.links a{margin-right:8px}
.sc-links .lbl{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
a{color:var(--accent)}
.cat{font-size:.72rem;padding:1px 7px;border-radius:5px;background:var(--chip);white-space:nowrap}
.cat-settlement{color:#2563eb}.cat-school{color:#c026d3}.cat-worship{color:#7c3aed}
.cat-health{color:#dc2626}.cat-market{color:#ea580c}.cat-water{color:#0d9488}
.empty{color:var(--accent);font-size:.85rem;padding:6px 0}
.hidden{display:none}
</style>
</head>
<body>
<div class="wrap">
  <h1>Listing landmarks — field navigation</h1>
  <div class="sub">{{N_CELLS}} 5km grid cells · {{N_SUB}} selected 500m sub-cells · {{GENERATED}}<br>
    For each 500m sub-cell, the recognisable places nearby, nearest-first, with distance from the
    cell edge and direction. Click a landmark to open it in Google Maps or view its OSM attributes.</div>
  <div class="controls">
    <input id="q" placeholder="Search ward, village, landmark, 5km id, or 500m id (e.g. G_0618_1099)…" oninput="flt()">
    <select id="region" onchange="flt()"><option value="">All regions</option>{{REGION_OPTS}}</select>
    <select id="status" onchange="flt()"><option value="">All statuses</option>
      <option value="sampled">sampled</option><option value="replacement">replacement</option></select>
    <span class="count" id="count"></span>
  </div>
  <div class="controls startrow">
    <label for="origin">Start point&nbsp;(route from):</label>
    <input id="origin" placeholder="e.g. Ibumu ward office  ·  or  -7.3052, 36.1493" oninput="setOrigin()">
    <span class="count" id="ostate">routes start from device location</span>
  </div>
  {{CARDS}}
</div>
<script>
var CELLS={{CELLS_JSON}};   // per-5km-cell geojson + landmarks
var MAPS={};                // lazily-created Leaflet maps, keyed by 5km id

// Build one 5km cell's map the first time its section is opened (keeps 48 maps
// from loading at once). Grid = red; primary 500m = green, reserve = gold;
// dots = landmarks (popup lists distance+direction to each 500m cell).
function initMap(km5){
  if(MAPS[km5]){MAPS[km5].map.invalidateSize();return;}
  var d=CELLS[km5]; if(!d)return;
  var map=L.map('map_'+km5,{scrollWheelZoom:false});
  var ghyb=L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',{maxZoom:20,attribution:'© Google'});
  var osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'});
  ghyb.addTo(map);
  var cellLayer=L.geoJSON(d.cell5k,{style:{color:'#e11d48',weight:3,fill:false}}).addTo(map);
  function sqStyle(f){var p=f.properties.role==='primary';return {color:p?'#16a34a':'#d19a1e',
    weight:2,fillColor:p?'#16a34a':'#d19a1e',fillOpacity:0.12};}
  var subLayers={};
  function mk(role){return L.geoJSON(d.squares,{filter:function(f){return f.properties.role===role;},
    style:sqStyle,onEachFeature:function(f,l){l.bindPopup(f.properties.popup);subLayers[f.properties.grid_id]=l;}});}
  var prim=mk('primary').addTo(map), res=mk('reserve').addTo(map);
  var lmGroup=L.layerGroup(), lmMarkers={};
  d.landmarks.forEach(function(p){
    // landmarks outside the 5km cell get a red ring + a ⚠ label
    var mk=L.circleMarker([p.lat,p.lon],{radius:7,color:p.outside?'#e11d48':'#fff',
      weight:p.outside?2.5:1.5,fillColor:p.color,fillOpacity:1})
     .bindPopup(p.popup)
     .bindTooltip((p.outside?'⚠ ':'')+p.name,{permanent:true,direction:'right',offset:[6,0],
       className:'lmname'+(p.outside?' out':'')})
     .addTo(lmGroup);
    lmMarkers[p.lat.toFixed(6)+','+p.lon.toFixed(6)]=mk;});
  lmGroup.addTo(map);
  var overlays={'Primary 500m':prim,'Reserve 500m':res,'Landmarks':lmGroup};
  // road-access points: a marker at the snapped road point + a dashed connector
  // to the 500m centroid (the off-road walk). Only when road-snap data exists.
  var roadMarkers={};
  if(d.roads&&d.roads.length){
    var roadGroup=L.layerGroup();
    d.roads.forEach(function(r){
      L.polyline([[r.clat,r.clon],[r.rlat,r.rlon]],{color:'#0284c7',weight:2,
        dashArray:'4 5',opacity:.9}).addTo(roadGroup);
      var rm=L.circleMarker([r.rlat,r.rlon],{radius:6,color:'#0369a1',weight:2,
        fillColor:'#38bdf8',fillOpacity:1}).bindPopup(r.popup)
       .bindTooltip('road access',{direction:'right',offset:[6,0],className:'roadname'})
       .addTo(roadGroup);
      roadMarkers[r.rlat.toFixed(6)+','+r.rlon.toFixed(6)]=rm;
    });
    roadGroup.addTo(map); overlays['Road access']=roadGroup;
  }
  L.control.layers({'Satellite + labels':ghyb,'OSM street':osm},overlays,{collapsed:false}).addTo(map);
  map.fitBounds(cellLayer.getBounds().pad(0.05));
  map.on('popupopen',setOrigin);   // keep popup route links in sync with the Start point
  MAPS[km5]={map:map,subLayers:subLayers,lmMarkers:lmMarkers,roadMarkers:roadMarkers};
  setTimeout(function(){map.invalidateSize();},60);
}
// Open the cell (if collapsed), init its map, and frame one 500m sub-cell.
function focusSub(km5,gid){
  var m=MAPS[km5];
  if(!m){var det=document.getElementById('map_'+km5).closest('details');
    if(det&&!det.open)det.open=true; initMap(km5); m=MAPS[km5];}
  if(m&&m.subLayers[gid]){m.map.fitBounds(m.subLayers[gid].getBounds().pad(1.2));
    m.subLayers[gid].openPopup();}
}
// Open the cell, init its map, and pan/zoom to one landmark (brings far,
// outside-5km landmarks that sit beyond the default view into frame).
function focusLandmark(km5,key){
  var fresh=!MAPS[km5];
  var det=document.getElementById('map_'+km5).closest('details');
  if(det&&!det.open)det.open=true;
  initMap(km5);
  var m=MAPS[km5]; if(!m)return;
  function go(){var mk=m.lmMarkers[key];
    if(mk){m.map.setView(mk.getLatLng(),16);mk.openPopup();}
    else{var p=key.split(',');m.map.setView([parseFloat(p[0]),parseFloat(p[1])],16);}}
  if(fresh){m.map.invalidateSize();setTimeout(go,140);}else{go();}
}
// Same idea for a road-access point (opens its popup) ...
function focusRoad(km5,key){
  var fresh=!MAPS[km5];
  var det=document.getElementById('map_'+km5).closest('details');
  if(det&&!det.open)det.open=true;
  initMap(km5);
  var m=MAPS[km5]; if(!m)return;
  function go(){var mk=m.roadMarkers&&m.roadMarkers[key];
    if(mk){m.map.setView(mk.getLatLng(),17);mk.openPopup();}
    else{var p=key.split(',');m.map.setView([parseFloat(p[0]),parseFloat(p[1])],17);}}
  if(fresh){m.map.invalidateSize();setTimeout(go,140);}else{go();}
}
// ... and for the 500m centroid (no permanent marker, so pulse it briefly).
function focusCentroid(km5,lat,lon){
  var fresh=!MAPS[km5];
  var det=document.getElementById('map_'+km5).closest('details');
  if(det&&!det.open)det.open=true;
  initMap(km5);
  var m=MAPS[km5]; if(!m)return;
  function go(){m.map.setView([lat,lon],17);flashAt(m.map,lat,lon);}
  if(fresh){m.map.invalidateSize();setTimeout(go,140);}else{go();}
}
function flashAt(map,lat,lon){
  var c=L.circleMarker([lat,lon],{radius:12,color:'#e11d48',weight:3,fill:false}).addTo(map);
  setTimeout(function(){map.removeLayer(c);},2200);
}
// Copy a centroid "lat, lon" to the clipboard, with brief visual feedback.
function copyCoord(btn){
  var t=btn.dataset.coord;
  function done(){var o=btn.textContent;btn.textContent='copied ✓';btn.classList.add('ok');
    setTimeout(function(){btn.textContent=o;btn.classList.remove('ok');},1200);}
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(t).then(done,done);}
  else{var ta=document.createElement('textarea');ta.value=t;document.body.appendChild(ta);
    ta.select();try{document.execCommand('copy');}catch(e){}document.body.removeChild(ta);done();}
}
function flt(){
  var q=document.getElementById('q').value.toLowerCase().trim();
  var rg=document.getElementById('region').value, st=document.getElementById('status').value;
  var n=0, cells=document.querySelectorAll('.cell');
  cells.forEach(function(c){
    var ok=(!q||c.dataset.filter.indexOf(q)>=0)&&(!rg||c.dataset.region===rg)&&(!st||c.dataset.status===st);
    c.classList.toggle('hidden',!ok); if(ok)n++;
  });
  document.getElementById('count').textContent=n+' / '+cells.length+' cells shown';
}
// Rewrite every route link (Google + OSM) so directions start from the typed
// origin (a place name or "lat, lon"). Blank origin = the map app's own default
// (device location in Google; unset in OSM).
function setOrigin(){
  var o=document.getElementById('origin').value.trim();
  var enc=encodeURIComponent(o);
  document.querySelectorAll('a.dir').forEach(function(a){
    var d=encodeURIComponent(a.dataset.dest);
    var url;
    if(a.dataset.engine==='o'){
      url='https://www.openstreetmap.org/directions?';
      if(o) url+='from='+enc+'&';
      url+='to='+d+'&engine=fossgis_osrm_car';
    } else {
      url='https://www.google.com/maps/dir/?api=1&destination='+d+'&travelmode=driving';
      if(o) url+='&origin='+enc;
    }
    a.href=url;
  });
  document.getElementById('ostate').textContent = o ? ('routes start from: '+o)
                                                     : 'routes start from the map app default (your location)';
}
flt(); setOrigin();
</script>
</body>
</html>"""


def main():
    args = parse_args()
    config = load_config()
    data_dir = get_data_dir(config)

    lm_path = (Path(args.landmarks) if args.landmarks else
               data_dir / "01_input_data" / "boundaries" / "subcell_landmarks.csv")
    if not lm_path.exists():
        print(f"Error: landmark cache not found: {lm_path}\nRun scripts/build_landmarks.py first.")
        sys.exit(1)

    out_path = (Path(args.output) if args.output else
                get_output_dir(config) / "listing_maps" / "landmark_guide.html")

    grid = load_control_grid(data_dir)
    selected = load_selected_subcells(data_dir).to_crs(4326)
    lm = pd.read_csv(lm_path)

    # Road-snap cache: explicit --road-snaps path, else auto-load the default if present.
    road_path = (Path(args.road_snaps) if args.road_snaps else
                 data_dir / "01_input_data" / "boundaries" / "subcell_road_snaps.csv")
    roads = None
    if road_path.exists():
        roads = pd.read_csv(road_path)
        print(f"Loaded {len(roads)} road-snap rows from {road_path.name}")
    else:
        print("No road-snap cache found — guide will omit road access (run build_road_snaps.py to add it).")

    note = f"landmark cache: {lm_path.name}"
    html_out = build_html(grid, selected, lm, note, base_dir=out_path.parent, roads=roads)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8")
    n_roads = int(roads["road_lat"].notna().sum()) if roads is not None else 0
    print(f"Wrote landmark guide ({lm['5km_id'].nunique()} cells, {lm['grid_id'].nunique()} sub-cells, "
          f"{n_roads} road points) -> {out_path}")


if __name__ == "__main__":
    main()
