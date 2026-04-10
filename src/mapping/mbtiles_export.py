"""Export styled MBTiles for offline use in SurveyCTO.

Companion to ``map_generator.py`` — produces georeferenced raster tiles
(MBTiles) carrying the same overlays the static PNGs use:

- ``export_overview_mbtiles``  : 5km cell + selected sub-cells, zoom ~17
- ``export_detail_mbtiles``    : single 500m sub-cell + buildings, zoom ~19

Each function downloads Google Hybrid satellite tiles for the area
(roads, place labels, hamlets are baked into the tiles), burns the
vector overlays into the RGB pixels, then writes a single MBTiles file
with overviews built down to ~zoom-4. The resulting file is a
self-contained offline basemap that SurveyCTO can load directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import contextily as cx
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.geometry import LineString, box

TARGET_CRS = "EPSG:3857"
GOOGLE_HYBRID = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"

# RGB uint8 colours
RED = (255, 30, 30)
GREEN = (40, 200, 60)
GOLDENROD = (218, 165, 32)
YELLOW = (255, 230, 0)
ORANGE = (255, 140, 0)
WHITE = (255, 255, 255)

# Default zoom levels (Google Hybrid maxes out around 19-20)
DEFAULT_DETAIL_ZOOM = 19
DEFAULT_OVERVIEW_ZOOM = 15

# Overview pyramid factors fed to gdaladdo
DEFAULT_OVERVIEW_FACTORS = (2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bounds_with_padding(gdf_3857: gpd.GeoDataFrame, factor: float):
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    dx = (maxx - minx) * factor
    dy = (maxy - miny) * factor
    return minx - dx, miny - dy, maxx + dx, maxy + dy


def _download_satellite(bounds, zoom: int, output_tif: Path, source: str = GOOGLE_HYBRID) -> Path:
    west, south, east, north = bounds
    cx.bounds2raster(
        west, south, east, north,
        str(output_tif), zoom=zoom, source=source, ll=False,
    )
    return output_tif


def _blend(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 1.0) -> None:
    """Blend an RGB triple into ``rgb`` (shape (3,h,w)) where mask == 1."""
    sel = mask == 1
    if not sel.any():
        return
    if alpha >= 1.0:
        for c in range(3):
            rgb[c][sel] = color[c]
        return
    for c in range(3):
        rgb[c][sel] = (rgb[c][sel] * (1 - alpha) + color[c] * alpha).astype("uint8")


def _rasterize(geoms: Iterable, shape, transform) -> np.ndarray:
    return rasterize(
        ((g, 1) for g in geoms),
        out_shape=shape, transform=transform, fill=0, dtype="uint8",
    )


def _crosshair(point, px_size: float, arm_px: float, width_px: float):
    """Two perpendicular line segments through ``point``, buffered to width."""
    arm = px_size * arm_px
    half_w = px_size * width_px / 2
    h = LineString([(point.x - arm, point.y), (point.x + arm, point.y)]).buffer(half_w)
    v = LineString([(point.x, point.y - arm), (point.x, point.y + arm)]).buffer(half_w)
    return [h, v]


def _draw_crosshair(rgb, shape, transform, point, px_size, arm_px=14, width_px=2.0):
    """White-outlined red crosshair on ``rgb``."""
    arms = _crosshair(point, px_size, arm_px=arm_px, width_px=width_px)
    arms_outline = [g.buffer(px_size * 1.0) for g in arms]
    _blend(rgb, _rasterize(arms_outline, shape, transform), WHITE)
    _blend(rgb, _rasterize(arms, shape, transform), RED)


def _add_title_banner(rgb: np.ndarray, transform: Affine, title: str):
    """Extend the raster northward with a black banner containing ``title``.

    The banner sits *above* the original bbox so it never obscures map
    content. When the enumerator pans down in the offline viewer they're
    in the actual data; panning up reveals the title.

    Returns:
        ``(new_rgb, new_transform)`` — the original transform shifted so
        the existing pixels still map to their original geographic
        coordinates and the banner pixels live north of them.
    """
    from PIL import Image, ImageDraw, ImageFont

    h, w = rgb.shape[1], rgb.shape[2]
    font_size_px = max(20, w // 50)

    # Try a TTF font for readable text; fall back to PIL's default bitmap font
    font = None
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            font = ImageFont.truetype(candidate, font_size_px)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Measure text via a throwaway draw context
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = measure.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_px = max(8, font_size_px // 3)
    banner_h = text_h + pad_px * 2

    # Build extended array: banner rows on top, original below
    new_h = h + banner_h
    new_rgb = np.zeros((3, new_h, w), dtype="uint8")  # banner starts black
    new_rgb[:, banner_h:, :] = rgb

    # Render text onto the banner via PIL
    banner_img = Image.fromarray(new_rgb[:, :banner_h, :].transpose(1, 2, 0).copy())
    draw = ImageDraw.Draw(banner_img)
    text_x = max(0, (w - text_w) // 2 - bbox[0])
    text_y = pad_px - bbox[1]
    draw.text((text_x, text_y), title, fill=(255, 255, 255), font=font)
    new_rgb[:, :banner_h, :] = np.array(banner_img).transpose(2, 0, 1)

    # Shift transform so the original pixels map to the same geographic
    # coordinates and the banner extends northward. ``e`` is negative for a
    # north-up raster, so subtracting ``banner_h * e`` increases ``f``.
    new_transform = Affine(
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f - banner_h * transform.e,
    )
    return new_rgb, new_transform


def _read_rgb(tif_path: Path):
    """Read a GeoTIFF as (rgb_3band, transform, shape, profile)."""
    with rasterio.open(tif_path) as src:
        data = src.read()
        if data.shape[0] == 4:
            data = data[:3]
        elif data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        return data, src.transform, (src.height, src.width), src.profile.copy()


def _write_styled_geotiff(rgb: np.ndarray, profile: dict, output_path: Path) -> Path:
    profile = profile.copy()
    profile.update(count=3, dtype="uint8", photometric="RGB")
    profile.pop("nodata", None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgb)
    return output_path


def _geotiff_to_mbtiles(
    tif_path: Path,
    mbtiles_path: Path,
    overview_factors: Iterable[int] = DEFAULT_OVERVIEW_FACTORS,
) -> Path:
    """``gdal_translate`` + ``gdaladdo`` equivalent via rasterio."""
    if mbtiles_path.exists():
        mbtiles_path.unlink()  # MBTILES driver refuses to overwrite
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        profile.update(driver="MBTILES", TILE_FORMAT="PNG")
        profile.pop("nodata", None)
        data = src.read()
        if data.shape[0] == 4:
            data = data[:3]
            profile["count"] = 3
        with rasterio.open(mbtiles_path, "w", **profile) as dst:
            dst.write(data)
    with rasterio.open(mbtiles_path, "r+") as ds:
        ds.build_overviews(list(overview_factors), Resampling.nearest)
    return mbtiles_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_detail_mbtiles(
    subcell: gpd.GeoDataFrame,
    output_path: Path,
    buildings: gpd.GeoDataFrame | None = None,
    role: str = "primary",
    zoom: int = DEFAULT_DETAIL_ZOOM,
    buffer_factor: float = 0.15,
    title: str | None = None,
    keep_intermediate: bool = False,
) -> Path:
    """Export one 500m sub-cell as a styled MBTiles file.

    Mirrors :py:meth:`MapGenerator.generate_detail` overlays:
    yellow building footprints, sub-cell boundary in green (primary)
    or goldenrod (reserve), and a red crosshair on the centroid.

    Args:
        subcell: GeoDataFrame containing exactly one 500m sub-cell.
        output_path: Where to write the ``.mbtiles`` file.
        buildings: Building footprints (any CRS); clipped to bbox.
        role: ``"primary"`` or ``"reserve"`` — picks the boundary colour.
        zoom: Max zoom to download from the tile provider.
        buffer_factor: Pad bounding box by this fraction on each side.
        title: Optional heading rendered on a black banner *above* the
            geographic bbox (so it never obscures map content).
        keep_intermediate: Keep the temporary ``.tif`` files for inspection.

    Returns:
        Path to the written ``.mbtiles`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub_3857 = subcell.to_crs(TARGET_CRS)
    bounds = _bounds_with_padding(sub_3857, buffer_factor)

    raw_tif = output_path.with_name(output_path.stem + "_raw.tif")
    _download_satellite(bounds, zoom=zoom, output_tif=raw_tif)
    rgb, transform, shape, profile = _read_rgb(raw_tif)
    px_size = abs(transform.a)

    # Buildings (clipped) — fill + edge
    if buildings is not None:
        bbox_geom = box(*bounds)
        bldg_3857 = buildings.to_crs(TARGET_CRS)
        bldg_clip = bldg_3857[bldg_3857.intersects(bbox_geom)]
        if len(bldg_clip) > 0:
            fill_mask = _rasterize(bldg_clip.geometry, shape, transform)
            edge_geoms = [g.boundary.buffer(px_size * 0.7) for g in bldg_clip.geometry]
            edge_mask = _rasterize(edge_geoms, shape, transform)
            _blend(rgb, fill_mask, YELLOW, alpha=0.45)
            _blend(rgb, edge_mask, ORANGE, alpha=0.9)

    # Sub-cell boundary
    sub_geom = sub_3857.geometry.iloc[0]
    boundary_color = GREEN if role == "primary" else GOLDENROD
    boundary_mask = _rasterize(
        [sub_geom.boundary.buffer(px_size * 3)], shape, transform,
    )
    _blend(rgb, boundary_mask, boundary_color)

    # Centroid crosshair
    _draw_crosshair(rgb, shape, transform, sub_geom.centroid, px_size, arm_px=16, width_px=2.0)

    # Optional title banner above the bbox
    if title:
        rgb, transform = _add_title_banner(rgb, transform, title)
        profile["height"] = rgb.shape[1]
        profile["transform"] = transform

    styled_tif = output_path.with_name(output_path.stem + "_styled.tif")
    _write_styled_geotiff(rgb, profile, styled_tif)
    _geotiff_to_mbtiles(styled_tif, output_path)

    if not keep_intermediate:
        raw_tif.unlink(missing_ok=True)
        styled_tif.unlink(missing_ok=True)

    return output_path


def export_overview_mbtiles(
    grid_cell: gpd.GeoDataFrame,
    selected_subcells: gpd.GeoDataFrame,
    output_path: Path,
    zoom: int = DEFAULT_OVERVIEW_ZOOM,
    buffer_factor: float = 0.05,
    title: str | None = None,
    keep_intermediate: bool = False,
) -> Path:
    """Export a 5km overview cell as a styled MBTiles file.

    Mirrors :py:meth:`MapGenerator.generate_overview` overlays:
    red 5km cell boundary, primary sub-cells outlined green, reserve
    sub-cells outlined goldenrod, and a red crosshair at each sub-cell
    centroid. Sub-cells are *not* filled so the satellite imagery and
    place labels underneath stay legible.

    Args:
        grid_cell: GeoDataFrame containing exactly one 5km cell.
        selected_subcells: Sub-cells inside this cell, with a
            ``selection_role`` column (``"primary"``/``"reserve"``).
        output_path: Where to write the ``.mbtiles`` file.
        zoom: Max zoom to download. ``15`` keeps file size in check —
            higher values cause very large GeoTIFFs for a 5km bbox.
        buffer_factor: Pad bounding box by this fraction on each side.
        title: Optional heading rendered on a black banner *above* the
            geographic bbox (so it never obscures map content).
        keep_intermediate: Keep the temporary ``.tif`` files for inspection.

    Returns:
        Path to the written ``.mbtiles`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell_3857 = grid_cell.to_crs(TARGET_CRS)
    bounds = _bounds_with_padding(cell_3857, buffer_factor)

    raw_tif = output_path.with_name(output_path.stem + "_raw.tif")
    _download_satellite(bounds, zoom=zoom, output_tif=raw_tif)
    rgb, transform, shape, profile = _read_rgb(raw_tif)
    px_size = abs(transform.a)

    # 5km cell boundary (red, ~4 px)
    cell_boundary_mask = _rasterize(
        [cell_3857.geometry.iloc[0].boundary.buffer(px_size * 4)], shape, transform,
    )
    _blend(rgb, cell_boundary_mask, RED)

    # Selected sub-cells: outline only (no fill — keeps landmarks underneath visible)
    sub_3857 = selected_subcells.to_crs(TARGET_CRS)
    primary = sub_3857[sub_3857["selection_role"] == "primary"]
    reserve = sub_3857[sub_3857["selection_role"] == "reserve"]

    for sub_set, edge_color in ((reserve, GOLDENROD), (primary, GREEN)):
        if len(sub_set) == 0:
            continue
        edge_geoms = [g.boundary.buffer(px_size * 3) for g in sub_set.geometry]
        edge_mask = _rasterize(edge_geoms, shape, transform)
        _blend(rgb, edge_mask, edge_color)

    # Crosshair at each sub-cell centroid
    for _, row in sub_3857.iterrows():
        _draw_crosshair(
            rgb, shape, transform, row.geometry.centroid, px_size,
            arm_px=10, width_px=1.5,
        )

    # Optional title banner above the bbox
    if title:
        rgb, transform = _add_title_banner(rgb, transform, title)
        profile["height"] = rgb.shape[1]
        profile["transform"] = transform

    styled_tif = output_path.with_name(output_path.stem + "_styled.tif")
    _write_styled_geotiff(rgb, profile, styled_tif)
    _geotiff_to_mbtiles(styled_tif, output_path)

    if not keep_intermediate:
        raw_tif.unlink(missing_ok=True)
        styled_tif.unlink(missing_ok=True)

    return output_path
