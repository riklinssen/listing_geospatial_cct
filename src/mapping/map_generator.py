"""Generate high-resolution PNG maps for enumeration area grid cells.

Each map highlights a single control grid cell with its ID label,
overlays context layers (roads, buildings), and optionally adds a
basemap via contextily. Output is a tablet-optimized PNG.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False


# Default style constants
DEFAULT_FIG_WIDTH = 19.2   # inches at 100 dpi -> 1920 px
DEFAULT_FIG_HEIGHT = 10.8  # inches at 100 dpi -> 1080 px
DEFAULT_DPI = 100
HIGHLIGHT_COLOR = "#FF4444"
HIGHLIGHT_EDGE_WIDTH = 3
GRID_EDGE_COLOR = "#333333"
GRID_EDGE_WIDTH = 0.8
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 14


class MapGenerator:
    """Generates high-resolution static maps for individual grid cells."""

    def __init__(
        self,
        output_dir: Path,
        fig_width: float = DEFAULT_FIG_WIDTH,
        fig_height: float = DEFAULT_FIG_HEIGHT,
        dpi: int = DEFAULT_DPI,
        add_basemap: bool = True,
        basemap_source=None,
        zoom: int | str | None = None,
        add_scalebar: bool = False,
    ):
        """Initialize the map generator.

        Args:
            output_dir: Directory to save generated map PNGs.
            fig_width: Figure width in inches.
            fig_height: Figure height in inches.
            dpi: Dots per inch for output PNG.
            add_basemap: Whether to add an online basemap tile layer.
            basemap_source: Contextily tile provider (e.g. cx.providers.OpenStreetMap.Mapnik).
                            Defaults to Esri.WorldImagery.
            zoom: Tile zoom level. Use an int for a fixed level, or "auto" to
                  let contextily choose. Defaults to contextily's auto behaviour.
            add_scalebar: Whether to add a distance scale bar to the map.
        """
        self.output_dir = Path(output_dir)
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.dpi = dpi
        self.add_basemap = add_basemap and HAS_CONTEXTILY
        self.basemap_source = basemap_source
        self.zoom = zoom
        self.add_scalebar = add_scalebar and HAS_SCALEBAR

    def generate_map(
        self,
        grid_cell: gpd.GeoDataFrame,
        grid_id: str,
        label: str | None = None,
        all_grid_cells: gpd.GeoDataFrame | None = None,
        roads: gpd.GeoDataFrame | None = None,
        buildings: gpd.GeoDataFrame | None = None,
        buffer_factor: float = 0.3,
        show: bool = False,
    ) -> Path:
        """Generate a single map PNG for one grid cell.

        Args:
            grid_cell: GeoDataFrame containing the single target grid cell.
            grid_id: Numeric/short identifier used for file naming.
            label: Display label on the map (defaults to grid_id).
            all_grid_cells: All grid cells for context (optional).
            roads: Road network layer (optional).
            buildings: Building footprints layer (optional).
            buffer_factor: How much to buffer the view around the cell
                           (fraction of cell extent).

        Returns:
            Path to the saved PNG file.
        """
        if label is None:
            label = grid_id
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))

        # Reproject everything to Web Mercator for basemap compatibility
        target_crs = "EPSG:3857"
        cell_merc = grid_cell.to_crs(target_crs)

        # Calculate view bounds with buffer
        bounds = cell_merc.total_bounds  # minx, miny, maxx, maxy
        dx = (bounds[2] - bounds[0]) * buffer_factor
        dy = (bounds[3] - bounds[1]) * buffer_factor
        ax.set_xlim(bounds[0] - dx, bounds[2] + dx)
        ax.set_ylim(bounds[1] - dy, bounds[3] + dy)

        # Plot context layers first (underneath)
        if buildings is not None:
            buildings_clip = self._clip_to_bounds(buildings, grid_cell, buffer_factor)
            if buildings_clip is not None and len(buildings_clip) > 0:
                buildings_clip.to_crs(target_crs).plot(
                    ax=ax, color="#CCCCCC", edgecolor="#999999", linewidth=0.3
                )

        if roads is not None:
            roads_clip = self._clip_to_bounds(roads, grid_cell, buffer_factor)
            if roads_clip is not None and len(roads_clip) > 0:
                roads_clip.to_crs(target_crs).plot(
                    ax=ax, color="#666666", linewidth=0.8
                )

        # Plot all grid cells for context (if provided)
        if all_grid_cells is not None:
            nearby = self._clip_to_bounds(all_grid_cells, grid_cell, buffer_factor)
            if nearby is not None and len(nearby) > 0:
                nearby.to_crs(target_crs).plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor=GRID_EDGE_COLOR,
                    linewidth=GRID_EDGE_WIDTH,
                )

        # Highlight the target grid cell
        cell_merc.plot(
            ax=ax,
            facecolor="none",
            edgecolor=HIGHLIGHT_COLOR,
            linewidth=HIGHLIGHT_EDGE_WIDTH,
        )

        # Add basemap tiles
        if self.add_basemap:
            try:
                source = self.basemap_source or cx.providers.Esri.WorldImagery
                basemap_kwargs = {"ax": ax, "source": source}
                if self.zoom is not None:
                    basemap_kwargs["zoom"] = self.zoom
                cx.add_basemap(**basemap_kwargs)
            except Exception as e:
                print(f"Could not add basemap for {grid_id}: {e}")

        # Scale bar (EPSG:3857 units are metres)
        if self.add_scalebar:
            ax.add_artist(ScaleBar(1, location="lower right", box_alpha=0.7))

        # Title and legend
        ax.set_title(f"Grid Cell: {label}", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_axis_off()

        legend_patch = mpatches.Patch(
            edgecolor=HIGHLIGHT_COLOR, facecolor="none", linewidth=2,
            label=label        )
        ax.legend(handles=[legend_patch], loc="upper right", fontsize=10)

        # Save
        cell_dir = self.output_dir / f"grid_cell_{grid_id}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        output_path = cell_dir / f"grid_cell_{grid_id}_map.png"

        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        plt.close(fig)

        return output_path

    def _clip_to_bounds(
        self,
        layer: gpd.GeoDataFrame,
        reference: gpd.GeoDataFrame,
        buffer_factor: float,
    ) -> gpd.GeoDataFrame | None:
        """Clip a layer to the buffered bounding box of a reference geometry.

        Args:
            layer: Layer to clip.
            reference: Reference geometry to define the clip region.
            buffer_factor: Buffer around the reference bounds.

        Returns:
            Clipped GeoDataFrame or None if clipping fails.
        """
        try:
            ref_bounds = reference.to_crs(layer.crs).total_bounds
            dx = (ref_bounds[2] - ref_bounds[0]) * buffer_factor
            dy = (ref_bounds[3] - ref_bounds[1]) * buffer_factor
            from shapely.geometry import box
            clip_box = box(
                ref_bounds[0] - dx,
                ref_bounds[1] - dy,
                ref_bounds[2] + dx,
                ref_bounds[3] + dy,
            )
            return gpd.clip(layer, clip_box)
        except Exception:
            return layer
