"""Generate high-resolution PNG maps for enumeration area grid cells.

Supports two map types:
- **Overview (5km):** Shows the full 5km cell with selected 500m sub-cells
  highlighted (primary=green, reserve=yellow) on a Google Hybrid basemap
  that includes village/place labels.
- **Detail (500m):** Zoomed-in view of a single selected 500m sub-cell with
  building footprints overlaid on ESRI satellite imagery.

Output is tablet-optimized PNG (1920×1080 at 100 dpi).
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

# Building overlay style (from notebook 05 testing)
BUILDING_FILL = "yellow"
BUILDING_EDGE = "orange"
BUILDING_ALPHA = 0.6
BUILDING_LINEWIDTH = 0.5

# Sub-cell selection colours
PRIMARY_FILL = "lime"
PRIMARY_EDGE = "green"
RESERVE_FILL = "yellow"
RESERVE_EDGE = "goldenrod"
SELECTION_ALPHA = 0.4
SELECTION_LINEWIDTH = 2

# Basemap providers
ESRI_SATELLITE = cx.providers.Esri.WorldImagery if HAS_CONTEXTILY else None
GOOGLE_HYBRID = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
CARTODB_LABELS = cx.providers.CartoDB.PositronOnlyLabels if HAS_CONTEXTILY else None

TARGET_CRS = "EPSG:3857"


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
        self.output_dir = Path(output_dir)
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.dpi = dpi
        self.add_basemap = add_basemap and HAS_CONTEXTILY
        self.basemap_source = basemap_source
        self.zoom = zoom
        self.add_scalebar = add_scalebar and HAS_SCALEBAR

    # ------------------------------------------------------------------
    # Overview map (5km cell with selected sub-cells)
    # ------------------------------------------------------------------

    def generate_overview(
        self,
        grid_cell: gpd.GeoDataFrame,
        grid_id: str,
        selected_subcells: gpd.GeoDataFrame,
        all_grid_cells: gpd.GeoDataFrame | None = None,
        label: str | None = None,
        buffer_factor: float = 0.05,
    ) -> Path:
        """Generate an overview map of a 5km cell with selected sub-cells.

        Uses Google Hybrid basemap (satellite + place labels) and highlights
        primary sub-cells in green and reserve sub-cells in yellow.

        Args:
            grid_cell: GeoDataFrame of the single 5km cell.
            grid_id: Cell identifier for file naming.
            selected_subcells: Selected 500m sub-cells for this cell
                (must have 'selection_role' column).
            all_grid_cells: Neighbouring cells for context (optional).
            label: Display label (defaults to grid_id).
            buffer_factor: View buffer around the cell.

        Returns:
            Path to the saved PNG.
        """
        if label is None:
            label = grid_id
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))

        cell_wm = grid_cell.to_crs(TARGET_CRS)
        bounds = cell_wm.total_bounds
        dx = (bounds[2] - bounds[0]) * buffer_factor
        dy = (bounds[3] - bounds[1]) * buffer_factor
        ax.set_xlim(bounds[0] - dx, bounds[2] + dx)
        ax.set_ylim(bounds[1] - dy, bounds[3] + dy)

        # Context cells
        if all_grid_cells is not None:
            nearby = self._clip_to_bounds(all_grid_cells, grid_cell, buffer_factor)
            if nearby is not None and len(nearby) > 0:
                nearby.to_crs(TARGET_CRS).plot(
                    ax=ax, facecolor="none",
                    edgecolor=GRID_EDGE_COLOR, linewidth=GRID_EDGE_WIDTH,
                )

        # 5km cell boundary
        cell_wm.plot(ax=ax, facecolor="none", edgecolor=HIGHLIGHT_COLOR,
                     linewidth=HIGHLIGHT_EDGE_WIDTH)

        # Selected sub-cells
        primary = selected_subcells[selected_subcells["selection_role"] == "primary"]
        reserve = selected_subcells[selected_subcells["selection_role"] == "reserve"]

        if len(reserve) > 0:
            reserve.to_crs(TARGET_CRS).plot(
                ax=ax, facecolor=RESERVE_FILL, edgecolor=RESERVE_EDGE,
                linewidth=SELECTION_LINEWIDTH, alpha=SELECTION_ALPHA,
            )
        if len(primary) > 0:
            primary.to_crs(TARGET_CRS).plot(
                ax=ax, facecolor=PRIMARY_FILL, edgecolor=PRIMARY_EDGE,
                linewidth=SELECTION_LINEWIDTH, alpha=SELECTION_ALPHA,
            )

        # Label each sub-cell and mark centroid as starting point
        import matplotlib.patheffects as pe
        for _, row in selected_subcells.to_crs(TARGET_CRS).iterrows():
            centroid = row.geometry.centroid
            # Centroid marker (red dot)
            ax.plot(centroid.x, centroid.y, marker="o", color="red",
                    markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=5)
            # Label with role + ID
            role_letter = "P" if row["selection_role"] == "primary" else "R"
            subcell_id = row.get("grid_id", "")
            lbl = f"{role_letter}\n{subcell_id}" if subcell_id else role_letter
            ax.annotate(
                lbl, xy=(centroid.x, centroid.y), xytext=(0, 14),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=3, foreground="black")],
            )

        # Basemap: Google Hybrid for village labels
        if self.add_basemap:
            try:
                cx.add_basemap(ax, source=GOOGLE_HYBRID, zoom=14)
            except Exception as e:
                print(f"  Could not add basemap for overview {grid_id}: {e}")

        if self.add_scalebar:
            ax.add_artist(ScaleBar(1, location="lower right", box_alpha=0.7))

        ax.set_title(f"Overview: Cell {label}", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_axis_off()

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=PRIMARY_FILL, edgecolor=PRIMARY_EDGE,
                           alpha=SELECTION_ALPHA, label="Primary (P)"),
            mpatches.Patch(facecolor=RESERVE_FILL, edgecolor=RESERVE_EDGE,
                           alpha=SELECTION_ALPHA, label="Reserve (R)"),
            mpatches.Patch(facecolor="none", edgecolor=HIGHLIGHT_COLOR,
                           linewidth=2, label="5km cell"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.8)

        output_path = self._save_flat(fig, "overview_5km")
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Detail map (500m sub-cell with buildings)
    # ------------------------------------------------------------------

    def generate_detail(
        self,
        subcell: gpd.GeoDataFrame,
        grid_id: str,
        subcell_index: int,
        role: str,
        buildings: gpd.GeoDataFrame | None = None,
        label: str | None = None,
        buffer_factor: float = 0.15,
    ) -> Path:
        """Generate a detail map of a single 500m sub-cell.

        Uses ESRI satellite basemap with building footprints overlaid
        in yellow/orange.

        Args:
            subcell: GeoDataFrame of the single 500m sub-cell.
            grid_id: Parent 5km cell ID (for folder naming).
            subcell_index: Index of this sub-cell (1-based, for filename).
            role: "primary" or "reserve".
            buildings: Building footprints to overlay (optional).
            label: Display label (defaults to auto-generated).
            buffer_factor: View buffer around the sub-cell.

        Returns:
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))

        subcell_wm = subcell.to_crs(TARGET_CRS)
        bounds = subcell_wm.total_bounds
        dx = (bounds[2] - bounds[0]) * buffer_factor
        dy = (bounds[3] - bounds[1]) * buffer_factor
        ax.set_xlim(bounds[0] - dx, bounds[2] + dx)
        ax.set_ylim(bounds[1] - dy, bounds[3] + dy)

        # Building footprints
        if buildings is not None:
            buildings_clip = self._clip_to_bounds(buildings, subcell, buffer_factor)
            if buildings_clip is not None and len(buildings_clip) > 0:
                buildings_clip.to_crs(TARGET_CRS).plot(
                    ax=ax, color=BUILDING_FILL, edgecolor=BUILDING_EDGE,
                    linewidth=BUILDING_LINEWIDTH, alpha=BUILDING_ALPHA,
                )

        # Sub-cell boundary
        edge_color = PRIMARY_EDGE if role == "primary" else RESERVE_EDGE
        subcell_wm.plot(ax=ax, facecolor="none", edgecolor=edge_color, linewidth=3)

        # Centroid marker — starting point for enumerators
        import matplotlib.patheffects as pe
        centroid = subcell_wm.geometry.iloc[0].centroid
        ax.plot(centroid.x, centroid.y, marker="+", color="red",
                markersize=20, markeredgewidth=3, zorder=5)
        ax.annotate(
            "START", xy=(centroid.x, centroid.y), xytext=(0, 16),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="red",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )

        # Basemap: Google Hybrid (satellite + labels — best rural Africa coverage)
        if self.add_basemap:
            try:
                cx.add_basemap(ax, source=GOOGLE_HYBRID, zoom="auto")
            except Exception as e:
                print(f"  Could not add basemap for detail {grid_id}/{subcell_index}: {e}")

        if self.add_scalebar:
            ax.add_artist(ScaleBar(1, location="lower right", box_alpha=0.7))

        building_count = subcell.iloc[0].get("building_count", "?")
        subcell_id = subcell.iloc[0].get("grid_id", "")
        if label is None:
            label = f"5km: {grid_id} — 500m: {subcell_id} ({role}) — {building_count} buildings"
        ax.set_title(label, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_axis_off()

        filename = f"subcell_{subcell_id}_{role}" if subcell_id else f"subcell_{subcell_index}_{role}"
        output_path = self._save_flat(fig, filename)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Legacy single-map method (kept for backwards compatibility)
    # ------------------------------------------------------------------

    def generate_map(
        self,
        grid_cell: gpd.GeoDataFrame,
        grid_id: str,
        label: str | None = None,
        all_grid_cells: gpd.GeoDataFrame | None = None,
        subgrid: gpd.GeoDataFrame | None = None,
        roads: gpd.GeoDataFrame | None = None,
        buildings: gpd.GeoDataFrame | None = None,
        buffer_factor: float = 0.3,
        show: bool = False,
    ) -> Path:
        """Generate a single map PNG for one grid cell (legacy interface)."""
        if label is None:
            label = grid_id
        fig, ax = plt.subplots(1, 1, figsize=(self.fig_width, self.fig_height))

        cell_merc = grid_cell.to_crs(TARGET_CRS)

        bounds = cell_merc.total_bounds
        dx = (bounds[2] - bounds[0]) * buffer_factor
        dy = (bounds[3] - bounds[1]) * buffer_factor
        ax.set_xlim(bounds[0] - dx, bounds[2] + dx)
        ax.set_ylim(bounds[1] - dy, bounds[3] + dy)

        if buildings is not None:
            buildings_clip = self._clip_to_bounds(buildings, grid_cell, buffer_factor)
            if buildings_clip is not None and len(buildings_clip) > 0:
                buildings_clip.to_crs(TARGET_CRS).plot(
                    ax=ax, color=BUILDING_FILL, edgecolor=BUILDING_EDGE,
                    linewidth=BUILDING_LINEWIDTH, alpha=BUILDING_ALPHA,
                )

        if roads is not None:
            roads_clip = self._clip_to_bounds(roads, grid_cell, buffer_factor)
            if roads_clip is not None and len(roads_clip) > 0:
                roads_clip.to_crs(TARGET_CRS).plot(
                    ax=ax, color="#666666", linewidth=0.8,
                )

        if all_grid_cells is not None:
            nearby = self._clip_to_bounds(all_grid_cells, grid_cell, buffer_factor)
            if nearby is not None and len(nearby) > 0:
                nearby.to_crs(TARGET_CRS).plot(
                    ax=ax, facecolor="none",
                    edgecolor=GRID_EDGE_COLOR, linewidth=GRID_EDGE_WIDTH,
                )

        cell_merc.plot(
            ax=ax, facecolor="none",
            edgecolor=HIGHLIGHT_COLOR, linewidth=HIGHLIGHT_EDGE_WIDTH,
        )

        if subgrid is not None and "5km_id" in subgrid.columns:
            cell_subgrid = subgrid[subgrid["5km_id"] == int(grid_id)]
            if len(cell_subgrid) > 0:
                cell_subgrid.to_crs(TARGET_CRS).plot(
                    ax=ax, facecolor="none",
                    edgecolor="#0066CC", linewidth=0.6, alpha=0.8,
                )

        if self.add_basemap:
            try:
                source = self.basemap_source or cx.providers.Esri.WorldImagery
                basemap_kwargs = {"ax": ax, "source": source}
                if self.zoom is not None:
                    basemap_kwargs["zoom"] = self.zoom
                cx.add_basemap(**basemap_kwargs)
            except Exception as e:
                print(f"Could not add basemap for {grid_id}: {e}")

        if self.add_scalebar:
            ax.add_artist(ScaleBar(1, location="lower right", box_alpha=0.7))

        ax.set_title(f"Grid Cell: {label}", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_axis_off()

        legend_patch = mpatches.Patch(
            edgecolor=HIGHLIGHT_COLOR, facecolor="none", linewidth=2,
            label=label,
        )
        ax.legend(handles=[legend_patch], loc="upper right", fontsize=10)

        output_path = self._save(fig, grid_id, "map")

        if show:
            plt.show()
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_flat(self, fig: Figure, name: str) -> Path:
        """Save figure directly into output_dir (no subfolder)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.1)
        return output_path

    def _save(self, fig: Figure, grid_id: str, name: str) -> Path:
        """Save figure to a grid_cell subfolder (legacy)."""
        cell_dir = self.output_dir / f"grid_cell_{grid_id}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        output_path = cell_dir / f"grid_cell_{grid_id}_{name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.1)
        return output_path

    def _clip_to_bounds(
        self,
        layer: gpd.GeoDataFrame,
        reference: gpd.GeoDataFrame,
        buffer_factor: float,
    ) -> gpd.GeoDataFrame | None:
        """Clip a layer to the buffered bounding box of a reference geometry."""
        try:
            ref_bounds = reference.to_crs(layer.crs).total_bounds
            dx = (ref_bounds[2] - ref_bounds[0]) * buffer_factor
            dy = (ref_bounds[3] - ref_bounds[1]) * buffer_factor
            from shapely.geometry import box
            clip_box = box(
                ref_bounds[0] - dx, ref_bounds[1] - dy,
                ref_bounds[2] + dx, ref_bounds[3] + dy,
            )
            return gpd.clip(layer, clip_box)
        except Exception:
            return layer
