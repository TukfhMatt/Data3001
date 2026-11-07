"""
Shared map and figure helpers. Every figure runs headless and is saved at 150 dpi.
Maps use cartopy when it is installed and plain matplotlib otherwise; MAP_KW and
TRANSFORM_KW are empty in that case, so calls read the same either way.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    MAP_KW = {"projection": ccrs.PlateCarree()}
    TRANSFORM_KW = {"transform": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    cfeature = None
    MAP_KW = {}
    TRANSFORM_KW = {}

LAND = "#e6e6e6"
DROGUE_COLORS = {"drogued": "#2a78d6", "undrogued": "#eb6834"}


def map_axes(ax, box, land=LAND, gridlines=False, coast_zorder=3):
    """Zoom a map to the box, with land and coastline on top of the data."""
    if not ccrs:
        return
    ax.set_extent(list(box), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color=land, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=coast_zorder)
    if gridlines:
        add_gridlines(ax)


def add_gridlines(ax):
    """Labelled gridlines along the left and bottom edges."""
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = gl.right_labels = False


def save_figure(path, trim=True):
    """Save the current figure and close it, trimmed to its contents unless trim is False."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight" if trim else None)
    plt.close()
