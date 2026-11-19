"""
Agulhas Region — Currents Slide
The shipping map of shipping_overlay redrawn for a slide: commercial shipping
lanes under streamlines of the mean surface current, thicker where it is faster
and fading out below MIN_SPEED, with the main currents named and the port
hotspots starred. The mean current is from one drogue type on a finer grid than
the matrices (CURRENT_RES), since the Agulhas Current is narrow. Not part of the
pipeline; it reads the shipping density saved by shipping_overlay and the drifter
subset, so run shipping_overlay first.

    python -m agulhas currents_slide                   # undrogued drifters
    python -m agulhas currents_slide --type drogued
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

from agulhas.config import DROGUE_TYPES, parse_args, region_prefix, setting_tag
from agulhas.drifters import Drifters, in_box
from agulhas.grid import Grid
from agulhas.plotting import MAP_KW, TRANSFORM_KW, add_gridlines, map_axes, save_figure
from agulhas.release_points.shipping_overlay import HOTSPOTS, mean_currents

CURRENT_RES = 0.5   # degrees, grid of the mean current
MIN_SPEED = 0.15    # m/s; slower mean flow, mostly eddy noise, fades out below this
FULL_SPEED = 1.0    # m/s; streamlines are widest from this speed up
CURRENT_COLOR = "#c2410c"
LABEL_COLOR = "#7c2d12"
PORT_COLOR = "#facc15"
# Named currents (label: lon, lat); those outside the box are left out
CURRENT_NAMES = {
    "Agulhas\nCurrent": (33.6, -32.6),
    "Retroflection": (17.0, -37.0),
    "Agulhas Return Current": (40.0, -42.6),
    "Mozambique\nChannel": (40.0, -24.0),
    "East\nMadagascar\nCurrent": (52.6, -19.5),
    "Benguela\nCurrent": (12.6, -28.5),
}


def add_options(p):
    p.add_argument("--type", default="undrogued", choices=list(DROGUE_TYPES), help="drifters behind the mean current")


# ── Inputs ───────────────────────────────────────────────────────────────────

def load_shipping(path):
    """Shipping density on its 0.1° display grid, relative to the busiest cell (NaN where none)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run shipping_overlay first.")
    d = np.load(path)
    ship, fine = d["ship_fine"], float(d["fine"])
    lon = np.arange(ship.shape[1] + 1) * fine + float(d["lon_min"])
    lat = np.arange(ship.shape[0] + 1) * fine + float(d["lat_min"])
    return lon, lat, np.where(ship > 0, ship / ship.max(), np.nan)


# ── Map ──────────────────────────────────────────────────────────────────────

def plot_currents(ax, grid, current):
    """Streamlines of the mean current, width by speed, transparent below MIN_SPEED.
    Cells without enough data count as still water, so streamlines end there."""
    u, v = np.nan_to_num(current[0]), np.nan_to_num(current[1])
    speed = np.hypot(u, v)
    opacity = np.clip((speed - 0.6 * MIN_SPEED) / (0.6 * MIN_SPEED), 0, 1)
    fade = LinearSegmentedColormap.from_list("fade", [CURRENT_COLOR + "00"] * 2 + [CURRENT_COLOR + "ff"] * 2)
    width = 1.0 + 5 * np.clip((speed - MIN_SPEED) / (FULL_SPEED - MIN_SPEED), 0, 1)
    ax.streamplot(grid.lon_centres, grid.lat_centres, u, v, density=2.4, color=0.2 + 0.6 * opacity, cmap=fade,
                  norm=Normalize(0, 1), linewidth=width, arrowsize=1.3, zorder=4, **TRANSFORM_KW)


def plot_labels(ax, box):
    halo = [pe.withStroke(linewidth=4, foreground="white")]
    for label, (lon, lat) in CURRENT_NAMES.items():
        if in_box(lon, lat, box):
            ax.text(lon, lat, label, fontsize=15, fontweight="bold", fontstyle="italic", color=LABEL_COLOR,
                    ha="center", va="center", zorder=8, path_effects=halo, **TRANSFORM_KW)
    for lon, lat in HOTSPOTS.values():
        if in_box(lon, lat, box):
            ax.plot(lon, lat, marker="*", ms=17, mfc=PORT_COLOR, mec="#222222", zorder=7, **TRANSFORM_KW)


def plot_slide(shipping, grid, current, path):
    lon, lat, rel = shipping
    lo, hi = np.nanpercentile(rel, [60, 99.5])
    with plt.rc_context({"font.size": 15}):
        draw_slide(lon, lat, rel, lo, hi, grid, current)
        save_figure(path)


def draw_slide(lon, lat, rel, lo, hi, grid, current):
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw=MAP_KW)
    mesh = ax.pcolormesh(lon, lat, rel, cmap="Blues", norm=LogNorm(vmin=lo, vmax=hi), **TRANSFORM_KW)
    plot_currents(ax, grid, current)
    plot_labels(ax, grid.box)
    map_axes(ax, grid.box)
    add_gridlines(ax)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02, extend="both")
    cb.set_label("Commercial ship traffic (relative, log scale)")
    cb.set_ticks([])
    cb.minorticks_off()
    handles = [plt.Line2D([], [], color=CURRENT_COLOR, lw=4,
                          label=f"Mean surface current, over {MIN_SPEED:g} m/s (thicker = faster)"),
               plt.Line2D([], [], ls="", marker="*", ms=17, mfc=PORT_COLOR, mec="#222222", label="Major port")]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out = f"figures/{region_prefix(args.data)}_currents_slide_{args.type}_{tag}.png"

    shipping = load_shipping(f"data/shipping_{tag}.npz")
    grid = Grid(args.box, CURRENT_RES)
    current = mean_currents(Drifters(args.data), grid)[args.type]
    plot_slide(shipping, grid, current, out)
    print(f"Slide map saved → {out}")


if __name__ == "__main__":
    main()
