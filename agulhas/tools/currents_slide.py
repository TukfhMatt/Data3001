"""
Agulhas Region — Currents Slide
The shipping map of shipping_overlay redrawn for a slide: commercial shipping
lanes under the mean surface current, drawn as arrows (length by speed, one per
ARROW_SPACING) or as streamlines (thicker where faster, fading out below
MIN_SPEED), with the main currents named and the ports marked and named. The mean current is from one drogue type on a finer grid than
the matrices (CURRENT_RES), since the Agulhas Current is narrow. Not part of the
pipeline; it reads the shipping density saved by shipping_overlay and the drifter
subset, so run shipping_overlay first.

    python -m agulhas currents_slide                   # undrogued drifters
    python -m agulhas currents_slide --type drogued
    python -m agulhas currents_slide --style streamlines
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
from agulhas.release_points.shipping_overlay import mean_currents

CURRENT_RES = 0.5   # degrees, grid of the mean current
MIN_SPEED = 0.15    # m/s; slower mean flow, mostly eddy noise, fades out below this
FULL_SPEED = 1.0    # m/s; streamlines are widest from this speed up
ARROW_SPACING = 1.5  # degrees between arrows
ARROW_KEY = 0.5     # m/s, the length shown in the arrow key
MIN_ARROW = 0.05    # m/s; slower cells get no arrow, which would draw as a dot
CURRENT_COLOR = "#c2410c"
LABEL_COLOR = "#7c2d12"
PORT_COLOR = "#222222"
# Named currents (label: lon, lat); those outside the box are left out
CURRENT_NAMES = {
    "Agulhas\nCurrent": (33.6, -32.6),
    "Retroflection": (17.0, -37.0),
    "Agulhas Return Current": (40.0, -42.6),
    "Mozambique\nChannel": (40.0, -24.0),
    "East\nMadagascar\nCurrent": (52.6, -19.5),
    "Benguela\nCurrent": (12.6, -28.5),
}
# Port cities, set about 0.3° inland so the points sit clear of the coast at this scale, with where each name
# sits beside its point (offset in points, alignment); those outside the box are left out
PORTS = {
    "Richards Bay": (31.75, -28.6, (-10, 3), "right"),
    "Durban": (30.7, -29.75, (-10, -3), "right"),
    "Gqeberha": (25.5, -33.55, (8, 10), "left"),
    "Cape Town": (18.8, -33.8, (10, -4), "left"),
    "Saldanha Bay": (18.35, -32.85, (10, 6), "left"),
}


def add_options(p):
    p.add_argument("--type", default="undrogued", choices=list(DROGUE_TYPES), help="drifters behind the mean current")
    p.add_argument("--style", default="arrows", choices=["arrows", "streamlines"], help="how the current is drawn")


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

def plot_arrows(ax, grid, current):
    """One arrow per ARROW_SPACING, its length the mean speed; none where data are too few
    or the flow is slower than MIN_ARROW. The key sits on land, clear of the arrows."""
    step = max(1, int(round(ARROW_SPACING / grid.res)))
    sub = (slice(step // 2, None, step), slice(step // 2, None, step))
    lon, lat = np.meshgrid(grid.lon_centres[sub[1]], grid.lat_centres[sub[0]])
    u, v = current[0][sub], current[1][sub]
    slow = ~(np.hypot(u, v) >= MIN_ARROW)
    q = ax.quiver(lon, lat, np.where(slow, np.nan, u), np.where(slow, np.nan, v), color=CURRENT_COLOR, scale=13,
                  width=0.0024, headwidth=3.5, headlength=4, zorder=4, **TRANSFORM_KW)
    ax.quiverkey(q, 0.25, 0.80, ARROW_KEY, f"{ARROW_KEY:g} m/s", labelpos="E", coordinates="axes",
                 color=CURRENT_COLOR, zorder=9, fontproperties={"size": 12})


def plot_streamlines(ax, grid, current):
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
    for name, (lon, lat, offset, align) in PORTS.items():
        if in_box(lon, lat, box):
            ax.plot(lon, lat, marker="o", ms=9, mfc=PORT_COLOR, mec="white", mew=1.5, zorder=7, **TRANSFORM_KW)
            ax.annotate(name, (lon, lat), xytext=offset, textcoords="offset points", fontsize=13,
                        color="#222222", ha=align, va="center", zorder=8, path_effects=halo)


def plot_slide(shipping, grid, current, style, path):
    lon, lat, rel = shipping
    lo, hi = np.nanpercentile(rel, [60, 99.5])
    with plt.rc_context({"font.size": 15}):
        draw_slide(lon, lat, rel, lo, hi, grid, current, style)
        save_figure(path)


def draw_slide(lon, lat, rel, lo, hi, grid, current, style):
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw=MAP_KW)
    mesh = ax.pcolormesh(lon, lat, rel, cmap="Blues", norm=LogNorm(vmin=lo, vmax=hi), **TRANSFORM_KW)
    (plot_arrows if style == "arrows" else plot_streamlines)(ax, grid, current)
    plot_labels(ax, grid.box)
    map_axes(ax, grid.box)
    add_gridlines(ax)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02, extend="both")
    cb.set_label("Commercial ship traffic (relative, log scale)")
    cb.set_ticks([])
    cb.minorticks_off()
    label = ("Mean surface current (longer = faster)" if style == "arrows"
             else f"Mean surface current, over {MIN_SPEED:g} m/s (thicker = faster)")
    current_handle = (plt.Line2D([], [], ls="", marker=r"$\rightarrow$", ms=18, color=CURRENT_COLOR, label=label)
                      if style == "arrows" else plt.Line2D([], [], color=CURRENT_COLOR, lw=3, label=label))
    handles = [current_handle,
               plt.Line2D([], [], ls="", marker="o", ms=9, mfc=PORT_COLOR, mec="white", mew=1.5, label="Major port")]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95, fontsize=11, handlelength=1.5)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out = f"figures/{region_prefix(args.data)}_currents_slide_{args.type}_{args.style}_{tag}.png"

    shipping = load_shipping(f"data/shipping_{tag}.npz")
    grid = Grid(args.box, CURRENT_RES)
    current = mean_currents(Drifters(args.data), grid)[args.type]
    plot_slide(shipping, grid, current, args.style, out)
    print(f"Slide map saved → {out}")


if __name__ == "__main__":
    main()
