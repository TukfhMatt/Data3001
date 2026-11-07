"""
Agulhas Region — Worked Example
Something goes into the water at a point: where is it after a week, a month and
a year? This steps a saved matrix forward from any point and reports the answer
with what a reader needs to judge it: the reliability of the release cell, 95%
intervals from the atlas (risk_atlas), the coast that receives the stranding
(coastal_stranding), and for oil whether the result depends on the method (oil_wind). Not part of the pipeline;
it only reads its outputs, so run the pipeline (or at least transition_matrix) first.

    python -m agulhas worked_example --lon 40.5 --lat -16.5 --matrix oil
    python -m agulhas worked_example --lon -70.5 --lat 35.5 --matrix undrogued \
        --box -80 -40 25 45 --data data/gulfstream_gdp1h_subset.nc
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import (DISAGREE_TVD, OIL_WINDAGE_RANGE, operator_path, parse_args, place_label, region_prefix,
                            setting_tag)
from agulhas.drifters import in_box
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import (COAST_NAMES, EXIT_LABELS, STRANDED, absorption, cell_centres, load_operator, propagate,
                               reliable, state_of, to_grid)

HORIZONS = [7, 30, 365]
MAX_MOVE_KM = 150   # how far a release may be moved to reach an ocean cell
EDGES = list(EXIT_LABELS[:STRANDED])


def add_options(p):
    p.add_argument("--lon", type=float, required=True, help="release longitude")
    p.add_argument("--lat", type=float, required=True, help="release latitude")
    p.add_argument("--matrix", default="undrogued", choices=["drogued", "undrogued"] + list(OIL_WINDAGE_RANGE))


def km_from(op, lon, lat, states):
    """Distance in km from (lon, lat) to the centres of the given states."""
    cell_lon, cell_lat = cell_centres(op)
    dx = (cell_lon[states] - lon) * 111.32 * np.cos(np.radians(lat))
    return np.hypot(dx, (cell_lat[states] - lat) * 110.57)


# ── The matrix, and the cell the release falls in ────────────────────────────

def load_matrix(args):
    path = operator_path(args.matrix, args.res, args.tau, args.box)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Build it first: transition_matrix for drogued and undrogued, oil_wind for oil.")
    if not in_box(args.lon, args.lat, args.box):
        raise SystemExit(f"{place_label(args.lon, args.lat)} is outside the box {list(args.box)}.")
    return load_operator(path)


def release_state(op, lon, lat):
    """The release's ocean cell. A point on the coast or in a cell no drifter
    visited has no row, so it moves to the nearest ocean cell."""
    try:
        return state_of(op, lon, lat)
    except ValueError:
        state = int(np.argmin(km_from(op, lon, lat, np.arange(len(op["cell_flat"])))))
        km = km_from(op, lon, lat, state)
        if km > MAX_MOVE_KM:
            raise SystemExit(f"{place_label(lon, lat)} is {km:.0f} km from the nearest ocean cell: is it on land?")
        print(f"Note: no ocean cell at this point, so using the nearest, {km:.0f} km away.")
        return state


def print_release(op, state, args):
    """Where the release is, and whether its cell can be trusted."""
    cell_lon, cell_lat = cell_centres(op)
    here = place_label(cell_lon[state], cell_lat[state])
    print(f"Release: {place_label(args.lon, args.lat)}, in the {args.res:g}° cell at {here}; {args.matrix} matrix, "
          f"τ = {args.tau:g} days")
    print(f"Data behind this cell: {int(op['row_drifters'][state])} drifters.")
    ok = reliable(op)
    if not ok[state]:
        usable = np.where(ok)[0]
        near = usable[np.argmin(km_from(op, args.lon, args.lat, usable))]
        print(f"  WARNING: this cell is thin (fewer than 10 drifters, or no moves of its own), so treat the "
              f"numbers below as very uncertain.\n  The nearest reliable cell is {place_label(cell_lon[near], cell_lat[near])}, "
              f"{km_from(op, args.lon, args.lat, near):.0f} km away; rerun from there to compare.")


# ── Step forward: intervals come from the atlas, if it has been built ────────

def load_atlas_rows(path, matrix, state):
    """The atlas's rows for this release cell, or None."""
    if not os.path.exists(path):
        return None
    atlas = pd.read_csv(path, low_memory=False)
    atlas = atlas[(atlas["matrix"] == matrix) & (atlas["state"] == state)]
    return atlas if len(atlas) else None


def with_interval(atlas, value, horizon, outcome):
    if atlas is not None:
        row = atlas[(atlas["horizon"].astype(str) == str(horizon)) & (atlas["outcome"] == outcome)]
        if len(row):
            return f"{value:>5.1%} ({row['lo'].iloc[0]:.0%}–{row['hi'].iloc[0]:.0%})"
    return f"{value:>5.1%}"


def print_fates(op, state, dist, atlas, tau):
    n_cells = len(op["cell_flat"])
    print(f"\n{'':<12}{'afloat in R':<20}" + "".join(f"{'exit ' + e:<18}" for e in EDGES) + "stranded")
    for d in HORIZONS:
        p = dist[d]
        print(f"{d:>4} days   {with_interval(atlas, p[:n_cells].sum(), d, 'in_R'):<20}"
              + "".join(f"{with_interval(atlas, p[n_cells + k], d, e):<18}" for k, e in enumerate(EDGES))
              + with_interval(atlas, p[n_cells + STRANDED], d, "stranded_any"))
    steps, fate = absorption(op)
    print("\nEventually: " + ", ".join(f"exit {e} {fate[state, k]:.0%}" for k, e in enumerate(EDGES))
          + f", stranded {fate[state, STRANDED]:.0%}. Mean time afloat in R: {steps[state] * tau:.0f} days.")


def print_coasts(args, state, atlas):
    """Which coast receives the stranding, when the coast operator has been built."""
    coast_path = operator_path(f"{args.matrix}_coast", args.res, args.tau, args.box)
    if not os.path.exists(coast_path):
        return
    coast_fate = absorption(load_operator(coast_path))[1][state, STRANDED:]
    print("\nWhere it strands, eventually:")
    for k in np.argsort(-coast_fate):
        if coast_fate[k] >= 0.005:
            print(f"  {coast_fate[k]:>5.1%}  {COAST_NAMES[k]}")
    if atlas is not None:
        days = atlas[atlas["outcome"] == "days_to_strand"]
        if len(days) and np.isfinite(days["estimate"].iloc[0]):
            r = days.iloc[0]
            print(f"  Among material that strands, it takes {r['estimate']:.0f} days on average ({r['lo']:.0f}–{r['hi']:.0f}).")


def print_method_check(args, p0, dist):
    """For oil: does the answer depend on the method?"""
    if args.matrix not in OIL_WINDAGE_RANGE:
        return
    check = load_operator(operator_path(args.matrix.replace("oil", "oil_check"), args.res, args.tau, args.box))
    gap = 0.5 * np.abs(dist[30] - propagate(check, p0, days=30)).sum()
    print(f"\nMethod check at 30 days: oil against its drogued check differ by TVD {gap:.2f}"
          + (f", above {DISAGREE_TVD}, so this oil result depends on the method." if gap > DISAGREE_TVD
             else f", within {DISAGREE_TVD}."))


# ── Map ──────────────────────────────────────────────────────────────────────

def plot_release(op, dist, args, path):
    n_cells = len(op["cell_flat"])
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15, 4.6), subplot_kw=MAP_KW)
    for ax, d in zip(axes, HORIZONS):
        grid = to_grid(op, dist[d])
        mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], np.where(grid > 0, grid, np.nan), cmap="Blues",
                             vmin=0, vmax=max(np.nanmax(grid), 1e-6), **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.plot(args.lon, args.lat, marker="*", color="#d03b3b", ms=11, mec="white", ls="", zorder=5, **TRANSFORM_KW)
        ax.set_title(f"{d} days: {dist[d][:n_cells].sum():.0%} afloat in R, {dist[d][n_cells + STRANDED]:.0%} stranded",
                     fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=0.75, label="Share of the release per cell")
    fig.suptitle(f"Release at {place_label(args.lon, args.lat)} — {args.matrix}, {args.res:g}°, τ = {args.tau:g} d",
                 fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    slug = f"{args.lon:g}_{args.lat:g}".replace("-", "m").replace(".", "p")
    out = f"figures/{region_prefix(args.data)}_worked_example_{args.matrix}_{slug}_{tag}.png"

    op = load_matrix(args)
    state = release_state(op, args.lon, args.lat)
    print_release(op, state, args)

    p0 = np.zeros(op["P"].shape[0])
    p0[state] = 1.0
    dist = {d: propagate(op, p0, days=d) for d in HORIZONS}
    atlas = load_atlas_rows(f"data/risk_atlas_{tag}.csv", args.matrix, state)
    print_fates(op, state, dist, atlas, args.tau)
    print_coasts(args, state, atlas)
    print_method_check(args, p0, dist)

    plot_release(op, dist, args, out)
    print(f"\nMap saved → {out}")


if __name__ == "__main__":
    main()
