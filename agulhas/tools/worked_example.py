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
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

from agulhas.config import DISAGREE_TVD, OIL_WINDAGE_RANGE, operator_path, parse_args, place_label, region_prefix, setting_tag
from agulhas.transport import (COAST_NAMES, EXIT_LABELS, STRANDED, absorption, cell_centres, load_operator, propagate,
                       reliable, state_of, to_grid)


def add_options(p):
    p.add_argument("--lon", type=float, required=True, help="release longitude")
    p.add_argument("--lat", type=float, required=True, help="release latitude")
    p.add_argument("--matrix", default="undrogued", choices=["drogued", "undrogued"] + list(OIL_WINDAGE_RANGE))

HORIZONS = [7, 30, 365]
MAX_MOVE_KM = 150   # how far a release may be moved to reach an ocean cell


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
    RES, TAU = args.res, args.tau
    TAG = setting_tag(RES, TAU, args.box)

    # ── 1. The matrix, and the cell the release falls in ─────────────────────
    path = operator_path(args.matrix, RES, TAU, args.box)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Build it first: transition_matrix for drogued and undrogued, oil_wind for oil.")
    op = load_operator(path)
    cells = op["cell_flat"]
    n_cells = len(cells)
    cell_lon, cell_lat = cell_centres(op)
    ok = reliable(op)
    if not (LON_MIN <= args.lon < LON_MAX and LAT_MIN <= args.lat < LAT_MAX):
        raise SystemExit(f"{place_label(args.lon, args.lat)} is outside the box {list(args.box)}.")

    def km_to(states):
        dx = (cell_lon[states] - args.lon) * 111.32 * np.cos(np.radians(args.lat))
        return np.hypot(dx, (cell_lat[states] - args.lat) * 110.57)

    # A point on the coast or in a cell no drifter visited has no row, so use the nearest ocean cell
    try:
        state = state_of(op, args.lon, args.lat)
    except ValueError:
        state = int(np.argmin(km_to(np.arange(n_cells))))
        if km_to(state) > MAX_MOVE_KM:
            raise SystemExit(f"{place_label(args.lon, args.lat)} is {km_to(state):.0f} km from the nearest ocean cell: "
                             "is it on land?")
        print(f"Note: no ocean cell at this point, so using the nearest, {km_to(state):.0f} km away.")
    here = place_label(cell_lon[state], cell_lat[state])
    print(f"Release: {place_label(args.lon, args.lat)}, in the {RES:g}° cell at {here}; {args.matrix} matrix, τ = {TAU:g} days")

    # ── 2. Can this cell be trusted? ─────────────────────────────────────────
    print(f"Data behind this cell: {int(op['row_drifters'][state])} drifters.")
    if not ok[state]:
        usable = np.where(ok)[0]
        near = usable[np.argmin(km_to(usable))]
        print(f"  WARNING: this cell is thin (fewer than 10 drifters, or no moves of its own), so treat the "
              f"numbers below as very uncertain.\n  The nearest reliable cell is {place_label(cell_lon[near], cell_lat[near])}, "
              f"{km_to(near):.0f} km away; rerun from there to compare.")

    # ── 3. Step forward: intervals come from the atlas, if it has been built ─
    p0 = np.zeros(op["P"].shape[0])
    p0[state] = 1.0
    dist = {d: propagate(op, p0, days=d) for d in HORIZONS}
    atlas_path = f"data/risk_atlas_{TAG}.csv"
    atlas = None
    if os.path.exists(atlas_path):
        atlas = pd.read_csv(atlas_path, low_memory=False)
        atlas = atlas[(atlas["matrix"] == args.matrix) & (atlas["state"] == state)]
        atlas = atlas if len(atlas) else None

    def with_interval(value, horizon, outcome):
        if atlas is not None:
            row = atlas[(atlas["horizon"].astype(str) == str(horizon)) & (atlas["outcome"] == outcome)]
            if len(row):
                return f"{value:>5.1%} ({row['lo'].iloc[0]:.0%}–{row['hi'].iloc[0]:.0%})"
        return f"{value:>5.1%}"

    edges = list(EXIT_LABELS[:STRANDED])
    print(f"\n{'':<12}{'afloat in R':<20}" + "".join(f"{'exit ' + e:<18}" for e in edges) + "stranded")
    for d in HORIZONS:
        p = dist[d]
        print(f"{d:>4} days   {with_interval(p[:n_cells].sum(), d, 'in_R'):<20}"
              + "".join(f"{with_interval(p[n_cells + k], d, e):<18}" for k, e in enumerate(edges))
              + with_interval(p[n_cells + STRANDED], d, "stranded_any"))
    steps, fate = absorption(op)
    print("\nEventually: " + ", ".join(f"exit {e} {fate[state, k]:.0%}" for k, e in enumerate(edges))
          + f", stranded {fate[state, STRANDED]:.0%}. Mean time afloat in R: {steps[state] * TAU:.0f} days.")

    # ── 4. Which coast? ──────────────────────────────────────────────────────
    coast_path = operator_path(f"{args.matrix}_coast", RES, TAU, args.box)
    if os.path.exists(coast_path):
        coast_op = load_operator(coast_path)
        coast_fate = absorption(coast_op)[1][state, STRANDED:]
        print("\nWhere it strands, eventually:")
        for k in np.argsort(-coast_fate):
            if coast_fate[k] >= 0.005:
                print(f"  {coast_fate[k]:>5.1%}  {COAST_NAMES[k]}")
        if atlas is not None:
            days = atlas[atlas["outcome"] == "days_to_strand"]
            if len(days) and np.isfinite(days["estimate"].iloc[0]):
                r = days.iloc[0]
                print(f"  Among material that strands, it takes {r['estimate']:.0f} days on average ({r['lo']:.0f}–{r['hi']:.0f}).")

    # ── 5. For oil: does the answer depend on the method? ────────────────────
    if args.matrix in OIL_WINDAGE_RANGE:
        check = load_operator(operator_path(args.matrix.replace("oil", "oil_check"), RES, TAU, args.box))
        gap = 0.5 * np.abs(dist[30] - propagate(check, p0, days=30)).sum()
        print(f"\nMethod check at 30 days: oil against its drogued check differ by TVD {gap:.2f}"
              + (f", above {DISAGREE_TVD}, so this oil result depends on the method." if gap > DISAGREE_TVD
                 else f", within {DISAGREE_TVD}."))

    # ── 6. Map ───────────────────────────────────────────────────────────────
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        map_kw = {"projection": ccrs.PlateCarree()}
    except ImportError:
        ccrs, map_kw = None, {}
    kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15, 4.6), subplot_kw=map_kw)
    for ax, d in zip(axes, HORIZONS):
        grid = to_grid(op, dist[d])
        mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], np.where(grid > 0, grid, np.nan), cmap="Blues",
                             vmin=0, vmax=max(np.nanmax(grid), 1e-6), **kw)
        if ccrs:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.plot(args.lon, args.lat, marker="*", color="#d03b3b", ms=11, mec="white", ls="", zorder=5, **kw)
        ax.set_title(f"{d} days: {dist[d][:n_cells].sum():.0%} afloat in R, {dist[d][n_cells + STRANDED]:.0%} stranded", fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=0.75, label="Share of the release per cell")
    fig.suptitle(f"Release at {place_label(args.lon, args.lat)} — {args.matrix}, {RES:g}°, τ = {TAU:g} d", fontsize=12)
    slug = f"{args.lon:g}_{args.lat:g}".replace("-", "m").replace(".", "p")
    out = f"figures/{region_prefix(args.data)}_worked_example_{args.matrix}_{slug}_{TAG}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nMap saved → {out}")


if __name__ == "__main__":
    main()
