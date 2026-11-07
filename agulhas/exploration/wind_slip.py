"""
Agulhas Region — Undrogued Wind Slip
Measures how much faster and in which direction undrogued drifters move than
drogued drifters in the same grid cell: slip = undrogued − drogued mean
velocity per cell. Drogued drifters follow the ~15 m current with almost no
windage, so the slip is the extra surface (wind + wave) drift that undrogued
drifters, the oil proxy, already carry. It sets how much windage the oil
version still has to add.

Cells need ≥ MIN_DRIFTERS distinct drifters of each type. The two types do
not sample a cell at the same times, so part of a cell's slip is sampling
difference; latitude-band means and the bootstrap interval over whole
drifters show what is robust.

    python -m agulhas wind_slip --res 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import MIN_DRIFTERS, grid_tag, parse_args, region_prefix
from agulhas.drifters import Drifters, in_box
from agulhas.grid import Grid
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.stats import interval

N_BOOT = 500
SEED = 0
BAND_DEG = 5.0
BLOCK = 3.0        # arrows are block means: single cells carry sampling noise, blocks show the pattern


# ── Slip per cell ────────────────────────────────────────────────────────────
def per_drifter_means(drifters, grid):
    """Each drifter's mean velocity per cell, so a drifter that loiters for weeks does
    not dominate a cell."""
    lon, lat, ve, vn = drifters.lon, drifters.lat, drifters.ve, drifters.vn
    in_R = in_box(lon, lat, grid.box) & np.isfinite(ve) & np.isfinite(vn)
    df = pd.DataFrame({"cell": grid.cell_in_box(lon[in_R], lat[in_R]), "traj": drifters.traj_idx[in_R],
                       "drogued": drifters.drogue[in_R], "u": ve[in_R], "v": vn[in_R]})
    return df.groupby(["drogued", "cell", "traj"])[["u", "v"]].mean().reset_index()


def cell_means(pd_df):
    g = pd_df.groupby(["drogued", "cell"])
    out = g[["u", "v"]].mean()
    out["drifters"] = g.size()
    return out


def slip_table(cm):
    dro = cm.xs(True, level="drogued")
    und = cm.xs(False, level="drogued")
    both = dro.join(und, lsuffix="_dro", rsuffix="_und", how="inner")
    both = both[(both.drifters_dro >= MIN_DRIFTERS) & (both.drifters_und >= MIN_DRIFTERS)]
    both["su"] = both.u_und - both.u_dro
    both["sv"] = both.v_und - both.v_dro
    return both


def add_latitude(tab, grid):
    j, _ = np.divmod(tab.index.values, grid.n_lon)
    tab["lat"] = grid.lat_min + (j + 0.5) * grid.res


def cell_slip(per_drifter, grid):
    """Slip per cell with enough drifters of each type, with its position and speeds."""
    slip = slip_table(cell_means(per_drifter))
    slip["slip_speed"] = np.hypot(slip.su, slip.sv)
    slip["current_speed"] = np.hypot(slip.u_dro, slip.v_dro)
    add_latitude(slip, grid)
    _, i = np.divmod(slip.index.values, grid.n_lon)
    slip["lon"] = grid.lon_min + (i + 0.5) * grid.res
    return slip


# ── Latitude bands ───────────────────────────────────────────────────────────
def band_label(lo, hi):
    return f"{-hi:g}–{-lo:g}°S" if hi <= 0 else f"{lo:g}–{hi:g}°N" if lo >= 0 else f"{lo:g}–{hi:g}°"


def band_means(tab, bands):
    rows = []
    for lo, hi in bands:
        b = tab[(tab.lat >= lo) & (tab.lat < hi)]
        rows.append((b.su.mean(), b.sv.mean(), len(b)))
    return np.array(rows)


def resampled_band_means(per_drifter, bands, grid, rng):
    """Band means over N_BOOT resamples of drifters, each type resampled on its own and
    each drifter's cells kept together."""
    ids_of = {flag: per_drifter.loc[per_drifter.drogued == flag, "traj"].unique() for flag in (True, False)}
    boot = []
    for _ in range(N_BOOT):
        parts = []
        for flag, ids in ids_of.items():
            pick = pd.Series(rng.choice(ids, size=len(ids), replace=True)).value_counts()
            sub = per_drifter[(per_drifter.drogued == flag) & per_drifter.traj.isin(pick.index)]
            parts.append(sub.loc[sub.index.repeat(sub.traj.map(pick).values)])
        s = slip_table(cell_means(pd.concat(parts)))
        add_latitude(s, grid)
        boot.append(band_means(s, bands)[:, :2])
    return np.array(boot)


def band_table(est, boot, bands):
    """Print each band's mean slip with its interval, and return them as a table."""
    lo_ci, hi_ci = interval(boot)
    print("\nMean slip by latitude band (undrogued − drogued), cm/s, 95% interval over drifters:")
    print(f"{'Band':<12}{'cells':>6}{'east':>22}{'north':>22}{'speed':>8}{'toward':>9}")
    rows = []
    for k, (lo, hi) in enumerate(bands):
        su, sv, n = est[k]
        toward = (np.degrees(np.arctan2(su, sv)) + 360) % 360  # compass bearing the slip points to
        print(f"{band_label(lo, hi):<12}{int(n):>6}"
              f"{100 * su:>7.1f} ({100 * lo_ci[k, 0]:>5.1f}, {100 * hi_ci[k, 0]:>5.1f})"
              f"{100 * sv:>7.1f} ({100 * lo_ci[k, 1]:>5.1f}, {100 * hi_ci[k, 1]:>5.1f})"
              f"{100 * np.hypot(su, sv):>8.1f}{toward:>8.0f}°")
        rows.append({"lat_lo": lo, "lat_hi": hi, "cells": int(n), "slip_east": su, "slip_north": sv,
                     "east_lo": lo_ci[k, 0], "east_hi": hi_ci[k, 0],
                     "north_lo": lo_ci[k, 1], "north_hi": hi_ci[k, 1], "toward_deg": toward})
    return pd.DataFrame(rows)


# ── Map ──────────────────────────────────────────────────────────────────────
def plot_slip(slip, grid, path):
    """Slip speed per cell, with arrows of the mean slip per block."""
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=MAP_KW)
    sc = ax.scatter(slip.lon, slip.lat, c=100 * slip.slip_speed, cmap="Blues", s=55, marker="s",
                    vmin=0, vmax=np.percentile(100 * slip.slip_speed, 95), zorder=3, **TRANSFORM_KW)
    blk = slip.assign(bx=((slip.lon - grid.lon_min) // BLOCK), by=((slip.lat - grid.lat_min) // BLOCK))
    blk = blk.groupby(["bx", "by"]).agg(su=("su", "mean"), sv=("sv", "mean"), n=("su", "size")).reset_index()
    blk = blk[blk.n >= 3]
    q = ax.quiver(grid.lon_min + (blk.bx + 0.5) * BLOCK, grid.lat_min + (blk.by + 0.5) * BLOCK, blk.su, blk.sv,
                  color="#222222", scale=1.2, width=0.004, zorder=4, **TRANSFORM_KW)
    ax.quiverkey(q, 0.12, 0.93, 0.1, "10 cm/s slip", labelpos="E", coordinates="axes",
                 fontproperties={"size": 9})
    map_axes(ax, grid.box, gridlines=True, coast_zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, extend="max")
    cb.set_label("Slip speed, cm/s (undrogued − drogued mean velocity)")
    ax.set_title(f"Undrogued wind slip — colour per {grid.res:g}° cell (≥ {MIN_DRIFTERS} drifters of each type), "
                 f"arrows {BLOCK:g}° block means", fontsize=11)
    save_figure(path)
    print(f"Map saved → {path}")


def main(argv=None):
    args = parse_args(__doc__, grid=True, argv=argv)
    tag = grid_tag(args.res, args.box)
    grid = Grid(args.box, args.res)
    bands = [(lo, min(lo + BAND_DEG, grid.lat_max)) for lo in np.arange(grid.lat_min, grid.lat_max, BAND_DEG)]

    per_drifter = per_drifter_means(Drifters(args.data), grid)
    slip = cell_slip(per_drifter, grid)
    print(f"Cells with ≥ {MIN_DRIFTERS} drifters of each type: {len(slip)}")
    print(f"Median slip speed {100 * slip.slip_speed.median():.1f} cm/s; "
          f"median drogued current {100 * slip.current_speed.median():.1f} cm/s")

    boot = resampled_band_means(per_drifter, bands, grid, np.random.default_rng(SEED))
    table = band_table(band_means(slip, bands), boot, bands)

    slip.reset_index().to_csv(f"data/wind_slip_cells_{tag}.csv", index=False)
    table.to_csv(f"data/wind_slip_bands_{tag}.csv", index=False)
    print(f"\nSaved → data/wind_slip_cells_{tag}.csv, data/wind_slip_bands_{tag}.csv")
    plot_slip(slip, grid, f"figures/{region_prefix(args.data)}_wind_slip_{tag}.png")


if __name__ == "__main__":
    main()
