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

    .venv/bin/python scripts/07_wind_slip.py --res 1
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import xarray as xr

from config import MIN_DRIFTERS, box_tag, grid_shape, parse_args

args = parse_args(__doc__, grid=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES = args.res
N_BOOT = 500
BANDS = [(-45, -40), (-40, -35), (-35, -30), (-30, -25), (-25, -20), (-20, -15)]
rng = np.random.default_rng(0)

# ── 1. Load subset ───────────────────────────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
ve = ds["ve"].values.astype(float)
vn = ds["vn"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)

n_lon, n_lat = grid_shape(args.box, RES)
in_R = ((lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
        & np.isfinite(ve) & np.isfinite(vn))
cell = ((lat[in_R] - LAT_MIN) // RES).astype(int) * n_lon + ((lon[in_R] - LON_MIN) // RES).astype(int)
df = pd.DataFrame({"cell": cell, "traj": traj_idx[in_R], "drogued": drogue[in_R],
                   "u": ve[in_R], "v": vn[in_R]})

# ── 2. Per-drifter cell means, then cell means per type ──────────────────────
# Averaging per drifter first weights every drifter equally in a cell, so a
# drifter that loiters for weeks does not dominate it
per_drifter = df.groupby(["drogued", "cell", "traj"])[["u", "v"]].mean().reset_index()


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


slip = slip_table(cell_means(per_drifter))
slip["slip_speed"] = np.hypot(slip.su, slip.sv)
slip["current_speed"] = np.hypot(slip.u_dro, slip.v_dro)
j, i = np.divmod(slip.index.values, n_lon)
slip["lat"] = LAT_MIN + (j + 0.5) * RES
slip["lon"] = LON_MIN + (i + 0.5) * RES
print(f"Cells with ≥ {MIN_DRIFTERS} drifters of each type: {len(slip)}")
print(f"Median slip speed {100 * slip.slip_speed.median():.1f} cm/s; "
      f"median drogued current {100 * slip.current_speed.median():.1f} cm/s")


# ── 3. Latitude-band mean slip with a bootstrap over whole drifters ──────────
def band_means(tab):
    rows = []
    for lo, hi in BANDS:
        b = tab[(tab.lat >= lo) & (tab.lat < hi)]
        rows.append((b.su.mean(), b.sv.mean(), len(b)))
    return np.array(rows)


est = band_means(slip)
drifters = {flag: per_drifter.loc[per_drifter.drogued == flag, "traj"].unique() for flag in (True, False)}
boot = []
for _ in range(N_BOOT):
    # Resample drifters of each type independently, keeping each drifter's cells together
    parts = []
    for flag, ids in drifters.items():
        pick = pd.Series(rng.choice(ids, size=len(ids), replace=True)).value_counts()
        sub = per_drifter[(per_drifter.drogued == flag) & per_drifter.traj.isin(pick.index)]
        parts.append(sub.loc[sub.index.repeat(sub.traj.map(pick).values)])
    s = slip_table(cell_means(pd.concat(parts)))
    j, i = np.divmod(s.index.values, n_lon)
    s["lat"] = LAT_MIN + (j + 0.5) * RES
    boot.append(band_means(s)[:, :2])
boot = np.array(boot)
lo_ci, hi_ci = np.nanpercentile(boot, [2.5, 97.5], axis=0)

print("\nMean slip by latitude band (undrogued − drogued), cm/s, 95% interval over drifters:")
print(f"{'Band':<12}{'cells':>6}{'east':>22}{'north':>22}{'speed':>8}{'toward':>9}")
bands = []
for k, (lo, hi) in enumerate(BANDS):
    su, sv, n = est[k]
    toward = (np.degrees(np.arctan2(su, sv)) + 360) % 360  # compass bearing the slip points to
    print(f"{abs(hi)}–{abs(lo)}°S{'':<4}{int(n):>6}"
          f"{100 * su:>7.1f} ({100 * lo_ci[k, 0]:>5.1f}, {100 * hi_ci[k, 0]:>5.1f})"
          f"{100 * sv:>7.1f} ({100 * lo_ci[k, 1]:>5.1f}, {100 * hi_ci[k, 1]:>5.1f})"
          f"{100 * np.hypot(su, sv):>8.1f}{toward:>8.0f}°")
    bands.append({"lat_lo": lo, "lat_hi": hi, "cells": int(n), "slip_east": su, "slip_north": sv,
                  "east_lo": lo_ci[k, 0], "east_hi": hi_ci[k, 0],
                  "north_lo": lo_ci[k, 1], "north_hi": hi_ci[k, 1], "toward_deg": toward})

tag = f"{f'{RES:g}'.replace('.', 'p')}deg_{box_tag(args.box)}"
slip.reset_index().to_csv(f"data/wind_slip_cells_{tag}.csv", index=False)
pd.DataFrame(bands).to_csv(f"data/wind_slip_bands_{tag}.csv", index=False)
print(f"\nSaved → data/wind_slip_cells_{tag}.csv, data/wind_slip_bands_{tag}.csv")

# ── 4. Map ───────────────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=subplot_kw)
sc = ax.scatter(slip.lon, slip.lat, c=100 * slip.slip_speed, cmap="Blues", s=55, marker="s",
                vmin=0, vmax=np.percentile(100 * slip.slip_speed, 95), zorder=3, **kw)
# Arrows are 3° block means: single cells carry sampling noise, blocks show the pattern
BLOCK = 3.0
blk = slip.assign(bx=((slip.lon - LON_MIN) // BLOCK), by=((slip.lat - LAT_MIN) // BLOCK))
blk = blk.groupby(["bx", "by"]).agg(su=("su", "mean"), sv=("sv", "mean"), n=("su", "size")).reset_index()
blk = blk[blk.n >= 3]
q = ax.quiver(LON_MIN + (blk.bx + 0.5) * BLOCK, LAT_MIN + (blk.by + 0.5) * BLOCK, blk.su, blk.sv,
              color="#222222", scale=1.2, width=0.004, zorder=4, **kw)
ax.quiverkey(q, 0.12, 0.93, 0.1, "10 cm/s slip", labelpos="E", coordinates="axes",
             fontproperties={"size": 9})
if ccrs:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = gl.right_labels = False
cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, extend="max")
cb.set_label("Slip speed, cm/s (undrogued − drogued mean velocity)")
ax.set_title(f"Undrogued wind slip — colour per {RES:g}° cell (≥ {MIN_DRIFTERS} drifters of each type), arrows {BLOCK:g}° block means",
             fontsize=11)
out = f"figures/agulhas_wind_slip_{tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Map saved → {out}")
plt.close()
