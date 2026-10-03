"""
Agulhas Region — Residence Time in R
Expected time material released in each cell stays afloat in R before it
leaves across an edge or strands, for every grid and lag and both drogue
types. It comes exactly from the absorbing chain (transport.absorption):
N·1 is the expected number of τ steps up to and including the step in which
the material leaves, and the material is taken to leave halfway through that
step, so

    residence time = τ · (N·1 − ½)

Summaries are over reliable cells (not flagged, not empty).

Needs the matrices from 03.

    .venv/bin/python scripts/14_residence_time.py --res 0.5 1 2 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

from config import DROGUE_TYPES, RES, operator_path, parse_args, setting_tag
from transport import absorption, cell_centres, load_operator, reliable, to_grid

args = parse_args(__doc__, grid=True, lag=True, multi=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
TAG = setting_tag(args.res, args.tau, args.box)
MAP_RES = RES if RES in args.res else args.res[0]

os.makedirs("figures", exist_ok=True)

# ── 1. Residence time per grid, lag and drogue type ──────────────────────────
summary, cells_out, maps = [], [], {}
for tau in args.tau:
    for res in args.res:
        for name in DROGUE_TYPES:
            op = load_operator(operator_path(name, res, tau, args.box))
            steps, _ = absorption(op)
            days = tau * (steps - 0.5)
            ok = reliable(op)
            lon, lat = cell_centres(op)
            d = days[ok]
            top = np.argmax(np.where(ok, days, -np.inf))
            summary.append({"res": res, "tau": tau, "drogue": name, "reliable_cells": int(ok.sum()),
                            "mean_days": d.mean(), "median_days": np.median(d),
                            "q25_days": np.percentile(d, 25), "q75_days": np.percentile(d, 75),
                            "q90_days": np.percentile(d, 90), "max_days": d.max(),
                            "max_lon": lon[top], "max_lat": lat[top]})
            cells_out.append(pd.DataFrame({"res": res, "tau": tau, "drogue": name, "lon": lon, "lat": lat,
                                           "residence_days": days, "reliable": ok,
                                           "drifters": op["row_drifters"]}))
            if res == MAP_RES and tau == args.tau[0]:
                maps[name] = (op, np.where(ok, days, np.nan))

summary = pd.DataFrame(summary)
print("Expected residence time in R over reliable cells (days)")
print(f"{'grid':>5} {'τ':>4} {'type':<10}{'cells':>7}{'mean':>7}{'median':>8}{'q25–q75':>12}{'q90':>6}"
      f"{'max':>6}  longest at")
for r in summary.itertuples():
    print(f"{r.res:>4g}° {r.tau:>4g} {r.drogue:<10}{r.reliable_cells:>7}{r.mean_days:>7.0f}{r.median_days:>8.0f}"
          f"{r.q25_days:>7.0f}–{r.q75_days:<4.0f}{r.q90_days:>6.0f}{r.max_days:>6.0f}  "
          f"{r.max_lon:.1f}°E {-r.max_lat:.1f}°S")

out_summary = f"data/residence_time_{TAG}.csv"
out_cells = f"data/residence_time_cells_{TAG}.csv"
summary.to_csv(out_summary, index=False)
pd.concat(cells_out).to_csv(out_cells, index=False)

# ── 2. Map on the reference grid ─────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
vmax = np.nanpercentile(np.r_[maps["drogued"][1], maps["undrogued"][1]], 99)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=subplot_kw)
for ax, name in zip(axes, DROGUE_TYPES):
    op, days = maps[name]
    mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, days), cmap="viridis",
                         vmin=0, vmax=vmax, **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.set_title(f"{name.capitalize()}: median {np.nanmedian(days):.0f} days", fontsize=10)
cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="max")
cb.set_label("Expected time afloat in R (days)")
fig.suptitle(f"Residence time from each reliable cell — {MAP_RES:g}°, τ = {args.tau[0]:g} d", fontsize=12)
out_png = f"figures/agulhas_residence_time_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_summary}\nSaved → {out_cells}\nSaved → {out_png}")
