"""
Agulhas Region — Exit Zones
Where material released in each cell eventually ends up: across which edge
of R it leaves (W, E, S, N) or whether it strands on a coast inside R. The
eventual fate comes exactly from the absorbing chain (transport.absorption,
N·R); the fate after about one year comes from propagating P. For every grid
and lag and both drogue types: the mean fate over reliable cells, and the
share of reliable cells whose most likely fate is each exit, mapped on the
reference grid. Fates are read through exit_labels.

Needs the matrices from 03.

    .venv/bin/python scripts/15_exit_zones.py --res 0.5 1 2 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from config import DROGUE_TYPES, RES, operator_path, parse_args, setting_tag
from transport import absorption, cell_centres, load_operator, reliable, to_grid

args = parse_args(__doc__, grid=True, lag=True, multi=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
TAG = setting_tag(args.res, args.tau, args.box)
MAP_RES = RES if RES in args.res else args.res[0]
YEAR_DAYS = 365
FATE_COLORS = {"W": "#2a78d6", "E": "#eb6834", "S": "#1b9e77", "N": "#7b3294", "stranded": "#1a1a1a"}

os.makedirs("figures", exist_ok=True)

# ── 1. Eventual and one-year fates per grid, lag and drogue type ─────────────
summary, cells_out, maps = [], [], {}
for tau in args.tau:
    year_steps = int(round(YEAR_DAYS / tau))
    for res in args.res:
        for name in DROGUE_TYPES:
            op = load_operator(operator_path(name, res, tau, args.box))
            labels = [str(x) for x in op["exit_labels"]]
            n = len(op["cell_flat"])
            _, fate = absorption(op)
            year = np.linalg.matrix_power(op["P"], year_steps)[:n, n:]
            ok = reliable(op)
            dominant = np.argmax(fate, axis=1)
            row = {"res": res, "tau": tau, "drogue": name, "reliable_cells": int(ok.sum()),
                   "year_days": year_steps * tau, "in_R_1yr": 1 - year[ok].sum(axis=1).mean()}
            for k, lab in enumerate(labels):
                row[f"eventual_{lab}"] = fate[ok, k].mean()
                row[f"1yr_{lab}"] = year[ok, k].mean()
                row[f"dominant_{lab}"] = np.mean(dominant[ok] == k)
            summary.append(row)
            lon, lat = cell_centres(op)
            cells = pd.DataFrame({"res": res, "tau": tau, "drogue": name, "lon": lon, "lat": lat,
                                  "reliable": ok, "dominant": np.array(labels)[dominant]})
            for k, lab in enumerate(labels):
                cells[f"eventual_{lab}"] = fate[:, k]
                cells[f"1yr_{lab}"] = year[:, k]
            cells_out.append(cells)
            if res == MAP_RES and tau == args.tau[0]:
                maps[name] = (op, np.where(ok, dominant, np.nan), labels)

summary = pd.DataFrame(summary)
print("Mean eventual fate over reliable cells, and (in brackets) the share of cells where it is the most likely")
print(f"{'grid':>5} {'τ':>4} {'type':<10}{'cells':>7}" + "".join(f"{lab:>16}" for lab in labels)
      + f"{'in R at 1 yr':>14}")
for r in summary.to_dict("records"):
    print(f"{r['res']:>4g}° {r['tau']:>4g} {r['drogue']:<10}{r['reliable_cells']:>7}"
          + "".join(f"{r[f'eventual_{lab}']:>9.1%} ({r[f'dominant_{lab}']:>3.0%})" for lab in labels)
          + f"{r['in_R_1yr']:>14.1%}")

out_summary = f"data/exit_zones_{TAG}.csv"
out_cells = f"data/exit_zones_cells_{TAG}.csv"
summary.to_csv(out_summary, index=False)
pd.concat(cells_out).to_csv(out_cells, index=False)

# ── 2. Map of the most likely fate on the reference grid ─────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
cmap = ListedColormap([FATE_COLORS[lab] for lab in labels])
norm = BoundaryNorm(np.arange(len(labels) + 1) - 0.5, len(labels))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=subplot_kw)
for ax, name in zip(axes, DROGUE_TYPES):
    op, dom, labels = maps[name]
    ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, dom), cmap=cmap, norm=norm, **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.set_title(f"{name.capitalize()}", fontsize=10)
fig.legend(handles=[Patch(color=FATE_COLORS[lab], label=lab) for lab in labels], title="Most likely fate",
           loc="center right", frameon=False)
fig.suptitle(f"Eventual exit or stranding from each reliable cell — {MAP_RES:g}°, τ = {args.tau[0]:g} d",
             fontsize=12)
out_png = f"figures/agulhas_exit_zones_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_summary}\nSaved → {out_cells}\nSaved → {out_png}")
