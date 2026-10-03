"""
Agulhas Region — Release-Point Maps
Where material released at each candidate point from 04 is after about one
week, one month and one year, under the drogued and the undrogued matrix:
one figure per point (rows: drogue type; columns: 7, 30, 365 days), each
panel giving the share still afloat in R and the share that has left across
each edge or stranded, with the 95% interval from 06's drifter bootstrap.
Times are rounded to whole τ steps (1 week = 2, 1 month ≈ 9, 1 year ≈ 104
steps at τ = 3.5 d), and exits are read through exit_labels.

Needs the matrices from 03, data/release_candidates.csv from 04 and the
bootstrap intervals from 06 (same box, grid and lag).

    .venv/bin/python scripts/17_release_point_maps.py --res 1 --tau 3.5
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from config import DROGUE_TYPES, operator_path, parse_args, setting_tag
from transport import load_operator, release, to_grid

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
TAG = setting_tag(RES, TAU, args.box)
DAYS = [7, 30, 365]

os.makedirs("figures", exist_ok=True)

# ── 1. Release points, matrices and bootstrap intervals ──────────────────────
points = pd.read_csv("data/release_candidates.csv")
ops = {name: load_operator(operator_path(name, RES, TAU, args.box)) for name in DROGUE_TYPES}
labels = [str(x) for x in ops["drogued"]["exit_labels"]]
n_cells = len(ops["drogued"]["cell_flat"])
steps = {d: int(round(d / TAU)) for d in DAYS}
powers = {name: {d: np.linalg.matrix_power(op["P"], steps[d]) for d in DAYS} for name, op in ops.items()}
boot_csv = f"data/bootstrap_release_{TAG}.csv"
if not os.path.exists(boot_csv):
    raise SystemExit(f"{boot_csv} not found: run 06 with --res {RES:g} --tau {TAU:g} first.")
boot = pd.read_csv(boot_csv).set_index(["name", "drogue", "metric", "days"])

# ── 2. Propagate each point ──────────────────────────────────────────────────
rows, dists = [], {}
for pt in points.itertuples():
    for name, op in ops.items():
        p0 = release(op, pt.lon, pt.lat)
        for d in DAYS:
            p = p0 @ powers[name][d]
            dists[(pt.name, name, d)] = p
            fate = {"in_R": p[:n_cells].sum(), **{f"exit_{lab}": p[n_cells + k] for k, lab in enumerate(labels)}}
            for metric, value in fate.items():
                b = boot.loc[(pt.name, name, metric, d)]
                if abs(b.estimate - value) > 1e-6:
                    raise SystemExit(f"{pt.name} {name} {metric} {d} d differs from 06: rerun 06")
                rows.append({"name": pt.name, "kind": pt.kind, "lon": pt.lon, "lat": pt.lat, "drogue": name,
                             "days": d, "model_days": steps[d] * TAU, "metric": metric, "estimate": value,
                             "ci_low": b.ci_low, "ci_high": b.ci_high})
table = pd.DataFrame(rows)
out_csv = f"data/release_point_maps_{TAG}.csv"
table.to_csv(out_csv, index=False)


def fmt(name, drogue, metric, d):
    r = table[(table.name == name) & (table.drogue == drogue) & (table.metric == metric) & (table.days == d)].iloc[0]
    return f"{r.estimate:.0%} ({r.ci_low:.0%}–{r.ci_high:.0%})"


print(f"Release points after 1 year, estimate (95% interval), {RES:g}°, τ = {TAU:g} d")
metrics = ["in_R"] + [f"exit_{lab}" for lab in labels]
print(f"{'':<28}{'type':<11}" + "".join(f"{m.replace('exit_', ''):>16}" for m in metrics))
for pt in points.itertuples():
    for name in DROGUE_TYPES:
        print(f"{pt.name[:27]:<28}{name:<11}" + "".join(f"{fmt(pt.name, name, m, 365):>16}" for m in metrics))

# ── 3. One figure per point ──────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
saved = []
for i, pt in enumerate(points.itertuples(), start=1):
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.2), subplot_kw=subplot_kw)
    for row, name in enumerate(DROGUE_TYPES):
        op = ops[name]
        for col, d in enumerate(DAYS):
            ax = axes[row, col]
            grid = to_grid(op, dists[(pt.name, name, d)])
            mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], np.where(grid > 0, grid, np.nan),
                                 cmap="Blues", norm=LogNorm(1e-4, 1), **kw)
            ax.plot(pt.lon, pt.lat, marker="*", ms=12, color="#e8710a", mec="#222222", **kw)
            if ccrs:
                ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
            ax.set_title(f"{name.capitalize()}, {d} days: afloat in R {fmt(pt.name, name, 'in_R', d)}", fontsize=9)
            text = "\n".join(f"{lab} {fmt(pt.name, name, f'exit_{lab}', d)}" for lab in labels)
            ax.text(0.01, 0.02, text, transform=ax.transAxes, fontsize=7, va="bottom", zorder=5,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="min")
    cb.set_label(f"Probability per {RES:g}° cell (log)")
    fig.suptitle(f"Release at {pt.name} ({pt.lon:.1f}°E, {-pt.lat:.1f}°S) — {RES:g}°, τ = {TAU:g} d; "
                 f"95% drifter-bootstrap intervals", fontsize=12)
    slug = re.sub(r"[^a-z0-9]+", "_", pt.name.lower()).strip("_")
    out = f"figures/agulhas_release_{i:02d}_{slug}_{TAG}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(out)
print(f"\nSaved → {out_csv}\nSaved → {len(saved)} figures, figures/agulhas_release_*_{TAG}.png")
