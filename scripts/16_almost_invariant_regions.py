"""
Agulhas Region — Almost-Invariant Regions
Splits R into regions that exchange little material with each other over a
step, from the transition matrix of each drogue type (reliable cells only):

  1. affinity    — A = ½ (Q + Qᵀ) between reliable cells, self-moves removed,
                   normalised as S = D^(−½) A D^(−½)
  2. how many    — N_REGIONS regions. The leading eigenvalues of S fall off
                   smoothly, with no clear gap to fix the number, so it is a
                   choice; the eigenvalues and gaps are printed for checking
  3. clustering  — rows of the k leading eigenvectors, scaled to unit length,
                   split by k-means (best of N_STARTS random starts)
  4. validation  — with the directional P: the share of material that stays
                   in its region after one step, 30 days and 365 days, from a
                   uniform start over the region; and the same 30-day share
                   for longitude bands of the same sizes, as a baseline any
                   compact region would reach

Regions are numbered west to east. This is a spectral-clustering
approximation to almost-invariant sets, not an exact PCCA+ decomposition.

Needs the matrices from 03.

    .venv/bin/python scripts/16_almost_invariant_regions.py --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.sparse.linalg import eigsh

from config import DROGUE_TYPES, operator_path, parse_args, setting_tag
from transport import cell_centres, load_operator, reliable, to_grid

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
TAG = setting_tag(RES, TAU, args.box)
N_REGIONS = 4
N_EIG = 8                       # eigenvalues printed
N_STARTS = 20
KMEANS_ITER = 200
SEED = 3001
CHECK_DAYS = [30, 365]
REGION_COLORS = ["#2a78d6", "#eb6834", "#1b9e77", "#7b3294", "#e6ab02", "#666666"]

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(SEED)


def kmeans(x, k):
    """Best of N_STARTS k-means runs (lowest within-cluster sum of squares)."""
    best, best_cost = None, np.inf
    for _ in range(N_STARTS):
        centres = x[rng.choice(len(x), k, replace=False)]
        for _ in range(KMEANS_ITER):
            lab = np.argmin(((x[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
            new = np.array([x[lab == g].mean(axis=0) if np.any(lab == g) else x[rng.integers(len(x))]
                            for g in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        cost = ((x - centres[lab]) ** 2).sum()
        if cost < best_cost:
            best, best_cost = lab, cost
    return best


def stay(P, members, steps):
    """Share of a uniform start over `members` that is in `members` after `steps`."""
    p = np.zeros(P.shape[0])
    p[members] = 1 / len(members)
    for _ in range(steps):
        p = p @ P
    return p[members].sum()


# ── 1. Spectral regions per drogue type ──────────────────────────────────────
summary, cells_out, maps = [], [], {}
for name in DROGUE_TYPES:
    op = load_operator(operator_path(name, RES, TAU, args.box))
    P = op["P"]
    n = len(op["cell_flat"])
    ok = reliable(op)
    idx = np.where(ok)[0]
    Q = P[np.ix_(idx, idx)]
    A = (Q + Q.T) / 2
    np.fill_diagonal(A, 0)
    d = A.sum(axis=1)
    s = 1 / np.sqrt(np.where(d > 0, d, 1))
    vals, vecs = eigsh(s[:, None] * A * s[None, :], k=N_EIG, which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    gaps = vals[:-1] - vals[1:]                     # gaps[k − 1]: after the k-th eigenvalue
    k = N_REGIONS
    emb = vecs[:, :k] / np.maximum(np.linalg.norm(vecs[:, :k], axis=1, keepdims=True), 1e-12)
    lab = kmeans(emb, k)

    lon, lat = cell_centres(op)
    west_to_east = np.argsort([lon[idx][lab == g].mean() for g in range(k)])
    lab = np.argsort(west_to_east)[lab]
    region = np.full(n, -1)
    region[idx] = lab

    # Longitude bands over the same reliable cells, with the same sizes
    bands = np.full(n, -1)
    by_lon = idx[np.argsort(lon[idx], kind="stable")]
    edges = np.r_[0, np.cumsum([np.sum(lab == g) for g in range(k)])]
    for g in range(k):
        bands[by_lon[edges[g]:edges[g + 1]]] = g

    steps = {dd: int(round(dd / TAU)) for dd in CHECK_DAYS}
    print(f"\n{name}: {len(idx)} reliable cells; leading eigenvalues "
          + ", ".join(f"{v:.4f}" for v in vals)
          + f"\n  gaps after the 2nd–7th: " + ", ".join(f"{g:.4f}" for g in gaps[1:7])
          + f"; {k} regions")
    print(f"  {'region':>6}{'cells':>7}{'centre':>16}{'1 step':>8}{'30 d':>7}{'365 d':>7}{'band 30 d':>11}")
    for g in range(k):
        m = np.where(region == g)[0]
        row = {"drogue": name, "region": g + 1, "cells": len(m), "lon": lon[m].mean(), "lat": lat[m].mean(),
               "median_drifters": np.median(op["row_drifters"][m]),
               "one_step": P[np.ix_(m, m)].sum(axis=1).mean(),
               "band_30d": stay(P, np.where(bands == g)[0], steps[30]), "k": k, "gap_after_k": gaps[k - 1]}
        for dd in CHECK_DAYS:
            row[f"stay_{dd}d"] = stay(P, m, steps[dd])
        summary.append(row)
        print(f"  {g + 1:>6}{len(m):>7}{row['lon']:>8.1f}°E {-row['lat']:>4.1f}°S{row['one_step']:>8.0%}"
              f"{row['stay_30d']:>7.0%}{row['stay_365d']:>7.0%}{row['band_30d']:>11.0%}")
    cells_out.append(pd.DataFrame({"drogue": name, "lon": lon, "lat": lat, "reliable": ok,
                                   "region": np.where(region >= 0, region + 1, 0)}))
    maps[name] = (op, np.where(region >= 0, region, np.nan), k)

out_summary = f"data/almost_invariant_regions_{TAG}.csv"
out_cells = f"data/almost_invariant_cells_{TAG}.csv"
pd.DataFrame(summary).to_csv(out_summary, index=False)
pd.concat(cells_out).to_csv(out_cells, index=False)

# ── 2. Map ───────────────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=subplot_kw)
for ax, name in zip(axes, DROGUE_TYPES):
    op, region, k = maps[name]
    cmap = ListedColormap(REGION_COLORS[:k])
    ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, region), cmap=cmap,
                  norm=BoundaryNorm(np.arange(k + 1) - 0.5, k), **kw)
    lon, lat = cell_centres(op)
    for g in range(k):
        m = region == g
        ax.text(lon[m].mean(), lat[m].mean(), str(g + 1), ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=4, **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.set_title(f"{name.capitalize()}: {k} regions", fontsize=10)
fig.suptitle(f"Almost-invariant regions (reliable cells) — {RES:g}°, τ = {TAU:g} d", fontsize=12)
out_png = f"figures/agulhas_almost_invariant_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_summary}\nSaved → {out_cells}\nSaved → {out_png}")
