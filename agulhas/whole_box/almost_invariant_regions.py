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

Needs the matrices from transition_matrix.

    python -m agulhas almost_invariant_regions --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.sparse.linalg import eigsh

from agulhas.config import DROGUE_TYPES, operator_path, parse_args, place_label, region_prefix, setting_tag
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import cell_centres, load_operator, reliable, to_grid

N_REGIONS = 4
N_EIG = 8                       # eigenvalues printed
N_STARTS = 20
KMEANS_ITER = 200
SEED = 3001
CHECK_DAYS = [30, 365]
REGION_COLORS = ["#2a78d6", "#eb6834", "#1b9e77", "#7b3294", "#e6ab02", "#666666"]


# ── Clustering ───────────────────────────────────────────────────────────────
def kmeans(x, k, rng):
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


def spectral_embedding(P, idx, k):
    """Leading eigenvalues of S between the cells idx, and each cell's row of the
    k leading eigenvectors scaled to unit length: (eigenvalues, embedding)."""
    Q = P[np.ix_(idx, idx)]
    A = (Q + Q.T) / 2
    np.fill_diagonal(A, 0)
    d = A.sum(axis=1)
    s = 1 / np.sqrt(np.where(d > 0, d, 1))
    vals, vecs = eigsh(s[:, None] * A * s[None, :], k=N_EIG, which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return vals, vecs[:, :k] / np.maximum(np.linalg.norm(vecs[:, :k], axis=1, keepdims=True), 1e-12)


def regions_west_to_east(emb, lon, k, rng):
    """Cluster the embedding, and number the regions by mean longitude."""
    lab = kmeans(emb, k, rng)
    west_to_east = np.argsort([lon[lab == g].mean() for g in range(k)])
    return np.argsort(west_to_east)[lab]


def longitude_bands(lon, idx, lab, k, n):
    """Bands of longitude over the same reliable cells, with the same sizes as the regions."""
    bands = np.full(n, -1)
    by_lon = idx[np.argsort(lon[idx], kind="stable")]
    edges = np.r_[0, np.cumsum([np.sum(lab == g) for g in range(k)])]
    for g in range(k):
        bands[by_lon[edges[g]:edges[g + 1]]] = g
    return bands


# ── Validation ───────────────────────────────────────────────────────────────
def stay(P, members, steps):
    """Share of a uniform start over `members` that is in `members` after `steps`."""
    p = np.zeros(P.shape[0])
    p[members] = 1 / len(members)
    for _ in range(steps):
        p = p @ P
    return p[members].sum()


def region_rows(name, op, region, bands, gap_after_k, k, tau):
    """How well each region of one drogue type keeps its material, one row each."""
    P = op["P"]
    lon, lat = cell_centres(op)
    steps = {days: int(round(days / tau)) for days in CHECK_DAYS}
    rows = []
    for g in range(k):
        m = np.where(region == g)[0]
        row = {"drogue": name, "region": g + 1, "cells": len(m), "lon": lon[m].mean(), "lat": lat[m].mean(),
               "median_drifters": np.median(op["row_drifters"][m]),
               "one_step": P[np.ix_(m, m)].sum(axis=1).mean(),
               "band_30d": stay(P, np.where(bands == g)[0], steps[30]), "k": k, "gap_after_k": gap_after_k}
        for days in CHECK_DAYS:
            row[f"stay_{days}d"] = stay(P, m, steps[days])
        rows.append(row)
    return rows


def print_regions(name, n_reliable, vals, gaps, rows, k):
    print(f"\n{name}: {n_reliable} reliable cells; leading eigenvalues "
          + ", ".join(f"{v:.4f}" for v in vals)
          + "\n  gaps after the 2nd–7th: " + ", ".join(f"{g:.4f}" for g in gaps[1:7])
          + f"; {k} regions")
    print(f"  {'region':>6}{'cells':>7}{'centre':>16}{'1 step':>8}{'30 d':>7}{'365 d':>7}{'band 30 d':>11}")
    for row in rows:
        print(f"  {row['region']:>6}{row['cells']:>7}{place_label(row['lon'], row['lat']):>16}{row['one_step']:>8.0%}"
              f"{row['stay_30d']:>7.0%}{row['stay_365d']:>7.0%}{row['band_30d']:>11.0%}")


# ── Map ──────────────────────────────────────────────────────────────────────
def plot_regions(maps, k, args, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=MAP_KW)
    for ax, name in zip(axes, DROGUE_TYPES):
        op, region = maps[name]
        ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, region), cmap=ListedColormap(REGION_COLORS[:k]),
                      norm=BoundaryNorm(np.arange(k + 1) - 0.5, k), **TRANSFORM_KW)
        lon, lat = cell_centres(op)
        for g in range(k):
            m = region == g
            ax.text(lon[m].mean(), lat[m].mean(), str(g + 1), ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=4, **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.set_title(f"{name.capitalize()}: {k} regions", fontsize=10)
    fig.suptitle(f"Almost-invariant regions (reliable cells) — {args.res:g}°, τ = {args.tau:g} d", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_summary = f"data/almost_invariant_regions_{tag}.csv"
    out_cells = f"data/almost_invariant_cells_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_almost_invariant_{tag}.png"
    rng = np.random.default_rng(SEED)
    k = N_REGIONS

    summary, cells, maps = [], [], {}
    for name in DROGUE_TYPES:
        op = load_operator(operator_path(name, args.res, args.tau, args.box))
        n = len(op["cell_flat"])
        ok = reliable(op)
        idx = np.where(ok)[0]
        lon, lat = cell_centres(op)
        vals, emb = spectral_embedding(op["P"], idx, k)
        gaps = vals[:-1] - vals[1:]                     # gaps[k − 1]: after the k-th eigenvalue
        lab = regions_west_to_east(emb, lon[idx], k, rng)
        region = np.full(n, -1)
        region[idx] = lab

        rows = region_rows(name, op, region, longitude_bands(lon, idx, lab, k, n), gaps[k - 1], k, args.tau)
        print_regions(name, len(idx), vals, gaps, rows, k)
        summary += rows
        cells.append(pd.DataFrame({"drogue": name, "lon": lon, "lat": lat, "reliable": ok,
                                   "region": np.where(region >= 0, region + 1, 0)}))
        maps[name] = (op, np.where(region >= 0, region, np.nan))

    pd.DataFrame(summary).to_csv(out_summary, index=False)
    pd.concat(cells).to_csv(out_cells, index=False)
    plot_regions(maps, k, args, out_png)
    print(f"\nSaved → {out_summary}\nSaved → {out_cells}\nSaved → {out_png}")


if __name__ == "__main__":
    main()
