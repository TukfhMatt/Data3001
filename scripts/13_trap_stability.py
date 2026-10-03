"""
Agulhas Region — Retention Hotspots and Their Stability
Where in R material is held longest. Each reliable cell is scored by the
probability that material released there is still afloat in R after 365
days; the top HOTSPOT_SHARE of reliable cells are the retention hotspots.
Two checks decide whether a hotspot is a feature of the flow or of the data:

  • across grids — hotspot sets of two grids rasterised onto a common
                   REF_RES grid and compared (Jaccard index and overlap
                   coefficient) only where both grids have reliable cells,
                   so cells flagged on one grid cannot count as disagreement
  • bootstrap    — on the reference grid, whole drifters are resampled
                   N_BOOT times and the matrix rebuilt each time: a 95%
                   interval on each cell's retention, and how often each cell
                   is a hotspot (share of resamples in its top HOTSPOT_SHARE)

Retention after a year also measures distance from the open edges of R, so a
hotspot is a place R holds material, not necessarily a closed eddy.

Needs the matrices from 03 and the subset from 00 (for the bootstrap).

    .venv/bin/python scripts/13_trap_stability.py --res 0.5 1 2 --tau 3.5
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import xarray as xr

from config import DROGUE_TYPES, RES, grid_shape, operator_path, parse_args, setting_tag
from transport import (cell_centres, drifter_counts, load_operator, move_indices, reliable,
                       resampled_counts, row_normalise, strand_times, to_grid, transition_states)

args = parse_args(__doc__, grid=True, lag=True, multi=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
TAG = setting_tag(args.res, args.tau, args.box)
HOTSPOT_SHARE = 0.10
RETENTION_DAYS = 365
REF_RES = 0.25                 # common grid for comparing hotspot sets
BOOT_RES = RES if RES in args.res else args.res[0]
N_BOOT = 1000
SEED = 0

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(SEED)


def retention(P, n_cells, steps):
    """Probability that material starting in each ocean state is still in R after `steps`."""
    Q = P[:n_cells, :n_cells]
    s = np.ones(n_cells)
    for _ in range(steps):
        s = Q @ s
    return s


def hotspots(score, ok):
    """Top HOTSPOT_SHARE of the cells in ok by score."""
    return ok & (score >= np.quantile(score[ok], 1 - HOTSPOT_SHARE))


# ── 1. Retention and hotspots per grid ───────────────────────────────────────
summary, cells_out, raster = [], [], {}
ref_lon = np.arange(LON_MIN + REF_RES / 2, LON_MAX, REF_RES)
ref_lat = np.arange(LAT_MIN + REF_RES / 2, LAT_MAX, REF_RES)
pt_lon, pt_lat = [a.ravel() for a in np.meshgrid(ref_lon, ref_lat)]
for tau in args.tau:
    steps = int(round(RETENTION_DAYS / tau))
    for res in args.res:
        for name in DROGUE_TYPES:
            op = load_operator(operator_path(name, res, tau, args.box))
            n = len(op["cell_flat"])
            ok = reliable(op)
            score = retention(op["P"], n, steps)
            hot = hotspots(score, ok)
            lon, lat = cell_centres(op)
            summary.append({"res": res, "tau": tau, "drogue": name, "reliable_cells": int(ok.sum()),
                            "median_retention": np.median(score[ok]),
                            "hotspot_threshold": score[hot].min(), "hotspot_cells": int(hot.sum()),
                            "hotspot_lon": np.mean(lon[hot]), "hotspot_lat": np.mean(lat[hot])})
            cells_out.append(pd.DataFrame({"res": res, "tau": tau, "drogue": name, "lon": lon, "lat": lat,
                                           "retention_365d": score, "reliable": ok, "hotspot": hot}))
            # Each REF_RES point takes the state of the cell it falls in
            n_lon, n_lat = grid_shape(args.box, res)
            flat = ((pt_lat - LAT_MIN) // res).astype(int) * n_lon + ((pt_lon - LON_MIN) // res).astype(int)
            state = np.full(n_lon * n_lat, -1)
            state[op["cell_flat"]] = np.arange(n)
            st = state[flat]
            raster[(tau, res, name)] = (np.where(st >= 0, ok[st], False), np.where(st >= 0, hot[st], False))

summary = pd.DataFrame(summary)
print(f"Retention after {RETENTION_DAYS} days; hotspots = top {HOTSPOT_SHARE:.0%} of reliable cells")
print(f"{'grid':>5} {'τ':>4} {'type':<10}{'reliable':>9}{'median':>8}{'threshold':>10}{'hotspot centre':>18}")
for r in summary.itertuples():
    print(f"{r.res:>4g}° {r.tau:>4g} {r.drogue:<10}{r.reliable_cells:>9}{r.median_retention:>8.1%}"
          f"{r.hotspot_threshold:>10.1%}{r.hotspot_lon:>9.1f}°E {-r.hotspot_lat:.1f}°S")

# ── 2. Hotspot agreement across grids ────────────────────────────────────────
pairs = []
for tau in args.tau:
    for name in DROGUE_TYPES:
        for a, b in combinations(args.res, 2):
            ok_a, hot_a = raster[(tau, a, name)]
            ok_b, hot_b = raster[(tau, b, name)]
            both = ok_a & ok_b
            ha, hb = hot_a & both, hot_b & both
            inter, union = (ha & hb).sum(), (ha | hb).sum()
            pairs.append({"tau": tau, "drogue": name, "grid_a": a, "grid_b": b,
                          "common_area_share": both.mean(),
                          "jaccard": inter / union if union else np.nan,
                          "overlap": inter / min(ha.sum(), hb.sum()) if min(ha.sum(), hb.sum()) else np.nan})
pairs = pd.DataFrame(pairs)
print(f"\nHotspot agreement where both grids are reliable (rasterised at {REF_RES:g}°)")
for r in pairs.itertuples():
    print(f"  τ = {r.tau:g} d {r.drogue:<10} {r.grid_a:g}° vs {r.grid_b:g}°: Jaccard {r.jaccard:.0%}, "
          f"overlap {r.overlap:.0%} (over {r.common_area_share:.0%} of R)")

# ── 3. Drifter bootstrap on the reference grid ───────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
n_lon, _ = grid_shape(args.box, BOOT_RES)
cell_flat = np.where(in_R, ((lat - LAT_MIN) // BOOT_RES).astype(int) * n_lon
                     + ((lon - LON_MIN) // BOOT_RES).astype(int), -1)
strand_t = strand_times(ds, traj_idx, t, in_R)

boot_rows, maps = [], {}
for tau in args.tau:
    steps = int(round(RETENTION_DAYS / tau))
    start, end, stranded = move_indices(traj_idx, t, in_R, tau, strand_t)
    for name, flag in DROGUE_TYPES.items():
        op = load_operator(operator_path(name, BOOT_RES, tau, args.box))
        cells, n = op["cell_flat"], len(op["cell_flat"])
        n_states = op["P"].shape[0]
        same = (drogue[start] == flag) & (drogue[end] == flag)
        s, e = start[same], end[same]
        f, to = transition_states(cell_flat, cells, s, e, lon, lat, args.box, stranded[same])
        counts = drifter_counts(traj_idx[s], f, to, n_states)
        C = np.bincount(counts[1], weights=counts[2], minlength=n_states**2).reshape(n_states, n_states)
        assert np.array_equal(C, op["C"]), f"rebuilt counts differ from the saved {name} matrix"

        ok = reliable(op)
        score = retention(op["P"], n, steps)
        hot = hotspots(score, ok)
        samples = np.empty((N_BOOT, n))
        in_top = np.zeros(n)
        for b in range(N_BOOT):
            Pb, _ = row_normalise(resampled_counts(counts, n_states, rng), cells, n_lon)
            samples[b] = retention(Pb, n, steps)
            in_top += hotspots(samples[b], ok)
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        freq = in_top / N_BOOT
        clon, clat = cell_centres(op)
        boot_rows.append(pd.DataFrame({"tau": tau, "drogue": name, "lon": clon, "lat": clat,
                                       "reliable": ok, "drifters": op["row_drifters"], "hotspot": hot,
                                       "retention_365d": score, "ci_low": lo, "ci_high": hi,
                                       "hotspot_frequency": freq}))
        maps[(tau, name)] = (op, np.where(ok, freq, np.nan))

        print(f"\n{name}, {BOOT_RES:g}°, τ = {tau:g} d: {counts[3]:,} drifters resampled {N_BOOT} times")
        print(f"  hotspot cells a hotspot in ≥ 50% of resamples: {(freq[hot] >= 0.5).sum()} of {hot.sum()}; "
              f"other cells a hotspot in ≥ 50%: {(freq[ok & ~hot] >= 0.5).sum()}")
        print(f"  {'cell':>15}{'drifters':>10}{'retention (95%)':>22}{'hotspot freq':>14}")
        for k in np.argsort(-score * ok)[:8]:
            print(f"  {clon[k]:>6.1f}°E {-clat[k]:>4.1f}°S{op['row_drifters'][k]:>10}"
                  f"{score[k]:>9.0%} ({lo[k]:.0%}–{hi[k]:.0%}){freq[k]:>14.0%}")

# ── 4. Save and map hotspot frequency ────────────────────────────────────────
out_summary = f"data/retention_hotspots_{TAG}.csv"
out_pairs = f"data/retention_hotspot_agreement_{TAG}.csv"
out_cells = f"data/retention_cells_{TAG}.csv"
out_boot = f"data/retention_bootstrap_{TAG}.csv"
summary.to_csv(out_summary, index=False)
pairs.to_csv(out_pairs, index=False)
pd.concat(cells_out).to_csv(out_cells, index=False)
pd.concat(boot_rows).to_csv(out_boot, index=False)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
tau0 = args.tau[0]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=subplot_kw)
for ax, name in zip(axes, DROGUE_TYPES):
    op, freq = maps[(tau0, name)]
    mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, freq),
                         cmap="magma_r", vmin=0, vmax=1, **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.set_title(f"{name.capitalize()}: share of resamples in the top {HOTSPOT_SHARE:.0%}", fontsize=10)
cb = fig.colorbar(mesh, ax=axes, shrink=0.8)
cb.set_label("Hotspot frequency (reliable cells)")
fig.suptitle(f"Retention hotspots after {RETENTION_DAYS} days — {BOOT_RES:g}°, τ = {tau0:g} d, "
             f"{N_BOOT} drifter resamples", fontsize=12)
out_png = f"figures/agulhas_retention_hotspots_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_summary}\nSaved → {out_pairs}\nSaved → {out_cells}\nSaved → {out_boot}\nSaved → {out_png}")
