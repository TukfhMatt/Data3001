"""
Agulhas Region — Source and Sink Cells
Which cells of R gather material and which shed it over one step. Every row of
P sums to 1, so the rows say nothing here; the column sums do. If each ocean
cell starts with one unit of material, the column sum of cell j,

    inflow_j = Σ_i P[i, j]  over the ocean cells i,

is what cell j holds after one τ. A cell above 1 gathers material (a sink) and
a cell below 1 sheds it (a source), judged by whether the 95% interval from
N_BOOT resamples of whole drifters excludes 1. The intervals are not corrected
for the number of cells compared, so the agreement between the two drogue types
is printed as a check.

Cells on the edge of R are not judged: material entering R from outside is not
in the matrix, so they would look like sources whatever the flow does.

The even starting load is a measure of the flow's convergence, not of where
material really is. Retention over a year is in 13 and eventual fates in 15.
Inflow counts every row that reaches the cell, including thin ones, so only the
receiving cell's own reliability is checked.

Needs the matrices from 03 and the subset from 00 (for the bootstrap).

    .venv/bin/python scripts/30_source_sink.py --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import xarray as xr

from config import DROGUE_TYPES, grid_shape, operator_path, parse_args, place_label, region_prefix, setting_tag
from transport import (cell_centres, drifter_counts, load_operator, move_indices, reliable, resampled_counts,
                       row_normalise, same_drogue, strand_times, to_grid, transition_states)

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
TAG = setting_tag(RES, TAU, args.box)
N_BOOT = 1000
SEED = 0
TOP = 5        # cells printed at each end

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(SEED)
n_lon, _ = grid_shape(args.box, RES)


def inflow(P, n):
    """Column sums of the ocean block of P."""
    return P[:n, :n].sum(axis=0)


# ── 1. Moves of each drifter, to resample ────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
cell_flat = np.where(in_R, ((lat - LAT_MIN) // RES).astype(int) * n_lon
                     + ((lon - LON_MIN) // RES).astype(int), -1)
start, end, stranded = move_indices(traj_idx, t, in_R, TAU, strand_times(ds, traj_idx, t, in_R))

# ── 2. Inflow per cell with intervals ────────────────────────────────────────
tables, ops = [], {}
for name, flag in DROGUE_TYPES.items():
    op = ops[name] = load_operator(operator_path(name, RES, TAU, args.box))
    cells, n = op["cell_flat"], len(op["cell_flat"])
    n_states = op["P"].shape[0]
    same = same_drogue(drogue, start, end, stranded, flag)
    s, e = start[same], end[same]
    f, to = transition_states(cell_flat, cells, s, e, lon, lat, args.box, stranded[same])
    counts = drifter_counts(traj_idx[s], f, to, n_states)
    C = np.bincount(counts[1], weights=counts[2], minlength=n_states**2).reshape(n_states, n_states)
    assert np.array_equal(C, op["C"]), f"rebuilt counts differ from the saved {name} matrix"

    boot = np.empty((N_BOOT, n))
    for b in range(N_BOOT):
        boot[b] = inflow(row_normalise(resampled_counts(counts, n_states, rng), cells, n_lon)[0], n)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    clon, clat = cell_centres(op)
    value = inflow(op["P"], n)
    on_edge = (np.minimum(clon - LON_MIN, LON_MAX - clon) < RES) | (np.minimum(clat - LAT_MIN, LAT_MAX - clat) < RES)
    kind = np.where(on_edge, "edge", np.where(lo > 1, "sink", np.where(hi < 1, "source", "neither")))
    tables.append(pd.DataFrame({"drogue": name, "lon": clon, "lat": clat, "inflow": value, "lo": lo, "hi": hi,
                                "kind": kind, "reliable": reliable(op), "row_drifters": op["row_drifters"]}))
    print(f"{name}: {counts[3]:,} drifters resampled {N_BOOT} times")

table = pd.concat(tables, ignore_index=True)

# ── 3. Save and summarise ────────────────────────────────────────────────────
out_csv = f"data/source_sink_{TAG}.csv"
table.to_csv(out_csv, index=False, float_format="%.5g")

good = table[table.reliable & (table.kind != "edge")]
for name in DROGUE_TYPES:
    part = good[good.drogue == name]
    shares = part.kind.value_counts().reindex(["sink", "source", "neither"], fill_value=0)
    print(f"\n{name}: {len(part)} reliable cells off the edge — " + ", ".join(f"{k} {v}" for k, v in shares.items()))
    ranked = part[part.kind != "neither"].sort_values("inflow")
    for label, rows in [("strongest sources", ranked.head(TOP)), ("strongest sinks", ranked.tail(TOP)[::-1])]:
        print(f"  {label}:")
        for r in rows.itertuples():
            print(f"    {place_label(r.lon, r.lat):>15}  {r.inflow:.2f} ({r.lo:.2f}–{r.hi:.2f}), "
                  f"{r.row_drifters} drifters")
both = good.pivot(index=["lon", "lat"], columns="drogue", values=["inflow", "kind"]).dropna()
kind_d, kind_u = both[("kind", "drogued")], both[("kind", "undrogued")]
judged = (kind_d != "neither") & (kind_u != "neither")
print(f"\nOver {len(both)} cells reliable in both types: correlation of inflow "
      f"{np.corrcoef(both[('inflow', 'drogued')].astype(float), both[('inflow', 'undrogued')].astype(float))[0, 1]:.2f}; "
      f"source or sink in both {judged.sum()}, same kind {(kind_d[judged] == kind_u[judged]).sum()}")

# ── 4. Map ───────────────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
span = np.nanpercentile(np.abs(good.inflow - 1), 98)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=subplot_kw)
for ax, (name, op), part in zip(axes, ops.items(), tables):
    ok = part.reliable.values & (part.kind.values != "edge")
    grid = to_grid(op, np.r_[np.where(ok, part.inflow.values, np.nan), np.zeros(op["P"].shape[0] - len(part))])
    mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], grid, cmap="RdBu_r", vmin=1 - span, vmax=1 + span, **kw)
    marked = part[ok & (part.kind != "neither")]
    ax.scatter(marked.lon, marked.lat, s=3, color="#222222", linewidths=0, zorder=4, **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.set_title(f"{name.capitalize()}: dot = 95% interval excludes 1", fontsize=10)
cb = fig.colorbar(mesh, ax=axes, shrink=0.8)
cb.set_label("Held after one step from a load of 1")
fig.suptitle(f"Source and sink cells — {RES:g}°, τ = {TAU:g} d, reliable cells off the edge of R; above 1 gathers, below 1 sheds", fontsize=12)
out_png = f"figures/{region_prefix(args.data)}_source_sink_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_csv}\nSaved → {out_png}")
