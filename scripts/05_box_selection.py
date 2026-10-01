"""
Agulhas Region — Bounding-Box Selection
Searches every 1°-aligned box that contains the Agulhas core and scores each
on the oil (undrogued) drifter data:

  • data      — undrogued observations and the share of ocean cells backed by
                ≥ 10 distinct drifters (reliable transition-matrix rows)
  • relevance — mean commercial shipping density per ocean cell
  • coherence — probability that oil in the box is still in it after one
                τ = 3.5-day step, and the implied mean residence time

Boxes must stay SEARCH_MARGIN inside the subset extent: positions beyond the
subset are not on disk, so a box touching its edge would never see oil leave
through that edge and its retention would be biased high.

Observations are hourly (native GDP resolution, gap ≤ 6 h). Boxes that pass
the data constraints are reduced to the Pareto set on (shipping ↑,
retention ↑, size ↓) and the best balance of shipping and retention is
reported. A fixed set of named options, including the analysis box R, is scored
side by side.
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.colors import ListedColormap, LogNorm
import rasterio
from rasterio.windows import from_bounds
import xarray as xr

GDP_LOCAL = "data/agulhas_gdp1h_subset.nc"
SHIP_ZIP = "data/shipping/shipdensity_commercial.zip"

# Grid = padded subset extent (the only area with drifter data on disk)
G_LON_MIN, G_LON_MAX = -5.0, 65.0
G_LAT_MIN, G_LAT_MAX = -60.0, -5.0
RES = 1.0
TAU_DAYS = 3.5
SEARCH_MARGIN = 5.0   # ° kept between a candidate box and the subset edge

# Every candidate box must contain the Agulhas core: the current from
# Richards Bay / Durban to the retroflection
CORE_LON = (18.0, 33.0)
CORE_LAT = (-40.0, -28.0)

MIN_OBS = 1_000_000   # undrogued hourly observations
MIN_RELIABLE = 0.90   # share of ocean cells with ≥ MIN_DRIFTERS drifters
MIN_DRIFTERS = 10

# Named options scored side by side (lon_min, lon_max, lat_min, lat_max)
OPTIONS = {
    "Wide southern box":            (5, 50, -50, -20),
    "Shipping-focused":             (10, 55, -41, -15),
    "Shipping-focused + Return Current": (10, 55, -45, -15),
    "Compact core":                 (14, 39, -40, -18),
}
# The analysis box R (BOX in scripts/config.py), highlighted in the figure
CHOSEN = "Shipping-focused + Return Current"
# Map / scatter style per option: colour, line style, line width, marker
OPTION_STYLE = {
    "Wide southern box":                 ("#3a3a3a", "--", 1.8, "s"),
    "Shipping-focused":                  ("#7b3294", "-.", 1.8, "D"),
    "Shipping-focused + Return Current": ("#e8710a", "-", 3.0, "*"),
    "Compact core":                      ("#1b9e77", (0, (4, 2)), 1.8, "o"),
}

os.makedirs("figures", exist_ok=True)
n_lon = int(round((G_LON_MAX - G_LON_MIN) / RES))
n_lat = int(round((G_LAT_MAX - G_LAT_MIN) / RES))

# ── 1. Per-cell drifter statistics (undrogued) ───────────────────────────────
ds = xr.open_dataset(GDP_LOCAL)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
undrogued = ~ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)

inside = (lon >= G_LON_MIN) & (lon < G_LON_MAX) & (lat >= G_LAT_MIN) & (lat < G_LAT_MAX)
ci = np.where(inside, ((lon - G_LON_MIN) // RES).astype(int), -1)
cj = np.where(inside, ((lat - G_LAT_MIN) // RES).astype(int), -1)
cell = np.where(inside, cj * n_lon + ci, -1)

m = inside & undrogued
obs = np.bincount(cell[m], minlength=n_lat * n_lon).reshape(n_lat, n_lon)
pairs = np.unique(np.stack([cell[m], traj_idx[m]]), axis=1)
drifters = np.bincount(pairs[0], minlength=n_lat * n_lon).reshape(n_lat, n_lon)
ocean = (np.bincount(cell[inside], minlength=n_lat * n_lon) > 0).reshape(n_lat, n_lon)
reliable = ocean & (drifters >= MIN_DRIFTERS)
print(f"Grid {n_lon}×{n_lat} cells: {ocean.sum()} ocean, {reliable.sum()} reliable (undrogued)")

# ── 2. τ-step transitions (undrogued at both ends) ───────────────────────────
key = traj_idx.astype(np.int64) * 10**10 + t
target = key + int(TAU_DAYS * 86400)
j = np.minimum(np.searchsorted(key, target), len(key) - 1)
ok = m & (key[j] == target) & inside[j] & undrogued[j]
s, e = np.where(ok)[0], j[ok]
print(f"Undrogued τ-transitions: {len(s):,}")

# A transition stays in box [a, b) × [c, d) (cell indices) iff both endpoints
# do: min(i) ≥ a, max(i) < b, min(j) ≥ c, max(j) < d. Histogram transitions
# by their own bounding cells; cumulative sums then give "stays in box" counts.
imin, imax = np.minimum(ci[s], ci[e]), np.maximum(ci[s], ci[e])
jmin, jmax = np.minimum(cj[s], cj[e]), np.maximum(cj[s], cj[e])
H = np.zeros((n_lon, n_lon, n_lat, n_lat), np.int32)
np.add.at(H, (imin, imax, jmin, jmax), 1)
stay = H.astype(np.int64)
del H
stay = np.flip(np.cumsum(np.flip(stay, 0), 0), 0)   # imin ≥ a
stay = np.cumsum(stay, 1)                            # imax ≤ b-1
stay = np.flip(np.cumsum(np.flip(stay, 2), 2), 2)   # jmin ≥ c
stay = np.cumsum(stay, 3)                            # jmax ≤ d-1

starts = np.bincount(cj[s] * n_lon + ci[s], minlength=n_lat * n_lon).reshape(n_lat, n_lon)

# ── 3. Shipping density on the grid ──────────────────────────────────────────
with zipfile.ZipFile(SHIP_ZIP) as z:
    tif = next(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))
ship = np.zeros((n_lat, n_lon))
with rasterio.open(f"zip://{SHIP_ZIP}!/{tif}") as src:
    k = int(round(RES / src.res[0]))
    for jj in range(n_lat):  # one 1° strip at a time keeps memory small
        la0 = G_LAT_MIN + jj * RES
        win = from_bounds(G_LON_MIN, la0, G_LON_MAX, la0 + RES, src.transform)
        a = src.read(1, window=win.round_offsets().round_lengths()).astype(np.float64)
        a[(a == src.nodata) | (a < 0)] = 0
        a = a[:k, : n_lon * k]
        if a.shape[1] < n_lon * k:  # window rounding can drop a column
            a = np.pad(a, ((0, 0), (0, n_lon * k - a.shape[1])))
        ship[jj] = a.reshape(a.shape[0], n_lon, k).sum(axis=(0, 2))
ship /= ship[ocean].max()  # relative to the busiest ocean cell
print("Shipping density read")

# ── 4. Score every candidate box ─────────────────────────────────────────────
def integral(arr):
    out = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1))
    out[1:, 1:] = arr.astype(float).cumsum(0).cumsum(1)
    return out


I = {"obs": integral(obs), "ocean": integral(ocean), "reliable": integral(reliable),
     "ship": integral(ship * ocean), "starts": integral(starts)}


def box_sum(name, a, b, c, d):
    S = I[name]
    return S[d, b] - S[c, b] - S[d, a] + S[c, a]


ix = lambda x: int(round((x - G_LON_MIN) / RES))
jy = lambda y: int(round((y - G_LAT_MIN) / RES))
mg = int(round(SEARCH_MARGIN / RES))
A_range = range(mg, ix(CORE_LON[0]) + 1)
B_range = range(ix(CORE_LON[1]), n_lon - mg + 1)
cc, dd = np.meshgrid(np.arange(mg, jy(CORE_LAT[0]) + 1),
                     np.arange(jy(CORE_LAT[1]), n_lat - mg + 1), indexing="ij")
cc, dd = cc.ravel(), dd.ravel()

rows = []
for a in A_range:
    for b in B_range:
        n_ocean = box_sum("ocean", a, b, cc, dd)
        rows.append(pd.DataFrame({
            "lon_min": G_LON_MIN + a * RES, "lon_max": G_LON_MIN + b * RES,
            "lat_min": G_LAT_MIN + cc * RES, "lat_max": G_LAT_MIN + dd * RES,
            "ocean_cells": n_ocean,
            "obs": box_sum("obs", a, b, cc, dd),
            "reliable_frac": box_sum("reliable", a, b, cc, dd) / np.maximum(n_ocean, 1),
            "ship_per_cell": box_sum("ship", a, b, cc, dd) / np.maximum(n_ocean, 1),
            "retention": stay[a, b - 1, cc, dd - 1] / np.maximum(box_sum("starts", a, b, cc, dd), 1),
        }))
boxes = pd.concat(rows, ignore_index=True)
boxes["residence_days"] = TAU_DAYS / np.maximum(1 - boxes["retention"], 1e-9)
boxes["retained_1yr"] = boxes["retention"] ** (365 / TAU_DAYS)
print(f"Candidate boxes (contain core, ≥ {SEARCH_MARGIN:g}° inside subset edge): {len(boxes):,}")
limits = (G_LON_MIN + mg * RES, G_LON_MAX - mg * RES, G_LAT_MIN + mg * RES, G_LAT_MAX - mg * RES)
print(f"Search limits: {limits[0]:g}–{limits[1]:g}°E, {-limits[3]:g}–{-limits[2]:g}°S")


def pareto_front(df):
    """Pareto set on (shipping ↑, retention ↑, ocean cells ↓)."""
    vals = df[["ship_per_cell", "retention", "ocean_cells"]].to_numpy() * [1, 1, -1]
    front = []
    for idx in np.argsort(-vals[:, 0], kind="stable"):
        # Boxes earlier in this order have ≥ shipping, so a box is dominated
        # if a kept box also has ≥ retention and ≤ size
        if not any(vals[f, 1] >= vals[idx, 1] and vals[f, 2] >= vals[idx, 2] for f in front):
            front.append(idx)
    return df.iloc[front].sort_values("ship_per_cell", ascending=False)


def touches_limit(bx):
    return [side for side, hit in [("W", bx.lon_min == limits[0]), ("E", bx.lon_max == limits[1]),
                                   ("S", bx.lat_min == limits[2]), ("N", bx.lat_max == limits[3])] if hit]


cols = ["lon_min", "lon_max", "lat_min", "lat_max", "ocean_cells", "obs",
        "reliable_frac", "ship_per_cell", "retention", "residence_days", "retained_1yr"]
fmt = {"obs": "{:,.0f}".format,
       "reliable_frac": "{:.0%}".format, "ship_per_cell": "{:.3f}".format,
       "retention": "{:.3f}".format, "residence_days": "{:.0f}".format,
       "retained_1yr": "{:.1%}".format, "ocean_cells": "{:.0f}".format}

def pick(lo0, lo1, la0, la1):
    return boxes[(boxes.lon_min == lo0) & (boxes.lon_max == lo1) &
                 (boxes.lat_min == la0) & (boxes.lat_max == la1)].iloc[0]


norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)

feas = boxes[(boxes["obs"] >= MIN_OBS) & (boxes["reliable_frac"] >= MIN_RELIABLE)]
print(f"\nFeasible boxes (≥ {MIN_OBS:,} undrogued hourly obs, ≥ {MIN_RELIABLE:.0%} reliable cells): "
      f"{len(feas):,}")
par = pareto_front(feas)
par = par.assign(score=norm(par["ship_per_cell"]) * norm(par["retention"]))
rec = par.sort_values("score", ascending=False).iloc[0]
par.to_csv("data/box_pareto.csv", index=False)
print(f"Pareto-optimal boxes: {len(par)} (saved → data/box_pareto.csv); top 12 by shipping:")
print(par[cols].head(12).to_string(index=False, formatters=fmt))
print("\nSearch top score (best normalised shipping × retention on the Pareto set):")
print(rec[cols].to_frame().T.to_string(index=False, formatters=fmt))
hit = touches_limit(rec)
if hit:
    print(f"  ⚠ touches the search limit on side(s) {', '.join(hit)}; "
          f"larger boxes retain more oil, so the optimum may lie beyond the search area")

opts = pd.DataFrame([pick(*v).rename(k) for k, v in OPTIONS.items()] + [rec.rename("Search top score")])
opts["passes"] = (opts["obs"] >= MIN_OBS) & (opts["reliable_frac"] >= MIN_RELIABLE)
print("\nNamed options side by side:")
print(opts[cols + ["passes"]].to_string(formatters=fmt))
opts.to_csv("data/box_options.csv")
print("Saved → data/box_options.csv")

# ── 5. Figure: map + trade-off ───────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    map_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    map_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}

fig = plt.figure(figsize=(17, 8))
ax = fig.add_subplot(1, 2, 1, **map_kw)
ax2 = fig.add_subplot(1, 2, 2)

lon_e = G_LON_MIN + np.arange(n_lon + 1) * RES
lat_e = G_LAT_MIN + np.arange(n_lat + 1) * RES
ax.pcolormesh(lon_e, lat_e, np.where(reliable, 1.0, np.where(ocean, 0.0, np.nan)),
              cmap=ListedColormap(["#d9d9d9", "#f7fbff"]), vmin=0, vmax=1, **kw)
sh = np.where(ocean & (ship > 0), ship, np.nan)
mesh = ax.pcolormesh(lon_e, lat_e, sh, cmap="Blues",
                     norm=LogNorm(*np.nanpercentile(sh, [30, 99.5])), **kw)


def outline(bx, color, style, width, label):
    x = [bx.lon_min, bx.lon_max, bx.lon_max, bx.lon_min, bx.lon_min]
    y = [bx.lat_min, bx.lat_min, bx.lat_max, bx.lat_max, bx.lat_min]
    ax.plot(x, y, color=color, linestyle=style, linewidth=width, label=label, zorder=6, **kw)


box_label = lambda bx: f"{bx.lon_min:g}–{bx.lon_max:g}°E, {-bx.lat_max:g}–{-bx.lat_min:g}°S"
for name, (color, style, width, _) in OPTION_STYLE.items():
    bx = pick(*OPTIONS[name])
    tag = " (analysis box R)" if name == CHOSEN else ""
    if name != CHOSEN:
        rb = pick(*OPTIONS[CHOSEN])
        shared = [side for side, same in [("W", bx.lon_min == rb.lon_min), ("E", bx.lon_max == rb.lon_max),
                                          ("S", bx.lat_min == rb.lat_min), ("N", bx.lat_max == rb.lat_max)] if same]
        if len(shared) >= 2:  # outline hidden under box R on those sides
            tag = f" (shares {'/'.join(shared)} edges with R)"
    outline(bx, color, style, width, f"{name}{tag}  {box_label(bx)}")
outline(pd.Series({"lon_min": CORE_LON[0], "lon_max": CORE_LON[1],
                   "lat_min": CORE_LAT[0], "lat_max": CORE_LAT[1]}),
        "#3a3a3a", ":", 1.2, "Required core")
outline(pd.Series(dict(zip(["lon_min", "lon_max", "lat_min", "lat_max"], limits))),
        "#9e9e9e", (0, (1, 3)), 1.0, "Search limit")

if ccrs:
    ax.set_extent([G_LON_MIN, G_LON_MAX, G_LAT_MIN, G_LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = gl.right_labels = False
cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03, extend="both", orientation="horizontal")
cb.set_label("Commercial ship density relative to busiest 1° cell (log)")
ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95)
ax.set_title(f"Candidate boxes — grey = ocean cells with < {MIN_DRIFTERS} undrogued drifters",
             fontsize=10)

# Trade-off across all feasible boxes
ax2.scatter(feas["ship_per_cell"], feas["residence_days"], s=4, c="#c6dbef",
            label=f"Feasible boxes ({len(feas):,})", rasterized=True)
ax2.scatter(par["ship_per_cell"], par["residence_days"], s=16, c="#2171b5",
            label=f"Pareto-optimal ({len(par)})")
ax2.set_yscale("log")
ax2.set_xlabel("Mean shipping density per ocean cell (relative)")
ax2.set_ylabel(f"Mean residence time in box (days, from 1-step retention, τ = {TAU_DAYS} d)")
for name, (color, _, _, marker) in OPTION_STYLE.items():
    bx = pick(*OPTIONS[name])
    big = name == CHOSEN
    ax2.scatter([bx.ship_per_cell], [bx.residence_days], s=220 if big else 80, marker=marker,
                c=color, edgecolor="#222222", zorder=6 if big else 5,
                label=f"{name}{' (analysis box R)' if big else ''}")
    ax2.annotate(f"{bx.residence_days:.0f} d", (bx.ship_per_cell, bx.residence_days),
                 xytext=(8, -3), textcoords="offset points", fontsize=8, color="#222222",
                 path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground="white")])
ax2.set_title(f"Trade-off, boxes with ≥ {MIN_OBS/1e6:g}M undrogued hourly obs\n"
              "right = more shipping risk, up = oil stays in the box longer", fontsize=10)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, loc="lower left")

fig.suptitle("Bounding-box selection for the oil (undrogued) transition matrix", fontsize=12)
out = "figures/agulhas_box_selection.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.close()
