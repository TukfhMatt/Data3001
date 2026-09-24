"""
Agulhas Region — Grid Coverage
Counts drifters per grid cell in box R at 1° and 0.5° resolution, split by
drogue status, to choose the transition-matrix grid.

Observations along one trajectory are strongly correlated, so the main
measure is the number of *distinct drifters* per cell, not raw obs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr

GDP_LOCAL = "data/agulhas_gdp6h_subset.nc"
os.makedirs("figures", exist_ok=True)

# ── 1. Load subset, restrict to R ────────────────────────────────────────────
LON_MIN, LON_MAX = 10.0, 40.0
LAT_MIN, LAT_MAX = -45.0, -25.0

ds = xr.open_dataset(GDP_LOCAL)
lon = ds["lon"].values
lat = ds["lat"].values
drogue = ds["drogue_status"].values.astype(bool)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)

in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)

RESOLUTIONS = [1.0, 0.5]
THRESHOLDS  = [5, 10, 20]   # distinct drifters per cell
SUBSETS = {
    "Drogued (container)":  in_R &  drogue,
    "Undrogued (oil)":      in_R & ~drogue,
}


# ── 2. Per-cell counts ───────────────────────────────────────────────────────
def cell_counts(mask, res):
    """Return (obs count, distinct drifter count) grids, shape (n_lat, n_lon)."""
    n_lon = int(round((LON_MAX - LON_MIN) / res))
    n_lat = int(round((LAT_MAX - LAT_MIN) / res))
    i = ((lon[mask] - LON_MIN) // res).astype(int)
    j = ((lat[mask] - LAT_MIN) // res).astype(int)
    cell = j * n_lon + i
    obs = np.bincount(cell, minlength=n_lat * n_lon)
    pairs = np.unique(np.stack([cell, traj_idx[mask]]), axis=1)
    drifters = np.bincount(pairs[0], minlength=n_lat * n_lon)
    return obs.reshape(n_lat, n_lon), drifters.reshape(n_lat, n_lon)


results = {}
for res in RESOLUTIONS:
    # Ocean cells = cells visited by any drifter (drogued or not) in R
    _, any_drifters = cell_counts(in_R, res)
    ocean = any_drifters > 0
    for name, mask in SUBSETS.items():
        obs, drifters = cell_counts(mask, res)
        results[(name, res)] = (obs, drifters, ocean)

# ── 3. Summary table ─────────────────────────────────────────────────────────
print(f"Distinct drifters per ocean cell in R ({LON_MIN:g}–{LON_MAX:g}°E, "
      f"{-LAT_MIN:g}–{-LAT_MAX:g}°S)\n")
header = f"{'Subset':<22}{'Res':>5}{'Cells':>7}{'Median':>8}" + "".join(f"{'≥' + str(t):>7}" for t in THRESHOLDS)
print(header)
print("-" * len(header))
for (name, res), (obs, drifters, ocean) in results.items():
    d = drifters[ocean]
    frac = "".join(f"{(d >= t).mean():>7.0%}" for t in THRESHOLDS)
    print(f"{name:<22}{res:>4}°{ocean.sum():>7}{np.median(d):>8.0f}{frac}")

np.savez(
    "data/grid_coverage.npz",
    lon_min=LON_MIN, lon_max=LON_MAX, lat_min=LAT_MIN, lat_max=LAT_MAX,
    **{f"{'drogued' if n.startswith('Drogued') else 'undrogued'}_{r}_{k}": v
       for (n, r), (o, d, oc) in results.items()
       for k, v in [("obs", o), ("drifters", d), ("ocean", oc)]},
)
print("\nCounts saved → data/grid_coverage.npz")

# ── 4. Maps ──────────────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}

MIN_DRIFTERS = 10  # cells below this are shown in grey
cmap = plt.get_cmap("Blues").copy()
cmap.set_under("#d9d9d9")
vmax = max(d.max() for _, d, _ in results.values())
norm = LogNorm(vmin=MIN_DRIFTERS, vmax=vmax)

fig, axes = plt.subplots(len(SUBSETS), len(RESOLUTIONS), figsize=(13, 8.5),
                         subplot_kw=subplot_kw, constrained_layout=True)
for r, name in enumerate(SUBSETS):
    for c, res in enumerate(RESOLUTIONS):
        ax = axes[r, c]
        obs, drifters, ocean = results[(name, res)]
        grid = np.where(ocean, np.maximum(drifters, 0.5), np.nan)  # 0.5 → "under"
        lon_edges = np.arange(LON_MIN, LON_MAX + res / 2, res)
        lat_edges = np.arange(LAT_MIN, LAT_MAX + res / 2, res)
        kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
        mesh = ax.pcolormesh(lon_edges, lat_edges, grid, cmap=cmap, norm=norm,
                             edgecolor="white", linewidth=0.2, **kw)
        if ccrs:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color="#f0f0f0", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
            gl.top_labels = gl.right_labels = False
        pct = (drifters[ocean] >= MIN_DRIFTERS).mean()
        ax.set_title(f"{name} — {res}° grid  ({pct:.0%} of cells ≥ {MIN_DRIFTERS} drifters)",
                     fontsize=10)

cb = fig.colorbar(mesh, ax=axes, shrink=0.6, extend="min")
cb.set_label(f"Distinct drifters per cell (grey = fewer than {MIN_DRIFTERS})")
fig.suptitle("Drifter coverage per grid cell — Agulhas box R", fontsize=13)

out = "figures/agulhas_grid_coverage.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Map saved → {out}")
plt.close()
