"""
Agulhas Region — GDP Drifter Exploration
Analyses drifter coverage in the Agulhas region using the local subset
produced by 00_fetch_agulhas_subset.py (GDP hourly via CloudDrift,
thinned to 6-hourly, padded box).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import xarray as xr

# ── 1. Load dataset ──────────────────────────────────────────────────────────
GDP_LOCAL = "data/agulhas_gdp6h_subset.nc"

os.makedirs("figures", exist_ok=True)

if not os.path.exists(GDP_LOCAL):
    raise SystemExit(f"{GDP_LOCAL} not found — run scripts/00_fetch_agulhas_subset.py first.")

print("Opening dataset...")
ds = xr.open_dataset(GDP_LOCAL)

print(ds)
print(f"\nTrajectories in padded box : {ds.sizes['traj']:,}")
print(f"Observations in padded box : {ds.sizes['obs']:,}")

# ── 2. Agulhas bounding box ──────────────────────────────────────────────────
LON_MIN, LON_MAX = 10.0, 40.0
LAT_MIN, LAT_MAX = -45.0, -25.0

lon = ds["lon"].values
lat = ds["lat"].values

mask = (
    (lon >= LON_MIN) & (lon <= LON_MAX) &
    (lat >= LAT_MIN) & (lat <= LAT_MAX)
)

print(f"\nObservations in Agulhas box : {mask.sum():,}")
print(f"Fraction of padded subset   : {mask.mean():.2%}")

# ── 3. Trajectories passing through the box ──────────────────────────────────
rowsize      = ds["rowsize"].values
traj_idx     = np.repeat(np.arange(len(rowsize)), rowsize)
trajs_in_box = np.unique(traj_idx[mask])
print(f"Trajectories passing through Agulhas box: {len(trajs_in_box):,}")

# ── 4. Drogued vs undrogued ──────────────────────────────────────────────────
drogue = ds["drogue_status"].values  # bool or int: True/1 = drogued

mask_undrogued = mask & ~drogue.astype(bool)
mask_drogued   = mask &  drogue.astype(bool)

print(f"\nUndrogued obs in box (oil proxy) : {mask_undrogued.sum():,}")
print(f"Drogued obs in box (15m current) : {mask_drogued.sum():,}")

# ── 5. Map ───────────────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    use_cartopy = True
except ImportError:
    use_cartopy = False
    print("Cartopy not available — plain matplotlib fallback.")

if use_cartopy:
    fig, ax = plt.subplots(figsize=(10, 8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([LON_MIN - 2, LON_MAX + 2, LAT_MIN - 2, LAT_MAX + 2],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    ax.scatter(lon[mask_drogued],   lat[mask_drogued],   s=0.3,
               c="steelblue",  alpha=0.4, label="Drogued (15m)",              zorder=4)
    ax.scatter(lon[mask_undrogued], lat[mask_undrogued], s=0.3,
               c="darkorange", alpha=0.5, label="Undrogued (surface/oil proxy)", zorder=5)
    ax.set_title("GDP Drifter Observations — Agulhas Region", fontsize=13)
    ax.legend(markerscale=10, loc="lower right")
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(lon[mask_drogued],   lat[mask_drogued],   s=0.3,
               c="steelblue",  alpha=0.4, label="Drogued (15m)")
    ax.scatter(lon[mask_undrogued], lat[mask_undrogued], s=0.3,
               c="darkorange", alpha=0.5, label="Undrogued (surface/oil proxy)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("GDP Drifter Observations — Agulhas Region", fontsize=13)
    ax.legend(markerscale=10)

plt.tight_layout()
out_map = "figures/agulhas_gdp_coverage.png"
plt.savefig(out_map, dpi=150, bbox_inches="tight")
print(f"\nMap saved → {out_map}")
plt.close()

# ── 6. Temporal coverage ─────────────────────────────────────────────────────
import pandas as pd

time_in_box = ds["time"].values[mask]
years = pd.DatetimeIndex(time_in_box).year

fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(years, bins=np.arange(years.min(), years.max() + 2) - 0.5,
        color="steelblue", edgecolor="white")
ax.set_xlabel("Year")
ax.set_ylabel("Observations")
ax.set_title("Temporal Distribution of GDP Drifter Obs in Agulhas Box")
plt.tight_layout()
out_time = "figures/agulhas_gdp_temporal.png"
plt.savefig(out_time, dpi=150, bbox_inches="tight")
print(f"Temporal plot saved → {out_time}")
plt.close()

print("\nDone.")
