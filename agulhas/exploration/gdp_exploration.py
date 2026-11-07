"""
Agulhas Region — GDP Drifter Exploration
Analyses drifter coverage in the Agulhas region using the local subset
produced by fetch_gdp_subset (GDP hourly via CloudDrift,
native hourly resolution, padded box).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from agulhas.config import box_tag, parse_args, region_prefix
from agulhas.plotting import MAP_KW, ccrs, cfeature, save_figure

MAP_PAD = 2      # degrees shown around the box


def open_subset(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run agulhas/sources/fetch_gdp_subset.py first.")
    print("Opening dataset...")
    ds = xr.open_dataset(path)
    print(ds)
    print(f"\nTrajectories in padded box : {ds.sizes['traj']:,}")
    print(f"Observations in padded box : {ds.sizes['obs']:,}")
    return ds


def plot_coverage(lon, lat, drogued, undrogued, box, path):
    """Every observation in the box, coloured by drogue status."""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=MAP_KW)
    if ccrs:
        lon_min, lon_max, lat_min, lat_max = box
        ax.set_extent([lon_min - MAP_PAD, lon_max + MAP_PAD, lat_min - MAP_PAD, lat_max + MAP_PAD],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="lightgray", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    else:
        print("Cartopy not available — plain matplotlib fallback.")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.scatter(lon[drogued], lat[drogued], s=0.3, c="steelblue", alpha=0.4, label="Drogued (15m)", zorder=4)
    ax.scatter(lon[undrogued], lat[undrogued], s=0.3, c="darkorange", alpha=0.5,
               label="Undrogued (surface + wind slip)", zorder=5)
    ax.set_title(f"GDP Drifter Observations — {box_tag(box)}", fontsize=13)
    ax.legend(markerscale=10, loc="lower right")
    plt.tight_layout()
    save_figure(path)
    print(f"\nMap saved → {path}")


def plot_years(times, box, path):
    """Observations in the box per year."""
    years = pd.DatetimeIndex(times).year
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(years, bins=np.arange(years.min(), years.max() + 2) - 0.5, color="steelblue", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Observations")
    ax.set_title(f"Temporal Distribution of GDP Drifter Obs in {box_tag(box)}")
    plt.tight_layout()
    save_figure(path)
    print(f"Temporal plot saved → {path}")


def main(argv=None):
    args = parse_args(__doc__, argv=argv)
    tag = f"{region_prefix(args.data)}_gdp_%s_{box_tag(args.box)}.png"
    ds = open_subset(args.data)

    lon, lat = ds["lon"].values, ds["lat"].values
    lon_min, lon_max, lat_min, lat_max = args.box
    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    print(f"\nObservations in box : {mask.sum():,}")
    print(f"Fraction of padded subset   : {mask.mean():.2%}")

    rowsize = ds["rowsize"].values
    traj_idx = np.repeat(np.arange(len(rowsize)), rowsize)
    print(f"Trajectories passing through box: {len(np.unique(traj_idx[mask])):,}")

    drogue = ds["drogue_status"].values.astype(bool)
    undrogued, drogued = mask & ~drogue, mask & drogue
    print(f"\nUndrogued obs in box (+ wind slip): {undrogued.sum():,}")
    print(f"Drogued obs in box (15m current) : {drogued.sum():,}")

    plot_coverage(lon, lat, drogued, undrogued, args.box, "figures/" + tag % "coverage")
    plot_years(ds["time"].values[mask], args.box, "figures/" + tag % "temporal")
    print("\nDone.")


if __name__ == "__main__":
    main()
