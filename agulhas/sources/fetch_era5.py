"""
Agulhas Region — ERA5 10 m Wind at Drifter Positions
Requests ERA5 hourly-reanalysis 10 m wind components (u10, v10) from the
Copernicus Climate Data Store API one year at a time, interpolates them to
every drifter position in that year, and deletes the raw year file. Only the
wind at drifter positions is kept, so at most one year of ERA5 (~100 MB) is
on disk at once. The oil version uses these winds to add the windage
undrogued drifters do not already carry.

  • area   — R plus a 5° margin, so the wind is known along every 3.5-day
             transition, including the part after a drifter leaves R
  • grid   — 0.5° (finer than the 1° transition grid), bilinear in space
  • times  — 00, 06, 12, 18 UTC, linear in time to the hourly positions

Needs a CDS account, the ERA5 licence accepted on the dataset page, and the
API token in ~/.cdsapirc (url + key). Years already processed are skipped,
so an interrupted run resumes where it stopped.

    python -m agulhas fetch_era5
"""

import os
import shutil

import cdsapi
import numpy as np
import xarray as xr

from agulhas.config import parse_args, wind_paths
from agulhas.drifters import Drifters, in_box

MARGIN = 5.0          # ° around R
GRID = 0.5            # ° ERA5 output grid
TIMES = ["00:00", "06:00", "12:00", "18:00"]
MIN_FREE_GB = 0.5


def request_year(client, year, area, path):
    """Download one year of ERA5 10 m wind over the area (N, W, S, E)."""
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": TIMES,
            "area": area,
            "grid": [GRID, GRID],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        path,
    )


def wind_at_positions(raw, lon, lat, time):
    """Interpolate a raw year to the positions: bilinear in space, linear in time.
    ERA5 stores latitude north → south; the extra `expver` / `number` coordinates stop
    xarray from interpolating to points, so they are dropped."""
    era = xr.open_dataset(raw).sortby("latitude").drop_vars(["expver", "number"], errors="ignore")
    tname = "valid_time" if "valid_time" in era.dims else "time"
    pts = dict(longitude=xr.DataArray(lon, dims="obs"), latitude=xr.DataArray(lat, dims="obs"),
               **{tname: xr.DataArray(time, dims="obs")})
    w = era[["u10", "v10"]].interp(**pts)
    if w["u10"].dims != ("obs",):
        raise SystemExit(f"Interpolation kept dims {w['u10'].dims}; expected one value per position.")
    u10, v10 = w["u10"].values.astype(np.float32), w["v10"].values.astype(np.float32)
    era.close()
    return u10, v10


def fetch_year(client, year, area, positions, out_dir):
    """Wind at one year's positions, saved per year; the raw year is deleted, and a year
    already processed is skipped, so an interrupted run resumes."""
    lon, lat, time, in_area, obs_year = positions
    part = f"{out_dir}/wind_at_drifters_{year}.npz"
    if os.path.exists(part):
        print(f"{year}: processed, skipped")
        return
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"Only {free_gb:.1f} GB free — stopping before {year}. Free space and rerun.")
    raw = f"{out_dir}/era5_wind10m_{year}.nc"
    request_year(client, year, area, raw)
    idx = np.where(in_area & (obs_year == year))[0]
    u10, v10 = wind_at_positions(raw, lon[idx], lat[idx], time[idx])
    np.savez_compressed(part, obs=idx, u10=u10, v10=v10)
    os.remove(raw)
    print(f"{year}: {len(idx):,} positions, {np.isnan(u10).mean():.2%} without wind → {part}")


def combine_years(years, n_obs, out_dir, area, data, path):
    """One file of wind aligned with the subset's observations."""
    u10 = np.full(n_obs, np.nan, np.float32)
    v10 = np.full(n_obs, np.nan, np.float32)
    for year in years:
        with np.load(f"{out_dir}/wind_at_drifters_{year}.npz") as f:
            u10[f["obs"]], v10[f["obs"]] = f["u10"], f["v10"]
    xr.Dataset(
        {"u10": ("obs", u10, {"units": "m s-1", "long_name": "ERA5 10 m eastward wind at drifter position"}),
         "v10": ("obs", v10, {"units": "m s-1", "long_name": "ERA5 10 m northward wind at drifter position"})},
        attrs={"source": "ERA5 hourly single levels (Copernicus CDS), 0.5°, 6-hourly, interpolated",
               "subset": data, "area_NWSE": str(area)},
    ).to_netcdf(path)
    print(f"\nWind at {np.isfinite(u10).sum():,} of {n_obs:,} drifter positions → {path}")


def main(argv=None):
    args = parse_args(__doc__, argv=argv)
    lon_min, lon_max, lat_min, lat_max = args.box
    out_dir, out_path = wind_paths(args.data)
    os.makedirs(out_dir, exist_ok=True)

    drifters = Drifters(args.data)
    lon, lat, time = drifters.lon, drifters.lat, drifters.ds["time"].values
    obs_year = drifters.year
    years = np.unique(obs_year[in_box(lon, lat, args.box)])
    area = [lat_max + MARGIN, lon_min - MARGIN, lat_min - MARGIN, lon_max + MARGIN]  # N, W, S, E
    in_area = (lon >= area[1]) & (lon <= area[3]) & (lat >= area[2]) & (lat <= area[0])
    print(f"Years {years.min()}–{years.max()} ({len(years)}), area N/W/S/E {area}, {GRID}° grid, "
          f"{len(TIMES)} times a day; {in_area.sum():,} drifter positions in the area")

    client = cdsapi.Client()
    for year in years:
        fetch_year(client, year, area, (lon, lat, time, in_area, obs_year), out_dir)
    combine_years(years, len(lon), out_dir, area, args.data, out_path)


if __name__ == "__main__":
    main()
