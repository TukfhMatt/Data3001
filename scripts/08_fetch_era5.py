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

    .venv/bin/python scripts/08_fetch_era5.py
"""

import os
import shutil

import cdsapi
import numpy as np
import xarray as xr

from config import parse_args

args = parse_args(__doc__)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box

OUT_DIR = "data/era5"                   # per-year wind at drifter positions
OUT_PATH = "data/drifter_wind10m.nc"    # all years, aligned with the subset's obs
MARGIN = 5.0          # ° around R
GRID = 0.5            # ° ERA5 output grid
TIMES = ["00:00", "06:00", "12:00", "18:00"]
MIN_FREE_GB = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Drifter positions and years ───────────────────────────────────────────
ds = xr.open_dataset(args.data)
lon, lat = ds["lon"].values.astype(float), ds["lat"].values.astype(float)
time = ds["time"].values
obs_year = time.astype("datetime64[Y]").astype(int) + 1970
in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
years = np.unique(obs_year[in_R])
area = [LAT_MAX + MARGIN, LON_MIN - MARGIN, LAT_MIN - MARGIN, LON_MAX + MARGIN]  # N, W, S, E
in_area = (lon >= area[1]) & (lon <= area[3]) & (lat >= area[2]) & (lat <= area[0])
print(f"Years {years.min()}–{years.max()} ({len(years)}), area N/W/S/E {area}, {GRID}° grid, "
      f"{len(TIMES)} times a day; {in_area.sum():,} drifter positions in the area")

# ── 2. One request per year → wind at drifter positions → delete raw file ────
client = cdsapi.Client()
for year in years:
    part = f"{OUT_DIR}/wind_at_drifters_{year}.npz"
    if os.path.exists(part):
        print(f"{year}: processed, skipped")
        continue
    free_gb = shutil.disk_usage(OUT_DIR).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"Only {free_gb:.1f} GB free — stopping before {year}. Free space and rerun.")
    raw = f"{OUT_DIR}/era5_wind10m_{year}.nc"
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
        raw,
    )
    # ERA5 stores latitude north → south; the extra `expver` / `number`
    # coordinates stop xarray from interpolating to points, so they are dropped
    era = xr.open_dataset(raw).sortby("latitude").drop_vars(["expver", "number"], errors="ignore")
    tname = "valid_time" if "valid_time" in era.dims else "time"
    idx = np.where(in_area & (obs_year == year))[0]
    pts = dict(longitude=xr.DataArray(lon[idx], dims="obs"),
               latitude=xr.DataArray(lat[idx], dims="obs"),
               **{tname: xr.DataArray(time[idx], dims="obs")})
    w = era[["u10", "v10"]].interp(**pts)  # bilinear in space, linear in time
    if w["u10"].dims != ("obs",):
        raise SystemExit(f"Interpolation kept dims {w['u10'].dims}; expected one value per position.")
    np.savez_compressed(part, obs=idx, u10=w["u10"].values.astype(np.float32),
                        v10=w["v10"].values.astype(np.float32))
    era.close()
    os.remove(raw)
    print(f"{year}: {len(idx):,} positions, {np.isnan(w['u10'].values).mean():.2%} without wind → {part}")

# ── 3. Combine into one file aligned with the subset's observations ──────────
u10 = np.full(len(lon), np.nan, np.float32)
v10 = np.full(len(lon), np.nan, np.float32)
for year in years:
    with np.load(f"{OUT_DIR}/wind_at_drifters_{year}.npz") as f:
        u10[f["obs"]], v10[f["obs"]] = f["u10"], f["v10"]
xr.Dataset(
    {"u10": ("obs", u10, {"units": "m s-1", "long_name": "ERA5 10 m eastward wind at drifter position"}),
     "v10": ("obs", v10, {"units": "m s-1", "long_name": "ERA5 10 m northward wind at drifter position"})},
    attrs={"source": "ERA5 hourly single levels (Copernicus CDS), 0.5°, 6-hourly, interpolated",
           "subset": args.data, "area_NWSE": str(area)},
).to_netcdf(OUT_PATH)
print(f"\nWind at {np.isfinite(u10).sum():,} of {len(u10):,} drifter positions → {OUT_PATH}")
