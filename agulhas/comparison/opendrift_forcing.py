"""
Agulhas Region — OpenDrift Forcing
Downloads what the OpenDrift comparison runs on, for 2000, 2005, 2010, 2015 and
2020: GLORYS12 daily surface currents at 1/12° from Copernicus Marine, and ERA5
10 m wind with the same area, grid and times as fetch_era5 (R plus 5°, 0.5°,
6-hourly). Each year runs to 3 January of the next year, so a December start can
reach 31.5 days. Finished files are skipped, so an interrupted run resumes.

Needs the OpenDrift environment (requirements-opendrift.txt), a Copernicus Marine
login (copernicusmarine login) and the ERA5 token in ~/.cdsapirc. About 3 GB.

    .venv-opendrift/bin/python -m agulhas opendrift_forcing
"""

import os
import time

from agulhas.opendrift_common import FORCING, YEARS

GLORYS = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
AREA = (5, 60, -50, -10)     # lon_min, lon_max, lat_min, lat_max: R plus 5°
ATTEMPTS = 5


def fetch(path, get):
    """Run a download to a temporary name, with retries, so an interrupted one never looks finished."""
    if os.path.exists(path):
        print(f"{path} exists", flush=True)
        return
    part = path + ".part.nc"
    for attempt in range(ATTEMPTS):
        try:
            get(part)
            os.replace(part, path)
            print(f"{path}: {os.path.getsize(path) / 1e6:.0f} MB, {time.strftime('%H:%M')}", flush=True)
            return
        except Exception as e:
            print(f"{path}: attempt {attempt + 1} failed: {str(e)[:200]}", flush=True)
            time.sleep(120)
    raise SystemExit(f"{path}: gave up after {ATTEMPTS} attempts")


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    import cdsapi
    import copernicusmarine

    os.makedirs(FORCING, exist_ok=True)
    lon0, lon1, lat0, lat1 = AREA

    # ── 1. Currents ──────────────────────────────────────────────────────────
    for year in YEARS:
        def get(part, year=year):
            copernicusmarine.subset(dataset_id=GLORYS, variables=["uo", "vo"], minimum_longitude=lon0, maximum_longitude=lon1,
                                    minimum_latitude=lat0, maximum_latitude=lat1, minimum_depth=0, maximum_depth=1,
                                    start_datetime=f"{year}-01-01T00:00:00", end_datetime=f"{year + 1}-01-03T23:59:59",
                                    output_directory=os.path.dirname(part), output_filename=os.path.basename(part),
                                    disable_progress_bar=True)
        fetch(f"{FORCING}/glorys_{year}.nc", get)

    # ── 2. Wind ──────────────────────────────────────────────────────────────
    client = cdsapi.Client(quiet=True)
    base = {"product_type": ["reanalysis"], "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "time": ["00:00", "06:00", "12:00", "18:00"], "area": [lat1, lon0, lat0, lon1], "grid": [0.5, 0.5],
            "data_format": "netcdf", "download_format": "unarchived"}
    for year in YEARS:
        whole = {"year": [str(year)], "month": [f"{m:02d}" for m in range(1, 13)], "day": [f"{d:02d}" for d in range(1, 32)]}
        january = {"year": [str(year + 1)], "month": ["01"], "day": ["01", "02", "03"]}
        for name, dates in ((f"era5_{year}.nc", whole), (f"era5_{year + 1}01.nc", january)):
            fetch(f"{FORCING}/{name}", lambda part, dates=dates: client.retrieve("reanalysis-era5-single-levels", {**base, **dates}, part))


if __name__ == "__main__":
    main()
