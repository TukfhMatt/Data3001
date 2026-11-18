"""
Agulhas Region — OpenDrift Forcing
Downloads what the OpenDrift comparison runs on, for 2000, 2005, 2010, 2015 and
2020 and the year after each (the year-long runs of opendrift_year and
opendrift_drifters_year end in it): GLORYS12 daily surface currents at 1/12° from
Copernicus Marine, and ERA5 10 m wind with the same area, grid and times as
fetch_era5 (R plus 5°, 0.5°, 6-hourly). Each year runs to 3 January of the next
year, so a December start can reach 31.5 days. Finished files are skipped, so an
interrupted run resumes.

The files are published on Zenodo (about 5 GB, checked against their SHA-256), so
a run needs no account. --rebuild requests them from the sources instead: it needs
a Copernicus Marine login (copernicusmarine login) and the ERA5 token in
~/.cdsapirc, and the ERA5 requests can wait hours in the CDS queue.

    .venv-opendrift/bin/python -m agulhas opendrift_forcing [--rebuild]
"""

import os
import tarfile
import time

from agulhas.config import parse_args
from agulhas.download import download
from agulhas.opendrift_common import FORCING, YEARS

GLORYS = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
AREA = (5, 60, -50, -10)     # lon_min, lon_max, lat_min, lat_max: R plus 5°
ATTEMPTS = 5
FORCING_YEARS = sorted(set(YEARS) | {year + 1 for year in YEARS})
ZENODO = "https://zenodo.org/records/24266249/files"
ARCHIVES = {          # name → SHA-256 of the tar, which holds the files at its top level
    "agulhas_opendrift_forcing_currents.tar": "91d62e99d29e63c9b54326aa6125c6e504285e39c3c5b25a0d8a1a35089a61b4",
    "agulhas_opendrift_forcing_currents_next.tar": "a9654fd30c9c88db8bfc99b50a8dfe2f4090154ef7f6c55795565701c6ddc1f8",
    "agulhas_opendrift_forcing_wind.tar": "c30f77b50aaba88de322be0edf5b0c50d2277ed9885cfd6d47609c246d0ca7fa",
    "agulhas_opendrift_forcing_wind_next.tar": "af2632dfce5927bfa6b2ea95fbed8a4ace5767cbdb80804508a46f91a31c0a10",
}


# ── Published files ──────────────────────────────────────────────────────────

def expected_files():
    """Every file the comparison steps read, as rebuild names them."""
    names = [f"glorys_{year}.nc" for year in FORCING_YEARS] + [f"era5_{year}.nc" for year in FORCING_YEARS]
    return names + [f"era5_{year + 1}01.nc" for year in FORCING_YEARS]


def missing_files():
    """The expected files not yet in the forcing folder."""
    return [name for name in expected_files() if not os.path.exists(f"{FORCING}/{name}")]


def unpack(archive, wanted):
    """Extract the wanted files from a tar of plain files into the forcing folder, keeping any already there."""
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile() and member.name in wanted and not os.path.exists(f"{FORCING}/{member.name}"):
                member.name = os.path.basename(member.name)
                tar.extract(member, FORCING)
                print(f"{FORCING}/{member.name}", flush=True)


def download_published():
    """Download each published archive, check it and unpack the files still missing."""
    wanted = set(missing_files())
    for name, checksum in ARCHIVES.items():
        archive = f"{FORCING}/{name}"
        print(f"Downloading {ZENODO}/{name}", flush=True)
        download(f"{ZENODO}/{name}", archive, checksum)
        unpack(archive, wanted)
        os.remove(archive)


# ── Rebuild from the sources ─────────────────────────────────────────────────

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


def add_options(p):
    p.add_argument("--rebuild", action="store_true", help="request the files from Copernicus Marine and the CDS, not Zenodo")


def rebuild():
    """Request every missing file from Copernicus Marine and the CDS."""
    import cdsapi
    import copernicusmarine

    os.makedirs(FORCING, exist_ok=True)
    lon0, lon1, lat0, lat1 = AREA

    # ── 1. Currents ──────────────────────────────────────────────────────────
    for year in FORCING_YEARS:
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
    for year in FORCING_YEARS:
        whole = {"year": [str(year)], "month": [f"{m:02d}" for m in range(1, 13)], "day": [f"{d:02d}" for d in range(1, 32)]}
        january = {"year": [str(year + 1)], "month": ["01"], "day": ["01", "02", "03"]}
        for name, dates in ((f"era5_{year}.nc", whole), (f"era5_{year + 1}01.nc", january)):
            fetch(f"{FORCING}/{name}", lambda part, dates=dates: client.retrieve("reanalysis-era5-single-levels", {**base, **dates}, part))


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, extra=add_options, argv=argv)
    if args.rebuild:
        rebuild()
    elif missing_files():
        download_published()
    left = missing_files()
    if left:
        raise SystemExit(f"{len(left)} forcing files are not in the published archives, e.g. {left[0]}; "
                         "run with --rebuild (needs a Copernicus Marine login and ~/.cdsapirc).")
    print(f"All {len(expected_files())} forcing files are in {FORCING}/")


if __name__ == "__main__":
    main()
