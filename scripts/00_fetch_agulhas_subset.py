"""
Agulhas Region — Fetch GDP subset via CloudDrift
Streams the NOAA GDP hourly dataset (zarr on AWS S3, same source as
clouddrift.datasets.gdp1h) chunk by chunk, keeps only observations inside
a padded Agulhas box, thins them to 6-hourly, and saves a small local file.

Transfers ~1.2 GB once (lon/lat must be scanned in full); the saved subset
is what every later script reads.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

ZARR_URL = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp-v2.01.zarr"
OUT_PATH = "data/agulhas_gdp6h_subset.nc"

# ── 1. Region ────────────────────────────────────────────────────────────────
# Analysis box R is 10–40°E, 45–25°S. Save a 10° margin so positions just
# after a drifter leaves R are kept (needed for the "outside R" state and
# exit-edge statistics).
LON_MIN, LON_MAX = 0.0, 50.0
LAT_MIN, LAT_MAX = -55.0, -15.0

STEP_S = 6 * 3600  # thin hourly → 6-hourly
OBS_VARS = ["time", "lon", "lat", "ve", "vn", "drogue_status"]

os.makedirs("data", exist_ok=True)

# ── 2. Open store ────────────────────────────────────────────────────────────
root = zarr.open_group(ZARR_URL, mode="r")
n_obs = root["lon"].shape[0]
chunk = root["lon"].chunks[0]
n_chunks = int(np.ceil(n_obs / chunk))

traj_ids = root["ID"][:]
rowsize = root["rowsize"][:].astype(np.int64)
traj_start = np.concatenate([[0], np.cumsum(rowsize)[:-1]])
print(f"Trajectories: {len(traj_ids):,}   Observations: {n_obs:,}   Chunks: {n_chunks}")


# ── 3. Scan chunks ───────────────────────────────────────────────────────────
def scan(k):
    sl = slice(k * chunk, min((k + 1) * chunk, n_obs))
    lon = root["lon"][sl]
    lat = root["lat"][sl]
    box = (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)
    if not box.any():
        return None
    time = root["time"][sl]
    keep = box & (np.round(time).astype(np.int64) % STEP_S == 0)
    if not keep.any():
        return None
    out = {"obs_index": np.arange(sl.start, sl.stop)[keep],
           "time": time[keep], "lon": lon[keep], "lat": lat[keep]}
    for v in ["ve", "vn", "drogue_status"]:
        out[v] = root[v][sl][keep]
    return out


with ThreadPoolExecutor(max_workers=8) as pool:
    parts = [p for p in tqdm(pool.map(scan, range(n_chunks)), total=n_chunks,
                             desc="Scanning chunks") if p is not None]

sub = {v: np.concatenate([p[v] for p in parts]) for v in ["obs_index"] + OBS_VARS}

# ── 4. Rebuild ragged array for the subset ───────────────────────────────────
traj_of_obs = np.searchsorted(traj_start, sub["obs_index"], side="right") - 1
kept_traj, sub_rowsize = np.unique(traj_of_obs, return_counts=True)

ds = xr.Dataset(
    data_vars={
        "rowsize": ("traj", sub_rowsize.astype(np.int64)),
        "lon": ("obs", sub["lon"].astype(np.float32)),
        "lat": ("obs", sub["lat"].astype(np.float32)),
        "ve": ("obs", sub["ve"].astype(np.float32)),
        "vn": ("obs", sub["vn"].astype(np.float32)),
        "drogue_status": ("obs", sub["drogue_status"].astype(bool)),
    },
    coords={
        "id": ("traj", traj_ids[kept_traj]),
        "time": ("obs", sub["time"].astype("datetime64[s]").astype("datetime64[ns]")),
    },
    attrs={
        "source": ZARR_URL,
        "description": "GDP hourly subset thinned to 6-hourly, padded Agulhas box",
        "lon_range": f"{LON_MIN}..{LON_MAX}",
        "lat_range": f"{LAT_MIN}..{LAT_MAX}",
    },
)
ds.to_netcdf(OUT_PATH)

print(f"\nKept trajectories : {ds.sizes['traj']:,}")
print(f"Kept observations : {ds.sizes['obs']:,}")
print(f"Saved → {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")
