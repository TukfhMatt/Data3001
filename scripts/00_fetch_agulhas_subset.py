"""
Agulhas Region — Fetch GDP subset via CloudDrift
Streams the NOAA GDP hourly dataset (zarr on AWS S3, same source as
clouddrift.datasets.gdp1h) chunk by chunk, keeps only observations inside
a padded Agulhas box at the native hourly resolution, drops positions
interpolated across long gaps between satellite fixes, and saves a local file.

Transfers ~1.2 GB once (lon/lat must be scanned in full); the saved subset
is what every later script reads. Each drifter also keeps GDP's per-drifter
fate: how its record ended (`typedeath`, 1 = ran aground) and its last good
date (`end_date`), which the stranded state in 03 is built from.

    .venv/bin/python scripts/00_fetch_agulhas_subset.py
    .venv/bin/python scripts/00_fetch_agulhas_subset.py --metadata-only   # refresh the per-drifter fields only
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

ZARR_URL = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp-v2.01.zarr"
OUT_PATH = "data/agulhas_gdp1h_subset.nc"

# ── 1. Region ────────────────────────────────────────────────────────────────
# The saved area holds every candidate box from 05_box_selection.py (search
# area 0–60°E, 55–10°S) plus a margin, so positions just after a drifter
# leaves the box are kept (needed for the exit states and exit-edge statistics).
LON_MIN, LON_MAX = -5.0, 65.0
LAT_MIN, LAT_MAX = -60.0, -5.0

STEP_S = 3600       # keep every hourly position (set 6 * 3600 to thin to 6-hourly)
MAX_GAP_S = 6 * 3600  # drop positions interpolated across > 6 h between fixes
OBS_VARS = ["time", "lon", "lat", "ve", "vn", "drogue_status", "gap"]
TYPEDEATH = ("0 buoy still alive, 1 ran aground, 2 picked up by vessel, 3 stopped transmitting, "
             "4 sporadic transmissions, 5 bad batteries, 6 inactive status")

parser = argparse.ArgumentParser(description="Fetch the GDP hourly subset for the Agulhas region")
parser.add_argument("--metadata-only", action="store_true",
                    help="add the per-drifter fields to an existing subset without rescanning")
args = parser.parse_args()

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


def drifter_fields(ids):
    """Per-drifter fate for the given GDP IDs, in the same order."""
    pos = np.searchsorted(traj_ids, ids) if np.all(np.diff(traj_ids) > 0) else \
        np.array([np.where(traj_ids == i)[0][0] for i in ids])
    if not np.array_equal(traj_ids[pos], ids):
        raise SystemExit("Subset drifter IDs not found in the GDP store.")
    end = root["end_date"][:][pos].astype(np.float64)
    return {
        "typedeath": ("traj", root["typedeath"][:][pos].astype(np.int8),
                      {"long_name": "How the drifter's record ended", "flag_meanings": TYPEDEATH}),
        "end_date": ("traj", np.round(end).astype("datetime64[s]").astype("datetime64[ns]"),
                     {"long_name": "Last good date of the drifter's full record (DAC quality control)"}),
    }


if args.metadata_only:
    ds = xr.load_dataset(OUT_PATH)
    ds = ds.assign_coords(drifter_fields(ds["id"].values))
    tmp = OUT_PATH + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, OUT_PATH)
    print(f"Per-drifter fields added for {ds.sizes['traj']:,} drifters; ran aground: "
          f"{int((ds['typedeath'].values == 1).sum()):,} → {OUT_PATH}")
    raise SystemExit


# ── 3. Scan chunks ───────────────────────────────────────────────────────────
def scan(k):
    sl = slice(k * chunk, min((k + 1) * chunk, n_obs))
    lon = root["lon"][sl]
    lat = root["lat"][sl]
    box = (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)
    if not box.any():
        return None
    time = root["time"][sl]
    gap = root["gap"][sl]
    keep = box & (np.round(time).astype(np.int64) % STEP_S == 0) & (gap <= MAX_GAP_S)
    n_gap = int((box & (gap > MAX_GAP_S)).sum())
    if not keep.any():
        return {"n_gap": n_gap}
    out = {"obs_index": np.arange(sl.start, sl.stop)[keep], "time": time[keep],
           "lon": lon[keep], "lat": lat[keep], "gap": gap[keep], "n_gap": n_gap}
    for v in ["ve", "vn", "drogue_status"]:
        out[v] = root[v][sl][keep]
    return out


with ThreadPoolExecutor(max_workers=8) as pool:
    parts = [p for p in tqdm(pool.map(scan, range(n_chunks)), total=n_chunks,
                             desc="Scanning chunks") if p is not None]

n_gap = sum(p["n_gap"] for p in parts)
parts = [p for p in parts if "obs_index" in p]
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
        "gap": ("obs", sub["gap"].astype(np.float32), {"units": "s",
                "long_name": "Time interval between previous and next location"}),
    },
    coords={
        "id": ("traj", traj_ids[kept_traj]),
        **drifter_fields(traj_ids[kept_traj]),
        "time": ("obs", sub["time"].astype("datetime64[s]").astype("datetime64[ns]")),
    },
    attrs={
        "source": ZARR_URL,
        "description": f"GDP hourly subset (step {STEP_S // 3600} h, gap <= {MAX_GAP_S // 3600} h), padded Agulhas box",
        "lon_range": f"{LON_MIN}..{LON_MAX}",
        "lat_range": f"{LAT_MIN}..{LAT_MAX}",
    },
)
ds.to_netcdf(OUT_PATH)

print(f"\nDropped (gap > {MAX_GAP_S // 3600} h): {n_gap:,}")
print(f"Kept trajectories : {ds.sizes['traj']:,}")
print(f"Kept observations : {ds.sizes['obs']:,}")
print(f"Saved → {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")
