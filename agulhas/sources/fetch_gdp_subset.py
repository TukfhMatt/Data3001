"""
Agulhas Region — Fetch GDP subset via CloudDrift
Streams the NOAA GDP hourly dataset (zarr on AWS S3, same source as
clouddrift.datasets.gdp1h) chunk by chunk, keeps only observations inside
a padded Agulhas box at the native hourly resolution, drops positions
interpolated across long gaps between satellite fixes, and saves a local file.

Transfers ~1.2 GB once (lon/lat must be scanned in full); the saved subset
is what every later script reads. Each drifter also keeps GDP's per-drifter
fate: how its record ended (`typedeath`, 1 = ran aground) and its last good
date (`end_date`), which the stranded state in transition_matrix is built from.

    python -m agulhas fetch_gdp_subset
    python -m agulhas fetch_gdp_subset --metadata-only   # refresh the per-drifter fields only
    python -m agulhas fetch_gdp_subset --fetch-box -90 -30 15 55 \
        --out data/gulfstream_gdp1h_subset.nc

Another region needs both --fetch-box (the box of interest plus a margin of
about 10°, so moves leaving it are seen) and its own --out file; every later
script then takes that file with --data.
"""

import argparse
import os
import time as time_module
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm
from zarr.storage import FSStore

ZARR_URL = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp-v2.01.zarr"
DEFAULT_OUT = "data/agulhas_gdp1h_subset.nc"
# The saved area holds every candidate box from box_selection (search area
# 0–60°E, 55–10°S) plus a margin, so positions just after a drifter leaves the
# box are kept (needed for the exit states and exit-edge statistics).
FETCH_BOX = [-5.0, 65.0, -60.0, -5.0]   # lon_min, lon_max, lat_min, lat_max
STEP_S = 3600       # keep every hourly position (set 6 * 3600 to thin to 6-hourly)
MAX_GAP_S = 6 * 3600  # drop positions interpolated across > 6 h between fixes
OBS_VARS = ["time", "lon", "lat", "ve", "vn", "drogue_status", "gap"]
MAX_TRIES = 5       # attempts per chunk before stopping
TYPEDEATH = ("0 buoy still alive, 1 ran aground, 2 picked up by vessel, 3 stopped transmitting, "
             "4 sporadic transmissions, 5 bad batteries, 6 inactive status")


class GdpStore:
    """The global GDP zarr store, read one chunk at a time."""

    def __init__(self):
        # FSStore's default treats any OSError on a read as a missing chunk and fills
        # it with NaN, which would silently drop that chunk's drifters. Only a chunk
        # that is truly absent (KeyError) may count as missing; other errors are raised.
        self.root = zarr.open_group(FSStore(ZARR_URL, mode="r", exceptions=(KeyError,)), mode="r")
        self.n_obs = self.root["lon"].shape[0]
        self.chunk = self.root["lon"].chunks[0]
        self.n_chunks = int(np.ceil(self.n_obs / self.chunk))
        self.traj_ids = self.root["ID"][:]
        rowsize = self.root["rowsize"][:].astype(np.int64)
        self.traj_start = np.concatenate([[0], np.cumsum(rowsize)[:-1]])
        print(f"Trajectories: {len(self.traj_ids):,}   Observations: {self.n_obs:,}   Chunks: {self.n_chunks}")

    def drifter_fields(self, ids):
        """Per-drifter fate for the given GDP IDs, in the same order."""
        traj_ids = self.traj_ids
        pos = np.searchsorted(traj_ids, ids) if np.all(np.diff(traj_ids) > 0) else \
            np.array([np.where(traj_ids == i)[0][0] for i in ids])
        if not np.array_equal(traj_ids[pos], ids):
            raise SystemExit("Subset drifter IDs not found in the GDP store.")
        end = self.root["end_date"][:][pos].astype(np.float64)
        return {
            "typedeath": ("traj", self.root["typedeath"][:][pos].astype(np.int8),
                          {"long_name": "How the drifter's record ended", "flag_meanings": TYPEDEATH}),
            "end_date": ("traj", np.round(end).astype("datetime64[s]").astype("datetime64[ns]"),
                         {"long_name": "Last good date of the drifter's full record (DAC quality control)"}),
        }

    def scan(self, k, box):
        """What chunk k holds inside the box: its kept observations, or None if it holds nothing."""
        root = self.root
        lon_min, lon_max, lat_min, lat_max = box
        sl = slice(k * self.chunk, min((k + 1) * self.chunk, self.n_obs))
        lon = root["lon"][sl]
        lat = root["lat"][sl]
        inside = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        if not inside.any():
            return None
        time = root["time"][sl]
        gap = root["gap"][sl]
        keep = inside & (np.round(time).astype(np.int64) % STEP_S == 0) & (gap <= MAX_GAP_S)
        n_gap = int((inside & (gap > MAX_GAP_S)).sum())
        if not keep.any():
            return {"n_gap": n_gap}
        out = {"obs_index": np.arange(sl.start, sl.stop)[keep], "time": time[keep],
               "lon": lon[keep], "lat": lat[keep], "gap": gap[keep], "n_gap": n_gap}
        for v in ["ve", "vn", "drogue_status"]:
            out[v] = root[v][sl][keep]
        return out

    def scan_retrying(self, k, box):
        """Scan one chunk, retrying read errors; a chunk that keeps failing stops the run."""
        for attempt in range(MAX_TRIES):
            try:
                return self.scan(k, box)
            except Exception as e:
                if attempt == MAX_TRIES - 1:
                    raise SystemExit(f"Chunk {k} failed after {MAX_TRIES} tries ({e!r}); nothing saved, rerun.")
                time_module.sleep(2 ** attempt)


def parse_options(argv):
    parser = argparse.ArgumentParser(description="Fetch a GDP hourly subset for one region")
    parser.add_argument("--fetch-box", nargs=4, type=float, default=FETCH_BOX,
                        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"), help="area to save")
    parser.add_argument("--out", default=DEFAULT_OUT, help="subset file to write")
    parser.add_argument("--metadata-only", action="store_true",
                        help="add the per-drifter fields to an existing subset without rescanning")
    args = parser.parse_args(argv)
    if args.fetch_box != FETCH_BOX and args.out == DEFAULT_OUT:
        raise SystemExit(f"--fetch-box needs its own --out file, or it would overwrite {DEFAULT_OUT}.")
    return args


def refresh_drifter_fields(store, path):
    """Add the per-drifter fields to an existing subset."""
    ds = xr.load_dataset(path)
    ds = ds.assign_coords(store.drifter_fields(ds["id"].values))
    tmp = path + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, path)
    print(f"Per-drifter fields added for {ds.sizes['traj']:,} drifters; ran aground: "
          f"{int((ds['typedeath'].values == 1).sum()):,} → {path}")


def scan_all(store, box):
    """Every chunk's kept observations joined up: (arrays by name, positions dropped for their gap)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = [p for p in tqdm(pool.map(lambda k: store.scan_retrying(k, box), range(store.n_chunks)),
                                 total=store.n_chunks, desc="Scanning chunks") if p is not None]
    n_gap = sum(p["n_gap"] for p in parts)
    parts = [p for p in parts if "obs_index" in p]
    return {v: np.concatenate([p[v] for p in parts]) for v in ["obs_index"] + OBS_VARS}, n_gap


def build_subset(store, sub, box):
    """Rebuild the ragged array for the kept observations."""
    lon_min, lon_max, lat_min, lat_max = box
    traj_of_obs = np.searchsorted(store.traj_start, sub["obs_index"], side="right") - 1
    kept_traj, sub_rowsize = np.unique(traj_of_obs, return_counts=True)
    return xr.Dataset(
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
            "id": ("traj", store.traj_ids[kept_traj]),
            **store.drifter_fields(store.traj_ids[kept_traj]),
            "time": ("obs", sub["time"].astype("datetime64[s]").astype("datetime64[ns]")),
        },
        attrs={
            "source": ZARR_URL,
            "description": f"GDP hourly subset (step {STEP_S // 3600} h, gap <= {MAX_GAP_S // 3600} h), padded box",
            "lon_range": f"{lon_min}..{lon_max}",
            "lat_range": f"{lat_min}..{lat_max}",
        },
    )


def main(argv=None):
    args = parse_options(argv)
    os.makedirs("data", exist_ok=True)
    store = GdpStore()
    if args.metadata_only:
        refresh_drifter_fields(store, args.out)
        return

    sub, n_gap = scan_all(store, args.fetch_box)
    ds = build_subset(store, sub, args.fetch_box)
    ds.to_netcdf(args.out)
    print(f"\nDropped (gap > {MAX_GAP_S // 3600} h): {n_gap:,}")
    print(f"Kept trajectories : {ds.sizes['traj']:,}")
    print(f"Kept observations : {ds.sizes['obs']:,}")
    print(f"Saved → {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
