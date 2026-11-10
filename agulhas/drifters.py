"""
The GDP drifter subset: one object that loads each array when a step first asks
for it, so a step holds only the arrays it uses.

    drifters = Drifters(args.data)
    in_R = in_box(drifters.lon, drifters.lat, args.box)
    start, end, stranded = drifters.moves(in_R, tau)

Observations are sorted by (trajectory, time), and every time is on the hour.
"""

from functools import cached_property
import hashlib

import numpy as np
import xarray as xr

from agulhas.transport import move_indices, strand_times


def in_box(lon, lat, box):
    """Boolean per position: inside the box, with half-open edges [min, max)."""
    lon_min, lon_max, lat_min, lat_max = box
    return (lon >= lon_min) & (lon < lon_max) & (lat >= lat_min) & (lat < lat_max)


class Drifters:
    """Arrays of one subset file. Positions and velocities are float64; `ds` is
    the dataset itself, for the per-drifter fields."""

    def __init__(self, path):
        self.path = path
        self.ds = xr.open_dataset(path)

    @cached_property
    def fingerprint(self):
        """SHA-256 of the positions and times as stored: a wind file fits only the subset it was built for."""
        h = hashlib.sha256()
        for name in ("lon", "lat"):
            h.update(np.ascontiguousarray(self.ds[name].values).tobytes())
        h.update(self.ds["time"].values.astype("datetime64[s]").astype(np.int64).tobytes())
        return h.hexdigest()

    @cached_property
    def lon(self):
        return self.ds["lon"].values.astype(float)

    @cached_property
    def lat(self):
        return self.ds["lat"].values.astype(float)

    @cached_property
    def ve(self):
        return self.ds["ve"].values.astype(float)

    @cached_property
    def vn(self):
        return self.ds["vn"].values.astype(float)

    @cached_property
    def drogue(self):
        return self.ds["drogue_status"].values.astype(bool)

    @cached_property
    def t(self):
        """Time of each observation, in seconds."""
        return self.ds["time"].values.astype("datetime64[s]").astype(np.int64)

    @cached_property
    def year(self):
        return self.ds["time"].values.astype("datetime64[Y]").astype(int) + 1970

    @cached_property
    def month(self):
        """Calendar month of each observation, 1–12."""
        return self.ds["time"].values.astype("datetime64[M]").astype(int) % 12 + 1

    @cached_property
    def traj_idx(self):
        """Per observation, the number of its drifter in this subset."""
        return np.repeat(np.arange(self.ds.sizes["traj"]), self.ds["rowsize"].values)

    def strand_times(self, in_R):
        """Per observation, when its drifter ran aground inside the box, or -1."""
        return strand_times(self.ds, self.traj_idx, self.t, in_R)

    def moves(self, in_R, tau, strand_t=None):
        """All one-step moves starting inside the box: (start, end, stranded)."""
        if strand_t is None:
            strand_t = self.strand_times(in_R)
        return move_indices(self.traj_idx, self.t, in_R, tau, strand_t)
