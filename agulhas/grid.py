"""The grid of a box: its cell edges, and the cell each position falls in."""

import numpy as np

from agulhas.config import grid_shape
from agulhas.drifters import in_box


class Grid:
    """Cells of size res from the box's south-west corner. Cell index is
    j * n_lon + i, with half-open cells; a side that is not a multiple of res ends
    in a partial cell clipped at the box edge."""

    def __init__(self, box, res):
        self.box = tuple(box)
        self.res = res
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = box
        self.n_lon, self.n_lat = grid_shape(box, res)
        self.n_cells = self.n_lon * self.n_lat

    @property
    def lon_edges(self):
        return np.minimum(self.lon_min + np.arange(self.n_lon + 1) * self.res, self.lon_max)

    @property
    def lat_edges(self):
        return np.minimum(self.lat_min + np.arange(self.n_lat + 1) * self.res, self.lat_max)

    @property
    def lon_centres(self):
        """Centre of each column; a partial last column is centred inside the box."""
        e = self.lon_edges
        return (e[:-1] + e[1:]) / 2

    @property
    def lat_centres(self):
        e = self.lat_edges
        return (e[:-1] + e[1:]) / 2

    def cell_in_box(self, lon, lat):
        """Cell index of positions known to be inside the box."""
        return (((lat - self.lat_min) // self.res).astype(int) * self.n_lon
                + ((lon - self.lon_min) // self.res).astype(int))

    def cell_of(self, lon, lat, inside=None):
        """Cell index of every position, -1 outside the box. `inside` saves
        recomputing the box test when the step already holds it."""
        if inside is None:
            inside = in_box(lon, lat, self.box)
        return np.where(inside, self.cell_in_box(lon, lat), -1)
