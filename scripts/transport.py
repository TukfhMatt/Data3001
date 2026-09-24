"""
Helpers for using a saved Agulhas transport operator.

    from transport import load_operator, release, propagate, to_grid

    op = load_operator("data/P_drogued_1deg_3p5d.npz")
    p0 = release(op, lon=31.5, lat=-30.5)   # all mass in one cell
    p  = propagate(op, p0, days=30)         # distribution after 30 days
    grid = to_grid(op, p)                   # (n_lat, n_lon) map, NaN off-ocean

States 0..n_cells-1 are ocean grid cells in R; the last four are absorbing
"exited R" states in the order of op["exit_labels"] (W, E, S, N).
"""

import numpy as np


def load_operator(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def state_of(op, lon, lat):
    """State index for a lon/lat inside R (raises if not an ocean cell)."""
    res = float(op["res"])
    i = int((lon - op["lon_edges"][0]) // res)
    j = int((lat - op["lat_edges"][0]) // res)
    n_lon = len(op["lon_edges"]) - 1
    match = np.where(op["cell_flat"] == j * n_lon + i)[0]
    if match.size == 0:
        raise ValueError(f"({lon}, {lat}) is not an ocean cell of this operator")
    return int(match[0])


def release(op, lon, lat):
    """Initial distribution with all mass in the cell containing (lon, lat)."""
    p0 = np.zeros(op["P"].shape[0])
    p0[state_of(op, lon, lat)] = 1.0
    return p0


def propagate(op, p0, days=None, steps=None):
    """Push distribution p0 forward by `steps` (or the nearest whole number of
    steps to `days`). Returns the distribution as a row vector over states."""
    if steps is None:
        steps = int(round(days / float(op["tau_days"])))
    return p0 @ np.linalg.matrix_power(op["P"], steps)


def to_grid(op, p):
    """Map the ocean-cell part of a distribution back onto the lon/lat grid."""
    n_lon = len(op["lon_edges"]) - 1
    n_lat = len(op["lat_edges"]) - 1
    grid = np.full(n_lat * n_lon, np.nan)
    grid[op["cell_flat"]] = p[: len(op["cell_flat"])]
    return grid.reshape(n_lat, n_lon)
