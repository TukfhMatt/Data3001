"""
Helpers for using a saved Agulhas transport operator.

    from transport import load_operator, release, propagate, to_grid

    op = load_operator("data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz")
    p0 = release(op, lon=31.5, lat=-30.5)   # all mass in one cell
    p  = propagate(op, p0, days=30)         # distribution after 30 days
    grid = to_grid(op, p)                   # (n_lat, n_lon) map, NaN off-ocean

States 0..n_cells-1 are ocean grid cells in R; the last four are absorbing
"exited R" states in the order of op["exit_labels"] (W, E, S, N).

The building helpers at the end (pair_indices, transition_states,
row_normalise) are shared by 03, which builds the matrices, and 06, which
rebuilds them from resampled drifters.
"""

import numpy as np

EXIT_LABELS = np.array(["W", "E", "S", "N"])


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


def coastal_states(op):
    """Boolean per ocean state: the cell has a non-ocean neighbour inside the
    grid (the matrix has no beaching state, so these stand in for the coast)."""
    n_lon = len(op["lon_edges"]) - 1
    n_lat = len(op["lat_edges"]) - 1
    ocean = np.zeros(n_lat * n_lon, bool)
    ocean[op["cell_flat"]] = True
    ocean = ocean.reshape(n_lat, n_lon)
    pad = np.pad(~ocean, 1, constant_values=False)
    land_nb = np.zeros_like(ocean)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            land_nb |= pad[1 + dj: 1 + dj + n_lat, 1 + di: 1 + di + n_lon]
    return (ocean & land_nb).ravel()[op["cell_flat"]]


# ── Building helpers ─────────────────────────────────────────────────────────
def pair_indices(traj_idx, t, in_box, tau_days):
    """Indices (start, end) of obs pairs from one drifter exactly τ apart, with
    the start inside the box. Data must be sorted by (trajectory, time), so a
    combined key can be binary-searched; t is in seconds."""
    tau_s = int(round(tau_days * 86400))
    key = traj_idx.astype(np.int64) * 10**10 + t
    target = key + tau_s
    j = np.minimum(np.searchsorted(key, target), len(key) - 1)
    has_pair = in_box & (key[j] == target)
    return np.where(has_pair)[0], j[has_pair]


def transition_states(cell_flat, cells, start, end, lon, lat, box):
    """(from_state, to_state) for each pair. cell_flat is the grid index of
    every obs (-1 outside the box) and cells the ocean cells that become
    states; an end outside the box goes to the exit state on the side it
    left by (largest excursion wins)."""
    lon_min, lon_max, lat_min, lat_max = box
    n_cells = len(cells)
    state_of_cell = np.full(cell_flat.max() + 1, -1)
    state_of_cell[cells] = np.arange(n_cells)
    le, la = lon[end], lat[end]
    side = np.argmax(np.stack([lon_min - le, le - lon_max, lat_min - la, la - lat_max]), axis=0)
    from_state = state_of_cell[cell_flat[start]]
    to_state = np.where(cell_flat[end] >= 0,
                        state_of_cell[np.maximum(cell_flat[end], 0)],
                        n_cells + side)
    return from_state, to_state


def row_normalise(C, n_cells):
    """Row-stochastic P from counts C. Ocean rows with no data hold mass in
    place; exit states are absorbing. Returns (P, empty ocean-state indices)."""
    row_obs = C.sum(axis=1)
    P = np.zeros_like(C)
    ok = row_obs > 0
    P[ok] = C[ok] / row_obs[ok, None]
    empty = np.where(~ok[:n_cells])[0]
    P[empty, empty] = 1.0
    P[n_cells:, n_cells:] = np.eye(C.shape[0] - n_cells)
    return P, empty
