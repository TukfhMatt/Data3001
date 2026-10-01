"""
Helpers for using a saved Agulhas transport operator.

    from transport import load_operator, release, propagate, to_grid

    op = load_operator("data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz")
    p0 = release(op, lon=31.5, lat=-30.5)   # all mass in one cell
    p  = propagate(op, p0, days=30)         # distribution after 30 days
    grid = to_grid(op, p)                   # (n_lat, n_lon) map, NaN off-ocean

States 0..n_cells-1 are ocean grid cells in R; the last five are absorbing,
in the order of op["exit_labels"]: left R across the W, E, S or N edge, and
"stranded" (ran aground on a coast inside R).

The building helpers at the end (strand_times, move_indices,
transition_states, row_normalise) are shared by every script that builds a
matrix (03, 06, 09, 10), so all builds match 03 exactly.
"""

import numpy as np

EXIT_LABELS = np.array(["W", "E", "S", "N", "stranded"])
STRANDED = len(EXIT_LABELS) - 1          # offset of the stranded state after the ocean cells
STRAND_MATCH_S = 86400                   # last subset obs within this of GDP's end date


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
    grid (near-shore cells, used for coastal exposure)."""
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
def last_obs_index(traj_idx):
    """Per obs, the index of its drifter's last observation in the subset."""
    last = np.r_[np.where(np.diff(traj_idx) != 0)[0], len(traj_idx) - 1]
    return np.repeat(last, np.diff(np.r_[-1, last]))


def strand_times(ds, traj_idx, t, in_box):
    """Per obs, the time (s) its drifter ran aground inside the box, or -1.
    A drifter strands when GDP records its death as "ran aground"
    (typedeath 1) and its last observation in the subset lies inside the box
    and within STRAND_MATCH_S of GDP's end date, so the subset holds the true
    end of the record; that last observation gives the time and place."""
    last = last_obs_index(traj_idx)
    end_s = ds["end_date"].values.astype("datetime64[s]").astype(np.int64)[traj_idx]
    aground = ((ds["typedeath"].values == 1)[traj_idx] & in_box[last]
               & (np.abs(t[last] - end_s) <= STRAND_MATCH_S))
    return np.where(aground, t[last], -1)


def move_indices(traj_idx, t, in_box, tau_days, strand_t):
    """All one-step moves starting inside the box: (start, end, stranded).
    Ordinary moves pair obs from one drifter exactly τ apart. A start whose
    drifter runs aground inside the box less than τ later has no such pair;
    it moves to the stranded state, with end = the drifter's last obs (so
    drogue-type and drifter filters apply to both kinds alike)."""
    s1, e1 = pair_indices(traj_idx, t, in_box, tau_days)
    tau_s = int(round(tau_days * 86400))
    s2 = np.where(in_box & (strand_t >= 0) & (t <= strand_t) & (t > strand_t - tau_s))[0]
    s2 = np.setdiff1d(s2, s1, assume_unique=True)
    e2 = last_obs_index(traj_idx)[s2]
    order = np.argsort(np.r_[s1, s2], kind="stable")
    return (np.r_[s1, s2][order], np.r_[e1, e2][order],
            np.r_[np.zeros(len(s1), bool), np.ones(len(s2), bool)][order])


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


def transition_states(cell_flat, cells, start, end, lon, lat, box, stranded=None):
    """(from_state, to_state) for each move. cell_flat is the grid index of
    every obs (-1 outside the box) and cells the ocean cells that become
    states; an end outside the box goes to the exit state on the side it
    left by (largest excursion wins), and moves marked `stranded` go to the
    stranded state."""
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
    if stranded is not None:
        to_state = np.where(stranded, n_cells + STRANDED, to_state)
    return from_state, to_state


def row_normalise(C, cells, n_lon, max_ring=3):
    """Row-stochastic P from counts C. An ocean cell with no outgoing moves of
    its own (its drifters' records ended there for reasons other than running
    aground) takes the pooled counts of the nearest ring of ocean cells
    around it that has data, up to max_ring cells away; beyond that it holds
    mass in place. Absorbing states stay put. Returns (P, empty ocean-state
    indices)."""
    n_cells = len(cells)
    row_obs = C.sum(axis=1)
    P = np.zeros_like(C)
    ok = row_obs > 0
    P[ok] = C[ok] / row_obs[ok, None]
    empty = np.where(~ok[:n_cells])[0]
    j, i = np.divmod(np.asarray(cells), n_lon)
    has_data = ok[:n_cells]
    for st in empty:
        for r in range(1, max_ring + 1):
            nb = np.where(has_data & (np.abs(j - j[st]) <= r) & (np.abs(i - i[st]) <= r))[0]
            if nb.size:
                pooled = C[nb].sum(axis=0)
                P[st] = pooled / pooled.sum()
                break
        else:
            P[st, st] = 1.0
    P[n_cells:, n_cells:] = np.eye(C.shape[0] - n_cells)
    return P, empty
