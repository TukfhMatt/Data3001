"""
Agulhas Region — Transition Matrix (Ulam's method)
Builds the surface-transport operator P for box R from GDP drifter pairs
separated by a lag τ, separately for drogued (15 m current, no wind) and
undrogued (surface current + wind slip) drifters, for every grid size and
lag given (defaults: 0.5°, 1° and 2°; τ = 3.5 days; see scripts/config.py).

P[i, j] = probability that material in state i is in state j after τ.
States: ocean cells of R, then four absorbing "exited R" states (W, E, S, N).
Load and iterate with scripts/transport.py.

    .venv/bin/python scripts/03_transition_matrix.py --res 1 --tau 2 3.5 5
"""

import numpy as np
import xarray as xr

from config import DROGUE_TYPES, MIN_DRIFTERS, grid_shape, operator_path, parse_args
from transport import propagate, release, to_grid

args = parse_args(__doc__, grid=True, lag=True, multi=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
EXIT_LABELS = np.array(["W", "E", "S", "N"])
TEST_RELEASE = (31.5, -30.5)   # off Durban

# ── 1. Load subset ───────────────────────────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
years = ds["time"].values.astype("datetime64[Y]").astype(int) + 1970
in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)


# ── 2. Pair each obs with the same drifter τ later ───────────────────────────
def pairs_at(tau_days):
    """Indices (start, end) of obs pairs from one drifter exactly τ apart.
    Data is sorted by (trajectory, time), so a combined key can be binary-searched."""
    tau_s = int(round(tau_days * 86400))
    key = traj_idx.astype(np.int64) * 10**10 + t
    target = key + tau_s
    j = np.minimum(np.searchsorted(key, target), len(key) - 1)
    has_pair = in_R & (key[j] == target)
    return np.where(has_pair)[0], j[has_pair]


def exit_state(lon_e, lat_e):
    """Which side of R a point outside R lies on (largest excursion wins)."""
    excursion = np.stack([LON_MIN - lon_e, lon_e - LON_MAX,
                          LAT_MIN - lat_e, lat_e - LAT_MAX])
    return np.argmax(excursion, axis=0)


# ── 3. Build P for each grid, lag and drogue type ────────────────────────────
def build(name, drogue_flag, res, tau, start, end):
    # If a side is not a multiple of res, the last row/column is a partial cell
    n_lon, n_lat = grid_shape(args.box, res)
    lon_edges = np.minimum(LON_MIN + np.arange(n_lon + 1) * res, LON_MAX)
    lat_edges = np.minimum(LAT_MIN + np.arange(n_lat + 1) * res, LAT_MAX)
    cell_flat = np.where(
        in_R,
        ((lat - LAT_MIN) // res).astype(int) * n_lon + ((lon - LON_MIN) // res).astype(int),
        -1,
    )

    same_type = (drogue[start] == drogue_flag) & (drogue[end] == drogue_flag)
    s, e = start[same_type], end[same_type]

    # Ocean cells: any cell in R where a drifter of either type was observed,
    # so both drogue types share the same states
    cells = np.unique(cell_flat[cell_flat >= 0])
    n_cells = len(cells)
    n_states = n_cells + len(EXIT_LABELS)
    state_of_cell = np.full(n_lat * n_lon, -1)
    state_of_cell[cells] = np.arange(n_cells)

    from_state = state_of_cell[cell_flat[s]]
    to_state = np.where(
        cell_flat[e] >= 0,
        state_of_cell[np.maximum(cell_flat[e], 0)],
        n_cells + exit_state(lon[e], lat[e]),
    )

    C = np.zeros((n_states, n_states))
    np.add.at(C, (from_state, to_state), 1)

    # Distinct drifters contributing to each row
    pairs = np.unique(np.stack([from_state, traj_idx[s]]), axis=1)
    row_drifters = np.bincount(pairs[0], minlength=n_states)
    row_obs = C.sum(axis=1)

    P = np.zeros_like(C)
    ok = row_obs > 0
    P[ok] = C[ok] / row_obs[ok, None]
    # Ocean cells with no outgoing data: hold mass in place (flagged below)
    empty = np.where(~ok[:n_cells])[0]
    P[empty, empty] = 1.0
    # Exit states are absorbing
    P[n_cells:, n_cells:] = np.eye(len(EXIT_LABELS))

    flagged = row_drifters[:n_cells] < MIN_DRIFTERS
    out = operator_path(name, res, tau, args.box)
    np.savez_compressed(
        out,
        P=P, C=C,
        row_drifters=row_drifters[:n_cells], row_obs=row_obs[:n_cells],
        flagged=flagged, empty=np.isin(np.arange(n_cells), empty),
        cell_flat=cells, lon_edges=lon_edges, lat_edges=lat_edges,
        res=res, tau_days=tau, exit_labels=EXIT_LABELS, box=np.array(args.box),
        drogued=drogue_flag,
        years=np.array([years[s].min(), years[s].max()]),
        source="NOAA GDP hourly v2.01 via CloudDrift, hourly, gap <= 6 h",
    )

    # ── Checks ──
    print(f"\n=== {name}, {res:g}°, τ = {tau:g} d ===")
    print(f"Transitions used      : {len(s):,}  ({years[s].min()}–{years[s].max()})")
    print(f"Ocean cells / states  : {n_cells} / {n_states}")
    print(f"Rows sum to 1         : {np.allclose(P.sum(axis=1), 1)}")
    print(f"Cells with no data    : {len(empty)}")
    print(f"Cells < {MIN_DRIFTERS} drifters    : {flagged.sum()} ({flagged.mean():.0%})")
    exit_per_step = P[:n_cells, n_cells:].sum(axis=1)
    print(f"Mean exit prob / step : {exit_per_step.mean():.1%}")
    print(f"Stay-in-same-cell prob: {np.diag(P)[:n_cells].mean():.1%} (mean over cells)")
    print(f"Saved → {out}")
    return {"P": P, "cell_flat": cells, "lon_edges": lon_edges,
            "lat_edges": lat_edges, "res": res, "tau_days": tau}


ops = {}
for tau in args.tau:
    start, end = pairs_at(tau)
    if len(start) == 0:
        raise SystemExit(f"No pairs at τ = {tau} d: τ must be a multiple of the data's time step.")
    for res in args.res:
        for name, flag in DROGUE_TYPES.items():
            ops[(name, res, tau)] = build(name, flag, res, tau, start, end)

# ── 4. Sanity check: one release off Durban ──────────────────────────────────
lo, la = TEST_RELEASE
if LON_MIN <= lo < LON_MAX and LAT_MIN <= la < LAT_MAX:
    print(f"\nSanity check — release at {lo}°E, {abs(la)}°S (off Durban)")
    for (name, res, tau), op in ops.items():
        n_cells = len(op["cell_flat"])
        p0 = release(op, lo, la)
        print(f"  {name}, {res:g}°, τ = {tau:g} d:")
        for days in [7, 30, 365]:
            p = propagate(op, p0, days=days)
            grid = to_grid(op, p)
            j, i = np.unravel_index(np.nanargmax(grid), grid.shape)
            exits = ", ".join(f"{l} {v:.0%}" for l, v in zip(EXIT_LABELS, p[n_cells:]))
            print(f"    {days:>3} d: in R {p[:n_cells].sum():5.1%} | exited {exits} | "
                  f"peak cell {op['lon_edges'][i:i+2].mean():.2f}°E {op['lat_edges'][j:j+2].mean():.2f}°")
