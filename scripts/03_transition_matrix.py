"""
Agulhas Region — Transition Matrix (Ulam's method)
Builds the surface-transport operator P for box R from GDP drifter pairs
separated by a lag τ, separately for drogued (15 m current, no wind) and
undrogued (surface current + wind slip) drifters, for every grid size and
lag given (defaults: 0.5°, 1° and 2°; τ = 3.5 days; see scripts/config.py).

P[i, j] = probability that material in state i is in state j after τ.
States: ocean cells of R, then five absorbing states: exited R across the W,
E, S or N edge, and stranded (the drifter ran aground inside R within τ,
from GDP's record of how each drifter ended). A cell with no outgoing moves
of its own takes the pooled moves of the ocean cells around it.
Load and iterate with scripts/transport.py.

    .venv/bin/python scripts/03_transition_matrix.py --res 1 --tau 2 3.5 5
"""

import numpy as np
import xarray as xr

from config import DROGUE_TYPES, MIN_DRIFTERS, grid_shape, operator_path, parse_args
from transport import (EXIT_LABELS, STRANDED, move_indices, propagate, release, row_normalise,
                       strand_times, to_grid, transition_states)

args = parse_args(__doc__, grid=True, lag=True, multi=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
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
strand_t = strand_times(ds, traj_idx, t, in_R)


# ── 2. Build P for each grid, lag and drogue type ────────────────────────────
def build(name, drogue_flag, res, tau, start, end, stranded):
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
    s, e, st = start[same_type], end[same_type], stranded[same_type]

    # Ocean cells: any cell in R where a drifter of either type was observed,
    # so both drogue types share the same states
    cells = np.unique(cell_flat[cell_flat >= 0])
    n_cells = len(cells)
    n_states = n_cells + len(EXIT_LABELS)
    from_state, to_state = transition_states(cell_flat, cells, s, e, lon, lat, args.box, st)

    C = np.zeros((n_states, n_states))
    np.add.at(C, (from_state, to_state), 1)

    # Distinct drifters contributing to each row
    pairs = np.unique(np.stack([from_state, traj_idx[s]]), axis=1)
    row_drifters = np.bincount(pairs[0], minlength=n_states)
    row_obs = C.sum(axis=1)

    # Ocean cells with no outgoing moves take their neighbours' (flagged below)
    P, empty = row_normalise(C, cells, n_lon)

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
    print(f"Transitions used      : {len(s):,}  ({years[s].min()}–{years[s].max()}), "
          f"{st.sum():,} into stranded from {len(np.unique(traj_idx[s[st]]))} drifters")
    print(f"Ocean cells / states  : {n_cells} / {n_states}")
    print(f"Rows sum to 1         : {np.allclose(P.sum(axis=1), 1)}")
    print(f"Cells with no moves   : {len(empty)} (rows pooled from neighbouring cells)")
    print(f"Cells < {MIN_DRIFTERS} drifters    : {flagged.sum()} ({flagged.mean():.0%})")
    exit_per_step = P[:n_cells, n_cells:n_cells + STRANDED].sum(axis=1)
    print(f"Mean exit prob / step : {exit_per_step.mean():.1%}")
    print(f"Mean strand prob/step : {P[:n_cells, n_cells + STRANDED].mean():.2%} "
          f"(cells with any stranding: {(P[:n_cells, n_cells + STRANDED] > 0).sum()})")
    print(f"Stay-in-same-cell prob: {np.diag(P)[:n_cells].mean():.1%} (mean over cells)")
    print(f"Saved → {out}")
    return {"P": P, "cell_flat": cells, "lon_edges": lon_edges,
            "lat_edges": lat_edges, "res": res, "tau_days": tau}


ops = {}
for tau in args.tau:
    start, end, stranded = move_indices(traj_idx, t, in_R, tau, strand_t)
    if (~stranded).sum() == 0:
        raise SystemExit(f"No pairs at τ = {tau} d: τ must be a multiple of the data's time step.")
    for res in args.res:
        for name, flag in DROGUE_TYPES.items():
            ops[(name, res, tau)] = build(name, flag, res, tau, start, end, stranded)

# ── 3. Sanity check: one release off Durban ──────────────────────────────────
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
            print(f"    {days:>3} d: in R {p[:n_cells].sum():5.1%} | {exits} | "
                  f"peak cell {op['lon_edges'][i:i+2].mean():.2f}°E {op['lat_edges'][j:j+2].mean():.2f}°")
