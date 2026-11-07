"""
Agulhas Region — Transition Matrix (Ulam's method)
Builds the surface-transport operator P for box R from GDP drifter pairs
separated by a lag τ, separately for drogued (15 m current, no wind) and
undrogued (surface current + wind slip) drifters, for every grid size and
lag given (defaults: 0.5°, 1° and 2°; τ = 3.5 days; see agulhas/config.py).

P[i, j] = probability that material in state i is in state j after τ.
States: ocean cells of R, then five absorbing states: exited R across the W,
E, S or N edge, and stranded (the drifter ran aground inside R within τ,
from GDP's record of how each drifter ended). A cell with no outgoing moves
of its own takes the pooled moves of the ocean cells around it.
Load and iterate with agulhas/transport.py.

    python -m agulhas transition_matrix --res 1 --tau 2 3.5 5
"""

import numpy as np

from agulhas.config import DROGUE_TYPES, MIN_DRIFTERS, operator_path, parse_args
from agulhas.drifters import Drifters, in_box
from agulhas.grid import Grid
from agulhas.transport import (EXIT_LABELS, STRANDED, propagate, release, row_normalise, same_drogue, to_grid,
                               transition_states)

TEST_RELEASE = (31.5, -30.5)   # off Durban
TEST_DAYS = [7, 30, 365]
SOURCE = "NOAA GDP hourly v2.01 via CloudDrift, hourly, gap <= 6 h"


def build(drifters, in_R, grid, moves, flag, tau):
    """One drogue type's matrix on one grid, as the fields of its saved file. Ocean cells
    are any cell of R where a drifter of either type was seen, so both types share states."""
    start, end, stranded = moves
    traj_idx = drifters.traj_idx
    cell_flat = grid.cell_of(drifters.lon, drifters.lat, in_R)
    same_type = same_drogue(drifters.drogue, start, end, stranded, flag)
    s, e, st = start[same_type], end[same_type], stranded[same_type]

    cells = np.unique(cell_flat[cell_flat >= 0])
    n_cells = len(cells)
    n_states = n_cells + len(EXIT_LABELS)
    from_state, to_state = transition_states(cell_flat, cells, s, e, drifters.lon, drifters.lat, grid.box, st)

    C = np.zeros((n_states, n_states))
    np.add.at(C, (from_state, to_state), 1)
    pairs = np.unique(np.stack([from_state, traj_idx[s]]), axis=1)   # distinct drifters behind each row
    row_drifters = np.bincount(pairs[0], minlength=n_states)
    row_obs = C.sum(axis=1)
    P, empty = row_normalise(C, cells, grid.n_lon)                   # empty cells take their neighbours' moves
    years = drifters.year[s]
    return dict(
        P=P, C=C,
        row_drifters=row_drifters[:n_cells], row_obs=row_obs[:n_cells],
        flagged=row_drifters[:n_cells] < MIN_DRIFTERS, empty=np.isin(np.arange(n_cells), empty),
        cell_flat=cells, lon_edges=grid.lon_edges, lat_edges=grid.lat_edges,
        res=grid.res, tau_days=tau, exit_labels=EXIT_LABELS, box=np.array(grid.box), drogued=flag,
        years=np.array([years.min(), years.max()]), source=SOURCE,
    )


def print_checks(name, op, tau, n_moves, n_stranded, stranded_drifters, path):
    P, n_cells = op["P"], len(op["cell_flat"])
    flagged = op["flagged"]
    print(f"\n=== {name}, {op['res']:g}°, τ = {tau:g} d ===")
    print(f"Transitions used      : {n_moves:,}  ({op['years'][0]}–{op['years'][1]}), "
          f"{n_stranded:,} into stranded from {stranded_drifters} drifters")
    print(f"Ocean cells / states  : {n_cells} / {P.shape[0]}")
    print(f"Rows sum to 1         : {np.allclose(P.sum(axis=1), 1)}")
    print(f"Cells with no moves   : {op['empty'].sum()} (rows pooled from neighbouring cells)")
    print(f"Cells < {MIN_DRIFTERS} drifters    : {flagged.sum()} ({flagged.mean():.0%})")
    print(f"Mean exit prob / step : {P[:n_cells, n_cells:n_cells + STRANDED].sum(axis=1).mean():.1%}")
    print(f"Mean strand prob/step : {P[:n_cells, n_cells + STRANDED].mean():.2%} "
          f"(cells with any stranding: {(P[:n_cells, n_cells + STRANDED] > 0).sum()})")
    print(f"Stay-in-same-cell prob: {np.diag(P)[:n_cells].mean():.1%} (mean over cells)")
    print(f"Saved → {path}")


def build_and_save(drifters, in_R, grid, moves, name, flag, tau, box):
    op = build(drifters, in_R, grid, moves, flag, tau)
    path = operator_path(name, grid.res, tau, box)
    np.savez_compressed(path, **op)
    start, end, stranded = moves
    keep = same_drogue(drifters.drogue, start, end, stranded, flag)
    s, st = start[keep], stranded[keep]
    print_checks(name, op, tau, len(s), st.sum(), len(np.unique(drifters.traj_idx[s[st]])), path)
    return op


def print_sanity_check(ops, box):
    """Where one release off Durban goes under each matrix, when the box holds it."""
    lo, la = TEST_RELEASE
    if not in_box(lo, la, box):
        return
    print(f"\nSanity check — release at {lo}°E, {abs(la)}°S (off Durban)")
    for (name, res, tau), op in ops.items():
        n_cells = len(op["cell_flat"])
        p0 = release(op, lo, la)
        print(f"  {name}, {res:g}°, τ = {tau:g} d:")
        for days in TEST_DAYS:
            p = propagate(op, p0, days=days)
            grid = to_grid(op, p)
            j, i = np.unravel_index(np.nanargmax(grid), grid.shape)
            exits = ", ".join(f"{l} {v:.0%}" for l, v in zip(EXIT_LABELS, p[n_cells:]))
            print(f"    {days:>3} d: in R {p[:n_cells].sum():5.1%} | {exits} | "
                  f"peak cell {op['lon_edges'][i:i+2].mean():.2f}°E {op['lat_edges'][j:j+2].mean():.2f}°")


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, multi=True, argv=argv)
    drifters = Drifters(args.data)
    in_R = in_box(drifters.lon, drifters.lat, args.box)
    strand_t = drifters.strand_times(in_R)

    ops = {}
    for tau in args.tau:
        moves = drifters.moves(in_R, tau, strand_t)
        if (~moves[2]).sum() == 0:
            raise SystemExit(f"No pairs at τ = {tau} d: τ must be a multiple of the data's time step.")
        for res in args.res:
            grid = Grid(args.box, res)
            for name, flag in DROGUE_TYPES.items():
                ops[(name, res, tau)] = build_and_save(drifters, in_R, grid, moves, name, flag, tau, args.box)
    print_sanity_check(ops, args.box)


if __name__ == "__main__":
    main()
