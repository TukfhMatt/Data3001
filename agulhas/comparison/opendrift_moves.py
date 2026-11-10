"""
Agulhas Region — OpenDrift Moves Between the Matrix's States
For opendrift_checks, which were added after opendrift_compare's fixed rule gave
its result. Repeats opendrift_points' runs with the same particles and seeds, and
counts every particle's moves between the oil coast matrix's states over one lag
τ, from every hour of the run, as transition_matrix counts the drifters' moves.
A matrix built from them has OpenDrift's physics and the matrix's method, so it
separates the method's own error from the difference in data.

Each run's states at day 7 and day 31.5 must give opendrift_points' saved counts;
the step stops if they do not. Finished starts are skipped, so an interrupted run
resumes, and runs for different years can go side by side with --years.

    .venv-opendrift/bin/python -m agulhas opendrift_moves --years 2000 2005
"""

from datetime import datetime
import os
import time

import numpy as np
import pandas as pd

from agulhas.comparison.opendrift_points import MatrixStates, seed_points
from agulhas.config import parse_args, release_points_path
from agulhas.opendrift_common import FORCING, HOURS, N_COASTS, OUT, YEARS, hourly_states, simulate
from agulhas.transport import check_coast_box


# ── 1. Moves over one lag, from every hour ───────────────────────────────────

def count_moves(state, n_cells, lag):
    """Moves from an ocean cell to any state lag hours later, (n_cells, n_cells + 4 + coasts),
    and the number of particles behind each row. Moves to a position with no state are left out."""
    n_to = n_cells + 4 + N_COASTS
    fr, to = state[:, :-lag], state[:, lag:]
    keep = (fr < n_cells) & (to < n_to)
    moves = np.bincount(fr[keep] * n_to + to[keep], minlength=n_cells * n_to).reshape(n_cells, n_to)
    particle = np.broadcast_to(np.arange(len(state))[:, None], fr.shape)[keep]
    pairs = np.unique(particle.astype(np.int64) * n_cells + fr[keep])
    return moves.astype(np.int32), np.bincount(pairs % n_cells, minlength=n_cells)


def check_counts(state, marker, states, n_points, saved):
    """Stop if the states at day 7 and day 31.5 do not give opendrift_points' counts."""
    for h, hour in enumerate(HOURS.values()):
        for k in range(n_points):
            if not (np.bincount(state[marker == k, hour], minlength=states.n_states) == saved[k, h]).all():
                raise SystemExit(f"Release point {k} at hour {hour} differs from opendrift_points' run.")


# ── 2. One run for each start date ───────────────────────────────────────────

def run_start(year, month, points, land, states, args, path):
    """One OceanDrift run from the 1st of the month, as opendrift_points runs it, with its moves saved to path."""
    began = time.time()
    seed = year * 100 + month
    x, y, marks = seed_points(land, points, args.res, np.random.default_rng(seed))
    lon, lat, marker, stranded, last = simulate(datetime(year, month, 1), x, y, marks, max(HOURS.values()), seed)
    state = hourly_states(lon, lat, stranded, last, states.cell_state, states.n_cells, states.box, states.n_lon)
    check_counts(state, marker, states, len(points), np.load(f"{OUT}/points/{year}{month:02d}.npz")["counts"])
    moves, particles = count_moves(state, states.n_cells, int(round(args.tau * 24)))
    np.savez_compressed(path + ".tmp.npz", moves=moves, particles=particles)
    os.replace(path + ".tmp.npz", path)
    print(f"{year}-{month:02d}: {time.time() - began:.0f} s, {moves.sum()} moves", flush=True)


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    from roaring_landmask import RoaringLandmask

    def add_options(p):
        p.add_argument("--years", type=int, nargs="+", default=YEARS, help="years to run")

    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    check_coast_box(args.box)
    os.makedirs(f"{OUT}/moves", exist_ok=True)
    points = pd.read_csv(release_points_path(args.box))
    states = MatrixStates(args)
    land = RoaringLandmask.new()
    for year in args.years:
        for month in range(1, 13):
            path = f"{OUT}/moves/{year}{month:02d}.npz"
            if not os.path.exists(path):
                run_start(year, month, points, land, states, args, path)
    print(f"Forcing read from {FORCING}; moves saved → {OUT}/moves/")


if __name__ == "__main__":
    main()
