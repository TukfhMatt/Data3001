"""
Agulhas Region — OpenDrift Runs at the Release Points
Compares the oil matrix with a particle model driven by each date's real currents
and winds. One OceanDrift run for each start date (the 1st of each month in five
years, 60 starts), with 500 particles per release point spread over the ocean part
of its 1° cell, the same information the matrix has. Oil moves with the GLORYS12
surface current plus 3.5% of the ERA5 wind, with no weathering. Particles are sorted
into the matrix's states at day 7 and day 31.5 (steps 2 and 9), and
opendrift_compare reads the counts.

Finished starts are skipped, so an interrupted run resumes. Runs for different
years can go side by side with --years.

    .venv-opendrift/bin/python -m agulhas opendrift_points --years 2000 2005
"""

from datetime import datetime
import os
import time

import numpy as np
import pandas as pd

from agulhas.opendrift_common import FORCING, HOURS, N_COASTS, OUT, YEARS, classify, simulate
from agulhas.config import grid_shape, operator_path, parse_args, release_points_path
from agulhas.transport import check_coast_box, load_operator

N_PER_POINT = 500


def seed_cell(land, lon, lat, res, n, rng):
    """Uniform over the ocean part of the cell centred on (lon, lat)."""
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(lon - res / 2, lon + res / 2, 4 * n)
        y = rng.uniform(lat - res / 2, lat + res / 2, 4 * n)
        keep = ~land.contains_many(x, y)
        xs.extend(x[keep].tolist())
        ys.extend(y[keep].tolist())
    return np.array(xs[:n]), np.array(ys[:n])


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    from roaring_landmask import RoaringLandmask

    def add_options(p):
        p.add_argument("--years", type=int, nargs="+", default=YEARS, help="years to run")

    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    check_coast_box(args.box)
    os.makedirs(f"{OUT}/points", exist_ok=True)
    points = pd.read_csv(release_points_path(args.box))
    op = load_operator(operator_path("oil_coast", args.res, args.tau, args.box))
    cells = op["cell_flat"].astype(int)
    n_cells = len(cells)
    n_states = n_cells + 4 + N_COASTS + 1
    cell_state = {int(f): s for s, f in enumerate(cells)}
    n_lon, _ = grid_shape(args.box, args.res)
    land = RoaringLandmask.new()

    # ── 1. One run for each start date ───────────────────────────────────────
    for year in args.years:
        for month in range(1, 13):
            path = f"{OUT}/points/{year}{month:02d}.npz"
            if os.path.exists(path):
                continue
            began = time.time()
            seed = year * 100 + month
            rng = np.random.default_rng(seed)
            xs, ys, marks = [], [], []
            for k, p in points.iterrows():
                x, y = seed_cell(land, p.lon, p.lat, args.res, N_PER_POINT, rng)
                xs.append(x)
                ys.append(y)
                marks.append(np.full(N_PER_POINT, k))
            lon, lat, marker, stranded, last = simulate(datetime(year, month, 1), np.concatenate(xs), np.concatenate(ys),
                                                        np.concatenate(marks), max(HOURS.values()), seed)

            # ── 2. Counts in each state, at each horizon ─────────────────────
            counts = np.zeros((len(points), len(HOURS), n_states), int)
            for h, hour in enumerate(HOURS.values()):
                state = classify(lon, lat, stranded, last, hour, cell_state, n_cells, args.box, n_lon)
                for k in range(len(points)):
                    counts[k, h] = np.bincount(state[marker == k], minlength=n_states)
            np.savez_compressed(path + ".tmp.npz", counts=counts)
            os.replace(path + ".tmp.npz", path)
            print(f"{year}-{month:02d}: {time.time() - began:.0f} s, no matching state {counts[:, :, -1].sum()}", flush=True)
    print(f"Forcing read from {FORCING}; counts saved → {OUT}/points/")


if __name__ == "__main__":
    main()
