"""
Agulhas Region — OpenDrift Runs for a Year
The same runs as opendrift_points, lasting 364 days (the matrix's 104 steps), so the
oil matrix can be set against OpenDrift over the horizon the project reports. One
OceanDrift run for each start date, 500 particles per release point over the ocean part
of its 1° cell, oil at 3.5% of the ERA5 wind with no weathering. A run needs the
following year's GLORYS12 and ERA5 files as well (opendrift_forcing). Particles are
sorted into the matrix's states every 7 days to day 364, and the counts are saved for
comparison. A year of hourly output for 6,500 particles takes a few GB of memory, so
run one at a time.

Start dates are the 1st of --months in each of --years (January and July in the five
years by default, 10 starts), since each run lasts about ten times as long as a
31.5-day one. Finished starts are skipped, so an interrupted run resumes.

    .venv-opendrift/bin/python -m agulhas opendrift_year --years 2000 --months 1
"""

from datetime import datetime
import os
import time

import numpy as np
import pandas as pd

from agulhas.comparison.opendrift_points import MatrixStates, seed_points
from agulhas.config import parse_args, release_points_path
from agulhas.opendrift_common import FORCING, OIL_WIND, OUT, YEARS, simulate
from agulhas.transport import check_coast_box

YEAR_HOURS = {day: day * 24 for day in range(7, 365, 7)}     # every week to 364 days
FOLDER = f"{OUT}/points_year"


# ── 1. One run for each start date ───────────────────────────────────────────

def run_start(year, month, points, land, states, res, path):
    """One year-long OceanDrift run from the 1st of the month, with its counts saved to path."""
    began = time.time()
    seed = year * 100 + month
    x, y, marks = seed_points(land, points, res, np.random.default_rng(seed))
    lon, lat, marker, stranded, last = simulate(datetime(year, month, 1), x, y, marks, max(YEAR_HOURS.values()), seed,
                                                wind=OIL_WIND, next_year=True)
    counts = states.counts(lon, lat, marker, stranded, last, len(points), YEAR_HOURS)
    np.savez_compressed(path + ".tmp.npz", counts=counts)
    os.replace(path + ".tmp.npz", path)
    print(f"{year}-{month:02d}: {time.time() - began:.0f} s, no matching state {counts[:, :, -1].sum()}", flush=True)


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    from roaring_landmask import RoaringLandmask

    def add_options(p):
        p.add_argument("--years", type=int, nargs="+", default=YEARS, help="start years to run")
        p.add_argument("--months", type=int, nargs="+", default=[1, 7], help="start months to run")

    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    check_coast_box(args.box)
    os.makedirs(FOLDER, exist_ok=True)
    points = pd.read_csv(release_points_path(args.box))
    states = MatrixStates(args)
    land = RoaringLandmask.new()
    for year in args.years:
        for month in args.months:
            path = f"{FOLDER}/{year}{month:02d}.npz"
            if not os.path.exists(path):
                run_start(year, month, points, land, states, args.res, path)
    print(f"Forcing read from {FORCING}; counts saved → {FOLDER}/")


if __name__ == "__main__":
    main()
