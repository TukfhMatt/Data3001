"""
Agulhas Region — OpenDrift Against the Matrix on Held-out Drifters for a Year
opendrift_drifters and opendrift_score taken to 364 days (the matrix's 104 steps), so
OpenDrift's forecast error can be set beside the matrix's over the horizon the project
reports. The test starts are held_out's undrogued drifters at 00 UTC on each day they
are in R, in the five forcing years, with the true position at every whole step of τ up
to 364 days. A run needs the following year's GLORYS12 and ERA5 files as well
(opendrift_forcing).

Each run lasts about ten times as long as a 28-day one, so a random sample of at most
N_PER_BATCH starts in each half month is used, the same for every method. OpenDrift is
seeded as in opendrift_drifters (at the drifter's position, and over its 1° cell) with
2% of the ERA5 wind. Scored as in opendrift_score: the distance from a forecast's centre
(of the material still in R) to the drifter's actual position, for starts where the
drifter is still in R at that step, with 95% intervals from resampling whole drifters.
Finished runs are skipped, so an interrupted run resumes.

    .venv-opendrift/bin/python -m agulhas opendrift_drifters_year --years 2000
"""

from datetime import datetime
import os

import numpy as np

from agulhas.comparison.opendrift_drifters import batches, fold_starts, run_batch
from agulhas.comparison.opendrift_points import MatrixStates
from agulhas.comparison.opendrift_score import SEED as SCORE_SEED
from agulhas.comparison.opendrift_score import errors, load_forecasts, median_table
from agulhas.config import parse_args, setting_tag
from agulhas.drifters import Drifters
from agulhas.opendrift_common import FORCING, OUT, YEARS
from agulhas.transport import check_coast_box
from agulhas.validation.held_out import SEED, HeldOut, folds

YEAR_DAYS = 364
N_PER_BATCH = 100
DIR = f"{OUT}/drifters_year"


def year_end(year):
    """The last time a run starting in a year may reach: the wind runs to the end of the next year."""
    return datetime(year + 1, 12, 31)


# ── 1. A sample of the test starts ───────────────────────────────────────────

def sample_starts(starts, rng):
    """At most N_PER_BATCH starts from each half month, in the order of the full set."""
    keep = np.concatenate([rng.permutation(index)[:N_PER_BATCH] for _, index in batches(starts, YEARS)])
    keep.sort()
    n = len(starts["traj"])
    return {k: (v[keep] if np.ndim(v) and len(v) == n else v) for k, v in starts.items()}


def load_starts(args):
    """The sampled test starts, built from the subset and held_out's folds the first time and read back after."""
    path = f"{DIR}/starts_{setting_tag(args.res, args.tau, args.box)}.npz"
    if not os.path.exists(path):
        ho = HeldOut(Drifters(args.data), args.box, args.res, args.tau, horizon_days=YEAR_DAYS)
        parts = [fold_starts(ho, f, np.setdiff1d(trajs, test), test, end=year_end)
                 for name, _, f, trajs, test in folds(ho, np.random.default_rng(SEED)) if name == "undrogued"]
        starts = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.argsort(starts["t0"], kind="stable")
        starts = {k: v[order] for k, v in starts.items()}
        starts.update(n_cells=ho.n_cells, tau=ho.tau, steps=np.array(ho.horizon_steps))
        total = len(order)
        starts = sample_starts(starts, np.random.default_rng(SEED))
        np.savez_compressed(path + ".tmp.npz", **starts)
        os.replace(path + ".tmp.npz", path)
        print(f"{len(starts['traj']):,} of {total:,} test starts, from {len(np.unique(starts['traj']))} drifters → {path}")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# ── 2. Score ─────────────────────────────────────────────────────────────────

def print_summary(table, n_starts):
    cols = ["days", "starts", "drifters", "median_matrix", "median_opendrift_exact", "median_opendrift_cell", "median_persistence"]
    print(table[table["days"].round().isin([7, 28, 91, 182, 273, 364])][cols].round(1).to_string(index=False))
    for m in ("opendrift_exact", "opendrift_cell"):
        print(f"{m}: beats the matrix at {(table[f'diff_{m}_lo'] > 0).sum()} of {len(table)} horizons, "
              f"the matrix beats it at {(table[f'diff_{m}_hi'] < 0).sum()}")
    print(f"{n_starts:,} sampled starts.")


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    from roaring_landmask import RoaringLandmask

    def add_options(p):
        p.add_argument("--years", type=int, nargs="+", default=YEARS, help="years to run")

    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    check_coast_box(args.box)
    os.makedirs(DIR, exist_ok=True)
    tag = setting_tag(args.res, args.tau, args.box)
    starts = load_starts(args)
    states = MatrixStates(args)
    land = RoaringLandmask.new()
    for name, index in batches(starts, args.years):
        path = f"{DIR}/{name}.npz"
        if not os.path.exists(path):
            run_batch(index, starts, land, states, args, path, next_year=True)
    print(f"Forcing read from {FORCING}; forecasts saved → {DIR}/")
    if set(args.years) == set(YEARS):
        forecast, truth, traj = load_forecasts(tag, base=DIR)
        err, _, scorable = errors(forecast, truth)
        days = np.arange(1, truth.shape[1] + 1) * args.tau
        table = median_table(err, scorable, traj, days, np.random.default_rng(SCORE_SEED))
        table.to_csv(f"{OUT}/drifters_year_score_{tag}.csv", index=False)
        print_summary(table, len(traj))


if __name__ == "__main__":
    main()
