"""
Agulhas Region — OpenDrift Runs From Held-out Drifters
Part 3 of the OpenDrift comparison: OpenDrift and the matrix forecast the same
real drifters, so each is scored against where the drifter actually went.

The test starts are held_out's: the undrogued drifters of its folds, at 00 UTC
on each day they are in R, with the true outcome at every whole step of τ up to
held_out.HORIZON_DAYS. Only starts in the five forcing years whose horizon ends
before the forcing does are kept. For each start the step saves the fold
matrix's forecast (built without that drifter, as held_out builds it): its centre
of mass given the material is still in R, and the shares in R, exited and stranded.

OpenDrift moves each start with the GLORYS12 surface current plus 2% of the ERA5
wind, the undrogued drifters' measured windage, from two seedings: N_EXACT
particles at the drifter's position, and N_CELL spread over the ocean part of
its 1° cell, the information the matrix has. One run per half month holds every
start in it, each particle released at its start and stopped after the horizon.
Finished runs are skipped, so an interrupted run resumes.

    .venv-opendrift/bin/python -m agulhas opendrift_drifters --years 2000 2005

--diffusivity 1 or 50 repeats the runs at that horizontal diffusivity (m²/s) into drifters_d{value}.
"""

from datetime import datetime, timedelta
import os
import time

import numpy as np

from agulhas.comparison.opendrift_points import MatrixStates, seed_cell
from agulhas.config import DROGUE_TYPES, parse_args, setting_tag
from agulhas.drifters import Drifters
from agulhas.opendrift_common import DIFFUSIVITY, FORCING, N_COASTS, OUT, YEARS, hourly_states, simulate
from agulhas.transport import STRANDED, check_coast_box
from agulhas.validation.held_out import SEED, HeldOut, folds

WIND = 0.02                 # undrogued drifters' windage (oil_wind), with no oil on top
N_EXACT = 10
N_CELL = 20
DIR = f"{OUT}/drifters"


def folder(diffusivity=DIFFUSIVITY):
    """Where the forecasts at a horizontal diffusivity are kept; the starts are shared."""
    return DIR if diffusivity == DIFFUSIVITY else f"{DIR}_d{diffusivity:g}"


def forcing_end(year):
    """The last time a run may reach: the forcing runs to 3 January of the next year."""
    return datetime(year + 1, 1, 2)


# ── 1. Test starts and the fold matrix's forecasts ───────────────────────────

def matrix_forecast(ho, P, start_state):
    """The fold matrix's forecast at each scored step: centre of mass given in R (starts, steps, 2),
    and the shares in R, exited and stranded (starts, steps)."""
    n, K = len(start_state), len(ho.horizon_steps)
    com, shares = np.empty((n, K, 2)), np.empty((3, n, K))
    Pk = np.eye(ho.n_states)
    for i, _ in enumerate(ho.horizon_steps):
        Pk = Pk @ P
        pred = Pk[start_state]
        mass = pred[:, :ho.n_cells]
        w = mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)
        com[:, i] = np.c_[w @ ho.cell_lon, w @ ho.cell_lat]
        shares[:, :, i] = [mass.sum(axis=1), pred[:, ho.n_cells:ho.n_cells + STRANDED].sum(axis=1), pred[:, ho.n_cells + STRANDED]]
    return com, shares


def fold_starts(ho, f, train_traj, test_traj):
    """One fold's test starts in the forcing years, with their true outcomes and the fold matrix's forecasts."""
    flag = DROGUE_TYPES["undrogued"]
    s0, state, pos = ho.test_samples(flag, test_traj)
    t0 = ho.d.t[s0].astype("datetime64[s]").astype(datetime)
    horizon = timedelta(days=max(ho.horizon_steps) * ho.tau)
    keep = (state[:, 0] >= 0) & np.array([t.year in YEARS and t + horizon <= forcing_end(t.year) for t in t0])
    P, _ = ho.train_matrix(flag, train_traj)
    com, shares = matrix_forecast(ho, P, state[keep, 0])
    return {"fold": np.full(keep.sum(), f), "traj": ho.d.traj_idx[s0[keep]], "t0": ho.d.t[s0[keep]],
            "state": state[keep], "pos": pos[keep], "matrix_com": com,
            "matrix_in_R": shares[0], "matrix_exit": shares[1], "matrix_strand": shares[2]}


def load_starts(args):
    """The test starts, built from the subset and held_out's folds the first time and read back after."""
    path = f"{DIR}/starts_{setting_tag(args.res, args.tau, args.box)}.npz"
    if not os.path.exists(path):
        ho = HeldOut(Drifters(args.data), args.box, args.res, args.tau)
        parts = [fold_starts(ho, f, np.setdiff1d(trajs, test), test)
                 for name, _, f, trajs, test in folds(ho, np.random.default_rng(SEED)) if name == "undrogued"]
        starts = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.argsort(starts["t0"], kind="stable")
        starts = {k: v[order] for k, v in starts.items()}
        starts.update(n_cells=ho.n_cells, tau=ho.tau, steps=np.array(ho.horizon_steps))
        np.savez_compressed(path + ".tmp.npz", **starts)
        os.replace(path + ".tmp.npz", path)
        print(f"{len(order):,} test starts from {len(np.unique(starts['traj']))} held-out undrogued drifters → {path}")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# ── 2. One OpenDrift run per half month ──────────────────────────────────────

def seed_starts(land, lon0, lat0, box, res, rng):
    """N_EXACT particles at each start's position and N_CELL over the ocean part of its cell, start by start."""
    xs, ys = [], []
    for x, y in zip(lon0, lat0):
        cx = box[0] + (np.floor((x - box[0]) / res) + 0.5) * res
        cy = box[2] + (np.floor((y - box[2]) / res) + 0.5) * res
        cell_x, cell_y = seed_cell(land, cx, cy, res, N_CELL, rng)
        xs.append(np.r_[np.full(N_EXACT, x), cell_x])
        ys.append(np.r_[np.full(N_EXACT, y), cell_y])
    return np.concatenate(xs), np.concatenate(ys)


def forecasts(state, lon, lat, begin_hour, steps, lag, n_cells):
    """Each start's forecast from its two seedings: centre of mass of the particles afloat in R
    (seedings, starts, steps, 2), and the shares in R, exited and stranded (3, seedings, starts, steps)."""
    n_per = N_EXACT + N_CELL
    n = len(begin_hour)
    com, shares = np.full((2, n, len(steps), 2), np.nan), np.empty((3, 2, n, len(steps)))
    groups = [slice(0, N_EXACT), slice(N_EXACT, n_per)]
    for j, k in enumerate(steps):
        hour = np.repeat(begin_hour, n_per) + k * lag
        rows = np.arange(len(hour))
        s = state[rows, hour].reshape(n, n_per)
        x, y = lon[rows, hour].reshape(n, n_per), lat[rows, hour].reshape(n, n_per)
        for g, cols in enumerate(groups):
            in_R = s[:, cols] < n_cells
            count = in_R.sum(axis=1)
            with np.errstate(invalid="ignore"):
                com[g, :, j] = np.c_[np.where(in_R, x[:, cols], 0).sum(axis=1) / count, np.where(in_R, y[:, cols], 0).sum(axis=1) / count]
            shares[:, g, :, j] = [in_R.mean(axis=1), ((s[:, cols] >= n_cells) & (s[:, cols] < n_cells + 4)).mean(axis=1),
                                  ((s[:, cols] >= n_cells + 4) & (s[:, cols] < n_cells + 4 + N_COASTS)).mean(axis=1)]
    return com, shares


def run_batch(index, starts, land, states, args, path, diffusivity=DIFFUSIVITY):
    """One OceanDrift run for the starts at index, with each start's forecasts saved to path."""
    began = time.time()
    t0 = starts["t0"][index]
    lon0, lat0 = starts["pos"][index, 0, 0], starts["pos"][index, 0, 1]
    seed = int(os.path.basename(path)[:6]) * 10 + int(os.path.basename(path)[7])
    x, y = seed_starts(land, lon0, lat0, args.box, args.res, np.random.default_rng(seed))
    lag = int(round(float(starts["tau"]) * 24))
    span = int(max(starts["steps"])) * lag
    begin = datetime(1970, 1, 1) + timedelta(seconds=int(t0.min()))
    begin_hour = ((t0 - t0.min()) // 3600).astype(int)
    times = [begin + timedelta(hours=int(h)) for h in np.repeat(begin_hour, N_EXACT + N_CELL)]
    lon, lat, _, stranded, last = simulate(begin, x, y, np.zeros(len(x), int), int(begin_hour.max()) + span, seed,
                                           times=times, wind=WIND, max_age_hours=span + 1, diffusivity=diffusivity)
    first = np.argmax(np.isfinite(lon), axis=1)
    if not (first == np.repeat(begin_hour, N_EXACT + N_CELL)).all():
        raise SystemExit(f"{path}: particles were not released at their starts' hours.")
    state = hourly_states(lon, lat, stranded, last, states.cell_state, states.n_cells, states.box, states.n_lon)
    com, shares = forecasts(state, lon, lat, begin_hour, starts["steps"], lag, states.n_cells)
    np.savez_compressed(path + ".tmp.npz", index=index, com=com, in_R=shares[0], exit=shares[1], strand=shares[2],
                        on_land=stranded & (last == np.repeat(begin_hour, N_EXACT + N_CELL)))
    os.replace(path + ".tmp.npz", path)
    print(f"{os.path.basename(path)[:8]}: {len(index)} starts, {time.time() - began:.0f} s", flush=True)


def batches(starts, years):
    """Starts by half month, (name, index), in the years asked for."""
    t0 = starts["t0"].astype("datetime64[s]").astype(datetime)
    key = np.array([f"{t.year}{t.month:02d}_{1 if t.day < 16 else 2}" for t in t0])
    for name in sorted(set(key)):
        if int(name[:4]) in years:
            yield name, np.where(key == name)[0]


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    from roaring_landmask import RoaringLandmask

    def add_options(p):
        p.add_argument("--years", type=int, nargs="+", default=YEARS, help="years to run")
        p.add_argument("--diffusivity", type=float, default=DIFFUSIVITY, help="horizontal diffusivity, m²/s")

    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    check_coast_box(args.box)
    os.makedirs(DIR, exist_ok=True)
    out = folder(args.diffusivity)
    os.makedirs(out, exist_ok=True)
    starts = load_starts(args)
    states = MatrixStates(args)
    land = RoaringLandmask.new()
    for name, index in batches(starts, args.years):
        path = f"{out}/{name}.npz"
        if not os.path.exists(path):
            run_batch(index, starts, land, states, args, path, args.diffusivity)
    print(f"Forcing read from {FORCING}; forecasts saved → {out}/")


if __name__ == "__main__":
    main()
