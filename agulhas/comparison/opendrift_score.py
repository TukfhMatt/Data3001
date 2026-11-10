"""
Agulhas Region — OpenDrift Against the Matrix on Held-out Drifters
Scores opendrift_drifters' forecasts against where each held-out undrogued
drifter actually went. The rule was fixed before any run was scored.

Score: the distance from a forecast's centre (of the material still in R) to the
drifter's actual position, at each whole step of τ up to 28 days, for starts
where the drifter is still in R at that step. Compared: the fold matrix,
OpenDrift seeded at the drifter's exact position, OpenDrift seeded over its 1°
cell (the information the matrix has), and persistence (the start itself).

OpenDrift beats the matrix at a horizon if the 95% interval of the difference in
median error (matrix − OpenDrift) excludes zero. Intervals resample whole
drifters. Starts with no OpenDrift particle in R at a step cannot be scored for
OpenDrift; every method is scored on the same starts and their number is printed.

    python -m agulhas opendrift_score
    python -m agulhas opendrift_score --diffusivity 1     # runs from opendrift_drifters --diffusivity 1
"""

import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.comparison.opendrift_drifters import DIR, folder
from agulhas.opendrift_common import DIFFUSIVITY
from agulhas.config import parse_args, setting_tag
from agulhas.plotting import save_figure
from agulhas.stats import haversine_km

N_BOOT = 1000
SEED = 0
METHODS = {"matrix": "#2a78d6", "opendrift_exact": "#eb6834", "opendrift_cell": "#2e9e6b", "persistence": "#7a7a7a"}


# ── 1. Forecasts of every start ──────────────────────────────────────────────

def load_forecasts(tag, diffusivity=DIFFUSIVITY):
    """The starts' truth and each method's forecast centre: positions (starts, steps, 2) by method, truth, and drifter."""
    with np.load(f"{DIR}/starts_{tag}.npz") as f:
        starts = {k: f[k] for k in f.files}
    n, steps = len(starts["traj"]), len(starts["steps"])
    exact, cell = np.full((n, steps, 2), np.nan), np.full((n, steps, 2), np.nan)
    done = np.zeros(n, bool)
    for path in sorted(glob.glob(f"{folder(diffusivity)}/20*_?.npz")):
        with np.load(path) as f:
            exact[f["index"]], cell[f["index"]] = f["com"][0], f["com"][1]
            done[f["index"]] = True
    if not done.all():
        raise SystemExit(f"{int((~done).sum())} of {n} starts have no OpenDrift run: finish opendrift_drifters first.")
    persistence = np.repeat(starts["pos"][:, :1], steps, axis=1)
    forecast = {"matrix": starts["matrix_com"], "opendrift_exact": exact, "opendrift_cell": cell, "persistence": persistence}
    return forecast, starts["pos"][:, 1:], starts["traj"]


def errors(forecast, truth):
    """Distance in km from each method's forecast to the drifter, (starts, steps) by method, and which starts are scorable."""
    err = {m: haversine_km(c[..., 0], c[..., 1], truth[..., 0], truth[..., 1]) for m, c in forecast.items()}
    in_R = np.isfinite(truth[..., 0])
    both = in_R & np.isfinite(err["opendrift_exact"]) & np.isfinite(err["opendrift_cell"])
    return err, in_R, both


# ── 2. Medians and intervals over whole drifters ─────────────────────────────

def median_table(err, scorable, traj, days, rng):
    """Median error of each method and its difference from the matrix, with 95% intervals from resampling drifters."""
    ids, inverse = np.unique(traj, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(ids))]
    draws = [np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))]) for _ in range(N_BOOT)]
    rows = []
    for k, d in enumerate(days):
        ok = scorable[:, k]
        e = {m: np.where(ok, err[m][:, k], np.nan) for m in METHODS}
        row = {"days": d, "starts": int(ok.sum()), "drifters": len(np.unique(traj[ok]))}
        for m in METHODS:
            row[f"median_{m}"] = np.nanmedian(e[m])
        for m in ("opendrift_exact", "opendrift_cell"):
            boot = [np.nanmedian(e["matrix"][i]) - np.nanmedian(e[m][i]) for i in draws]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row.update({f"diff_{m}": row["median_matrix"] - row[f"median_{m}"], f"diff_{m}_lo": lo, f"diff_{m}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


# ── 3. Summary ───────────────────────────────────────────────────────────────

def print_summary(table, n_starts, n_in_R):
    cols = ["days", "starts", "drifters"] + [f"median_{m}" for m in METHODS]
    print(table[cols].round(1).to_string(index=False))
    for m in ("opendrift_exact", "opendrift_cell"):
        better = (table[f"diff_{m}_lo"] > 0).sum()
        worse = (table[f"diff_{m}_hi"] < 0).sum()
        print(f"{m}: beats the matrix at {better} of {len(table)} horizons, the matrix beats it at {worse}; "
              f"difference at 28 days {table[f'diff_{m}'].iloc[-1]:.1f} km ({table[f'diff_{m}_lo'].iloc[-1]:.1f} to {table[f'diff_{m}_hi'].iloc[-1]:.1f})")
    print(f"Scored on the starts where the drifter is in R and OpenDrift has particles in R: {n_in_R:,} start-steps in R, "
          f"{int(table['starts'].sum()):,} scored of {n_starts:,} starts.")


def plot_errors(table, png):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, color in METHODS.items():
        a.plot(table["days"], table[f"median_{m}"], "o-", color=color, label=m.replace("_", " "))
    a.set_xlabel("Days after the start")
    a.set_ylabel("Median distance to the drifter (km)")
    a.legend(fontsize=8)
    for m in ("opendrift_exact", "opendrift_cell"):
        b.plot(table["days"], table[f"diff_{m}"], "o-", color=METHODS[m], label=f"matrix − {m.replace('_', ' ')}")
        b.fill_between(table["days"], table[f"diff_{m}_lo"], table[f"diff_{m}_hi"], color=METHODS[m], alpha=0.2)
    b.axhline(0, color="#10263B", lw=1)
    b.set_xlabel("Days after the start")
    b.set_ylabel("Matrix error − OpenDrift error (km)")
    b.legend(fontsize=8)
    fig.suptitle("Held-out undrogued drifters: forecast error of the matrix and OpenDrift")
    save_figure(png)


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    def add_options(p):
        p.add_argument("--diffusivity", type=float, default=DIFFUSIVITY, help="horizontal diffusivity of the runs to score, m²/s")

    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    name = "" if args.diffusivity == DIFFUSIVITY else f"_d{args.diffusivity:g}"
    out = f"data/opendrift/drifters_score{name}_{tag}.csv"
    png = f"figures/agulhas_opendrift_drifters{name}_{tag}.png"
    forecast, truth, traj = load_forecasts(tag, args.diffusivity)
    err, in_R, scorable = errors(forecast, truth)
    days = np.arange(1, truth.shape[1] + 1) * args.tau
    table = median_table(err, scorable, traj, days, np.random.default_rng(SEED))
    table.to_csv(out, index=False)
    print_summary(table, len(traj), int(in_R.sum()))
    plot_errors(table, png)
    print(f"Saved → {out}, {png}")


if __name__ == "__main__":
    main()
