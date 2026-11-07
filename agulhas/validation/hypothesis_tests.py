"""
Agulhas Region — Hypothesis Tests
Applies the decision rule of each hypothesis in the project proposal:

  • H1 (pathways): the interval of P(exit E) − P(exit W), eventual fate averaged
    over reliable cells, excludes zero. Also the share of cells where it does.
  • H2 (windage): undrogued stranding exceeds drogued from the same release
    cell, with non-overlapping intervals, in most lane cells (the traffic points
    from shipping_overlay). Within a year as the headline, plus eventually and the interval
    of the difference. Drogued stranding is a lower bound (drogue_loss), which favours H2.
  • H3 (skill): the operator has a lower median position error than both
    persistence and mean-current advection in at least 4 of 5 folds, at each
    horizon of 14 days or more. Read from the folds saved by held_out; the baselines
    from the cell centre, the same information the matrix has, are reported too.

Intervals come from N_BOOT resamples of whole drifters, drawn independently for
each drogue type (as bootstrap). Needs transition_matrix, shipping_overlay and held_out.

    python -m agulhas hypothesis_tests --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, operator_path, parse_args, region_prefix, release_points_path, setting_tag
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import save_figure
from agulhas.transport import (EXIT_LABELS, STRANDED, load_operator, reliable, resampled_counts, row_normalise,
                               state_of, stranding_chain)

N_BOOT = 1000
SEED = 0
YEAR_DAYS = 365
H3_MIN_DAYS = 14
H3_MIN_WINS = 4
SERIES = {"drogued": ("#2a78d6", "o"), "undrogued": ("#eb6834", "s")}
COL_E, COL_W = list(EXIT_LABELS).index("E"), list(EXIT_LABELS).index("W")


def interval(x):
    return np.percentile(x, [2.5, 97.5], axis=0)


# ── Full data and drifter resamples ──────────────────────────────────────────

def outcomes(P, n, ok, lane_states, year_steps):
    """E − W per reliable cell, and stranding per lane within a year and eventually."""
    fate = stranding_chain(P, n, STRANDED)[0]
    p = np.eye(P.shape[0])[lane_states]
    for _ in range(year_steps):
        p = p @ P
    return fate[ok, COL_E] - fate[ok, COL_W], p[:, n + STRANDED], fate[lane_states, STRANDED]


def resample_outcomes(op, counts, lanes, n_lon, year_steps, rng):
    """One drogue type's outcomes on the full data, and on N_BOOT resamples of its drifters."""
    n, n_states = len(op["cell_flat"]), op["P"].shape[0]
    ok = reliable(op)
    lane_states = np.array([state_of(op, r.lon, r.lat) for r in lanes.itertuples()])
    full = outcomes(op["P"], n, ok, lane_states, year_steps)
    draws = [np.empty((N_BOOT,) + x.shape, dtype=np.float32) for x in full]
    for b in range(N_BOOT):
        Pb = row_normalise(resampled_counts(counts, n_states, rng), op["cell_flat"], n_lon)[0]
        for d, x in zip(draws, outcomes(Pb, n, ok, lane_states, year_steps)):
            d[b] = x
    return {"cells": int(ok.sum()), "full": full, "draws": draws}


def run_resamples(moves, lanes, args, rng):
    year_steps = int(round(YEAR_DAYS / args.tau))
    results = {}
    for name, flag in DROGUE_TYPES.items():
        op = load_operator(operator_path(name, args.res, args.tau, args.box))
        counts = moves.counts_for(op, name, flag)
        results[name] = resample_outcomes(op, counts, lanes, moves.grid.n_lon, year_steps, rng)
        print(f"{name:<10} {counts[3]:,} drifters resampled {N_BOOT} times")
    return results


# ── The three decision rules ─────────────────────────────────────────────────

def test_h1(results):
    """H1: east against west."""
    rows = []
    for name, res in results.items():
        lo, hi = interval(res["draws"][0].mean(axis=1))
        cell_lo, cell_hi = interval(res["draws"][0])
        rows.append({"drogue": name, "cells": res["cells"], "mean_E_minus_W": res["full"][0].mean(), "lo": lo, "hi": hi,
                     "share_cells_east": (cell_lo > 0).mean(), "share_cells_west": (cell_hi < 0).mean(),
                     "share_cells_unclear": ((cell_lo <= 0) & (cell_hi >= 0)).mean(), "supported": lo > 0})
    return pd.DataFrame(rows)


def test_h2(results, lanes):
    """H2: undrogued against drogued stranding in lane cells, within a year and eventually."""
    rows = []
    dr, un = results["drogued"], results["undrogued"]
    for k, (label, days) in enumerate([("one year", YEAR_DAYS), ("eventually", "eventual")], start=1):
        dr_lo, dr_hi = interval(dr["draws"][k])
        un_lo, un_hi = interval(un["draws"][k])
        diff_lo, diff_hi = interval(un["draws"][k] - dr["draws"][k])
        for i, r in enumerate(lanes.itertuples()):
            rows.append({"name": r.name, "lon": r.lon, "lat": r.lat, "horizon": days,
                         "drogued": dr["full"][k][i], "drogued_lo": dr_lo[i], "drogued_hi": dr_hi[i],
                         "undrogued": un["full"][k][i], "undrogued_lo": un_lo[i], "undrogued_hi": un_hi[i],
                         "diff_lo": diff_lo[i], "diff_hi": diff_hi[i],
                         "non_overlapping": un_lo[i] > dr_hi[i], "diff_above_zero": diff_lo[i] > 0})
    return pd.DataFrame(rows)


def test_h3(folds):
    """H3: folds won against each baseline, at horizons of H3_MIN_DAYS or more."""
    folds = folds[folds["days"] >= H3_MIN_DAYS]
    win = lambda col: (folds["err_markov_km"] < folds[col]).astype(int)
    folds = folds.assign(both=win("err_persist_km") & win("err_advect_km"), advect=win("err_advect_km"),
                         persist=win("err_persist_km"), advect_centre=win("err_advect_centre_km"),
                         persist_centre=win("err_persist_centre_km"))
    h3 = folds.groupby(["type", "days"], as_index=False).agg(
        folds=("fold", "nunique"), advect=("advect", "sum"), persist=("persist", "sum"), both=("both", "sum"),
        advect_centre=("advect_centre", "sum"), persist_centre=("persist_centre", "sum"),
        err_markov_km=("err_markov_km", "mean"), err_advect_km=("err_advect_km", "mean"),
        err_persist_km=("err_persist_km", "mean"))
    h3["supported"] = h3["both"] >= H3_MIN_WINS
    return h3


# ── Figure: lanes (H2) and folds (H3) ────────────────────────────────────────

def plot_lanes(ax, h2_year, n_lanes, h2_wins):
    order = h2_year.sort_values("undrogued")["name"].tolist()
    for name, (color, marker) in SERIES.items():
        shift = 0.14 if name == "undrogued" else -0.14
        rows = h2_year.set_index("name").loc[order]
        y = np.arange(len(order)) + shift
        ax.hlines(y, 100 * rows[f"{name}_lo"], 100 * rows[f"{name}_hi"], color=color, lw=2)
        ax.scatter(100 * rows[name], y, color=color, marker=marker, s=45, edgecolor="white", zorder=3, label=name.capitalize())
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Stranded within a year (%), with 95% interval")
    ax.set_title(f"H2: non-overlapping intervals in {h2_wins} of {n_lanes} lane cells", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


def plot_folds(ax, folds):
    ax.axhline(0, color="#777777", lw=1)
    for name, (color, marker) in SERIES.items():
        g = folds[folds["type"] == name]
        shift = 0.35 if name == "undrogued" else -0.35
        ax.scatter(g["days"] + shift, g["err_advect_km"] - g["err_markov_km"], color=color, marker=marker, s=26,
                   alpha=0.8, label=name.capitalize())
    ax.axvline(H3_MIN_DAYS - 1.75, color="#777777", lw=1, ls=":")
    ax.text(H3_MIN_DAYS - 1.5, ax.get_ylim()[1], "H3 horizons →", fontsize=9, va="top", color="#555555")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Advection error − operator error (km), one mark per fold")
    ax.set_title("H3: above zero, the operator beats mean-current advection", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


def plot_tests(h2_year, folds, n_lanes, h2_wins, args, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_lanes(axes[0], h2_year, n_lanes, h2_wins)
    plot_folds(axes[1], folds)
    fig.suptitle(f"Hypothesis tests — {args.res:g}°, τ = {args.tau:g} d", fontsize=12)
    save_figure(path)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(h1, h2, h3, n_lanes, h2_wins, h2_supported):
    pd.set_option("display.width", 200)
    print("\nH1: P(exit E) − P(exit W), eventual fate, mean over reliable cells")
    print(h1.round(3).to_string(index=False))
    print(f"\nH2: stranding in {n_lanes} lane cells; non-overlapping within a year in {h2_wins} "
          f"→ {'supported' if h2_supported else 'not supported'} (drogued is a lower bound)")
    show = h2[["name", "horizon", "drogued", "drogued_lo", "drogued_hi", "undrogued", "undrogued_lo", "undrogued_hi",
               "non_overlapping", "diff_above_zero"]]
    print(show.round(3).to_string(index=False))
    print(f"\nH3: folds won of {int(h3['folds'].max())}; supported = beats both baselines in at least {H3_MIN_WINS}")
    print(h3.drop(columns="folds").round({c: 0 for c in h3.columns if c.startswith("err_")}).to_string(index=False))


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    folds_path = f"data/validation_folds_{tag}.csv"
    out_h1, out_h2, out_h3 = [f"data/hypothesis_h{k}_{tag}.csv" for k in (1, 2, 3)]
    out_png = f"figures/{region_prefix(args.data)}_hypothesis_tests_{tag}.png"
    if not os.path.exists(folds_path):
        raise SystemExit(f"{folds_path} not found: run held_out at this grid and lag first.")

    moves = MoveSet(Drifters(args.data), args.box, args.res, args.tau)
    points = pd.read_csv(release_points_path(args.box))
    lanes = points[points["kind"] == "traffic"].reset_index(drop=True)
    results = run_resamples(moves, lanes, args, np.random.default_rng(SEED))

    folds = pd.read_csv(folds_path)
    h1, h2, h3 = test_h1(results), test_h2(results, lanes), test_h3(folds)
    h2_year = h2[h2["horizon"] == YEAR_DAYS]
    h2_wins = int(h2_year["non_overlapping"].sum())
    h1.to_csv(out_h1, index=False)
    h2.to_csv(out_h2, index=False)
    h3.to_csv(out_h3, index=False)

    plot_tests(h2_year, folds, len(lanes), h2_wins, args, out_png)
    print_summary(h1, h2, h3, len(lanes), h2_wins, h2_wins > len(lanes) / 2)
    print(f"\nSaved → {out_h1}, {out_h2}, {out_h3}, {out_png}")


if __name__ == "__main__":
    main()
