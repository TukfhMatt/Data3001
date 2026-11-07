"""
Agulhas Region — Seasonal Transition Matrices

Builds separate transport operators for the four Southern Hemisphere seasons:

    DJF = Dec-Feb (summer)
    MAM = Mar-May (autumn)
    JJA = Jun-Aug (winter)
    SON = Sep-Nov (spring)

A move is assigned to the season of its START observation. Matrix building
matches transition_matrix: the same ocean states, five absorbing states
(W/E/S/N/stranded), neighbour pooling and MIN_DRIFTERS reliability rule.

For each drogue type the script reports:
  - seasonal data support
  - row-wise TVD against the annual operator
  - pairwise TVD between seasons
  - pairwise TVD on cells reliable in all four seasons
  - a permutation test of the pairwise TVD: season labels are shuffled
    between blocks (one drifter's moves in one season of one year), so the
    shuffled TVD is what four matrices of the same sizes differ by through
    sampling alone
  - 30-day release-point outcomes as a within-season sensitivity test

The 30-day calculation repeatedly applies one seasonal operator and is not a
calendar-year seasonal forecast.

    python -m agulhas seasonal_matrices --res 1 --tau 3.5
    python -m agulhas seasonal_matrices --res 2 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import (DROGUE_TYPES, MIN_DRIFTERS, box_tag, operator_path, parse_args, region_prefix,
                            release_points_path, setting_tag)
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import save_figure
from agulhas.transport import (EXIT_LABELS, STRANDED, cell_centres, load_operator, reliable, row_normalise,
                               state_of)

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
SEASON_NAMES = {
    "DJF": "Summer",
    "MAM": "Autumn",
    "JJA": "Winter",
    "SON": "Spring",
}
SEASON_LIST = list(SEASONS)
SEASON_PAIRS = [(i, j) for i in range(len(SEASON_LIST)) for j in range(i + 1, len(SEASON_LIST))]
N_PERM = 200
SEED = 0
RELEASE_DAYS = 30
SOURCE = "NOAA GDP hourly v2.01 via CloudDrift, hourly, gap <= 6 h"


def banner(title):
    print("\n========================================")
    print(title)
    print("========================================")


class SeasonalMoves:
    """All moves from R on one grid and lag, with the month each starts in. A move
    belongs to the season of its start observation. Every season and both drogue
    types share the ocean states of the annual matrices."""

    def __init__(self, data, args):
        drifters = Drifters(data)
        for v in ("typedeath", "end_date"):
            if v not in drifters.ds:
                raise SystemExit(f"{v} is missing from {data}. Refresh the subset metadata first:\n"
                                 "  python -m agulhas fetch_gdp_subset --metadata-only")
        self.moves = MoveSet(drifters, args.box, args.res, args.tau)
        self.d, self.args = drifters, args
        self.cells = np.unique(self.moves.cell_flat[self.moves.cell_flat >= 0])
        self.n_cells = len(self.cells)
        self.n_states = self.n_cells + len(EXIT_LABELS)
        self.start_month = drifters.month[self.moves.start]

    def season_moves(self, flag, season):
        """Moves of one drogue type that start in the season: (start, end, stranded)."""
        mv = self.moves
        use = np.isin(self.start_month, SEASONS[season]) & mv.of_type_mask(flag)
        return mv.start[use], mv.end[use], mv.stranded[use]

    def build(self, name, flag, season):
        """Build and save one seasonal matrix as transition_matrix builds the annual one,
        and describe it: (the saved operator, its support summary)."""
        args, d, n_cells, n_states = self.args, self.d, self.n_cells, self.n_states
        s, e, st = self.season_moves(flag, season)
        if len(s) == 0:
            raise SystemExit(f"No transitions for {name} {season}")
        from_state, to_state = self.moves.states(self.cells, s, e, st)
        C = np.zeros((n_states, n_states))
        np.add.at(C, (from_state, to_state), 1)
        pairs = np.unique(np.stack([from_state, d.traj_idx[s]], axis=1), axis=0)
        row_drifters = np.bincount(pairs[:, 0], minlength=n_states)
        row_obs = C.sum(axis=1)
        P, empty = row_normalise(C, self.cells, self.moves.grid.n_lon)
        empty_mask = np.isin(np.arange(n_cells), empty)
        flagged = row_drifters[:n_cells] < MIN_DRIFTERS
        years = d.year[s]
        out = operator_path(f"{name}_{season}", args.res, args.tau, args.box)
        np.savez_compressed(
            out, P=P, C=C, row_drifters=row_drifters[:n_cells], row_obs=row_obs[:n_cells],
            flagged=flagged, empty=empty_mask, cell_flat=self.cells,
            lon_edges=self.moves.grid.lon_edges, lat_edges=self.moves.grid.lat_edges,
            res=args.res, tau_days=args.tau, exit_labels=EXIT_LABELS, box=np.array(args.box), drogued=flag,
            season=season, season_name=SEASON_NAMES[season], season_months=np.array(SEASONS[season]),
            season_rule="season of transition start observation",
            years=np.array([years.min(), years.max()]), source=SOURCE,
        )
        summary = support_summary(name, season, P, row_obs[:n_cells], ~flagged & ~empty_mask, empty_mask,
                                  (len(s), len(np.unique(d.traj_idx[s])), years))
        print(f"Rows sum to 1         : {np.allclose(P.sum(axis=1), 1)}")
        print(f"Saved → {out}")
        return load_operator(out), summary


def support_summary(name, season, P, row_obs, ok, empty_mask, sizes):
    """Print and return one seasonal matrix's data support; sizes are (moves, drifters, move years)."""
    n_cells = len(row_obs)
    n_moves, n_drifters, years = sizes
    active = row_obs > 0
    exit_prob = P[:n_cells, n_cells:n_cells + STRANDED].sum(axis=1)
    strand_prob = P[:n_cells, n_cells + STRANDED]
    print(f"\n=== {name}, {season} ({SEASON_NAMES[season]}) ===")
    print(f"Transitions used      : {n_moves:,} ({years.min()}–{years.max()})")
    print(f"Distinct drifters     : {n_drifters:,}")
    print(f"Active rows           : {active.sum()} / {n_cells}")
    print(f"Reliable rows         : {ok.sum()} / {n_cells} ({ok.mean():.1%})")
    print(f"Cells with no moves   : {empty_mask.sum()}")
    print(f"Mean exit prob / step : {exit_prob.mean():.1%}")
    print(f"Mean strand prob/step : {strand_prob.mean():.2%}")
    return {
        "season": season, "season_name": SEASON_NAMES[season], "drogue": name,
        "transitions": n_moves, "drifters": n_drifters,
        "active_rows": int(active.sum()), "reliable_rows": int(ok.sum()),
        "reliable_share_all": float(ok.mean()),
        "reliable_share_active": float(ok[active].mean()) if active.any() else np.nan,
        "empty_rows": int(empty_mask.sum()),
        "mean_exit_per_step": float(exit_prob.mean()), "mean_strand_per_step": float(strand_prob.mean()),
    }


def print_header(sm, args):
    print("========================================")
    print("SEASONAL TRANSITION MATRICES")
    print("========================================")
    print(f"Region: {args.box}")
    print(f"Grid: {args.res:g}°")
    print(f"Lag: {args.tau:g} days")
    print(f"Ocean cells / states: {sm.n_cells} / {sm.n_states}")
    print(f"Candidate transitions: {len(sm.moves.start):,}")


def load_annual(sm, args):
    """The annual matrices, which must have the seasonal matrices' states."""
    ops = {}
    for name in DROGUE_TYPES:
        path = operator_path(name, args.res, args.tau, args.box)
        rebuild = f"  python -m agulhas transition_matrix --res {args.res:g} --tau {args.tau:g}"
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run:\n{rebuild}")
        op = load_operator(path)
        if op["P"].shape != (sm.n_states, sm.n_states):
            raise SystemExit(f"{path} has shape {op['P'].shape}, but the current model expects "
                             f"{(sm.n_states, sm.n_states)}.\nRebuild it with:\n{rebuild}")
        if not np.array_equal(op["cell_flat"], sm.cells):
            raise SystemExit(f"{path} uses a different ocean-cell state space. Rebuild it with:\n{rebuild}")
        if not np.array_equal(op["exit_labels"].astype(str), EXIT_LABELS.astype(str)):
            raise SystemExit(f"{path} has outdated absorbing states. Rebuild it with:\n{rebuild}")
        ops[name] = op
    return ops


def build_seasons(sm):
    seasonal_ops, rows = {}, []
    for season in SEASON_LIST:
        for name, flag in DROGUE_TYPES.items():
            seasonal_ops[(season, name)], summary = sm.build(name, flag, season)
            rows.append(summary)
    return seasonal_ops, pd.DataFrame(rows)


# ── Seasonal against annual, and season against season ─────────────────────

def compare_annual(annual_ops, seasonal_ops, cells):
    """Row TVD of every seasonal matrix against the annual one, per cell."""
    banner("SEASONAL VS ANNUAL MATRIX")
    n_cells = len(cells)
    rows = []
    for name in DROGUE_TYPES:
        annual = annual_ops[name]
        annual_reliable = reliable(annual)
        cell_lon, cell_lat = cell_centres(annual)
        for season in SEASON_LIST:
            seasonal = seasonal_ops[(season, name)]
            seasonal_reliable = reliable(seasonal)
            common = annual_reliable & seasonal_reliable
            tvd = 0.5 * np.abs(seasonal["P"][:n_cells] - annual["P"][:n_cells]).sum(axis=1)
            values = tvd[common]
            print(f"{name:10s} {season}: {common.sum():4d} common reliable rows | "
                  f"median TVD {np.median(values):.3f} | 90th pct {np.percentile(values, 90):.3f}")
            for i in range(n_cells):
                rows.append({"drogue": name, "season": season, "season_name": SEASON_NAMES[season], "state": i,
                             "cell_flat": int(cells[i]), "lon": float(cell_lon[i]), "lat": float(cell_lat[i]),
                             "annual_reliable": bool(annual_reliable[i]), "season_reliable": bool(seasonal_reliable[i]),
                             "common_reliable": bool(common[i]), "tvd_vs_annual": float(tvd[i])})
    return pd.DataFrame(rows)


def compare_pairs(seasonal_ops, n_cells):
    """Row TVD between every pair of seasons, on cells reliable in both and on cells
    reliable in all four (the same cells for every comparison)."""
    banner("PAIRWISE SEASON TVD")
    rows = []
    for name in DROGUE_TYPES:
        ok = {season: reliable(seasonal_ops[(season, name)]) for season in SEASON_LIST}
        all_four = np.logical_and.reduce([ok[s] for s in SEASON_LIST])
        print(f"\n{name}: cells reliable in all four seasons = {all_four.sum()} / {n_cells}")
        for i, j in SEASON_PAIRS:
            s1, s2 = SEASON_LIST[i], SEASON_LIST[j]
            pair_common = ok[s1] & ok[s2]
            tvd = 0.5 * np.abs(seasonal_ops[(s1, name)]["P"][:n_cells] - seasonal_ops[(s2, name)]["P"][:n_cells]).sum(axis=1)
            pair_values, four_values = tvd[pair_common], tvd[all_four]
            rows.append({
                "drogue": name, "season_1": s1, "season_2": s2,
                "pair_common_rows": int(pair_common.sum()),
                "median_tvd_pair_common": float(np.median(pair_values)),
                "p90_tvd_pair_common": float(np.percentile(pair_values, 90)),
                "all_four_common_rows": int(all_four.sum()),
                "median_tvd_all_four": float(np.median(four_values)) if len(four_values) else np.nan,
                "p90_tvd_all_four": float(np.percentile(four_values, 90)) if len(four_values) else np.nan,
            })
            if len(four_values):
                print(f"{s1} vs {s2}: pair n={pair_common.sum():4d}, median={np.median(pair_values):.3f} | "
                      f"all-four n={all_four.sum():4d}, median={np.median(four_values):.3f}")
            else:
                print("no all-four common reliable rows")
    return pd.DataFrame(rows)


# ── Permutation test of seasonal differences ─────────────────────────────────

def season_blocks(sm):
    """Without seasonality, a block (one drifter's moves in one season of one year)
    is equally likely to carry any season label. Shuffling the labels of whole
    blocks keeps each block's moves and their correlation together, so the
    shuffled pairwise TVD is the TVD that sampling alone produces.
    Returns each move's block and each block's season."""
    season_index = np.zeros(13, int)
    for k, season in enumerate(SEASON_LIST):
        season_index[SEASONS[season]] = k
    start = sm.moves.start
    season_year = sm.d.year[start] + (sm.start_month == 12)      # December joins the next DJF
    block_key = (sm.d.traj_idx[start].astype(np.int64) * 10000 + season_year) * 4 + season_index[sm.start_month]
    block_ids, block_of_move = np.unique(block_key, return_inverse=True)
    return block_of_move, (block_ids % 4).astype(int)


class SeasonShuffle:
    """One drogue type's moves grouped into (block, from, to) triples, so a shuffle
    only relabels blocks. Cells compared for each pair of seasons: reliable in both
    seasons of the observed data."""

    def __init__(self, sm, flag, block_of_move, ok):
        n_states = self.n_states = sm.n_states
        self.n_cells = sm.n_cells
        keep = sm.moves.of_type_mask(flag)
        mv = sm.moves
        f, to = mv.states(sm.cells, mv.start[keep], mv.end[keep], mv.stranded[keep])
        key = block_of_move[keep].astype(np.int64) * n_states**2 + f.astype(np.int64) * n_states + to
        uniq, self.n_moves = np.unique(key, return_counts=True)
        self.trip_block = uniq // n_states**2
        self.trip_ft = uniq % n_states**2
        self.ocean = self.trip_ft // n_states < self.n_cells
        self.pair_masks = [ok[i] & ok[j] for i, j in SEASON_PAIRS]

    def pair_medians(self, labels):
        """Median row TVD per season pair, with block b given season labels[b]."""
        n_states, n_cells = self.n_states, self.n_cells
        season_of_trip = labels[self.trip_block]
        rows = []
        for k in range(len(SEASON_LIST)):
            m = self.ocean & (season_of_trip == k)
            C = np.bincount(self.trip_ft[m], weights=self.n_moves[m], minlength=n_states**2).reshape(n_states, n_states)[:n_cells]
            rows.append((C / np.maximum(C.sum(axis=1), 1)[:, None], C.sum(axis=1) > 0))
        out = []
        for (i, j), mask in zip(SEASON_PAIRS, self.pair_masks):
            ok = mask & rows[i][1] & rows[j][1]      # a shuffle can empty a row
            out.append(np.median(0.5 * np.abs(rows[i][0][ok] - rows[j][0][ok]).sum(axis=1)))
        return np.array(out)


def permutation_test(sm, seasonal_ops):
    block_of_move, block_season = season_blocks(sm)
    rng = np.random.default_rng(SEED)
    rows, observed_null = [], {}
    banner(f"PERMUTATION TEST ({N_PERM} shuffles of {len(block_season):,} season blocks)")
    for name, flag in DROGUE_TYPES.items():
        shuffle = SeasonShuffle(sm, flag, block_of_move, [reliable(seasonal_ops[(s, name)]) for s in SEASON_LIST])
        observed = shuffle.pair_medians(block_season)
        null = np.array([shuffle.pair_medians(rng.permutation(block_season)) for _ in range(N_PERM)])
        observed_null[name] = (observed, null)
        print(f"\n{name}:")
        for p, (i, j) in enumerate(SEASON_PAIRS):
            p_value = (1 + np.sum(null[:, p] >= observed[p])) / (1 + N_PERM)
            n = shuffle.pair_masks[p].sum()
            rows.append({"drogue": name, "season_1": SEASON_LIST[i], "season_2": SEASON_LIST[j], "cells": int(n),
                         "observed_median_tvd": float(observed[p]), "null_median_tvd": float(np.median(null[:, p])),
                         "null_p95_tvd": float(np.percentile(null[:, p], 95)), "p_value": float(p_value)})
            print(f"  {SEASON_LIST[i]} vs {SEASON_LIST[j]}: n={n:4d} | "
                  f"observed {observed[p]:.3f} | shuffled {np.median(null[:, p]):.3f} "
                  f"(95th pct {np.percentile(null[:, p], 95):.3f}) | p = {p_value:.3f}")
    return pd.DataFrame(rows), observed_null


# ── 30-day release-point seasonal sensitivity ───────────────────────────────

def release_scenario(seasonal_ops, points, tau, n_cells):
    """Each release point after RELEASE_DAYS under each seasonal matrix. Repeating one
    season's matrix is a within-season sensitivity test, not a seasonal forecast."""
    steps = int(round(RELEASE_DAYS / tau))
    model_days = steps * tau
    banner("30-DAY RELEASE-POINT SEASONAL SCENARIO")
    print(f"30 requested days = {steps} steps = {model_days:g} model days")
    rows = []
    for pt in points.itertuples():
        for name in DROGUE_TYPES:
            for season in SEASON_LIST:
                op = seasonal_ops[(season, name)]
                try:
                    s = state_of(op, pt.lon, pt.lat)
                except ValueError:
                    continue
                p0 = np.zeros(op["P"].shape[0])
                p0[s] = 1.0
                p = p0 @ np.linalg.matrix_power(op["P"], steps)
                row = {"name": pt.name, "lon": pt.lon, "lat": pt.lat, "drogue": name, "season": season,
                       "season_name": SEASON_NAMES[season], "requested_days": RELEASE_DAYS, "steps": steps,
                       "model_days": model_days, "release_row_drifters": int(op["row_drifters"][s]),
                       "release_reliable": not bool(op["flagged"][s]) and not bool(op["empty"][s]),
                       "in_R": float(p[:n_cells].sum())}
                for k, label in enumerate(op["exit_labels"].astype(str)):
                    row[f"exit_{label}"] = float(p[n_cells + k])
                rows.append(row)
    return rows


# ── Summary figure ───────────────────────────────────────────────────────────

def plot_support(ax, support, width):
    x = np.arange(len(SEASON_LIST))
    for offset, name in enumerate(DROGUE_TYPES):
        sub = support[support["drogue"] == name].set_index("season").loc[SEASON_LIST]
        ax.bar(x + (offset - 0.5) * width, sub["reliable_share_all"], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(SEASON_LIST)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of ocean rows reliable")
    ax.set_title("Seasonal data support")
    ax.legend()


def plot_change(ax, cell_table, width):
    x = np.arange(len(SEASON_LIST))
    tvd_summary = cell_table[cell_table["common_reliable"]].groupby(["drogue", "season"])["tvd_vs_annual"].median()
    for offset, name in enumerate(DROGUE_TYPES):
        values = [tvd_summary.get((name, season), np.nan) for season in SEASON_LIST]
        ax.bar(x + (offset - 0.5) * width, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(SEASON_LIST)
    ax.set_ylabel("Median row TVD vs annual matrix")
    ax.set_title("Seasonal change in transport")
    ax.legend()


def plot_shuffled(ax, observed_null, width):
    x_pairs = np.arange(len(SEASON_PAIRS))
    for offset, (name, colour) in enumerate(zip(DROGUE_TYPES, ["tab:blue", "tab:orange"])):
        observed, null = observed_null[name]
        xs = x_pairs + (offset - 0.5) * width
        ax.vlines(xs, np.percentile(null, 2.5, axis=0), np.percentile(null, 97.5, axis=0),
                  color=colour, alpha=0.4, lw=6, label=f"{name}: shuffled (95% range)")
        ax.scatter(xs, observed, color=colour, edgecolor="k", zorder=3, label=f"{name}: observed")
    ax.set_xticks(x_pairs)
    ax.set_xticklabels([f"{SEASON_LIST[i]}–{SEASON_LIST[j]}" for i, j in SEASON_PAIRS])
    ax.set_ylabel("Median row TVD between seasons")
    ax.set_title("Seasonal differences against shuffled seasons")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)


def plot_summary(support, cell_table, observed_null, args, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    width = 0.35
    plot_support(axes[0], support, width)
    plot_change(axes[1], cell_table, width)
    plot_shuffled(axes[2], observed_null, width)
    fig.suptitle(f"Seasonal transport — {box_tag(args.box)}, {args.res:g}°, τ = {args.tau:g} d")
    fig.tight_layout()
    save_figure(path)


def print_done(saved, release_file, figure_file):
    banner("DONE")
    for label, path in saved:
        print(f"{label} → {path}")
    if release_file:
        print(f"30-day fate → {release_file}")
    print(f"Figure      → {figure_file}")
    print("\n30-day release results are within-season sensitivity scenarios; "
          "they are not calendar-year forecasts.")


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_support = f"data/seasonal_support_{tag}.csv"
    out_cells = f"data/seasonal_tvd_cells_{tag}.csv"
    out_pairs = f"data/seasonal_pairwise_tvd_{tag}.csv"
    out_perm = f"data/seasonal_permutation_{tag}.csv"
    out_release = f"data/seasonal_release_30d_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_seasonal_summary_{tag}.png"

    sm = SeasonalMoves(args.data, args)
    print_header(sm, args)
    annual_ops = load_annual(sm, args)
    seasonal_ops, support = build_seasons(sm)
    support.to_csv(out_support, index=False)
    cell_table = compare_annual(annual_ops, seasonal_ops, sm.cells)
    cell_table.to_csv(out_cells, index=False)
    compare_pairs(seasonal_ops, sm.n_cells).to_csv(out_pairs, index=False)
    perm_table, observed_null = permutation_test(sm, seasonal_ops)
    perm_table.to_csv(out_perm, index=False)

    release_rows = []
    if os.path.exists(release_points_path(args.box)):
        release_rows = release_scenario(seasonal_ops, pd.read_csv(release_points_path(args.box)), args.tau, sm.n_cells)
        pd.DataFrame(release_rows).to_csv(out_release, index=False)
    else:
        print(f"\n{release_points_path(args.box)} not found: skipping the release-point scenario (run shipping_overlay).")

    plot_summary(support, cell_table, observed_null, args, out_png)
    print_done([("Support    ", out_support), ("Cell TVD   ", out_cells), ("Pairwise   ", out_pairs),
                ("Permutation", out_perm)], out_release if release_rows else None, out_png)


if __name__ == "__main__":
    main()
