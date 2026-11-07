"""
Agulhas Region — Chapman-Kolmogorov Test

Checks the Markov consistency of the transport operator by comparing

    P(tau)^2

with a transition matrix estimated directly from drifter pairs separated by

    2 * tau.

For each drogue type, the comparison is made row by row using total variation
distance (TVD):

    TVD_i = 0.5 * sum_j | P(tau)^2[i,j] - P(2*tau)[i,j] |

Only ocean rows reliable in both the tau and 2*tau operators are used for the
main summary statistics.

The script also reports differences in:
  - probability of remaining inside R
  - each absorbing fate (W/E/S/N/stranded)
  - share of reliable rows with TVD below 0.25 and 0.50

Part of any row's TVD is sampling noise in the direct 2*tau row, which is
estimated from a finite number of drifters. Resampling whole drifters
(N_BOOT times, as in bootstrap) gives each row's noise level: the TVD between a
bootstrap 2*tau row and the full-data row. A row departs from the Markov
prediction by more than noise when its TVD exceeds the 95th percentile of its
bootstrap TVD.

The TVD thresholds are descriptive summaries, not pass/fail criteria.

The direct 2*tau matrices must already exist. For the baseline setting:

    python -m agulhas transition_matrix --res 1 --tau 7
    python -m agulhas chapman_kolmogorov --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, operator_path, parse_args, region_prefix, setting_tag
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import save_figure
from agulhas.transport import cell_centres, load_operator, reliable, resampled_counts

N_BOOT = 1000
SEED = 0


def print_header(args):
    print("========================================")
    print("CHAPMAN-KOLMOGOROV TEST")
    print("========================================")
    print(f"Region: {args.box}")
    print(f"Grid: {args.res:g}°")
    print(f"Base lag: {args.tau:g} days")
    print(f"Direct comparison lag: {2 * args.tau:g} days")
    print("\nComparing:")
    print(f"  P({args.tau:g} d)^2")
    print("against")
    print(f"  P({2 * args.tau:g} d)")


# ── Operators ────────────────────────────────────────────────────────────────

def load_pair(name, args):
    """The τ and directly estimated 2τ operators of one drogue type, which must share their states."""
    res, tau = args.res, args.tau
    tau_path = operator_path(name, res, tau, args.box)
    direct_path = operator_path(name, res, 2 * tau, args.box)
    for path, lag, what in [(tau_path, tau, "Build it first with"), (direct_path, 2 * tau, "Build the direct 2*tau operator first with")]:
        if not os.path.exists(path):
            raise SystemExit(f"\nMissing {path}\n{what}:\n"
                             f"  python -m agulhas transition_matrix --res {res:g} --tau {lag:g}")
    op_tau, op_direct = load_operator(tau_path), load_operator(direct_path)
    if op_tau["P"].shape != op_direct["P"].shape:
        raise SystemExit(f"\nState-space mismatch for {name}:\n"
                         f"  tau matrix:    {op_tau['P'].shape}\n"
                         f"  2*tau matrix:  {op_direct['P'].shape}\n"
                         "Rebuild both matrices with transition_matrix.")
    if not np.array_equal(op_tau["cell_flat"], op_direct["cell_flat"]):
        raise SystemExit(f"\nOcean-cell state-space mismatch for {name}.\nRebuild both matrices using the same grid and box.")
    if not np.array_equal(op_tau["exit_labels"].astype(str), op_direct["exit_labels"].astype(str)):
        raise SystemExit(f"\nAbsorbing-state mismatch for {name}.\nRebuild both matrices using the current pipeline.")
    return op_tau, op_direct


# ── Sampling noise of the direct 2τ rows ─────────────────────────────────────

def row_noise(op_direct, counts, rng):
    """Each row's bootstrap TVD against its full-data row: the TVD that sampling alone
    produces for that row (median and 95th percentile)."""
    n_cells, n_states = len(op_direct["cell_flat"]), op_direct["P"].shape[0]
    rows = op_direct["C"][:n_cells]
    total = rows.sum(axis=1)
    P_rows = rows / np.maximum(total, 1)[:, None]
    boot_tvd = np.full((N_BOOT, n_cells), np.nan)
    for b in range(N_BOOT):
        Cb = resampled_counts(counts, n_states, rng)[:n_cells]
        tb = Cb.sum(axis=1)
        ok = (tb > 0) & (total > 0)  # a resample can miss every drifter of a row
        boot_tvd[b, ok] = 0.5 * np.abs(Cb[ok] / tb[ok, None] - P_rows[ok]).sum(axis=1)
    with np.errstate(all="ignore"):
        return {"median": np.nanmedian(boot_tvd, axis=0), "p95": np.nanpercentile(boot_tvd, 95, axis=0)}


def all_noise(operators, args):
    """Rebuild the direct 2τ counts per drifter (they must match transition_matrix
    exactly), then resample whole drifters."""
    moves = MoveSet(Drifters(args.data), args.box, args.res, 2 * args.tau)
    rng = np.random.default_rng(SEED)
    noise = {}
    print()
    for name, flag in DROGUE_TYPES.items():
        op_direct = operators[name][1]
        try:
            counts = moves.counts_for(op_direct, name, flag)
        except AssertionError:
            raise SystemExit(f"Rebuilt {2 * args.tau:g}-day counts differ from the saved {name} matrix. "
                             f"Rebuild it with:\n  python -m agulhas transition_matrix "
                             f"--res {args.res:g} --tau {2 * args.tau:g}")
        noise[name] = row_noise(op_direct, counts, rng)
        print(f"{name}: {counts[3]:,} drifters resampled {N_BOOT} times for the {2 * args.tau:g}-day rows")
    return noise


# ── Chapman-Kolmogorov comparison ────────────────────────────────────────────

def compare(op_tau, op_direct, noise):
    """Row by row, two τ steps against one directly estimated 2τ step."""
    n_cells = len(op_tau["cell_flat"])
    P_two_step = op_tau["P"] @ op_tau["P"]
    P_direct = op_direct["P"]
    c = {"reliable_tau": reliable(op_tau), "reliable_direct": reliable(op_direct)}
    c["common"] = c["reliable_tau"] & c["reliable_direct"]
    # Full row-wise TVD, including ocean and all absorbing states
    c["tvd"] = 0.5 * np.abs(P_two_step[:n_cells] - P_direct[:n_cells]).sum(axis=1)
    c["in_R_two_step"] = P_two_step[:n_cells, :n_cells].sum(axis=1)
    c["in_R_direct"] = P_direct[:n_cells, :n_cells].sum(axis=1)
    c["absdiff_in_R"] = np.abs(c["in_R_two_step"] - c["in_R_direct"])
    c["fate_two_step"] = P_two_step[:n_cells, n_cells:]
    c["fate_direct"] = P_direct[:n_cells, n_cells:]
    c["fate_absdiff"] = np.abs(c["fate_two_step"] - c["fate_direct"])
    c["mean_fate_absdiff"] = c["fate_absdiff"].mean(axis=1)
    c["noise_median"], c["noise_p95"] = noise["median"], noise["p95"]
    c["above_noise"] = c["tvd"] > noise["p95"]
    c["excess"] = c["tvd"] - noise["median"]
    return c


def summary_row(name, op_tau, c, args):
    """The comparison over common reliable rows. The TVD thresholds are descriptive, not pass/fail."""
    n_cells = len(op_tau["cell_flat"])
    common, values = c["common"], c["tvd"][c["common"]]
    summary = {
        "drogue": name, "res": args.res, "tau_days": args.tau, "direct_tau_days": 2 * args.tau,
        "n_cells": n_cells,
        "reliable_tau": int(c["reliable_tau"].sum()), "reliable_2tau": int(c["reliable_direct"].sum()),
        "common_reliable": int(common.sum()), "common_reliable_share": float(common.mean()),
        "mean_tvd": float(np.mean(values)), "median_tvd": float(np.median(values)),
        "p90_tvd": float(np.percentile(values, 90)), "max_tvd": float(np.max(values)),
        "share_tvd_below_0p25": float(np.mean(values < 0.25)), "share_tvd_below_0p50": float(np.mean(values < 0.50)),
        "median_noise_tvd": float(np.median(c["noise_median"][common])),
        "median_excess_tvd": float(np.median(c["excess"][common])),
        "share_above_noise_p95": float(c["above_noise"][common].mean()),
        "median_absdiff_in_R": float(np.median(c["absdiff_in_R"][common])),
        "median_fate_mae": float(np.median(c["mean_fate_absdiff"][common])),
    }
    for k, label in enumerate(op_tau["exit_labels"].astype(str)):
        summary[f"median_absdiff_{label}"] = float(np.median(c["fate_absdiff"][common, k]))
    return summary


def cell_rows(name, op_tau, op_direct, c):
    cell_lon, cell_lat = cell_centres(op_tau)
    exit_labels = op_tau["exit_labels"].astype(str)
    rows = []
    for i in range(len(op_tau["cell_flat"])):
        row = {
            "drogue": name, "state": i, "cell_flat": int(op_tau["cell_flat"][i]),
            "lon": float(cell_lon[i]), "lat": float(cell_lat[i]),
            "reliable_tau": bool(c["reliable_tau"][i]), "reliable_2tau": bool(c["reliable_direct"][i]),
            "common_reliable": bool(c["common"][i]), "tvd": float(c["tvd"][i]),
            "noise_tvd_median": float(c["noise_median"][i]), "noise_tvd_p95": float(c["noise_p95"][i]),
            "above_noise": bool(c["above_noise"][i]),
            "in_R_two_step": float(c["in_R_two_step"][i]), "in_R_direct": float(c["in_R_direct"][i]),
            "absdiff_in_R": float(c["absdiff_in_R"][i]), "mean_fate_absdiff": float(c["mean_fate_absdiff"][i]),
            "row_drifters_tau": int(op_tau["row_drifters"][i]), "row_drifters_2tau": int(op_direct["row_drifters"][i]),
        }
        for k, label in enumerate(exit_labels):
            row[f"two_step_{label}"] = float(c["fate_two_step"][i, k])
            row[f"direct_{label}"] = float(c["fate_direct"][i, k])
            row[f"absdiff_{label}"] = float(c["fate_absdiff"][i, k])
        rows.append(row)
    return rows


def print_comparison(name, s, exit_labels):
    print("\n----------------------------------------")
    print(name.upper())
    print("----------------------------------------")
    n = s["n_cells"]
    print(f"Ocean cells              : {n}")
    print(f"Reliable at tau          : {s['reliable_tau']} / {n} ({s['reliable_tau'] / n:.1%})")
    print(f"Reliable at 2*tau        : {s['reliable_2tau']} / {n} ({s['reliable_2tau'] / n:.1%})")
    print(f"Common reliable rows     : {s['common_reliable']} / {n} ({s['common_reliable_share']:.1%})")
    print(f"Median TVD               : {s['median_tvd']:.3f}")
    print(f"Mean TVD                 : {s['mean_tvd']:.3f}")
    print(f"90th percentile TVD      : {s['p90_tvd']:.3f}")
    print(f"Max TVD                  : {s['max_tvd']:.3f}")
    print(f"Rows with TVD < 0.25     : {s['share_tvd_below_0p25']:.1%}")
    print(f"Rows with TVD < 0.50     : {s['share_tvd_below_0p50']:.1%}")
    print(f"Median noise TVD (2*tau) : {s['median_noise_tvd']:.3f}")
    print(f"Median TVD above noise   : {s['median_excess_tvd']:.3f}")
    print(f"Rows above noise (95%)   : {s['share_above_noise_p95']:.1%}")
    print(f"Median |Δ in-R prob|     : {s['median_absdiff_in_R']:.3f}")
    print(f"Median fate MAE          : {s['median_fate_mae']:.3f}")
    for label in exit_labels:
        print(f"Median |Δ {label:8s}|       : {s[f'median_absdiff_{label}']:.3f}")


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_tvd_maps(results, args, path):
    """Row TVD on the grid, common reliable rows only, the colour scale capped at the
    95th percentile so a handful of extreme cells do not dominate it."""
    vmax = float(np.percentile(np.concatenate([c["tvd"][c["common"]] for _, c in results.values()]), 95))
    if vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, len(DROGUE_TYPES), figsize=(13, 5), sharex=True, sharey=True)
    if len(DROGUE_TYPES) == 1:
        axes = [axes]
    for ax, name in zip(axes, DROGUE_TYPES):
        op, c = results[name]
        common = c["common"]
        n_lon, n_lat = len(op["lon_edges"]) - 1, len(op["lat_edges"]) - 1
        grid = np.full(n_lat * n_lon, np.nan)
        grid[op["cell_flat"][common]] = c["tvd"][common]
        image = ax.imshow(grid.reshape(n_lat, n_lon), origin="lower",
                          extent=[op["lon_edges"][0], op["lon_edges"][-1], op["lat_edges"][0], op["lat_edges"][-1]],
                          aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"{name}\nmedian TVD = {np.median(c['tvd'][common]):.3f}, n = {common.sum()}")
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(image, ax=axes, label="Row-wise TVD: P(tau)^2 vs direct P(2 tau)", shrink=0.85)
    fig.suptitle(f"Chapman-Kolmogorov test — {args.res:g}°, tau = {args.tau:g} d")
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.90, wspace=0.12)
    save_figure(path)


def plot_noise(results, args, path):
    """Row TVD against the 95th percentile of its direct row's bootstrap TVD."""
    fig, axes = plt.subplots(1, len(DROGUE_TYPES), figsize=(11, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, DROGUE_TYPES):
        c = results[name][1]
        common = c["common"]
        x, y = c["noise_p95"][common], c["tvd"][common]
        above = y > x
        ax.scatter(x[~above], y[~above], s=8, alpha=0.5, color="tab:grey", label="within noise")
        ax.scatter(x[above], y[above], s=8, alpha=0.6, color="tab:red", label="above noise")
        lim = max(x.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="k", lw=1, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{name}: {above.mean():.0%} of {common.sum()} rows above noise")
        ax.set_xlabel(f"95th percentile bootstrap TVD of the direct {2 * args.tau:g}-day row")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel(f"TVD: P({args.tau:g} d)^2 vs direct P({2 * args.tau:g} d)")
    fig.suptitle(f"Chapman-Kolmogorov TVD against sampling noise — {args.res:g}°, tau = {args.tau:g} d")
    fig.tight_layout()
    save_figure(path)


def print_interpretation():
    print("\n========================================")
    print("DONE")
    print("========================================")
    print("\nInterpretation:")
    print("TVD = 0 means exact Chapman-Kolmogorov agreement.")
    print("Larger TVD means the directly observed 2*tau transition "
          "distribution differs more from two applications of P(tau).")
    print("The TVD < 0.25 and TVD < 0.50 shares are descriptive summaries, "
          "not formal pass/fail thresholds.")
    print("Rows above noise have a TVD larger than the 95th percentile of their "
          "direct row's bootstrap TVD, so sampling alone is unlikely to explain it. "
          "The noise level ignores the smaller noise in P(tau)^2 and is not "
          "corrected for the number of rows tested.")


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    prefix = region_prefix(args.data)
    out_cells = f"data/chapman_kolmogorov_cells_{tag}.csv"
    out_summary = f"data/chapman_kolmogorov_summary_{tag}.csv"
    out_map = f"figures/{prefix}_chapman_kolmogorov_{tag}.png"
    out_noise = f"figures/{prefix}_chapman_kolmogorov_noise_{tag}.png"

    print_header(args)
    operators = {name: load_pair(name, args) for name in DROGUE_TYPES}
    noise = all_noise(operators, args)

    summaries, cells, results = [], [], {}
    for name in DROGUE_TYPES:
        op_tau, op_direct = operators[name]
        c = compare(op_tau, op_direct, noise[name])
        if c["common"].sum() == 0:
            raise SystemExit(f"No common reliable rows for {name}.")
        summaries.append(summary_row(name, op_tau, c, args))
        print_comparison(name, summaries[-1], op_tau["exit_labels"].astype(str))
        cells += cell_rows(name, op_tau, op_direct, c)
        results[name] = (op_tau, c)
    pd.DataFrame(cells).to_csv(out_cells, index=False)
    pd.DataFrame(summaries).to_csv(out_summary, index=False)
    print(f"\nCell results saved → {out_cells}")
    print(f"Summary saved      → {out_summary}")

    plot_tvd_maps(results, args, out_map)
    print(f"Figure saved       → {out_map}")
    plot_noise(results, args, out_noise)
    print(f"Figure saved       → {out_noise}")
    print_interpretation()


if __name__ == "__main__":
    main()
