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
(N_BOOT times, as in 06) gives each row's noise level: the TVD between a
bootstrap 2*tau row and the full-data row. A row departs from the Markov
prediction by more than noise when its TVD exceeds the 95th percentile of its
bootstrap TVD.

The TVD thresholds are descriptive summaries, not pass/fail criteria.

The direct 2*tau matrices must already exist. For the baseline setting:

    python scripts/03_transition_matrix.py --res 1 --tau 7
    python scripts/20_chapman_kolmogorov.py --res 1 --tau 3.5
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from config import DROGUE_TYPES, grid_shape, operator_path, parse_args, setting_tag

from transport import (
    cell_centres,
    drifter_counts,
    load_operator,
    move_indices,
    reliable,
    resampled_counts,
    same_drogue,
    strand_times,
    transition_states,
)


# ── 1. Settings ───────────────────────────────────────────────────────────────
args = parse_args(__doc__, grid=True, lag=True)
RES = args.res
TAU = args.tau
DIRECT_TAU = 2 * TAU
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
N_BOOT = 1000
SEED = 0
TAG = setting_tag(RES, TAU, args.box)
CELL_FILE = f"data/chapman_kolmogorov_cells_{TAG}.csv"
SUMMARY_FILE = f"data/chapman_kolmogorov_summary_{TAG}.csv"
FIGURE_FILE = f"figures/agulhas_chapman_kolmogorov_{TAG}.png"
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
print("========================================")
print("CHAPMAN-KOLMOGOROV TEST")
print("========================================")
print(f"Region: {args.box}")
print(f"Grid: {RES:g}°")
print(f"Base lag: {TAU:g} days")
print(f"Direct comparison lag: {DIRECT_TAU:g} days")
print("\nComparing:")
print(f"  P({TAU:g} d)^2")
print("against")
print(f"  P({DIRECT_TAU:g} d)")


# ── 2. Load and check operators ──────────────────────────────────────────────
operators = {}

for name in DROGUE_TYPES:
    tau_path = operator_path(name, RES, TAU, args.box)
    direct_path = operator_path(name, RES, DIRECT_TAU, args.box)

    if not os.path.exists(tau_path):
        raise SystemExit(
            f"\nMissing {tau_path}\n"
            f"Build it first with:\n"
            f"  python scripts/03_transition_matrix.py "
            f"--res {RES:g} --tau {TAU:g}"
        )

    if not os.path.exists(direct_path):
        raise SystemExit(
            f"\nMissing {direct_path}\n"
            f"Build the direct 2*tau operator first with:\n"
            f"  python scripts/03_transition_matrix.py "
            f"--res {RES:g} --tau {DIRECT_TAU:g}"
        )

    op_tau = load_operator(tau_path)
    op_direct = load_operator(direct_path)

    if op_tau["P"].shape != op_direct["P"].shape:
        raise SystemExit(
            f"\nState-space mismatch for {name}:\n"
            f"  tau matrix:    {op_tau['P'].shape}\n"
            f"  2*tau matrix:  {op_direct['P'].shape}\n"
            "Rebuild both matrices using the current version of script 03."
        )

    if not np.array_equal(op_tau["cell_flat"], op_direct["cell_flat"]):
        raise SystemExit(
            f"\nOcean-cell state-space mismatch for {name}.\n"
            "Rebuild both matrices using the same grid and box."
        )

    if not np.array_equal(op_tau["exit_labels"].astype(str), op_direct["exit_labels"].astype(str)):
        raise SystemExit(
            f"\nAbsorbing-state mismatch for {name}.\n"
            "Rebuild both matrices using the current pipeline."
        )

    operators[name] = (op_tau, op_direct)


# ── 3. Sampling noise of the direct 2*tau rows ───────────────────────────────
# Rebuild the direct 2*tau counts per drifter (they must match 03 exactly),
# then resample whole drifters: each row's bootstrap TVD against the full-data
# row is the TVD that sampling alone produces for that row.
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
n_lon, _ = grid_shape(args.box, RES)
cell_flat = np.where(
    in_R, ((lat - LAT_MIN) // RES).astype(int) * n_lon + ((lon - LON_MIN) // RES).astype(int), -1
)
start, end, stranded = move_indices(
    traj_idx, t, in_R, DIRECT_TAU, strand_times(ds, traj_idx, t, in_R)
)

rng = np.random.default_rng(SEED)
noise = {}
print()
for name, flag in DROGUE_TYPES.items():
    op_direct = operators[name][1]
    cells = op_direct["cell_flat"]
    n_cells = len(cells)
    n_states = op_direct["P"].shape[0]
    keep = same_drogue(drogue, start, end, stranded, flag)
    s, e = start[keep], end[keep]
    f, to = transition_states(cell_flat, cells, s, e, lon, lat, args.box, stranded[keep])
    counts = drifter_counts(traj_idx[s], f, to, n_states)
    C = np.bincount(counts[1], weights=counts[2], minlength=n_states**2).reshape(n_states, n_states)
    if not np.array_equal(C, op_direct["C"]):
        raise SystemExit(
            f"Rebuilt {DIRECT_TAU:g}-day counts differ from the saved {name} matrix. "
            f"Rebuild it with:\n  python scripts/03_transition_matrix.py "
            f"--res {RES:g} --tau {DIRECT_TAU:g}"
        )

    rows = C[:n_cells]
    total = rows.sum(axis=1)
    P_rows = rows / np.maximum(total, 1)[:, None]
    boot_tvd = np.full((N_BOOT, n_cells), np.nan)
    for b in range(N_BOOT):
        Cb = resampled_counts(counts, n_states, rng)[:n_cells]
        tb = Cb.sum(axis=1)
        ok = (tb > 0) & (total > 0)  # a resample can miss every drifter of a row
        boot_tvd[b, ok] = 0.5 * np.abs(Cb[ok] / tb[ok, None] - P_rows[ok]).sum(axis=1)

    with np.errstate(all="ignore"):
        noise[name] = {
            "median": np.nanmedian(boot_tvd, axis=0),
            "p95": np.nanpercentile(boot_tvd, 95, axis=0),
        }

    print(
        f"{name}: {counts[3]:,} drifters resampled {N_BOOT} times "
        f"for the {DIRECT_TAU:g}-day rows"
    )


# ── 4. Chapman-Kolmogorov comparison ─────────────────────────────────────────
cell_rows = []
summary_rows = []
plot_data = {}

for name in DROGUE_TYPES:
    op_tau, op_direct = operators[name]
    P_tau = op_tau["P"]
    P_direct = op_direct["P"]
    n_cells = len(op_tau["cell_flat"])

    # Chapman-Kolmogorov prediction:
    # two tau steps should approximate one directly estimated 2*tau step.
    P_two_step = P_tau @ P_tau
    reliable_tau = reliable(op_tau)
    reliable_direct = reliable(op_direct)
    common = reliable_tau & reliable_direct

    if common.sum() == 0:
        raise SystemExit(f"No common reliable rows for {name}.")

    # Full row-wise TVD, including ocean and all absorbing states.
    tvd = 0.5 * np.abs(P_two_step[:n_cells] - P_direct[:n_cells]).sum(axis=1)

    # Difference in probability of remaining inside R.
    in_R_two_step = P_two_step[:n_cells, :n_cells].sum(axis=1)
    in_R_direct = P_direct[:n_cells, :n_cells].sum(axis=1)
    absdiff_in_R = np.abs(in_R_two_step - in_R_direct)

    # Differences across absorbing fates.
    fate_two_step = P_two_step[:n_cells, n_cells:]
    fate_direct = P_direct[:n_cells, n_cells:]
    fate_absdiff = np.abs(fate_two_step - fate_direct)
    mean_fate_absdiff = fate_absdiff.mean(axis=1)

    # Sampling noise of each direct 2*tau row (section 3).
    noise_median = noise[name]["median"]
    noise_p95 = noise[name]["p95"]
    above_noise = tvd > noise_p95
    excess = tvd - noise_median
    cell_lon, cell_lat = cell_centres(op_tau)
    values = tvd[common]

    # Descriptive shares of reliable rows below convenient TVD thresholds.
    # These are not formal pass/fail thresholds.
    share_below_025 = float(np.mean(values < 0.25))
    share_below_050 = float(np.mean(values < 0.50))
    print("\n----------------------------------------")
    print(name.upper())
    print("----------------------------------------")
    print(f"Ocean cells              : {n_cells}")

    print(
        f"Reliable at tau          : "
        f"{reliable_tau.sum()} / "
        f"{n_cells} "
        f"({reliable_tau.mean():.1%})"
    )

    print(
        f"Reliable at 2*tau        : "
        f"{reliable_direct.sum()} / "
        f"{n_cells} "
        f"({reliable_direct.mean():.1%})"
    )

    print(f"Common reliable rows     : {common.sum()} / {n_cells} ({common.mean():.1%})")

    print(f"Median TVD               : {np.median(values):.3f}")
    print(f"Mean TVD                 : {np.mean(values):.3f}")
    print(f"90th percentile TVD      : {np.percentile(values, 90):.3f}")
    print(f"Max TVD                  : {np.max(values):.3f}")
    print(f"Rows with TVD < 0.25     : {share_below_025:.1%}")
    print(f"Rows with TVD < 0.50     : {share_below_050:.1%}")
    print(f"Median noise TVD (2*tau) : {np.median(noise_median[common]):.3f}")
    print(f"Median TVD above noise   : {np.median(excess[common]):.3f}")
    print(f"Rows above noise (95%)   : {above_noise[common].mean():.1%}")
    print(f"Median |Δ in-R prob|     : {np.median(absdiff_in_R[common]):.3f}")
    print(f"Median fate MAE          : {np.median(mean_fate_absdiff[common]):.3f}")
    exit_labels = op_tau["exit_labels"].astype(str)

    for k, label in enumerate(exit_labels):
        print(f"Median |Δ {label:8s}|       : {np.median(fate_absdiff[common, k]):.3f}")

    summary = {
        "drogue": name,
        "res": RES,
        "tau_days": TAU,
        "direct_tau_days": DIRECT_TAU,
        "n_cells": n_cells,
        "reliable_tau": int(reliable_tau.sum()),
        "reliable_2tau": int(reliable_direct.sum()),
        "common_reliable": int(common.sum()),
        "common_reliable_share": float(common.mean()),
        "mean_tvd": float(np.mean(values)),
        "median_tvd": float(np.median(values)),
        "p90_tvd": float(np.percentile(values, 90)),
        "max_tvd": float(np.max(values)),
        "share_tvd_below_0p25": share_below_025,
        "share_tvd_below_0p50": share_below_050,
        "median_noise_tvd": float(np.median(noise_median[common])),
        "median_excess_tvd": float(np.median(excess[common])),
        "share_above_noise_p95": float(above_noise[common].mean()),
        "median_absdiff_in_R": float(np.median(absdiff_in_R[common])),
        "median_fate_mae": float(np.median(mean_fate_absdiff[common])),
    }

    for k, label in enumerate(exit_labels):
        summary[f"median_absdiff_{label}"] = float(np.median(fate_absdiff[common, k]))

    summary_rows.append(summary)

    for i in range(n_cells):
        row = {
            "drogue": name,
            "state": i,
            "cell_flat": int(op_tau["cell_flat"][i]),
            "lon": float(cell_lon[i]),
            "lat": float(cell_lat[i]),
            "reliable_tau": bool(reliable_tau[i]),
            "reliable_2tau": bool(reliable_direct[i]),
            "common_reliable": bool(common[i]),
            "tvd": float(tvd[i]),
            "noise_tvd_median": float(noise_median[i]),
            "noise_tvd_p95": float(noise_p95[i]),
            "above_noise": bool(above_noise[i]),
            "in_R_two_step": float(in_R_two_step[i]),
            "in_R_direct": float(in_R_direct[i]),
            "absdiff_in_R": float(absdiff_in_R[i]),
            "mean_fate_absdiff": float(mean_fate_absdiff[i]),
            "row_drifters_tau": int(op_tau["row_drifters"][i]),
            "row_drifters_2tau": int(op_direct["row_drifters"][i]),
        }

        for k, label in enumerate(exit_labels):
            row[f"two_step_{label}"] = float(fate_two_step[i, k])
            row[f"direct_{label}"] = float(fate_direct[i, k])
            row[f"absdiff_{label}"] = float(fate_absdiff[i, k])

        cell_rows.append(row)

    plot_data[name] = {"op": op_tau, "tvd": tvd, "common": common, "noise_p95": noise_p95}


# ── 5. Save tables ────────────────────────────────────────────────────────────
cell_table = pd.DataFrame(cell_rows)
summary_table = pd.DataFrame(summary_rows)
cell_table.to_csv(CELL_FILE, index=False)
summary_table.to_csv(SUMMARY_FILE, index=False)
print(f"\nCell results saved → {CELL_FILE}")
print(f"Summary saved      → {SUMMARY_FILE}")


# ── 6. TVD maps ───────────────────────────────────────────────────────────────
all_plot_values = np.concatenate([d["tvd"][d["common"]] for d in plot_data.values()])

# Use the 95th percentile so a handful of extreme cells do not dominate
# the colour scale.
vmax = float(np.percentile(all_plot_values, 95))

if vmax <= 0:
    vmax = 1.0


fig, axes = plt.subplots(1, len(DROGUE_TYPES), figsize=(13, 5), sharex=True, sharey=True)

if len(DROGUE_TYPES) == 1:
    axes = [axes]


last_image = None

for ax, name in zip(axes, DROGUE_TYPES):
    d = plot_data[name]
    op = d["op"]
    tvd = d["tvd"]
    common = d["common"]
    n_lon = len(op["lon_edges"]) - 1
    n_lat = len(op["lat_edges"]) - 1
    grid = np.full(n_lat * n_lon, np.nan)
    grid[op["cell_flat"][common]] = tvd[common]
    grid = grid.reshape(n_lat, n_lon)

    last_image = ax.imshow(
        grid,
        origin="lower",
        extent=[op["lon_edges"][0], op["lon_edges"][-1], op["lat_edges"][0], op["lat_edges"][-1]],
        aspect="auto",
        vmin=0,
        vmax=vmax,
    )

    values = tvd[common]
    ax.set_title(f"{name}\nmedian TVD = {np.median(values):.3f}, n = {common.sum()}")
    ax.set_xlabel("Longitude")

axes[0].set_ylabel("Latitude")

fig.colorbar(
    last_image, ax=axes, label=("Row-wise TVD: " "P(tau)^2 vs direct P(2 tau)"), shrink=0.85
)

fig.suptitle(f"Chapman-Kolmogorov test — {RES:g}°, tau = {TAU:g} d")
fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.90, wspace=0.12)
plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved       → {FIGURE_FILE}")


# ── 7. TVD against sampling noise ────────────────────────────────────────────
fig, axes = plt.subplots(1, len(DROGUE_TYPES), figsize=(11, 5), sharex=True, sharey=True)

for ax, name in zip(axes, DROGUE_TYPES):
    d = plot_data[name]
    common = d["common"]
    x = d["noise_p95"][common]
    y = d["tvd"][common]
    above = y > x
    ax.scatter(x[~above], y[~above], s=8, alpha=0.5, color="tab:grey", label="within noise")
    ax.scatter(x[above], y[above], s=8, alpha=0.6, color="tab:red", label="above noise")
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="k", lw=1, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"{name}: {above.mean():.0%} of {common.sum()} rows above noise")
    ax.set_xlabel(f"95th percentile bootstrap TVD of the direct {DIRECT_TAU:g}-day row")
    ax.legend(frameon=False, loc="upper left")

axes[0].set_ylabel(f"TVD: P({TAU:g} d)^2 vs direct P({DIRECT_TAU:g} d)")
fig.suptitle(f"Chapman-Kolmogorov TVD against sampling noise — {RES:g}°, tau = {TAU:g} d")
fig.tight_layout()
NOISE_FIGURE_FILE = f"figures/agulhas_chapman_kolmogorov_noise_{TAG}.png"
plt.savefig(NOISE_FIGURE_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved       → {NOISE_FIGURE_FILE}")


# ── 8. Interpretation reminder ────────────────────────────────────────────────
print("\n========================================")
print("DONE")
print("========================================")
print("\nInterpretation:")
print("TVD = 0 means exact Chapman-Kolmogorov agreement.")

print(
    "Larger TVD means the directly observed 2*tau transition "
    "distribution differs more from two applications of P(tau)."
)

print(
    "The TVD < 0.25 and TVD < 0.50 shares are descriptive summaries, "
    "not formal pass/fail thresholds."
)

print(
    "Rows above noise have a TVD larger than the 95th percentile of their "
    "direct row's bootstrap TVD, so sampling alone is unlikely to explain it. "
    "The noise level ignores the smaller noise in P(tau)^2 and is not "
    "corrected for the number of rows tested."
)
