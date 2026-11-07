"""
Agulhas Region — Drifter Bootstrap
Puts confidence intervals on the reported numbers by resampling whole
drifters (not observations, which are strongly correlated along a track).
For each drogue type, drifters are drawn with replacement, the transition
matrix is rebuilt from their transitions, and the release points from shipping_overlay are
propagated again. Repeated N_BOOT times, this gives:

  • 95% intervals on each release point's fate: share still in R, exited
    W / E / S / N and stranded after 7, 30 and 365 days, and 30-day
    coastal exposure
  • whether drogued and undrogued differ at each point (interval of the
    difference excludes 0)
  • row stability: median total-variation distance (TVD) between bootstrap
    rows and the full-data row, as a data-driven check of the 10-drifter flag

Needs the matrices from transition_matrix (same box, grid and lag) and the release points
from shipping_overlay (same box, default grid and lag).

    python -m agulhas bootstrap --res 1 --tau 3.5
"""

from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from agulhas.config import (DROGUE_TYPES, MIN_DRIFTERS, operator_path, parse_args, region_prefix,
                            release_points_path, setting_tag)
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import DROGUE_COLORS, MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import coastal_states, load_operator, resampled_counts, row_normalise, state_of

N_BOOT = 1000
SEED = 0
HORIZONS = [7, 30, 365]          # days
UNSTABLE_TVD = 0.2               # rows whose median bootstrap TVD exceeds this are unstable
YEAR_METRICS = [("in_R", "Still in R"), ("exit_W", "Exited west (Atlantic)"), ("exit_E", "Exited east"),
                ("exit_stranded", "Stranded")]


# ── Fates of the release points ──────────────────────────────────────────────

def fates(P, starts, coastal, exit_labels, tau):
    """Fate of each release point: {(metric, days): values}, one per point."""
    n_cells = len(coastal)
    steps = {d: int(round(d / tau)) for d in HORIZONS}
    p = np.zeros((len(starts), P.shape[0]))
    p[np.arange(len(starts)), starts] = 1.0
    out, exposure = {}, 0.0
    for step in range(1, steps[max(HORIZONS)] + 1):
        p = p @ P
        if step <= steps[30]:
            exposure = exposure + p[:, :n_cells][:, coastal].sum(axis=1) * tau
        for days in HORIZONS:
            if step == steps[days]:
                out[("in_R", days)] = p[:, :n_cells].sum(axis=1)
                for k, lab in enumerate(exit_labels):
                    out[(f"exit_{lab}", days)] = p[:, n_cells + k]
    out[("coastal_exposure_days", 30)] = exposure
    return out


# ── Bootstrap by drifter ─────────────────────────────────────────────────────

def resample(op, counts, fates_of, n_lon, rng):
    """Release-point fates on the full data and on N_BOOT resamples of the drifters,
    and each row's median TVD from its full-data row."""
    P, n_cells, n_states = op["P"], len(op["cell_flat"]), op["P"].shape[0]
    estimate = fates_of(P)
    n_points = len(next(iter(estimate.values())))
    samples = {k: np.empty((N_BOOT, n_points)) for k in estimate}
    tvd = np.empty((N_BOOT, n_cells))
    for b in range(N_BOOT):
        Pb, _ = row_normalise(resampled_counts(counts, n_states, rng), op["cell_flat"], n_lon)
        tvd[b] = 0.5 * np.abs(Pb[:n_cells] - P[:n_cells]).sum(axis=1)
        for k, v in fates_of(Pb).items():
            samples[k][b] = v
    return estimate, samples, np.median(tvd, axis=0)


def run_bootstrap(ops, moves, cand, tau):
    op0 = ops["drogued"]   # both share the same ocean cells
    starts = np.array([state_of(op0, r.lon, r.lat) for r in cand.itertuples()])
    fates_of = partial(fates, starts=starts, coastal=coastal_states(op0), exit_labels=op0["exit_labels"], tau=tau)
    rng = np.random.default_rng(SEED)
    estimate, boots, row_tvd = {}, {}, {}
    for name, flag in DROGUE_TYPES.items():
        counts = moves.counts_for(ops[name], name, flag)
        estimate[name], boots[name], row_tvd[name] = resample(ops[name], counts, fates_of, moves.grid.n_lon, rng)
        print(f"{name}: {counts[3]:,} drifters resampled {N_BOOT} times")
    return estimate, boots, row_tvd


# ── Release-point intervals ──────────────────────────────────────────────────

def interval_table(estimate, boots, cand):
    rows = []
    for name in DROGUE_TYPES:
        for (metric, days), est in estimate[name].items():
            lo, hi = np.percentile(boots[name][(metric, days)], [2.5, 97.5], axis=0)
            for i, r in enumerate(cand.itertuples()):
                rows.append({"name": r.name, "kind": r.kind, "lon": r.lon, "lat": r.lat,
                             "drogue": name, "metric": metric, "days": days,
                             "estimate": est[i], "ci_low": lo[i], "ci_high": hi[i]})
    return pd.DataFrame(rows)


def difference_table(estimate, boots, cand):
    """Drogued − undrogued: the two resamples are independent, so pair them by index."""
    rows = []
    for key in estimate["drogued"]:
        diff = boots["drogued"][key] - boots["undrogued"][key]
        lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
        for i, r in enumerate(cand.itertuples()):
            rows.append({"name": r.name, "metric": key[0], "days": key[1],
                         "diff": estimate["drogued"][key][i] - estimate["undrogued"][key][i],
                         "ci_low": lo[i], "ci_high": hi[i], "significant": lo[i] > 0 or hi[i] < 0})
    return pd.DataFrame(rows)


def print_release_fates(table, diffs, cand, args):
    def fmt(name, metric, days, pct=True):
        sel = table[(table.drogue == name) & (table.metric == metric) & (table.days == days)].set_index("name")
        f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.1f}")
        return sel.apply(lambda r: f"{f(r.estimate)} ({f(r.ci_low)}–{f(r.ci_high)})", axis=1)

    print(f"\nRelease points after 1 year, estimate (95% interval), {args.res:g}°, τ = {args.tau:g} d")
    for metric, label in YEAR_METRICS:
        show = pd.DataFrame({d: fmt(d, metric, 365) for d in DROGUE_TYPES})
        sig = diffs[(diffs.metric == metric) & (diffs.days == 365)].set_index("name")["significant"]
        show["differ"] = sig.map({True: "yes", False: "no"})
        print(f"\n{label}:\n{show.loc[cand.name].to_string()}")

    show = pd.DataFrame({d: fmt(d, "coastal_exposure_days", 30, pct=False) for d in DROGUE_TYPES})
    sig = diffs[diffs.metric == "coastal_exposure_days"].set_index("name")["significant"]
    show["differ"] = sig.map({True: "yes", False: "no"})
    print(f"\nCoastal exposure over 30 days (material-days):\n{show.loc[cand.name].to_string()}")


# ── Row stability against drifter count ──────────────────────────────────────

def print_row_stability(ops, row_tvd):
    print(f"\nRow stability: median bootstrap TVD per row (unstable if > {UNSTABLE_TVD})")
    bins = [0, MIN_DRIFTERS, 20, 50, np.inf]
    labels = [f"< {MIN_DRIFTERS}", f"{MIN_DRIFTERS}–19", "20–49", "≥ 50"]
    for name in DROGUE_TYPES:
        op = ops[name]
        ok = ~op["empty"]
        grp = pd.cut(op["row_drifters"][ok], bins, right=False, labels=labels)
        df = pd.DataFrame({"drifters": grp, "tvd": row_tvd[name][ok]})
        summary = df.groupby("drifters", observed=False)["tvd"].agg(
            rows="size", median_tvd="median", unstable=lambda x: (x > UNSTABLE_TVD).mean())
        unstable = row_tvd[name][ok] > UNSTABLE_TVD
        flagged = op["flagged"][ok]
        print(f"\n{name}: {unstable.sum()} of {ok.sum()} rows unstable; "
              f"{(unstable & ~flagged).sum()} of them pass the {MIN_DRIFTERS}-drifter rule, "
              f"{(flagged & ~unstable).sum()} flagged rows are stable")
        print(summary.to_string(formatters={"median_tvd": "{:.2f}".format, "unstable": "{:.0%}".format}))


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_row_stability(ops, row_tvd, cand, args, path):
    """Row stability map (sequential blue, grey = no outgoing data)."""
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    cmap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), subplot_kw=MAP_KW, constrained_layout=True)
    for ax, name in zip(axes, DROGUE_TYPES):
        op = ops[name]
        n_lon, n_lat = len(op["lon_edges"]) - 1, len(op["lat_edges"]) - 1
        grid = np.full(n_lat * n_lon, np.nan)
        grid[op["cell_flat"]] = np.where(op["empty"], np.nan, row_tvd[name])
        mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], grid.reshape(n_lat, n_lon),
                             cmap=cmap, vmin=0, vmax=0.5, **TRANSFORM_KW)
        ax.scatter(cand.lon, cand.lat, s=40, c="white", edgecolor="#222222", linewidth=1, zorder=5, **TRANSFORM_KW)
        map_axes(ax, args.box, land="#f0f0f0", gridlines=True)
        unstable = (row_tvd[name] > UNSTABLE_TVD) & ~op["empty"]
        ax.set_title(f"{name.capitalize()} — {unstable.mean():.0%} of rows with median TVD > {UNSTABLE_TVD}",
                     fontsize=11)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="max")
    cb.set_label("Median bootstrap TVD of the row (0 = stable; grey = no data)")
    fig.suptitle(f"Row stability under drifter resampling ({N_BOOT} bootstraps, {args.res:g}°, τ = {args.tau:g} d); "
                 "dots = release points", fontsize=12)
    save_figure(path)


def plot_year_fates(table, cand, args, path):
    """One-year fate per release point with 95% intervals."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.42 * len(cand) + 1.8), sharey=True, constrained_layout=True)
    y = np.arange(len(cand))[::-1]
    offset = {"drogued": 0.15, "undrogued": -0.15}
    for ax, (metric, label) in zip(axes, YEAR_METRICS):
        for name in DROGUE_TYPES:
            sel = table[(table.drogue == name) & (table.metric == metric) & (table.days == 365)]
            sel = sel.set_index("name").loc[cand.name]
            ax.errorbar(sel.estimate * 100, y + offset[name],
                        xerr=[(sel.estimate - sel.ci_low) * 100, (sel.ci_high - sel.estimate) * 100],
                        fmt="o", ms=6, color=DROGUE_COLORS[name], elinewidth=2, capsize=0, label=name.capitalize())
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("% of released material after 1 year")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cand.name)
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle(f"One-year fate of each release point, 95% drifter-bootstrap intervals "
                 f"({args.res:g}°, τ = {args.tau:g} d)", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    prefix = region_prefix(args.data)
    out_csv = f"data/bootstrap_release_{tag}.csv"
    out_diff = f"data/bootstrap_difference_{tag}.csv"
    out_rows = f"data/bootstrap_rows_{tag}.npz"
    out_map = f"figures/{prefix}_bootstrap_rows_{tag}.png"
    out_fate = f"figures/{prefix}_bootstrap_fates_{tag}.png"

    moves = MoveSet(Drifters(args.data), args.box, args.res, args.tau)
    ops = {name: load_operator(operator_path(name, args.res, args.tau, args.box)) for name in DROGUE_TYPES}
    cand = pd.read_csv(release_points_path(args.box))
    estimate, boots, row_tvd = run_bootstrap(ops, moves, cand, args.tau)

    table, diffs = interval_table(estimate, boots, cand), difference_table(estimate, boots, cand)
    table.to_csv(out_csv, index=False)
    diffs.to_csv(out_diff, index=False)
    print_release_fates(table, diffs, cand, args)
    print_row_stability(ops, row_tvd)
    np.savez_compressed(out_rows, cell_flat=ops["drogued"]["cell_flat"], n_boot=N_BOOT, unstable_tvd=UNSTABLE_TVD,
                        **{f"row_tvd_{n}": v for n, v in row_tvd.items()})

    plot_row_stability(ops, row_tvd, cand, args, out_map)
    plot_year_fates(table, cand, args, out_fate)
    print(f"\nIntervals saved   → {out_csv}")
    print(f"Differences saved → {out_diff}")
    print(f"Row TVD saved     → {out_rows}")
    print(f"Maps saved        → {out_map}, {out_fate}")


if __name__ == "__main__":
    main()
