"""
Agulhas Region — Calendar-Ordered Seasonal Forecast
Where material from each release point is after one year when every τ step uses
the matrix of the season it falls in, released in each of the twelve months,
against the annual matrix. A step takes the season of its start, as in
seasonal_matrices: a release on 1 July runs through JJA, then SON, DJF and MAM.
Where a seasonal row is flagged (fewer than MIN_DRIFTERS drifters) or empty, the
step uses the annual row, so a thin seasonal matrix adds no noise; the share of
such rows is printed per season.

Each estimate is compared with the 95% drifter-bootstrap interval of the annual
estimate. A calendar-ordered estimate inside that interval is no evidence of a
seasonal effect, and the seasonal matrices' own sampling noise is not in these
point estimates; the permutation test in seasonal_matrices says whether
seasonality exceeds noise.

Needs the annual matrices from transition_matrix, the seasonal ones from
seasonal_matrices (same box, grid and lag), the release points from
shipping_overlay and the intervals from bootstrap.

    python -m agulhas calendar_forecast --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import (DROGUE_TYPES, operator_path, parse_args, region_prefix, release_points_path,
                            setting_tag)
from agulhas.operators.seasonal_matrices import SEASONS
from agulhas.plotting import DROGUE_COLORS, save_figure
from agulhas.release_points.release_point_maps import load_intervals
from agulhas.transport import load_operator, release

DAYS = 365                        # the horizon, as in release_point_maps
DAYS_PER_MONTH = 365.25 / 12      # a step falls in the month its start day falls in
MONTHS = range(1, 13)             # release months
PLOTTED = ["in_R", "exit_E", "exit_stranded"]   # fates drawn in the figure


# ── 1. The season of each step ───────────────────────────────────────────────

def season_of(month):
    """DJF, MAM, JJA or SON for a calendar month (1-12)."""
    return next(season for season, months in SEASONS.items() if month in months)


def season_sequence(start_month, days, tau):
    """The season of each τ step of a release in start_month; a step takes the
    season of the month its start day falls in."""
    steps = int(round(days / tau))
    return [season_of((start_month - 1 + int(k * tau // DAYS_PER_MONTH)) % 12 + 1) for k in range(steps)]


# ── 2. Matrices and forecasts ────────────────────────────────────────────────

def blend(annual, seasonal):
    """The seasonal matrix with the annual row wherever the seasonal row is
    flagged or empty: (matrix, share of cell rows replaced)."""
    n = len(annual["cell_flat"])
    if seasonal["P"].shape != annual["P"].shape or not np.array_equal(seasonal["cell_flat"], annual["cell_flat"]):
        raise SystemExit("A seasonal matrix does not share the annual matrix's states: rerun seasonal_matrices.")
    P = seasonal["P"].copy()
    weak = seasonal["flagged"].astype(bool) | seasonal["empty"].astype(bool)
    P[:n][weak] = annual["P"][:n][weak]
    return P, float(weak.mean())


def propagate_calendar(p0, mats, sequence):
    """Push p0 through the matrix of each step's season, in order."""
    p = p0
    for season in sequence:
        p = p @ mats[season]
    return p


def fate_shares(p, labels, n_cells):
    """Share still afloat in R and the share in each absorbing state."""
    return {"in_R": p[:n_cells].sum(), **{f"exit_{lab}": p[n_cells + k] for k, lab in enumerate(labels)}}


def load_seasonal(args):
    """{drogue type: {season: operator}} from seasonal_matrices."""
    ops = {}
    for name in DROGUE_TYPES:
        ops[name] = {}
        for season in SEASONS:
            path = operator_path(f"{name}_{season}", args.res, args.tau, args.box)
            try:
                ops[name][season] = load_operator(path)
            except FileNotFoundError:
                raise SystemExit(f"{path} not found: run seasonal_matrices with --res {args.res:g} "
                                 f"--tau {args.tau:g} first.")
    return ops


def forecast_table(points, annual, seasonal, boot, tau):
    """Long table: every point, drogue type, release month and fate, with the
    annual estimate and its interval. Stops if the annual estimate differs from
    the one the bootstrap was computed around. Also returns the share of rows
    replaced by the annual row, per (type, season)."""
    labels = [str(x) for x in annual["drogued"]["exit_labels"]]
    n_cells = len(annual["drogued"]["cell_flat"])
    steps = int(round(DAYS / tau))
    rows, weak = [], {}
    for name, op in annual.items():
        blended = {s: blend(op, seasonal[name][s]) for s in SEASONS}
        mats = {s: m for s, (m, _) in blended.items()}
        weak.update({(name, s): share for s, (_, share) in blended.items()})
        reference = np.linalg.matrix_power(op["P"], steps)
        for pt in points.itertuples():
            p0 = release(op, pt.lon, pt.lat)
            annual_share = fate_shares(p0 @ reference, labels, n_cells)
            for month in MONTHS:
                p = propagate_calendar(p0, mats, season_sequence(month, DAYS, tau))
                for metric, value in fate_shares(p, labels, n_cells).items():
                    b = boot.loc[(pt.name, name, metric, DAYS)]
                    if abs(b.estimate - annual_share[metric]) > 1e-6:
                        raise SystemExit(f"{pt.name} {name} {metric} differs from bootstrap: rerun it")
                    rows.append({"name": pt.name, "kind": pt.kind, "drogue": name, "start_month": month,
                                 "first_season": season_of(month), "metric": metric, "estimate": value,
                                 "annual": annual_share[metric], "ci_low": b.ci_low, "ci_high": b.ci_high,
                                 "outside": bool(value < b.ci_low or value > b.ci_high)})
    return pd.DataFrame(rows), weak


# ── 3. Summary and figure ────────────────────────────────────────────────────

def print_summary(table, weak):
    print(f"Calendar-ordered forecast after {DAYS} days against the annual matrix "
          f"(release in each month; a step takes the season of its start)")
    for name, d in table.groupby("drogue"):
        gap = (d.estimate - d.annual).abs()
        worst = d.loc[gap.idxmax()]
        print(f"  {name:<10} {d.outside.mean():.1%} of estimates outside the annual 95% interval; "
              f"median departure {gap.median() * 100:.1f} points, largest {gap.max() * 100:.1f} "
              f"({worst['name']}, {worst.metric}, release month {worst.start_month})")
    print("Seasonal rows replaced by the annual row (flagged or empty):")
    for name in DROGUE_TYPES:
        print(f"  {name:<10}" + "  ".join(f"{s} {weak[(name, s)]:.0%}" for s in SEASONS))


def plot_differences(table, path):
    """Calendar minus annual estimate by release month, one line per point; the
    band is the median annual 95% half-width. Rows: drogue type; columns: fate."""
    fig, axes = plt.subplots(len(DROGUE_TYPES), len(PLOTTED), figsize=(14, 7), sharex=True)
    for row, name in enumerate(DROGUE_TYPES):
        for col, metric in enumerate(PLOTTED):
            ax = axes[row, col]
            d = table[(table.drogue == name) & (table.metric == metric)].copy()
            if d.empty:
                ax.set_visible(False)
                continue
            d["diff"] = (d.estimate - d.annual) * 100
            for _, g in d.groupby("name"):
                ax.plot(g.start_month, g["diff"], color="#9aa0a6", lw=0.8)
            median = d.groupby("start_month")["diff"].median()
            ax.plot(median.index, median.values, color=DROGUE_COLORS[name], lw=2.2)
            half = ((d.ci_high - d.ci_low) / 2 * 100).median()
            ax.axhspan(-half, half, color=DROGUE_COLORS[name], alpha=0.15)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{name.capitalize()}: {metric.replace('exit_', 'exit ')}", fontsize=10)
            ax.set_xticks(list(MONTHS))
            if col == 0:
                ax.set_ylabel("Calendar − annual (points)")
            if row == len(DROGUE_TYPES) - 1:
                ax.set_xlabel("Release month")
    fig.suptitle(f"Calendar-ordered seasonal forecast after {DAYS} days against the annual matrix "
                 f"(grey: release points; line: median; band: median annual 95% half-width)", fontsize=11)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_csv = f"data/calendar_forecast_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_calendar_forecast_{tag}.png"

    points = pd.read_csv(release_points_path(args.box))
    annual = {name: load_operator(operator_path(name, args.res, args.tau, args.box)) for name in DROGUE_TYPES}
    table, weak = forecast_table(points, annual, load_seasonal(args), load_intervals(tag, args.res, args.tau),
                                 args.tau)
    table.to_csv(out_csv, index=False)
    print_summary(table, weak)
    plot_differences(table, out_png)
    print(f"Saved → {out_csv}, {out_png}")


if __name__ == "__main__":
    main()
