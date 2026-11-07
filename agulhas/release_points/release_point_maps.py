"""
Agulhas Region — Release-Point Maps
Where material released at each candidate point from shipping_overlay is after about one
week, one month and one year, under the drogued and the undrogued matrix:
one figure per point (rows: drogue type; columns: 7, 30, 365 days), each
panel giving the share still afloat in R and the share that has left across
each edge or stranded, with the 95% interval from the drifter bootstrap.
Times are rounded to whole τ steps (1 week = 2, 1 month ≈ 9, 1 year ≈ 104
steps at τ = 3.5 d), and exits are read through exit_labels.

Needs the matrices from transition_matrix and the intervals from bootstrap (same
box, grid and lag), and the release points from shipping_overlay (same box,
default grid and lag).

    python -m agulhas release_point_maps --res 1 --tau 3.5
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from agulhas.config import (DROGUE_TYPES, operator_path, parse_args, place_label, region_prefix, release_points_path,
                            setting_tag)
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import load_operator, release, to_grid

DAYS = [7, 30, 365]


def load_intervals(tag, res, tau):
    path = f"data/bootstrap_release_{tag}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run bootstrap with --res {res:g} --tau {tau:g} first.")
    return pd.read_csv(path).set_index(["name", "drogue", "metric", "days"])


def propagate_points(points, ops, boot, tau):
    """Every point under every matrix and horizon: (long table with intervals, distributions).
    Stops if an estimate differs from the one the intervals were computed around."""
    labels = [str(x) for x in ops["drogued"]["exit_labels"]]
    n_cells = len(ops["drogued"]["cell_flat"])
    steps = {d: int(round(d / tau)) for d in DAYS}
    powers = {name: {d: np.linalg.matrix_power(op["P"], steps[d]) for d in DAYS} for name, op in ops.items()}
    rows, dists = [], {}
    for pt in points.itertuples():
        for name, op in ops.items():
            p0 = release(op, pt.lon, pt.lat)
            for d in DAYS:
                p = p0 @ powers[name][d]
                dists[(pt.name, name, d)] = p
                fate = {"in_R": p[:n_cells].sum(), **{f"exit_{lab}": p[n_cells + k] for k, lab in enumerate(labels)}}
                for metric, value in fate.items():
                    b = boot.loc[(pt.name, name, metric, d)]
                    if abs(b.estimate - value) > 1e-6:
                        raise SystemExit(f"{pt.name} {name} {metric} {d} d differs from bootstrap: rerun it")
                    rows.append({"name": pt.name, "kind": pt.kind, "lon": pt.lon, "lat": pt.lat, "drogue": name,
                                 "days": d, "model_days": steps[d] * tau, "metric": metric, "estimate": value,
                                 "ci_low": b.ci_low, "ci_high": b.ci_high})
    return pd.DataFrame(rows), dists


def formatted(table, name, drogue, metric, d):
    """'estimate (low–high)' for one point, matrix, metric and horizon."""
    r = table[(table.name == name) & (table.drogue == drogue) & (table.metric == metric) & (table.days == d)].iloc[0]
    return f"{r.estimate:.0%} ({r.ci_low:.0%}–{r.ci_high:.0%})"


def print_one_year(table, points, labels, res, tau):
    print(f"Release points after 1 year, estimate (95% interval), {res:g}°, τ = {tau:g} d")
    metrics = ["in_R"] + [f"exit_{lab}" for lab in labels]
    print(f"{'':<28}{'type':<11}" + "".join(f"{m.replace('exit_', ''):>16}" for m in metrics))
    for pt in points.itertuples():
        for name in DROGUE_TYPES:
            print(f"{pt.name[:27]:<28}{name:<11}"
                  + "".join(f"{formatted(table, pt.name, name, m, 365):>16}" for m in metrics))


def plot_point(pt, ops, dists, table, labels, args, path):
    """One figure for one point: rows are drogue types, columns are horizons."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.2), subplot_kw=MAP_KW)
    for row, name in enumerate(DROGUE_TYPES):
        op = ops[name]
        for col, d in enumerate(DAYS):
            ax = axes[row, col]
            grid = to_grid(op, dists[(pt.name, name, d)])
            mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], np.where(grid > 0, grid, np.nan),
                                 cmap="Blues", norm=LogNorm(1e-4, 1), **TRANSFORM_KW)
            ax.plot(pt.lon, pt.lat, marker="*", ms=12, color="#e8710a", mec="#222222", **TRANSFORM_KW)
            map_axes(ax, args.box)
            ax.set_title(f"{name.capitalize()}, {d} days: afloat in R {formatted(table, pt.name, name, 'in_R', d)}",
                         fontsize=9)
            text = "\n".join(f"{lab} {formatted(table, pt.name, name, f'exit_{lab}', d)}" for lab in labels)
            ax.text(0.01, 0.02, text, transform=ax.transAxes, fontsize=7, va="bottom", zorder=5,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="min")
    cb.set_label(f"Probability per {args.res:g}° cell (log)")
    fig.suptitle(f"Release at {pt.name} ({place_label(pt.lon, pt.lat)}) — {args.res:g}°, τ = {args.tau:g} d; "
                 f"95% drifter-bootstrap intervals", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    prefix = region_prefix(args.data)
    out_csv = f"data/release_point_maps_{tag}.csv"

    points = pd.read_csv(release_points_path(args.box))
    ops = {name: load_operator(operator_path(name, args.res, args.tau, args.box)) for name in DROGUE_TYPES}
    labels = [str(x) for x in ops["drogued"]["exit_labels"]]
    table, dists = propagate_points(points, ops, load_intervals(tag, args.res, args.tau), args.tau)
    table.to_csv(out_csv, index=False)
    print_one_year(table, points, labels, args.res, args.tau)

    for i, pt in enumerate(points.itertuples(), start=1):
        slug = re.sub(r"[^a-z0-9]+", "_", pt.name.lower()).strip("_")
        plot_point(pt, ops, dists, table, labels, args, f"figures/{prefix}_release_{i:02d}_{slug}_{tag}.png")
    print(f"\nSaved → {out_csv}\nSaved → {len(points)} figures, figures/{prefix}_release_*_{tag}.png")


if __name__ == "__main__":
    main()
