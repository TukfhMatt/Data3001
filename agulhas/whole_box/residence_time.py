"""
Agulhas Region — Residence Time in R
Expected time material released in each cell stays afloat in R before it
leaves across an edge or strands, for every grid and lag and both drogue
types. It comes exactly from the absorbing chain (transport.absorption):
N·1 is the expected number of τ steps up to and including the step in which
the material leaves, and the material is taken to leave halfway through that
step, so

    residence time = τ · (N·1 − ½)

Summaries are over reliable cells (not flagged, not empty).

Needs the matrices from transition_matrix.

    python -m agulhas residence_time --res 0.5 1 2 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, parse_args, place_label, reference_grid, region_prefix, setting_tag
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import absorption, cell_centres, each_operator, reliable, to_grid


def residence_of(res, tau, name, op):
    """One matrix's residence time: (summary row, table of cells, days per cell)."""
    steps, _ = absorption(op)
    days = tau * (steps - 0.5)
    ok = reliable(op)
    lon, lat = cell_centres(op)
    d = days[ok]
    top = np.argmax(np.where(ok, days, -np.inf))
    row = {"res": res, "tau": tau, "drogue": name, "reliable_cells": int(ok.sum()),
           "mean_days": d.mean(), "median_days": np.median(d),
           "q25_days": np.percentile(d, 25), "q75_days": np.percentile(d, 75),
           "q90_days": np.percentile(d, 90), "max_days": d.max(),
           "max_lon": lon[top], "max_lat": lat[top]}
    cells = pd.DataFrame({"res": res, "tau": tau, "drogue": name, "lon": lon, "lat": lat,
                          "residence_days": days, "reliable": ok, "drifters": op["row_drifters"]})
    return row, cells, days


def print_summary(summary):
    print("Expected residence time in R over reliable cells (days)")
    print(f"{'grid':>5} {'τ':>4} {'type':<10}{'cells':>7}{'mean':>7}{'median':>8}{'q25–q75':>12}{'q90':>6}"
          f"{'max':>6}  longest at")
    for r in summary.itertuples():
        print(f"{r.res:>4g}° {r.tau:>4g} {r.drogue:<10}{r.reliable_cells:>7}{r.mean_days:>7.0f}{r.median_days:>8.0f}"
              f"{r.q25_days:>7.0f}–{r.q75_days:<4.0f}{r.q90_days:>6.0f}{r.max_days:>6.0f}  "
              f"{place_label(r.max_lon, r.max_lat)}")


def plot_residence(maps, args, map_res, path):
    """Residence time per reliable cell on the reference grid, per drogue type."""
    vmax = np.nanpercentile(np.r_[maps["drogued"][1], maps["undrogued"][1]], 99)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=MAP_KW)
    for ax, name in zip(axes, DROGUE_TYPES):
        op, days = maps[name]
        mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, days), cmap="viridis",
                             vmin=0, vmax=vmax, **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.set_title(f"{name.capitalize()}: median {np.nanmedian(days):.0f} days", fontsize=10)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="max")
    cb.set_label("Expected time afloat in R (days)")
    fig.suptitle(f"Residence time from each reliable cell — {map_res:g}°, τ = {args.tau[0]:g} d", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, multi=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    map_res = reference_grid(args)
    out_summary = f"data/residence_time_{tag}.csv"
    out_cells = f"data/residence_time_cells_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_residence_time_{tag}.png"

    rows, cells, maps = [], [], {}
    for res, tau, name, op in each_operator(args):
        row, cell_table, days = residence_of(res, tau, name, op)
        rows.append(row)
        cells.append(cell_table)
        if res == map_res and tau == args.tau[0]:
            maps[name] = (op, np.where(reliable(op), days, np.nan))

    summary = pd.DataFrame(rows)
    print_summary(summary)
    summary.to_csv(out_summary, index=False)
    pd.concat(cells).to_csv(out_cells, index=False)
    plot_residence(maps, args, map_res, out_png)
    print(f"\nSaved → {out_summary}\nSaved → {out_cells}\nSaved → {out_png}")


if __name__ == "__main__":
    main()
