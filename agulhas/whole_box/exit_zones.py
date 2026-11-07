"""
Agulhas Region — Exit Zones
Where material released in each cell eventually ends up: across which edge
of R it leaves (W, E, S, N) or whether it strands on a coast inside R. The
eventual fate comes exactly from the absorbing chain (transport.absorption,
N·R); the fate after about one year comes from propagating P. For every grid
and lag and both drogue types: the mean fate over reliable cells, and the
share of reliable cells whose most likely fate is each exit, mapped on the
reference grid. Fates are read through exit_labels.

Needs the matrices from transition_matrix.

    python -m agulhas exit_zones --res 0.5 1 2 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from agulhas.config import DROGUE_TYPES, parse_args, reference_grid, region_prefix, setting_tag
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import absorption, cell_centres, each_operator, reliable, to_grid

YEAR_DAYS = 365
FATE_COLORS = {"W": "#2a78d6", "E": "#eb6834", "S": "#1b9e77", "N": "#7b3294", "stranded": "#1a1a1a"}


def fates_of(res, tau, name, op):
    """One matrix's fates: (summary row, table of cells, most likely fate per cell)."""
    year_steps = int(round(YEAR_DAYS / tau))
    labels = exit_labels(op)
    n = len(op["cell_flat"])
    _, fate = absorption(op)
    year = np.linalg.matrix_power(op["P"], year_steps)[:n, n:]
    ok = reliable(op)
    dominant = np.argmax(fate, axis=1)
    row = {"res": res, "tau": tau, "drogue": name, "reliable_cells": int(ok.sum()),
           "year_days": year_steps * tau, "in_R_1yr": 1 - year[ok].sum(axis=1).mean()}
    for k, lab in enumerate(labels):
        row[f"eventual_{lab}"] = fate[ok, k].mean()
        row[f"1yr_{lab}"] = year[ok, k].mean()
        row[f"dominant_{lab}"] = np.mean(dominant[ok] == k)
    lon, lat = cell_centres(op)
    cells = pd.DataFrame({"res": res, "tau": tau, "drogue": name, "lon": lon, "lat": lat,
                          "reliable": ok, "dominant": np.array(labels)[dominant]})
    for k, lab in enumerate(labels):
        cells[f"eventual_{lab}"] = fate[:, k]
        cells[f"1yr_{lab}"] = year[:, k]
    return row, cells, dominant


def exit_labels(op):
    return [str(x) for x in op["exit_labels"]]


def print_summary(summary, labels):
    print("Mean eventual fate over reliable cells, and (in brackets) the share of cells where it is the most likely")
    print(f"{'grid':>5} {'τ':>4} {'type':<10}{'cells':>7}" + "".join(f"{lab:>16}" for lab in labels)
          + f"{'in R at 1 yr':>14}")
    for r in summary.to_dict("records"):
        print(f"{r['res']:>4g}° {r['tau']:>4g} {r['drogue']:<10}{r['reliable_cells']:>7}"
              + "".join(f"{r[f'eventual_{lab}']:>9.1%} ({r[f'dominant_{lab}']:>3.0%})" for lab in labels)
              + f"{r['in_R_1yr']:>14.1%}")


def plot_most_likely_fate(maps, labels, args, map_res, path):
    """Most likely fate per reliable cell on the reference grid, per drogue type."""
    cmap = ListedColormap([FATE_COLORS[lab] for lab in labels])
    norm = BoundaryNorm(np.arange(len(labels) + 1) - 0.5, len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=MAP_KW)
    for ax, name in zip(axes, DROGUE_TYPES):
        op, dom = maps[name]
        ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, dom), cmap=cmap, norm=norm, **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.set_title(f"{name.capitalize()}", fontsize=10)
    fig.legend(handles=[Patch(color=FATE_COLORS[lab], label=lab) for lab in labels], title="Most likely fate",
               loc="center right", frameon=False)
    fig.suptitle(f"Eventual exit or stranding from each reliable cell — {map_res:g}°, τ = {args.tau[0]:g} d",
                 fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, multi=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    map_res = reference_grid(args)
    out_summary = f"data/exit_zones_{tag}.csv"
    out_cells = f"data/exit_zones_cells_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_exit_zones_{tag}.png"

    rows, cells, maps = [], [], {}
    for res, tau, name, op in each_operator(args):
        row, cell_table, dominant = fates_of(res, tau, name, op)
        rows.append(row)
        cells.append(cell_table)
        if res == map_res and tau == args.tau[0]:
            maps[name] = (op, np.where(reliable(op), dominant, np.nan))

    summary = pd.DataFrame(rows)
    labels = exit_labels(op)
    print_summary(summary, labels)
    summary.to_csv(out_summary, index=False)
    pd.concat(cells).to_csv(out_cells, index=False)
    plot_most_likely_fate(maps, labels, args, map_res, out_png)
    print(f"\nSaved → {out_summary}\nSaved → {out_cells}\nSaved → {out_png}")


if __name__ == "__main__":
    main()
