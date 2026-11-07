"""
Agulhas Region — Whole-Box Data Support
How well the data support each row of the transition matrices, for every
grid and lag and both drogue types, before the whole-box analysis
draws conclusions from them:

  • active cells       — cells with outgoing moves of their own (not empty)
  • transition support — moves behind each active row (median, share ≥ 50)
  • drifter support    — distinct drifters behind each active row (median,
                         share ≥ MIN_DRIFTERS, i.e. not flagged)

Move counts overstate the support, since the hourly moves along one track
are strongly correlated; the distinct-drifter count is the one that matters.

Needs the matrices from transition_matrix.

    python -m agulhas whole_box_analysis --res 0.5 1 2 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, MIN_DRIFTERS, parse_args, region_prefix, setting_tag
from agulhas.plotting import DROGUE_COLORS, save_figure
from agulhas.transport import each_operator

MIN_MOVES = 50


def support_row(res, tau, name, op):
    """Row support of one matrix, over its active cells."""
    active = ~op["empty"].astype(bool)
    moves, drifters = op["row_obs"][active], op["row_drifters"][active]
    return {
        "res": res, "tau": tau, "drogue": name,
        "ocean_cells": len(active), "active_cells": int(active.sum()),
        "empty_cells": int((~active).sum()), "moves": int(op["row_obs"].sum()),
        "median_moves": np.median(moves), "share_ge50_moves": np.mean(moves >= MIN_MOVES),
        "median_drifters": np.median(drifters),
        "q25_drifters": np.percentile(drifters, 25), "q75_drifters": np.percentile(drifters, 75),
        "share_reliable": np.mean(drifters >= MIN_DRIFTERS),
    }


def print_support(table):
    print(f"Row support over active cells (≥ {MIN_MOVES} moves; ≥ {MIN_DRIFTERS} distinct drifters = reliable)")
    print(f"{'grid':>5} {'τ':>4} {'type':<10}{'active':>8}{'empty':>7}{'median moves':>14}"
          f"{'≥50 moves':>11}{'median drifters':>17}{'reliable':>10}")
    for r in table.itertuples():
        print(f"{r.res:>4g}° {r.tau:>4g} {r.drogue:<10}{r.active_cells:>8}{r.empty_cells:>7}"
              f"{r.median_moves:>14,.0f}{r.share_ge50_moves:>11.1%}{r.median_drifters:>17.0f}{r.share_reliable:>10.1%}")


def plot_support(table, args, path):
    """Share of reliable rows per grid, one panel per lag."""
    fig, axes = plt.subplots(1, len(args.tau), figsize=(5.5 * len(args.tau), 4), squeeze=False)
    for ax, tau in zip(axes[0], args.tau):
        sub = table[table.tau == tau]
        x = np.arange(len(args.res))
        for k, name in enumerate(DROGUE_TYPES):
            vals = sub[sub.drogue == name].set_index("res").loc[args.res, "share_reliable"] * 100
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=DROGUE_COLORS[name], label=name)
            ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}°" for r in args.res])
        ax.set_ylim(0, 105)
        ax.set_ylabel(f"Active cells with ≥ {MIN_DRIFTERS} drifters (%)")
        ax.set_title(f"τ = {tau:g} d", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Whole-box drifter support per grid", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, multi=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_csv = f"data/whole_box_support_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_whole_box_support_{tag}.png"

    table = pd.DataFrame([support_row(*matrix) for matrix in each_operator(args)])
    table.to_csv(out_csv, index=False)
    print_support(table)
    plot_support(table, args, out_png)
    print(f"\nSaved → {out_csv}\nSaved → {out_png}")


if __name__ == "__main__":
    main()
