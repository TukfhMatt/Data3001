"""
Agulhas Region — Coastal Cell Support
Data support in coastal cells, where stranding and coastal exposure are
decided, against the open ocean. A coastal cell is an ocean cell with a land
neighbour on the grid (transport.coastal_states), the same definition
shipping_overlay, bootstrap and oil_wind use for coastal exposure. For every
grid and lag and both drogue types: coastal cells, the share that are empty
(rows pooled from neighbours), and the share of active coastal rows that are
reliable (≥ MIN_DRIFTERS drifters), next to the same share offshore.

Needs the matrices from transition_matrix.

    python -m agulhas coastal_coverage --res 0.5 1 2 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, MIN_DRIFTERS, parse_args, region_prefix, setting_tag
from agulhas.plotting import DROGUE_COLORS, save_figure
from agulhas.transport import coastal_states, each_operator


def support_row(res, tau, name, op):
    """Support of one matrix's coastal cells, against its offshore cells."""
    coast = coastal_states(op)
    active = ~op["empty"].astype(bool)
    ok = op["row_drifters"] >= MIN_DRIFTERS
    return {
        "res": res, "tau": tau, "drogue": name,
        "coastal_cells": int(coast.sum()), "coastal_empty": int((coast & ~active).sum()),
        "coastal_median_drifters": np.median(op["row_drifters"][coast & active]),
        "coastal_share_reliable": ok[coast & active].mean(),
        "offshore_share_reliable": ok[~coast & active].mean(),
    }


def print_support(table):
    print(f"Coastal cells (land neighbour on the grid); reliable = ≥ {MIN_DRIFTERS} drifters, over active rows")
    print(f"{'grid':>5} {'τ':>4} {'type':<10}{'coastal':>8}{'empty':>7}{'median drifters':>17}"
          f"{'reliable coast':>16}{'reliable offshore':>19}")
    for r in table.itertuples():
        print(f"{r.res:>4g}° {r.tau:>4g} {r.drogue:<10}{r.coastal_cells:>8}{r.coastal_empty:>7}"
              f"{r.coastal_median_drifters:>17.0f}{r.coastal_share_reliable:>16.1%}{r.offshore_share_reliable:>19.1%}")


def plot_support(table, args, path):
    """Reliable share per grid, coastal bars against offshore marks, one panel per lag."""
    fig, axes = plt.subplots(1, len(args.tau), figsize=(6 * len(args.tau), 4), squeeze=False)
    for ax, tau in zip(axes[0], args.tau):
        sub = table[table.tau == tau]
        x = np.arange(len(args.res))
        for k, name in enumerate(DROGUE_TYPES):
            s = sub[sub.drogue == name].set_index("res").loc[args.res]
            off = (k - 0.5) * 0.38
            bars = ax.bar(x + off, s.coastal_share_reliable * 100, 0.36, color=DROGUE_COLORS[name],
                          label=f"{name}, coastal")
            ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
            ax.plot(x + off, s.offshore_share_reliable * 100, "_", ms=22, mew=2, color="#222222",
                    label="offshore" if k == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}°" for r in args.res])
        ax.set_ylim(0, 105)
        ax.set_ylabel(f"Active cells with ≥ {MIN_DRIFTERS} drifters (%)")
        ax.set_title(f"τ = {tau:g} d", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Coastal cells: drifter support against offshore", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, multi=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_csv = f"data/coastal_support_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_coastal_support_{tag}.png"

    table = pd.DataFrame([support_row(*matrix) for matrix in each_operator(args)])
    table.to_csv(out_csv, index=False)
    print_support(table)
    plot_support(table, args, out_png)
    print(f"\nSaved → {out_csv}\nSaved → {out_png}")


if __name__ == "__main__":
    main()
