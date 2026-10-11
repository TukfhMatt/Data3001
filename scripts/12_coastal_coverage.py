"""
Agulhas Region — Coastal Cell Support
Data support in coastal cells, where stranding and coastal exposure are
decided, against the open ocean. A coastal cell is an ocean cell with a land
neighbour on the grid (transport.coastal_states), the same definition 04, 06
and 09 use for coastal exposure. For every grid and lag and both drogue types:
coastal cells, the share that are empty (rows pooled from neighbours), and
the share of active coastal rows that are reliable (≥ MIN_DRIFTERS drifters),
next to the same share offshore.

Needs the matrices from 03.

    .venv/bin/python scripts/12_coastal_coverage.py --res 0.5 1 2 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

from config import DROGUE_TYPES, MIN_DRIFTERS, operator_path, parse_args, setting_tag
from transport import coastal_states, load_operator

args = parse_args(__doc__, grid=True, lag=True, multi=True)
TAG = setting_tag(args.res, args.tau, args.box)
COLORS = {"drogued": "#2a78d6", "undrogued": "#eb6834"}   # categorical slots 1–2

os.makedirs("figures", exist_ok=True)

# ── 1. Coastal vs offshore support ───────────────────────────────────────────
rows = []
for tau in args.tau:
    for res in args.res:
        for name in DROGUE_TYPES:
            op = load_operator(operator_path(name, res, tau, args.box))
            coast = coastal_states(op)
            active = ~op["empty"].astype(bool)
            ok = op["row_drifters"] >= MIN_DRIFTERS
            rows.append({
                "res": res, "tau": tau, "drogue": name,
                "coastal_cells": int(coast.sum()), "coastal_empty": int((coast & ~active).sum()),
                "coastal_median_drifters": np.median(op["row_drifters"][coast & active]),
                "coastal_share_reliable": ok[coast & active].mean(),
                "offshore_share_reliable": ok[~coast & active].mean(),
            })
table = pd.DataFrame(rows)
out_csv = f"data/coastal_support_{TAG}.csv"
table.to_csv(out_csv, index=False)

print(f"Coastal cells (land neighbour on the grid); reliable = ≥ {MIN_DRIFTERS} drifters, over active rows")
print(f"{'grid':>5} {'τ':>4} {'type':<10}{'coastal':>8}{'empty':>7}{'median drifters':>17}"
      f"{'reliable coast':>16}{'reliable offshore':>19}")
for r in table.itertuples():
    print(f"{r.res:>4g}° {r.tau:>4g} {r.drogue:<10}{r.coastal_cells:>8}{r.coastal_empty:>7}"
          f"{r.coastal_median_drifters:>17.0f}{r.coastal_share_reliable:>16.1%}{r.offshore_share_reliable:>19.1%}")

# ── 2. Figure: reliable share, coastal vs offshore ───────────────────────────
fig, axes = plt.subplots(1, len(args.tau), figsize=(6 * len(args.tau), 4), squeeze=False)
for ax, tau in zip(axes[0], args.tau):
    sub = table[table.tau == tau]
    x = np.arange(len(args.res))
    for k, name in enumerate(DROGUE_TYPES):
        s = sub[sub.drogue == name].set_index("res").loc[args.res]
        off = (k - 0.5) * 0.38
        bars = ax.bar(x + off, s.coastal_share_reliable * 100, 0.36, color=COLORS[name], label=f"{name}, coastal")
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
out_png = f"figures/agulhas_coastal_support_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_csv}\nSaved → {out_png}")
