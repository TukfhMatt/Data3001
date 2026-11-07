"""
Agulhas Region — Source and Sink Cells
Which cells of R gather material and which shed it over one step. Every row of
P sums to 1, so the rows say nothing here; the column sums do. If each ocean
cell starts with one unit of material, the column sum of cell j,

    inflow_j = Σ_i P[i, j]  over the ocean cells i,

is what cell j holds after one τ. A cell above 1 gathers material (a sink) and
a cell below 1 sheds it (a source), judged by whether the 95% interval from
N_BOOT resamples of whole drifters excludes 1. The intervals are not corrected
for the number of cells compared, so the agreement between the two drogue types
is printed as a check.

Cells on the edge of R are not judged: material entering R from outside is not
in the matrix, so they would look like sources whatever the flow does.

The even starting load is a measure of the flow's convergence, not of where
material really is. Retention over a year is in trap_stability and
eventual fates in exit_zones.
Inflow counts every row that reaches the cell, including thin ones, so only the
receiving cell's own reliability is checked.

Needs the matrices from transition_matrix and the subset from fetch_gdp_subset
(for the bootstrap).

    python -m agulhas source_sink --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, operator_path, parse_args, place_label, region_prefix, setting_tag
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import cell_centres, load_operator, reliable, resampled_counts, row_normalise, to_grid

N_BOOT = 1000
SEED = 0
TOP = 5        # cells printed at each end


def inflow(P, n):
    """Column sums of the ocean block of P."""
    return P[:n, :n].sum(axis=0)


def inflow_table(name, flag, op, moves, rng):
    """Inflow per cell of one drogue type, with its interval and what it makes the cell."""
    cells, n = op["cell_flat"], len(op["cell_flat"])
    n_states = op["P"].shape[0]
    counts = moves.counts_for(op, name, flag)
    boot = np.empty((N_BOOT, n))
    for b in range(N_BOOT):
        boot[b] = inflow(row_normalise(resampled_counts(counts, n_states, rng), cells, moves.grid.n_lon)[0], n)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    clon, clat = cell_centres(op)
    grid = moves.grid
    on_edge = ((np.minimum(clon - grid.lon_min, grid.lon_max - clon) < grid.res)
               | (np.minimum(clat - grid.lat_min, grid.lat_max - clat) < grid.res))
    kind = np.where(on_edge, "edge", np.where(lo > 1, "sink", np.where(hi < 1, "source", "neither")))
    print(f"{name}: {counts[3]:,} drifters resampled {N_BOOT} times")
    return pd.DataFrame({"drogue": name, "lon": clon, "lat": clat, "inflow": inflow(op["P"], n), "lo": lo, "hi": hi,
                         "kind": kind, "reliable": reliable(op), "row_drifters": op["row_drifters"]})


def print_summary(good):
    """Sources and sinks per drogue type, then how far the two types agree."""
    for name in DROGUE_TYPES:
        part = good[good.drogue == name]
        shares = part.kind.value_counts().reindex(["sink", "source", "neither"], fill_value=0)
        print(f"\n{name}: {len(part)} reliable cells off the edge — " + ", ".join(f"{k} {v}" for k, v in shares.items()))
        ranked = part[part.kind != "neither"].sort_values("inflow")
        for label, rows in [("strongest sources", ranked.head(TOP)), ("strongest sinks", ranked.tail(TOP)[::-1])]:
            print(f"  {label}:")
            for r in rows.itertuples():
                print(f"    {place_label(r.lon, r.lat):>15}  {r.inflow:.2f} ({r.lo:.2f}–{r.hi:.2f}), "
                      f"{r.row_drifters} drifters")
    both = good.pivot(index=["lon", "lat"], columns="drogue", values=["inflow", "kind"]).dropna()
    kind_d, kind_u = both[("kind", "drogued")], both[("kind", "undrogued")]
    judged = (kind_d != "neither") & (kind_u != "neither")
    print(f"\nOver {len(both)} cells reliable in both types: correlation of inflow "
          f"{np.corrcoef(both[('inflow', 'drogued')].astype(float), both[('inflow', 'undrogued')].astype(float))[0, 1]:.2f}; "
          f"source or sink in both {judged.sum()}, same kind {(kind_d[judged] == kind_u[judged]).sum()}")


def plot_inflow(ops, tables, span, args, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=MAP_KW)
    for ax, (name, op), part in zip(axes, ops.items(), tables):
        ok = part.reliable.values & (part.kind.values != "edge")
        grid = to_grid(op, np.r_[np.where(ok, part.inflow.values, np.nan), np.zeros(op["P"].shape[0] - len(part))])
        mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], grid, cmap="RdBu_r", vmin=1 - span, vmax=1 + span,
                             **TRANSFORM_KW)
        marked = part[ok & (part.kind != "neither")]
        ax.scatter(marked.lon, marked.lat, s=3, color="#222222", linewidths=0, zorder=4, **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.set_title(f"{name.capitalize()}: dot = 95% interval excludes 1", fontsize=10)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8)
    cb.set_label("Held after one step from a load of 1")
    fig.suptitle(f"Source and sink cells — {args.res:g}°, τ = {args.tau:g} d, reliable cells off the edge of R; "
                 "above 1 gathers, below 1 sheds", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_csv = f"data/source_sink_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_source_sink_{tag}.png"
    rng = np.random.default_rng(SEED)

    moves = MoveSet(Drifters(args.data), args.box, args.res, args.tau)
    ops = {name: load_operator(operator_path(name, args.res, args.tau, args.box)) for name in DROGUE_TYPES}
    tables = [inflow_table(name, flag, ops[name], moves, rng) for name, flag in DROGUE_TYPES.items()]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out_csv, index=False, float_format="%.5g")

    good = table[table.reliable & (table.kind != "edge")]
    print_summary(good)
    plot_inflow(ops, tables, np.nanpercentile(np.abs(good.inflow - 1), 98), args, out_png)
    print(f"\nSaved → {out_csv}\nSaved → {out_png}")


if __name__ == "__main__":
    main()
