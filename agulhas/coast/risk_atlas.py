"""
Agulhas Region — Release-Risk Atlas
For every ocean cell of R as a release cell, and for drogued, undrogued and oil
material: the share still afloat in R, exited across each edge, and stranded
on each named coast after 7, 30 and 365 days and eventually, plus the expected
days to strand among material that strands. Every value carries a 95% interval
from N_BOOT resamples of whole drifters (as bootstrap). Each row also carries the
cell's reliability, so a reader can see which entries rest on thin data
(`flagged`: fewer than 10 drifters; `empty`: no moves of its own, row pooled
from neighbouring cells). The figure shows reliable cells only.

Needs the coast operators and per-drifter counts from coastal_stranding.

    python -m agulhas risk_atlas --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from agulhas.config import grid_shape, operator_path, parse_args, setting_tag
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.stats import interval
from agulhas.transport import (COAST_LABELS, EXIT_LABELS, STRANDED, cell_centres, check_coast_box, days_to_strand,
                               load_coast_counts, load_operator, reliable, resampled_counts, row_normalise,
                               stranding_chain, to_grid)

N_BOOT = 1000
SEED = 0
HORIZONS = [7, 30, 365]
FIGURE_MATRICES = ["drogued", "undrogued", "oil"]
MATRICES = FIGURE_MATRICES + ["oil_3pct", "oil_4pct"]   # oil windage 3.5%, 3% and 4% of the wind
OUTCOMES = np.r_[["in_R"], EXIT_LABELS[:STRANDED], COAST_LABELS, ["stranded_any"]]


# ── All outcomes of one matrix, from every release cell ─────────────────────

def powers(P, wanted, n_cells):
    """Ocean rows of P^k for each k in wanted, by repeated squaring."""
    base, k = P, 1
    pending = {w: None for w in wanted}
    while k <= max(wanted):
        for w in wanted:
            if w & k:
                pending[w] = base if pending[w] is None else pending[w] @ base
        base, k = base @ base, k * 2
    return [pending[w][:n_cells].astype(np.float32) for w in wanted]


def outcomes(P, n_cells, steps, tau):
    """(horizons, cells, outcomes) array, eventual fate (cells, outcomes) with
    in_R left at 0, and days to strand per cell."""
    by_horizon = np.zeros((len(steps), n_cells, len(OUTCOMES)), dtype=np.float32)
    for h, Pk in enumerate(powers(P, steps, n_cells)):
        by_horizon[h, :, 0] = Pk[:, :n_cells].sum(axis=1)
        by_horizon[h, :, 1:-1] = Pk[:, n_cells:]
        by_horizon[h, :, -1] = Pk[:, n_cells + STRANDED:].sum(axis=1)
    fate, visits = stranding_chain(P, n_cells, STRANDED)
    eventual = np.zeros((n_cells, len(OUTCOMES)), dtype=np.float32)
    eventual[:, 1:-1] = fate
    b = fate[:, STRANDED:].sum(axis=1)
    eventual[:, -1] = b
    days = days_to_strand(visits, b, tau).astype(np.float32)
    return by_horizon, eventual, days


# ── Full data and drifter resamples ──────────────────────────────────────────

def resample(op, counts, steps, tau, n_lon, rng):
    """Outcomes on the full data, and their 95% intervals over N_BOOT drifter resamples:
    ((by horizon, eventual, days), (lo, hi) of each)."""
    n_cells, n_states = len(op["cell_flat"]), op["P"].shape[0]
    full = outcomes(op["P"], n_cells, steps, tau)
    boot = [np.zeros((N_BOOT,) + x.shape, dtype=np.float32) for x in full]
    for b in range(N_BOOT):
        Pb = row_normalise(resampled_counts(counts, n_states, rng), op["cell_flat"], n_lon)[0]
        boot[0][b], boot[1][b], boot[2][b] = outcomes(Pb, n_cells, steps, tau)
    bounds = (np.percentile(boot[0], [2.5, 97.5], axis=0), np.percentile(boot[1], [2.5, 97.5], axis=0),
              interval(boot[2]))
    return full, bounds


def atlas_rows(name, op, full, bounds):
    """One matrix's long table: a row per cell, horizon and outcome."""
    (full_h, full_e, full_d), ((lo_h, hi_h), (lo_e, hi_e), (lo_d, hi_d)) = full, bounds
    cell_lon, cell_lat = cell_centres(op)
    meta = pd.DataFrame({"matrix": name, "state": np.arange(len(op["cell_flat"])), "lon": cell_lon, "lat": cell_lat,
                         "reliable": reliable(op), "flagged": op["flagged"].astype(bool),
                         "empty": op["empty"].astype(bool), "row_drifters": op["row_drifters"]})
    frames = []
    for h, days in enumerate(HORIZONS):
        for k, outcome in enumerate(OUTCOMES):
            frames.append(meta.assign(horizon=days, outcome=outcome, estimate=full_h[h, :, k],
                                      lo=lo_h[h, :, k], hi=hi_h[h, :, k]))
    for k, outcome in enumerate(OUTCOMES[1:], start=1):
        frames.append(meta.assign(horizon="eventual", outcome=outcome, estimate=full_e[:, k],
                                  lo=lo_e[:, k], hi=hi_e[:, k]))
    frames.append(meta.assign(horizon="eventual", outcome="days_to_strand", estimate=full_d, lo=lo_d, hi=hi_d))
    return frames


def build_atlas(ops, counts, args, rng):
    """The atlas table, and each matrix's outcomes by horizon on the full data for the figure."""
    steps = [int(round(d / args.tau)) for d in HORIZONS]
    n_lon, _ = grid_shape(args.box, args.res)
    frames, maps = [], {}
    for name in MATRICES:
        full, bounds = resample(ops[name], counts[name], steps, args.tau, n_lon, rng)
        maps[name] = full[0]
        frames += atlas_rows(name, ops[name], full, bounds)
        print(f"{name:<10} {counts[name][3]:,} drifters resampled {N_BOOT} times")
    return pd.concat(frames, ignore_index=True), maps


# ── Figure: stranding by release cell and horizon ────────────────────────────

def plot_atlas(ops, maps, args, path):
    any_col = len(OUTCOMES) - 1
    norm = PowerNorm(0.5, vmin=0, vmax=1)
    fig, axes = plt.subplots(len(FIGURE_MATRICES), len(HORIZONS), figsize=(14, 9.5), subplot_kw=MAP_KW)
    for r, name in enumerate(FIGURE_MATRICES):
        op = ops[name]
        n_cells = len(op["cell_flat"])
        for h, days in enumerate(HORIZONS):
            ax = axes[r, h]
            values = np.where(reliable(op), maps[name][h, :, any_col], np.nan)
            grid = to_grid(op, np.r_[values, np.zeros(op["P"].shape[0] - n_cells)])
            mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], grid, cmap="Blues", norm=norm, **TRANSFORM_KW)
            map_axes(ax, args.box)
            ax.set_title(f"{name}, {days} days", fontsize=10)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.7, ticks=[0, 0.05, 0.2, 0.5, 1])
    cb.ax.set_yticklabels(["0%", "5%", "20%", "50%", "100%"])
    cb.set_label("Share of a release from this cell stranded by then (reliable cells only)")
    fig.suptitle(f"Release-risk atlas: stranding — {args.res:g}°, τ = {args.tau:g} d", fontsize=12)
    save_figure(path)


def print_summary(atlas, ops):
    pd.set_option("display.width", 200)
    any_rows = atlas[(atlas.outcome == "stranded_any") & atlas.reliable]
    summary = any_rows.assign(width=any_rows.hi - any_rows.lo).groupby(["matrix", "horizon"], sort=False).agg(
        cells=("state", "size"), median=("estimate", "median"), p90=("estimate", lambda x: x.quantile(0.9)),
        median_interval_width=("width", "median"))
    print("\nStranding (any coast) from reliable release cells")
    print(summary.round(3).to_string())
    n_cells = len(ops["drogued"]["cell_flat"])
    print("\nReliable cells: " + ", ".join(f"{n} {reliable(ops[n]).sum()} of {n_cells}" for n in MATRICES))


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    tag = setting_tag(args.res, args.tau, args.box)
    out_atlas = f"data/risk_atlas_{tag}.csv"
    out_png = f"figures/agulhas_risk_atlas_{tag}.png"

    ops = {name: load_operator(operator_path(f"{name}_coast", args.res, args.tau, args.box)) for name in MATRICES}
    counts = load_coast_counts(f"data/coast_counts_{tag}.npz", MATRICES, ops["drogued"]["P"].shape[0])
    atlas, maps = build_atlas(ops, counts, args, np.random.default_rng(SEED))
    atlas.to_csv(out_atlas, index=False, float_format="%.5g")

    plot_atlas(ops, maps, args, out_png)
    print_summary(atlas, ops)
    print(f"\nSaved → {out_atlas} ({len(atlas):,} rows), {out_png}")


if __name__ == "__main__":
    main()
