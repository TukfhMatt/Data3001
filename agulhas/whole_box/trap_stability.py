"""
Agulhas Region — Retention Hotspots and Their Stability
Where in R material is held longest. Each reliable cell is scored by the
probability that material released there is still afloat in R after 365
days; the top HOTSPOT_SHARE of reliable cells are the retention hotspots.
Two checks decide whether a hotspot is a feature of the flow or of the data:

  • across grids — hotspot sets of two grids rasterised onto a common
                   REF_RES grid and compared (Jaccard index and overlap
                   coefficient) only where both grids have reliable cells,
                   so cells flagged on one grid cannot count as disagreement
  • bootstrap    — on the reference grid, whole drifters are resampled
                   N_BOOT times and the matrix rebuilt each time: a 95%
                   interval on each cell's retention, and how often each cell
                   is a hotspot (share of resamples in its top HOTSPOT_SHARE)

Retention after a year also measures distance from the open edges of R, so a
hotspot is a place R holds material, not necessarily a closed eddy.

Needs the matrices from transition_matrix and the subset from fetch_gdp_subset
(for the bootstrap).

    python -m agulhas trap_stability --res 0.5 1 2 --tau 3.5
"""

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import (DROGUE_TYPES, grid_shape, operator_path, parse_args, place_label, reference_grid,
                            region_prefix, setting_tag)
from agulhas.drifters import Drifters, in_box
from agulhas.moves import MoveSet
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import (cell_centres, each_operator, load_operator, reliable, resampled_counts,
                               row_normalise, to_grid)

HOTSPOT_SHARE = 0.10
RETENTION_DAYS = 365
REF_RES = 0.25                 # common grid for comparing hotspot sets
N_BOOT = 1000
SEED = 0


def retention(P, n_cells, steps):
    """Probability that material starting in each ocean state is still in R after `steps`."""
    Q = P[:n_cells, :n_cells]
    s = np.ones(n_cells)
    for _ in range(steps):
        s = Q @ s
    return s


def hotspots(score, ok):
    """Top HOTSPOT_SHARE of the cells in ok by score."""
    return ok & (score >= np.quantile(score[ok], 1 - HOTSPOT_SHARE))


# ── Retention and hotspots per grid ──────────────────────────────────────────
def common_points(box):
    """Longitude and latitude of the common REF_RES grid."""
    lon_min, lon_max, lat_min, lat_max = box
    ref_lon = np.arange(lon_min + REF_RES / 2, lon_max, REF_RES)
    ref_lat = np.arange(lat_min + REF_RES / 2, lat_max, REF_RES)
    return [a.ravel() for a in np.meshgrid(ref_lon, ref_lat)]


def rasterise(op, box, res, points, ok, hot):
    """Reliable and hotspot flags at each REF_RES point, from the cell it falls in."""
    pt_lon, pt_lat = points
    n_lon, n_lat = grid_shape(box, res)
    flat = ((pt_lat - box[2]) // res).astype(int) * n_lon + ((pt_lon - box[0]) // res).astype(int)
    state = np.full(n_lon * n_lat, -1)
    state[op["cell_flat"]] = np.arange(len(op["cell_flat"]))
    st = state[flat]
    return np.where(st >= 0, ok[st], False), np.where(st >= 0, hot[st], False)


def hotspots_of(res, tau, name, op):
    """One matrix's hotspots: (summary row, table of cells, reliable flags, hotspot flags)."""
    ok = reliable(op)
    score = retention(op["P"], len(op["cell_flat"]), int(round(RETENTION_DAYS / tau)))
    hot = hotspots(score, ok)
    lon, lat = cell_centres(op)
    row = {"res": res, "tau": tau, "drogue": name, "reliable_cells": int(ok.sum()),
           "median_retention": np.median(score[ok]),
           "hotspot_threshold": score[hot].min(), "hotspot_cells": int(hot.sum()),
           "hotspot_lon": np.mean(lon[hot]), "hotspot_lat": np.mean(lat[hot])}
    cells = pd.DataFrame({"res": res, "tau": tau, "drogue": name, "lon": lon, "lat": lat,
                          "retention_365d": score, "reliable": ok, "hotspot": hot})
    return row, cells, ok, hot


def print_hotspots(summary):
    print(f"Retention after {RETENTION_DAYS} days; hotspots = top {HOTSPOT_SHARE:.0%} of reliable cells")
    print(f"{'grid':>5} {'τ':>4} {'type':<10}{'reliable':>9}{'median':>8}{'threshold':>10}{'hotspot centre':>18}")
    for r in summary.itertuples():
        print(f"{r.res:>4g}° {r.tau:>4g} {r.drogue:<10}{r.reliable_cells:>9}{r.median_retention:>8.1%}"
              f"{r.hotspot_threshold:>10.1%}{place_label(r.hotspot_lon, r.hotspot_lat):>18}")


# ── Agreement across grids ───────────────────────────────────────────────────
def grid_agreement(raster, args):
    """Hotspot agreement between each pair of grids, where both are reliable."""
    pairs = []
    for tau in args.tau:
        for name in DROGUE_TYPES:
            for a, b in combinations(args.res, 2):
                ok_a, hot_a = raster[(tau, a, name)]
                ok_b, hot_b = raster[(tau, b, name)]
                both = ok_a & ok_b
                ha, hb = hot_a & both, hot_b & both
                inter, union = (ha & hb).sum(), (ha | hb).sum()
                pairs.append({"tau": tau, "drogue": name, "grid_a": a, "grid_b": b,
                              "common_area_share": both.mean(),
                              "jaccard": inter / union if union else np.nan,
                              "overlap": inter / min(ha.sum(), hb.sum()) if min(ha.sum(), hb.sum()) else np.nan})
    return pd.DataFrame(pairs)


def print_agreement(pairs):
    print(f"\nHotspot agreement where both grids are reliable (rasterised at {REF_RES:g}°)")
    for r in pairs.itertuples():
        print(f"  τ = {r.tau:g} d {r.drogue:<10} {r.grid_a:g}° vs {r.grid_b:g}°: Jaccard {r.jaccard:.0%}, "
              f"overlap {r.overlap:.0%} (over {r.common_area_share:.0%} of R)")


# ── Drifter bootstrap on the reference grid ──────────────────────────────────
def bootstrap_retention(op, counts, n_lon, steps, ok, rng):
    """Resampled retention per cell, and how often each cell is a hotspot: (samples, frequency)."""
    n, n_states = len(op["cell_flat"]), op["P"].shape[0]
    samples = np.empty((N_BOOT, n))
    in_top = np.zeros(n)
    for b in range(N_BOOT):
        Pb, _ = row_normalise(resampled_counts(counts, n_states, rng), op["cell_flat"], n_lon)
        samples[b] = retention(Pb, n, steps)
        in_top += hotspots(samples[b], ok)
    return samples, in_top / N_BOOT


def print_bootstrap(title, n_drifters, cells):
    """The bootstrap of one matrix, from its table of cells."""
    score, ok, hot, freq, lo, hi = (cells[c].values for c in ("retention_365d", "reliable", "hotspot",
                                                              "hotspot_frequency", "ci_low", "ci_high"))
    print(f"\n{title}: {n_drifters:,} drifters resampled {N_BOOT} times")
    print(f"  hotspot cells a hotspot in ≥ 50% of resamples: {(freq[hot] >= 0.5).sum()} of {hot.sum()}; "
          f"other cells a hotspot in ≥ 50%: {(freq[ok & ~hot] >= 0.5).sum()}")
    print(f"  {'cell':>15}{'drifters':>10}{'retention (95%)':>22}{'hotspot freq':>14}")
    for k in np.argsort(-score * ok)[:8]:
        print(f"  {place_label(cells.lon.values[k], cells.lat.values[k]):>15}{cells.drifters.values[k]:>10}"
              f"{score[k]:>9.0%} ({lo[k]:.0%}–{hi[k]:.0%}){freq[k]:>14.0%}")


def run_bootstrap(args, res, rng):
    """Bootstrap each drogue type and lag on one grid: (tables of cells, maps of hotspot frequency)."""
    drifters = Drifters(args.data)
    in_R = in_box(drifters.lon, drifters.lat, args.box)
    strand_t = drifters.strand_times(in_R)
    rows, maps = [], {}
    for tau in args.tau:
        steps = int(round(RETENTION_DAYS / tau))
        moves = MoveSet(drifters, args.box, res, tau, strand_t)
        for name, flag in DROGUE_TYPES.items():
            op = load_operator(operator_path(name, res, tau, args.box))
            counts = moves.counts_for(op, name, flag)
            ok = reliable(op)
            score = retention(op["P"], len(op["cell_flat"]), steps)
            hot = hotspots(score, ok)
            samples, freq = bootstrap_retention(op, counts, moves.grid.n_lon, steps, ok, rng)
            lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
            clon, clat = cell_centres(op)
            rows.append(pd.DataFrame({"tau": tau, "drogue": name, "lon": clon, "lat": clat,
                                      "reliable": ok, "drifters": op["row_drifters"], "hotspot": hot,
                                      "retention_365d": score, "ci_low": lo, "ci_high": hi,
                                      "hotspot_frequency": freq}))
            maps[(tau, name)] = (op, np.where(ok, freq, np.nan))
            print_bootstrap(f"{name}, {res:g}°, τ = {tau:g} d", counts[3], rows[-1])
    return rows, maps


# ── Map ──────────────────────────────────────────────────────────────────────
def plot_hotspot_frequency(maps, args, res, path):
    tau0 = args.tau[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=MAP_KW)
    for ax, name in zip(axes, DROGUE_TYPES):
        op, freq = maps[(tau0, name)]
        mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], to_grid(op, freq),
                             cmap="magma_r", vmin=0, vmax=1, **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.set_title(f"{name.capitalize()}: share of resamples in the top {HOTSPOT_SHARE:.0%}", fontsize=10)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8)
    cb.set_label("Hotspot frequency (reliable cells)")
    fig.suptitle(f"Retention hotspots after {RETENTION_DAYS} days — {res:g}°, τ = {tau0:g} d, "
                 f"{N_BOOT} drifter resamples", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, multi=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_summary = f"data/retention_hotspots_{tag}.csv"
    out_pairs = f"data/retention_hotspot_agreement_{tag}.csv"
    out_cells = f"data/retention_cells_{tag}.csv"
    out_boot = f"data/retention_bootstrap_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_retention_hotspots_{tag}.png"
    boot_res = reference_grid(args)
    points = common_points(args.box)

    rows, cells, raster = [], [], {}
    for res, tau, name, op in each_operator(args):
        row, cell_table, ok, hot = hotspots_of(res, tau, name, op)
        rows.append(row)
        cells.append(cell_table)
        raster[(tau, res, name)] = rasterise(op, args.box, res, points, ok, hot)
    summary = pd.DataFrame(rows)
    print_hotspots(summary)

    pairs = grid_agreement(raster, args)
    print_agreement(pairs)

    boot_rows, maps = run_bootstrap(args, boot_res, np.random.default_rng(SEED))

    summary.to_csv(out_summary, index=False)
    pairs.to_csv(out_pairs, index=False)
    pd.concat(cells).to_csv(out_cells, index=False)
    pd.concat(boot_rows).to_csv(out_boot, index=False)
    plot_hotspot_frequency(maps, args, boot_res, out_png)
    print(f"\nSaved → {out_summary}\nSaved → {out_pairs}\nSaved → {out_cells}\nSaved → {out_boot}\nSaved → {out_png}")


if __name__ == "__main__":
    main()
