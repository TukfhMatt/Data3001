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

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from agulhas.config import grid_shape, operator_path, parse_args, setting_tag
from agulhas.transport import (COAST_LABELS, EXIT_LABELS, STRANDED, cell_centres, check_coast_box, load_operator, reliable,
                       resampled_counts, row_normalise, stranding_chain, to_grid)

N_BOOT = 1000
SEED = 0
HORIZONS = [7, 30, 365]
FIGURE_MATRICES = ["drogued", "undrogued", "oil"]
MATRICES = FIGURE_MATRICES + ["oil_3pct", "oil_4pct"]   # oil windage 3.5%, 3% and 4% of the wind
OUTCOMES = np.r_[["in_R"], EXIT_LABELS[:STRANDED], COAST_LABELS, ["stranded_any"]]


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
    RES, TAU = args.res, args.tau
    TAG = setting_tag(RES, TAU, args.box)

    os.makedirs("figures", exist_ok=True)

    ops = {name: load_operator(operator_path(f"{name}_coast", RES, TAU, args.box)) for name in MATRICES}
    op0 = ops["drogued"]
    cells = op0["cell_flat"]
    n_cells = len(cells)
    n_states = op0["P"].shape[0]
    n_lon, _ = grid_shape(args.box, RES)
    cell_lon, cell_lat = cell_centres(op0)
    steps = [int(round(d / TAU)) for d in HORIZONS]
    counts_file = np.load(f"data/coast_counts_{TAG}.npz")
    if int(counts_file["n_states"]) != n_states:
        raise SystemExit("data/coast_counts_….npz does not match the coast operators; rerun coastal_stranding.")

    # ── 1. All outcomes of one matrix, from every release cell ───────────────
    def powers(P, wanted):
        """Ocean rows of P^k for each k in wanted, by repeated squaring."""
        base, k = P, 1
        pending = {w: None for w in wanted}
        while k <= max(wanted):
            for w in wanted:
                if w & k:
                    pending[w] = base if pending[w] is None else pending[w] @ base
            base, k = base @ base, k * 2
        return [pending[w][:n_cells].astype(np.float32) for w in wanted]

    def outcomes(P):
        """(horizons, cells, outcomes) array, eventual fate (cells, outcomes) with
    in_R left at 0, and days to strand per cell."""
        by_horizon = np.zeros((len(steps), n_cells, len(OUTCOMES)), dtype=np.float32)
        for h, Pk in enumerate(powers(P, steps)):
            by_horizon[h, :, 0] = Pk[:, :n_cells].sum(axis=1)
            by_horizon[h, :, 1:-1] = Pk[:, n_cells:]
            by_horizon[h, :, -1] = Pk[:, n_cells + STRANDED:].sum(axis=1)
        fate, visits = stranding_chain(P, n_cells, STRANDED)
        eventual = np.zeros((n_cells, len(OUTCOMES)), dtype=np.float32)
        eventual[:, 1:-1] = fate
        b = fate[:, STRANDED:].sum(axis=1)
        eventual[:, -1] = b
        with np.errstate(invalid="ignore", divide="ignore"):
            days = np.where(b > 0, visits / b * TAU, np.nan).astype(np.float32)
        return by_horizon, eventual, days

    # ── 2. Full data and drifter resamples ───────────────────────────────────
    rng = np.random.default_rng(SEED)
    frames, maps = [], {}
    for name in MATRICES:
        op = ops[name]
        full_h, full_e, full_d = outcomes(op["P"])
        maps[name] = full_h
        counts = tuple(counts_file[f"{name}_{part}"] for part in ("d_idx", "ft", "n"))
        counts += (int(counts_file[f"{name}_n_drifters"]),)
        boot_h = np.zeros((N_BOOT,) + full_h.shape, dtype=np.float32)
        boot_e = np.zeros((N_BOOT,) + full_e.shape, dtype=np.float32)
        boot_d = np.zeros((N_BOOT,) + full_d.shape, dtype=np.float32)
        for b in range(N_BOOT):
            Pb = row_normalise(resampled_counts(counts, n_states, rng), cells, n_lon)[0]
            boot_h[b], boot_e[b], boot_d[b] = outcomes(Pb)
        lo_h, hi_h = np.percentile(boot_h, [2.5, 97.5], axis=0)
        lo_e, hi_e = np.percentile(boot_e, [2.5, 97.5], axis=0)
        with np.errstate(all="ignore"):
            lo_d, hi_d = np.nanpercentile(boot_d, [2.5, 97.5], axis=0)
        del boot_h, boot_e, boot_d

        # ── 3. One long table per matrix ─────────────────────────────────────
        meta = pd.DataFrame({"matrix": name, "state": np.arange(n_cells), "lon": cell_lon, "lat": cell_lat,
                             "reliable": reliable(op), "flagged": op["flagged"].astype(bool),
                             "empty": op["empty"].astype(bool), "row_drifters": op["row_drifters"]})
        for h, days in enumerate(HORIZONS):
            for k, outcome in enumerate(OUTCOMES):
                frames.append(meta.assign(horizon=days, outcome=outcome, estimate=full_h[h, :, k],
                                          lo=lo_h[h, :, k], hi=hi_h[h, :, k]))
        for k, outcome in enumerate(OUTCOMES[1:], start=1):
            frames.append(meta.assign(horizon="eventual", outcome=outcome, estimate=full_e[:, k],
                                      lo=lo_e[:, k], hi=hi_e[:, k]))
        frames.append(meta.assign(horizon="eventual", outcome="days_to_strand", estimate=full_d,
                                  lo=lo_d, hi=hi_d))
        print(f"{name:<10} {counts[3]:,} drifters resampled {N_BOOT} times")

    atlas = pd.concat(frames, ignore_index=True)
    out_atlas = f"data/risk_atlas_{TAG}.csv"
    atlas.to_csv(out_atlas, index=False, float_format="%.5g")

    # ── 4. Figure: stranding by release cell and horizon ─────────────────────
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        map_kw = {"projection": ccrs.PlateCarree()}
    except ImportError:
        ccrs, map_kw = None, {}
    kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
    any_col = len(OUTCOMES) - 1
    norm = PowerNorm(0.5, vmin=0, vmax=1)
    fig, axes = plt.subplots(len(FIGURE_MATRICES), len(HORIZONS), figsize=(14, 9.5), subplot_kw=map_kw)
    for r, name in enumerate(FIGURE_MATRICES):
        op = ops[name]
        for h, days in enumerate(HORIZONS):
            ax = axes[r, h]
            values = np.where(reliable(op), maps[name][h, :, any_col], np.nan)
            grid = to_grid(op, np.r_[values, np.zeros(n_states - n_cells)])
            mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], grid, cmap="Blues", norm=norm, **kw)
            if ccrs:
                ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
            ax.set_title(f"{name}, {days} days", fontsize=10)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.7, ticks=[0, 0.05, 0.2, 0.5, 1])
    cb.ax.set_yticklabels(["0%", "5%", "20%", "50%", "100%"])
    cb.set_label("Share of a release from this cell stranded by then (reliable cells only)")
    fig.suptitle(f"Release-risk atlas: stranding — {RES:g}°, τ = {TAU:g} d", fontsize=12)
    out_png = f"figures/agulhas_risk_atlas_{TAG}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Summary ───────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    any_rows = atlas[(atlas.outcome == "stranded_any") & atlas.reliable]
    summary = any_rows.assign(width=any_rows.hi - any_rows.lo).groupby(["matrix", "horizon"], sort=False).agg(
        cells=("state", "size"), median=("estimate", "median"), p90=("estimate", lambda x: x.quantile(0.9)),
        median_interval_width=("width", "median"))
    print("\nStranding (any coast) from reliable release cells")
    print(summary.round(3).to_string())
    print("\nReliable cells: " + ", ".join(f"{n} {reliable(ops[n]).sum()} of {n_cells}" for n in MATRICES))
    print(f"\nSaved → {out_atlas} ({len(atlas):,} rows), {out_png}")


if __name__ == "__main__":
    main()
