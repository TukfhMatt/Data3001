"""
Agulhas Region — Grid Coverage
Counts drifters per grid cell in box R at each grid size (defaults 0.5°, 1°
and 2°; see agulhas/config.py), split by drogue status, to choose the
transition-matrix grid.

Observations along one trajectory are strongly correlated, so the main
measure is the number of *distinct drifters* per cell, not raw obs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from agulhas.config import MIN_DRIFTERS, box_tag, grid_tag, parse_args, region_prefix
from agulhas.drifters import Drifters, in_box
from agulhas.grid import Grid
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure

THRESHOLDS = [5, 10, 20]   # distinct drifters per cell
LAND = "#f0f0f0"


def cell_counts(lon, lat, traj_idx, mask, grid):
    """(obs count, distinct drifter count) grids of the masked observations, shape (n_lat, n_lon)."""
    cell = grid.cell_in_box(lon[mask], lat[mask])
    obs = np.bincount(cell, minlength=grid.n_cells)
    pairs = np.unique(np.stack([cell, traj_idx[mask]]), axis=1)
    drifters = np.bincount(pairs[0], minlength=grid.n_cells)
    return obs.reshape(grid.n_lat, grid.n_lon), drifters.reshape(grid.n_lat, grid.n_lon)


def coverage(drifters, box, resolutions):
    """Per subset and grid: (obs, distinct drifters, ocean), where ocean cells are those
    visited by any drifter in R. Uses the positions as stored, single precision."""
    lon, lat, traj_idx = drifters.ds["lon"].values, drifters.ds["lat"].values, drifters.traj_idx
    in_R = in_box(lon, lat, box)
    subsets = {"Drogued (15 m current)": in_R & drifters.drogue,
               "Undrogued (+ wind slip)": in_R & ~drifters.drogue}
    results = {}
    for res in resolutions:
        grid = Grid(box, res)
        _, any_drifters = cell_counts(lon, lat, traj_idx, in_R, grid)
        ocean = any_drifters > 0
        for name, mask in subsets.items():
            obs, counts = cell_counts(lon, lat, traj_idx, mask, grid)
            results[(name, res)] = (obs, counts, ocean)
    return results


def print_coverage(results, box):
    print(f"Distinct drifters per ocean cell in R ({box_tag(box)})\n")
    header = f"{'Subset':<26}{'Res':>5}{'Cells':>7}{'Median':>8}" + "".join(f"{'≥' + str(t):>7}" for t in THRESHOLDS)
    print(header)
    print("-" * len(header))
    for (name, res), (_, drifters, ocean) in results.items():
        d = drifters[ocean]
        frac = "".join(f"{(d >= t).mean():>7.0%}" for t in THRESHOLDS)
        print(f"{name:<26}{res:>4}°{ocean.sum():>7}{np.median(d):>8.0f}{frac}")


def save_counts(results, box, path):
    lon_min, lon_max, lat_min, lat_max = box
    np.savez(path, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
             **{f"{'drogued' if n.startswith('Drogued') else 'undrogued'}_{r}_{k}": v
                for (n, r), (o, d, oc) in results.items()
                for k, v in [("obs", o), ("drifters", d), ("ocean", oc)]})
    print(f"\nCounts saved → {path}")


def plot_coverage(results, box, resolutions, path):
    """Distinct drifters per cell: one row per subset, one column per grid."""
    subsets = list(dict.fromkeys(name for name, _ in results))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("#d9d9d9")
    norm = LogNorm(vmin=MIN_DRIFTERS, vmax=max(d.max() for _, d, _ in results.values()))
    fig, axes = plt.subplots(len(subsets), len(resolutions), figsize=(6.5 * len(resolutions), 8.5),
                             subplot_kw=MAP_KW, constrained_layout=True)
    for r, name in enumerate(subsets):
        for c, res in enumerate(resolutions):
            ax = axes[r][c] if len(resolutions) > 1 else axes[r]
            _, drifters, ocean = results[(name, res)]
            grid = Grid(box, res)
            values = np.where(ocean, np.maximum(drifters, 0.5), np.nan)  # 0.5 → "under"
            mesh = ax.pcolormesh(grid.lon_edges, grid.lat_edges, values, cmap=cmap, norm=norm,
                                 edgecolor="white", linewidth=0.2, **TRANSFORM_KW)
            map_axes(ax, box, land=LAND, gridlines=True)
            pct = (drifters[ocean] >= MIN_DRIFTERS).mean()
            ax.set_title(f"{name} — {res}° grid  ({pct:.0%} of cells ≥ {MIN_DRIFTERS} drifters)", fontsize=10)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.6, extend="min")
    cb.set_label(f"Distinct drifters per cell (grey = fewer than {MIN_DRIFTERS})")
    fig.suptitle(f"Drifter coverage per grid cell — {box_tag(box)}", fontsize=13)
    save_figure(path)
    print(f"Map saved → {path}")


def main(argv=None):
    args = parse_args(__doc__, grid=True, multi=True, argv=argv)
    tag = grid_tag(sorted(args.res), args.box)
    resolutions = sorted(args.res, reverse=True)

    results = coverage(Drifters(args.data), args.box, resolutions)
    print_coverage(results, args.box)
    save_counts(results, args.box, f"data/grid_coverage_{tag}.npz")
    plot_coverage(results, args.box, resolutions, f"figures/{region_prefix(args.data)}_grid_coverage_{tag}.png")


if __name__ == "__main__":
    main()
