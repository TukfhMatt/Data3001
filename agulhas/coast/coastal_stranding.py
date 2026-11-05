"""
Agulhas Region — Stranding by Coast
Splits the stranded state of each operator by the coast where drifters ran
aground, so stranding can be reported per named coast (transport.COAST_LABELS:
six coasts of R, each with ≥ 13 grounded drifters). A stranding move takes the
coast of its drifter's last observation (transport.coast_of); every other
move is unchanged, so the coast columns sum to the operator's stranded column
exactly, which is checked.

Built for drogued and undrogued drifters (from the subset, as transition_matrix) and for oil
and the oil check (from the per-drifter moves saved by oil_wind), and saved as
operator_path("{name}_coast", …) with exit_labels W, E, S, N and the six
coasts. Reports, for every cell and for the release points from
shipping_overlay:
  • eventual stranding on each coast, exact from the absorbing chain
  • stranding on each coast within 7, 30 and 365 days
  • expected time to strand, among material that strands
with 95% intervals for the release points from N_BOOT resamples of whole
drifters (as bootstrap). Drogued stranding is a lower bound (see drogue_loss).

Needs the matrices from transition_matrix and oil_wind, data/oil_moves_….npz
from oil_wind and the release points from shipping_overlay (same box, default
grid and lag).

    python -m agulhas coastal_stranding --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import xarray as xr

from agulhas.config import DROGUE_TYPES, MIN_DRIFTERS, grid_shape, operator_path, parse_args, release_points_path, setting_tag
from agulhas.transport import (COAST_COLORS, COAST_LABELS, COAST_NAMES, EXIT_LABELS, STRANDED, cell_centres, check_coast_box, coast_of,
                       last_obs_index, load_operator, move_indices, reliable, resampled_counts,
                       row_normalise, same_drogue, state_of, strand_times, stranding_chain, to_grid,
                       transition_states)

N_BOOT = 1000
SEED = 0
HORIZONS = [7, 30, 365]
MATRICES = ["drogued", "undrogued", "oil", "oil_check"]
WINDAGE_MATRICES = ["oil_3pct", "oil_4pct"]   # split for risk_atlas and oil_windage, which report them
COAST_EXIT_LABELS = np.r_[EXIT_LABELS[:STRANDED], COAST_LABELS]


def interval(draws, axis=0):
    with np.errstate(all="ignore"):
        return np.nanpercentile(draws, [2.5, 97.5], axis=axis)


def fmt(e, lo, hi):
    return f"{e:.0%} ({lo:.0%}–{hi:.0%})"


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
    RES, TAU = args.res, args.tau
    TAG = setting_tag(RES, TAU, args.box)

    os.makedirs("figures", exist_ok=True)

    # ── 1. Load drifters; coast of each grounding ────────────────────────────
    ds = xr.open_dataset(args.data)
    lon = ds["lon"].values.astype(float)
    lat = ds["lat"].values.astype(float)
    drogue = ds["drogue_status"].values.astype(bool)
    t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
    traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
    in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
    n_lon, _ = grid_shape(args.box, RES)
    cell_flat = np.where(in_R, ((lat - LAT_MIN) // RES).astype(int) * n_lon
                         + ((lon - LON_MIN) // RES).astype(int), -1)

    strand_t = strand_times(ds, traj_idx, t, in_R)
    last = np.unique(last_obs_index(traj_idx))
    grounded = last[strand_t[last] >= 0]
    coast_of_traj = np.full(ds.sizes["traj"], -1)
    coast_of_traj[traj_idx[grounded]] = coast_of(lon[grounded], lat[grounded])
    start, end, stranded = move_indices(traj_idx, t, in_R, TAU, strand_t)

    base = {name: load_operator(operator_path(name, RES, TAU, args.box)) for name in MATRICES + WINDAGE_MATRICES}
    cells = base["drogued"]["cell_flat"]
    n_cells = len(cells)
    n_base = n_cells + len(EXIT_LABELS)
    n_states = n_cells + len(COAST_EXIT_LABELS)
    strand_col = n_cells + STRANDED

    # ── 2. Coast operators: stranding moves split by the drifter's coast ─────
    def split_moves(traj, ft, n):
        """Moves per (drifter, from, to) on the base layout → coast layout, as
    counts for resampled_counts: (drifter number, flat index, moves, drifters)."""
        f, to = np.divmod(ft, n_base)
        strands = to == strand_col
        if (coast_of_traj[traj[strands]] < 0).any():
            raise SystemExit("A stranding move belongs to a drifter with no grounding coast.")
        to = np.where(strands, strand_col + coast_of_traj[traj], to)
        drifters, d_idx = np.unique(traj, return_inverse=True)
        return d_idx, f * n_states + to, n, len(drifters)

    def matrix(counts):
        C = np.bincount(counts[1], weights=counts[2], minlength=n_states**2).reshape(n_states, n_states)
        return C, row_normalise(C, cells, n_lon)

    oil_moves = np.load(f"data/oil_moves_{TAG}.npz")
    counts, ops, support = {}, {}, []
    for name in MATRICES + WINDAGE_MATRICES:
        if name in DROGUE_TYPES:
            keep = same_drogue(drogue, start, end, stranded, DROGUE_TYPES[name])
            s = start[keep]
            f, to = transition_states(cell_flat, cells, s, end[keep], lon, lat, args.box, stranded[keep])
            key = traj_idx[s].astype(np.int64) * n_base**2 + f * n_base + to
            uniq, n_moves = np.unique(key, return_counts=True)
            traj, ft = uniq // n_base**2, uniq % n_base**2
        else:
            traj, ft, n_moves = (oil_moves[f"{name}_{k}"] for k in ("traj", "ft", "n"))
            if int(oil_moves["n_states"]) != n_base:
                raise SystemExit("data/oil_moves_….npz does not match the matrices; rerun oil_wind.")
        counts[name] = split_moves(traj, ft, n_moves)
        C, (P, empty) = matrix(counts[name])

        # The split must reproduce the saved operator exactly
        P0 = base[name]["P"]
        same_rest = np.allclose(P[:n_cells, :strand_col], P0[:n_cells, :strand_col])
        same_strand = np.allclose(P[:n_cells, strand_col:].sum(axis=1), P0[:n_cells, strand_col])
        if not (same_rest and same_strand and np.array_equal(np.sort(empty), np.where(base[name]["empty"])[0])):
            raise SystemExit(f"The coast split of {name} does not reproduce {operator_path(name, RES, TAU, args.box)}.")

        op = dict(base[name], P=P, C=C, exit_labels=COAST_EXIT_LABELS, coast_names=COAST_NAMES)
        out = operator_path(f"{name}_coast", RES, TAU, args.box)
        np.savez_compressed(out, **op)
        ops[name] = load_operator(out)

        strand_traj = traj[ft % n_base == strand_col]
        for k, label in enumerate(COAST_LABELS):
            n_dr = len(np.unique(strand_traj[coast_of_traj[strand_traj] == k]))
            support.append({"matrix": name, "coast": label, "grounded_drifters": n_dr,
                            "thin": n_dr < MIN_DRIFTERS})
        print(f"{name:<10} split into {len(COAST_LABELS)} coasts; matches the saved operator → {out}")
    support = pd.DataFrame(support)

    # Per-drifter counts on the coast layout, for the traffic-weighted bootstrap in traffic_exposure
    out_counts = f"data/coast_counts_{TAG}.npz"
    np.savez_compressed(out_counts, n_states=n_states, **{
        f"{name}_{part}": value for name, c in counts.items()
        for part, value in zip(("d_idx", "ft", "n", "n_drifters"), c)})

    # ── 3. Stranding by coast: eventual, by horizon, time to strand ──────────
    points = pd.read_csv(release_points_path(args.box))[["name", "lon", "lat"]]
    release_states = np.array([state_of(ops["drogued"], r.lon, r.lat) for r in points.itertuples()])
    steps = {d: int(round(d / TAU)) for d in HORIZONS}
    coast_cols = slice(strand_col, n_states)

    def strand_summary(P, rows=None):
        """Eventual stranding per coast and expected days to strand (among
    material that strands) for ocean states `rows` (all if None), plus
    stranding per coast within each horizon for the release points."""
        fate, visits = stranding_chain(P, n_cells, STRANDED)
        b = fate[:, STRANDED:].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            days = np.where(b > 0, visits / b * TAU, np.nan)
        p = np.zeros((len(release_states), n_states))
        p[np.arange(len(release_states)), release_states] = 1.0
        by_horizon = {}
        for step in range(1, max(steps.values()) + 1):
            p = p @ P
            for d, k in steps.items():
                if step == k:
                    by_horizon[d] = p[:, coast_cols].copy()
        rows = slice(None) if rows is None else rows
        return fate[rows, STRANDED:], days[rows], by_horizon

    rng = np.random.default_rng(SEED)
    full, boot = {}, {}
    for name in MATRICES:
        full[name] = strand_summary(ops[name]["P"])
        draws = {"eventual": [], "days": [], **{d: [] for d in HORIZONS}}
        for _ in range(N_BOOT):
            Pb = row_normalise(resampled_counts(counts[name], n_states, rng), cells, n_lon)[0]
            ev, days, hz = strand_summary(Pb, release_states)
            draws["eventual"].append(ev)
            draws["days"].append(days)
            for d in HORIZONS:
                draws[d].append(hz[d])
        boot[name] = {k: np.array(v) for k, v in draws.items()}
        print(f"{name:<10} {counts[name][3]:,} drifters resampled {len(boot[name]['days'])} times")

    # ── 4. Tables ────────────────────────────────────────────────────────────

    point_rows = []
    for name in MATRICES:
        ev_full, days_full, hz_full = full[name]
        b = boot[name]
        estimates = [("eventual", ev_full[release_states], b["eventual"])]
        estimates += [(d, hz_full[d], b[d]) for d in HORIZONS]
        for horizon, est, draws in estimates:
            any_est, any_draws = est.sum(axis=1), draws.sum(axis=2)
            for k, label in enumerate(np.r_[COAST_LABELS, ["any"]]):
                e = any_est if label == "any" else est[:, k]
                lo, hi = interval(any_draws if label == "any" else draws[:, :, k])
                for i, r in enumerate(points.itertuples()):
                    point_rows.append({"name": r.name, "lon": r.lon, "lat": r.lat, "matrix": name,
                                       "horizon": horizon, "coast": label, "estimate": e[i],
                                       "lo": lo[i], "hi": hi[i]})
        lo, hi = interval(b["days"])
        for i, r in enumerate(points.itertuples()):
            point_rows.append({"name": r.name, "lon": r.lon, "lat": r.lat, "matrix": name,
                               "horizon": "days_to_strand", "coast": "any",
                               "estimate": days_full[release_states][i], "lo": lo[i], "hi": hi[i]})
    point_table = pd.DataFrame(point_rows)

    cell_lon, cell_lat = cell_centres(ops["drogued"])
    cell_rows = []
    for name in MATRICES:
        ev, days, _ = full[name]
        table = pd.DataFrame(ev, columns=COAST_LABELS)
        table.insert(0, "matrix", name)
        table.insert(1, "lon", cell_lon)
        table.insert(2, "lat", cell_lat)
        table.insert(3, "reliable", reliable(ops[name]))
        table["any"] = ev.sum(axis=1)
        table["days_to_strand"] = days
        cell_rows.append(table)
    cell_table = pd.concat(cell_rows)

    out_points = f"data/coastal_stranding_points_{TAG}.csv"
    out_cells = f"data/coastal_stranding_cells_{TAG}.csv"
    out_support = f"data/coastal_stranding_support_{TAG}.csv"
    point_table.to_csv(out_points, index=False)
    cell_table.to_csv(out_cells, index=False)
    support.to_csv(out_support, index=False)

    # ── 5. Map: where material from each cell strands ────────────────────────
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        subplot_kw = {"projection": ccrs.PlateCarree()}
    except ImportError:
        ccrs = None
        subplot_kw = {}
    kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
    MIN_SHOWN, SOLID = 0.05, 0.20       # pale from 5% eventual stranding, solid from 20%
    cmap = ListedColormap(COAST_COLORS)
    norm = BoundaryNorm(np.arange(len(COAST_LABELS) + 1) - 0.5, len(COAST_LABELS))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), subplot_kw=subplot_kw)
    for ax, name in zip(axes.ravel(), MATRICES):
        ev = full[name][0]
        total = ev.sum(axis=1)
        rel = reliable(ops[name])
        for lo, hi, alpha in [(MIN_SHOWN, SOLID, 0.35), (SOLID, np.inf, 1.0)]:
            top = np.where(rel & (total >= lo) & (total < hi), ev.argmax(axis=1), np.nan).astype(float)
            ax.pcolormesh(ops[name]["lon_edges"], ops[name]["lat_edges"], to_grid(ops[name], top),
                          cmap=cmap, norm=norm, alpha=alpha, linewidth=0, **kw)
        if ccrs:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.set_title(f"{name}: ≥ {MIN_SHOWN:.0%} strands from {(rel & (total >= MIN_SHOWN)).sum()} "
                     f"of {rel.sum()} reliable cells, ≥ {SOLID:.0%} from {(rel & (total >= SOLID)).sum()}",
                     fontsize=10)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COAST_COLORS, COAST_NAMES)],
               title=f"Coast that receives most stranding (solid: ≥ {SOLID:.0%} eventually strands; "
                     f"pale: {MIN_SHOWN:.0%}–{SOLID:.0%}; blank: less, or unreliable cell)",
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Where material from each cell eventually strands — {RES:g}°, τ = {TAU:g} d", fontsize=12)
    fig.subplots_adjust(bottom=0.12, top=0.93, hspace=0.15, wspace=0.05)
    out_png = f"figures/agulhas_coastal_stranding_{TAG}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. Summary ───────────────────────────────────────────────────────────
    pd.set_option("display.width", 220)
    print(f"\nGrounded drifters behind each coast ({MIN_DRIFTERS}+ = not thin):")
    print(support.pivot(index="coast", columns="matrix", values="grounded_drifters")
          .reindex(index=COAST_LABELS, columns=MATRICES).to_string())

    print("\nEventual stranding by coast, mean over reliable cells:")
    mean_rows = {name: [f"{x:.1%}" for x in full[name][0][reliable(ops[name])].mean(axis=0)]
                 for name in MATRICES}
    print(pd.DataFrame(mean_rows, index=COAST_LABELS).to_string())

    for name in ["drogued", "undrogued", "oil"]:
        sub = point_table[(point_table.matrix == name) & (point_table.horizon == "eventual")]
        show = sub.pivot(index="name", columns="coast", values="estimate")
        lo = sub.pivot(index="name", columns="coast", values="lo")
        hi = sub.pivot(index="name", columns="coast", values="hi")
        cols = [c for c in COAST_LABELS if show[c].max() >= 0.02] + ["any"]
        table = pd.DataFrame({c: [fmt(show.at[n, c], lo.at[n, c], hi.at[n, c]) for n in points.name]
                              for c in cols}, index=points.name)
        days = point_table[(point_table.matrix == name) & (point_table.horizon == "days_to_strand")]
        table["days to strand"] = [f"{e:.0f} ({a:.0f}–{b:.0f})" for e, a, b in
                                   days.set_index("name").loc[points.name, ["estimate", "lo", "hi"]].values]
        print(f"\n{name}: eventual stranding by coast (coasts with ≥ 2% from some point), 95% interval")
        print(table.to_string())

    print(f"\nSaved → {out_points}, {out_cells}, {out_support}, {out_counts}, {out_png}")
    print("Coast operators → " + ", ".join(operator_path(f"{n}_coast", RES, TAU, args.box) for n in MATRICES))


if __name__ == "__main__":
    main()
