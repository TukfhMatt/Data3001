"""
Agulhas Region — Oil Transition Matrix (drifters + ERA5 windage)
Surface oil moves at the surface current plus ALPHA_OIL of the 10 m wind.
Undrogued drifters already carry part of that windage, so the oil matrix is
built from undrogued transitions whose end points are moved by the windage
they lack:

    shift = (ALPHA_OIL − α_undrogued) × mean ERA5 wind along the pair × τ

α_undrogued is measured, not assumed: per cell, the slip (undrogued −
drogued mean velocity) is regressed on the mean wind at the undrogued
positions, with a 95% interval from resampling whole drifters. Drogued
drifters are treated as windage-free. The slip holds everything that moves
the surface differently from 15 m (direct windage, wave-driven Stokes drift
and wind-driven shear in the top 15 m), so ALPHA_OIL is taken as oil's drift
relative to the 15 m current.

A check matrix applies the full ALPHA_OIL to drogued transitions instead. Both
are also built across the range of oil windage in the literature (3–4% of the
wind), and a release point where oil and its check disagree strongly at 30 days
is flagged as depending on the method.
The two should agree where both have enough data; a large disagreement at a
release point means its oil result rests on the method choice.

Moves into the stranded state (the drifter ran aground inside R within τ)
are kept as they are. Shifted end points that land on land or on a cell no
drifter visited keep their unshifted end; their share is printed. Surface
mass decays as m(t) = exp(−t / DECAY_DAYS), applied when reporting, so P
itself conserves probability like the drifter matrices.

Release-point results carry 95% intervals from resampling whole drifters
(the oil matrix here, undrogued from bootstrap's output), and the number of
grounded drifters behind each release cell's stranding, since stranding
rests on far fewer drifters than the row's drifter count.

    python -m agulhas oil_wind --res 1 --tau 3.5
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr

from agulhas.config import (DISAGREE_TVD, MIN_DRIFTERS, OIL_WINDAGE_RANGE, operator_path, parse_args, region_prefix,
                            release_points_path, setting_tag, wind_paths)
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.transport import (EXIT_LABELS, STRANDED, coastal_states, drifter_counts, load_operator, propagate,
                               release, resampled_counts, row_normalise, same_drogue, state_of, to_grid,
                               transition_states)

ALPHA_OIL = OIL_WINDAGE_RANGE["oil"]     # oil drift = surface current + 3.5% of wind
DECAY_DAYS = 14.0                        # surface oil e-folding time
HORIZONS = [7, 30, 365]
FAST_CURRENT = 0.3                       # m/s of mean drogued current, for the check of α without fast cells
EAST_LON = 35.0                          # for the check of α east of the Agulhas Current
N_BOOT = 1000                            # drifter resamples for α and the oil intervals
SEED = 0
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0
SOURCE = "NOAA GDP hourly v2.01 + ERA5 10 m wind (Copernicus CDS)"


def load_winds(path, drifters):
    """ERA5 10 m wind at every observation of the subset, from fetch_era5. Stops if the
    file was built for another subset: winds are matched to positions by order alone."""
    wind = xr.open_dataset(path)
    u10 = wind["u10"].values.astype(float)
    v10 = wind["v10"].values.astype(float)
    built_for = wind.attrs.get("subset_fingerprint")
    if len(u10) != len(drifters.lon) or (built_for and built_for != drifters.fingerprint):
        raise SystemExit(f"{path} was built for another subset than {drifters.path}: "
                         "delete it and rerun fetch_era5.")
    return u10, v10


# ── Undrogued windage α from slip against wind ───────────────────────────────

class SlipFit:
    """Per-drifter mean velocity and wind per cell and drogue type, from which α is
    fitted for any weighting of the drifters. Per-drifter means come first, so a
    drifter that loiters does not dominate a cell."""

    def __init__(self, moves, u10, v10):
        d = moves.drifters
        ve, vn = d.ve, d.vn
        m = moves.in_R & np.isfinite(ve) & np.isfinite(vn) & np.isfinite(u10)
        df = pd.DataFrame({"cell": moves.cell_flat[m], "traj": d.traj_idx[m], "drogued": d.drogue[m],
                           "u": ve[m], "v": vn[m], "wu": u10[m], "wv": v10[m]})
        self.per = per = df.groupby(["drogued", "cell", "traj"]).mean().reset_index()
        # Per drogue type and cell: the weighted mean of its drifters' means, so the
        # bootstrap can reweight drifters without regrouping
        self.group = per["cell"].values * 2 + per["drogued"].values.astype(int)
        self.n_groups = 2 * (per["cell"].max() + 1)
        self.traj_ids, self.traj_pos = np.unique(per["traj"].values, return_inverse=True)
        dro_g, und_g = np.arange(1, self.n_groups, 2), np.arange(0, self.n_groups, 2)
        n_per = np.bincount(self.group, minlength=self.n_groups)
        # Cells with ≥ MIN_DRIFTERS of each type; the set is fixed by the data, so
        # the bootstrap varies only the drifters inside it
        self.fit_d = dro_g[(n_per[dro_g] >= MIN_DRIFTERS) & (n_per[und_g] >= MIN_DRIFTERS)]
        self.fit_u = self.fit_d - 1

    def alpha(self, w, keep=None):
        """α from drifter weights w (1 each for the data, bootstrap counts otherwise):
        slip (undrogued − drogued cell mean) regressed through the origin on the
        wind at the undrogued positions, over the fitting cells (or those in keep)."""
        wt = w[self.traj_pos]
        tot = np.maximum(np.bincount(self.group, weights=wt, minlength=self.n_groups), 1e-12)
        mean = {c: np.bincount(self.group, weights=wt * self.per[c].values, minlength=self.n_groups) / tot
                for c in ["u", "v", "wu", "wv"]}
        ok = (tot[self.fit_d] >= 1) & (tot[self.fit_u] >= 1)    # a resample can miss every drifter of a cell
        ok = ok if keep is None else ok & keep
        d, u = self.fit_d[ok], self.fit_u[ok]
        su, sv = mean["u"][u] - mean["u"][d], mean["v"][u] - mean["v"][d]
        wu, wv = mean["wu"][u], mean["wv"][u]
        return float((su * wu + sv * wv).sum() / (wu ** 2 + wv ** 2).sum())


def measure_alpha(fit, grid, rng):
    """α with its 95% interval over resampled whole drifters (neighbouring cells share
    drifters, so cells are not independent), and α without fast cells and in the east."""
    ones = np.ones(len(fit.traj_ids))
    alpha_u = fit.alpha(ones)
    n = len(fit.traj_ids)
    alpha_ci = np.percentile([fit.alpha(np.bincount(rng.integers(n, size=n), minlength=n)) for _ in range(N_BOOT)],
                             [2.5, 97.5])
    one = fit.per.groupby(fit.group)[["u", "v"]].mean()      # drifter means per drogue type and cell
    fit_speed = np.hypot(one["u"].reindex(fit.fit_d).values, one["v"].reindex(fit.fit_d).values)
    fit_lon = grid.lon_min + ((fit.fit_d // 2) % grid.n_lon + 0.5) * grid.res
    alpha_slow = fit.alpha(ones, fit_speed <= FAST_CURRENT)
    alpha_east = fit.alpha(ones, fit_lon >= EAST_LON)
    print(f"Check: without cells of mean drogued current above {FAST_CURRENT:g} m/s α = {100 * alpha_slow:.2f}% "
          f"({(fit_speed <= FAST_CURRENT).sum()} cells); east of {EAST_LON:g}°E α = {100 * alpha_east:.2f}% "
          f"({(fit_lon >= EAST_LON).sum()} cells)")
    print(f"Undrogued windage α = {100 * alpha_u:.2f}% of 10 m wind "
          f"(95% over drifters {100 * alpha_ci[0]:.2f}–{100 * alpha_ci[1]:.2f}%, {len(fit.fit_d)} cells)")
    print(f"Oil windage {100 * ALPHA_OIL:.1f}% → added to undrogued: {100 * (ALPHA_OIL - alpha_u):.2f}%, "
          f"to drogued: {100 * ALPHA_OIL:.1f}%")
    return alpha_u


# ── Oil matrices: drifter moves with wind-shifted end points ─────────────────

class OilMoves:
    """Every move from R, split into stranding moves (kept as they are) and the
    rest, with the mean wind along each of the rest."""

    def __init__(self, moves, u10, v10, alpha_u, args):
        self.d, self.grid, self.args, self.alpha_u = moves.drifters, moves.grid, args, alpha_u
        self.cell_flat = moves.cell_flat
        self.cells = np.unique(self.cell_flat[moves.in_R])        # same ocean cells as transition_matrix
        self.n_states = len(self.cells) + len(EXIT_LABELS)
        stranded = moves.stranded
        self.strand_s, self.strand_e = moves.start[stranded], moves.end[stranded]
        self.start, self.end = moves.start[~stranded], moves.end[~stranded]
        # Mean wind over the positions from start up to (not including) end; the
        # subset is sorted by (trajectory, time), so these are the pair's own path
        ok_w = np.isfinite(u10)
        cs_u = np.concatenate([[0], np.cumsum(np.where(ok_w, u10, 0))])
        cs_v = np.concatenate([[0], np.cumsum(np.where(ok_w, v10, 0))])
        cs_n = np.concatenate([[0], np.cumsum(ok_w)])
        n_w = cs_n[self.end] - cs_n[self.start]
        self.mean_u = np.where(n_w > 0, (cs_u[self.end] - cs_u[self.start]) / np.maximum(n_w, 1), np.nan)
        self.mean_v = np.where(n_w > 0, (cs_v[self.end] - cs_v[self.start]) / np.maximum(n_w, 1), np.nan)
        self.has_wind = np.isfinite(self.mean_u)
        print(f"Pairs: {len(self.start):,}; with wind along the path: {self.has_wind.mean():.1%}")

    def shifted_ends(self, same, alpha):
        """End points moved by alpha of the wind over τ. Ends pushed onto land or an
        unvisited cell keep their drifter end. (lon, lat, cell, kept, shift in km)."""
        d, e = self.d, self.end[same]
        tau_s = self.args.tau * 86400
        dlat = alpha * self.mean_v[same] * tau_s / M_PER_DEG_LAT
        dlon = alpha * self.mean_u[same] * tau_s / (M_PER_DEG_LON_EQ * np.cos(np.radians(d.lat[e])))
        lo2, la2 = d.lon[e] + dlon, d.lat[e] + dlat
        c2 = self.grid.cell_of(lo2, la2)
        beached = (c2 >= 0) & ~np.isin(c2, self.cells)
        lo2, la2 = np.where(beached, d.lon[e], lo2), np.where(beached, d.lat[e], la2)
        c2 = np.where(beached, self.cell_flat[e], c2)
        shift_km = np.hypot(dlon * M_PER_DEG_LON_EQ * np.cos(np.radians(d.lat[e])), dlat * M_PER_DEG_LAT) / 1e3
        return lo2, la2, c2, beached, shift_km

    def states(self, flag, alpha):
        """(from, to) of every move of one drogue type on the oil matrix, the moves'
        starts, the stranding moves kept, and the shifted ends."""
        d, box = self.d, self.args.box
        same = (d.drogue[self.start] == flag) & (d.drogue[self.end] == flag) & self.has_wind
        s = self.start[same]
        ends = self.shifted_ends(same, alpha)
        lo2, la2, c2 = ends[:3]
        # Extend the obs arrays with the shifted ends so the shared helper applies
        n = len(d.lon)
        ext_cell = np.concatenate([self.cell_flat, c2])
        ext_lon, ext_lat = np.concatenate([d.lon, lo2]), np.concatenate([d.lat, la2])
        from_state, to_state = transition_states(ext_cell, self.cells, s, n + np.arange(len(s)), ext_lon, ext_lat, box)
        # Stranding moves of this drogue type, unshifted
        k = same_drogue(d.drogue, self.strand_s, self.strand_e, True, flag)
        f2, t2 = transition_states(self.cell_flat, self.cells, self.strand_s[k], self.strand_e[k], d.lon, d.lat, box,
                                   np.ones(k.sum(), bool))
        return np.r_[from_state, f2], np.r_[to_state, t2], np.r_[s, self.strand_s[k]], k, ends

    def build(self, name, flag, alpha, alpha_oil):
        """Build, save and describe one oil matrix: (path, per-drifter counts, moves
        per (drifter, from, to) for coastal_stranding)."""
        d, args, cells = self.d, self.args, self.cells
        from_state, to_state, s, k, (_, _, _, beached, shift_km) = self.states(flag, alpha)
        n_cells, n_states = len(cells), self.n_states
        C = np.zeros((n_states, n_states))
        np.add.at(C, (from_state, to_state), 1)
        pairs = np.unique(np.stack([from_state, d.traj_idx[s]]), axis=1)
        row_drifters = np.bincount(pairs[0], minlength=n_states)
        P, empty = row_normalise(C, cells, self.grid.n_lon)
        flagged = row_drifters[:n_cells] < MIN_DRIFTERS
        out = operator_path(name, args.res, args.tau, args.box)
        np.savez_compressed(
            out, P=P, C=C,
            row_drifters=row_drifters[:n_cells], row_obs=C.sum(axis=1)[:n_cells],
            flagged=flagged, empty=np.isin(np.arange(n_cells), empty),
            cell_flat=cells, lon_edges=self.grid.lon_edges, lat_edges=self.grid.lat_edges,
            res=args.res, tau_days=args.tau, exit_labels=EXIT_LABELS, box=np.array(args.box),
            drogued=flag, alpha_added=alpha, alpha_undrogued=self.alpha_u,
            alpha_oil=alpha_oil, decay_days=DECAY_DAYS,
            years=np.array([d.year[s].min(), d.year[s].max()]), source=SOURCE,
        )
        print(f"\n=== {name}: {'drogued' if flag else 'undrogued'} + {100 * alpha:.2f}% wind ===")
        print(f"Transitions          : {len(s):,}")
        print(f"Wind shift per step  : median {np.median(shift_km):.0f} km, 90th pct {np.percentile(shift_km, 90):.0f} km")
        print(f"Kept unshifted (land): {beached.mean():.2%}")
        print(f"Flagged rows         : {flagged.mean():.0%}; empty cells {len(empty)}")
        print(f"Mean exit prob / step: {P[:n_cells, n_cells:n_cells + STRANDED].sum(axis=1).mean():.1%}; "
              f"stranding {P[:n_cells, n_cells + STRANDED].mean():.2%} ({k.sum():,} stranding moves)")
        print(f"Saved → {out}")
        key = d.traj_idx[s].astype(np.int64) * n_states**2 + from_state * n_states + to_state
        uniq, n_moves = np.unique(key, return_counts=True)
        built = {"traj": uniq // n_states**2, "ft": uniq % n_states**2, "n": n_moves}
        return out, drifter_counts(d.traj_idx[s], from_state, to_state, n_states), built


def build_all(oil, path):
    """Oil and its check at 3.5%, then the rest of the windage range; saves every
    matrix's moves for coastal_stranding. Returns the 3.5% oil's counts."""
    _, counts, moves_a = oil.build("oil", False, ALPHA_OIL - oil.alpha_u, ALPHA_OIL)
    moves = {"oil": moves_a, "oil_check": oil.build("oil_check", True, ALPHA_OIL, ALPHA_OIL)[2]}
    for name, alpha_oil in OIL_WINDAGE_RANGE.items():
        if name != "oil":
            moves[name] = oil.build(name, False, alpha_oil - oil.alpha_u, alpha_oil)[2]
            check = name.replace("oil", "oil_check")
            moves[check] = oil.build(check, True, alpha_oil, alpha_oil)[2]
    np.savez_compressed(path, **{f"{name}_{k}": v for name, m in moves.items() for k, v in m.items()},
                        n_states=oil.n_states)
    return counts


# ── Undrogued, oil (A) and check (B) at the release points ───────────────────

def oil_fates(P, starts, n_cells, tau):
    """Share stranded after 30 days and still afloat in R after 365 days, per release point."""
    steps = {d: int(round(d / tau)) for d in HORIZONS}
    p = np.zeros((len(starts), P.shape[0]))
    p[np.arange(len(starts)), starts] = 1.0
    for step in range(1, steps[365] + 1):
        p = p @ P
        if step == steps[30]:
            stranded_30 = p[:, n_cells + STRANDED].copy()
    return stranded_30, p[:, :n_cells].sum(axis=1)


def oil_intervals(op, counts, starts, tau, rng):
    """Drifter bootstrap of the oil matrix, as bootstrap does for drogued and undrogued."""
    n_cells, n_states = len(op["cell_flat"]), op["P"].shape[0]
    n_lon = len(op["lon_edges"]) - 1
    boot = np.empty((N_BOOT, 2, len(starts)))
    for b in range(N_BOOT):
        boot[b] = oil_fates(row_normalise(resampled_counts(counts, n_states, rng), op["cell_flat"], n_lon)[0],
                            starts, n_cells, tau)
    print(f"\nOil matrix: {counts[3]:,} drifters resampled {N_BOOT} times")
    return np.percentile(boot, [2.5, 97.5], axis=0)


def grounded_drifters(counts, starts, n_cells, n_states):
    """Grounded drifters behind each release cell's own stranding moves (the same
    moves in the undrogued and oil matrices, since stranding moves are not shifted)."""
    d_idx, ft = counts[0], counts[1]
    return [len(np.unique(d_idx[(ft // n_states == st) & (ft % n_states == n_cells + STRANDED)])) for st in starts]


def undrogued_intervals(path, points):
    """Undrogued intervals from bootstrap, or None if it has not run."""
    if not os.path.exists(path):
        print(f"{path} not found: run bootstrap for undrogued intervals")
        return None
    bt = pd.read_csv(path)
    bt = bt[bt.drogue == "undrogued"].set_index(["metric", "days", "name"])
    return {key: bt.loc[key[0], key[1]].reindex(points.name)[["ci_low", "ci_high"]].values
            for key in [("exit_stranded", 30), ("in_R", 365)]}


def point_outcomes(op, point, coast, tau):
    """One matrix from one release point: the distribution at each horizon and the
    material-days in coastal cells over the first 30 days."""
    n_cells = len(op["cell_flat"])
    steps = {d: int(round(d / tau)) for d in HORIZONS}
    p = release(op, point.lon, point.lat)
    exposure, dist = 0.0, {}
    for step in range(1, max(steps.values()) + 1):
        p = p @ op["P"]
        if step <= steps[30]:
            exposure += p[:n_cells][coast].sum() * tau
        for d, k in steps.items():
            if step == k:
                dist[d] = p.copy()
    return dist, exposure


def compare_points(ops, points, tau):
    n_cells = len(ops["oil"]["cell_flat"])
    exit_col = {lab: n_cells + k for k, lab in enumerate(ops["oil"]["exit_labels"])}
    coast = coastal_states(ops["oil"])
    rows = []
    for _, pt in points.iterrows():
        row = {"name": pt["name"], "lon": pt.lon, "lat": pt.lat}
        dist = {}
        for key, op in ops.items():
            by_day, exposure = point_outcomes(op, pt, coast, tau)
            dist.update({(key, d): p for d, p in by_day.items()})
            p365 = dist[(key, 365)]
            row.update({f"{key}_coast_30d": exposure, f"{key}_inR_365d": p365[:n_cells].sum(),
                        f"{key}_W_365d": p365[exit_col["W"]], f"{key}_E_365d": p365[exit_col["E"]],
                        f"{key}_stranded_30d": dist[(key, 30)][exit_col["stranded"]],
                        f"{key}_stranded_365d": p365[exit_col["stranded"]]})
        for d in [7, 30]:
            row[f"tvd_oil_check_{d}d"] = 0.5 * np.abs(dist[("oil", d)] - dist[("oil_check", d)]).sum()
            row[f"tvd_oil_undrogued_{d}d"] = 0.5 * np.abs(dist[("oil", d)] - dist[("undrogued", d)]).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def add_intervals(cmp_, grounded, oil_ci, und_ci):
    cmp_["grounded_drifters_in_cell"] = grounded
    cmp_["method_flag"] = cmp_["tvd_oil_check_30d"] > DISAGREE_TVD
    oil_lo, oil_hi = oil_ci
    for j, (metric, col) in enumerate([("stranded_30d", "exit_stranded"), ("inR_365d", "in_R")]):
        cmp_[f"oil_{metric}_lo"], cmp_[f"oil_{metric}_hi"] = oil_lo[j], oil_hi[j]
        if und_ci is not None:
            days = 30 if metric == "stranded_30d" else 365
            ci = und_ci[(col, days)]
            cmp_[f"undrogued_{metric}_lo"], cmp_[f"undrogued_{metric}_hi"] = ci[:, 0], ci[:, 1]


def print_compare(cmp_, keys):
    print("\nRelease points — 30-day coastal exposure (material-days) and stranding, 1-year fate, A vs B distance")
    print(f"{'':<28}{'coast 30 d':^24}{'stranded 30 d':^24}{'in R 1 yr':^24}{'exit W 1 yr':^24}{'TVD A–B':^14}")
    print(f"{'':<28}" + f"{'undr':>8}{'oil A':>8}{'chk B':>8}" * 4 + f"{'7 d':>7}{'30 d':>7}")
    for _, r in cmp_.iterrows():
        flag = "  ⚠" if r.method_flag else ""
        print(f"{r['name'][:27]:<28}"
              + "".join(f"{r[f'{k}_coast_30d']:>8.1f}" for k in keys)
              + "".join(f"{100 * r[f'{k}_stranded_30d']:>7.1f}%" for k in keys)
              + "".join(f"{100 * r[f'{k}_inR_365d']:>7.0f}%" for k in keys)
              + "".join(f"{100 * r[f'{k}_W_365d']:>7.0f}%" for k in keys)
              + f"{r.tvd_oil_check_7d:>7.2f}{r.tvd_oil_check_30d:>7.2f}{flag}")


def print_intervals(cmp_):
    def ci(r, key, metric):
        lo, hi = r.get(f"{key}_{metric}_lo", np.nan), r.get(f"{key}_{metric}_hi", np.nan)
        return f"{100 * r[f'{key}_{metric}']:.0f}% ({100 * lo:.0f}–{100 * hi:.0f})"

    print("\nEstimate (95% drifter interval); grounded = drifters that ran aground from the release cell")
    print(f"{'':<28}{'stranded 30 d':^44}{'afloat in R 1 yr':^44}{'grounded':>9}")
    print(f"{'':<28}" + f"{'undrogued':>22}{'oil A':>22}" * 2)
    for _, r in cmp_.iterrows():
        print(f"{r['name'][:27]:<28}" + "".join(f"{ci(r, k, m):>22}" for m in ["stranded_30d", "inR_365d"]
                                              for k in ["undrogued", "oil"])
              + f"{r.grounded_drifters_in_cell:>9}")
    print(f"\nSurface oil remaining (exp(−t/{DECAY_DAYS:g} d)): "
          + ", ".join(f"{d} d {np.exp(-d / DECAY_DAYS):.1%}" for d in HORIZONS))


# ── Map: 30-day distribution from one release point for the three matrices ───

def shown_point(points):
    """Durban, or the first release point in a box without Durban."""
    shown = points[points["name"] == "Durban"]
    return (shown if len(shown) else points).iloc[0]


def plot_release(ops, shown, alpha_u, args, path):
    titles = {"undrogued": f"Undrogued ({100 * alpha_u:.1f}% wind, measured)",
              "oil": f"Oil A: undrogued + {100 * (ALPHA_OIL - alpha_u):.1f}% wind",
              "oil_check": f"Check B: drogued + {100 * ALPHA_OIL:.1f}% wind"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.3), subplot_kw=MAP_KW)
    for ax, (key, op) in zip(axes, ops.items()):
        n_cells = len(op["cell_flat"])
        p = propagate(op, release(op, shown.lon, shown.lat), days=30)
        grid = to_grid(op, p)
        mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], np.where(grid > 0, grid, np.nan), cmap="Blues",
                             norm=LogNorm(1e-4, 1e-1), **TRANSFORM_KW)
        ax.plot(shown.lon, shown.lat, marker="*", ms=12, color="#e8710a", mec="#222222", **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.set_title(f"{titles[key]}\nafloat in R {p[:n_cells].sum():.0%}, stranded {p[n_cells + STRANDED]:.0%}",
                     fontsize=10)
    cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="both")
    cb.set_label("Probability per 1° cell after 30 days (log)")
    fig.suptitle(f"{shown['name']} release after 30 days — undrogued, oil (A) and check (B)", fontsize=12)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_moves = f"data/oil_moves_{tag}.npz"
    out_compare = f"data/oil_compare_{tag}.csv"

    moves = MoveSet(Drifters(args.data), args.box, args.res, args.tau)
    u10, v10 = load_winds(wind_paths(args.data)[1], moves.drifters)
    print(f"Wind available at {np.isfinite(u10[moves.in_R]).mean():.1%} of positions in R")
    rng = np.random.default_rng(SEED)
    alpha_u = measure_alpha(SlipFit(moves, u10, v10), moves.grid, rng)
    oil = OilMoves(moves, u10, v10, alpha_u, args)
    counts = build_all(oil, out_moves)

    ops = {name: load_operator(operator_path(name, args.res, args.tau, args.box))
           for name in ["undrogued", "oil", "oil_check"]}
    points = pd.read_csv(release_points_path(args.box))[["name", "lon", "lat"]]
    starts = np.array([state_of(ops["oil"], r.lon, r.lat) for r in points.itertuples()])
    oil_ci = oil_intervals(ops["oil"], counts, starts, args.tau, rng)
    grounded = grounded_drifters(counts, starts, len(oil.cells), oil.n_states)
    und_ci = undrogued_intervals(f"data/bootstrap_release_{tag}.csv", points)
    cmp_ = compare_points(ops, points, args.tau)
    add_intervals(cmp_, grounded, oil_ci, und_ci)
    cmp_.to_csv(out_compare, index=False)
    print_compare(cmp_, list(ops))
    print_intervals(cmp_)
    print(f"⚠ = A and B differ by TVD > {DISAGREE_TVD} at 30 d (method_flag). Saved → {out_compare}, {out_moves}")

    shown = shown_point(points)
    slug = re.sub(r"[^a-z0-9]+", "_", shown["name"].lower()).strip("_")
    out_png = f"figures/{region_prefix(args.data)}_oil_{slug}_30d_{tag}.png"
    plot_release(ops, shown, alpha_u, args, out_png)
    print(f"Map saved → {out_png}")


if __name__ == "__main__":
    main()
