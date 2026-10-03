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

A check matrix applies the full ALPHA_OIL to drogued transitions instead.
The two should agree where both have enough data; a large disagreement at a
release point means its oil result rests on the method choice.

Moves into the stranded state (the drifter ran aground inside R within τ)
are kept as they are. Shifted end points that land on land or on a cell no
drifter visited keep their unshifted end; their share is printed. Surface
mass decays as m(t) = exp(−t / DECAY_DAYS), applied when reporting, so P
itself conserves probability like the drifter matrices.

Release-point results carry 95% intervals from resampling whole drifters
(the oil matrix here, undrogued from 06's output), and the number of
grounded drifters behind each release cell's stranding, since stranding
rests on far fewer drifters than the row's drifter count.

    .venv/bin/python scripts/09_oil_wind.py --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr

from config import MIN_DRIFTERS, box_tag, grid_shape, operator_path, parse_args
from transport import (EXIT_LABELS, STRANDED, coastal_states, load_operator, move_indices,
                       propagate, release, row_normalise, state_of, strand_times, to_grid,
                       transition_states)

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
WIND_PATH = "data/drifter_wind10m.nc"   # written by 08
ALPHA_OIL = 0.035                        # oil drift = surface current + 3.5% of wind
DECAY_DAYS = 14.0                        # surface oil e-folding time
HORIZONS = [7, 30, 365]
DISAGREE_TVD = 0.3                       # A vs B at 30 d above this is flagged
N_BOOT = 1000                            # drifter resamples for α and the oil intervals
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0
TAG = f"{f'{RES:g}'.replace('.', 'p')}deg_{f'{TAU:g}'.replace('.', 'p')}d_{box_tag(args.box)}"

# ── 1. Load drifters and winds ───────────────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
ve = ds["ve"].values.astype(float)
vn = ds["vn"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
years = ds["time"].values.astype("datetime64[Y]").astype(int) + 1970

wind = xr.open_dataset(WIND_PATH)
u10 = wind["u10"].values.astype(float)
v10 = wind["v10"].values.astype(float)
if len(u10) != len(lon):
    raise SystemExit(f"{WIND_PATH} does not match {args.data}: rerun 08 for this subset.")

n_lon, n_lat = grid_shape(args.box, RES)
lon_edges = np.minimum(LON_MIN + np.arange(n_lon + 1) * RES, LON_MAX)
lat_edges = np.minimum(LAT_MIN + np.arange(n_lat + 1) * RES, LAT_MAX)


def cell_of(lo, la):
    inside = (lo >= LON_MIN) & (lo < LON_MAX) & (la >= LAT_MIN) & (la < LAT_MAX)
    return np.where(inside, ((la - LAT_MIN) // RES).astype(int) * n_lon
                    + ((lo - LON_MIN) // RES).astype(int), -1)


cell_flat = cell_of(lon, lat)
in_R = cell_flat >= 0
cells = np.unique(cell_flat[in_R])        # same ocean cells as 03
print(f"Wind available at {np.isfinite(u10[in_R]).mean():.1%} of positions in R")

# ── 2. Undrogued windage α from slip vs wind ─────────────────────────────────
# Per-drifter means first, so a drifter that loiters does not dominate a cell
m = in_R & np.isfinite(ve) & np.isfinite(vn) & np.isfinite(u10)
df = pd.DataFrame({"cell": cell_flat[m], "traj": traj_idx[m], "drogued": drogue[m],
                   "u": ve[m], "v": vn[m], "wu": u10[m], "wv": v10[m]})
per = df.groupby(["drogued", "cell", "traj"]).mean().reset_index()
# Per drogue type and cell: the weighted mean of its drifters' means, so the
# bootstrap can reweight drifters without regrouping
group = per["cell"].values * 2 + per["drogued"].values.astype(int)
n_groups = 2 * (per["cell"].max() + 1)
traj_ids, traj_pos = np.unique(per["traj"].values, return_inverse=True)
dro_g, und_g = np.arange(1, n_groups, 2), np.arange(0, n_groups, 2)
n_per = np.bincount(group, minlength=n_groups)
# Cells with ≥ MIN_DRIFTERS of each type; the set is fixed by the data, so
# the bootstrap varies only the drifters inside it
fit_d = dro_g[(n_per[dro_g] >= MIN_DRIFTERS) & (n_per[und_g] >= MIN_DRIFTERS)]
fit_u = fit_d - 1


def fit_alpha(w):
    """α from drifter weights w (1 each for the data, bootstrap counts otherwise):
    slip (undrogued − drogued cell mean) regressed through the origin on the
    wind at the undrogued positions, over the fitting cells."""
    wt = w[traj_pos]
    tot = np.maximum(np.bincount(group, weights=wt, minlength=n_groups), 1e-12)
    mean = {c: np.bincount(group, weights=wt * per[c].values, minlength=n_groups) / tot
            for c in ["u", "v", "wu", "wv"]}
    ok = (tot[fit_d] >= 1) & (tot[fit_u] >= 1)    # a resample can miss every drifter of a cell
    d, u = fit_d[ok], fit_u[ok]
    su, sv = mean["u"][u] - mean["u"][d], mean["v"][u] - mean["v"][d]
    wu, wv = mean["wu"][u], mean["wv"][u]
    return float((su * wu + sv * wv).sum() / (wu ** 2 + wv ** 2).sum())


rng = np.random.default_rng(0)
alpha_u = fit_alpha(np.ones(len(traj_ids)))
# Whole drifters are resampled: neighbouring cells share drifters, so cells are not independent
alpha_ci = np.percentile([fit_alpha(np.bincount(rng.integers(len(traj_ids), size=len(traj_ids)),
                                                minlength=len(traj_ids)))
                          for _ in range(N_BOOT)], [2.5, 97.5])
n_alpha_cells = len(fit_d)
alpha_extra = ALPHA_OIL - alpha_u
print(f"Undrogued windage α = {100 * alpha_u:.2f}% of 10 m wind "
      f"(95% over drifters {100 * alpha_ci[0]:.2f}–{100 * alpha_ci[1]:.2f}%, {n_alpha_cells} cells)")
print(f"Oil windage {100 * ALPHA_OIL:.1f}% → added to undrogued: {100 * alpha_extra:.2f}%, "
      f"to drogued: {100 * ALPHA_OIL:.1f}%")

# ── 3. Pairs, mean wind along each pair, shifted end points ──────────────────
start, end, stranded = move_indices(traj_idx, t, in_R, TAU, strand_times(ds, traj_idx, t, in_R))
strand_s, strand_e = start[stranded], end[stranded]
start, end = start[~stranded], end[~stranded]
# Mean wind over the positions from start up to (not including) end; the
# subset is sorted by (trajectory, time), so these are the pair's own path
ok_w = np.isfinite(u10)
cs_u = np.concatenate([[0], np.cumsum(np.where(ok_w, u10, 0))])
cs_v = np.concatenate([[0], np.cumsum(np.where(ok_w, v10, 0))])
cs_n = np.concatenate([[0], np.cumsum(ok_w)])
n_w = cs_n[end] - cs_n[start]
mean_u = np.where(n_w > 0, (cs_u[end] - cs_u[start]) / np.maximum(n_w, 1), np.nan)
mean_v = np.where(n_w > 0, (cs_v[end] - cs_v[start]) / np.maximum(n_w, 1), np.nan)
has_wind = np.isfinite(mean_u)
print(f"Pairs: {len(start):,}; with wind along the path: {has_wind.mean():.1%}")


def build(name, drogue_flag, alpha):
    same = (drogue[start] == drogue_flag) & (drogue[end] == drogue_flag) & has_wind
    s, e = start[same], end[same]
    tau_s = TAU * 86400
    dlat = alpha * mean_v[same] * tau_s / M_PER_DEG_LAT
    dlon = alpha * mean_u[same] * tau_s / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat[e])))
    lo2, la2 = lon[e] + dlon, lat[e] + dlat
    c2 = cell_of(lo2, la2)
    # End points pushed onto land or an unvisited cell keep their drifter end
    beached = (c2 >= 0) & ~np.isin(c2, cells)
    lo2, la2, c2 = np.where(beached, lon[e], lo2), np.where(beached, lat[e], la2), np.where(beached, cell_flat[e], c2)

    # Extend the obs arrays with the shifted ends so the shared helper applies
    n = len(lon)
    ext_cell = np.concatenate([cell_flat, c2])
    ext_lon, ext_lat = np.concatenate([lon, lo2]), np.concatenate([lat, la2])
    from_state, to_state = transition_states(ext_cell, cells, s, n + np.arange(len(s)),
                                             ext_lon, ext_lat, args.box)
    # Stranding moves of this drogue type, unshifted
    k = (drogue[strand_s] == drogue_flag) & (drogue[strand_e] == drogue_flag)
    f2, t2 = transition_states(cell_flat, cells, strand_s[k], strand_e[k], lon, lat, args.box,
                               np.ones(k.sum(), bool))
    from_state, to_state = np.r_[from_state, f2], np.r_[to_state, t2]
    s = np.r_[s, strand_s[k]]
    n_cells = len(cells)
    n_states = n_cells + len(EXIT_LABELS)
    C = np.zeros((n_states, n_states))
    np.add.at(C, (from_state, to_state), 1)
    pairs = np.unique(np.stack([from_state, traj_idx[s]]), axis=1)
    row_drifters = np.bincount(pairs[0], minlength=n_states)
    P, empty = row_normalise(C, cells, n_lon)
    flagged = row_drifters[:n_cells] < MIN_DRIFTERS
    out = operator_path(name, RES, TAU, args.box)
    np.savez_compressed(
        out, P=P, C=C,
        row_drifters=row_drifters[:n_cells], row_obs=C.sum(axis=1)[:n_cells],
        flagged=flagged, empty=np.isin(np.arange(n_cells), empty),
        cell_flat=cells, lon_edges=lon_edges, lat_edges=lat_edges,
        res=RES, tau_days=TAU, exit_labels=EXIT_LABELS, box=np.array(args.box),
        drogued=drogue_flag, alpha_added=alpha, alpha_undrogued=alpha_u,
        alpha_oil=ALPHA_OIL, decay_days=DECAY_DAYS,
        years=np.array([years[s].min(), years[s].max()]),
        source="NOAA GDP hourly v2.01 + ERA5 10 m wind (Copernicus CDS)",
    )
    shift_km = np.hypot(dlon * M_PER_DEG_LON_EQ * np.cos(np.radians(lat[e])), dlat * M_PER_DEG_LAT) / 1e3
    print(f"\n=== {name}: {'drogued' if drogue_flag else 'undrogued'} + {100 * alpha:.2f}% wind ===")
    print(f"Transitions          : {len(s):,}")
    print(f"Wind shift per step  : median {np.median(shift_km):.0f} km, 90th pct {np.percentile(shift_km, 90):.0f} km")
    print(f"Kept unshifted (land): {beached.mean():.2%}")
    print(f"Flagged rows         : {flagged.mean():.0%}; empty cells {len(empty)}")
    print(f"Mean exit prob / step: {P[:n_cells, n_cells:n_cells + STRANDED].sum(axis=1).mean():.1%}; "
          f"stranding {P[:n_cells, n_cells + STRANDED].mean():.2%} ({k.sum():,} stranding moves)")
    print(f"Saved → {out}")
    # Unique (drifter, from, to) triples with counts, for the bootstrap (as in 06)
    key = traj_idx[s].astype(np.int64) * n_states**2 + from_state * n_states + to_state
    uniq, cnt = np.unique(key, return_counts=True)
    _, d_idx = np.unique(uniq // n_states**2, return_inverse=True)
    return out, {"d_idx": d_idx, "ft": uniq % n_states**2, "n": cnt, "drifter": uniq // n_states**2}


path_a, counts_a = build("oil", False, alpha_extra)
path_b, _ = build("oil_check", True, ALPHA_OIL)

# ── 4. Compare undrogued, oil (A) and check (B) at the release points ────────
ops = {"undrogued": load_operator(operator_path("undrogued", RES, TAU, args.box)),
       "oil": load_operator(path_a), "oil_check": load_operator(path_b)}
n_cells = len(cells)
n_states = n_cells + len(EXIT_LABELS)
exit_col = {lab: n_cells + k for k, lab in enumerate(ops["oil"]["exit_labels"])}
coast = coastal_states(ops["oil"])
points = pd.read_csv("data/release_candidates.csv")[["name", "lon", "lat"]]
steps = {d: int(round(d / TAU)) for d in HORIZONS}
release_states = np.array([state_of(ops["oil"], r.lon, r.lat) for r in points.itertuples()])


def oil_fates(P):
    """Share stranded after 30 days and still afloat in R after 365 days, per release point."""
    p = np.zeros((len(release_states), n_states))
    p[np.arange(len(release_states)), release_states] = 1.0
    for step in range(1, steps[365] + 1):
        p = p @ P
        if step == steps[30]:
            stranded_30 = p[:, exit_col["stranded"]].copy()
    return stranded_30, p[:, :n_cells].sum(axis=1)


# Drifter bootstrap of the oil matrix, as 06 does for drogued and undrogued
boot = np.empty((N_BOOT, 2, len(release_states)))
n_drift = counts_a["d_idx"].max() + 1
for b in range(N_BOOT):
    w = np.bincount(rng.integers(n_drift, size=n_drift), minlength=n_drift)
    Cb = np.bincount(counts_a["ft"], weights=counts_a["n"] * w[counts_a["d_idx"]],
                     minlength=n_states**2).reshape(n_states, n_states)
    boot[b] = oil_fates(row_normalise(Cb, cells, n_lon)[0])
oil_lo, oil_hi = np.percentile(boot, [2.5, 97.5], axis=0)
print(f"\nOil matrix: {n_drift:,} drifters resampled {N_BOOT} times")

# Grounded drifters behind each release cell's own stranding moves (the same
# moves in the undrogued and oil matrices, since stranding moves are not shifted)
strand_to = n_cells + STRANDED
grounded = [len(np.unique(counts_a["drifter"][(counts_a["ft"] // n_states == st)
                                              & (counts_a["ft"] % n_states == strand_to)]))
            for st in release_states]

# Undrogued intervals come from 06
boot_csv = f"data/bootstrap_release_{TAG}.csv"
und_ci = None
if os.path.exists(boot_csv):
    bt = pd.read_csv(boot_csv)
    bt = bt[bt.drogue == "undrogued"].set_index(["metric", "days", "name"])
    und_ci = {key: bt.loc[key[0], key[1]].reindex(points.name)[["ci_low", "ci_high"]].values
              for key in [("exit_stranded", 30), ("in_R", 365)]}
else:
    print(f"{boot_csv} not found: run 06 for undrogued intervals")

rows = []
for _, pt in points.iterrows():
    row = {"name": pt["name"], "lon": pt.lon, "lat": pt.lat}
    dist = {}
    for key, op in ops.items():
        p = release(op, pt.lon, pt.lat)
        exposure = 0.0   # material-days in coastal cells over the first 30 days
        for step in range(1, max(steps.values()) + 1):
            p = p @ op["P"]
            if step <= steps[30]:
                exposure += p[:n_cells][coast].sum() * TAU
            for d, k in steps.items():
                if step == k:
                    dist[(key, d)] = p.copy()
        p365 = dist[(key, 365)]
        row.update({f"{key}_coast_30d": exposure, f"{key}_inR_365d": p365[:n_cells].sum(),
                    f"{key}_W_365d": p365[exit_col["W"]], f"{key}_E_365d": p365[exit_col["E"]],
                    f"{key}_stranded_30d": dist[(key, 30)][exit_col["stranded"]],
                    f"{key}_stranded_365d": p365[exit_col["stranded"]]})
    for d in [7, 30]:
        row[f"tvd_oil_check_{d}d"] = 0.5 * np.abs(dist[("oil", d)] - dist[("oil_check", d)]).sum()
        row[f"tvd_oil_undrogued_{d}d"] = 0.5 * np.abs(dist[("oil", d)] - dist[("undrogued", d)]).sum()
    rows.append(row)
cmp_ = pd.DataFrame(rows)
cmp_["grounded_drifters_in_cell"] = grounded
for j, (metric, col) in enumerate([("stranded_30d", "exit_stranded"), ("inR_365d", "in_R")]):
    cmp_[f"oil_{metric}_lo"], cmp_[f"oil_{metric}_hi"] = oil_lo[j], oil_hi[j]
    if und_ci is not None:
        days = 30 if metric == "stranded_30d" else 365
        ci = und_ci[(col, days)]
        cmp_[f"undrogued_{metric}_lo"], cmp_[f"undrogued_{metric}_hi"] = ci[:, 0], ci[:, 1]
cmp_.to_csv(f"data/oil_compare_{TAG}.csv", index=False)

print(f"\nRelease points — 30-day coastal exposure (material-days) and stranding, 1-year fate, A vs B distance")
print(f"{'':<28}{'coast 30 d':^24}{'stranded 30 d':^24}{'in R 1 yr':^24}{'exit W 1 yr':^24}{'TVD A–B':^14}")
print(f"{'':<28}" + f"{'undr':>8}{'oil A':>8}{'chk B':>8}" * 4 + f"{'7 d':>7}{'30 d':>7}")
for _, r in cmp_.iterrows():
    flag = "  ⚠" if r.tvd_oil_check_30d > DISAGREE_TVD else ""
    print(f"{r['name'][:27]:<28}"
          + "".join(f"{r[f'{k}_coast_30d']:>8.1f}" for k in ops)
          + "".join(f"{100 * r[f'{k}_stranded_30d']:>7.1f}%" for k in ops)
          + "".join(f"{100 * r[f'{k}_inR_365d']:>7.0f}%" for k in ops)
          + "".join(f"{100 * r[f'{k}_W_365d']:>7.0f}%" for k in ops)
          + f"{r.tvd_oil_check_7d:>7.2f}{r.tvd_oil_check_30d:>7.2f}{flag}")

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
print(f"⚠ = A and B differ by TVD > {DISAGREE_TVD} at 30 d. Saved → data/oil_compare_{TAG}.csv")

# ── 5. Map: 30-day distribution from Durban for the three matrices ───────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
durban = points[points["name"] == "Durban"].iloc[0]
titles = {"undrogued": f"Undrogued ({100 * alpha_u:.1f}% wind, measured)", "oil": f"Oil A: undrogued + {100 * alpha_extra:.1f}% wind",
          "oil_check": f"Check B: drogued + {100 * ALPHA_OIL:.1f}% wind"}
fig, axes = plt.subplots(1, 3, figsize=(18, 4.3), subplot_kw=subplot_kw)
for ax, (key, op) in zip(axes, ops.items()):
    p = propagate(op, release(op, durban.lon, durban.lat), days=30)
    grid = to_grid(op, p)
    mesh = ax.pcolormesh(lon_edges, lat_edges, np.where(grid > 0, grid, np.nan), cmap="Blues",
                         norm=LogNorm(1e-4, 1e-1), **kw)
    ax.plot(durban.lon, durban.lat, marker="*", ms=12, color="#e8710a", mec="#222222", **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.set_title(f"{titles[key]}\nafloat in R {p[:n_cells].sum():.0%}, stranded {p[n_cells + STRANDED]:.0%}",
                 fontsize=10)
cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="both")
cb.set_label("Probability per 1° cell after 30 days (log)")
fig.suptitle("Durban release after 30 days — undrogued, oil (A) and check (B)", fontsize=12)
out = f"figures/agulhas_oil_durban_30d_{TAG}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Map saved → {out}")
plt.close()
