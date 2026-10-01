"""
Agulhas Region — Oil Transition Matrix (drifters + ERA5 windage)
Surface oil moves at the surface current plus ALPHA_OIL of the 10 m wind.
Undrogued drifters already carry part of that windage, so the oil matrix is
built from undrogued transitions whose end points are moved by the windage
they lack:

    shift = (ALPHA_OIL − α_undrogued) × mean ERA5 wind along the pair × τ

α_undrogued is measured, not assumed: per cell, the slip (undrogued −
drogued mean velocity) is regressed on the mean wind at the undrogued
positions. Drogued drifters are treated as windage-free.

A check matrix applies the full ALPHA_OIL to drogued transitions instead.
The two should agree where both have enough data; a large disagreement at a
release point means its oil result rests on the method choice.

Moves into the stranded state (the drifter ran aground inside R within τ)
are kept as they are. Shifted end points that land on land or on a cell no
drifter visited keep their unshifted end; their share is printed. Surface mass decays as m(t) = exp(−t / DECAY_DAYS), applied when
reporting, so P itself conserves probability like the drifter matrices.

    .venv/bin/python scripts/09_oil_wind.py --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr

from config import MIN_DRIFTERS, box_tag, grid_shape, operator_path, parse_args
from transport import (EXIT_LABELS, STRANDED, coastal_states, load_operator, move_indices,
                       propagate, release, row_normalise, strand_times, to_grid,
                       transition_states)

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
WIND_PATH = "data/drifter_wind10m.nc"   # written by 08
ALPHA_OIL = 0.035                        # oil drift = surface current + 3.5% of wind
DECAY_DAYS = 14.0                        # surface oil e-folding time
HORIZONS = [7, 30, 365]
DISAGREE_TVD = 0.3                       # A vs B at 30 d above this is flagged
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
g = per.groupby(["drogued", "cell"])
cm = g[["u", "v", "wu", "wv"]].mean().assign(n=g.size())
dro, und = cm.xs(True, level="drogued"), cm.xs(False, level="drogued")
both = dro.join(und, lsuffix="_d", rsuffix="_u", how="inner")
both = both[(both.n_d >= MIN_DRIFTERS) & (both.n_u >= MIN_DRIFTERS)]
su, sv = both.u_u - both.u_d, both.v_u - both.v_d          # slip
wu, wv = both.wu_u, both.wv_u                              # wind at undrogued positions


def fit_alpha(idx):
    return float((su.values[idx] * wu.values[idx] + sv.values[idx] * wv.values[idx]).sum()
                 / (wu.values[idx] ** 2 + wv.values[idx] ** 2).sum())


rng = np.random.default_rng(0)
all_idx = np.arange(len(both))
alpha_u = fit_alpha(all_idx)
alpha_ci = np.percentile([fit_alpha(rng.choice(all_idx, len(all_idx))) for _ in range(1000)], [2.5, 97.5])
alpha_extra = ALPHA_OIL - alpha_u
print(f"Undrogued windage α = {100 * alpha_u:.2f}% of 10 m wind "
      f"(95% over cells {100 * alpha_ci[0]:.2f}–{100 * alpha_ci[1]:.2f}%, {len(both)} cells)")
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
    return out


path_a = build("oil", False, alpha_extra)
path_b = build("oil_check", True, ALPHA_OIL)

# ── 4. Compare undrogued, oil (A) and check (B) at the release points ────────
ops = {"undrogued": load_operator(operator_path("undrogued", RES, TAU, args.box)),
       "oil": load_operator(path_a), "oil_check": load_operator(path_b)}
n_cells = len(cells)
coast = coastal_states(ops["oil"])
points = pd.read_csv("data/release_candidates.csv")[["name", "lon", "lat"]]
steps = {d: int(round(d / TAU)) for d in HORIZONS}

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
                    f"{key}_W_365d": p365[n_cells], f"{key}_E_365d": p365[n_cells + 1],
                    f"{key}_stranded_30d": dist[(key, 30)][n_cells + STRANDED],
                    f"{key}_stranded_365d": p365[n_cells + STRANDED]})
    for d in [7, 30]:
        row[f"tvd_oil_check_{d}d"] = 0.5 * np.abs(dist[("oil", d)] - dist[("oil_check", d)]).sum()
        row[f"tvd_oil_undrogued_{d}d"] = 0.5 * np.abs(dist[("oil", d)] - dist[("undrogued", d)]).sum()
    rows.append(row)
cmp_ = pd.DataFrame(rows)
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
titles = {"undrogued": "Undrogued (≈1% wind)", "oil": f"Oil A: undrogued + {100 * alpha_extra:.1f}% wind",
          "oil_check": f"Check B: drogued + {100 * ALPHA_OIL:.1f}% wind"}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), subplot_kw=subplot_kw)
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
    ax.set_title(f"{titles[key]}\nin R {p[:n_cells].sum():.0%}", fontsize=10)
cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="both")
cb.set_label("Probability per 1° cell after 30 days (log)")
fig.suptitle("Durban release after 30 days — undrogued, oil (A) and check (B)", fontsize=12)
out = f"figures/agulhas_oil_durban_30d_{TAG}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Map saved → {out}")
plt.close()
