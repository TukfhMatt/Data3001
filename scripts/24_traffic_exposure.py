"""
Agulhas Region — Traffic-Weighted Coastal Exposure
If releases follow ship traffic, which coasts are most exposed, from which
shipping routes, and how soon? Releases are spread over the cells of R in
proportion to commercial shipping density (World Bank / IMF AIS, 2015–2021)
and pushed through the coast operators from 23, for all traffic and for six
routes (ROUTE_NAMES).

Weights. The raster's values are undocumented in scale and bimodal: about 1–30
and 1e4–1e7 per pixel, with almost nothing between, and the high mode traces
the shipping lanes. They are used only as a relative density: the mean over a
cell's pixels, capped at the 99th percentile of ocean cells so one port cannot
dominate. Weight on a cell that is not reliable in both drifter matrices moves
to the nearest such cell (as the release points in 04) and keeps its route; the
share moved and how far are reported. Two sensitivity runs change one choice
each: weights from the share of a cell's pixels in the high mode, which does
not depend on the scale, and dropping the weight on unreliable cells instead of
moving it. Pixel coverage drops
about 8-fold at exactly 40°S, a source artifact, so the share of weight south
of 40°S is reported too. The data predate the Red Sea diversions of late 2023.

Reports per release and matrix, with 95% intervals from N_BOOT resamples of
whole drifters (none for the oil check):
  • stranding on each coast, eventually and within 7, 30 and 365 days
  • expected days to strand, among material that strands
  • material-days in each coast's near-shore cells over the first 30 days (per
    pooled coastal segment from 21, without intervals, in a separate table)
  • each route's share of the stranding on each coast
  • for oil, the mass stranded on each coast with exp(−t / decay_days)

Needs the coast operators and counts from 23, and the segments from 21.

    .venv/bin/python scripts/24_traffic_exposure.py --res 1 --tau 3.5
"""

import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio
from rasterio.windows import from_bounds

from config import grid_shape, operator_path, parse_args, setting_tag
from transport import (COAST_LABELS, COAST_NAMES, EXIT_LABELS, STRANDED, cell_centres, coast_of,
                       load_operator, reliable, resampled_counts, row_normalise, stranding_chain,
                       to_grid)

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
TAG = setting_tag(RES, TAU, args.box)
SHIP_ZIP = "data/shipping/shipdensity_commercial.zip"
LANE_MIN = 1e3                  # pixels at or above this are in the raster's high mode
CAP_PERCENTILE = 99
ARTIFACT_LAT = -40.0
N_BOOT = 1000
SEED = 0
HORIZONS = [7, 30, 365]
NEARSHORE_DAYS = 30
BOOT_MATRICES = ["drogued", "undrogued", "oil"]
MATRICES = BOOT_MATRICES + ["oil_check"]
ROUTES = np.array(["cape_west", "cape_east", "natal", "mozambique_channel",
                   "south_madagascar", "east_madagascar"])
ROUTE_NAMES = np.array(["Cape route, west of Cape Agulhas (and west coast)",
                        "Cape route, east of Cape Agulhas",
                        "Natal coast", "Mozambique Channel",
                        "South of Madagascar", "East of Madagascar"])
SERIES_COLORS = {"drogued": "#2a78d6", "undrogued": "#eb6834", "oil": "#1baf7a"}
SERIES_MARKERS = {"drogued": "o", "undrogued": "s", "oil": "D"}

os.makedirs("figures", exist_ok=True)


def route_of(lon, lat):
    """Route index (into ROUTES) of a position: a partition of R."""
    return np.select([lon < 20, (lon < 30) & (lat < -32), lon < 34, (lat >= -24) & (lon < 44),
                      (lon >= 50) | (lat >= -24)], [0, 1, 2, 3, 5], 4)


# ── 1. Shipping density and lane share per grid cell ─────────────────────────
with zipfile.ZipFile(SHIP_ZIP) as z:
    tif = next(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))
with rasterio.open(f"zip://{SHIP_ZIP}!/{tif}") as src:
    window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform).round_offsets().round_lengths()
    raw = src.read(1, window=window).astype(np.float32)
    nodata, px = src.nodata, src.res[0]
if nodata is not None:
    raw[raw == nodata] = 0
raw[~np.isfinite(raw) | (raw < 0)] = 0
raw = raw[::-1]                 # raster rows run north→south; flip so row 0 is LAT_MIN

n_lon, n_lat = grid_shape(args.box, RES)


def block_mean(a):
    """Mean of the pixels in each grid cell; a partial last row/column is
    zero-padded, as in 04, so it keeps the pixels it has."""
    k = int(round(RES / px))
    ny, nx = -(-a.shape[0] // k), -(-a.shape[1] // k)
    a = np.pad(a, ((0, ny * k - a.shape[0]), (0, nx * k - a.shape[1])))
    return a.reshape(ny, k, nx, k).mean(axis=(1, 3), dtype=np.float64)[:n_lat, :n_lon].ravel()


density = block_mean(raw)
lane_share = block_mean(raw >= LANE_MIN)
del raw

# ── 2. Coast operators, release weights ──────────────────────────────────────
ops = {name: load_operator(operator_path(f"{name}_coast", RES, TAU, args.box)) for name in MATRICES}
op0 = ops["drogued"]
cells = op0["cell_flat"]
n_cells = len(cells)
n_states = op0["P"].shape[0]
n_coasts = len(COAST_LABELS)
coast_cols = slice(n_cells + STRANDED, n_states)
exit_labels = EXIT_LABELS[:STRANDED]
cell_lon, cell_lat = cell_centres(op0)

ok_cells = reliable(ops["drogued"]) & reliable(ops["undrogued"])
if not ok_cells.any():
    raise SystemExit("No cell is reliable in both drifter matrices.")
lo_e, la_e = op0["lon_edges"], op0["lat_edges"]
grid_j, grid_i = np.divmod(np.arange(n_lat * n_lon), n_lon)
grid_lon = (lo_e[grid_i] + lo_e[grid_i + 1]) / 2
grid_lat = (la_e[grid_j] + la_e[grid_j + 1]) / 2
grid_route = route_of(grid_lon, grid_lat)
state_of_grid = np.full(n_lat * n_lon, -1)
state_of_grid[cells] = np.arange(n_cells)
ok_state = np.where(ok_cells)[0]


def nearest_ok(k):
    """State of the reliable cell nearest grid cell k, as in 04."""
    dx = (cell_lon[ok_state] - grid_lon[k]) * 111.32 * np.cos(np.radians(grid_lat[k]))
    dy = (cell_lat[ok_state] - grid_lat[k]) * 110.57
    return ok_state[np.argmin(dx ** 2 + dy ** 2)]


def release_weights(values, move=True):
    """Releases × ocean states: all traffic, then each route, each row summing
    to 1, with weight off reliable cells moved to the nearest reliable cell
    (or dropped). Also the route shares, and the share of weight moved with its
    weighted median and 90th percentile distance in km."""
    has = np.where(values > 0)[0]
    on_ok = (state_of_grid[has] >= 0) & ok_cells[state_of_grid[has]]
    if not move:
        has = has[on_ok]
        on_ok = on_ok[on_ok]
    target = np.array([state_of_grid[k] if good else nearest_ok(k) for k, good in zip(has, on_ok)])
    W = np.zeros((len(ROUTES) + 1, n_cells))
    np.add.at(W[0], target, values[has])
    for r in range(len(ROUTES)):
        m = grid_route[has] == r
        np.add.at(W[r + 1], target[m], values[has][m])
    shares = W[1:].sum(axis=1) / W[0].sum()
    dx = (cell_lon[target] - grid_lon[has]) * 111.32 * np.cos(np.radians(grid_lat[has]))
    km = np.hypot(dx, (cell_lat[target] - grid_lat[has]) * 110.57)
    moved_weight = values[has] * ~on_ok
    moved = (0.0, 0.0, 0.0)
    if moved_weight.sum() > 0:
        order = np.argsort(km)
        cum = np.cumsum(moved_weight[order]) / moved_weight.sum()
        moved = (moved_weight.sum() / values[has].sum(),
                 km[order][np.searchsorted(cum, 0.5)], km[order][np.searchsorted(cum, 0.9)])
    return W / W.sum(axis=1, keepdims=True), shares, moved


ocean_grid = state_of_grid >= 0
cap = np.percentile(density[ocean_grid], CAP_PERCENTILE)
capped = np.minimum(density, cap)
weights = {"density": release_weights(capped),
           "lane_share": release_weights(lane_share),
           "reliable_only": release_weights(capped, move=False)}
SENSITIVITY = ["lane_share", "reliable_only"]
W, route_share, moved_stats = weights["density"]
RELEASES = np.r_[["all"], ROUTES]
south_share = W[0][cell_lat < ARTIFACT_LAT].sum()

# ── 3. Near-shore cells of each coast, from the pooled segments of 21 ────────
seg = np.load(f"data/coastal_segments_{TAG}.npz", allow_pickle=True)
segment_of_state = seg["segment_of_state"]
coastal = seg["coastal"].astype(bool)
segment_ids = np.unique(segment_of_state[coastal])
in_segment = np.stack([segment_of_state == s for s in segment_ids], axis=1).astype(float)
coast_of_cell = coast_of(cell_lon, cell_lat)
in_coast = np.stack([coastal & (coast_of_cell == k) for k in range(n_coasts)], axis=1).astype(float)
steps = {d: int(round(d / TAU)) for d in HORIZONS}
near_steps = int(round(NEARSHORE_DAYS / TAU))


# ── 4. Exposure from one matrix and set of releases ──────────────────────────
def exposure(P, W, shares, decay_days=None):
    fate, visits = stranding_chain(P, n_cells, STRANDED)
    wf = W @ fate
    b = wf[:, STRANDED:].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        days = np.where(b > 0, W @ visits / b * TAU, np.nan)
    p = np.zeros((len(W), n_states))
    p[:, :n_cells] = W
    prev = p[:, coast_cols].copy()
    near = np.zeros((len(W), n_cells))
    mass = np.zeros((len(W), n_coasts))
    out = {"eventual": wf[:, STRANDED:], "exit": wf[:, :STRANDED], "days": days}
    for step in range(1, steps[max(HORIZONS)] + 1):
        p = p @ P
        if step <= near_steps:
            near += p[:, :n_cells] * TAU
        new = p[:, coast_cols]
        if decay_days:
            mass += (new - prev) * np.exp(-step * TAU / decay_days)
        prev = new.copy()
        for d, k in steps.items():
            if step == k:
                out[d] = new.copy()
                out[f"in_R_{d}"] = p[:, :n_cells].sum(axis=1)
    out["nearshore"] = near @ in_coast
    out["segments"] = near @ in_segment
    if decay_days:
        out["mass"] = mass
    # Each route's share of the traffic-weighted stranding on each coast
    by_route = shares[:, None] * wf[1:, STRANDED:]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["source"] = by_route / by_route.sum(axis=0, keepdims=True)
    return out


counts_file = np.load(f"data/coast_counts_{TAG}.npz")
if int(counts_file["n_states"]) != n_states:
    raise SystemExit("data/coast_counts_….npz does not match the coast operators; rerun 23.")
decay = {"oil": float(ops["oil"]["decay_days"])}

rng = np.random.default_rng(SEED)
full, sens, intervals = {}, {}, {}
for name in MATRICES:
    full[name] = exposure(ops[name]["P"], W, route_share, decay.get(name))
    sens[name] = {w: exposure(ops[name]["P"], weights[w][0], weights[w][1], decay.get(name))
                  for w in SENSITIVITY}
    if name not in BOOT_MATRICES:
        continue
    counts = tuple(counts_file[f"{name}_{part}"] for part in ("d_idx", "ft", "n"))
    counts += (int(counts_file[f"{name}_n_drifters"]),)
    draws = {}
    for _ in range(N_BOOT):
        Pb = row_normalise(resampled_counts(counts, n_states, rng), cells, n_lon)[0]
        for key, value in exposure(Pb, W, route_share, decay.get(name)).items():
            if key != "segments":
                draws.setdefault(key, []).append(value)
    with np.errstate(all="ignore"):
        intervals[name] = {k: np.nanpercentile(np.array(v), [2.5, 97.5], axis=0) for k, v in draws.items()}
    print(f"{name:<10} {counts[3]:,} drifters resampled {N_BOOT} times")


# ── 5. Tables ────────────────────────────────────────────────────────────────
def rows_for(name, weighting, out, ci):
    """Long table: release, metric, horizon, coast (or exit edge or route)."""
    rows = []

    def add(metric, horizon, target, release, value, lo=np.nan, hi=np.nan):
        rows.append({"matrix": name, "weights": weighting, "release": release, "metric": metric,
                     "horizon": horizon, "target": target, "estimate": float(value),
                     "lo": float(lo), "hi": float(hi)})

    def bounds(key, *index):
        return ci[key][(slice(None),) + index] if ci and key in ci else (np.nan, np.nan)

    for r, release in enumerate(RELEASES):
        for horizon in ["eventual"] + HORIZONS:
            for k, coast in enumerate(COAST_LABELS):
                add("stranded", horizon, coast, release, out[horizon][r, k], *bounds(horizon, r, k))
        add("days_to_strand", "eventual", "any", release, out["days"][r], *bounds("days", r))
        for k, label in enumerate(exit_labels):
            add("exit", "eventual", label, release, out["exit"][r, k], *bounds("exit", r, k))
        for d in HORIZONS:
            add("in_R", d, "R", release, out[f"in_R_{d}"][r], *bounds(f"in_R_{d}", r))
        for k, coast in enumerate(COAST_LABELS):
            add("nearshore_days", NEARSHORE_DAYS, coast, release, out["nearshore"][r, k], *bounds("nearshore", r, k))
            if "mass" in out:
                add("mass_stranded", "eventual", coast, release, out["mass"][r, k], *bounds("mass", r, k))
    for k, coast in enumerate(COAST_LABELS):
        for r, route in enumerate(ROUTES):
            add("source_share", "eventual", coast, route, out["source"][r, k], *bounds("source", r, k))
    return rows


table = pd.DataFrame([row for name in MATRICES
                      for row in rows_for(name, "density", full[name], intervals.get(name))
                      + [r for w in SENSITIVITY for r in rows_for(name, w, sens[name][w], None)]])
out_table = f"data/traffic_exposure_{TAG}.csv"
table.to_csv(out_table, index=False)

segment_table = pd.DataFrame([
    {"matrix": name, "release": release, "segment_id": int(s) + 1,   # numbered from 1 in 21's tables
     "nearshore_days": full[name]["segments"][r, j]}
    for name in MATRICES for r, release in enumerate(RELEASES) for j, s in enumerate(segment_ids)])
out_segments = f"data/traffic_nearshore_segments_{TAG}.csv"
segment_table.to_csv(out_segments, index=False)

weight_table = pd.DataFrame({"state": np.arange(n_cells), "lon": cell_lon, "lat": cell_lat,
                             "reliable_both": ok_cells, "weight_density": W[0],
                             "weight_lane_share": weights["lane_share"][0][0],
                             "route": ROUTES[route_of(cell_lon, cell_lat)]})
out_weights = f"data/traffic_weights_{TAG}.csv"
weight_table.to_csv(out_weights, index=False)


# ── 6. Figure: weights, exposed coasts, route to coast ───────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    map_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs, map_kw = None, {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
INK, MUTED = "#0b0b0b", "#52514e"
short_coasts = ["Atlantic & Cape", "SA south & east", "S Mozambique", "C & N Mozambique",
                "W & S Madagascar", "E Madagascar"]
short_routes = ["All traffic", "Cape west", "Cape east", "Natal", "Moz. Channel", "S of Madag.", "E of Madag."]

fig = plt.figure(figsize=(19, 5.6))
ax = fig.add_subplot(1, 3, 1, **map_kw)
grid = to_grid(op0, np.r_[W[0], np.zeros(n_states - n_cells)])
pos = grid[grid > 0]
mesh = ax.pcolormesh(lo_e, la_e, np.ma.masked_less_equal(grid, 0), cmap="Blues",
                     norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-4), vmax=pos.max()), **kw)
if ccrs:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
for x, y in [([20, 20], [-45, -15]), ([20, 30, 30], [-32, -32, -45]), ([34, 34], [-45, -24]),
             ([34, 50], [-24, -24]), ([44, 44], [-24, -15]), ([50, 50], [-45, -24])]:
    ax.plot(x, y, color=MUTED, lw=0.8, ls="--", zorder=4, **kw)
for label, lo, la in [("Cape W", 14.5, -42), ("Cape E", 25, -42), ("Natal", 31.8, -42),
                      ("Moz. Channel", 39, -22.8), ("S of Madag.", 42, -42), ("E of Madag.", 52.5, -42)]:
    ax.text(lo, la, label, ha="center", fontsize=8, color=INK, zorder=5, **kw)
fig.colorbar(mesh, ax=ax, shrink=0.8, label="Share of traffic-weighted release per cell (log)")
ax.set_title("Where releases start", fontsize=11, color=INK)

ax = fig.add_subplot(1, 3, 2)
y = np.arange(n_coasts)
for offset, name in zip([-0.22, 0, 0.22], BOOT_MATRICES):
    est = full[name]["eventual"][0]
    lo, hi = intervals[name]["eventual"][:, 0]
    ax.hlines(y + offset, lo, hi, color=SERIES_COLORS[name], lw=2)
    ax.scatter(est, y + offset, s=45, marker=SERIES_MARKERS[name], color=SERIES_COLORS[name],
               edgecolor="white", linewidth=1, zorder=3, label=name)
ax.set_yticks(y)
ax.set_yticklabels(short_coasts)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="x", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.set_xlabel("Share of traffic-weighted release that eventually strands there (95% interval)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Which coasts are exposed", fontsize=11, color=INK)

ax = fig.add_subplot(1, 3, 3)
heat = full["oil"]["eventual"]
im = ax.imshow(heat, cmap="Blues", vmin=0, vmax=max(heat.max(), 0.01), aspect="auto")
for r in range(heat.shape[0]):
    for k in range(heat.shape[1]):
        ax.text(k, r, f"{heat[r, k]:.0%}", ha="center", va="center", fontsize=9,
                color="white" if heat[r, k] > 0.6 * heat.max() else INK)
ax.set_xticks(np.arange(n_coasts))
ax.set_xticklabels(short_coasts, rotation=30, ha="right")
ax.set_yticks(np.arange(len(short_routes)))
ax.set_yticklabels(short_routes)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_title("Oil: from each route to each coast", fontsize=11, color=INK)

fig.suptitle(f"Traffic-weighted coastal exposure — {RES:g}°, τ = {TAU:g} d", fontsize=13, color=INK)
fig.tight_layout()
out_png = f"figures/agulhas_traffic_exposure_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()


# ── 7. Summary ───────────────────────────────────────────────────────────────
pd.set_option("display.width", 220)


def fmt(name, key, *index):
    v = full[name][key][index]
    if name not in intervals:
        return f"{v:.0%}"
    lo, hi = intervals[name][key][(slice(None),) + index]
    return f"{v:.0%} ({lo:.0%}–{hi:.0%})"


print(f"\nWeights: cap at the {CAP_PERCENTILE}th percentile of ocean cells; "
      f"{moved_stats[0]:.1%} of weight moved to the nearest reliable cell "
      f"(weighted median {moved_stats[1]:.0f} km, 90th percentile {moved_stats[2]:.0f} km); "
      f"{south_share:.2%} of weight south of {abs(ARTIFACT_LAT):g}°S")
print("Route shares of traffic: " + ", ".join(f"{r} {s:.0%}" for r, s in zip(ROUTES, route_share)))

print("\nAll traffic: stranding on each coast, eventually and within 30 days (95% interval)")
print(pd.DataFrame({f"{name} {h}": [fmt(name, h, 0, k) for k in range(n_coasts)]
                    for name in BOOT_MATRICES for h in ["eventual", 30]}, index=COAST_LABELS).to_string())

print("\nStranding (any coast) and days to strand, by release")
rows = {}
for r, release in enumerate(RELEASES):
    row = {}
    for name in BOOT_MATRICES:
        for h in [30, "eventual"]:
            est = full[name][h][r].sum()
            row[f"{name} {h}"] = f"{est:.0%}"
        lo, hi = intervals[name]["days"][:, r]
        row[f"{name} days"] = f"{full[name]['days'][r]:.0f} ({lo:.0f}–{hi:.0f})"
    rows[release] = row
print(pd.DataFrame(rows).T.to_string())

print("\nOil: eventual stranding from each route on each coast")
print(pd.DataFrame(full["oil"]["eventual"], index=RELEASES, columns=COAST_LABELS)
      .round(3).to_string())

print("\nOil: share of each coast's stranding from each route")
print(pd.DataFrame(full["oil"]["source"], index=ROUTES, columns=COAST_LABELS).round(2).to_string())

print(f"\nNear-shore material-days over {NEARSHORE_DAYS} days, all traffic")
print(pd.DataFrame({name: [f"{full[name]['nearshore'][0, k]:.2f}" for k in range(n_coasts)]
                    for name in MATRICES}, index=COAST_LABELS).to_string())

print(f"\nOil mass stranded with exp(−t / {decay['oil']:g} d), all traffic: "
      + ", ".join(f"{c} {m:.2%}" for c, m in zip(COAST_LABELS, full["oil"]["mass"][0])))

print("\nSensitivity: all-traffic eventual stranding with lane-pixel weights and without moved weight")
print(pd.DataFrame({f"{name} {label}": [f"{out[name]['eventual'][0, k]:.1%}" for k in range(n_coasts)]
                    for name in BOOT_MATRICES
                    for label, out in [("density", full),
                                       ("lanes", {n: sens[n]["lane_share"] for n in BOOT_MATRICES}),
                                       ("dropped", {n: sens[n]["reliable_only"] for n in BOOT_MATRICES})]},
                   index=COAST_LABELS).to_string())

print(f"\nSaved → {out_table}, {out_segments}, {out_weights}, {out_png}")
