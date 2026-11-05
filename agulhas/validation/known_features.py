"""
Agulhas Region — Known Features of the Circulation
Whether the data and the matrices show the circulation described in the plan
and the literature. The ranges are widened by one 1° cell on each side, except
the bearings and the leakage share, and were fixed before this script first ran:

  • current speed    — fastest mean current on the Agulhas core rows
                       (CORE box) is at least MIN_CORE_SPEED m/s
  • current path     — those fastest cells head south-west (bearing 180–270°)
  • release drift    — material released at Durban has moved south-west
                       after 7 days (centre of the part still in R)
  • retroflection    — going west from JET_START, the westward jet (u below
                       −JET_SPEED in the RETRO latitudes) ends at 15–21°E
  • fate divide      — going west from JET_START along the RETURN latitudes,
                       P(exit E) falls below P(exit W) at 15–21°E
  • Return Current   — median latitude of the fastest eastward cell, per
                       longitude over RETURN_LONS, is at 37–41°S
  • rings            — mean current over the Cape Basin (RINGS box) heads
                       north-west (bearing 270–360°)
  • leakage share    — of material from the core rows that exits east or
                       west, the share going west overlaps 21–30%, the range of
                       15 Sv (Richardson 2007) and 21.3 Sv (Daher et al. 2020)
                       against 70 Sv (Bryden et al. 2005)

The rules describe the water, so the drogued result is the test and the
undrogued one is shown beside it: wind also moves undrogued drifters. The
leakage share compares surface material with a transport over the upper
1,000 m, so it is a check of size only. The Cape Basin mean holds the Benguela
Current as well as the rings, so its bearing cannot separate them. Exits are at
the box edges, 10°E and 55°E, so the fate divide also depends on where R was
drawn.

Each number has a 95% interval from N_BOOT resamples of whole drifters: the
mean-current numbers resample each drifter's sums per cell, and the matrix
numbers rebuild the matrix. Only reliable cells are used.

Needs the matrices from 03, the release points from 04 and the subset from 00.

    .venv/bin/python scripts/31_known_features.py --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import xarray as xr

from config import (DROGUE_TYPES, grid_shape, operator_path, parse_args, place_label, region_prefix,
                    release_points_path, setting_tag)
from transport import (EXIT_LABELS, STRANDED, cell_centres, check_coast_box, drifter_counts, load_operator,
                       move_indices, reliable, resampled_counts, row_normalise, same_drogue, state_of, strand_times,
                       stranding_chain, transition_states)

args = parse_args(__doc__, grid=True, lag=True)
check_coast_box(args.box)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
TAG = setting_tag(RES, TAU, args.box)
N_BOOT = 1000
SEED = 0

CORE = (26, 36, -34, -27)        # Agulhas core rows, Durban's latitude either side
MIN_CORE_SPEED = 1.0             # m/s, a 1° mean of a current peaking near 2 m/s
RINGS = (10, 18, -38, -30)       # Cape Basin
RETRO_LATS = (-42, -34)
RETURN_LATS = (-45, -34)
RETURN_LONS = (25, 55)
JET_START = 28                   # °E, in the jet south of the Agulhas Bank
JET_SPEED = 0.25                 # m/s
DRIFT_DAYS = 7
EXPECT = {"core_speed": (MIN_CORE_SPEED, np.inf), "core_bearing": (180, 270), "drift_bearing": (180, 270),
          "retroflection_lon": (15, 21), "fate_divide_lon": (15, 21), "return_lat": (-41, -37),
          "rings_bearing": (270, 360), "leakage_share": (0.21, 0.30)}
OVERLAP = {"leakage_share"}      # passes if the interval overlaps the range, not if the estimate is inside

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(SEED)
n_lon, n_lat = grid_shape(args.box, RES)
WEST, EAST = list(EXIT_LABELS).index("W"), list(EXIT_LABELS).index("E")


def bearing(u, v):
    """Compass bearing the flow heads towards, in degrees."""
    return np.degrees(np.arctan2(u, v)) % 360


def in_box(lon, lat, box):
    return (lon >= box[0]) & (lon < box[1]) & (lat >= box[2]) & (lat < box[3])


# ── 1. Data, and each drifter's velocity sums per cell ───────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
ve = ds["ve"].values.astype(float)
vn = ds["vn"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
in_R = in_box(lon, lat, args.box)
cell_flat = np.where(in_R, ((lat - LAT_MIN) // RES).astype(int) * n_lon
                     + ((lon - LON_MIN) // RES).astype(int), -1)
start, end, stranded = move_indices(traj_idx, t, in_R, TAU, strand_times(ds, traj_idx, t, in_R))
n_grid = n_lon * n_lat


def velocity_sums(flag):
    """Per (drifter, cell) sums of u and v and the observation count."""
    m = in_R & (drogue == flag) & np.isfinite(ve) & np.isfinite(vn)
    key = traj_idx[m].astype(np.int64) * n_grid + cell_flat[m]
    groups, inverse = np.unique(key, return_inverse=True)
    return (groups // n_grid, groups % n_grid, np.bincount(inverse, ve[m]), np.bincount(inverse, vn[m]),
            np.bincount(inverse))


def mean_current(sums, weights, ok_grid):
    """(u, v) on the grid from drifter weights, NaN outside the reliable cells."""
    traj, cell, su, sv, count = sums
    w = weights[traj]
    total = np.bincount(cell, w * count, n_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.bincount(cell, w * su, n_grid) / total
        v = np.bincount(cell, w * sv, n_grid) / total
    return (np.where(ok_grid, u, np.nan).reshape(n_lat, n_lon), np.where(ok_grid, v, np.nan).reshape(n_lat, n_lon))


# ── 2. The features ──────────────────────────────────────────────────────────
grid_lon = LON_MIN + (np.arange(n_lon) + 0.5) * RES
grid_lat = LAT_MIN + (np.arange(n_lat) + 0.5) * RES


def core_peaks(u, v):
    """(row, column) of the fastest cell on each core row."""
    speed = np.hypot(u, v)
    cols = (grid_lon >= CORE[0]) & (grid_lon < CORE[1])
    out = []
    for j in np.where((grid_lat >= CORE[2]) & (grid_lat < CORE[3]))[0]:
        row = np.where(cols, speed[j], np.nan)
        if np.isfinite(row).any():
            out.append((j, int(np.nanargmax(row))))
    return out


def current_features(u, v):
    peaks = core_peaks(u, v)
    rows, cols = np.array(peaks).T
    speed = np.hypot(u, v)

    # The jet ends at the first column west of JET_START with no westward flow
    lats = (grid_lat >= RETRO_LATS[0]) & (grid_lat < RETRO_LATS[1])
    west_edge = np.nan
    for i in np.where(grid_lon < JET_START)[0][::-1]:
        column = u[lats, i]
        if not np.isfinite(column).any() or np.nanmin(column) >= -JET_SPEED:
            break
        west_edge = grid_lon[i]

    lats = (grid_lat >= RETURN_LATS[0]) & (grid_lat < RETURN_LATS[1])
    peak_lat = []
    for i in np.where((grid_lon >= RETURN_LONS[0]) & (grid_lon < RETURN_LONS[1]))[0]:
        column = np.where(lats, u[:, i], np.nan)
        if np.isfinite(column).any():
            peak_lat.append(grid_lat[np.nanargmax(column)])

    rings_lon = (grid_lon >= RINGS[0]) & (grid_lon < RINGS[1])
    rings_lat = (grid_lat >= RINGS[2]) & (grid_lat < RINGS[3])
    ring_u, ring_v = u[np.ix_(rings_lat, rings_lon)], v[np.ix_(rings_lat, rings_lon)]
    return {"core_speed": np.nanmax(speed[rows, cols]),
            "core_bearing": bearing(np.nanmean(u[rows, cols]), np.nanmean(v[rows, cols])),
            "retroflection_lon": west_edge, "return_lat": np.median(peak_lat),
            "rings_bearing": bearing(np.nanmean(ring_u), np.nanmean(ring_v))}


def matrix_features(P, op, ok, core_states, durban):
    n = len(op["cell_flat"])
    clon, clat = cell_centres(op)
    fate = stranding_chain(P, n, STRANDED)[0]

    # Centre of the part still in R after DRIFT_DAYS
    p = np.zeros(P.shape[0])
    p[durban] = 1
    for _ in range(int(round(DRIFT_DAYS / TAU))):
        p = p @ P
    held = p[:n] / p[:n].sum()
    move_e = (held @ clon - clon[durban]) * np.cos(np.radians(clat[durban]))
    move_n = held @ clat - clat[durban]

    # Fate divide: mean P(E) − P(W) per longitude along the Return Current latitudes
    band = ok & (clat >= RETURN_LATS[0]) & (clat < RETURN_LATS[1])
    divide = np.nan
    for x in grid_lon[grid_lon < JET_START][::-1]:
        here = band & np.isclose(clon, x)
        if not here.any():
            continue
        if (fate[here, EAST] - fate[here, WEST]).mean() < 0:
            break
        divide = x

    west, east = fate[core_states, WEST].sum(), fate[core_states, EAST].sum()
    return {"drift_bearing": bearing(move_e, move_n), "fate_divide_lon": divide, "leakage_share": west / (west + east)}


def circular_interval(values):
    """95% interval of bearings, unwrapped about their circular mean."""
    a = np.radians(values[np.isfinite(values)])
    centre = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    off = np.degrees((a - centre + np.pi) % (2 * np.pi) - np.pi)
    lo, hi = np.percentile(off, [2.5, 97.5]) + np.degrees(centre)
    return lo % 360, hi % 360


# ── 3. Full data and resamples, per drogue type ──────────────────────────────
points = pd.read_csv(release_points_path(args.box))
durban_point = points[points.name == "Durban"].iloc[0]
rows, drawn = [], {}
for name, flag in DROGUE_TYPES.items():
    op = load_operator(operator_path(name, RES, TAU, args.box))
    cells, n = op["cell_flat"], len(op["cell_flat"])
    n_states = op["P"].shape[0]
    ok = reliable(op)
    ok_grid = np.zeros(n_grid, bool)
    ok_grid[cells[ok]] = True

    sums = velocity_sums(flag)
    n_traj = ds.sizes["traj"]
    present = np.unique(sums[0])
    u, v = mean_current(sums, np.ones(n_traj), ok_grid)
    peaks = core_peaks(u, v)
    state_of_grid = np.full(n_grid, -1)
    state_of_grid[cells] = np.arange(n)
    core_states = np.array([state_of_grid[j * n_lon + i] for j, i in peaks])
    durban = state_of(op, durban_point.lon, durban_point.lat)
    drawn[name] = (u, v, peaks)

    full = {**current_features(u, v), **matrix_features(op["P"], op, ok, core_states, durban)}

    same = same_drogue(drogue, start, end, stranded, flag)
    s, e = start[same], end[same]
    f, to = transition_states(cell_flat, cells, s, e, lon, lat, args.box, stranded[same])
    counts = drifter_counts(traj_idx[s], f, to, n_states)
    C = np.bincount(counts[1], weights=counts[2], minlength=n_states**2).reshape(n_states, n_states)
    assert np.array_equal(C, op["C"]), f"rebuilt counts differ from the saved {name} matrix"

    samples = {k: np.empty(N_BOOT) for k in full}
    for b in range(N_BOOT):
        w = np.zeros(n_traj)
        w[present] = np.bincount(rng.integers(len(present), size=len(present)), minlength=len(present))
        bu, bv = mean_current(sums, w, ok_grid)
        result = current_features(bu, bv)
        Pb = row_normalise(resampled_counts(counts, n_states, rng), cells, n_lon)[0]
        result.update(matrix_features(Pb, op, ok, core_states, durban))
        for k, x in result.items():
            samples[k][b] = x

    for k, value in full.items():
        if k.endswith("_bearing"):
            lo, hi = circular_interval(samples[k])
        else:
            lo, hi = np.nanpercentile(samples[k], [2.5, 97.5])
        low, high = EXPECT[k]
        inside = (hi >= low and lo <= high) if k in OVERLAP else (low <= value <= high)
        rows.append({"feature": k, "drogue": name, "estimate": value, "lo": lo, "hi": hi,
                     "expected_low": low, "expected_high": high, "agrees": bool(inside),
                     "undefined_share": float(np.isnan(samples[k]).mean())})
    print(f"{name}: {len(present):,} drifters resampled {N_BOOT} times")

table = pd.DataFrame(rows)

# ── 4. Save and summarise ────────────────────────────────────────────────────
out_csv = f"data/known_features_{TAG}.csv"
table.to_csv(out_csv, index=False, float_format="%.5g")
print(f"\n{'feature':<20}{'expected':>14}{'drogued':>26}{'undrogued':>26}")
for k in EXPECT:
    low, high = EXPECT[k]
    cells_txt = ""
    for name in DROGUE_TYPES:
        r = table[(table.feature == k) & (table.drogue == name)].iloc[0]
        cells_txt += f"{r.estimate:>10.2f} ({r.lo:.2f}–{r.hi:.2f}) {'yes' if r.agrees else 'NO':<3}"
    print(f"{k:<20}{f'{low:g} to {high:g}':>14}{cells_txt}")

# ── 5. Map of the features on the drogued mean current ───────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
lon_edges = LON_MIN + np.arange(n_lon + 1) * RES
lat_edges = LAT_MIN + np.arange(n_lat + 1) * RES
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), subplot_kw=subplot_kw)
for ax, (name, (u, v, peaks)) in zip(axes, drawn.items()):
    mesh = ax.pcolormesh(lon_edges, lat_edges, np.hypot(u, v), cmap="Blues", vmin=0, vmax=1.2, **kw)
    ax.quiver(grid_lon[::2], grid_lat[::2], u[::2, ::2], v[::2, ::2], color="#3a3a3a", scale=12, width=0.0018, **kw)
    ax.plot([grid_lon[i] for _, i in peaks], [grid_lat[j] for j, _ in peaks], "o-", color="#eb6834", ms=3, lw=1,
            label="Fastest core cells", **kw)
    feature = table[table.drogue == name].set_index("feature").estimate
    ax.plot([feature["retroflection_lon"]] * 2, RETRO_LATS, color="#1baf7a", lw=2, label="End of westward jet", **kw)
    ax.plot([feature["fate_divide_lon"]] * 2, RETURN_LATS, color="#e87ba4", lw=2, ls="--", label="Fate divide", **kw)
    ax.plot(RETURN_LONS, [feature["return_lat"]] * 2, color="#eda100", lw=2, label="Return Current latitude", **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.set_title(name.capitalize(), fontsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
cb = fig.colorbar(mesh, ax=axes, shrink=0.8)
cb.set_label("Mean speed, m/s (reliable cells)")
fig.suptitle(f"Known features of the circulation — {RES:g}°", fontsize=12)
out_png = f"figures/{region_prefix(args.data)}_known_features_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nDurban release point: {place_label(durban_point.lon, durban_point.lat)}")
print(f"Saved → {out_csv}\nSaved → {out_png}")
