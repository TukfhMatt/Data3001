"""
Agulhas Region — Drifter Bootstrap
Puts confidence intervals on the reported numbers by resampling whole
drifters (not observations, which are strongly correlated along a track).
For each drogue type, drifters are drawn with replacement, the transition
matrix is rebuilt from their transitions, and the release points from 04 are
propagated again. Repeated N_BOOT times, this gives:

  • 95% intervals on each release point's fate: share still in R, exited
    W / E / S / N and stranded after 7, 30 and 365 days, and 30-day
    coastal exposure
  • whether drogued and undrogued differ at each point (interval of the
    difference excludes 0)
  • row stability: median total-variation distance (TVD) between bootstrap
    rows and the full-data row, as a data-driven check of the 10-drifter flag

Needs the matrices from 03 and data/release_candidates.csv from 04 (same box,
grid and lag).

    .venv/bin/python scripts/06_bootstrap.py --res 1 --tau 3.5
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr

from config import DROGUE_TYPES, MIN_DRIFTERS, box_tag, grid_shape, operator_path, parse_args
from transport import (EXIT_LABELS, coastal_states, load_operator, move_indices, row_normalise,
                       state_of, strand_times, transition_states)

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU_DAYS = args.res, args.tau
N_BOOT = 1000
SEED = 0
HORIZONS = [7, 30, 365]          # days
UNSTABLE_TVD = 0.2               # rows whose median bootstrap TVD exceeds this are unstable
COLORS = {"drogued": "#2a78d6", "undrogued": "#eb6834"}   # categorical slots 1–2
TAG = f"{RES:g}".replace(".", "p") + f"deg_{TAU_DAYS:g}".replace(".", "p") + f"d_{box_tag(args.box)}"

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(SEED)

# ── 1. Load subset and pair obs τ apart ──────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)

n_lon, n_lat = grid_shape(args.box, RES)
cell_flat = np.where(
    in_R,
    ((lat - LAT_MIN) // RES).astype(int) * n_lon + ((lon - LON_MIN) // RES).astype(int),
    -1,
)
start, end, stranded = move_indices(traj_idx, t, in_R, TAU_DAYS, strand_times(ds, traj_idx, t, in_R))

ops = {name: load_operator(operator_path(name, RES, TAU_DAYS, args.box)) for name in DROGUE_TYPES}
op0 = ops["drogued"]   # both share the same ocean cells
cells = op0["cell_flat"]
n_cells = len(cells)
n_states = n_cells + len(EXIT_LABELS)
coastal = coastal_states(op0)

cand = pd.read_csv("data/release_candidates.csv")
release_states = np.array([state_of(op0, r.lon, r.lat) for r in cand.itertuples()])
steps = {d: int(round(d / TAU_DAYS)) for d in HORIZONS}
steps_30 = steps[30]


# ── 2. Per-drifter transition counts ─────────────────────────────────────────
def drifter_counts(flag):
    """Unique (drifter, from, to) triples with counts, for one drogue type.
    Summing them over drifters gives the count matrix C behind P."""
    same = (drogue[start] == flag) & (drogue[end] == flag)
    s, e = start[same], end[same]
    f, to = transition_states(cell_flat, cells, s, e, lon, lat, args.box, stranded[same])
    key = traj_idx[s].astype(np.int64) * n_states**2 + f * n_states + to
    uniq, n = np.unique(key, return_counts=True)
    d = uniq // n_states**2
    ft = uniq % n_states**2
    drifters, d_idx = np.unique(d, return_inverse=True)
    return d_idx, ft, n, len(drifters)


def fates(P):
    """Fate of each release point: {(metric, days): values}, one per point."""
    p = np.zeros((len(release_states), n_states))
    p[np.arange(len(release_states)), release_states] = 1.0
    out, exposure = {}, 0.0
    for step in range(1, steps[max(HORIZONS)] + 1):
        p = p @ P
        if step <= steps_30:
            exposure = exposure + p[:, :n_cells][:, coastal].sum(axis=1) * TAU_DAYS
        for days in HORIZONS:
            if step == steps[days]:
                out[("in_R", days)] = p[:, :n_cells].sum(axis=1)
                for k, lab in enumerate(op0["exit_labels"]):
                    out[(f"exit_{lab}", days)] = p[:, n_cells + k]
    out[("coastal_exposure_days", 30)] = exposure
    return out


# ── 3. Bootstrap by drifter ──────────────────────────────────────────────────
estimate, boots, row_tvd = {}, {}, {}
for name, flag in DROGUE_TYPES.items():
    d_idx, ft, n, n_drifters = drifter_counts(flag)
    C = np.bincount(ft, weights=n, minlength=n_states**2).reshape(n_states, n_states)
    assert np.array_equal(C, ops[name]["C"]), f"rebuilt counts differ from the saved {name} matrix"
    P = ops[name]["P"]
    estimate[name] = fates(P)

    samples = {k: np.empty((N_BOOT, len(release_states))) for k in estimate[name]}
    tvd = np.empty((N_BOOT, n_cells))
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(n_drifters, size=n_drifters), minlength=n_drifters)
        Cb = np.bincount(ft, weights=n * w[d_idx], minlength=n_states**2).reshape(n_states, n_states)
        Pb, _ = row_normalise(Cb, cells, n_lon)
        tvd[b] = 0.5 * np.abs(Pb[:n_cells] - P[:n_cells]).sum(axis=1)
        for k, v in fates(Pb).items():
            samples[k][b] = v
    boots[name] = samples
    row_tvd[name] = np.median(tvd, axis=0)
    print(f"{name}: {n_drifters:,} drifters resampled {N_BOOT} times")

# ── 4. Release-point intervals ───────────────────────────────────────────────
rows = []
for name in DROGUE_TYPES:
    for (metric, days), est in estimate[name].items():
        lo, hi = np.percentile(boots[name][(metric, days)], [2.5, 97.5], axis=0)
        for i, r in enumerate(cand.itertuples()):
            rows.append({"name": r.name, "kind": r.kind, "lon": r.lon, "lat": r.lat,
                         "drogue": name, "metric": metric, "days": days,
                         "estimate": est[i], "ci_low": lo[i], "ci_high": hi[i]})
table = pd.DataFrame(rows)
# Drogued − undrogued: the two resamples are independent, so pair them by index
diff_rows = []
for key in estimate["drogued"]:
    diff = boots["drogued"][key] - boots["undrogued"][key]
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    for i, r in enumerate(cand.itertuples()):
        diff_rows.append({"name": r.name, "metric": key[0], "days": key[1],
                          "diff": estimate["drogued"][key][i] - estimate["undrogued"][key][i],
                          "ci_low": lo[i], "ci_high": hi[i], "significant": lo[i] > 0 or hi[i] < 0})
diffs = pd.DataFrame(diff_rows)

out_csv = f"data/bootstrap_release_{TAG}.csv"
out_diff = f"data/bootstrap_difference_{TAG}.csv"
table.to_csv(out_csv, index=False)
diffs.to_csv(out_diff, index=False)


def fmt(name, metric, days, pct=True):
    sel = table[(table.drogue == name) & (table.metric == metric) & (table.days == days)].set_index("name")
    f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.1f}")
    return sel.apply(lambda r: f"{f(r.estimate)} ({f(r.ci_low)}–{f(r.ci_high)})", axis=1)


print(f"\nRelease points after 1 year, estimate (95% interval), {RES:g}°, τ = {TAU_DAYS:g} d")
for metric, label in [("in_R", "Still in R"), ("exit_W", "Exited west (Atlantic)"),
                      ("exit_E", "Exited east"), ("exit_stranded", "Stranded")]:
    show = pd.DataFrame({d: fmt(d, metric, 365) for d in DROGUE_TYPES})
    sig = diffs[(diffs.metric == metric) & (diffs.days == 365)].set_index("name")["significant"]
    show["differ"] = sig.map({True: "yes", False: "no"})
    print(f"\n{label}:\n{show.loc[cand.name].to_string()}")

show = pd.DataFrame({d: fmt(d, "coastal_exposure_days", 30, pct=False) for d in DROGUE_TYPES})
sig = diffs[diffs.metric == "coastal_exposure_days"].set_index("name")["significant"]
show["differ"] = sig.map({True: "yes", False: "no"})
print(f"\nCoastal exposure over 30 days (material-days):\n{show.loc[cand.name].to_string()}")

# ── 5. Row stability vs drifter count ────────────────────────────────────────
print(f"\nRow stability: median bootstrap TVD per row (unstable if > {UNSTABLE_TVD})")
bins = [0, MIN_DRIFTERS, 20, 50, np.inf]
labels = [f"< {MIN_DRIFTERS}", f"{MIN_DRIFTERS}–19", "20–49", "≥ 50"]
for name in DROGUE_TYPES:
    op = ops[name]
    ok = ~op["empty"]
    grp = pd.cut(op["row_drifters"][ok], bins, right=False, labels=labels)
    df = pd.DataFrame({"drifters": grp, "tvd": row_tvd[name][ok]})
    summary = df.groupby("drifters", observed=False)["tvd"].agg(
        rows="size", median_tvd="median", unstable=lambda x: (x > UNSTABLE_TVD).mean())
    unstable = row_tvd[name][ok] > UNSTABLE_TVD
    flagged = op["flagged"][ok]
    print(f"\n{name}: {unstable.sum()} of {ok.sum()} rows unstable; "
          f"{(unstable & ~flagged).sum()} of them pass the {MIN_DRIFTERS}-drifter rule, "
          f"{(flagged & ~unstable).sum()} flagged rows are stable")
    print(summary.to_string(formatters={"median_tvd": "{:.2f}".format, "unstable": "{:.0%}".format}))

out_rows = f"data/bootstrap_rows_{TAG}.npz"
np.savez_compressed(out_rows, cell_flat=cells, n_boot=N_BOOT, unstable_tvd=UNSTABLE_TVD,
                    **{f"row_tvd_{n}": v for n, v in row_tvd.items()})

# ── 6. Figures ───────────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}

# 6a. Row stability map (sequential blue, grey = no outgoing data)
cmap = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
cmap.set_bad("#d9d9d9")
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), subplot_kw=subplot_kw, constrained_layout=True)
for ax, name in zip(axes, DROGUE_TYPES):
    op = ops[name]
    grid = np.full(n_lat * n_lon, np.nan)
    grid[cells] = np.where(op["empty"], np.nan, row_tvd[name])
    mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], grid.reshape(n_lat, n_lon),
                         cmap=cmap, vmin=0, vmax=0.5, **kw)
    ax.scatter(cand.lon, cand.lat, s=40, c="white", edgecolor="#222222", linewidth=1, zorder=5, **kw)
    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#f0f0f0", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.top_labels = gl.right_labels = False
    unstable = (row_tvd[name] > UNSTABLE_TVD) & ~op["empty"]
    ax.set_title(f"{name.capitalize()} — {unstable.mean():.0%} of rows with median TVD > {UNSTABLE_TVD}",
                 fontsize=11)
cb = fig.colorbar(mesh, ax=axes, shrink=0.8, extend="max")
cb.set_label("Median bootstrap TVD of the row (0 = stable; grey = no data)")
fig.suptitle(f"Row stability under drifter resampling ({N_BOOT} bootstraps, {RES:g}°, τ = {TAU_DAYS:g} d); "
             "dots = release points", fontsize=12)
out_map = f"figures/agulhas_bootstrap_rows_{TAG}.png"
plt.savefig(out_map, dpi=150, bbox_inches="tight")
plt.close()

# 6b. One-year fate per release point with 95% intervals
metrics = [("in_R", "Still in R"), ("exit_W", "Exited west (Atlantic)"), ("exit_E", "Exited east"),
           ("exit_stranded", "Stranded")]
fig, axes = plt.subplots(1, 4, figsize=(19, 0.42 * len(cand) + 1.8), sharey=True, constrained_layout=True)
y = np.arange(len(cand))[::-1]
offset = {"drogued": 0.15, "undrogued": -0.15}
for ax, (metric, label) in zip(axes, metrics):
    for name in DROGUE_TYPES:
        sel = table[(table.drogue == name) & (table.metric == metric) & (table.days == 365)]
        sel = sel.set_index("name").loc[cand.name]
        ax.errorbar(sel.estimate * 100, y + offset[name],
                    xerr=[(sel.estimate - sel.ci_low) * 100, (sel.ci_high - sel.estimate) * 100],
                    fmt="o", ms=6, color=COLORS[name], elinewidth=2, capsize=0, label=name.capitalize())
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("% of released material after 1 year")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_yticks(y)
axes[0].set_yticklabels(cand.name)
axes[0].legend(loc="lower right", fontsize=9, frameon=False)
fig.suptitle(f"One-year fate of each release point, 95% drifter-bootstrap intervals "
             f"({RES:g}°, τ = {TAU_DAYS:g} d)", fontsize=12)
out_fate = f"figures/agulhas_bootstrap_fates_{TAG}.png"
plt.savefig(out_fate, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nIntervals saved   → {out_csv}")
print(f"Differences saved → {out_diff}")
print(f"Row TVD saved     → {out_rows}")
print(f"Maps saved        → {out_map}, {out_fate}")
