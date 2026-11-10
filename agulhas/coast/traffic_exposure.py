"""
Agulhas Region — Traffic-Weighted Coastal Exposure
If releases follow ship traffic, which coasts are most exposed, from which
shipping routes, and how soon? Releases are spread over the cells of R in
proportion to commercial shipping density (World Bank / IMF AIS, 2015–2021)
and pushed through the coast operators from coastal_stranding, for all traffic and for six
routes (ROUTE_NAMES).

Weights. The raster's values are undocumented in scale and bimodal: about 1–30
and 1e4–1e7 per pixel, with almost nothing between, and the high mode traces
the shipping lanes. They are used only as a relative density: the mean over a
cell's pixels, capped at the 99th percentile of ocean cells so one port cannot
dominate. Weight on a cell that is not reliable in both drifter matrices moves
to the nearest such cell (as the release points in shipping_overlay) and keeps
its route; the share moved and how far are reported. Two sensitivity runs change one choice
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
    pooled coastal segment from coastal_pooling, without intervals, in a separate table)
  • each route's share of the stranding on each coast
  • for oil, the mass stranded on each coast for heavy fuel oil and for fast loss (config.oil_mass_left)

Needs the coast operators and counts from coastal_stranding, and the segments
from coastal_pooling.

    python -m agulhas traffic_exposure --res 1 --tau 3.5
"""

from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from agulhas.config import OIL_MASS_CASES, grid_shape, oil_mass_left, operator_path, parse_args, setting_tag
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.shipping import blocks, read_window
from agulhas.stats import interval
from agulhas.transport import (COAST_LABELS, EXIT_LABELS, STRANDED, cell_centres, check_coast_box, coast_of,
                               load_coast_counts, load_operator, reliable, resampled_counts, row_normalise,
                               stranding_chain, to_grid)

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
SENSITIVITY = ["lane_share", "reliable_only"]
RELEASES = np.r_[["all"], ROUTES]
EXIT_NAMES = EXIT_LABELS[:STRANDED]
SERIES_COLORS = {"drogued": "#2a78d6", "undrogued": "#eb6834", "oil": "#1baf7a"}
SERIES_MARKERS = {"drogued": "o", "undrogued": "s", "oil": "D"}
INK, MUTED = "#0b0b0b", "#52514e"
SHORT_COASTS = ["Atlantic & Cape", "SA south & east", "S Mozambique", "C & N Mozambique",
                "W & S Madagascar", "E Madagascar"]
SHORT_ROUTES = ["All traffic", "Cape west", "Cape east", "Natal", "Moz. Channel", "S of Madag.", "E of Madag."]

# Where weight can go: every grid cell's centre, route and ocean state, and the reliable states
Geometry = namedtuple("Geometry", "grid_lon grid_lat grid_route state_of_grid ok_cells ok_state cell_lon cell_lat")
# Near-shore cells of each coast and of each pooled segment, as (states × coasts or segments) indicators
Nearshore = namedtuple("Nearshore", "in_coast in_segment segment_ids")


def route_of(lon, lat):
    """Route index (into ROUTES) of a position: a partition of R."""
    return np.select([lon < 20, (lon < 30) & (lat < -32), lon < 34, (lat >= -24) & (lon < 44),
                      (lon >= 50) | (lat >= -24)], [0, 1, 2, 3, 5], 4)


# ── Shipping density and lane share per grid cell ────────────────────────────

def load_density(box, res):
    """Mean pixel value per grid cell, and the share of its pixels in the high mode."""
    n_lon, n_lat = grid_shape(box, res)
    raw, px, _ = read_window(box, dtype=np.float32)
    density = blocks(raw, res, px).mean(axis=(1, 3), dtype=np.float64)[:n_lat, :n_lon].ravel()
    lane_share = blocks(raw >= LANE_MIN, res, px).mean(axis=(1, 3), dtype=np.float64)[:n_lat, :n_lon].ravel()
    return density, lane_share


# ── Release weights ──────────────────────────────────────────────────────────

def geometry(ops):
    op0 = ops["drogued"]
    cells = op0["cell_flat"]
    lo_e, la_e = op0["lon_edges"], op0["lat_edges"]
    n_lon, n_lat = len(lo_e) - 1, len(la_e) - 1
    ok_cells = reliable(ops["drogued"]) & reliable(ops["undrogued"])
    if not ok_cells.any():
        raise SystemExit("No cell is reliable in both drifter matrices.")
    grid_j, grid_i = np.divmod(np.arange(n_lat * n_lon), n_lon)
    grid_lon = (lo_e[grid_i] + lo_e[grid_i + 1]) / 2
    grid_lat = (la_e[grid_j] + la_e[grid_j + 1]) / 2
    state_of_grid = np.full(n_lat * n_lon, -1)
    state_of_grid[cells] = np.arange(len(cells))
    cell_lon, cell_lat = cell_centres(op0)
    return Geometry(grid_lon, grid_lat, route_of(grid_lon, grid_lat), state_of_grid, ok_cells,
                    np.where(ok_cells)[0], cell_lon, cell_lat)


def nearest_ok(k, geo):
    """State of the reliable cell nearest grid cell k, as in shipping_overlay."""
    dx = (geo.cell_lon[geo.ok_state] - geo.grid_lon[k]) * 111.32 * np.cos(np.radians(geo.grid_lat[k]))
    dy = (geo.cell_lat[geo.ok_state] - geo.grid_lat[k]) * 110.57
    return geo.ok_state[np.argmin(dx ** 2 + dy ** 2)]


def release_weights(values, geo, move=True):
    """Releases × ocean states: all traffic, then each route, each row summing
    to 1, with weight off reliable cells moved to the nearest reliable cell
    (or dropped). Also the route shares, and the share of weight moved with its
    weighted median and 90th percentile distance in km."""
    has = np.where(values > 0)[0]
    on_ok = (geo.state_of_grid[has] >= 0) & geo.ok_cells[geo.state_of_grid[has]]
    if not move:
        has = has[on_ok]
        on_ok = on_ok[on_ok]
    target = np.array([geo.state_of_grid[k] if good else nearest_ok(k, geo) for k, good in zip(has, on_ok)])
    W = np.zeros((len(ROUTES) + 1, len(geo.cell_lon)))
    np.add.at(W[0], target, values[has])
    for r in range(len(ROUTES)):
        m = geo.grid_route[has] == r
        np.add.at(W[r + 1], target[m], values[has][m])
    shares = W[1:].sum(axis=1) / W[0].sum()
    dx = (geo.cell_lon[target] - geo.grid_lon[has]) * 111.32 * np.cos(np.radians(geo.grid_lat[has]))
    km = np.hypot(dx, (geo.cell_lat[target] - geo.grid_lat[has]) * 110.57)
    moved_weight = values[has] * ~on_ok
    moved = (0.0, 0.0, 0.0)
    if moved_weight.sum() > 0:
        order = np.argsort(km)
        cum = np.cumsum(moved_weight[order]) / moved_weight.sum()
        moved = (moved_weight.sum() / values[has].sum(),
                 km[order][np.searchsorted(cum, 0.5)], km[order][np.searchsorted(cum, 0.9)])
    return W / W.sum(axis=1, keepdims=True), shares, moved


def all_weights(density, lane_share, geo):
    """The headline weights (capped density, moved) and the two sensitivity runs."""
    cap = np.percentile(density[geo.state_of_grid >= 0], CAP_PERCENTILE)
    capped = np.minimum(density, cap)
    return {"density": release_weights(capped, geo),
            "lane_share": release_weights(lane_share, geo),
            "reliable_only": release_weights(capped, geo, move=False)}


def load_nearshore(path, geo):
    """Near-shore cells of each coast, from coastal_pooling's segments."""
    seg = np.load(path, allow_pickle=True)
    segment_of_state = seg["segment_of_state"]
    coastal = seg["coastal"].astype(bool)
    segment_ids = np.unique(segment_of_state[coastal])
    in_segment = np.stack([segment_of_state == s for s in segment_ids], axis=1).astype(float)
    coast_of_cell = coast_of(geo.cell_lon, geo.cell_lat)
    in_coast = np.stack([coastal & (coast_of_cell == k) for k in range(len(COAST_LABELS))], axis=1).astype(float)
    return Nearshore(in_coast, in_segment, segment_ids)


# ── Exposure from one matrix and set of releases ────────────────────────────

def exposure(P, weights, near_cells, tau, oil=False):
    """Stranding, exits, days to strand, near-shore material-days and (with
    oil=True) stranded oil mass for each case of OIL_MASS_CASES, per release; and each route's share of each
    coast's stranding. weights are (W, route shares)."""
    W, shares = weights
    n_cells = W.shape[1]
    steps = {d: int(round(d / tau)) for d in HORIZONS}
    coast_cols = slice(n_cells + STRANDED, P.shape[0])
    fate, visits = stranding_chain(P, n_cells, STRANDED)
    wf = W @ fate
    b = wf[:, STRANDED:].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        days = np.where(b > 0, W @ visits / b * tau, np.nan)
    p = np.zeros((len(W), P.shape[0]))
    p[:, :n_cells] = W
    prev = p[:, coast_cols].copy()
    near = np.zeros((len(W), n_cells))
    mass = {case: np.zeros((len(W), len(COAST_LABELS))) for case in OIL_MASS_CASES}
    out = {"eventual": wf[:, STRANDED:], "exit": wf[:, :STRANDED], "days": days}
    for step in range(1, steps[max(HORIZONS)] + 1):
        p = p @ P
        if step <= int(round(NEARSHORE_DAYS / tau)):
            near += p[:, :n_cells] * tau
        new = p[:, coast_cols]
        if oil:
            for case in OIL_MASS_CASES:
                mass[case] += (new - prev) * oil_mass_left(step * tau, case)
        prev = new.copy()
        for d, k in steps.items():
            if step == k:
                out[d] = new.copy()
                out[f"in_R_{d}"] = p[:, :n_cells].sum(axis=1)
    out["nearshore"] = near @ near_cells.in_coast
    out["segments"] = near @ near_cells.in_segment
    if oil:
        out["mass"], out["mass_fast"] = mass["heavy_fuel"], mass["fast_loss"]
    # Each route's share of the traffic-weighted stranding on each coast
    by_route = shares[:, None] * wf[1:, STRANDED:]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["source"] = by_route / by_route.sum(axis=0, keepdims=True)
    return out


def run_exposure(ops, weights, near_cells, counts, args):
    """Every matrix under the headline weights and the sensitivity runs, with
    intervals over N_BOOT drifter resamples for BOOT_MATRICES."""
    n_lon, _ = grid_shape(args.box, args.res)
    headline = weights["density"][:2]
    rng = np.random.default_rng(SEED)
    full, sens, intervals = {}, {}, {}
    for name in MATRICES:
        P = ops[name]["P"]
        full[name] = exposure(P, headline, near_cells, args.tau, name == "oil")
        sens[name] = {w: exposure(P, weights[w][:2], near_cells, args.tau, name == "oil") for w in SENSITIVITY}
        if name not in BOOT_MATRICES:
            continue
        draws = {}
        for _ in range(N_BOOT):
            Pb = row_normalise(resampled_counts(counts[name], P.shape[0], rng), ops[name]["cell_flat"], n_lon)[0]
            for key, value in exposure(Pb, headline, near_cells, args.tau, name == "oil").items():
                if key != "segments":
                    draws.setdefault(key, []).append(value)
        intervals[name] = {k: interval(np.array(v)) for k, v in draws.items()}
        print(f"{name:<10} {counts[name][3]:,} drifters resampled {N_BOOT} times")
    return full, sens, intervals


# ── Tables ───────────────────────────────────────────────────────────────────

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
        for k, label in enumerate(EXIT_NAMES):
            add("exit", "eventual", label, release, out["exit"][r, k], *bounds("exit", r, k))
        for d in HORIZONS:
            add("in_R", d, "R", release, out[f"in_R_{d}"][r], *bounds(f"in_R_{d}", r))
        for k, coast in enumerate(COAST_LABELS):
            add("nearshore_days", NEARSHORE_DAYS, coast, release, out["nearshore"][r, k], *bounds("nearshore", r, k))
            if "mass" in out:
                add("mass_stranded", "eventual", coast, release, out["mass"][r, k], *bounds("mass", r, k))
                add("mass_stranded_fast", "eventual", coast, release, out["mass_fast"][r, k], *bounds("mass_fast", r, k))
    for k, coast in enumerate(COAST_LABELS):
        for r, route in enumerate(ROUTES):
            add("source_share", "eventual", coast, route, out["source"][r, k], *bounds("source", r, k))
    return rows


def exposure_table(full, sens, intervals):
    return pd.DataFrame([row for name in MATRICES
                         for row in rows_for(name, "density", full[name], intervals.get(name))
                         + [r for w in SENSITIVITY for r in rows_for(name, w, sens[name][w], None)]])


def segment_table(full, segment_ids):
    return pd.DataFrame([
        {"matrix": name, "release": release, "segment_id": int(s) + 1,   # numbered from 1 in coastal_pooling's tables
         "nearshore_days": full[name]["segments"][r, j]}
        for name in MATRICES for r, release in enumerate(RELEASES) for j, s in enumerate(segment_ids)])


def weight_table(weights, geo):
    return pd.DataFrame({"state": np.arange(len(geo.cell_lon)), "lon": geo.cell_lon, "lat": geo.cell_lat,
                         "reliable_both": geo.ok_cells, "weight_density": weights["density"][0][0],
                         "weight_lane_share": weights["lane_share"][0][0],
                         "route": ROUTES[route_of(geo.cell_lon, geo.cell_lat)]})


# ── Figure: weights, exposed coasts, route to coast ─────────────────────────

def plot_weights(ax, op, W, box):
    n_cells = len(op["cell_flat"])
    grid = to_grid(op, np.r_[W[0], np.zeros(op["P"].shape[0] - n_cells)])
    pos = grid[grid > 0]
    mesh = ax.pcolormesh(op["lon_edges"], op["lat_edges"], np.ma.masked_less_equal(grid, 0), cmap="Blues",
                         norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-4), vmax=pos.max()), **TRANSFORM_KW)
    map_axes(ax, box)
    for x, y in [([20, 20], [-45, -15]), ([20, 30, 30], [-32, -32, -45]), ([34, 34], [-45, -24]),
                 ([34, 50], [-24, -24]), ([44, 44], [-24, -15]), ([50, 50], [-45, -24])]:
        ax.plot(x, y, color=MUTED, lw=0.8, ls="--", zorder=4, **TRANSFORM_KW)
    for label, lo, la in [("Cape W", 14.5, -42), ("Cape E", 25, -42), ("Natal", 31.8, -42),
                          ("Moz. Channel", 39, -22.8), ("S of Madag.", 42, -42), ("E of Madag.", 52.5, -42)]:
        ax.text(lo, la, label, ha="center", fontsize=8, color=INK, zorder=5, **TRANSFORM_KW)
    ax.figure.colorbar(mesh, ax=ax, shrink=0.8, label="Share of traffic-weighted release per cell (log)")
    ax.set_title("Where releases start", fontsize=11, color=INK)


def plot_exposed(ax, full, intervals):
    y = np.arange(len(COAST_LABELS))
    for offset, name in zip([-0.22, 0, 0.22], BOOT_MATRICES):
        est = full[name]["eventual"][0]
        lo, hi = intervals[name]["eventual"][:, 0]
        ax.hlines(y + offset, lo, hi, color=SERIES_COLORS[name], lw=2)
        ax.scatter(est, y + offset, s=45, marker=SERIES_MARKERS[name], color=SERIES_COLORS[name],
                   edgecolor="white", linewidth=1, zorder=3, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(SHORT_COASTS)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Share of traffic-weighted release that eventually strands there (95% interval)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Which coasts are exposed", fontsize=11, color=INK)


def plot_routes(ax, heat):
    ax.imshow(heat, cmap="Blues", vmin=0, vmax=max(heat.max(), 0.01), aspect="auto")
    for r in range(heat.shape[0]):
        for k in range(heat.shape[1]):
            ax.text(k, r, f"{heat[r, k]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if heat[r, k] > 0.6 * heat.max() else INK)
    ax.set_xticks(np.arange(len(COAST_LABELS)))
    ax.set_xticklabels(SHORT_COASTS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(SHORT_ROUTES)))
    ax.set_yticklabels(SHORT_ROUTES)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Oil: from each route to each coast", fontsize=11, color=INK)


def plot_exposure(ops, W, full, intervals, args, path):
    fig = plt.figure(figsize=(19, 5.6))
    plot_weights(fig.add_subplot(1, 3, 1, **MAP_KW), ops["drogued"], W, args.box)
    plot_exposed(fig.add_subplot(1, 3, 2), full, intervals)
    plot_routes(fig.add_subplot(1, 3, 3), full["oil"]["eventual"])
    fig.suptitle(f"Traffic-weighted coastal exposure — {args.res:g}°, τ = {args.tau:g} d", fontsize=13, color=INK)
    fig.tight_layout()
    save_figure(path)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_weights(weights, south_share):
    _, route_share, moved_stats = weights["density"]
    print(f"\nWeights: cap at the {CAP_PERCENTILE}th percentile of ocean cells; "
          f"{moved_stats[0]:.1%} of weight moved to the nearest reliable cell "
          f"(weighted median {moved_stats[1]:.0f} km, 90th percentile {moved_stats[2]:.0f} km); "
          f"{south_share:.2%} of weight south of {abs(ARTIFACT_LAT):g}°S")
    print("Route shares of traffic: " + ", ".join(f"{r} {s:.0%}" for r, s in zip(ROUTES, route_share)))


def print_stranding(full, intervals):
    def fmt(name, key, *index):
        v = full[name][key][index]
        if name not in intervals:
            return f"{v:.0%}"
        lo, hi = intervals[name][key][(slice(None),) + index]
        return f"{v:.0%} ({lo:.0%}–{hi:.0%})"

    print("\nAll traffic: stranding on each coast, eventually and within 30 days (95% interval)")
    print(pd.DataFrame({f"{name} {h}": [fmt(name, h, 0, k) for k in range(len(COAST_LABELS))]
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


def print_oil_and_sensitivity(full, sens):
    print("\nOil: eventual stranding from each route on each coast")
    print(pd.DataFrame(full["oil"]["eventual"], index=RELEASES, columns=COAST_LABELS)
          .round(3).to_string())

    print("\nOil: share of each coast's stranding from each route")
    print(pd.DataFrame(full["oil"]["source"], index=ROUTES, columns=COAST_LABELS).round(2).to_string())

    print(f"\nNear-shore material-days over {NEARSHORE_DAYS} days, all traffic")
    print(pd.DataFrame({name: [f"{full[name]['nearshore'][0, k]:.2f}" for k in range(len(COAST_LABELS))]
                        for name in MATRICES}, index=COAST_LABELS).to_string())

    for key, label in [("mass", "heavy fuel oil"), ("mass_fast", "fast loss, 14 d")]:
        print(f"\nOil mass stranded, {label}, all traffic: "
              + ", ".join(f"{c} {m:.2%}" for c, m in zip(COAST_LABELS, full["oil"][key][0])))

    print("\nSensitivity: all-traffic eventual stranding with lane-pixel weights and without moved weight")
    print(pd.DataFrame({f"{name} {label}": [f"{out[name]['eventual'][0, k]:.1%}" for k in range(len(COAST_LABELS))]
                        for name in BOOT_MATRICES
                        for label, out in [("density", full),
                                           ("lanes", {n: sens[n]["lane_share"] for n in BOOT_MATRICES}),
                                           ("dropped", {n: sens[n]["reliable_only"] for n in BOOT_MATRICES})]},
                       index=COAST_LABELS).to_string())


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    tag = setting_tag(args.res, args.tau, args.box)
    out_releases = f"data/traffic_releases_{tag}.npz"
    out_table = f"data/traffic_exposure_{tag}.csv"
    out_segments = f"data/traffic_nearshore_segments_{tag}.csv"
    out_weights = f"data/traffic_weights_{tag}.csv"
    out_png = f"figures/agulhas_traffic_exposure_{tag}.png"

    density, lane_share = load_density(args.box, args.res)
    ops = {name: load_operator(operator_path(f"{name}_coast", args.res, args.tau, args.box)) for name in MATRICES}
    geo = geometry(ops)
    weights = all_weights(density, lane_share, geo)
    W, route_share, _ = weights["density"]
    np.savez_compressed(out_releases, W=W, route_share=route_share, releases=RELEASES)

    near_cells = load_nearshore(f"data/coastal_segments_{tag}.npz", geo)
    counts = load_coast_counts(f"data/coast_counts_{tag}.npz", BOOT_MATRICES, ops["drogued"]["P"].shape[0])
    full, sens, intervals = run_exposure(ops, weights, near_cells, counts, args)
    exposure_table(full, sens, intervals).to_csv(out_table, index=False)
    segment_table(full, near_cells.segment_ids).to_csv(out_segments, index=False)
    weight_table(weights, geo).to_csv(out_weights, index=False)

    plot_exposure(ops, W, full, intervals, args, out_png)
    pd.set_option("display.width", 220)
    print_weights(weights, W[0][geo.cell_lat < ARTIFACT_LAT].sum())
    print_stranding(full, intervals)
    print_oil_and_sensitivity(full, sens)
    print(f"\nSaved → {out_table}, {out_segments}, {out_weights}, {out_png}")


if __name__ == "__main__":
    main()
