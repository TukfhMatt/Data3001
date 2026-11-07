"""
Agulhas Region — Shipping Traffic vs Surface Currents
Overlays commercial shipping density (World Bank / IMF AIS, 2015–2021) on
the mean current from drogued and from undrogued drifters, then picks
candidate release points: the busiest shipping cells plus known hotspots,
ranked by how much material each transition matrix (drogued and undrogued)
carries to near-shore cells. Grid, lag and box come from agulhas/config.py.

Shipping data: "Global Shipping Traffic Density", World Bank Data Catalog,
dataset 0037580, CC-BY 4.0. Described as counts of AIS positions per 0.005°
cell, but the stored values are far larger than raw position counts could be
(lane pixels ~1e4–1e7), so they are treated as a relative density only.

Candidates are only placed in cells that are reliable in both matrices (≥ 10
distinct drifters, has outgoing data), so the two versions are compared at
the same points. Hotspots in unreliable cells are moved to the nearest
reliable cell and the shift is reported.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

from agulhas.config import (DROGUE_TYPES, MIN_DRIFTERS, box_tag, operator_path, parse_args,
                            place_label, region_prefix, release_points_path, setting_tag)
from agulhas.drifters import Drifters, in_box
from agulhas.grid import Grid
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.shipping import blocks, read_window
from agulhas.transport import coastal_states, load_operator, propagate, state_of

FINE = 0.1           # display grid for the shipping map
MIN_CURRENT_OBS = 300  # cells with fewer hourly obs of a drogue type get no current arrow
N_TRAFFIC = 8        # busiest-cell candidates to pick
MIN_SPACING = 2.0    # degrees between chosen candidates
# Known hotspots: offshore of each port / anchorage (lon, lat). Those outside
# the box are skipped, so another region is picked by traffic alone.
HOTSPOTS = {
    "Richards Bay":           (32.6, -28.9),
    "Durban":                 (31.4, -30.0),
    "Algoa Bay (STS bunkering)": (25.9, -33.9),
    "Cape Town":              (18.2, -33.9),
    "Saldanha Bay":           (17.8, -33.0),
}
LABELS = {"drogued": "drogued drifters, 15 m current", "undrogued": "undrogued drifters, current + wind slip"}


# ── Shipping density ─────────────────────────────────────────────────────────

def load_shipping(grid):
    """Shipping density summed on the display grid and on the analysis grid."""
    raw, px, tif = read_window(grid.box)
    ship_fine = blocks(raw, FINE, px).sum(axis=(1, 3))
    ship_grid = blocks(raw, grid.res, px).sum(axis=(1, 3))[:grid.n_lat, :grid.n_lon]   # drop any stray rounding pixels
    print(f"Shipping raster: {tif}, pixel {px:.4f}° (values used as relative density)")
    return ship_fine, ship_grid


# ── Mean current per drogue type ─────────────────────────────────────────────

def mean_currents(drifters, grid):
    """Per drogue type: mean east and north velocity per cell (NaN under
    MIN_CURRENT_OBS), and the hourly observations behind each cell."""
    ve, vn = drifters.ve, drifters.vn
    inside = in_box(drifters.lon, drifters.lat, grid.box) & np.isfinite(ve) & np.isfinite(vn)
    shape = (grid.n_lat, grid.n_lon)
    currents = {}
    for name, flag in DROGUE_TYPES.items():
        m = inside & (drifters.drogue == flag)
        cell = grid.cell_in_box(drifters.lon[m], drifters.lat[m])
        cnt = np.bincount(cell, minlength=grid.n_cells)
        u = np.bincount(cell, ve[m], grid.n_cells) / np.maximum(cnt, 1)
        v = np.bincount(cell, vn[m], grid.n_cells) / np.maximum(cnt, 1)
        ok = cnt >= MIN_CURRENT_OBS
        u, v = np.where(ok, u, np.nan).reshape(shape), np.where(ok, v, np.nan).reshape(shape)
        currents[name] = (u, v, cnt.reshape(shape))
        print(f"Mean current ({name}): {ok.sum()} cells, max speed {np.nanmax(np.hypot(u, v)):.2f} m/s")
    return currents


# ── Candidate release points ─────────────────────────────────────────────────

def cell_masks(ops, grid):
    """On the grid: ocean cells, near-shore cells (ocean cells with a non-ocean
    neighbour, which stand in for coastal impact) and cells reliable in both
    matrices (enough drifters behind the row, and real outgoing data)."""
    op = ops["drogued"]   # both operators share the same ocean cells
    masks = {}
    for key, values in [("ocean", True), ("coastal", coastal_states(op)),
                        ("reliable", np.logical_and.reduce([~o["flagged"] & ~o["empty"] for o in ops.values()]))]:
        flat = np.zeros(grid.n_cells, bool)
        flat[op["cell_flat"]] = values
        masks[key] = flat.reshape(grid.n_lat, grid.n_lon)
    return masks


def snap_to_reliable(lo, la, reliable_grid, grid):
    """Centre of the nearest reliable cell to (lo, la) and the distance in km."""
    jj, ii = np.nonzero(reliable_grid)
    cx, cy = grid.lon_centres[ii], grid.lat_centres[jj]
    dx = (cx - lo) * 111.32 * np.cos(np.radians(la))
    dy = (cy - la) * 110.57
    k = np.argmin(dx ** 2 + dy ** 2)
    return cx[k], cy[k], float(np.hypot(dx[k], dy[k]))


def pick_candidates(grid, reliable_grid, ship_grid):
    """Known hotspots inside the box, moved to reliable cells, then the busiest
    reliable shipping cells at least MIN_SPACING apart: (kind, name, lon, lat)."""
    print(f"\nHotspots moved to the nearest cell reliable in both matrices "
          f"(≥ {MIN_DRIFTERS} drifters, has outgoing data):")
    candidates = []
    for name, (lo, la) in HOTSPOTS.items():
        if not in_box(lo, la, grid.box):
            continue
        j, i = int((la - grid.lat_min) // grid.res), int((lo - grid.lon_min) // grid.res)
        if reliable_grid[j, i]:
            clo, cla = grid.lon_centres[i], grid.lat_centres[j]
            print(f"  {name:<28} unchanged")
        else:
            clo, cla, km = snap_to_reliable(lo, la, reliable_grid, grid)
            print(f"  {name:<28} ({lo}, {la}) → ({clo:.1f}, {cla:.1f}), ~{km:.0f} km")
        candidates.append(("hotspot", name, clo, cla))

    traffic = np.where(reliable_grid, ship_grid, -1).ravel()
    for flat in np.argsort(traffic)[::-1]:
        if len([c for c in candidates if c[0] == "traffic"]) == N_TRAFFIC:
            break
        j, i = divmod(flat, grid.n_lon)
        clo, cla = grid.lon_centres[i], grid.lat_centres[j]
        if all(max(abs(clo - c[2]), abs(cla - c[3])) >= MIN_SPACING for c in candidates):
            candidates.append(("traffic", f"Lane {place_label(clo, cla)}", clo, cla))
    return candidates


def fate(o, s, coastal_state):
    """Coastal exposure over 30 days and state after 1 year, released in state s."""
    n_cells = len(o["cell_flat"])
    tau = float(o["tau_days"])
    week_step = int(round(7 / tau))
    p = np.zeros(o["P"].shape[0])
    p[s] = 1.0
    exposure, coast_7, c = 0.0, 0.0, 0.0
    for step in range(1, int(round(30 / tau)) + 1):
        p = p @ o["P"]
        c = p[:n_cells][coastal_state].sum()
        exposure += c * tau
        if step == week_step:
            coast_7 = c
    p365 = propagate(o, np.eye(o["P"].shape[0])[s], days=365)
    return {"coastal_frac_7d": coast_7, "coastal_frac_30d": c,
            "coastal_exposure_days_30d": exposure, "in_R_365d": p365[:n_cells].sum(),
            **{f"exit_{l}_365d": v for l, v in zip(o["exit_labels"], p365[n_cells:])},
            "row_drifters": int(o["row_drifters"][s])}


def rank_candidates(candidates, ops, masks, ship_grid, grid):
    """Shipping and the fate under each matrix of every candidate, ranked by drogued coastal exposure."""
    op = ops["drogued"]
    coastal_state = coastal_states(op)
    rows = []
    for kind, name, lo, la in candidates:
        s = state_of(op, lo, la)
        j, i = divmod(int(op["cell_flat"][s]), grid.n_lon)
        row = {"kind": kind, "name": name, "lon": lo, "lat": la,
               "ship_density": ship_grid[j, i],
               "ship_density_rel": ship_grid[j, i] / ship_grid[masks["ocean"]].max(),
               "start_coastal": bool(masks["coastal"][j, i])}
        for dname, o in ops.items():
            row.update({f"{dname}_{k}": v for k, v in fate(o, s, coastal_state).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("drogued_coastal_exposure_days_30d", ascending=False)


def print_candidates(table, res, tau):
    print(f"\nCandidate release points ({res:g}°, τ = {tau:g} d), ranked by drogued coastal exposure")
    print("ship = density relative to the busiest cell in R; coastal = ocean cells next to land;")
    print("exposure = material-days spent in coastal cells over the first 30 days\n")
    show = table[["kind", "name", "lon", "lat", "ship_density_rel", "start_coastal"]
                 + [f"{d}_{k}" for k in ("coastal_exposure_days_30d", "in_R_365d") for d in DROGUE_TYPES]]
    show.columns = ["kind", "name", "lon", "lat", "ship", "coastal",
                    "exp_30d_dro", "exp_30d_und", "inR_365d_dro", "inR_365d_und"]
    print(show.to_string(index=False, formatters={
        "ship": "{:.0%}".format,
        "exp_30d_dro": "{:.1f}".format, "exp_30d_und": "{:.1f}".format,
        "inR_365d_dro": "{:.0%}".format, "inR_365d_und": "{:.0%}".format}))


def save_shipping(path, ship_grid, ship_fine, currents, masks, grid):
    np.savez_compressed(path, ship=ship_grid, ship_fine=ship_fine,
                        **{f"{k}_{d}": a for d, c in currents.items()
                           for k, a in zip(("u", "v", "current_obs"), c)},
                        coastal=masks["coastal"], lon_min=grid.lon_min, lat_min=grid.lat_min,
                        res=grid.res, fine=FINE)


# ── Maps (one per drogue type) ───────────────────────────────────────────────

def plot_shipping_map(table, ship_fine, current, dname, grid, tau, path):
    u, v, _ = current
    fine_lon = np.arange(ship_fine.shape[1] + 1) * FINE + grid.lon_min
    fine_lat = np.arange(ship_fine.shape[0] + 1) * FINE + grid.lat_min
    # Relative to the busiest 0.1° cell; colour range clipped to percentiles of
    # non-zero cells so the lanes stand out from the broad background
    rel = np.where(ship_fine > 0, ship_fine / ship_fine.max(), np.nan)
    lo_c, hi_c = np.nanpercentile(rel, [60, 99.5])
    thin = max(1, int(round(2 / grid.res)))   # one arrow per ~2°
    sub = (slice(None, None, thin), slice(None, None, thin))
    halo = [pe.withStroke(linewidth=3, foreground="white")]

    fig, ax = plt.subplots(figsize=(13, 9), subplot_kw=MAP_KW)
    mesh = ax.pcolormesh(fine_lon, fine_lat, rel, cmap="Blues", norm=LogNorm(vmin=lo_c, vmax=hi_c), **TRANSFORM_KW)
    q = ax.quiver(grid.lon_centres[sub[1]], grid.lat_centres[sub[0]], u[sub], v[sub], color="#3a3a3a", alpha=0.85,
                  scale=10, width=0.0025, zorder=4, **TRANSFORM_KW)
    ax.quiverkey(q, 0.40, 0.93, 0.5, "0.5 m/s mean current", labelpos="E",
                 coordinates="axes", fontproperties={"size": 9})
    for kind, marker, face, label in [("hotspot", "*", "#e8710a", "Known hotspot (moved to nearest reliable cell)"),
                                      ("traffic", "o", "#ffffff", "Busiest shipping cell (reliable cells only)")]:
        sel = table[table["kind"] == kind]
        ax.scatter(sel["lon"], sel["lat"], marker=marker, s=180 if marker == "*" else 70,
                   c=face, edgecolor="#222222", linewidth=1, zorder=6, label=label, **TRANSFORM_KW)
    col = f"{dname}_coastal_exposure_days_30d"
    for _, r in table.iterrows():
        text = f"{r['name']}  {r[col]:.1f} d" if r["kind"] == "hotspot" else f"{r[col]:.1f} d"
        # A label beside the box's east edge goes to the left of its point, or the edge would cut it
        ax.annotate(text, (r["lon"], r["lat"]), xytext=(-26, -12) if r["lon"] > grid.lon_max - 2 else (7, -12),
                    textcoords="offset points", fontsize=8, color="#222222", zorder=7, path_effects=halo)
    map_axes(ax, grid.box, gridlines=True)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02, extend="both")
    cb.set_label("Commercial ship density relative to busiest 0.1° cell, AIS 2015–2021 (log)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Commercial shipping and mean current ({LABELS[dname]}) — {box_tag(grid.box)}\n"
                 f"Labels: material-days spent in near-shore cells over the first 30 days after release "
                 f"({grid.res:g}°, τ = {tau:g} d)", fontsize=11)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    points_out = release_points_path(args.box, args.res, args.tau)
    ship_out = f"data/shipping_{tag}.npz"
    grid = Grid(args.box, args.res)

    ship_fine, ship_grid = load_shipping(grid)
    currents = mean_currents(Drifters(args.data), grid)
    ops = {name: load_operator(operator_path(name, args.res, args.tau, args.box)) for name in DROGUE_TYPES}
    masks = cell_masks(ops, grid)
    candidates = pick_candidates(grid, masks["reliable"], ship_grid)
    table = rank_candidates(candidates, ops, masks, ship_grid, grid)
    table.to_csv(points_out, index=False)
    print_candidates(table, args.res, args.tau)
    print(f"\nTable saved → {points_out}")
    save_shipping(ship_out, ship_grid, ship_fine, currents, masks, grid)
    print(f"Shipping and currents saved → {ship_out}")

    for dname, current in currents.items():
        out = f"figures/{region_prefix(args.data)}_shipping_currents_{dname}_{tag}.png"
        plot_shipping_map(table, ship_fine, current, dname, grid, args.tau, out)
        print(f"Map saved → {out}")


if __name__ == "__main__":
    main()
