"""
Agulhas Region — Spill Hindcasts
Sets the oil matrix against two recorded spills off the Cape, as a check of coast
and timescale. The matrix holds average conditions over 1995–2022, so it cannot
reproduce the winds of one event, and a single event can neither confirm nor
reject a probability. One 1° cell also holds the Treasure wreck, Robben Island
and Dassen Island. The question is whether the matrix favours what happened.

  • MV Treasure, 23 June 2000 (33.67°S, 18.33°E): heavy fuel oil reached Robben
    and Dassen islands, drifting towards Dassen by 2 July (9 days).
  • Castillo de Bellver, 6 August 1983 (33.28°S, 17.50°E): light crude that first
    drifted towards the coast, then went offshore with a wind shift into the
    north-west flowing Benguela Current, with little coastal pollution.

Decision rule, fixed before this script was first run. For each spill, on the
oil matrix at 3.5% of the wind, with 95% intervals from N_BOOT resamples of
whole drifters:

  1. Stranding: the share stranded within 30 days is above 50% for Treasure, and
     below 50% for Castillo de Bellver.
  2. Coast: Atlantic and Cape receives the most stranding within 30 days.
  3. Direction: material still in R after 14 days has moved north and west.

Rules 1 and 3 read "consistent" if the whole interval is on the side of what
happened, "inconsistent" if the whole interval is on the other side, and
"unclear" otherwise. Each spill is judged at its own cell and at the nearest
reliable cell (as shipping_overlay picks), and a thin spill cell cannot give a clear result
alone. Oil at 3% and 4%, undrogued material and the drogued check are reported
alongside, as is the mean over reliable neighbouring cells.

Needs oil_wind, seasonal_matrices and coastal_stranding.

    python -m agulhas spill_hindcasts --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

from agulhas.config import OIL_WINDAGE_RANGE, grid_shape, operator_path, parse_args, setting_tag
from agulhas.transport import (COAST_LABELS, STRANDED, cell_centres, check_coast_box, load_operator, reliable,
                       release, propagate, resampled_counts, row_normalise, state_of, to_grid)

N_BOOT = 1000
SEED = 0
HORIZONS = [7, 14, 30]
THRESHOLD = 0.5
HEADLINE = "oil"
MATRICES = list(OIL_WINDAGE_RANGE) + ["undrogued", "oil_check"]
EXPECTED_COAST = "atlantic_cape"
# Positions as published on Wikipedia; ITOPF ("about 70 miles north-west of Cape
# Town") and Cedre agree with them on the area. Sources disagree on tonnes spilled.
SPILLS = [
    {"name": "Treasure", "date": "2000-06-23", "lon": 18.33167, "lat": -33.67167, "strands": True,
     "record": "heavy fuel oil reached Robben and Dassen islands; drifting towards Dassen by day 9"},
    {"name": "Castillo de Bellver", "date": "1983-08-06", "lon": 17.50467, "lat": -33.27583, "strands": False,
     "record": "light crude went offshore into the Benguela Current; little coastal pollution"},
]
ISLANDS = {"Robben I.": (18.37, -33.80), "Dassen I.": (18.09, -33.42)}


def km_between(lon0, lat0, lon1, lat1):
    dx = (lon1 - lon0) * 111.32 * np.cos(np.radians(lat0))
    return np.hypot(dx, (lat1 - lat0) * 110.57)


def interval(x):
    with np.errstate(all="ignore"):
        return np.nanpercentile(x, [2.5, 97.5], axis=0)


def side_verdict(lo, hi, threshold, above):
    """Does the interval lie on the side (above or below threshold) of what happened?"""
    if (lo > threshold) if above else (hi < threshold):
        return "consistent"
    if (hi < threshold) if above else (lo > threshold):
        return "inconsistent"
    return "unclear"


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
    RES, TAU = args.res, args.tau
    TAG = setting_tag(RES, TAU, args.box)
    RULE_STRAND_DAYS, RULE_DIRECTION_DAYS = 30, 14

    os.makedirs("figures", exist_ok=True)

    # ── 1. Operators, counts and release sites ───────────────────────────────
    ops = {name: load_operator(operator_path(f"{name}_coast", RES, TAU, args.box)) for name in MATRICES}
    op0 = ops[HEADLINE]
    cells = op0["cell_flat"]
    n_cells, n_states = len(cells), op0["P"].shape[0]
    n_lon, _ = grid_shape(args.box, RES)
    cell_lon, cell_lat = cell_centres(op0)
    ok = reliable(op0)
    counts_file = np.load(f"data/coast_counts_{TAG}.npz")
    if int(counts_file["n_states"]) != n_states:
        raise SystemExit("data/coast_counts_….npz does not match the coast operators; rerun coastal_stranding.")

    sites = []
    for spill in SPILLS:
        state = state_of(op0, spill["lon"], spill["lat"])
        usable = np.where(ok)[0]
        nearest = usable[np.argmin(km_between(spill["lon"], spill["lat"], cell_lon[usable], cell_lat[usable]))]
        near_ring = (np.abs(cell_lon - cell_lon[state]) <= RES) & (np.abs(cell_lat - cell_lat[state]) <= RES)
        ring = np.where(near_ring & ok & (np.arange(n_cells) != state))[0]
        for label, states in [("spill cell", [state]), ("nearest reliable", [nearest]), ("reliable neighbours", ring)]:
            weights = np.zeros(n_cells)
            weights[states] = 1.0 / len(states)
            sites.append({"spill": spill["name"], "site": label, "weights": weights, "cells": len(states),
                          "drifters": int(op0["row_drifters"][states].min()), "reliable": bool(ok[states].all()),
                          "lon": float(weights @ cell_lon), "lat": float(weights @ cell_lat)})
    W = np.array([s["weights"] for s in sites])
    start_lon, start_lat = W @ cell_lon, W @ cell_lat
    steps = {d: int(round(d / TAU)) for d in HORIZONS}

    # ── 2. Outcomes of one matrix ────────────────────────────────────────────
    def outcomes(P):
        """Per site and horizon: stranding per coast, share in R and exits, and the
    move in km of the centre of the material still in R."""
        p = np.zeros((len(W), n_states))
        p[:, :n_cells] = W
        out = {}
        for step in range(1, max(steps.values()) + 1):
            p = p @ P
            for d, k in steps.items():
                if step == k:
                    ocean = p[:, :n_cells]
                    in_R = ocean.sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        lon_now, lat_now = ocean @ cell_lon / in_R, ocean @ cell_lat / in_R
                    out[d] = {"coasts": p[:, n_cells + STRANDED:].copy(), "in_R": in_R, "exits": p[:, n_cells:n_cells + STRANDED].copy(),
                              "east": (lon_now - start_lon) * 111.32 * np.cos(np.radians(start_lat)),
                              "north": (lat_now - start_lat) * 110.57}
        return out

    full, draws = {}, {}
    for name in MATRICES:
        counts = tuple(counts_file[f"{name}_{part}"] for part in ("d_idx", "ft", "n"))
        counts += (int(counts_file[f"{name}_n_drifters"]),)
        full[name] = outcomes(ops[name]["P"])
        rng = np.random.default_rng(SEED)
        collected = [outcomes(row_normalise(resampled_counts(counts, n_states, rng), cells, n_lon)[0]) for _ in range(N_BOOT)]
        draws[name] = {d: {key: np.array([c[d][key] for c in collected]) for key in full[name][d]} for d in HORIZONS}
        print(f"{name:<10} {counts[3]:,} drifters resampled {N_BOOT} times")

    # ── 3. Table of outcomes ─────────────────────────────────────────────────
    rows = []
    for name in MATRICES:
        for d in HORIZONS:
            f, b = full[name][d], draws[name][d]
            any_lo, any_hi = interval(b["coasts"].sum(axis=2))
            coast_lo, coast_hi = interval(b["coasts"])
            in_lo, in_hi = interval(b["in_R"])
            east_lo, east_hi = interval(b["east"])
            north_lo, north_hi = interval(b["north"])
            for i, s in enumerate(sites):
                base = {"spill": s["spill"], "site": s["site"], "matrix": name, "days": d}
                rows.append({**base, "metric": "stranded", "target": "any", "estimate": f["coasts"][i].sum(),
                             "lo": any_lo[i], "hi": any_hi[i]})
                rows += [{**base, "metric": "stranded", "target": c, "estimate": f["coasts"][i, k],
                          "lo": coast_lo[i, k], "hi": coast_hi[i, k]} for k, c in enumerate(COAST_LABELS)]
                rows.append({**base, "metric": "in_R", "target": "R", "estimate": f["in_R"][i], "lo": in_lo[i], "hi": in_hi[i]})
                rows.append({**base, "metric": "move_km", "target": "east", "estimate": f["east"][i], "lo": east_lo[i], "hi": east_hi[i]})
                rows.append({**base, "metric": "move_km", "target": "north", "estimate": f["north"][i], "lo": north_lo[i], "hi": north_hi[i]})
    table = pd.DataFrame(rows)

    # ── 4. The decision rule ─────────────────────────────────────────────────

    def pick(spill, site, matrix, days, metric, target):
        g = table[(table["spill"] == spill) & (table["site"] == site) & (table["matrix"] == matrix)
                  & (table["days"] == days) & (table["metric"] == metric) & (table["target"] == target)]
        return g.iloc[0]

    verdicts = []
    for spill in SPILLS:
        for site in ["spill cell", "nearest reliable"]:
            info = next(s for s in sites if s["spill"] == spill["name"] and s["site"] == site)
            for matrix in OIL_WINDAGE_RANGE:
                stranded = pick(spill["name"], site, matrix, RULE_STRAND_DAYS, "stranded", "any")
                shares = [pick(spill["name"], site, matrix, RULE_STRAND_DAYS, "stranded", c)["estimate"] for c in COAST_LABELS]
                east = pick(spill["name"], site, matrix, RULE_DIRECTION_DAYS, "move_km", "east")
                north = pick(spill["name"], site, matrix, RULE_DIRECTION_DAYS, "move_km", "north")
                parts = [side_verdict(east["lo"], east["hi"], 0.0, False), side_verdict(north["lo"], north["hi"], 0.0, True)]
                direction = "inconsistent" if "inconsistent" in parts else "consistent" if parts == ["consistent"] * 2 else "unclear"
                verdicts.append({
                    "spill": spill["name"], "site": site, "matrix": matrix, "reliable": info["reliable"],
                    "drifters": info["drifters"], "lon": info["lon"], "lat": info["lat"],
                    "stranded_30d": stranded["estimate"], "lo": stranded["lo"], "hi": stranded["hi"],
                    "rule1_stranding": side_verdict(stranded["lo"], stranded["hi"], THRESHOLD, spill["strands"]),
                    "top_coast": COAST_LABELS[int(np.argmax(shares))],
                    "rule2_coast": "consistent" if COAST_LABELS[int(np.argmax(shares))] == EXPECTED_COAST else "inconsistent",
                    "east_km_14d": east["estimate"], "east_lo": east["lo"], "east_hi": east["hi"],
                    "north_km_14d": north["estimate"], "north_lo": north["lo"], "north_hi": north["hi"],
                    "rule3_direction": direction})
    verdicts = pd.DataFrame(verdicts)
    out_table, out_verdicts = f"data/spill_hindcasts_{TAG}.csv", f"data/spill_hindcast_verdicts_{TAG}.csv"
    table.to_csv(out_table, index=False)
    verdicts.to_csv(out_verdicts, index=False)

    # ── 5. Figure: oil after 30 days, where it was released ──────────────────
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        map_kw = {"projection": ccrs.PlateCarree()}
    except ImportError:
        ccrs, map_kw = None, {}
    kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
    # A spill cell that is itself reliable needs no second panel
    panels = [s for s in sites if s["site"] == "spill cell" or (s["site"] == "nearest reliable" and not next(
        x for x in sites if x["spill"] == s["spill"] and x["site"] == "spill cell")["reliable"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 5), subplot_kw=map_kw)
    for ax, s in zip(axes.ravel(), panels):
        spill = next(x for x in SPILLS if x["name"] == s["spill"])
        p = propagate(op0, release(op0, s["lon"], s["lat"]), days=30)
        grid = to_grid(op0, p)
        mesh = ax.pcolormesh(op0["lon_edges"], op0["lat_edges"], np.where(grid > 0, grid, np.nan), cmap="Blues",
                             vmin=0, vmax=0.08, **kw)
        if ccrs:
            ax.set_extent([LON_MIN, 26, -40, -26], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.plot(spill["lon"], spill["lat"], marker="*", color="#d03b3b", ms=12, mec="white", zorder=5, ls="", **kw)
        if s["spill"] == "Treasure":
            for lo, la in ISLANDS.values():
                ax.plot(lo, la, marker="^", color="#222222", ms=5, zorder=5, ls="", **kw)
        v = verdicts[(verdicts["spill"] == s["spill"]) & (verdicts["site"] == s["site"]) & (verdicts["matrix"] == HEADLINE)].iloc[0]
        ax.set_title(f"{s['spill']}, {s['site']} ({s['drifters']} drifters)\n"
                     f"stranded within 30 days {v.stranded_30d:.0%} ({v.lo:.0%}–{v.hi:.0%})", fontsize=10)
    fig.colorbar(mesh, ax=axes, shrink=0.75, extend="max", label="Share of the release per cell at 30 days")
    fig.suptitle(f"Spill hindcasts, oil at 3.5%: star = spill, triangles = islands oiled — {RES:g}°, τ = {TAU:g} d", fontsize=12)
    out_png = f"figures/agulhas_spill_hindcasts_{TAG}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. Summary ───────────────────────────────────────────────────────────
    pd.set_option("display.width", 220)
    print("\nDecision rule (oil at each windage):")
    show = verdicts[["spill", "site", "matrix", "reliable", "drifters", "stranded_30d", "lo", "hi", "rule1_stranding",
                     "top_coast", "rule2_coast", "east_km_14d", "north_km_14d", "rule3_direction"]]
    print(show.round(2).to_string(index=False))
    print("\nDirection intervals at 14 days, km (oil, 3.5%):")
    print(verdicts[verdicts["matrix"] == HEADLINE][["spill", "site", "east_km_14d", "east_lo", "east_hi", "north_km_14d",
                                                    "north_lo", "north_hi"]].round(0).to_string(index=False))
    print("\nStranded by horizon, any coast (%), all matrices:")
    any_rows = table[(table["metric"] == "stranded") & (table["target"] == "any")]
    text = any_rows.assign(text=[f"{100 * e:.0f} ({100 * lo:.0f}–{100 * hi:.0f})" for e, lo, hi
                                 in zip(any_rows["estimate"], any_rows["lo"], any_rows["hi"])])
    print(text.pivot_table(index=["spill", "site", "days"], columns="matrix", values="text", aggfunc="first")[MATRICES].to_string())
    # Both spills were in the austral winter: the same releases on the winter undrogued matrix from seasonal_matrices
    winter = load_operator(operator_path("undrogued_JJA", RES, TAU, args.box))
    print("\nUndrogued stranded within 30 days, any coast (%), all year against June–August (estimates only):")
    for i, s in enumerate(sites):
        q = np.zeros(winter["P"].shape[0])
        q[:len(winter["cell_flat"])] = s["weights"]
        q = propagate(winter, q, days=RULE_STRAND_DAYS)
        print(f"  {s['spill']:<20} {s['site']:<20} {100 * full['undrogued'][RULE_STRAND_DAYS]['coasts'][i].sum():>5.0f} "
              f"{100 * q[len(winter['cell_flat']) + STRANDED]:>5.0f}")
    print("\nSites:")
    print(pd.DataFrame(sites).drop(columns="weights").round(2).to_string(index=False))
    print(f"\nSaved → {out_table}, {out_verdicts}, {out_png}")


if __name__ == "__main__":
    main()
