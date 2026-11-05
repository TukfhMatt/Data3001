"""
Agulhas Region — Oil Explorer
Builds a single web page: click anywhere in R and watch where oil released there
goes over a year, step by step, in the matrix's own τ steps. The page holds the
oil matrix with its stranding split by coast (coastal_stranding), and does the
matrix steps in the browser, so it needs no server. It offers two views: where the oil drifts
(transport only, as in the atlas) and how much is left with exp(−t / 14 d).

It reports what a reader needs to judge the answer: the drifters behind the
release cell, whether the cell is thin, the atlas's 95% intervals at 7 days, 30
days and 1 year (risk_atlas), and whether the result depends on the method
(oil_wind). Not part of the pipeline; it only reads its outputs, so run the pipeline first.
The page itself is agulhas/tools/oil_explorer.html, with the data filled in here.

    python -m agulhas oil_explorer
"""

import base64
import json
import os

import numpy as np
import pandas as pd

from agulhas.config import (DISAGREE_TVD, MIN_DRIFTERS, operator_path, parse_args, release_points_path, setting_tag)
from agulhas.transport import COAST_LABELS, COAST_NAMES, EXIT_LABELS, STRANDED, check_coast_box, load_operator, reliable

HORIZONS = [7, 30, 365]
SIMPLIFY_DEG = 0.03   # coastline detail kept on the page


def encode(array, dtype):
    """Base64 of an array's bytes, read back with a typed array in the browser."""
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode()


def per_mille(values):
    return np.round(values.to_numpy() * 1000).astype(int).tolist()


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
    RES, TAU = args.res, args.tau
    TAG = setting_tag(RES, TAU, args.box)
    TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oil_explorer.html")

    # ── 1. The matrix, stored row by row so the page holds exact values ──────
    op = load_operator(operator_path("oil_coast", RES, TAU, args.box))
    P = op["P"]
    n_cells = len(op["cell_flat"])
    rows, cols = np.nonzero(P)
    row_start = np.searchsorted(rows, np.arange(P.shape[0] + 1))
    print(f"Oil matrix: {P.shape[0]} states, {len(rows):,} non-zero entries")

    # ── 2. Does the answer depend on the method? (as in worked_example) ──────
    check = load_operator(operator_path("oil_check", RES, TAU, args.box))
    steps_30 = int(round(30 / TAU))
    oil_30 = np.linalg.matrix_power(load_operator(operator_path("oil", RES, TAU, args.box))["P"], steps_30)
    check_30 = np.linalg.matrix_power(check["P"], steps_30)
    method_tvd = 0.5 * np.abs(oil_30 - check_30).sum(axis=1)[:n_cells]
    print(f"Cells where oil and its check differ by TVD > {DISAGREE_TVD} at 30 days: {(method_tvd > DISAGREE_TVD).sum()}")

    # ── 3. Intervals from the atlas ──────────────────────────────────────────
    atlas = pd.read_csv(f"data/risk_atlas_{TAG}.csv", low_memory=False)
    atlas = atlas[atlas["matrix"] == "oil"].sort_values("state")

    intervals = {}
    for horizon in HORIZONS:
        part = atlas[atlas["horizon"] == str(horizon)]
        intervals[int(round(horizon / TAU))] = {o: {"lo": per_mille(g["lo"]), "hi": per_mille(g["hi"])}
                                                for o, g in part.groupby("outcome")}
    part = atlas[atlas["horizon"] == "eventual"]
    eventual = {o: {"est": per_mille(g["estimate"]), "lo": per_mille(g["lo"]), "hi": per_mille(g["hi"])}
                for o, g in part[part["outcome"] != "days_to_strand"].groupby("outcome")}
    days = part[part["outcome"] == "days_to_strand"]
    days_to_strand = [None if not np.isfinite(e) else [round(e), round(lo), round(hi)]
                      for e, lo, hi in zip(days["estimate"], days["lo"], days["hi"])]

    # ── 4. Named release points and the coastline ────────────────────────────
    points = pd.read_csv(release_points_path(args.box))[["kind", "name", "lon", "lat"]]
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box as shapely_box
        clip = shapely_box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
        land = []
        for geom in shpreader.Reader(shpreader.natural_earth("50m", "physical", "land")).geometries():
            part = geom.intersection(clip).simplify(SIMPLIFY_DEG)
            for poly in getattr(part, "geoms", [part]):
                if poly.geom_type == "Polygon" and not poly.is_empty:
                    land.append(np.round(np.array(poly.exterior.coords), 2).tolist())
    except ImportError:
        land = []
        print("cartopy not found, so the map has no coastline.")

    # ── 5. Fill in the page ──────────────────────────────────────────────────
    data = {
        "box": [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], "res": RES, "tau": TAU,
        "nLon": len(op["lon_edges"]) - 1, "nLat": len(op["lat_edges"]) - 1,
        "nCells": n_cells, "nStates": P.shape[0],
        "cellFlat": op["cell_flat"].astype(int).tolist(),
        "rowStart": encode(row_start, "<u4"), "col": encode(cols, "<u2"), "val": encode(P[rows, cols], "<f8"),
        "edges": EXIT_LABELS[:STRANDED].tolist(), "coasts": COAST_LABELS.tolist(), "coastNames": COAST_NAMES.tolist(),
        "drifters": op["row_drifters"][:n_cells].astype(int).tolist(),
        "flagged": op["flagged"][:n_cells].astype(int).tolist(), "empty": op["empty"][:n_cells].astype(int).tolist(),
        "reliable": reliable(op)[:n_cells].astype(int).tolist(),
        "methodTvd": np.round(method_tvd, 3).tolist(), "disagreeTvd": DISAGREE_TVD, "minDrifters": MIN_DRIFTERS,
        "decayDays": float(op["decay_days"]), "alphaOil": float(op["alpha_oil"]),
        "alphaUndrogued": float(op["alpha_undrogued"]), "alphaAdded": float(op["alpha_added"]),
        "years": op["years"].astype(int).tolist(),
        "maxSteps": int(round(HORIZONS[-1] / TAU)),
        "intervals": intervals, "eventual": eventual, "daysToStrand": days_to_strand,
        "points": points.to_dict("records"), "land": land,
    }
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    marker = "/*DATA*/null"
    if marker not in page:
        raise SystemExit(f"{TEMPLATE} has no data marker.")
    page = page.replace(marker, json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    os.makedirs("figures", exist_ok=True)
    out = f"figures/agulhas_oil_explorer_{TAG}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Coastline: {len(land)} polygons. Page saved → {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
