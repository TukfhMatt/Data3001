"""
Agulhas Region — Oil Explorer
Builds a single web page: click anywhere in R and watch where oil released there
goes over a year, step by step, in the matrix's own τ steps. The page holds three
matrices with their stranding split by coast (coastal_stranding): oil, and the
undrogued and drogued drifters it is built from, which a toggle switches between, and
does the matrix steps in the browser, so it needs no server. For oil it offers two
views: where the oil drifts (transport only, as in the atlas) and how much is left for
heavy fuel oil (config.oil_mass_left, with its headline residue time).

It reports what a reader needs to judge the answer: the drifters behind the
release cell in the chosen matrix, whether the cell is thin, the atlas's 95%
intervals at 7 days, 30 days and 1 year (risk_atlas), and, for oil, whether the
result depends on the method (oil_wind). Not part of the pipeline; it only reads its outputs, so run the pipeline first.
The page itself is agulhas/tools/oil_explorer.html, with the data filled in here.

    python -m agulhas oil_explorer
"""

import base64
import json
import os

import numpy as np
import pandas as pd

from agulhas.config import (DISAGREE_TVD, MIN_DRIFTERS, OIL_EVAPORATES, OIL_EVAPORATION_DAYS, OIL_RESIDUE_DAYS, operator_path,
                            parse_args, release_points_path, setting_tag)
from agulhas.transport import COAST_LABELS, COAST_NAMES, EXIT_LABELS, STRANDED, check_coast_box, load_operator, reliable

HORIZONS = [7, 30, 365]
SIMPLIFY_DEG = 0.03   # coastline detail kept on the page
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oil_explorer.html")
MARKER = "/*DATA*/null"
MATRICES = ["oil", "undrogued", "drogued"]   # in the order of the page's toggle; oil is the headline


def encode(array, dtype):
    """Base64 of an array's bytes, read back with a typed array in the browser."""
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode()


def per_mille(values):
    return np.round(values.to_numpy() * 1000).astype(int).tolist()


# ── What the page holds ──────────────────────────────────────────────────────

def method_tvd(args, n_cells):
    """Per cell, how far oil and its check are apart at 30 days (as in worked_example)."""
    steps_30 = int(round(30 / args.tau))
    oil_30 = np.linalg.matrix_power(load_operator(operator_path("oil", args.res, args.tau, args.box))["P"], steps_30)
    check_30 = np.linalg.matrix_power(load_operator(operator_path("oil_check", args.res, args.tau, args.box))["P"],
                                      steps_30)
    return 0.5 * np.abs(oil_30 - check_30).sum(axis=1)[:n_cells]


def atlas_intervals(atlas, name, tau):
    """The atlas's intervals for one matrix per release cell, in per mille, keyed as the page reads them."""
    atlas = atlas[atlas["matrix"] == name].sort_values("state")
    intervals = {}
    for horizon in HORIZONS:
        part = atlas[atlas["horizon"] == str(horizon)]
        intervals[int(round(horizon / tau))] = {o: {"lo": per_mille(g["lo"]), "hi": per_mille(g["hi"])}
                                                for o, g in part.groupby("outcome")}
    part = atlas[atlas["horizon"] == "eventual"]
    eventual = {o: {"est": per_mille(g["estimate"]), "lo": per_mille(g["lo"]), "hi": per_mille(g["hi"])}
                for o, g in part[part["outcome"] != "days_to_strand"].groupby("outcome")}
    days = part[part["outcome"] == "days_to_strand"]
    days_to_strand = [None if not np.isfinite(e) else [round(e), round(lo), round(hi)]
                      for e, lo, hi in zip(days["estimate"], days["lo"], days["hi"])]
    return {"intervals": intervals, "eventual": eventual, "daysToStrand": days_to_strand}


def coastline(box):
    """Land polygons clipped to the box, or none when cartopy is missing."""
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box as shapely_box
    except ImportError:
        print("cartopy not found, so the map has no coastline.")
        return []
    lon_min, lon_max, lat_min, lat_max = box
    clip = shapely_box(lon_min, lat_min, lon_max, lat_max)
    land = []
    for geom in shpreader.Reader(shpreader.natural_earth("50m", "physical", "land")).geometries():
        part = geom.intersection(clip).simplify(SIMPLIFY_DEG)
        for poly in getattr(part, "geoms", [part]):
            if poly.geom_type == "Polygon" and not poly.is_empty:
                land.append(np.round(np.array(poly.exterior.coords), 2).tolist())
    return land


def matrix_data(op, intervals, tvd=None):
    """One matrix as the page reads it, stored row by row so the page holds exact values."""
    P = op["P"]
    n_cells = len(op["cell_flat"])
    rows, cols = np.nonzero(P)
    row_start = np.searchsorted(rows, np.arange(P.shape[0] + 1))
    return {
        "rowStart": encode(row_start, "<u4"), "col": encode(cols, "<u2"), "val": encode(P[rows, cols], "<f8"),
        "drifters": op["row_drifters"][:n_cells].astype(int).tolist(),
        "flagged": op["flagged"][:n_cells].astype(int).tolist(), "empty": op["empty"][:n_cells].astype(int).tolist(),
        "reliable": reliable(op)[:n_cells].astype(int).tolist(),
        "methodTvd": None if tvd is None else np.round(tvd, 3).tolist(),
        **intervals,
    }


def page_data(ops, args, tvd, atlas, points, land):
    """Everything the page reads: what the matrices share, then each matrix in the order of MATRICES."""
    oil = ops["oil"]
    for name, op in ops.items():
        if not np.array_equal(op["cell_flat"], oil["cell_flat"]):
            raise SystemExit(f"The {name} and oil matrices have different ocean cells.")
    return {
        "box": list(args.box), "res": args.res, "tau": args.tau,
        "nLon": len(oil["lon_edges"]) - 1, "nLat": len(oil["lat_edges"]) - 1,
        "nCells": len(oil["cell_flat"]), "nStates": oil["P"].shape[0],
        "cellFlat": oil["cell_flat"].astype(int).tolist(),
        "edges": EXIT_LABELS[:STRANDED].tolist(), "coasts": COAST_LABELS.tolist(), "coastNames": COAST_NAMES.tolist(),
        "disagreeTvd": DISAGREE_TVD, "minDrifters": MIN_DRIFTERS,
        "evaporates": OIL_EVAPORATES, "evaporationDays": OIL_EVAPORATION_DAYS,
        "residueDays": OIL_RESIDUE_DAYS["mass"], "residueRange": [min(OIL_RESIDUE_DAYS.values()), max(OIL_RESIDUE_DAYS.values())],
        "alphaOil": float(oil["alpha_oil"]),
        "alphaUndrogued": float(oil["alpha_undrogued"]), "alphaAdded": float(oil["alpha_added"]),
        "years": oil["years"].astype(int).tolist(),
        "maxSteps": int(round(HORIZONS[-1] / args.tau)),
        "matrices": {name: matrix_data(ops[name], atlas_intervals(atlas, name, args.tau), tvd if name == "oil" else None)
                     for name in MATRICES},
        "points": points.to_dict("records"), "land": land,
    }


def write_page(data, path):
    """The page template with the data filled in."""
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read()
    if MARKER not in page:
        raise SystemExit(f"{TEMPLATE} has no data marker.")
    page = page.replace(MARKER, json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    tag = setting_tag(args.res, args.tau, args.box)
    out = f"figures/agulhas_oil_explorer_{tag}.html"

    ops = {name: load_operator(operator_path(f"{name}_coast", args.res, args.tau, args.box)) for name in MATRICES}
    for name, op in ops.items():
        print(f"{name.capitalize()} matrix: {op['P'].shape[0]} states, {np.count_nonzero(op['P']):,} non-zero entries")
    tvd = method_tvd(args, len(ops["oil"]["cell_flat"]))
    print(f"Cells where oil and its check differ by TVD > {DISAGREE_TVD} at 30 days: {(tvd > DISAGREE_TVD).sum()}")
    atlas = pd.read_csv(f"data/risk_atlas_{tag}.csv", low_memory=False)
    points = pd.read_csv(release_points_path(args.box))[["kind", "name", "lon", "lat"]]
    land = coastline(args.box)

    write_page(page_data(ops, args, tvd, atlas, points, land), out)
    print(f"Coastline: {len(land)} polygons. Page saved → {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
