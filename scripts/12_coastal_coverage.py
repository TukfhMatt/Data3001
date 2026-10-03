from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import (
    BOX,
    RESOLUTIONS,
    TAU_DAYS,
    MIN_DRIFTERS,
    operator_path
)


# ============================================================
# SQ2 COASTAL CELL COVERAGE
#
# Coastal cell:
# an ocean-state grid cell that intersects
# the Natural Earth 10m coastline.
#
# Drogued matrices only
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


print("========================================")
print("SQ2 COASTAL CELL COVERAGE")
print("========================================")

print("\nRegion:", BOX)
print("Lag:", TAU_DAYS, "days")
print("Minimum drifters:", MIN_DRIFTERS)


# ============================================================
# LOAD COASTLINE
# ============================================================

try:
    import cartopy.io.shapereader as shpreader

    from shapely.geometry import box
    from shapely.ops import unary_union
    from shapely.prepared import prep

except ImportError:

    print("\nERROR:")
    print("This script needs cartopy and shapely.")
    print("")
    print("Try:")
    print("pip install cartopy shapely")

    raise SystemExit


print("\nLoading Natural Earth 10m coastline...")


coast_file = shpreader.natural_earth(
    resolution="10m",
    category="physical",
    name="coastline"
)


reader = shpreader.Reader(
    coast_file
)


coast_geoms = list(
    reader.geometries()
)


coast_shape = unary_union(
    coast_geoms
)


coast_ready = prep(
    coast_shape
)


print("Coastline loaded.")


# ============================================================
# STORAGE
# ============================================================

all_results = {}


# ============================================================
# LOOP THROUGH THREE GRID SIZES
# ============================================================

for res in RESOLUTIONS:

    print("\n\n========================================")
    print(f"GRID SIZE: {res}°")
    print("========================================")


    matrix_name = operator_path(
        "drogued",
        res,
        TAU_DAYS,
        BOX
    )


    matrix_file = PROJECT_DIR / matrix_name


    print("\nLoading:")
    print(matrix_file)


    if not matrix_file.exists():

        print("\nERROR: matrix file not found.")

        raise SystemExit


    data = np.load(
        matrix_file,
        allow_pickle=True
    )


    needed = [
        "cell_flat",
        "lon_edges",
        "lat_edges",
        "row_obs",
        "row_drifters",
        "flagged",
        "empty"
    ]


    for name in needed:

        if name not in data.files:

            print(f"\nERROR: {name} missing from matrix.")

            raise SystemExit


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    cells = data["cell_flat"].astype(int)

    lon_edges = data["lon_edges"]
    lat_edges = data["lat_edges"]

    row_obs = data["row_obs"]
    row_drifters = data["row_drifters"]

    flagged = data["flagged"].astype(bool)
    empty = data["empty"].astype(bool)


    n_lon = len(lon_edges) - 1


    # --------------------------------------------------------
    # FIND COASTAL CELLS
    # --------------------------------------------------------

    coastal = np.zeros(
        len(cells),
        dtype=bool
    )


    print("\nChecking coastline intersections...")


    for k, flat_cell in enumerate(cells):

        i = flat_cell % n_lon
        j = flat_cell // n_lon


        cell_box = box(
            lon_edges[i],
            lat_edges[j],
            lon_edges[i + 1],
            lat_edges[j + 1]
        )


        if coast_ready.intersects(cell_box):

            coastal[k] = True


    # --------------------------------------------------------
    # ACTIVE COASTAL CELLS
    # --------------------------------------------------------

    active = row_obs > 0

    coastal_active = (
        coastal
        & active
    )


    coastal_obs = row_obs[
        coastal_active
    ]


    coastal_drifters = row_drifters[
        coastal_active
    ]


    n_coastal = int(
        coastal.sum()
    )


    n_coastal_active = int(
        coastal_active.sum()
    )


    n_coastal_empty = int(
        np.count_nonzero(
            coastal & empty
        )
    )


    # --------------------------------------------------------
    # TRANSITION SUPPORT
    # --------------------------------------------------------

    coast_30 = int(
        np.count_nonzero(
            coastal_obs >= 30
        )
    )


    coast_50 = int(
        np.count_nonzero(
            coastal_obs >= 50
        )
    )


    coast_100 = int(
        np.count_nonzero(
            coastal_obs >= 100
        )
    )


    # --------------------------------------------------------
    # DRIFTER SUPPORT
    # --------------------------------------------------------

    coast_d10 = int(
        np.count_nonzero(
            coastal_drifters >= 10
        )
    )


    coast_d20 = int(
        np.count_nonzero(
            coastal_drifters >= 20
        )
    )


    # --------------------------------------------------------
    # FLAGGED
    # --------------------------------------------------------

    coast_flagged = int(
        np.count_nonzero(
            coastal_active
            & flagged
        )
    )


    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------

    if n_coastal_active > 0:

        pct_30 = (
            coast_30
            / n_coastal_active
            * 100
        )

        pct_50 = (
            coast_50
            / n_coastal_active
            * 100
        )

        pct_100 = (
            coast_100
            / n_coastal_active
            * 100
        )

        pct_d10 = (
            coast_d10
            / n_coastal_active
            * 100
        )

        pct_d20 = (
            coast_d20
            / n_coastal_active
            * 100
        )

        pct_flagged = (
            coast_flagged
            / n_coastal_active
            * 100
        )

        med_obs = np.median(
            coastal_obs
        )

        med_drifters = np.median(
            coastal_drifters
        )

    else:

        pct_30 = 0
        pct_50 = 0
        pct_100 = 0

        pct_d10 = 0
        pct_d20 = 0
        pct_flagged = 0

        med_obs = 0
        med_drifters = 0


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    all_results[res] = {

        "coastal_cells": n_coastal,

        "active_coastal": n_coastal_active,

        "empty_coastal": n_coastal_empty,

        "median_transitions": med_obs,

        "pct30": pct_30,

        "pct50": pct_50,

        "pct100": pct_100,

        "median_drifters": med_drifters,

        "pct_d10": pct_d10,

        "pct_d20": pct_d20,

        "flagged": coast_flagged,

        "pct_flagged": pct_flagged
    }


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nCoastal ocean cells: "
        f"{n_coastal:,}"
    )

    print(
        f"Active coastal cells: "
        f"{n_coastal_active:,}"
    )

    print(
        f"Empty coastal cells: "
        f"{n_coastal_empty:,}"
    )


    print("\n--- Coastal transition support ---")

    print(
        f"Median transitions / coastal cell: "
        f"{med_obs:,.0f}"
    )

    print(
        f"Coastal cells >=30 transitions: "
        f"{coast_30:,} ({pct_30:.1f}%)"
    )

    print(
        f"Coastal cells >=50 transitions: "
        f"{coast_50:,} ({pct_50:.1f}%)"
    )

    print(
        f"Coastal cells >=100 transitions: "
        f"{coast_100:,} ({pct_100:.1f}%)"
    )


    print("\n--- Coastal drifter support ---")

    print(
        f"Median distinct drifters / coastal cell: "
        f"{med_drifters:.0f}"
    )

    print(
        f"Coastal cells >=10 drifters: "
        f"{coast_d10:,} ({pct_d10:.1f}%)"
    )

    print(
        f"Coastal cells >=20 drifters: "
        f"{coast_d20:,} ({pct_d20:.1f}%)"
    )


    print("\n--- Coastal reliability ---")

    print(
        f"Flagged coastal cells: "
        f"{coast_flagged:,} ({pct_flagged:.1f}%)"
    )


# ============================================================
# THREE GRID COMPARISON
# ============================================================

print("\n\n========================================")
print("COASTAL THREE-GRID COMPARISON")
print("========================================")


print(
    "\nGrid | Coastal | Active | Median trans | "
    ">=50 trans | Median drifters | >=10 drifters | Flagged"
)


for res in RESOLUTIONS:

    r = all_results[res]


    print(
        f"{res:>4}° | "
        f"{r['coastal_cells']:>7} | "
        f"{r['active_coastal']:>6} | "
        f"{r['median_transitions']:>12,.0f} | "
        f"{r['pct50']:>9.1f}% | "
        f"{r['median_drifters']:>15.0f} | "
        f"{r['pct_d10']:>12.1f}% | "
        f"{r['pct_flagged']:>6.1f}%"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    DATA_DIR
    / "coastal_grid_support.csv"
)


with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)


    writer.writerow(
        [
            "grid_deg",
            "coastal_cells",
            "active_coastal_cells",
            "empty_coastal_cells",
            "median_transitions",
            "pct_coastal_ge30_transitions",
            "pct_coastal_ge50_transitions",
            "pct_coastal_ge100_transitions",
            "median_distinct_drifters",
            "pct_coastal_ge10_drifters",
            "pct_coastal_ge20_drifters",
            "flagged_coastal_cells",
            "pct_flagged_coastal"
        ]
    )


    for res in RESOLUTIONS:

        r = all_results[res]


        writer.writerow(
            [
                res,
                r["coastal_cells"],
                r["active_coastal"],
                r["empty_coastal"],
                r["median_transitions"],
                r["pct30"],
                r["pct50"],
                r["pct100"],
                r["median_drifters"],
                r["pct_d10"],
                r["pct_d20"],
                r["flagged"],
                r["pct_flagged"]
            ]
        )


# ============================================================
# SIMPLE FIGURE
# ============================================================

labels = [
    f"{r:g}°"
    for r in RESOLUTIONS
]


coast_drifter_support = [
    all_results[r]["pct_d10"]
    for r in RESOLUTIONS
]


plot_file = (
    FIGURE_DIR
    / "coastal_drifter_support.png"
)


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    labels,
    coast_drifter_support
)


plt.xlabel(
    "Grid resolution"
)

plt.ylabel(
    "Active coastal cells with >=10 distinct drifters (%)"
)

plt.title(
    "SQ2 — Coastal Cell Drifter Support"
)

plt.ylim(
    0,
    105
)

plt.tight_layout()


plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved:")
print(csv_file)
print(plot_file)

print("\nDONE.")
