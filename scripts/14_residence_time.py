from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import (
    BOX,
    RESOLUTIONS,
    TAU_DAYS,
    operator_path
)


# ============================================================
# WHOLE-BOX RESIDENCE TIME
#
# For every ocean cell:
# estimate the expected number of days material remains
# inside the study region before leaving.
#
# Empty rows are treated as unsupported rather than
# permanent physical traps.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


MAX_YEARS = 10
STOP_EPS = 0.000001


print("========================================")
print("WHOLE-BOX RESIDENCE TIME")
print("========================================")

print("\nRegion:", BOX)
print("Lag:", TAU_DAYS, "days")
print("Maximum integration:", MAX_YEARS, "years")


# ============================================================
# SCIPY
# ============================================================

try:
    from scipy.sparse import csr_matrix

except ImportError:

    print("\nERROR: scipy is needed.")
    print("Run:")
    print("pip install scipy")

    raise SystemExit


# ============================================================
# STORAGE
# ============================================================

results = {}

map_data = {}


# ============================================================
# LOOP THROUGH GRID SIZES
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
        "P",
        "row_obs",
        "row_drifters",
        "flagged",
        "empty",
        "cell_flat",
        "lon_edges",
        "lat_edges"
    ]


    for name in needed:

        if name not in data.files:

            print(f"\nERROR: {name} missing.")

            raise SystemExit


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    P = data["P"]

    row_obs = data["row_obs"]

    row_drifters = data["row_drifters"]

    flagged = data["flagged"].astype(bool)

    empty = data["empty"].astype(bool)

    cells = data["cell_flat"].astype(int)

    lon_edges = data["lon_edges"]

    lat_edges = data["lat_edges"]


    n_cells = len(cells)


    # --------------------------------------------------------
    # OCEAN-ONLY MATRIX
    # --------------------------------------------------------

    Q = P[
        :n_cells,
        :n_cells
    ].copy()


    # Empty rows are self-loops in the saved operator.
    #
    # That is useful for normal propagation, but would make
    # empty cells look like infinite residence-time traps.
    #
    # Set their outgoing probabilities to zero for this check.

    Q[
        empty,
        :
    ] = 0


    Q = csr_matrix(
        Q
    )


    # --------------------------------------------------------
    # EXPECTED RESIDENCE TIME
    # --------------------------------------------------------
    #
    # survive[i] =
    # probability that material starting in cell i
    # remains in supported ocean states after k steps.
    #
    # E[T] = tau * sum_k survival(k)
    # --------------------------------------------------------

    max_steps = int(
        round(
            MAX_YEARS
            * 365
            / TAU_DAYS
        )
    )


    survive = np.ones(
        n_cells
    )


    expected_steps = np.zeros(
        n_cells
    )


    steps_used = 0


    print("\nCalculating residence time...")


    for step in range(
        max_steps
    ):

        expected_steps += survive


        survive = (
            Q
            @ survive
        )


        steps_used = (
            step
            + 1
        )


        # Print roughly once per year

        if steps_used % int(
            round(
                365
                / TAU_DAYS
            )
        ) == 0:

            years_now = (
                steps_used
                * TAU_DAYS
                / 365
            )


            print(
                f"  {years_now:.1f} years | "
                f"maximum remaining probability "
                f"{survive.max():.6f}"
            )


        if survive.max() < STOP_EPS:

            print(
                "\nSurvival probability became negligible."
            )

            break


    residence_days = (
        expected_steps
        * TAU_DAYS
    )


    # --------------------------------------------------------
    # RELIABLE STARTING CELLS
    # --------------------------------------------------------

    active = (
        row_obs > 0
    )


    reliable = (
        active
        & ~flagged
        & ~empty
    )


    reliable_time = residence_days[
        reliable
    ]


    n_reliable = int(
        np.count_nonzero(
            reliable
        )
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    mean_time = np.mean(
        reliable_time
    )


    median_time = np.median(
        reliable_time
    )


    q25_time = np.percentile(
        reliable_time,
        25
    )


    q75_time = np.percentile(
        reliable_time,
        75
    )


    q90_time = np.percentile(
        reliable_time,
        90
    )


    max_time = np.max(
        reliable_time
    )


    print(
        f"\nReliable active cells: "
        f"{n_reliable:,}"
    )


    print(
        "\n--- Residence time ---"
    )


    print(
        f"Mean: "
        f"{mean_time:.1f} days"
    )


    print(
        f"Median: "
        f"{median_time:.1f} days"
    )


    print(
        f"25th percentile: "
        f"{q25_time:.1f} days"
    )


    print(
        f"75th percentile: "
        f"{q75_time:.1f} days"
    )


    print(
        f"90th percentile: "
        f"{q90_time:.1f} days"
    )


    print(
        f"Maximum: "
        f"{max_time:.1f} days"
    )


    print(
        f"Steps integrated: "
        f"{steps_used:,}"
    )


    print(
        f"Final maximum survival probability: "
        f"{survive.max():.8f}"
    )


    # --------------------------------------------------------
    # CELL CENTRES
    # --------------------------------------------------------

    n_lon = len(
        lon_edges
    ) - 1


    n_lat = len(
        lat_edges
    ) - 1


    center_lon = np.zeros(
        n_cells
    )


    center_lat = np.zeros(
        n_cells
    )


    for k, flat_cell in enumerate(
        cells
    ):

        i = (
            flat_cell
            % n_lon
        )

        j = (
            flat_cell
            // n_lon
        )


        center_lon[k] = (
            lon_edges[i]
            + lon_edges[i + 1]
        ) / 2


        center_lat[k] = (
            lat_edges[j]
            + lat_edges[j + 1]
        ) / 2


    # --------------------------------------------------------
    # TOP RESIDENCE CELLS
    # --------------------------------------------------------

    reliable_idx = np.where(
        reliable
    )[0]


    order = reliable_idx[
        np.argsort(
            residence_days[
                reliable_idx
            ]
        )[::-1]
    ]


    print(
        "\nTop residence-time cells:"
    )


    for rank, idx in enumerate(
        order[:10],
        start=1
    ):

        print(
            f"{rank:>2}. "
            f"{center_lon[idx]:>6.2f}°E, "
            f"{abs(center_lat[idx]):>5.2f}°S | "
            f"{residence_days[idx]:>7.1f} days | "
            f"drifters {int(row_drifters[idx])}"
        )


    # --------------------------------------------------------
    # SAVE CELL CSV
    # --------------------------------------------------------

    tag = (
        f"{res:g}"
        .replace(".", "p")
    )


    cell_file = (
        DATA_DIR
        / f"residence_time_{tag}deg.csv"
    )


    with open(
        cell_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "lon",
                "lat",
                "residence_days",
                "row_transitions",
                "distinct_drifters",
                "flagged",
                "empty"
            ]
        )


        for k in range(
            n_cells
        ):

            writer.writerow(
                [
                    center_lon[k],
                    center_lat[k],
                    residence_days[k],
                    int(row_obs[k]),
                    int(row_drifters[k]),
                    bool(flagged[k]),
                    bool(empty[k])
                ]
            )


    # --------------------------------------------------------
    # STORE SUMMARY
    # --------------------------------------------------------

    results[
        res
    ] = {

        "reliable": n_reliable,

        "mean": mean_time,

        "median": median_time,

        "q25": q25_time,

        "q75": q75_time,

        "q90": q90_time,

        "max": max_time,

        "steps": steps_used,

        "tail": survive.max()
    }


    map_data[
        res
    ] = {

        "residence": residence_days,

        "reliable": reliable,

        "cells": cells,

        "lon_edges": lon_edges,

        "lat_edges": lat_edges
    }


    print(
        "\nSaved:"
    )

    print(
        cell_file
    )


# ============================================================
# THREE-GRID SUMMARY
# ============================================================

print(
    "\n\n========================================"
)

print(
    "RESIDENCE-TIME COMPARISON"
)

print(
    "========================================"
)


print(
    "\nGrid | Reliable | Mean days | Median days | "
    "75th pct | 90th pct | Maximum"
)


for res in RESOLUTIONS:

    r = results[
        res
    ]


    print(
        f"{res:>4}° | "
        f"{r['reliable']:>8} | "
        f"{r['mean']:>9.1f} | "
        f"{r['median']:>11.1f} | "
        f"{r['q75']:>8.1f} | "
        f"{r['q90']:>8.1f} | "
        f"{r['max']:>7.1f}"
    )


# ============================================================
# SUMMARY CSV
# ============================================================

summary_file = (
    DATA_DIR
    / "residence_time_summary.csv"
)


with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "grid_deg",
            "reliable_cells",
            "mean_residence_days",
            "median_residence_days",
            "q25_days",
            "q75_days",
            "q90_days",
            "max_days",
            "steps_integrated",
            "final_max_survival"
        ]
    )


    for res in RESOLUTIONS:

        r = results[
            res
        ]


        writer.writerow(
            [
                res,
                r["reliable"],
                r["mean"],
                r["median"],
                r["q25"],
                r["q75"],
                r["q90"],
                r["max"],
                r["steps"],
                r["tail"]
            ]
        )


# ============================================================
# 1 DEGREE MAP
# ============================================================

res = 1.0

m = map_data[
    res
]


lon_edges = m[
    "lon_edges"
]

lat_edges = m[
    "lat_edges"
]

cells = m[
    "cells"
]

reliable = m[
    "reliable"
]

residence_days = m[
    "residence"
]


n_lon = len(
    lon_edges
) - 1

n_lat = len(
    lat_edges
) - 1


grid = np.full(
    n_lon * n_lat,
    np.nan
)


grid[
    cells[
        reliable
    ]
] = residence_days[
    reliable
]


grid = grid.reshape(
    n_lat,
    n_lon
)


plot_file = (
    FIGURE_DIR
    / "residence_time_1deg.png"
)


# ------------------------------------------------------------
# MAP WITH CARTOPY IF AVAILABLE
# ------------------------------------------------------------

try:

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature


    fig = plt.figure(
        figsize=(10, 7)
    )


    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )


    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid,
        transform=ccrs.PlateCarree()
    )


    ax.set_extent(
        [
            BOX[0],
            BOX[1],
            BOX[2],
            BOX[3]
        ],
        crs=ccrs.PlateCarree()
    )


    ax.add_feature(
        cfeature.LAND
    )


    ax.coastlines(
        linewidth=0.7
    )


    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.5
    )


    gl.top_labels = False
    gl.right_labels = False


except ImportError:

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid
    )


    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )


plt.colorbar(
    mesh,
    label="Expected residence time (days)"
)


plt.title(
    "Expected Residence Time — 1° Drogued Matrix"
)


plt.tight_layout()


plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:"
)

print(
    summary_file
)

print(
    plot_file
)


print(
    "\nDONE."
)
