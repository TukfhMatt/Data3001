from pathlib import Path
import csv
import numpy as np

from config import (
    BOX,
    RESOLUTIONS,
    TAU_DAYS,
    operator_path
)


# ============================================================
# SQ2 RETENTION / TRAP STABILITY
#
# We use long-term in-box retention as the trap score.
#
# For each starting ocean cell:
#   score = probability still inside R after 365 days
#
# Then:
#   - ignore empty rows
#   - only use reliable cells (not flagged)
#   - top 10% = retention hotspot
#   - compare hotspot locations across 0.5, 1 and 2 degree grids
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

DATA_DIR.mkdir(
    exist_ok=True
)


HOTSPOT_PERCENT = 10

CHECK_DAYS = [
    30,
    365
]

REFERENCE_RES = 0.25


print(
    "========================================"
)

print(
    "SQ2 RETENTION / TRAP STABILITY"
)

print(
    "========================================"
)

print(
    "\nRegion:",
    BOX
)

print(
    "Lag:",
    TAU_DAYS,
    "days"
)

print(
    "Hotspot definition: top",
    HOTSPOT_PERCENT,
    "% of reliable cells by 365-day retention"
)


# ============================================================
# SCIPY
# ============================================================

try:

    from scipy.sparse import csr_matrix

except ImportError:

    print(
        "\nERROR: scipy is needed."
    )

    print(
        "Run:"
    )

    print(
        "pip install scipy"
    )

    raise SystemExit


# ============================================================
# STORAGE
# ============================================================

results = {}

hotspot_info = {}


# ============================================================
# LOOP THROUGH MATRICES
# ============================================================

for res in RESOLUTIONS:

    print(
        "\n\n========================================"
    )

    print(
        f"GRID SIZE: {res}°"
    )

    print(
        "========================================"
    )


    matrix_name = operator_path(
        "drogued",
        res,
        TAU_DAYS,
        BOX
    )


    matrix_file = (
        PROJECT_DIR
        / matrix_name
    )


    print(
        "\nLoading:"
    )

    print(
        matrix_file
    )


    if not matrix_file.exists():

        print(
            "\nERROR: matrix file not found."
        )

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

            print(
                f"\nERROR: {name} missing."
            )

            raise SystemExit


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    P = data[
        "P"
    ]

    row_obs = data[
        "row_obs"
    ]

    row_drifters = data[
        "row_drifters"
    ]

    flagged = data[
        "flagged"
    ].astype(bool)

    empty = data[
        "empty"
    ].astype(bool)

    cells = data[
        "cell_flat"
    ].astype(int)

    lon_edges = data[
        "lon_edges"
    ]

    lat_edges = data[
        "lat_edges"
    ]


    n_cells = len(
        cells
    )


    # --------------------------------------------------------
    # OCEAN-ONLY TRANSITION MATRIX
    # --------------------------------------------------------

    Q = P[
        :n_cells,
        :n_cells
    ].copy()


    # Empty rows are self-loops in the normal model.
    # That is useful for propagation, but bad for trap detection:
    # an empty cell would look like 100% retention forever.
    #
    # Here we treat entering an empty state as unsupported,
    # rather than as a real physical trap.

    Q[
        empty,
        :
    ] = 0


    Q_sparse = csr_matrix(
        Q
    )


    del Q


    # --------------------------------------------------------
    # RETENTION PROBABILITY
    # --------------------------------------------------------
    #
    # survive[i] =
    # probability that material starting from cell i
    # is still in supported ocean states after k steps.
    #
    # s(k+1) = Q @ s(k)
    # s(0) = 1
    # --------------------------------------------------------

    steps_needed = {}


    for days in CHECK_DAYS:

        steps = int(
            round(
                days
                / TAU_DAYS
            )
        )

        steps_needed[
            steps
        ] = days


    max_steps = max(
        steps_needed.keys()
    )


    survive = np.ones(
        n_cells
    )


    retention = {}


    print(
        "\nPropagating retention..."
    )


    for step in range(
        1,
        max_steps + 1
    ):

        survive = (
            Q_sparse
            @ survive
        )


        if step in steps_needed:

            days = steps_needed[
                step
            ]

            retention[
                days
            ] = survive.copy()


            print(
                f"{days}-day retention calculated "
                f"({step} steps)"
            )


    # --------------------------------------------------------
    # RELIABLE CELLS
    # --------------------------------------------------------

    active = (
        row_obs > 0
    )


    reliable = (
        active
        & ~flagged
        & ~empty
    )


    n_reliable = int(
        np.count_nonzero(
            reliable
        )
    )


    retention_30 = retention[
        30
    ]

    retention_365 = retention[
        365
    ]


    reliable_30 = retention_30[
        reliable
    ]

    reliable_365 = retention_365[
        reliable
    ]


    # --------------------------------------------------------
    # HOTSPOT THRESHOLD
    # --------------------------------------------------------

    percentile_cut = (
        100
        - HOTSPOT_PERCENT
    )


    threshold = np.percentile(
        reliable_365,
        percentile_cut
    )


    hotspot = (
        reliable
        & (
            retention_365
            >= threshold
        )
    )


    n_hotspot = int(
        np.count_nonzero(
            hotspot
        )
    )


    # --------------------------------------------------------
    # CELL CENTRES
    # --------------------------------------------------------

    n_lon = (
        len(lon_edges)
        - 1
    )


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
    # SUMMARY
    # --------------------------------------------------------

    median_30 = np.median(
        reliable_30
    )


    median_365 = np.median(
        reliable_365
    )


    q75_365 = np.percentile(
        reliable_365,
        75
    )


    max_365 = np.max(
        reliable_365
    )


    print(
        f"\nReliable active cells: "
        f"{n_reliable:,}"
    )


    print(
        "\n--- Retention ---"
    )


    print(
        f"Median 30-day retention: "
        f"{median_30:.1%}"
    )


    print(
        f"Median 365-day retention: "
        f"{median_365:.1%}"
    )


    print(
        f"75th percentile 365-day retention: "
        f"{q75_365:.1%}"
    )


    print(
        f"Maximum 365-day retention: "
        f"{max_365:.1%}"
    )


    print(
        "\n--- Hotspots ---"
    )


    print(
        f"Top-{HOTSPOT_PERCENT}% threshold: "
        f"{threshold:.1%}"
    )


    print(
        f"Hotspot cells: "
        f"{n_hotspot:,}"
    )


    # --------------------------------------------------------
    # TOP 10 CELLS
    # --------------------------------------------------------

    reliable_indices = np.where(
        reliable
    )[0]


    order = reliable_indices[
        np.argsort(
            retention_365[
                reliable_indices
            ]
        )[
            ::-1
        ]
    ]


    top_n = min(
        10,
        len(order)
    )


    print(
        "\nTop retention cells:"
    )


    for rank in range(
        top_n
    ):

        idx = order[
            rank
        ]


        print(
            f"{rank + 1:>2}. "
            f"{center_lon[idx]:>6.2f}°E, "
            f"{abs(center_lat[idx]):>5.2f}°S | "
            f"30 d {retention_30[idx]:>6.1%} | "
            f"365 d {retention_365[idx]:>6.1%} | "
            f"drifters {int(row_drifters[idx])}"
        )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results[
        res
    ] = {

        "reliable": n_reliable,

        "median30": median_30,

        "median365": median_365,

        "q75_365": q75_365,

        "max365": max_365,

        "threshold": threshold,

        "hotspots": n_hotspot
    }


    hotspot_info[
        res
    ] = {

        "hotspot": hotspot,

        "cells": cells,

        "lon_edges": lon_edges,

        "lat_edges": lat_edges,

        "n_lon": n_lon
    }


# ============================================================
# COMMON REFERENCE GRID
#
# Rasterise all three hotspot sets onto the same fine grid
# so that different resolutions can be compared spatially.
# ============================================================

print(
    "\n\n========================================"
)

print(
    "HOTSPOT STABILITY ACROSS GRIDS"
)

print(
    "========================================"
)


lon_min, lon_max, lat_min, lat_max = BOX


ref_lon = np.arange(
    lon_min + REFERENCE_RES / 2,
    lon_max,
    REFERENCE_RES
)


ref_lat = np.arange(
    lat_min + REFERENCE_RES / 2,
    lat_max,
    REFERENCE_RES
)


xx, yy = np.meshgrid(
    ref_lon,
    ref_lat
)


points_lon = xx.ravel()

points_lat = yy.ravel()


reference = {}


for res in RESOLUTIONS:

    info = hotspot_info[
        res
    ]


    lon_edges = info[
        "lon_edges"
    ]

    lat_edges = info[
        "lat_edges"
    ]

    cells = info[
        "cells"
    ]

    hotspot = info[
        "hotspot"
    ]


    n_lon = (
        len(lon_edges)
        - 1
    )

    n_lat = (
        len(lat_edges)
        - 1
    )


    # state lookup:
    # flattened grid cell -> ocean state number

    lookup = np.full(
        n_lon * n_lat,
        -1,
        dtype=int
    )


    lookup[
        cells
    ] = np.arange(
        len(cells)
    )


    i = np.floor(
        (
            points_lon
            - lon_edges[0]
        )
        / res
    ).astype(int)


    j = np.floor(
        (
            points_lat
            - lat_edges[0]
        )
        / res
    ).astype(int)


    inside = (
        (i >= 0)
        & (i < n_lon)
        & (j >= 0)
        & (j < n_lat)
    )


    flat = (
        j * n_lon
        + i
    )


    ocean = np.zeros(
        len(points_lon),
        dtype=bool
    )


    hot = np.zeros(
        len(points_lon),
        dtype=bool
    )


    valid_points = np.where(
        inside
    )[0]


    states = lookup[
        flat[
            valid_points
        ]
    ]


    is_ocean = (
        states >= 0
    )


    ocean_points = valid_points[
        is_ocean
    ]


    ocean[
        ocean_points
    ] = True


    ocean_states = states[
        is_ocean
    ]


    hot[
        ocean_points
    ] = hotspot[
        ocean_states
    ]


    reference[
        res
    ] = {

        "ocean": ocean,

        "hot": hot
    }


# ============================================================
# PAIRWISE COMPARISON
# ============================================================

pairs = [
    (0.5, 1.0),
    (1.0, 2.0),
    (0.5, 2.0)
]


stability_rows = []


print(
    "\nGrid pair | Jaccard | Overlap coefficient"
)


for a, b in pairs:

    ocean_a = reference[
        a
    ][
        "ocean"
    ]

    ocean_b = reference[
        b
    ][
        "ocean"
    ]


    hot_a = reference[
        a
    ][
        "hot"
    ]

    hot_b = reference[
        b
    ][
        "hot"
    ]


    common_ocean = (
        ocean_a
        & ocean_b
    )


    a_common = (
        hot_a
        & common_ocean
    )


    b_common = (
        hot_b
        & common_ocean
    )


    intersection = np.count_nonzero(
        a_common
        & b_common
    )


    union = np.count_nonzero(
        a_common
        | b_common
    )


    size_a = np.count_nonzero(
        a_common
    )

    size_b = np.count_nonzero(
        b_common
    )


    if union > 0:

        jaccard = (
            intersection
            / union
        )

    else:

        jaccard = 0


    smaller = min(
        size_a,
        size_b
    )


    if smaller > 0:

        overlap = (
            intersection
            / smaller
        )

    else:

        overlap = 0


    print(
        f"{a:g}° vs {b:g}° | "
        f"{jaccard:>7.1%} | "
        f"{overlap:>18.1%}"
    )


    stability_rows.append(
        [
            a,
            b,
            intersection,
            union,
            jaccard,
            overlap
        ]
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_file = (
    DATA_DIR
    / "retention_hotspot_summary.csv"
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
            "median_30d_retention",
            "median_365d_retention",
            "q75_365d_retention",
            "max_365d_retention",
            "hotspot_threshold",
            "hotspot_cells"
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
                r["median30"],
                r["median365"],
                r["q75_365"],
                r["max365"],
                r["threshold"],
                r["hotspots"]
            ]
        )


stability_file = (
    DATA_DIR
    / "retention_hotspot_stability.csv"
)


with open(
    stability_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "grid_a",
            "grid_b",
            "intersection_points",
            "union_points",
            "jaccard",
            "overlap_coefficient"
        ]
    )


    writer.writerows(
        stability_rows
    )


print(
    "\nSaved:"
)

print(
    summary_file
)

print(
    stability_file
)


print(
    "\nDONE."
)
