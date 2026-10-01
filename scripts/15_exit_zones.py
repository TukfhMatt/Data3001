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
# WHOLE-BOX EXIT ZONES
#
# Current operator has four absorbing exit states:
# W, E, S, N
#
# For each starting ocean cell:
# calculate the probability of eventually leaving through
# each side of the study box.
#
# Empty rows are treated as unsupported rather than
# physical permanent states.
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


MAX_YEARS = 10

STOP_EPS = 0.00000001

MIN_RESOLVED = 0.80


print("========================================")
print("WHOLE-BOX EXIT ZONES")
print("========================================")

print("\nRegion:", BOX)
print("Lag:", TAU_DAYS, "days")
print("Maximum propagation:", MAX_YEARS, "years")


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
        "lat_edges",
        "exit_labels"
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

    exit_labels = data["exit_labels"].astype(str)


    n_cells = len(cells)

    n_exit = len(exit_labels)


    print(
        "\nExit states:",
        ", ".join(exit_labels)
    )


    # --------------------------------------------------------
    # SPLIT MATRIX
    #
    # Q = ocean -> ocean
    # R = ocean -> exit states
    # --------------------------------------------------------

    Q = P[
        :n_cells,
        :n_cells
    ].copy()


    R = P[
        :n_cells,
        n_cells:n_cells + n_exit
    ].copy()


    # --------------------------------------------------------
    # REMOVE EMPTY SELF-LOOPS
    #
    # Saved matrices make empty rows self-loops.
    # For exit analysis that would artificially trap material.
    #
    # Treat empty states as unsupported instead.
    # --------------------------------------------------------

    Q[
        empty,
        :
    ] = 0


    R[
        empty,
        :
    ] = 0


    Q = csr_matrix(
        Q
    )


    # --------------------------------------------------------
    # EVENTUAL EXIT PROBABILITY
    #
    # F = R + Q R + Q^2 R + ...
    #
    # Iteratively:
    # F_(k+1) = R + Q F_k
    #
    # Also save the result after ~365 days.
    # --------------------------------------------------------

    max_steps = int(
        round(
            MAX_YEARS
            * 365
            / TAU_DAYS
        )
    )


    year_steps = int(
        round(
            365
            / TAU_DAYS
        )
    )


    exit_prob = R.copy()

    exit_365 = None

    iterations = 0


    print("\nCalculating exit probabilities...")


    for step in range(
        1,
        max_steps + 1
    ):

        new_exit_prob = (
            R
            + Q @ exit_prob
        )


        change = np.max(
            np.abs(
                new_exit_prob
                - exit_prob
            )
        )


        exit_prob = new_exit_prob

        iterations = step


        if step == year_steps:

            exit_365 = exit_prob.copy()

            print(
                f"  1 year ({step} steps) saved"
            )


        if step % year_steps == 0:

            years_now = (
                step
                * TAU_DAYS
                / 365
            )


            print(
                f"  {years_now:.1f} years | "
                f"maximum change {change:.10f}"
            )


        if change < STOP_EPS:

            print(
                "\nExit probabilities converged."
            )

            break


    if exit_365 is None:

        exit_365 = exit_prob.copy()


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


    # --------------------------------------------------------
    # RESOLVED PROBABILITY
    #
    # Because unsupported empty states were removed,
    # some probability may disappear into unsupported cells.
    #
    # resolved = probability eventually reaching W/E/S/N
    # --------------------------------------------------------

    resolved = np.sum(
        exit_prob,
        axis=1
    )


    unresolved = (
        1
        - resolved
    )


    resolved_365 = np.sum(
        exit_365,
        axis=1
    )


    good_resolved = (
        reliable
        & (
            resolved >= MIN_RESOLVED
        )
    )


    n_good_resolved = int(
        np.count_nonzero(
            good_resolved
        )
    )


    # --------------------------------------------------------
    # DOMINANT EXIT
    # --------------------------------------------------------

    dominant = np.argmax(
        exit_prob,
        axis=1
    )


    dominant_365 = np.argmax(
        exit_365,
        axis=1
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
    # SUMMARY
    # --------------------------------------------------------

    reliable_resolved = resolved[
        reliable
    ]


    reliable_unresolved = unresolved[
        reliable
    ]


    print(
        f"\nReliable active cells: "
        f"{n_reliable:,}"
    )


    print(
        f"Cells with >= {MIN_RESOLVED:.0%} resolved exit probability: "
        f"{n_good_resolved:,} "
        f"({n_good_resolved / n_reliable * 100:.1f}%)"
    )


    print(
        f"Median eventual resolved probability: "
        f"{np.median(reliable_resolved):.1%}"
    )


    print(
        f"Median unsupported probability: "
        f"{np.median(reliable_unresolved):.1%}"
    )


    # --------------------------------------------------------
    # EXIT SHARE BY STARTING CELL
    # --------------------------------------------------------

    print(
        "\n--- Dominant eventual exit side ---"
    )


    side_counts = {}


    for side_number, label in enumerate(
        exit_labels
    ):

        count = int(
            np.count_nonzero(
                good_resolved
                & (
                    dominant
                    == side_number
                )
            )
        )


        if n_good_resolved > 0:

            pct = (
                count
                / n_good_resolved
                * 100
            )

        else:

            pct = 0


        side_counts[
            label
        ] = (
            count,
            pct
        )


        print(
            f"{label}: "
            f"{count:,} cells "
            f"({pct:.1f}%)"
        )


    # --------------------------------------------------------
    # MEAN EXIT PROBABILITY
    # --------------------------------------------------------

    print(
        "\n--- Mean eventual exit probability ---"
    )


    mean_probs = {}


    for side_number, label in enumerate(
        exit_labels
    ):

        values = exit_prob[
            reliable,
            side_number
        ]


        mean_value = np.mean(
            values
        )


        median_value = np.median(
            values
        )


        mean_probs[
            label
        ] = mean_value


        print(
            f"{label}: "
            f"mean {mean_value:.1%}, "
            f"median {median_value:.1%}"
        )


    # --------------------------------------------------------
    # 365-DAY EXIT PROBABILITY
    # --------------------------------------------------------

    print(
        "\n--- Mean exit probability after 365 days ---"
    )


    for side_number, label in enumerate(
        exit_labels
    ):

        value = np.mean(
            exit_365[
                reliable,
                side_number
            ]
        )


        print(
            f"{label}: "
            f"{value:.1%}"
        )


    # --------------------------------------------------------
    # STRONGEST FEEDER CELLS FOR EACH EXIT
    # --------------------------------------------------------

    print(
        "\n--- Strongest feeder cells by exit side ---"
    )


    reliable_idx = np.where(
        reliable
    )[0]


    for side_number, label in enumerate(
        exit_labels
    ):

        print(
            f"\n{label} exit:"
        )


        order = reliable_idx[
            np.argsort(
                exit_prob[
                    reliable_idx,
                    side_number
                ]
            )[::-1]
        ]


        for rank, idx in enumerate(
            order[:5],
            start=1
        ):

            print(
                f"{rank:>2}. "
                f"{center_lon[idx]:>6.2f}°E, "
                f"{abs(center_lat[idx]):>5.2f}°S | "
                f"{exit_prob[idx, side_number]:>6.1%} | "
                f"resolved {resolved[idx]:>6.1%} | "
                f"drifters {int(row_drifters[idx])}"
            )


    # --------------------------------------------------------
    # SAVE PER-CELL CSV
    # --------------------------------------------------------

    tag = (
        f"{res:g}"
        .replace(
            ".",
            "p"
        )
    )


    cell_file = (
        DATA_DIR
        / f"exit_zones_{tag}deg.csv"
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


        header = [
            "lon",
            "lat",
            "row_transitions",
            "distinct_drifters",
            "flagged",
            "empty",
            "resolved_probability",
            "unsupported_probability",
            "dominant_exit"
        ]


        for label in exit_labels:

            header.append(
                f"exit_{label}_prob"
            )


        writer.writerow(
            header
        )


        for k in range(
            n_cells
        ):

            row = [
                center_lon[k],
                center_lat[k],
                int(row_obs[k]),
                int(row_drifters[k]),
                bool(flagged[k]),
                bool(empty[k]),
                resolved[k],
                unresolved[k],
                exit_labels[
                    dominant[k]
                ]
            ]


            row.extend(
                exit_prob[k, :]
            )


            writer.writerow(
                row
            )


    # --------------------------------------------------------
    # STORE SUMMARY
    # --------------------------------------------------------

    results[
        res
    ] = {

        "reliable": n_reliable,

        "good_resolved": n_good_resolved,

        "median_resolved":
            np.median(
                reliable_resolved
            ),

        "mean_probs":
            mean_probs,

        "side_counts":
            side_counts,

        "iterations":
            iterations
    }


    map_data[
        res
    ] = {

        "cells":
            cells,

        "lon_edges":
            lon_edges,

        "lat_edges":
            lat_edges,

        "dominant":
            dominant,

        "resolved":
            resolved,

        "reliable":
            reliable
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
    "EXIT-ZONE COMPARISON"
)

print(
    "========================================"
)


print(
    "\nGrid | Reliable | >=80% resolved | "
    "W dominant | E dominant | S dominant | N dominant"
)


for res in RESOLUTIONS:

    r = results[
        res
    ]


    pieces = []


    for label in [
        "W",
        "E",
        "S",
        "N"
    ]:

        if label in r[
            "side_counts"
        ]:

            pct = r[
                "side_counts"
            ][
                label
            ][
                1
            ]

        else:

            pct = 0


        pieces.append(
            pct
        )


    print(
        f"{res:>4}° | "
        f"{r['reliable']:>8} | "
        f"{r['good_resolved']:>13} | "
        f"{pieces[0]:>9.1f}% | "
        f"{pieces[1]:>9.1f}% | "
        f"{pieces[2]:>9.1f}% | "
        f"{pieces[3]:>9.1f}%"
    )


# ============================================================
# SUMMARY CSV
# ============================================================

summary_file = (
    DATA_DIR
    / "exit_zone_summary.csv"
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
            "cells_ge80pct_resolved",
            "median_resolved_probability",
            "mean_W_probability",
            "mean_E_probability",
            "mean_S_probability",
            "mean_N_probability",
            "iterations"
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
                r["good_resolved"],
                r["median_resolved"],
                r["mean_probs"].get("W", 0),
                r["mean_probs"].get("E", 0),
                r["mean_probs"].get("S", 0),
                r["mean_probs"].get("N", 0),
                r["iterations"]
            ]
        )


# ============================================================
# 1 DEGREE DOMINANT EXIT MAP
# ============================================================

res = 1.0

m = map_data[
    res
]


cells = m[
    "cells"
]

lon_edges = m[
    "lon_edges"
]

lat_edges = m[
    "lat_edges"
]

dominant = m[
    "dominant"
]

resolved = m[
    "resolved"
]

reliable = m[
    "reliable"
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


map_cells = (
    reliable
    & (
        resolved >= MIN_RESOLVED
    )
)


grid[
    cells[
        map_cells
    ]
] = dominant[
    map_cells
]


grid = grid.reshape(
    n_lat,
    n_lon
)


plot_file = (
    FIGURE_DIR
    / "dominant_exit_zone_1deg.png"
)


try:

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from matplotlib.patches import Patch


    fig = plt.figure(
        figsize=(10, 7)
    )


    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )


    cmap = plt.get_cmap(
        "tab10",
        4
    )


    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
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


    handles = []


    labels_for_map = [
        "W",
        "E",
        "S",
        "N"
    ]


    for number, label in enumerate(
        labels_for_map
    ):

        handles.append(
            Patch(
                color=cmap(number),
                label=label
            )
        )


    ax.legend(
        handles=handles,
        title="Dominant exit",
        loc="lower left"
    )


except ImportError:

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid,
        cmap=plt.get_cmap(
            "tab10",
            4
        ),
        vmin=-0.5,
        vmax=3.5
    )


    ax.set_xlabel(
        "Longitude"
    )


    ax.set_ylabel(
        "Latitude"
    )


plt.title(
    "Dominant Eventual Exit Side — 1° Drogued Matrix"
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
