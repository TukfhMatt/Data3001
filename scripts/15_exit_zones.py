from pathlib import Path
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")

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
# The transport operator has four absorbing exit states:
#
#   W, E, S, N
#
# For each starting ocean cell, estimate the probability
# that material eventually exits through each side.
#
# Empty rows in the saved matrix are self-loops.
# Here they are treated as unsupported instead of
# permanent physical states.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent.parent


DATA_DIR = (
    PROJECT_DIR
    / "data"
)


FIGURE_DIR = (
    PROJECT_DIR
    / "figures"
)


DATA_DIR.mkdir(
    exist_ok=True
)


FIGURE_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MAX_YEARS = 10

STOP_EPS = 0.00000001

MIN_RESOLVED = 0.80


# ============================================================
# START
# ============================================================

print(
    "========================================"
)

print(
    "WHOLE-BOX EXIT ZONES"
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
    "Maximum propagation:",
    MAX_YEARS,
    "years"
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

map_data = {}


# ============================================================
# LOOP THROUGH GRID SIZES
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


    # --------------------------------------------------------
    # MATRIX PATH
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LOAD MATRIX
    # --------------------------------------------------------

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

            print(
                f"\nERROR: {name} missing."
            )

            raise SystemExit


    # --------------------------------------------------------
    # VARIABLES
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
    ].astype(
        bool
    )


    empty = data[
        "empty"
    ].astype(
        bool
    )


    cells = data[
        "cell_flat"
    ].astype(
        int
    )


    lon_edges = data[
        "lon_edges"
    ]


    lat_edges = data[
        "lat_edges"
    ]


    exit_labels = data[
        "exit_labels"
    ].astype(
        str
    )


    n_cells = len(
        cells
    )


    n_exit = len(
        exit_labels
    )


    print(
        "\nExit states:",
        ", ".join(
            exit_labels
        )
    )


    # ========================================================
    # SPLIT MATRIX
    #
    # Q = ocean -> ocean
    # R = ocean -> absorbing exit states
    # ========================================================

    Q = P[
        :n_cells,
        :n_cells
    ].copy()


    R = P[
        :n_cells,
        n_cells:
        n_cells + n_exit
    ].copy()


    # ========================================================
    # REMOVE EMPTY SELF-LOOPS
    #
    # Empty states have no observed outgoing transitions.
    # The normal operator gives them a self-loop so P stays
    # stochastic.
    #
    # For exit analysis this would falsely trap probability.
    # Treat them as unsupported instead.
    # ========================================================

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


    # ========================================================
    # EVENTUAL EXIT PROBABILITY
    #
    # Exit probability within k transitions:
    #
    # F_k = R + Q R + ... + Q^(k-1) R
    #
    # Recurrence:
    #
    # F_0 = 0
    # F_(k+1) = R + Q F_k
    #
    # IMPORTANT:
    # Starting from zero means step 104 really means
    # 104 transitions, not 105.
    # ========================================================

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


    model_year_days = (
        year_steps
        * TAU_DAYS
    )


    # FIX:
    # Start from zero, not R.
    exit_prob = np.zeros_like(
        R
    )


    exit_365 = None

    iterations = 0

    last_change = np.nan

    converged = False


    print(
        "\nCalculating exit probabilities..."
    )


    print(
        f"1-year snapshot: "
        f"{year_steps} steps = "
        f"{model_year_days:.1f} model days"
    )


    for step in range(
        1,
        max_steps + 1
    ):

        new_exit_prob = (
            R
            + Q @ exit_prob
        )


        change = float(
            np.max(
                np.abs(
                    new_exit_prob
                    - exit_prob
                )
            )
        )


        exit_prob = new_exit_prob

        iterations = step

        last_change = change


        # ----------------------------------------------------
        # SAVE ~1 YEAR
        # ----------------------------------------------------

        if step == year_steps:

            exit_365 = (
                exit_prob.copy()
            )


            print(
                f"  1 year "
                f"({step} steps = "
                f"{model_year_days:.1f} days) saved"
            )


        # ----------------------------------------------------
        # PRINT ABOUT ONCE PER YEAR
        # ----------------------------------------------------

        if (
            step % year_steps
            == 0
        ):

            years_now = (
                step
                * TAU_DAYS
                / 365
            )


            print(
                f"  {years_now:.1f} years | "
                f"maximum change "
                f"{change:.10f}"
            )


        # ----------------------------------------------------
        # DO NOT STOP BEFORE 1-YEAR RESULT EXISTS
        # ----------------------------------------------------

        if (
            step >= year_steps
            and change < STOP_EPS
        ):

            converged = True


            print(
                "\nExit probabilities converged."
            )


            break


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if exit_365 is None:

        print(
            "\nERROR: 1-year exit probability was not calculated."
        )

        raise SystemExit


    # ========================================================
    # RELIABLE CELLS
    # ========================================================

    active = (
        row_obs
        > 0
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


    if n_reliable == 0:

        print(
            "\nERROR: no reliable cells."
        )

        raise SystemExit


    # ========================================================
    # RESOLVED PROBABILITY
    #
    # Probability assigned to W/E/S/N after the long
    # integration.
    #
    # Anything left over may represent:
    #   - transport into unsupported empty cells
    #   - a very small unexited tail after finite integration
    # ========================================================

    resolved = np.sum(
        exit_prob,
        axis=1
    )


    unresolved = np.maximum(
        0,
        1
        - resolved
    )


    resolved_365 = np.sum(
        exit_365,
        axis=1
    )


    unresolved_365 = np.maximum(
        0,
        1
        - resolved_365
    )


    good_resolved = (
        reliable
        & (
            resolved
            >= MIN_RESOLVED
        )
    )


    n_good_resolved = int(
        np.count_nonzero(
            good_resolved
        )
    )


    # ========================================================
    # DOMINANT EXIT
    # ========================================================

    dominant = np.argmax(
        exit_prob,
        axis=1
    )


    dominant_365 = np.argmax(
        exit_365,
        axis=1
    )


    # ========================================================
    # CELL CENTRES
    # ========================================================

    n_lon = (
        len(
            lon_edges
        )
        - 1
    )


    n_lat = (
        len(
            lat_edges
        )
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


        center_lon[
            k
        ] = (
            lon_edges[i]
            + lon_edges[i + 1]
        ) / 2


        center_lat[
            k
        ] = (
            lat_edges[j]
            + lat_edges[j + 1]
        ) / 2


    # ========================================================
    # SUMMARY
    # ========================================================

    reliable_resolved = resolved[
        reliable
    ]


    reliable_unresolved = unresolved[
        reliable
    ]


    reliable_resolved_365 = resolved_365[
        reliable
    ]


    print(
        f"\nReliable active cells: "
        f"{n_reliable:,}"
    )


    print(
        f"Cells with >= "
        f"{MIN_RESOLVED:.0%} "
        f"resolved eventual exit probability: "
        f"{n_good_resolved:,} "
        f"({n_good_resolved / n_reliable * 100:.1f}%)"
    )


    print(
        f"Median eventual resolved probability: "
        f"{np.median(reliable_resolved):.1%}"
    )


    print(
        f"Median unresolved / unsupported probability: "
        f"{np.median(reliable_unresolved):.1%}"
    )


    print(
        f"Median resolved probability at "
        f"{model_year_days:.1f} days: "
        f"{np.median(reliable_resolved_365):.1%}"
    )


    print(
        f"Iterations used: "
        f"{iterations:,}"
    )


    print(
        f"Final maximum change: "
        f"{last_change:.10f}"
    )


    print(
        f"Converged: "
        f"{converged}"
    )


    # ========================================================
    # DOMINANT EVENTUAL EXIT
    # ========================================================

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


    # ========================================================
    # MEAN EVENTUAL EXIT PROBABILITY
    # ========================================================

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


        mean_value = float(
            np.mean(
                values
            )
        )


        median_value = float(
            np.median(
                values
            )
        )


        mean_probs[
            label
        ] = mean_value


        print(
            f"{label}: "
            f"mean {mean_value:.1%}, "
            f"median {median_value:.1%}"
        )


    # ========================================================
    # APPROXIMATELY ONE-YEAR EXIT PROBABILITY
    # ========================================================

    print(
        f"\n--- Mean exit probability after "
        f"{model_year_days:.1f} model days ---"
    )


    mean_probs_365 = {}


    for side_number, label in enumerate(
        exit_labels
    ):

        value = float(
            np.mean(
                exit_365[
                    reliable,
                    side_number
                ]
            )
        )


        mean_probs_365[
            label
        ] = value


        print(
            f"{label}: "
            f"{value:.1%}"
        )


    # ========================================================
    # STRONGEST FEEDER CELLS
    #
    # Only use reliable cells with enough eventual fate
    # resolved into W/E/S/N.
    # ========================================================

    print(
        "\n--- Strongest feeder cells by exit side ---"
    )


    good_idx = np.where(
        good_resolved
    )[0]


    for side_number, label in enumerate(
        exit_labels
    ):

        print(
            f"\n{label} exit:"
        )


        if len(
            good_idx
        ) == 0:

            print(
                "No sufficiently resolved reliable cells."
            )

            continue


        order = good_idx[
            np.argsort(
                exit_prob[
                    good_idx,
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


    # ========================================================
    # SAVE PER-CELL CSV
    # ========================================================

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
            "reliable",
            "resolved_probability",
            "unresolved_probability",
            "resolved_probability_1yr",
            "unresolved_probability_1yr",
            "dominant_exit"
        ]


        for label in exit_labels:

            header.append(
                f"exit_{label}_prob"
            )


        for label in exit_labels:

            header.append(
                f"exit_{label}_prob_1yr"
            )


        writer.writerow(
            header
        )


        for k in range(
            n_cells
        ):

            # Do not assign a fake dominant W label
            # to unsupported / unresolved rows.

            if (
                reliable[k]
                and resolved[k]
                >= MIN_RESOLVED
            ):

                dominant_name = (
                    exit_labels[
                        dominant[k]
                    ]
                )

            else:

                dominant_name = ""


            row = [
                center_lon[k],
                center_lat[k],
                int(
                    row_obs[k]
                ),
                int(
                    row_drifters[k]
                ),
                bool(
                    flagged[k]
                ),
                bool(
                    empty[k]
                ),
                bool(
                    reliable[k]
                ),
                resolved[k],
                unresolved[k],
                resolved_365[k],
                unresolved_365[k],
                dominant_name
            ]


            row.extend(
                exit_prob[
                    k,
                    :
                ]
            )


            row.extend(
                exit_365[
                    k,
                    :
                ]
            )


            writer.writerow(
                row
            )


    # ========================================================
    # STORE SUMMARY
    # ========================================================

    results[
        res
    ] = {

        "reliable":
            n_reliable,

        "good_resolved":
            n_good_resolved,

        "median_resolved":
            float(
                np.median(
                    reliable_resolved
                )
            ),

        "median_resolved_365":
            float(
                np.median(
                    reliable_resolved_365
                )
            ),

        "mean_probs":
            mean_probs,

        "mean_probs_365":
            mean_probs_365,

        "side_counts":
            side_counts,

        "iterations":
            iterations,

        "final_change":
            last_change,

        "converged":
            converged,

        "model_year_days":
            model_year_days
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

            pct = (
                r[
                    "side_counts"
                ][
                    label
                ][
                    1
                ]
            )

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
            "median_eventual_resolved_probability",
            "median_1yr_resolved_probability",
            "mean_eventual_W_probability",
            "mean_eventual_E_probability",
            "mean_eventual_S_probability",
            "mean_eventual_N_probability",
            "mean_1yr_W_probability",
            "mean_1yr_E_probability",
            "mean_1yr_S_probability",
            "mean_1yr_N_probability",
            "one_year_model_days",
            "iterations",
            "final_max_change",
            "converged"
        ]
    )


    for res in RESOLUTIONS:

        r = results[
            res
        ]


        writer.writerow(
            [
                res,
                r[
                    "reliable"
                ],
                r[
                    "good_resolved"
                ],
                r[
                    "median_resolved"
                ],
                r[
                    "median_resolved_365"
                ],
                r[
                    "mean_probs"
                ].get(
                    "W",
                    0
                ),
                r[
                    "mean_probs"
                ].get(
                    "E",
                    0
                ),
                r[
                    "mean_probs"
                ].get(
                    "S",
                    0
                ),
                r[
                    "mean_probs"
                ].get(
                    "N",
                    0
                ),
                r[
                    "mean_probs_365"
                ].get(
                    "W",
                    0
                ),
                r[
                    "mean_probs_365"
                ].get(
                    "E",
                    0
                ),
                r[
                    "mean_probs_365"
                ].get(
                    "S",
                    0
                ),
                r[
                    "mean_probs_365"
                ].get(
                    "N",
                    0
                ),
                r[
                    "model_year_days"
                ],
                r[
                    "iterations"
                ],
                r[
                    "final_change"
                ],
                r[
                    "converged"
                ]
            ]
        )


# ============================================================
# 1 DEGREE DOMINANT EXIT MAP
# ============================================================

res = 1.0


if res not in map_data:

    print(
        "\nWARNING: 1 degree matrix was not analysed."
    )

else:

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


    n_lon = (
        len(
            lon_edges
        )
        - 1
    )


    n_lat = (
        len(
            lat_edges
        )
        - 1
    )


    grid = np.full(
        n_lon
        * n_lat,
        np.nan
    )


    map_cells = (
        reliable
        & (
            resolved
            >= MIN_RESOLVED
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
            figsize=(
                10,
                7
            )
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
            shading="auto",
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
                    color=cmap(
                        number
                    ),
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
            figsize=(
                10,
                7
            )
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
            shading="auto"
        )


        ax.set_xlabel(
            "Longitude"
        )


        ax.set_ylabel(
            "Latitude"
        )


        ax.set_xlim(
            BOX[0],
            BOX[1]
        )


        ax.set_ylim(
            BOX[2],
            BOX[3]
        )


    ax.set_title(
        "Dominant Eventual Exit Side — 1° Drogued Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# ============================================================
# DONE
# ============================================================

print(
    "\nSaved:"
)

print(
    summary_file
)


if 1.0 in map_data:

    print(
        FIGURE_DIR
        / "dominant_exit_zone_1deg.png"
    )


print(
    "\nDONE."
)
