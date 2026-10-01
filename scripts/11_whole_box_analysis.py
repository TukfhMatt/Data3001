from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt


# ============================================================
# SQ2 WHOLE-BOX ANALYSIS
#
# Part A:
# Check outgoing transition support
# across 0.5, 1 and 2 degree grids
#
# Region: 10E–55E, 45S–15S
# Drogued only
# Tau = 3.5 days
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

FIGURE_DIR = PROJECT_DIR / "figures"


DATA_DIR.mkdir(
    exist_ok=True
)

FIGURE_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# MATRIX FILES
# ============================================================

matrix_files = {

    0.5:
        DATA_DIR
        / "P_drogued_0p5deg_3p5d_10E-55E_45S-15S.npz",

    1.0:
        DATA_DIR
        / "P_drogued_1deg_3p5d_10E-55E_45S-15S.npz",

    2.0:
        DATA_DIR
        / "P_drogued_2deg_3p5d_10E-55E_45S-15S.npz"
}


# ============================================================
# START
# ============================================================

print(
    "========================================"
)

print(
    "SQ2 WHOLE-BOX SUPPORT ANALYSIS"
)

print(
    "========================================"
)

print(
    "\nRegion: 10E–55E, 45S–15S"
)

print(
    "Dataset: drogued drifters"
)

print(
    "Lag: 3.5 days"
)


# ============================================================
# STORAGE
# ============================================================

results = {}


# ============================================================
# LOOP THROUGH GRID SIZES
# ============================================================

for grid_size, matrix_file in matrix_files.items():

    print(
        "\n\n========================================"
    )

    print(
        f"GRID SIZE: {grid_size}°"
    )

    print(
        "========================================"
    )


    print(
        "\nLoading:"
    )

    print(
        matrix_file
    )


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

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
        "empty"
    ]


    for name in needed:

        if name not in data.files:

            print(
                f"\nERROR: {name} is missing."
            )

            raise SystemExit


    # --------------------------------------------------------
    # LOAD ROW INFORMATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ACTIVE CELLS
    # --------------------------------------------------------

    active = (
        row_obs > 0
    )


    active_obs = row_obs[
        active
    ]

    active_drifters = row_drifters[
        active
    ]


    n_total = len(
        row_obs
    )


    n_active = int(
        np.count_nonzero(
            active
        )
    )


    n_empty = int(
        np.count_nonzero(
            empty
        )
    )


    # --------------------------------------------------------
    # TRANSITION SUPPORT
    # --------------------------------------------------------

    n_30 = int(
        np.count_nonzero(
            active_obs >= 30
        )
    )


    n_50 = int(
        np.count_nonzero(
            active_obs >= 50
        )
    )


    n_100 = int(
        np.count_nonzero(
            active_obs >= 100
        )
    )


    # --------------------------------------------------------
    # DISTINCT DRIFTER SUPPORT
    # --------------------------------------------------------

    n_d10 = int(
        np.count_nonzero(
            active_drifters >= 10
        )
    )


    n_d20 = int(
        np.count_nonzero(
            active_drifters >= 20
        )
    )


    n_d30 = int(
        np.count_nonzero(
            active_drifters >= 30
        )
    )


    # --------------------------------------------------------
    # FLAGGED ROWS
    # --------------------------------------------------------

    n_flagged_total = int(
        np.count_nonzero(
            flagged
        )
    )


    flagged_active = (
        flagged
        & active
    )


    n_flagged_active = int(
        np.count_nonzero(
            flagged_active
        )
    )


    # --------------------------------------------------------
    # SUMMARY STATS
    # --------------------------------------------------------

    median_obs = np.median(
        active_obs
    )


    q25_obs = np.percentile(
        active_obs,
        25
    )


    q75_obs = np.percentile(
        active_obs,
        75
    )


    median_drifters = np.median(
        active_drifters
    )


    q25_drifters = np.percentile(
        active_drifters,
        25
    )


    q75_drifters = np.percentile(
        active_drifters,
        75
    )


    total_transitions = int(
        np.sum(
            row_obs
        )
    )


    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------

    pct_30 = (
        n_30
        / n_active
        * 100
    )


    pct_50 = (
        n_50
        / n_active
        * 100
    )


    pct_100 = (
        n_100
        / n_active
        * 100
    )


    pct_d10 = (
        n_d10
        / n_active
        * 100
    )


    pct_d20 = (
        n_d20
        / n_active
        * 100
    )


    pct_d30 = (
        n_d30
        / n_active
        * 100
    )


    pct_flagged = (
        n_flagged_active
        / n_active
        * 100
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results[
        grid_size
    ] = {

        "total_cells": n_total,

        "active_cells": n_active,

        "empty_cells": n_empty,

        "total_transitions": total_transitions,

        "median_obs": median_obs,

        "q25_obs": q25_obs,

        "q75_obs": q75_obs,

        "n30": n_30,

        "pct30": pct_30,

        "n50": n_50,

        "pct50": pct_50,

        "n100": n_100,

        "pct100": pct_100,

        "median_drifters": median_drifters,

        "q25_drifters": q25_drifters,

        "q75_drifters": q75_drifters,

        "d10": n_d10,

        "pct_d10": pct_d10,

        "d20": n_d20,

        "pct_d20": pct_d20,

        "d30": n_d30,

        "pct_d30": pct_d30,

        "flagged_total": n_flagged_total,

        "flagged_active": n_flagged_active,

        "pct_flagged": pct_flagged
    }


    # --------------------------------------------------------
    # PRINT THIS GRID
    # --------------------------------------------------------

    print(
        f"\nOcean cells: "
        f"{n_total:,}"
    )


    print(
        f"Active origin cells: "
        f"{n_active:,}"
    )


    print(
        f"Empty cells: "
        f"{n_empty:,}"
    )


    print(
        f"Total outgoing transitions: "
        f"{total_transitions:,}"
    )


    print(
        "\n--- Transition support ---"
    )


    print(
        f"Median transitions / active cell: "
        f"{median_obs:,.0f}"
    )


    print(
        f"25th percentile: "
        f"{q25_obs:,.0f}"
    )


    print(
        f"75th percentile: "
        f"{q75_obs:,.0f}"
    )


    print(
        f"Cells >=30 transitions: "
        f"{n_30:,} "
        f"({pct_30:.1f}%)"
    )


    print(
        f"Cells >=50 transitions: "
        f"{n_50:,} "
        f"({pct_50:.1f}%)"
    )


    print(
        f"Cells >=100 transitions: "
        f"{n_100:,} "
        f"({pct_100:.1f}%)"
    )


    print(
        "\n--- Distinct drifter support ---"
    )


    print(
        f"Median drifters / active cell: "
        f"{median_drifters:.0f}"
    )


    print(
        f"25th percentile: "
        f"{q25_drifters:.0f}"
    )


    print(
        f"75th percentile: "
        f"{q75_drifters:.0f}"
    )


    print(
        f"Cells >=10 drifters: "
        f"{n_d10:,} "
        f"({pct_d10:.1f}%)"
    )


    print(
        f"Cells >=20 drifters: "
        f"{n_d20:,} "
        f"({pct_d20:.1f}%)"
    )


    print(
        f"Cells >=30 drifters: "
        f"{n_d30:,} "
        f"({pct_d30:.1f}%)"
    )


    print(
        "\n--- Reliability ---"
    )


    print(
        f"Flagged active rows "
        f"(<10 drifters): "
        f"{n_flagged_active:,} "
        f"({pct_flagged:.1f}%)"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n\n========================================"
)

print(
    "THREE-GRID COMPARISON"
)

print(
    "========================================"
)


print(
    "\nGrid | Active | Median trans | >=50 trans | "
    "Median drifters | >=10 drifters | Flagged"
)


for grid_size in matrix_files:

    r = results[
        grid_size
    ]


    print(
        f"{grid_size:>4}° | "
        f"{r['active_cells']:>6} | "
        f"{r['median_obs']:>12,.0f} | "
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
    / "whole_box_grid_support.csv"
)


with open(
    csv_file,
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
            "total_cells",
            "active_cells",
            "empty_cells",
            "total_transitions",
            "median_transitions",
            "q25_transitions",
            "q75_transitions",
            "pct_cells_ge30_transitions",
            "pct_cells_ge50_transitions",
            "pct_cells_ge100_transitions",
            "median_distinct_drifters",
            "q25_distinct_drifters",
            "q75_distinct_drifters",
            "pct_cells_ge10_drifters",
            "pct_cells_ge20_drifters",
            "pct_cells_ge30_drifters",
            "flagged_active_rows",
            "pct_flagged_active_rows"
        ]
    )


    for grid_size in matrix_files:

        r = results[
            grid_size
        ]


        writer.writerow(
            [
                grid_size,
                r["total_cells"],
                r["active_cells"],
                r["empty_cells"],
                r["total_transitions"],
                r["median_obs"],
                r["q25_obs"],
                r["q75_obs"],
                r["pct30"],
                r["pct50"],
                r["pct100"],
                r["median_drifters"],
                r["q25_drifters"],
                r["q75_drifters"],
                r["pct_d10"],
                r["pct_d20"],
                r["pct_d30"],
                r["flagged_active"],
                r["pct_flagged"]
            ]
        )


# ============================================================
# FIGURE 1
# TRANSITION SUPPORT
# ============================================================

grid_labels = [
    "0.5°",
    "1°",
    "2°"
]


transition_support = [
    results[0.5]["pct50"],
    results[1.0]["pct50"],
    results[2.0]["pct50"]
]


plot1 = (
    FIGURE_DIR
    / "whole_box_transition_support.png"
)


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    grid_labels,
    transition_support
)


plt.ylabel(
    "Active cells with >=50 transitions (%)"
)

plt.xlabel(
    "Grid resolution"
)

plt.title(
    "SQ2 — Outgoing Transition Support"
)

plt.ylim(
    0,
    105
)

plt.tight_layout()


plt.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# DISTINCT DRIFTER SUPPORT
# ============================================================

drifter_support = [
    results[0.5]["pct_d10"],
    results[1.0]["pct_d10"],
    results[2.0]["pct_d10"]
]


plot2 = (
    FIGURE_DIR
    / "whole_box_drifter_support.png"
)


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    grid_labels,
    drifter_support
)


plt.ylabel(
    "Active cells with >=10 distinct drifters (%)"
)

plt.xlabel(
    "Grid resolution"
)

plt.title(
    "SQ2 — Independent Drifter Support"
)

plt.ylim(
    0,
    105
)

plt.tight_layout()


plt.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print(
    "\n\nSaved:"
)

print(
    csv_file
)

print(
    plot1
)

print(
    plot2
)


print(
    "\nDONE."
)
