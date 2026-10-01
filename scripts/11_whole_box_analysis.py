from pathlib import Path
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# MATRIX FILE
#
# README example:
# P_drogued_1deg_3p5d_10E-55E_45S-15S.npz
# ============================================================

matrix_file = (
    DATA_DIR
    / "P_drogued_1deg_3p5d_10E-55E_45S-15S.npz"
)


print(
    "========================================"
)

print(
    "SQ2 WHOLE-BOX ANALYSIS"
)

print(
    "========================================"
)


print(
    "\nLooking for:"
)

print(
    matrix_file
)


# ============================================================
# CHECK FILE EXISTS
# ============================================================

if not matrix_file.exists():

    print(
        "\nERROR: matrix file not found."
    )

    print(
        "\nFiles currently in data/:"
    )

    for file in DATA_DIR.glob(
        "*.npz"
    ):

        print(
            file.name
        )

    raise SystemExit


# ============================================================
# LOAD MATRIX
# ============================================================

print(
    "\nLoading matrix..."
)


data = np.load(
    matrix_file,
    allow_pickle=True
)


print(
    "\nMatrix loaded successfully."
)


# ============================================================
# SEE WHAT IS INSIDE
# ============================================================

print(
    "\nAvailable fields:"
)

for name in data.files:

    print(
        "-",
        name
    )


# ============================================================
# REQUIRED FIELDS
# ============================================================

needed = [
    "P",
    "row_obs",
    "row_drifters",
    "flagged"
]


print(
    "\nChecking required fields..."
)


for name in needed:

    if name in data.files:

        print(
            f"{name}: OK"
        )

    else:

        print(
            f"{name}: MISSING"
        )


# ============================================================
# STOP IF IMPORTANT FIELDS ARE MISSING
# ============================================================

if (
    "row_obs" not in data.files
    or "row_drifters" not in data.files
):

    print(
        "\nCannot continue because "
        "row_obs or row_drifters is missing."
    )

    raise SystemExit


# ============================================================
# LOAD ROW SUPPORT
# ============================================================

row_obs = data[
    "row_obs"
]

row_drifters = data[
    "row_drifters"
]


print(
    "\nrow_obs shape:"
)

print(
    row_obs.shape
)


print(
    "\nrow_drifters shape:"
)

print(
    row_drifters.shape
)


# ============================================================
# ACTIVE CELLS
# ============================================================

active = (
    row_obs > 0
)


active_obs = row_obs[
    active
]

active_drifters = row_drifters[
    active
]


n_active = int(
    np.count_nonzero(
        active
    )
)


# ============================================================
# TRANSITION SUPPORT
# ============================================================

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


# ============================================================
# DISTINCT DRIFTER SUPPORT
# ============================================================

n_drifter_10 = int(
    np.count_nonzero(
        active_drifters >= 10
    )
)

n_drifter_20 = int(
    np.count_nonzero(
        active_drifters >= 20
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n\n========================================"
)

print(
    "1 DEGREE DROGUED MATRIX RESULTS"
)

print(
    "========================================"
)


print(
    f"\nActive origin cells: "
    f"{n_active:,}"
)


print(
    f"Median outgoing transitions: "
    f"{np.median(active_obs):,.0f}"
)


print(
    f"Median distinct drifters per row: "
    f"{np.median(active_drifters):,.0f}"
)


print(
    "\n--- Transition counts ---"
)


print(
    f"Cells >=30 transitions: "
    f"{n_30:,} "
    f"({n_30 / n_active * 100:.1f}%)"
)


print(
    f"Cells >=50 transitions: "
    f"{n_50:,} "
    f"({n_50 / n_active * 100:.1f}%)"
)


print(
    f"Cells >=100 transitions: "
    f"{n_100:,} "
    f"({n_100 / n_active * 100:.1f}%)"
)


print(
    "\n--- Distinct drifter support ---"
)


print(
    f"Cells >=10 drifters: "
    f"{n_drifter_10:,} "
    f"({n_drifter_10 / n_active * 100:.1f}%)"
)


print(
    f"Cells >=20 drifters: "
    f"{n_drifter_20:,} "
    f"({n_drifter_20 / n_active * 100:.1f}%)"
)


# ============================================================
# FLAGGED ROWS
# ============================================================

if "flagged" in data.files:

    flagged = data[
        "flagged"
    ]


    n_flagged = int(
        np.count_nonzero(
            flagged
        )
    )


    print(
        "\n--- Reliability flags ---"
    )


    print(
        f"Flagged rows: "
        f"{n_flagged:,}"
    )


print(
    "\nDONE."
)
