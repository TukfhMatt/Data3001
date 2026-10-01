from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import (
    BOX,
    TAU_DAYS,
    operator_path
)


# ============================================================
# ALMOST-INVARIANT REGIONS
#
# Baseline:
#   drogued
#   1 degree
#   tau = 3.5 days
#
# Method:
#   1. Keep reliable active ocean cells
#   2. Build symmetric transport affinity
#   3. Use spectral embedding
#   4. Cluster cells into candidate regions
#   5. Test each region using the original directional P
#
# Output:
#   - region summary CSV
#   - cell labels CSV
#   - map of candidate regions
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


RES = 1.0

N_REGIONS = 4

RANDOM_SEED = 3001

KMEANS_ITER = 100


# ============================================================
# START
# ============================================================

print("========================================")
print("ALMOST-INVARIANT REGIONS")
print("========================================")

print("\nRegion:", BOX)
print("Grid:", RES, "degree")
print("Lag:", TAU_DAYS, "days")
print("Candidate regions:", N_REGIONS)


# ============================================================
# SCIPY
# ============================================================

try:

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

except ImportError:

    print("\nERROR: scipy is needed.")
    print("Run:")
    print("pip install scipy")

    raise SystemExit


# ============================================================
# LOAD MATRIX
# ============================================================

matrix_name = operator_path(
    "drogued",
    RES,
    TAU_DAYS,
    BOX
)


matrix_file = (
    PROJECT_DIR
    / matrix_name
)


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

        print(
            f"\nERROR: {name} missing."
        )

        raise SystemExit


# ============================================================
# LOAD VARIABLES
# ============================================================

P = data["P"]

row_obs = data["row_obs"]

row_drifters = data["row_drifters"]

flagged = data["flagged"].astype(bool)

empty = data["empty"].astype(bool)

cells = data["cell_flat"].astype(int)

lon_edges = data["lon_edges"]

lat_edges = data["lat_edges"]


n_cells = len(cells)

n_states = P.shape[0]


print(
    f"\nOcean cells: {n_cells:,}"
)

print(
    f"Total states: {n_states:,}"
)


# ============================================================
# RELIABLE CELLS
# ============================================================

active = (
    row_obs > 0
)


reliable = (
    active
    & ~flagged
    & ~empty
)


reliable_idx = np.where(
    reliable
)[0]


n_reliable = len(
    reliable_idx
)


print(
    f"Reliable active cells: {n_reliable:,}"
)


if n_reliable <= N_REGIONS:

    print(
        "\nERROR: not enough reliable cells."
    )

    raise SystemExit


# ============================================================
# CELL CENTRES
# ============================================================

n_lon = (
    len(lon_edges)
    - 1
)


n_lat = (
    len(lat_edges)
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


# ============================================================
# OCEAN TRANSITION MATRIX
# ============================================================

Q = P[
    :n_cells,
    :n_cells
]


Q_reliable = Q[
    np.ix_(
        reliable_idx,
        reliable_idx
    )
]


# ============================================================
# TRANSPORT AFFINITY
#
# P is directional.
#
# For spectral clustering, make a symmetric matrix:
#
# A(i,j) = 0.5 * (Q(i,j) + Q(j,i))
#
# This matrix is only used to FIND candidate regions.
#
# Later we use the original directional P to test whether
# each region really retains material.
# ============================================================

print(
    "\nBuilding transport affinity matrix..."
)


A = (
    Q_reliable
    + Q_reliable.T
) / 2


# Remove self-transition for clustering

np.fill_diagonal(
    A,
    0
)


degree = np.sum(
    A,
    axis=1
)


zero_degree = (
    degree <= 0
)


if np.any(
    zero_degree
):

    print(
        "WARNING:",
        np.count_nonzero(zero_degree),
        "cells have zero affinity."
    )


safe_degree = np.where(
    degree > 0,
    degree,
    1
)


d_inv_sqrt = (
    1
    / np.sqrt(
        safe_degree
    )
)


# Normalised symmetric matrix:
#
# S = D^(-1/2) A D^(-1/2)

S = (
    d_inv_sqrt[:, None]
    * A
    * d_inv_sqrt[None, :]
)


S_sparse = csr_matrix(
    S
)


# ============================================================
# SPECTRAL EMBEDDING
# ============================================================

print(
    "\nCalculating leading eigenvectors..."
)


eigenvalues, eigenvectors = eigsh(
    S_sparse,
    k=N_REGIONS,
    which="LA"
)


# Sort biggest to smallest

eigen_order = np.argsort(
    eigenvalues
)[::-1]


eigenvalues = eigenvalues[
    eigen_order
]


eigenvectors = eigenvectors[
    :,
    eigen_order
]


print(
    "\nLeading eigenvalues:"
)


for number, value in enumerate(
    eigenvalues,
    start=1
):

    print(
        f"{number}: {value:.6f}"
    )


# ============================================================
# NORMALISE EMBEDDING
# ============================================================

row_norm = np.linalg.norm(
    eigenvectors,
    axis=1
)


row_norm = np.where(
    row_norm > 0,
    row_norm,
    1
)


embedding = (
    eigenvectors
    / row_norm[:, None]
)


# ============================================================
# SIMPLE K-MEANS
# ============================================================

print(
    "\nClustering spectral coordinates..."
)


rng = np.random.default_rng(
    RANDOM_SEED
)


start_points = rng.choice(
    n_reliable,
    size=N_REGIONS,
    replace=False
)


centres = embedding[
    start_points
].copy()


labels = np.full(
    n_reliable,
    -1,
    dtype=int
)


for iteration in range(
    KMEANS_ITER
):

    dist = np.zeros(
        (
            n_reliable,
            N_REGIONS
        )
    )


    for group in range(
        N_REGIONS
    ):

        diff = (
            embedding
            - centres[group]
        )


        dist[
            :,
            group
        ] = np.sum(
            diff * diff,
            axis=1
        )


    new_labels = np.argmin(
        dist,
        axis=1
    )


    if np.array_equal(
        new_labels,
        labels
    ):

        print(
            f"K-means converged after "
            f"{iteration} iterations."
        )

        break


    labels = new_labels


    new_centres = centres.copy()


    for group in range(
        N_REGIONS
    ):

        members = (
            labels == group
        )


        if np.any(
            members
        ):

            new_centres[
                group
            ] = np.mean(
                embedding[
                    members
                ],
                axis=0
            )

        else:

            random_cell = rng.integers(
                0,
                n_reliable
            )


            new_centres[
                group
            ] = embedding[
                random_cell
            ]


    centres = new_centres


else:

    print(
        "K-means reached maximum iterations."
    )


# ============================================================
# RENAME REGIONS WEST -> EAST
#
# Cluster labels are arbitrary.
#
# Renumber them by their mean longitude so Region 1 is
# generally the western-most region.
# ============================================================

old_mean_lon = []


for group in range(
    N_REGIONS
):

    members = reliable_idx[
        labels == group
    ]


    if len(
        members
    ) > 0:

        value = np.mean(
            center_lon[
                members
            ]
        )

    else:

        value = np.inf


    old_mean_lon.append(
        value
    )


west_to_east = np.argsort(
    old_mean_lon
)


rename = {}


for new_group, old_group in enumerate(
    west_to_east
):

    rename[
        old_group
    ] = new_group


labels = np.array(
    [
        rename[x]
        for x in labels
    ]
)


# ============================================================
# FULL REGION LABEL ARRAY
# ============================================================

region_label = np.full(
    n_cells,
    -1,
    dtype=int
)


region_label[
    reliable_idx
] = labels


# ============================================================
# VALIDATE REGIONS
#
# Use ORIGINAL directional P.
#
# Check:
#   one step
#   30 days
#   365 days
#
# For 30d / 365d, start uniformly across the cells
# belonging to each candidate region.
# ============================================================

steps_30 = int(
    round(
        30
        / TAU_DAYS
    )
)


steps_365 = int(
    round(
        365
        / TAU_DAYS
    )
)


print(
    "\nCalculating 30-day matrix..."
)


P30 = np.linalg.matrix_power(
    P,
    steps_30
)


print(
    "Calculating 365-day matrix..."
)


P365 = np.linalg.matrix_power(
    P,
    steps_365
)


print(
    "\n========================================"
)

print(
    "REGION RETENTION RESULTS"
)

print(
    "========================================"
)


print(
    "\nRegion | Cells | Mean lon | Mean lat | "
    "Drifters | 1-step | 30-day | 365-day"
)


region_results = {}


# ============================================================
# ANALYSE EACH REGION
# ============================================================

for group in range(
    N_REGIONS
):

    member_idx = np.where(
        region_label == group
    )[0]


    n_member = len(
        member_idx
    )


    if n_member == 0:

        print(
            f"\nWARNING: Region {group + 1} is empty."
        )

        continue


    # --------------------------------------------------------
    # ONE-STEP INTERNAL RETENTION
    # --------------------------------------------------------

    region_block = P[
        np.ix_(
            member_idx,
            member_idx
        )
    ]


    one_step_by_cell = np.sum(
        region_block,
        axis=1
    )


    one_step = np.mean(
        one_step_by_cell
    )


    # --------------------------------------------------------
    # START UNIFORMLY OVER REGION
    # --------------------------------------------------------

    p0 = np.zeros(
        n_states
    )


    p0[
        member_idx
    ] = (
        1
        / n_member
    )


    # --------------------------------------------------------
    # 30 DAYS
    # --------------------------------------------------------

    p30 = (
        p0
        @ P30
    )


    stay_30 = np.sum(
        p30[
            member_idx
        ]
    )


    inside_box_30 = np.sum(
        p30[
            :n_cells
        ]
    )


    # --------------------------------------------------------
    # 365 DAYS
    # --------------------------------------------------------

    p365 = (
        p0
        @ P365
    )


    stay_365 = np.sum(
        p365[
            member_idx
        ]
    )


    inside_box_365 = np.sum(
        p365[
            :n_cells
        ]
    )


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    mean_lon = np.mean(
        center_lon[
            member_idx
        ]
    )


    mean_lat = np.mean(
        center_lat[
            member_idx
        ]
    )


    # --------------------------------------------------------
    # DATA SUPPORT
    # --------------------------------------------------------

    mean_drifters = np.mean(
        row_drifters[
            member_idx
        ]
    )


    median_drifters = np.median(
        row_drifters[
            member_idx
        ]
    )


    mean_transitions = np.mean(
        row_obs[
            member_idx
        ]
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    region_results[
        group
    ] = {

        "cells":
            n_member,

        "mean_lon":
            mean_lon,

        "mean_lat":
            mean_lat,

        "mean_drifters":
            mean_drifters,

        "median_drifters":
            median_drifters,

        "mean_transitions":
            mean_transitions,

        "one_step":
            one_step,

        "stay30":
            stay_30,

        "inside30":
            inside_box_30,

        "stay365":
            stay_365,

        "inside365":
            inside_box_365
    }


    print(
        f"{group + 1:>6} | "
        f"{n_member:>5} | "
        f"{mean_lon:>8.2f} | "
        f"{mean_lat:>8.2f} | "
        f"{median_drifters:>8.0f} | "
        f"{one_step:>6.1%} | "
        f"{stay_30:>6.1%} | "
        f"{stay_365:>7.1%}"
    )


# ============================================================
# REGION DETAILS
# ============================================================

print(
    "\n========================================"
)

print(
    "REGION DETAILS"
)

print(
    "========================================"
)


for group in range(
    N_REGIONS
):

    if group not in region_results:

        continue


    r = region_results[
        group
    ]


    print(
        f"\nRegion {group + 1}"
    )


    print(
        f"Cells: "
        f"{r['cells']}"
    )


    print(
        f"Mean centre: "
        f"{r['mean_lon']:.2f}°E, "
        f"{abs(r['mean_lat']):.2f}°S"
    )


    print(
        f"Mean transitions/cell: "
        f"{r['mean_transitions']:.0f}"
    )


    print(
        f"Median distinct drifters/cell: "
        f"{r['median_drifters']:.0f}"
    )


    print(
        f"One-step internal retention: "
        f"{r['one_step']:.1%}"
    )


    print(
        f"30-day probability still in same region: "
        f"{r['stay30']:.1%}"
    )


    print(
        f"30-day probability still anywhere in R: "
        f"{r['inside30']:.1%}"
    )


    print(
        f"365-day probability still in same region: "
        f"{r['stay365']:.1%}"
    )


    print(
        f"365-day probability still anywhere in R: "
        f"{r['inside365']:.1%}"
    )


# ============================================================
# SAVE REGION SUMMARY
# ============================================================

summary_file = (
    DATA_DIR
    / "almost_invariant_region_summary.csv"
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
            "region",
            "cells",
            "mean_lon",
            "mean_lat",
            "mean_transitions",
            "mean_distinct_drifters",
            "median_distinct_drifters",
            "one_step_internal_retention",
            "30d_same_region",
            "30d_inside_box",
            "365d_same_region",
            "365d_inside_box"
        ]
    )


    for group in range(
        N_REGIONS
    ):

        if group not in region_results:

            continue


        r = region_results[
            group
        ]


        writer.writerow(
            [
                group + 1,
                r["cells"],
                r["mean_lon"],
                r["mean_lat"],
                r["mean_transitions"],
                r["mean_drifters"],
                r["median_drifters"],
                r["one_step"],
                r["stay30"],
                r["inside30"],
                r["stay365"],
                r["inside365"]
            ]
        )


# ============================================================
# SAVE CELL LABELS
# ============================================================

cell_file = (
    DATA_DIR
    / "almost_invariant_regions_1deg.csv"
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
            "region",
            "row_transitions",
            "distinct_drifters",
            "flagged",
            "empty",
            "reliable"
        ]
    )


    for k in range(
        n_cells
    ):

        if region_label[k] >= 0:

            region_number = (
                region_label[k]
                + 1
            )

        else:

            region_number = ""


        writer.writerow(
            [
                center_lon[k],
                center_lat[k],
                region_number,
                int(row_obs[k]),
                int(row_drifters[k]),
                bool(flagged[k]),
                bool(empty[k]),
                bool(reliable[k])
            ]
        )


# ============================================================
# MAP GRID
# ============================================================

grid = np.full(
    n_lon * n_lat,
    np.nan
)


valid = (
    region_label >= 0
)


grid[
    cells[
        valid
    ]
] = (
    region_label[
        valid
    ]
    + 1
)


grid = grid.reshape(
    n_lat,
    n_lon
)


plot_file = (
    FIGURE_DIR
    / "almost_invariant_regions_1deg.png"
)


# ============================================================
# MAP
# ============================================================

try:

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature


    fig = plt.figure(
        figsize=(10, 7)
    )


    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )


    cmap = plt.get_cmap(
        "tab10",
        N_REGIONS
    )


    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid,
        cmap=cmap,
        vmin=0.5,
        vmax=N_REGIONS + 0.5,
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


    cmap = plt.get_cmap(
        "tab10",
        N_REGIONS
    )


    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid,
        cmap=cmap,
        vmin=0.5,
        vmax=N_REGIONS + 0.5
    )


    ax.set_xlabel(
        "Longitude"
    )


    ax.set_ylabel(
        "Latitude"
    )


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(
    mesh,
    ax=ax,
    ticks=np.arange(
        1,
        N_REGIONS + 1
    )
)


cbar.set_label(
    "Candidate almost-invariant region"
)


ax.set_title(
    "Candidate Almost-Invariant Regions — 1° Drogued Matrix"
)


plt.tight_layout()


plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# DONE
# ============================================================

print(
    "\nSaved:"
)

print(
    summary_file
)

print(
    cell_file
)

print(
    plot_file
)


print(
    "\nDONE."
)
