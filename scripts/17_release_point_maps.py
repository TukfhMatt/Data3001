from pathlib import Path
import re
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import PowerNorm

from config import (
    BOX,
    TAU_DAYS,
    operator_path
)

from transport import (
    load_operator,
    release,
    to_grid
)


# ============================================================
# RELEASE-POINT MAPS
#
# Make transport maps from the candidate release points
# selected by 04_shipping_overlay.py.
#
# Baseline:
#   grid = 1 degree
#   tau = 3.5 days
#
# Times:
#   7 days
#   30 days
#   365 days
#
# Compare:
#   drogued
#   undrogued
#
# Each candidate gets one figure:
#
#          7 d       30 d       365 d
# drogued
# undrogued
#
# ============================================================


# ============================================================
# SETTINGS
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


FIGURE_DIR.mkdir(
    exist_ok=True
)


RES = 1.0


DAYS_LIST = [
    7,
    30,
    365
]


TYPES = [
    "drogued",
    "undrogued"
]


candidate_file = (
    DATA_DIR
    / "release_candidates.csv"
)


summary_file = (
    DATA_DIR
    / "release_point_map_summary.csv"
)


# ============================================================
# START
# ============================================================

print(
    "========================================"
)

print(
    "RELEASE-POINT MAPS"
)

print(
    "========================================"
)


print(
    "\nRegion:",
    BOX
)


print(
    "Grid:",
    RES,
    "degree"
)


print(
    "Lag:",
    TAU_DAYS,
    "days"
)


print(
    "Requested times:",
    DAYS_LIST
)


# ============================================================
# CHECK CANDIDATE FILE
# ============================================================

if not candidate_file.exists():

    print(
        "\nERROR:"
    )

    print(
        candidate_file
    )

    print(
        "was not found."
    )

    print(
        "\nRun 04_shipping_overlay.py first."
    )

    raise SystemExit


# ============================================================
# READ RELEASE POINTS
# ============================================================

points = pd.read_csv(
    candidate_file
)


print(
    "\nRelease candidate file:"
)

print(
    candidate_file
)


print(
    "\nColumns found:"
)

print(
    list(points.columns)
)


if len(points) == 0:

    print(
        "\nERROR: release_candidates.csv is empty."
    )

    raise SystemExit


# ============================================================
# FIND COLUMN NAMES
#
# Slightly flexible because old versions of script 04 may use
# different names.
# ============================================================

def find_col(options):

    lower_names = {
        str(x).lower(): x
        for x in points.columns
    }


    for name in options:

        if name.lower() in lower_names:

            return lower_names[
                name.lower()
            ]


    return None


lon_col = find_col(
    [
        "lon",
        "longitude",
        "release_lon",
        "candidate_lon",
        "cell_lon",
        "x"
    ]
)


lat_col = find_col(
    [
        "lat",
        "latitude",
        "release_lat",
        "candidate_lat",
        "cell_lat",
        "y"
    ]
)


name_col = find_col(
    [
        "name",
        "label",
        "site",
        "location",
        "candidate",
        "point",
        "release_point"
    ]
)


rank_col = find_col(
    [
        "rank",
        "overall_rank",
        "shipping_rank",
        "priority_rank"
    ]
)


# ============================================================
# CHECK COORDINATE COLUMNS
# ============================================================

if lon_col is None or lat_col is None:

    print(
        "\nERROR: could not find longitude/latitude columns."
    )


    print(
        "\nColumns available:"
    )


    for col in points.columns:

        print(
            " ",
            col
        )


    raise SystemExit


print(
    "\nUsing longitude column:",
    lon_col
)


print(
    "Using latitude column:",
    lat_col
)


if name_col is not None:

    print(
        "Using point-name column:",
        name_col
    )


if rank_col is not None:

    print(
        "Using rank column:",
        rank_col
    )


# ============================================================
# SORT BY RANK IF AVAILABLE
# ============================================================

if rank_col is not None:

    points = points.sort_values(
        rank_col
    ).reset_index(
        drop=True
    )


else:

    points = points.reset_index(
        drop=True
    )


# ============================================================
# CLEAN POINTS
# ============================================================

good_rows = []


for i, row in points.iterrows():

    try:

        lon = float(
            row[
                lon_col
            ]
        )


        lat = float(
            row[
                lat_col
            ]
        )


    except Exception:

        continue


    if not np.isfinite(
        lon
    ):

        continue


    if not np.isfinite(
        lat
    ):

        continue


    good_rows.append(
        i
    )


points = points.loc[
    good_rows
].reset_index(
    drop=True
)


if len(points) == 0:

    print(
        "\nERROR: no valid release points."
    )

    raise SystemExit


print(
    f"\nValid release points: "
    f"{len(points)}"
)


# ============================================================
# POINT NAMES
# ============================================================

def get_point_name(row, number):

    if name_col is not None:

        value = row[
            name_col
        ]


        if pd.notna(
            value
        ):

            text = str(
                value
            ).strip()


            if text != "":

                return text


    return (
        f"Point {number}"
    )


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_name(text):

    text = str(
        text
    ).strip()


    text = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        text
    )


    text = text.strip(
        "_"
    )


    if text == "":

        text = "point"


    return text.lower()


# ============================================================
# LOAD MATRICES
# ============================================================

ops = {}


for kind in TYPES:

    matrix_name = operator_path(
        kind,
        RES,
        TAU_DAYS,
        BOX
    )


    matrix_file = (
        PROJECT_DIR
        / matrix_name
    )


    print(
        f"\nLoading {kind} matrix:"
    )

    print(
        matrix_file
    )


    if not matrix_file.exists():

        print(
            "\nERROR: matrix does not exist."
        )

        raise SystemExit


    ops[
        kind
    ] = load_operator(
        matrix_file
    )


# ============================================================
# PRECALCULATE MATRIX POWERS
#
# README convention:
#
# 7 d   -> about 2 steps
# 30 d  -> about 9 steps
# 365 d -> about 104 steps
#
# Use the exact same rounding rule as the model.
# ============================================================

powers = {}


print(
    "\nCalculating matrix powers..."
)


for kind in TYPES:

    P = ops[
        kind
    ][
        "P"
    ]


    powers[
        kind
    ] = {}


    for days in DAYS_LIST:

        steps = int(
            round(
                days
                / TAU_DAYS
            )
        )


        real_days = (
            steps
            * TAU_DAYS
        )


        print(
            f"  {kind:10s} | "
            f"{days:3d} requested days | "
            f"{steps:3d} steps | "
            f"{real_days:.1f} model days"
        )


        powers[
            kind
        ][
            days
        ] = np.linalg.matrix_power(
            P,
            steps
        )


# ============================================================
# CARTOPY
# ============================================================

try:

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAVE_CARTOPY = True


except ImportError:

    HAVE_CARTOPY = False


print(
    "\nCartopy:",
    HAVE_CARTOPY
)


# ============================================================
# SUMMARY ROWS
# ============================================================

summary_rows = []


# ============================================================
# LOOP THROUGH RELEASE POINTS
# ============================================================

for point_number, row in points.iterrows():

    point_id = (
        point_number
        + 1
    )


    lon = float(
        row[
            lon_col
        ]
    )


    lat = float(
        row[
            lat_col
        ]
    )


    point_name = get_point_name(
        row,
        point_id
    )


    print(
        "\n\n========================================"
    )

    print(
        f"RELEASE POINT {point_id}: "
        f"{point_name}"
    )

    print(
        "========================================"
    )


    print(
        f"Location: "
        f"{lon:.2f}°E, "
        f"{abs(lat):.2f}°S"
    )


    # --------------------------------------------------------
    # FIRST CALCULATE ALL DISTRIBUTIONS
    # --------------------------------------------------------

    distributions = {}

    all_grid_values = []


    for kind in TYPES:

        op = ops[
            kind
        ]


        try:

            p0 = release(
                op,
                lon=lon,
                lat=lat
            )


        except Exception as e:

            print(
                f"\nERROR creating {kind} release:"
            )

            print(
                e
            )

            raise


        n_cells = len(
            op[
                "cell_flat"
            ]
        )


        exit_labels = [
            str(x)
            for x in op[
                "exit_labels"
            ]
        ]


        distributions[
            kind
        ] = {}


        for days in DAYS_LIST:

            Pn = powers[
                kind
            ][
                days
            ]


            p = (
                p0
                @ Pn
            )


            grid = to_grid(
                op,
                p
            )


            inside = float(
                np.sum(
                    p[
                        :n_cells
                    ]
                )
            )


            exit_values = p[
                n_cells:
                n_cells
                + len(exit_labels)
            ]


            exits = {
                label:
                float(
                    exit_values[j]
                )
                for j, label
                in enumerate(
                    exit_labels
                )
            }


            distributions[
                kind
            ][
                days
            ] = {

                "p":
                    p,

                "grid":
                    grid,

                "inside":
                    inside,

                "exits":
                    exits
            }


            positive = grid[
                np.isfinite(
                    grid
                )
                & (
                    grid > 0
                )
            ]


            if positive.size > 0:

                all_grid_values.extend(
                    positive.tolist()
                )


            steps = int(
                round(
                    days
                    / TAU_DAYS
                )
            )


            model_days = (
                steps
                * TAU_DAYS
            )


            summary_rows.append(
                {
                    "point":
                        point_name,

                    "lon":
                        lon,

                    "lat":
                        lat,

                    "type":
                        kind,

                    "requested_days":
                        days,

                    "steps":
                        steps,

                    "model_days":
                        model_days,

                    "inside_R":
                        inside,

                    "exit_W":
                        exits.get(
                            "W",
                            np.nan
                        ),

                    "exit_E":
                        exits.get(
                            "E",
                            np.nan
                        ),

                    "exit_S":
                        exits.get(
                            "S",
                            np.nan
                        ),

                    "exit_N":
                        exits.get(
                            "N",
                            np.nan
                        )
                }
            )


            print(
                f"\n{kind} — {days} days"
            )


            print(
                f"  inside R: "
                f"{inside:.1%}"
            )


            for label in exit_labels:

                print(
                    f"  exit {label}: "
                    f"{exits[label]:.1%}"
                )


    # --------------------------------------------------------
    # COLOUR NORMALISATION
    #
    # Square-root scaling makes low probabilities visible,
    # especially at 365 days, while keeping one common scale
    # across the six panels.
    # --------------------------------------------------------

    if len(
        all_grid_values
    ) > 0:

        max_prob = float(
            np.max(
                all_grid_values
            )
        )


    else:

        max_prob = 1.0


    if max_prob <= 0:

        max_prob = 1.0


    norm = PowerNorm(
        gamma=0.5,
        vmin=0,
        vmax=max_prob
    )


    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    if HAVE_CARTOPY:

        projection = ccrs.PlateCarree()


        fig, axes = plt.subplots(
            2,
            3,
            figsize=(
                15,
                9
            ),
            subplot_kw={
                "projection":
                    projection
            }
        )


    else:

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(
                15,
                9
            )
        )


    last_mesh = None


    for row_number, kind in enumerate(
        TYPES
    ):

        op = ops[
            kind
        ]


        lon_edges = op[
            "lon_edges"
        ]


        lat_edges = op[
            "lat_edges"
        ]


        for col_number, days in enumerate(
            DAYS_LIST
        ):

            ax = axes[
                row_number,
                col_number
            ]


            result = distributions[
                kind
            ][
                days
            ]


            grid = result[
                "grid"
            ]


            inside = result[
                "inside"
            ]


            exits = result[
                "exits"
            ]


            if HAVE_CARTOPY:

                last_mesh = ax.pcolormesh(
                    lon_edges,
                    lat_edges,
                    grid,
                    cmap="viridis",
                    norm=norm,
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
                    cfeature.LAND,
                    facecolor="lightgray"
                )


                ax.coastlines(
                    linewidth=0.6
                )


                gl = ax.gridlines(
                    draw_labels=True,
                    linewidth=0.25,
                    alpha=0.4
                )


                gl.top_labels = False

                gl.right_labels = False


                if col_number > 0:

                    gl.left_labels = False


                ax.scatter(
                    lon,
                    lat,
                    marker="*",
                    s=100,
                    edgecolor="black",
                    linewidth=0.8,
                    transform=ccrs.PlateCarree(),
                    zorder=10
                )


            else:

                last_mesh = ax.pcolormesh(
                    lon_edges,
                    lat_edges,
                    grid,
                    cmap="viridis",
                    norm=norm,
                    shading="auto"
                )


                ax.scatter(
                    lon,
                    lat,
                    marker="*",
                    s=100,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=10
                )


                ax.set_xlim(
                    BOX[0],
                    BOX[1]
                )


                ax.set_ylim(
                    BOX[2],
                    BOX[3]
                )


                if row_number == 1:

                    ax.set_xlabel(
                        "Longitude"
                    )


                if col_number == 0:

                    ax.set_ylabel(
                        "Latitude"
                    )


            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            title = (
                f"{kind.capitalize()} — "
                f"{days} days\n"
                f"inside R {inside:.1%}"
            )


            ax.set_title(
                title,
                fontsize=10
            )


            # ------------------------------------------------
            # EXIT TEXT
            # ------------------------------------------------

            exit_text = (
                f"W {exits.get('W', 0):.1%}   "
                f"E {exits.get('E', 0):.1%}\n"
                f"S {exits.get('S', 0):.1%}   "
                f"N {exits.get('N', 0):.1%}"
            )


            ax.text(
                0.02,
                0.02,
                exit_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                bbox={
                    "facecolor":
                        "white",

                    "alpha":
                        0.75,

                    "edgecolor":
                        "none"
                }
            )


    # --------------------------------------------------------
    # FIGURE TITLE
    # --------------------------------------------------------

    fig.suptitle(
        (
            f"Surface Transport from {point_name}\n"
            f"Release: {lon:.2f}°E, "
            f"{abs(lat):.2f}°S — "
            f"1° grid, τ = {TAU_DAYS:g} days"
        ),
        fontsize=14
    )


    # --------------------------------------------------------
    # COMMON COLORBAR
    # --------------------------------------------------------

    if last_mesh is not None:

        cbar = fig.colorbar(
            last_mesh,
            ax=axes.ravel().tolist(),
            orientation="vertical",
            fraction=0.02,
            pad=0.02
        )


        cbar.set_label(
            "Probability mass in grid cell"
        )


    fig.subplots_adjust(
        top=0.88,
        right=0.90,
        wspace=0.12,
        hspace=0.22
    )


    # --------------------------------------------------------
    # SAVE FIGURE
    # --------------------------------------------------------

    file_name = (
        f"release_map_"
        f"{point_id:02d}_"
        f"{safe_name(point_name)}_"
        f"1deg.png"
    )


    out_file = (
        FIGURE_DIR
        / file_name
    )


    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        "\nSaved figure:"
    )

    print(
        out_file
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n\n========================================"
)

print(
    "RELEASE-POINT SUMMARY"
)

print(
    "========================================"
)


for point_name in summary_df[
    "point"
].unique():

    print(
        f"\n{point_name}"
    )


    here = summary_df[
        summary_df[
            "point"
        ]
        == point_name
    ]


    for kind in TYPES:

        part = here[
            here[
                "type"
            ]
            == kind
        ]


        print(
            f"\n  {kind}:"
        )


        for _, r in part.iterrows():

            print(
                f"    "
                f"{int(r['requested_days']):>3} d | "
                f"inside {r['inside_R']:.1%} | "
                f"W {r['exit_W']:.1%} | "
                f"E {r['exit_E']:.1%} | "
                f"S {r['exit_S']:.1%} | "
                f"N {r['exit_N']:.1%}"
            )


print(
    "\nSaved summary:"
)

print(
    summary_file
)


print(
    "\nDONE."
)
