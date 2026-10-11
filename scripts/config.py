"""
Shared settings for the Agulhas pipeline and the command-line options that
override them. Scripts 01–04 read the region, grid and lag from here, so a
run on another box, grid or lag needs no code change, e.g.

    .venv/bin/python scripts/03_transition_matrix.py --box 10 55 -41 -15 --res 1 --tau 5

Defaults are the current analysis settings. The box must lie inside the area
fetched by 00 (5°W–65°E, 60–5°S), with some margin for exits.
"""

import argparse

import numpy as np

GDP_LOCAL = "data/agulhas_gdp1h_subset.nc"   # written by 00
BOX = (10.0, 55.0, -45.0, -15.0)             # lon_min, lon_max, lat_min, lat_max (group choice, see 05)
RESOLUTIONS = [0.5, 1.0, 2.0]                # grids compared in 02 and built in 03
RES = 1.0                                    # grid for single-grid steps (04)
TAU_DAYS = 3.5
MIN_DRIFTERS = 10                            # rows with fewer distinct drifters are flagged
DROGUE_TYPES = {"drogued": True, "undrogued": False}


def parse_args(doc, grid=False, lag=False, multi=False):
    """Parse --box and --data, plus --res / --tau when the script uses them.
    With multi=True, --res and --tau take several values (lists)."""
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    p.add_argument("--box", nargs=4, type=float, default=list(BOX),
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    p.add_argument("--data", default=GDP_LOCAL, help="subset written by 00")
    if grid:
        p.add_argument("--res", type=float, nargs="+" if multi else None,
                       default=RESOLUTIONS if multi else RES, help="grid size in degrees")
    if lag:
        p.add_argument("--tau", type=float, nargs="+" if multi else None,
                       default=[TAU_DAYS] if multi else TAU_DAYS, help="lag in days")
    return p.parse_args()


def grid_shape(box, res):
    """(n_lon, n_lat) cells of size res from the box's south-west corner. When a
    side is not a multiple of res, the last row/column is a partial cell cut
    off at the box edge (e.g. 45° wide at 2° → 22 full columns + one 1° column)."""
    n = lambda span: int(np.ceil(span / res - 1e-9))
    return n(box[1] - box[0]), n(box[3] - box[2])


def _tag(x):
    return f"{x:g}".replace(".", "p")


def box_tag(box):
    """e.g. (5, 50, -50, -20) → '5E-50E_50S-20S'."""
    lo = lambda x: f"{abs(x):g}{'W' if x < 0 else 'E'}"
    la = lambda x: f"{abs(x):g}{'S' if x < 0 else 'N'}"
    return f"{lo(box[0])}-{lo(box[1])}_{la(box[2])}-{la(box[3])}"


def setting_tag(res, tau, box):
    """File tag for one or several grids and lags, so outputs of different
    settings never overwrite each other, e.g. '1deg_3p5d_10E-55E_45S-15S' or
    '0p5-1-2deg_3p5d_10E-55E_45S-15S'."""
    join = lambda xs: "-".join(_tag(x) for x in np.atleast_1d(xs))
    return f"{join(res)}deg_{join(tau)}d_{box_tag(box)}"


def operator_path(name, res, tau, box):
    """Saved transition matrix for one drogue type, grid, lag and region."""
    return f"data/P_{name}_{_tag(res)}deg_{_tag(tau)}d_{box_tag(box)}.npz"
