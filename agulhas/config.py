"""
Shared settings for the Agulhas pipeline and the command-line options that
override them. Steps read the region, grid and lag from here, so a run on
another box, grid or lag needs no code change, e.g.

    python -m agulhas transition_matrix --box 10 55 -41 -15 --res 1 --tau 5

Defaults are the current analysis settings. The box must lie inside the area
fetched by fetch_gdp_subset (by default 5°W–65°E, 60–5°S), with some margin for exits; a box
elsewhere needs its own subset from fetch_gdp_subset, passed with --data.
"""

import argparse
import os

import numpy as np

SHIP_URL = ("https://datacatalogfiles.worldbank.org/ddh-published/0037580/5/"
            "DR0045405/shipdensity_commercial_.zip")
SHIP_ZIP = "data/shipping/shipdensity_commercial.zip"
GDP_LOCAL = "data/agulhas_gdp1h_subset.nc"   # written by fetch_gdp_subset
# ERA5 winds at GDP_LOCAL's positions, published with the repository so a run needs no CDS account
WIND_URL = "https://github.com/TukfhMatt/Data3001/releases/download/data-v1/drifter_wind10m.nc"
WIND_SHA256 = "08f83d03d6d5cec26eff78ab1619184ee4bf5705f66c0056ca11b5f88ecb18f3"
WIND_SUBSET = "70bb9d892104087878384940dd1724a1b798a09f0198a30c3169e2ce6845ce82"   # Drifters.fingerprint they fit
BOX = (10.0, 55.0, -45.0, -15.0)             # lon_min, lon_max, lat_min, lat_max (group choice, see box_selection)
RESOLUTIONS = [0.5, 1.0, 2.0]                # grids compared in grid_coverage and built in transition_matrix
RES = 1.0                                    # grid for single-grid steps (shipping_overlay)
TAU_DAYS = 3.5
MIN_DRIFTERS = 10                            # rows with fewer distinct drifters are flagged
OIL_WINDAGE_RANGE = {"oil_3pct": 0.03, "oil": 0.035, "oil_4pct": 0.04}   # oil matrices, by share of the wind
DISAGREE_TVD = 0.3                           # oil against its check at 30 d above this depends on the method
# Oil mass left t days after release, for heavy (bunker) fuel oil: at most about 15% evaporates within days
# (Mackay et al. 1980, 85% non-volatile at API 15), and the rest floats as emulsion and tar that is lost over
# months. Floating tar stays at sea for 1–4 months (Sleeter and Butler 1982), so the residue decays with a
# 60-day e-folding time, bounded by 30 and 120 days. The residue time decides the mass reaching a coast,
# since half of the year's stranding comes after about 8 weeks. No loss at all is the share of parcels that strand.
OIL_EVAPORATES = 0.15
OIL_EVAPORATION_DAYS = 2.0
OIL_RESIDUE_DAYS = {"mass_30d": 30.0, "mass": 60.0, "mass_120d": 120.0}   # "mass" is the headline
DROGUE_TYPES = {"drogued": True, "undrogued": False}


def oil_mass_left(days, residue_days=OIL_RESIDUE_DAYS["mass"]):
    """Share of the oil's mass left `days` after release: the part that evaporates plus the residue."""
    t = np.asarray(days, float)
    return OIL_EVAPORATES * np.exp(-t / OIL_EVAPORATION_DAYS) + (1 - OIL_EVAPORATES) * np.exp(-t / residue_days)


def parse_args(doc, grid=False, lag=False, multi=False, extra=None, argv=None):
    """Parse --box and --data, plus --res / --tau when the step uses them.
    With multi=True, --res and --tau take several values (lists). extra adds a
    step's own options to the parser, and argv replaces the command line."""
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    p.add_argument("--box", nargs=4, type=float, default=list(BOX),
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    p.add_argument("--data", default=GDP_LOCAL, help="subset written by fetch_gdp_subset")
    if grid:
        p.add_argument("--res", type=float, nargs="+" if multi else None,
                       default=RESOLUTIONS if multi else RES, help="grid size in degrees")
    if lag:
        p.add_argument("--tau", type=float, nargs="+" if multi else None,
                       default=[TAU_DAYS] if multi else TAU_DAYS, help="lag in days")
    if extra:
        extra(p)
    args = p.parse_args(argv)
    check_box_in_subset(args.box, args.data)
    return args


def check_box_in_subset(box, data):
    """Stop if the box is outside the area the subset was fetched for, which
    would otherwise give empty or partial matrices without an error."""
    if not os.path.exists(data):
        return
    import xarray as xr
    with xr.open_dataset(data) as ds:
        lon_min, lon_max = map(float, ds.attrs["lon_range"].split(".."))
        lat_min, lat_max = map(float, ds.attrs["lat_range"].split(".."))
    if box[0] < lon_min or box[1] > lon_max or box[2] < lat_min or box[3] > lat_max:
        raise SystemExit(f"Box {list(box)} is outside {data} (lon {lon_min:g} to {lon_max:g}, "
                         f"lat {lat_min:g} to {lat_max:g}). Fetch it with fetch_gdp_subset and pass that file with --data.")


def reference_grid(args):
    """The grid a many-grid step draws its maps on: the default grid if it was given."""
    return RES if RES in args.res else args.res[0]


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


def _join(xs):
    return "-".join(_tag(x) for x in np.atleast_1d(xs))


def grid_tag(res, box):
    """File tag for a step that depends on the grid but not the lag, e.g.
    '0p5-1-2deg_10E-55E_45S-15S'."""
    return f"{_join(res)}deg_{box_tag(box)}"


def setting_tag(res, tau, box):
    """File tag for one or several grids and lags, so outputs of different
    settings never overwrite each other, e.g. '1deg_3p5d_10E-55E_45S-15S' or
    '0p5-1-2deg_3p5d_10E-55E_45S-15S'."""
    return f"{_join(res)}deg_{_join(tau)}d_{box_tag(box)}"


def lag_tag(tau, box):
    """File tag for a step that depends on the lag but not the grid, e.g. '3p5d_10E-55E_45S-15S'."""
    return f"{_tag(tau)}d_{box_tag(box)}"


def region_prefix(data):
    """First word of the subset's file name, which starts figure names, e.g.
    'gulfstream' for data/gulfstream_gdp1h_subset.nc."""
    return os.path.basename(data).split("_")[0]


def place_label(lon, lat):
    """e.g. (-70.5, 35.5) → '70.5°W 35.5°N'."""
    return f"{abs(lon):.1f}°{'W' if lon < 0 else 'E'} {abs(lat):.1f}°{'S' if lat < 0 else 'N'}"


def wind_paths(data):
    """Per-year folder and combined file of ERA5 wind at a subset's positions
    (fetch_era5), so each subset keeps its own."""
    if data == GDP_LOCAL:
        return "data/era5", "data/drifter_wind10m.nc"
    stem = os.path.splitext(os.path.basename(data))[0]
    return f"data/era5_{stem}", f"data/drifter_wind10m_{stem}.nc"


def operator_path(name, res, tau, box):
    """Saved transition matrix for one drogue type, grid, lag and region."""
    return f"data/P_{name}_{_tag(res)}deg_{_tag(tau)}d_{box_tag(box)}.npz"


def release_points_path(box, res=RES, tau=TAU_DAYS):
    """Candidate release points from shipping_overlay, by the box, grid and lag they were
    picked on. Later steps read their box's points picked at the default grid
    and lag, whatever grid and lag they run on, so results at different
    settings compare the same points."""
    return f"data/release_candidates_{setting_tag(res, tau, box)}.csv"
