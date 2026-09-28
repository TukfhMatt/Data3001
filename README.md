# Data3001 — Agulhas Surface Transport

Something goes into the water in the Agulhas region off South Africa: a container lost from a ship, a life raft, an oil slick. Where has the surface flow taken it after a week, a month, a year? Which parts of the region feed which others, and where does material collect or leave?

This project builds a **surface-transport operator** for that region: a transition matrix, estimated from real ocean drifter tracks, that anyone can load and step forward in time.


---

## Region

Box **R**: 10°E–55°E, 45°S–15°S, gridded at 0.5°, 1° and 2° (3,780 / 970 / 263 ocean cells). The box, grid and lag are command-line options (see below), so the same code runs on any box.

It covers the inflow to the Agulhas Current from the southern Mozambique Channel and south of Madagascar, the current along the South African east coast, the retroflection south of Africa where most of the flow turns back east, leakage into the South Atlantic, and the Agulhas Return Current.

The box was chosen by the group (28 Sep 2026) from four options scored in `05_box_selection.py`. It gives the best balance of shipping risk and oil retention: the longest mean residence (136 days), the most oil still in the box after a year (6.5%), 97% of the Cape Basin eddy activity (Agulhas rings), and the coastal, Mozambique Channel and south-of-Madagascar shipping routes. Earlier boxes were 10–40°E, 45–25°S and then 5–50°E, 50–20°S.

## Data

- **NOAA Global Drifter Program (GDP)**, hourly product v2.01, read from the public AWS S3 zarr store (the same source as `clouddrift.datasets.gdp1h()`), at the native hourly resolution (positions interpolated across > 6 h between satellite fixes are dropped, 0.3% of the data). Coverage in R, 1995–2022:

  | | Hourly observations |
  |---|---|
  | All | 5,952,447 |
  | … drogued | 1,834,513 |
  | … undrogued | 4,117,934 |
  | Distinct drifters | 1,503 |

  Consecutive hourly positions are strongly correlated; the independent sample size is the number of drifters (~1,300 undrogued), not the observation count.

- Drifters are split by **drogue status**, and a matrix is built for each:
  - **Drogued**: a sea anchor at 15 m is attached, so the drifter follows the near-surface current with little wind effect. This is passive water transport (e.g. a submerged container).
  - **Undrogued**: the drogue has been lost, so the drifter rides at the surface and slips downwind by about 10 cm/s (roughly 1% of wind speed: eastward under the westerlies, north-westward under the trades). Closest match to oil, but oil drifts at about 3–3.5% of wind speed, so this is a partial wind effect.
  - The two are not pooled: an "all drifters" matrix mixes 46–84% undrogued data depending on the cell, so its wind effect would follow sampling history rather than physics.
- Planned: ERA5 10 m winds (Copernicus Climate Data Store) for the oil and life-raft windage models.

---

## Setup

Python 3.9, with a virtual environment in `.venv/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas xarray netCDF4 h5netcdf "zarr<3" fsspec aiohttp \
            matplotlib cartopy tqdm clouddrift rasterio
```

`cartopy` is optional; the plots fall back to plain matplotlib without coastlines.

## Running the pipeline

Run everything **from the repository root**, since the scripts use relative paths to `data/` and `figures/`:

```bash
source .venv/bin/activate
python scripts/00_fetch_agulhas_subset.py   # once: ~1.2 GB streamed, 492 MB saved
python scripts/01_agulhas_gdp_exploration.py
python scripts/02_grid_coverage.py
python scripts/03_transition_matrix.py
python scripts/04_shipping_overlay.py
```

Scripts `01`–`04` take the region, grid and lag as options (defaults in `scripts/config.py`: R = 10–55°E, 45–15°S; grids 0.5°, 1°, 2°; τ = 3.5 days):

```bash
python scripts/02_grid_coverage.py --res 0.5 1 2
python scripts/03_transition_matrix.py --box 10 55 -41 -15 --res 1 --tau 2 3.5 5   # several grids / lags in one run
python scripts/04_shipping_overlay.py --res 1 --tau 3.5                           # needs that matrix from 03
```

τ must be a whole number of hours. If a box side is not a multiple of the grid size (e.g. 45° at 2°), the last row or column is a partial cell cut off at the box edge.

`data/` and `figures/` are not in git. After cloning, run `00` and then `03` to rebuild the matrices.

| Script | What it does | Output |
|---|---|---|
| `00_fetch_agulhas_subset.py` | Streams GDP hourly data chunk by chunk, keeps 5°W–65°E, 60–5°S (any candidate box plus a margin) at hourly resolution, drops positions with gaps > 6 h | `data/agulhas_gdp1h_subset.nc` |
| `01_agulhas_gdp_exploration.py` | Observation and drifter counts in R, drogued vs undrogued, coverage map, observations per year | `figures/agulhas_gdp_coverage.png`, `figures/agulhas_gdp_temporal.png` |
| `02_grid_coverage.py` | Distinct drifters per cell at each grid size (default 0.5°, 1°, 2°), used to choose the grid | `figures/agulhas_grid_coverage.png`, `data/grid_coverage.npz` |
| `03_transition_matrix.py` | Builds transition matrices for drogued and undrogued drifters at every grid and lag given (default 0.5°, 1°, 2°; τ = 3.5 days), then runs a test release | `data/P_{drogued,undrogued}_{res}deg_{tau}d_{box}.npz`, e.g. `data/P_drogued_1deg_3p5d_5E-50E_50S-20S.npz` |
| `04_shipping_overlay.py` | Overlays commercial shipping density (World Bank / IMF AIS, 2015–2021) on the mean current from each drogue type, picks candidate release points (busiest lanes + known hotspots) in cells reliable in both matrices, and ranks them by near-shore exposure under the drogued and the undrogued matrix. Needs the shipping zip, see below. | `figures/agulhas_shipping_currents_{drogued,undrogued}.png`, `data/release_candidates.csv`, `data/shipping_{res}deg.npz` |
| `05_box_selection.py` | Scores every candidate bounding box containing the Agulhas core on undrogued data volume, reliable-cell share, shipping density and oil retention; reports the Pareto set, a recommended box and a side-by-side of named options | `figures/agulhas_box_selection.png`, `data/box_pareto.csv`, `data/box_options.csv` |
| `config.py` | Default region, grids, lag and data path; command-line options; matrix file names | — |
| `transport.py` | Helpers to load and use a saved matrix | — |

Shipping data for `04` (458 MB, downloaded once, read directly from the zip):

```bash
mkdir -p data/shipping
curl -L -o data/shipping/shipdensity_commercial.zip \
  https://datacatalogfiles.worldbank.org/ddh-published/0037580/5/DR0045405/shipdensity_commercial_.zip
```

`notebooks/01_agulhas_gdp_exploration.ipynb` is an earlier interactive version of script 01 and is out of date: it calls a CloudDrift loader that does not run on this setup. Use the scripts.

---

## Using the transition matrix

Each `.npz` file is a self-contained operator. `P[i, j]` is the probability that material in state `i` is in state `j` one lag τ later (`tau_days`).

- **States `0 … n_cells − 1`**: ocean cells of R (1,112 at 1°; the mapping to the grid is in `cell_flat`).
- **Last 4 states**: absorbing "exited R" states, in the order of `exit_labels` (`W`, `E`, `S`, `N`).

```python
import sys; sys.path.append("scripts")
from transport import load_operator, release, propagate, to_grid

op = load_operator("data/P_drogued_1deg_3p5d_5E-50E_50S-20S.npz")
p0 = release(op, lon=31.5, lat=-30.5)      # all mass in the cell off Durban
p  = propagate(op, p0, days=30)            # rounded to whole 3.5-day steps
grid = to_grid(op, p)                      # 30 × 45 lat/lon array, NaN off-ocean
n_cells = len(op["cell_flat"])
print("still in R:", p[:n_cells].sum(), "exited:", dict(zip(op["exit_labels"], p[n_cells:])))
```

Other fields in each file:

| Field | Meaning |
|---|---|
| `C` | Raw transition counts behind `P` |
| `row_drifters`, `row_obs` | Distinct drifters and transitions behind each row |
| `flagged` | Rows built from fewer than 10 drifters (less reliable) |
| `empty` | Cells with no outgoing data; these hold mass in place |
| `lon_edges`, `lat_edges`, `res` | Grid definition |
| `box`, `tau_days`, `drogued`, `years`, `source` | Metadata |

At τ = 3.5 days: 1 week = 2 steps, 1 month ≈ 9 steps, 1 year ≈ 104 steps.

## Status

- [x] Data subset, exploration, grid coverage
- [x] Time step chosen: τ = 3.5 days (the velocity decorrelation time is about 1.2–1.4 days, estimated on the initial 10–40°E box)
- [x] First transition matrices (drogued and undrogued)
- [x] Region, grid and lag are parameters; matrices at 0.5°, 1° and 2° for both drogue types
- [x] Shipping lanes over currents; candidate release points chosen and ranked
- [ ] Release-point maps at 1 week, 1 month, 1 year
- [ ] Validation with held-out drifters; sensitivity to τ and grid size (the matrices can now be built, the comparison is not done)
- [ ] Oil version with ERA5 winds and mass decay
- [ ] Whole-box analysis: accumulation zones, exit zones, residence time, almost-invariant regions
