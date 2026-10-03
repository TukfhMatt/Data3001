# Data3001 — Agulhas Surface Transport

Something goes into the water in the Agulhas region off South Africa: a container lost from a ship, a life raft, an oil slick. Where has the surface flow taken it after a week, a month, a year? Which parts of the region feed which others, and where does material collect or leave?

This project builds a **surface-transport operator** for that region: a transition matrix, estimated from real ocean drifter tracks, that anyone can load and step forward in time.


---

## Region

Box **R**: 10°E–55°E, 45°S–15°S, gridded at 0.5°, 1° and 2° (3,780 / 970 / 263 ocean cells). The box, grid and lag are command-line options (see below), so the same code runs on any box.

It covers the inflow to the Agulhas Current from the southern Mozambique Channel and south of Madagascar, the current along the South African east coast, the retroflection south of Africa where most of the flow turns back east, leakage into the South Atlantic, and the Agulhas Return Current.

The box balances shipping risk against oil retention (scored in `05_box_selection.py`): oil stays in it 136 days on average and 6.5% is still inside after a year, it contains 97% of the Cape Basin eddy activity (Agulhas rings), and it covers the coastal, Mozambique Channel and south-of-Madagascar shipping routes.

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
  - **Undrogued**: the drogue has been lost, so the drifter rides at the surface and slips downwind by about 10 cm/s (2.0% of the ERA5 10 m wind: eastward under the westerlies, north-westward under the trades). Closest match to oil, but oil drifts at about 3.5% of wind speed, so the oil matrix adds the missing 1.5%.
  - The two are not pooled: an "all drifters" matrix mixes 46–84% undrogued data depending on the cell, so its wind effect would follow sampling history rather than physics.
- ERA5 10 m winds (Copernicus Climate Data Store) at every drifter position supply the oil matrix's wind term (`08`, `09`). Planned: a life-raft version with its own windage.

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
python scripts/05_box_selection.py        # compare candidate boxes and choose R
python scripts/06_bootstrap.py             # ~2 min at 1°
python scripts/07_wind_slip.py             # undrogued − drogued slip per cell
python scripts/08_fetch_era5.py            # ERA5 winds at drifter positions (needs ~/.cdsapirc)
python scripts/09_oil_wind.py              # oil matrix: undrogued + missing windage (+ drogued check)
python scripts/10_validation.py            # held-out validation, 5-fold by drifter
python scripts/11_whole_box_analysis.py     # grid support across 0.5°, 1° and 2°
python scripts/12_coastal_coverage.py       # coastal-cell data support
python scripts/13_trap_stability.py         # 365-day retention hotspots and grid stability
python scripts/14_residence_time.py        # expected whole-box residence time across grid sizes
python scripts/15_exit_zones.py            # eventual W/E/S/N exit probabilities and dominant exit zones
python scripts/16_almost_invariant_regions.py  # spectral candidate transport regions at 1°
python scripts/17_release_point_maps.py      # release maps at 7, 30 and 365 days
python scripts/18_sensitivity.py           # τ and grid sensitivity (needs 03 and 10 for every setting)
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
| `00_fetch_agulhas_subset.py` | Streams GDP hourly data chunk by chunk, keeps 5°W–65°E, 60–5°S (any candidate box plus a margin) at hourly resolution, drops positions with gaps > 6 h, and keeps each drifter's GDP fate (`typedeath`, `end_date`); `--metadata-only` refreshes those fields on an existing subset | `data/agulhas_gdp1h_subset.nc` |
| `01_agulhas_gdp_exploration.py` | Observation and drifter counts in R, drogued vs undrogued, coverage map, observations per year | `figures/agulhas_gdp_coverage.png`, `figures/agulhas_gdp_temporal.png` |
| `02_grid_coverage.py` | Distinct drifters per cell at each grid size (default 0.5°, 1°, 2°), used to choose the grid | `figures/agulhas_grid_coverage.png`, `data/grid_coverage.npz` |
| `03_transition_matrix.py` | Builds transition matrices for drogued and undrogued drifters at every grid and lag given (default 0.5°, 1°, 2°; τ = 3.5 days), with exit states for each edge and a stranded state from drifters that ran aground in R, then runs a test release | `data/P_{drogued,undrogued}_{res}deg_{tau}d_{box}.npz`, e.g. `data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz` |
| `04_shipping_overlay.py` | Overlays commercial shipping density (World Bank / IMF AIS, 2015–2021) on the mean current from each drogue type, picks candidate release points (busiest lanes + known hotspots) in cells reliable in both matrices, and ranks them by near-shore exposure under the drogued and the undrogued matrix. Needs the shipping zip, see below. | `figures/agulhas_shipping_currents_{drogued,undrogued}.png`, `data/release_candidates.csv`, `data/shipping_{res}deg.npz` |
| `05_box_selection.py` | Scores every candidate bounding box containing the Agulhas core on undrogued data volume, reliable-cell share, shipping density and oil retention; reports the Pareto set, a recommended box and a side-by-side of named options | `figures/agulhas_box_selection.png`, `data/box_pareto.csv`, `data/box_options.csv` |
| `06_bootstrap.py` | Resamples whole drifters 1,000 times per drogue type, rebuilds the matrix each time and re-propagates the release points from `04`: 95% intervals on each point's fate (still in R, exits, stranded, 30-day coastal exposure), whether drogued and undrogued differ, and how stable each row is (median bootstrap TVD) | `data/bootstrap_release_*.csv`, `data/bootstrap_difference_*.csv`, `data/bootstrap_rows_*.npz`, `figures/agulhas_bootstrap_fates_*.png`, `figures/agulhas_bootstrap_rows_*.png` |
| `07_wind_slip.py` | Undrogued − drogued mean velocity per cell (the extra surface wind drift undrogued drifters carry), latitude-band means with 95% intervals from resampling whole drifters | `figures/agulhas_wind_slip_{res}deg_{box}.png`, `data/wind_slip_cells_…csv`, `data/wind_slip_bands_…csv` |
| `08_fetch_era5.py` | Requests ERA5 10 m winds from the Copernicus CDS API one year at a time, interpolates them to every drifter position and deletes the raw year; needs a CDS account, the ERA5 licence and `~/.cdsapirc` | `data/drifter_wind10m.nc` |
| `09_oil_wind.py` | Oil matrix: undrogued transitions moved by the windage they lack (3.5% minus their measured windage) × the ERA5 wind along each 3.5-day path; a drogued + 3.5% check matrix; comparison at the release points, with drifter-bootstrap intervals and the number of grounded drifters behind each stranding share | `data/P_oil_…npz`, `data/P_oil_check_…npz`, `data/oil_compare_…csv`, `figures/agulhas_oil_durban_30d_…png` |
| `10_validation.py` | Held-out validation, 5-fold by drifter: position error against persistence and mean-current advection, log score against climatology, exit and stranding calibration | `data/validation_summary_…csv`, `figures/agulhas_validation_…png` |
| `11_whole_box_analysis.py` | Checks outgoing-transition and distinct-drifter support across the 0.5°, 1° and 2° drogued matrices. Reports active/empty cells, transition-count thresholds, independent drifter support and flagged low-support rows. | `data/whole_box_grid_support.csv`, `figures/whole_box_transition_support.png`, `figures/whole_box_drifter_support.png` |
| `12_coastal_coverage.py` | Identifies ocean-state cells intersecting the Natural Earth 10 m coastline and measures coastal transition and drifter support at all three grid sizes. | `data/coastal_grid_support.csv`, `figures/coastal_drifter_support.png` |
| `13_trap_stability.py` | Uses 365-day in-box retention as a hotspot score, excludes empty and flagged rows, selects the top 10% reliable cells, and compares hotspot locations across grid sizes using Jaccard and overlap coefficients. | `data/retention_hotspot_summary.csv`, `data/retention_hotspot_stability.csv` |
| `14_residence_time.py` | Estimates expected whole-box residence time from each reliable drogued starting cell at 0.5°, 1° and 2°. Empty rows are treated as unsupported, and the survival series is integrated for up to 10 years. | `data/residence_time_{res}deg.csv`, `data/residence_time_summary.csv`, `figures/residence_time_1deg.png` |
| `15_exit_zones.py` | Calculates eventual exit probabilities through the four absorbing box boundaries (`W`, `E`, `S`, `N`), identifies the dominant exit side for each reliable starting cell, and compares exit structure across grid sizes. | `data/exit_zones_{res}deg.csv`, `data/exit_zone_summary.csv`, `figures/dominant_exit_zone_1deg.png` |
| `16_almost_invariant_regions.py` | Uses spectral clustering of the 1° drogued transport-affinity matrix to identify four candidate almost-invariant regions, then validates them with the original directional transition matrix at one step, 30 days and 365 days. | `data/almost_invariant_region_summary.csv`, `data/almost_invariant_regions_1deg.csv`, `figures/almost_invariant_regions_1deg.png` |
| `17_release_point_maps.py` | Propagates all shipping-selected release points under the 1° drogued and undrogued operators at approximately 1 week, 1 month and 1 year, producing spatial probability maps and exit-fate summaries. | `data/release_point_map_summary.csv`, `figures/release_map_*_1deg.png` |
| `18_sensitivity.py` | Compares the reference (1°, τ = 3.5 d) with τ = 2 and 5 days and with 0.5° and 2° grids: held-out skill from `10` at common horizons, and the release points' one-year fates against the reference's bootstrap interval from `06` | `data/sensitivity_skill_{box}.csv`, `data/sensitivity_fates_{box}.csv`, `figures/agulhas_sensitivity_{box}.png` |
| `config.py` | Default region, grids, lag and data path; command-line options; matrix file names | — |
| `transport.py` | Helpers to load and use a saved matrix, plus the pairing and counting helpers shared by `03` and `06` | — |

Shipping data for `04` (458 MB, downloaded once, read directly from the zip):

```bash
mkdir -p data/shipping
curl -L -o data/shipping/shipdensity_commercial.zip \
  https://datacatalogfiles.worldbank.org/ddh-published/0037580/5/DR0045405/shipdensity_commercial_.zip
```

`notebooks/01_agulhas_gdp_exploration.ipynb` is not part of the pipeline and does not run on this setup; use the scripts.

---
## SQ2 whole-box feasibility

Whole-box transition support is strong across all three drogued-grid resolutions. Among active cells, 97.6% of 0.5° cells, 99.4% of 1° cells and 99.6% of 2° cells have at least 50 outgoing transitions. The median numbers of outgoing transitions are 432, 1,750 and 6,395 per active cell, respectively.

Independent drifter support is more sensitive to grid resolution. The share of active cells supported by at least 10 distinct drifters is 51.6% at 0.5°, 78.1% at 1° and 88.8% at 2°. Correspondingly, 48.4%, 21.9% and 11.2% of active rows are flagged as having fewer than 10 distinct drifters. This supports 1° as the main analysis grid, with 2° as a coarser robustness check; the 0.5° grid is substantially more weakly sampled.

Coastal cells are less well sampled than the box as a whole. At 1°, 93.2% of active coastal cells still have at least 50 outgoing transitions, but only 39.2% have at least 10 distinct drifters. At 2°, these figures improve to 97.9% and 52.1%, respectively. Coastal conclusions should therefore be interpreted more cautiously, especially at finer resolution.

Long-term retention analysis identifies a recurring high-retention region in the western Indian Ocean, approximately around 41–45°E and 29–33°S. The top-10% 365-day retention hotspots show relatively strong agreement between the 1° and 2° grids (48.7% Jaccard similarity; 73.3% overlap coefficient), but substantially weaker agreement involving the 0.5° grid (15.9% Jaccard for 0.5° vs 1°, and 12.0% for 0.5° vs 2°). This is consistent with the much weaker independent-drifter support at 0.5°.

Expected whole-box residence times are also broadly stable at the better-supported resolutions. Across reliable starting cells, the mean residence time is 236.9 days at 1° and 227.7 days at 2°, with medians of 233.1 and 226.2 days, respectively. The 0.5° estimate is somewhat lower at a mean of 209.7 days. The longest-residence cells again cluster broadly around 41–45°E and 29–33°S. After 10 years of integration, the maximum remaining survival probability is negligible at all three resolutions, so truncation of the residence-time estimate is small.

Exit behaviour is highly consistent across grid sizes. The eastern boundary is the dominant eventual exit for 80.5% of reliable 0.5° cells, 83.4% of reliable 1° cells and 81.4% of reliable 2° cells. At 1°, the mean eventual exit probabilities across reliable starting cells are 71.3% east, 20.0% west, 6.7% south and 2.1% north. All reliable cells at all three resolutions have at least 80% of their eventual fate resolved into the four absorbing exit states, indicating a robust large-scale eastward exit pattern.

A spectral analysis of the 1° drogued matrix identifies four candidate almost-invariant transport regions. Their one-step internal retention ranges from 80.8% to 94.3%, while 30-day retention within the same region ranges from 55.2% to 70.2%. By 365 days, only 2.4%–8.7% remains in the same region, showing that these structures are coherent on short-to-intermediate timescales rather than permanently isolated. These regions are therefore interpreted as candidate almost-invariant regions from a spectral-clustering approximation, rather than as an exact PCCA+ decomposition.

Overall, SQ2 is well supported for broad offshore transport, retention, residence-time and exit-pattern analysis at 1°–2° resolution. The 1° grid is used as the main analysis scale and the 2° grid provides a useful robustness check. The 0.5° grid remains useful as a sensitivity case but is too sparsely supported to claim stable fine-scale traps, while coastal results require additional caution. Whole-box diagnostics consistently show a persistent high-retention zone in the western Indian Ocean, residence times of roughly 7–8 months at the better-supported resolutions, predominantly eastward eventual exit, and several short-to-intermediate-timescale coherent transport regions.

## Using the transition matrix

Each `.npz` file is a self-contained operator. `P[i, j]` is the probability that material in state `i` is in state `j` one lag τ later (`tau_days`).

- **States `0 … n_cells − 1`**: ocean cells of R (970 at 1°; the mapping to the grid is in `cell_flat`).
- **Last 5 states**: absorbing, in the order of `exit_labels`: `W`, `E`, `S`, `N` (exited R across that edge) and `stranded` (ran aground on a coast inside R).

```python
import sys; sys.path.append("scripts")
from transport import load_operator, release, propagate, to_grid

op = load_operator("data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz")
p0 = release(op, lon=31.5, lat=-30.5)      # all mass in the cell off Durban
p  = propagate(op, p0, days=30)            # rounded to whole 3.5-day steps
grid = to_grid(op, p)                      # 30 × 45 lat/lon array, NaN off-ocean
n_cells = len(op["cell_flat"])
print("still in R:", p[:n_cells].sum(), "exited / stranded:", dict(zip(op["exit_labels"], p[n_cells:])))
```

Other fields in each file:

| Field | Meaning |
|---|---|
| `C` | Raw transition counts behind `P` |
| `row_drifters`, `row_obs` | Distinct drifters and transitions behind each row |
| `flagged` | Rows built from fewer than 10 drifters (less reliable) |
| `empty` | Cells with no outgoing moves of their own; their rows pool the moves of the nearest ring of ocean cells |
| `lon_edges`, `lat_edges`, `res` | Grid definition |
| `box`, `tau_days`, `drogued`, `years`, `source` | Metadata |

At τ = 3.5 days: 1 week = 2 steps, 1 month ≈ 9 steps, 1 year ≈ 104 steps.

## Status

- [x] Data subset, exploration, grid coverage
- [x] Time step: τ = 3.5 days (the velocity decorrelation time over the core region 10–40°E, 45–25°S is about 1.2–1.4 days)
- [x] Transition matrices (drogued and undrogued)
- [x] Region, grid and lag are parameters; matrices at 0.5°, 1° and 2° for both drogue types
- [x] Shipping lanes over currents; candidate release points chosen and ranked
- [x] Stranded state from GDP's record of drifters that ran aground in R (139 drifters, 137 of them undrogued)
- [x] Drifter bootstrap: 95% intervals on release-point fates, drogued vs undrogued differences, row stability
- [x] Release-point maps at 1 week, 1 month, 1 year
- [x] Held-out validation (5-fold by drifter): calibrated exits and stranding, beats persistence and mean-current advection by 28 days
- [x] Sensitivity to τ and grid size: τ = 2–5 days gives the same results; 1° is used (0.5° adds no skill and has 50% thin drogued rows, 2° loses short-range skill and shifts one-year fates)
- [ ] Seasonal matrices
- [x] ERA5 winds at every drifter position; oil matrix = undrogued + the missing 1.5% windage (undrogued carries 2.0%), with a drogued + 3.5% check
- [x] Whole-box analysis: accumulation zones, exit zones, residence time, almost-invariant regions
