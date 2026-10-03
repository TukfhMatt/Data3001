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
python scripts/11_whole_box_analysis.py    # row support per grid and drogue type
python scripts/12_coastal_coverage.py      # coastal-cell support against offshore
python scripts/13_trap_stability.py        # one-year retention hotspots, across grids and bootstrapped (~1 min)
python scripts/14_residence_time.py        # expected time afloat in R from each cell
python scripts/15_exit_zones.py            # eventual exit (W/E/S/N) or stranding from each cell
python scripts/16_almost_invariant_regions.py  # spectral regions that keep material to themselves
python scripts/17_release_point_maps.py    # release maps at 7, 30 and 365 days (needs 06)
python scripts/18_sensitivity.py           # τ and grid sensitivity (needs 03 and 10 for every setting)
```

Scripts `01`–`04` and `11`–`17` take the region, grid and lag as options (defaults in `scripts/config.py`: R = 10–55°E, 45–15°S; grids 0.5°, 1°, 2°; τ = 3.5 days):

```bash
python scripts/02_grid_coverage.py --res 0.5 1 2
python scripts/03_transition_matrix.py --box 10 55 -41 -15 --res 1 --tau 2 3.5 5   # several grids / lags in one run
python scripts/04_shipping_overlay.py --res 1 --tau 3.5                           # needs that matrix from 03
```

Outputs of `11`–`17` carry the grids and lags they were run on in their names (`{tag}`, e.g. `0p5-1-2deg_3p5d_10E-55E_45S-15S`), so runs at other settings do not overwrite each other.

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
| `11_whole_box_analysis.py` | Row support for every grid, lag and drogue type: active and empty cells, moves per row, distinct drifters per row and the share of reliable rows | `data/whole_box_support_{tag}.csv`, `figures/agulhas_whole_box_support_{tag}.png` |
| `12_coastal_coverage.py` | The same for coastal cells (a land neighbour on the grid, as in `04`, `06` and `09`), against offshore cells | `data/coastal_support_{tag}.csv`, `figures/agulhas_coastal_support_{tag}.png` |
| `13_trap_stability.py` | Scores each reliable cell by the share still afloat in R after 365 days; the top 10% are retention hotspots. Compares hotspots across grids where both grids are reliable, and bootstraps whole drifters on the 1° grid for 95% intervals and how often each cell is a hotspot | `data/retention_hotspots_{tag}.csv`, `data/retention_hotspot_agreement_{tag}.csv`, `data/retention_cells_{tag}.csv`, `data/retention_bootstrap_{tag}.csv`, `figures/agulhas_retention_hotspots_{tag}.png` |
| `14_residence_time.py` | Expected time afloat in R from each cell, exactly from the absorbing chain (`transport.absorption`), for every grid, lag and drogue type | `data/residence_time_{tag}.csv`, `data/residence_time_cells_{tag}.csv`, `figures/agulhas_residence_time_{tag}.png` |
| `15_exit_zones.py` | Eventual fate from each cell (left across W, E, S or N, or stranded), exactly from the absorbing chain, plus the fate after one year; mean fates and the most likely fate per cell | `data/exit_zones_{tag}.csv`, `data/exit_zones_cells_{tag}.csv`, `figures/agulhas_exit_zones_{tag}.png` |
| `16_almost_invariant_regions.py` | Four regions per drogue type that keep material to themselves, from spectral clustering of reliable cells; checked with the directional P (share staying after one step, 30 and 365 days) against longitude bands of the same sizes | `data/almost_invariant_regions_…csv`, `data/almost_invariant_cells_…csv`, `figures/agulhas_almost_invariant_…png` |
| `17_release_point_maps.py` | Maps of where material from each release point from `04` is after 7, 30 and 365 days under both matrices, with the share afloat in R, each exit and stranded, and 95% intervals from `06` | `data/release_point_maps_…csv`, `figures/agulhas_release_{nn}_{point}_….png` |
| `18_sensitivity.py` | Compares the reference (1°, τ = 3.5 d) with τ = 2 and 5 days and with 0.5° and 2° grids: held-out skill from `10` at common horizons, and the release points' one-year fates against the reference's bootstrap interval from `06` | `data/sensitivity_skill_{box}.csv`, `data/sensitivity_fates_{box}.csv`, `figures/agulhas_sensitivity_{box}.png` |
| `config.py` | Default region, grids, lag and data path; command-line options; matrix file names | — |
| `transport.py` | Helpers to load and use a saved matrix (release, propagation, cell centres, reliable cells, exact residence time and fate from the absorbing chain), plus the pairing, counting and drifter-bootstrap helpers shared by every script that builds a matrix | — |

Shipping data for `04` (458 MB, downloaded once, read directly from the zip):

```bash
mkdir -p data/shipping
curl -L -o data/shipping/shipdensity_commercial.zip \
  https://datacatalogfiles.worldbank.org/ddh-published/0037580/5/DR0045405/shipdensity_commercial_.zip
```

`notebooks/01_agulhas_gdp_exploration.ipynb` is not part of the pipeline and does not run on this setup; use the scripts.

---
## Whole-box results

At 1°, τ = 3.5 days, over reliable cells (≥ 10 drifters), drogued / undrogued unless stated. Scripts `11`–`17`.

- **Data support.** 78% of active drogued rows and 93% of undrogued rows are reliable at 1° (52% / 90% at 0.5°, 89% / 94% at 2°). Coastal cells are much thinner: 33% / 47% reliable at 1°, against 82% / 98% offshore, so coastal and stranding results carry wide intervals.
- **Retention hotspots.** Drogued material is held longest at 41–44°E, 30–33°S, south-west of Madagascar: about 67–69% is still in R after a year (95% intervals about 52–75%), and 71 of the 75 hotspot cells stay in the top 10% in at least half of 1,000 drifter resamples. Undrogued material is held longest further east, at 48–51°E, 30–33°S (55–56%, intervals 49–60%). Where both grids are reliable, the 1° and 2° hotspots agree (Jaccard 66% drogued, 73% undrogued); 0.5° agrees less for drogued (30–36%), whose rows are thin. Retention after a year also reflects distance from the open edges of R, so a hotspot is a place R holds material, not necessarily a closed eddy.
- **Residence time.** Material stays afloat in R for 233 / 203 days on average (median 230 / 213), with the longest stays in the hotspots (up to 539 / 459 days). The grids agree within about 40 days.
- **Exit zones.** Most material eventually leaves east with the Return Current: 71% / 62% on average, and east is the most likely fate in 83% / 75% of cells. The west (Atlantic leakage) takes 20% / 17%, the most likely fate in a strip along the west coast and the retroflection. 15% of undrogued material strands, and stranding is the most likely undrogued fate along the Mozambique Channel coast and east of Madagascar; drogued stranding is 0.7%, a lower bound (see `03`).
- **Almost-invariant regions.** Four regions for both drogue types: the retroflection and the Atlantic side, the Natal coast and Mozambique Channel, south of Madagascar and the Return Current, and east of Madagascar. They keep 81–94% of their material over one step and 55–70% over 30 days, and by a year only 1–9% remains in its own region. Three of the four keep 13–23 points more over 30 days than longitude bands of the same sizes; the western one matches its band, whose edge falls close to the retroflection anyway. The leading eigenvalues fall off smoothly, so four is a choice rather than a gap in the spectrum.
- **Release points.** After a year, 19% (14–26%) of drogued material from Durban is still in R, 51% has left east and 22% west, the expected behaviour. Undrogued material from the Mozambique Channel lane (40.5°E, 16.5°S) and the lanes south of Madagascar strands far more: 24–66% after a year, against 0–2% drogued.

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
