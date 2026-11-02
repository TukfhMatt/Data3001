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

  Consecutive hourly positions are strongly correlated; the independent sample size is the number of drifters (~1,170 undrogued), not the observation count.

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
pip install -r requirements.txt
```


`cartopy` is optional for most plots, which can fall back to plain matplotlib without coastlines. However, `21_coastal_pooling.py` requires both `cartopy` and `shapely` to apply its Natural Earth 50m geographic filter.

The script uses Natural Earth 50m land polygons to check whether an existing coastal-cell candidate intersects mapped land within its own grid cell or one of its eight neighbouring cells. The geographic filter excludes unsupported offshore candidates but does not add potentially missing coastal cells.

On the first run, Cartopy may need an internet connection to download the Natural Earth 50m land dataset. Subsequent runs can use the locally cached data.
  

## Running the pipeline

Run everything **from the repository root**, since the scripts use relative paths to `data/` and `figures/`:

```bash
source .venv/bin/activate
python scripts/00_fetch_agulhas_subset.py   # once: ~1.2 GB streamed, 492 MB saved
python scripts/01_agulhas_gdp_exploration.py
python scripts/02_grid_coverage.py
python scripts/03_transition_matrix.py
python scripts/03_transition_matrix.py --res 1 --tau 2 5 7  # other lags, needed by 10, 18 and 20
mkdir -p data/shipping                     # shipping data for 04, 05 and 24: 458 MB, read directly from the zip
curl -L -o data/shipping/shipdensity_commercial.zip \
  https://datacatalogfiles.worldbank.org/ddh-published/0037580/5/DR0045405/shipdensity_commercial_.zip
python scripts/04_shipping_overlay.py
python scripts/05_box_selection.py        # compare candidate boxes and choose R
python scripts/06_bootstrap.py             # ~2 min at 1°
python scripts/07_wind_slip.py             # undrogued − drogued slip per cell
python scripts/08_fetch_era5.py            # ERA5 winds at drifter positions (needs ~/.cdsapirc)
python scripts/09_oil_wind.py              # oil matrix: undrogued + missing windage (+ drogued check)
python scripts/10_validation.py            # held-out validation, 5-fold by drifter
python scripts/10_validation.py --res 1 --tau 2      # the other settings compared in 18
python scripts/10_validation.py --res 1 --tau 5
python scripts/10_validation.py --res 0.5 --tau 3.5
python scripts/10_validation.py --res 2 --tau 3.5
python scripts/11_whole_box_analysis.py    # row support per grid and drogue type
python scripts/12_coastal_coverage.py      # coastal-cell support against offshore
python scripts/13_trap_stability.py        # one-year retention hotspots, across grids and bootstrapped (~1 min)
python scripts/14_residence_time.py        # expected time afloat in R from each cell
python scripts/15_exit_zones.py            # eventual exit (W/E/S/N) or stranding from each cell
python scripts/16_almost_invariant_regions.py  # spectral regions that keep material to themselves
python scripts/17_release_point_maps.py    # release maps at 7, 30 and 365 days (needs 06)
python scripts/18_sensitivity.py           # τ and grid sensitivity
python scripts/19_seasonal_matrices.py     # DJF/MAM/JJA/SON seasonal operators and TVD comparison
python scripts/19_seasonal_matrices.py --res 2 --tau 3.5  # coarser seasonal robustness check
python scripts/20_chapman_kolmogorov.py    # P(tau)^2 vs direct P(2*tau) Markov-consistency check
python scripts/21_coastal_pooling.py       # pool thin coastal cells into short coastal segments
python scripts/22_drogue_loss.py           # when grounded drifters lost their drogue (stranding drogue rule)
python scripts/23_coastal_stranding.py     # split stranding into six named coasts (needs 09)
python scripts/24_traffic_exposure.py      # traffic-weighted exposure by coast and route (needs 21, 23)
python scripts/25_risk_atlas.py            # release-risk atlas: every cell and horizon with intervals (needs 23)
```

Scripts `01`–`04`, `11`–`17` and `19`–`21` take the region, grid and lag as options (defaults in `scripts/config.py`: R = 10–55°E, 45–15°S; grids 0.5°, 1° and 2°; τ = 3.5 days):

```bash
python scripts/02_grid_coverage.py --res 0.5 1 2
python scripts/03_transition_matrix.py --box 10 55 -41 -15 --res 1 --tau 2 3.5 5   # example: custom box and several lags
python scripts/04_shipping_overlay.py --res 1 --tau 3.5                           # needs that matrix from 03
python scripts/19_seasonal_matrices.py --res 1 --tau 3.5
python scripts/19_seasonal_matrices.py --res 2 --tau 3.5                         # coarser seasonal robustness check
python scripts/03_transition_matrix.py --res 1 --tau 7                           # direct 2τ matrix needed by 20
python scripts/20_chapman_kolmogorov.py --res 1 --tau 3.5                        # P(τ)^2 vs direct P(2τ)
python scripts/21_coastal_pooling.py --res 1 --tau 3.5                           # needs the annual matrices from 03
```

Outputs of `11`–`17` and `19`–`21` carry the grids and lags they were run on in their names (`{tag}`, e.g. `0p5-1-2deg_3p5d_10E-55E_45S-15S`), so runs at other settings do not overwrite each other. So do the coverage files from `02`, the release points and shipping grid from `04`, and the maps from `01` and `04`. Later steps read the release points for their box picked at the default grid and lag (`config.release_points_path`), whatever grid and lag they run on, so every setting is compared at the same points.

τ must be a whole number of hours. If a box side is not a multiple of the grid size (e.g. 45° at 2°), the last row or column is a partial cell cut off at the box edge.

`data/` and `figures/` are not in git. After cloning, run `00` and then `03` to rebuild the matrices.

### Another region

A box outside the area saved by `00` needs its own subset, with a margin of about 10° so moves leaving the box are seen. Then pass `--box` and `--data` together to every step; a box outside the data file stops with a message. The Gulf Stream run:

```bash
python scripts/00_fetch_agulhas_subset.py --fetch-box -90 -30 15 55 --out data/gulfstream_gdp1h_subset.nc
python scripts/03_transition_matrix.py --box -80 -40 25 45 --data data/gulfstream_gdp1h_subset.nc
```

`01`–`04`, `06`, `07` and `10`–`22` run this way, in the order of the list above. `04` picks release points by traffic alone outside R, since its named ports are all in R. `05` searches around the Agulhas core only, and `23`–`25` stop with a message outside the Agulhas area, because the named coasts and routes describe that region. `08` and `09` keep separate wind files for each subset, and need ERA5 for the extra years and area. Output names carry the box, though figure names still start with `agulhas_`.

| Script | What it does | Output |
|---|---|---|
| `00_fetch_agulhas_subset.py` | Streams GDP hourly data chunk by chunk, keeps 5°W–65°E, 60–5°S (any candidate box plus a margin) at hourly resolution, drops positions with gaps > 6 h, and keeps each drifter's GDP fate (`typedeath`, `end_date`); a chunk that keeps failing to read stops the run rather than dropping its drifters; `--fetch-box` and `--out` save another region to its own file; `--metadata-only` refreshes those fields on an existing subset | `data/agulhas_gdp1h_subset.nc` |
| `01_agulhas_gdp_exploration.py` | Observation and drifter counts in R, drogued vs undrogued, coverage map, observations per year | `figures/agulhas_gdp_coverage_{box}.png`, `figures/agulhas_gdp_temporal_{box}.png` |
| `02_grid_coverage.py` | Distinct drifters per cell at each grid size (default 0.5°, 1°, 2°), used to choose the grid | `figures/agulhas_grid_coverage_{grids}deg_{box}.png`, `data/grid_coverage_{grids}deg_{box}.npz` |
| `03_transition_matrix.py` | Builds transition matrices for drogued and undrogued drifters at every grid and lag given (default 0.5°, 1°, 2°; τ = 3.5 days), with exit states for each edge and a stranded state from drifters that ran aground in R, then runs a test release | `data/P_{drogued,undrogued}_{res}deg_{tau}d_{box}.npz`, e.g. `data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz` |
| `04_shipping_overlay.py` | Overlays commercial shipping density (World Bank / IMF AIS, 2015–2021) on the mean current from each drogue type, picks candidate release points (busiest lanes + known hotspots) in cells reliable in both matrices, and ranks them by near-shore exposure under the drogued and the undrogued matrix. Needs the shipping zip, see the run list above. | `figures/agulhas_shipping_currents_{drogued,undrogued}_{tag}.png`, `data/release_candidates_{tag}.csv`, `data/shipping_{tag}.npz` |
| `05_box_selection.py` | Scores every candidate bounding box containing the Agulhas core on undrogued data volume, reliable-cell share, shipping density and oil retention; reports the Pareto set, a recommended box and a side-by-side of named options | `figures/agulhas_box_selection.png`, `data/box_pareto.csv`, `data/box_options.csv` |
| `06_bootstrap.py` | Resamples whole drifters 1,000 times per drogue type, rebuilds the matrix each time and re-propagates the release points from `04`: 95% intervals on each point's fate (still in R, exits, stranded, 30-day coastal exposure), whether drogued and undrogued differ, and how stable each row is (median bootstrap TVD) | `data/bootstrap_release_*.csv`, `data/bootstrap_difference_*.csv`, `data/bootstrap_rows_*.npz`, `figures/agulhas_bootstrap_fates_*.png`, `figures/agulhas_bootstrap_rows_*.png` |
| `07_wind_slip.py` | Undrogued − drogued mean velocity per cell (the extra surface wind drift undrogued drifters carry), latitude-band means with 95% intervals from resampling whole drifters | `figures/agulhas_wind_slip_{res}deg_{box}.png`, `data/wind_slip_cells_…csv`, `data/wind_slip_bands_…csv` |
| `08_fetch_era5.py` | Requests ERA5 10 m winds from the Copernicus CDS API one year at a time, interpolates them to every drifter position and deletes the raw year; needs a CDS account, the ERA5 licence and `~/.cdsapirc` | `data/drifter_wind10m.nc` (another subset gets its own file and year folder, `config.wind_paths`) |
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
| `19_seasonal_matrices.py` | Builds separate DJF, MAM, JJA and SON transition matrices for both drogue types, assigning each move to the season of its start observation. Reports seasonal row support, TVD against the annual operator, pairwise seasonal TVD on common reliable cells, a permutation test of that TVD (season labels shuffled between blocks of one drifter's moves in one season of one year), and 30-day release-point sensitivity. The 2° run is used as a robustness check where 1° drogued seasonal support is sparse. | `data/P_{drogued,undrogued}_{DJF,MAM,JJA,SON}_…npz`, `data/seasonal_support_…csv`, `data/seasonal_tvd_cells_…csv`, `data/seasonal_pairwise_tvd_…csv`, `data/seasonal_permutation_…csv`, `data/seasonal_release_30d_…csv`, `figures/agulhas_seasonal_summary_…png` |
| `20_chapman_kolmogorov.py` | Chapman–Kolmogorov consistency check: compares two applications of the τ operator, `P(τ)^2`, with an operator estimated directly from drifter pairs at `2τ`. Reports row-wise TVD over cells reliable at both lags, descriptive shares below TVD 0.25 and 0.50, and differences in in-region and absorbing-state probabilities. Resamples whole drifters 1,000 times to give each direct `2τ` row's sampling noise, and reports the share of rows whose TVD exceeds the 95th percentile of that noise. For the baseline τ = 3.5 d, the direct 7-day matrices from `03` are required. | `data/chapman_kolmogorov_summary_{tag}.csv`, `data/chapman_kolmogorov_cells_{tag}.csv`, `figures/agulhas_chapman_kolmogorov_{tag}.png`, `figures/agulhas_chapman_kolmogorov_noise_{tag}.png` |
| `21_coastal_pooling.py` | Pools neighbouring thin coastal cells into short contiguous segments without modifying the main transition matrix. For this script only, the existing coastal-cell candidates are conservatively filtered using Natural Earth 50m land polygons: mapped land must intersect the candidate cell or one of its eight neighbouring cells. At the baseline setting (1°, τ = 3.5 days), this removes four offshore candidates, leaving 99 coastal cells grouped into 44 segments. Each segment contains at most four neighbouring cells and is evaluated using distinct-drifter support. Segments below the reliability threshold are retained and flagged. The filter does not add potentially missing coastal cells. Scripts `04`, `06`, `09` and `12` retain the original coastal-cell definition. The pooled segments support coastal summaries and future traffic-weighted coastal-risk analysis. | `data/coastal_pooling_cells_{tag}.csv`, `data/coastal_pooling_segments_{tag}.csv`, `data/coastal_pooling_summary_{tag}.csv`, `data/coastal_segments_{tag}.npz`, `figures/agulhas_coastal_pooling_{tag}.png` |
| `22_drogue_loss.py` | For drifters whose record ends in R, the time from drogue loss to the end of the record, for drifters that ran aground against drifters that stopped transmitting; counts each drogue type's stranding moves when a stranding move takes the drogue status of its start (the rule every matrix build uses, `transport.same_drogue`) and when both ends must match | `data/drogue_loss_{tau}d_{box}.csv`, `data/drogue_loss_summary_…csv`, `data/drogue_loss_stranding_moves_…csv`, `figures/agulhas_drogue_loss_…png` |
| `23_coastal_stranding.py` | Splits each operator's stranded state into six named coasts by where drifters ran aground, checks that the split reproduces the saved operator, and reports stranding on each coast (eventually and within 7, 30 and 365 days) and expected days to strand, for every cell and the release points from `04`, with 1,000 resamples of whole drifters | `data/P_{drogued,undrogued,oil,oil_check}_coast_…npz`, `data/coastal_stranding_points_…csv`, `data/coastal_stranding_cells_…csv`, `data/coastal_stranding_support_…csv`, `data/coast_counts_…npz`, `figures/agulhas_coastal_stranding_…png` |
| `24_traffic_exposure.py` | Spreads releases over R by shipping density, for all traffic and six routes: stranding by coast, days to strand, near-shore days, each route's share of each coast's stranding and oil's mass stranded, with 1,000 resamples of whole drifters; sensitivity runs with lane-pixel weights and without the weight moved off unreliable cells. Needs the shipping zip. | `data/traffic_exposure_…csv`, `data/traffic_nearshore_segments_…csv`, `data/traffic_weights_…csv`, `figures/agulhas_traffic_exposure_…png` |
| `25_risk_atlas.py` | Release-risk atlas: for every ocean cell, the share afloat in R, exited across each edge and stranded on each coast at 7, 30 and 365 days and eventually, and days to strand, with 95% intervals and each cell's reliability flags | `data/risk_atlas_…csv`, `figures/agulhas_risk_atlas_…png` |
| `config.py` | Default region, grids, lag and data path; command-line options; matrix file names | — |
| `transport.py` | Helpers to load and use a saved matrix (release, propagation, cell centres, reliable cells, exact residence time and fate from the absorbing chain), plus the pairing, counting and drifter-bootstrap helpers shared by every script that builds a matrix | — |

`notebooks/01_agulhas_gdp_exploration.ipynb` is not part of the pipeline and does not run on this setup; use the scripts.

---
## Whole-box results

At 1°, τ = 3.5 days, over reliable cells (≥ 10 drifters), drogued / undrogued unless stated. Scripts `11`–`17` and `19`–`21`; seasonal robustness also uses 2°.

- **Data support.** 78% of active drogued rows and 93% of undrogued rows are reliable at 1° (52% / 91% at 0.5°, 90% / 94% at 2°). Coastal cells are much thinner: 33% / 47% reliable at 1°, against 82% / 98% offshore, so coastal and stranding results carry wide intervals.
- **Retention hotspots.** Drogued material is held longest at 41–44°E, 30–33°S, south-west of Madagascar: about 63–65% is still in R after a year (95% intervals about 48–74%), and 70 of the 75 hotspot cells stay in the top 10% in at least half of 1,000 drifter resamples. Undrogued material is held longest further east, at 48–51°E, 30–33°S (55–56%, intervals 49–60%). Where both grids are reliable, the 1° and 2° hotspots agree (Jaccard 76% drogued, 73% undrogued); 0.5° agrees less for drogued (37–41%), whose rows are thin. Retention after a year also reflects distance from the open edges of R, so a hotspot is a place R holds material, not necessarily a closed eddy.
- **Residence time.** Material stays afloat in R for 223 / 203 days on average (median 223 / 213), with the longest stays in the hotspots (up to 518 / 459 days). The grids agree within about 40 days.
- **Exit zones.** Most material eventually leaves east with the Return Current: 69% / 62% on average, and east is the most likely fate in 83% / 75% of cells. The west (Atlantic leakage) takes 18% / 17%, the most likely fate in a strip along the west coast and the retroflection. 15% of undrogued material strands, and stranding is the most likely undrogued fate along the Mozambique Channel coast and east of Madagascar; drogued stranding is 4%, a lower bound that comes from the 18 drifters still drogued when they reached the coast (see `22`).
- **Almost-invariant regions.** Four regions for both drogue types: the retroflection and the Atlantic side, the Natal coast and Mozambique Channel, south of Madagascar and the Return Current, and east of Madagascar. They keep 81–94% of their material over one step and 54–70% over 30 days, and by a year only 1–8% remains in its own region. Three of the four keep 13–23 points more over 30 days than longitude bands of the same sizes; the western one matches its band, whose edge falls close to the retroflection anyway. The leading eigenvalues fall off smoothly, so four is a choice rather than a gap in the spectrum.
- **Release points.** After a year, 18% (13–24%) of drogued material from Durban is still in R, 49% has left east, 19% west and 10% has stranded, the expected behaviour. Undrogued material from the Mozambique Channel lane (40.5°E, 16.5°S) and the lanes south of Madagascar strands far more: 24–66% after a year, against 1–27% drogued. Along the South African coast drogued and undrogued stranding are alike (9–19% against 10–38%), since most of the drifters that reached the coast drogued grounded there.
- **Seasonality.** Four seasonal operators (DJF, MAM, JJA and SON) are compared with each other and with a permutation test that shuffles season labels between blocks of one drifter's moves in one season of one year. At 1°, undrogued seasonal support remains good (70–75% of ocean rows reliable), with 551 of 970 cells reliable in all four seasons; pairwise median TVD on those common cells is 0.35–0.38. Drogued seasonal coverage is much thinner at 1° (12–25% reliable; only 26 cells reliable in all four seasons), so those differences are treated cautiously. At 2°, drogued support improves to 41–51% per season and 72 of 263 cells are reliable in all four seasons, while broad seasonal differences remain (pairwise median TVD 0.20–0.24). Undrogued support at 2° is 84–88%, with pairwise median TVD 0.15–0.16. Against shuffled seasons, drogued transport shows no seasonal change beyond sampling noise (no season pair at 1°, one of six at 2°), while undrogued transport does change with season (4 of 6 pairs at 1°, 5 of 6 at 2°, p ≤ 0.05), consistent with seasonal winds acting through windage, but only by 0.01–0.02 TVD above the shuffled level. The annual operator therefore stays the baseline.
- **Markov consistency.** A Chapman–Kolmogorov check compares `P(3.5 d)^2` with transitions estimated directly at 7 days. Over cells reliable at both lags, median row-wise TVD is 0.243 for drogued and 0.173 for undrogued drifters (731 / 892 common reliable cells); the 90th percentiles are 0.374 / 0.261. Nearly all of it is sampling noise in the direct 7-day rows: resampling whole drifters puts the noise at 0.224 / 0.169 per row, and only 1.8% / 3.7% of rows exceed the 95th percentile of their own noise, about the share expected by chance. Within sampling noise the 3.5-day operator is Markov; the drogued rows differ more only because their data are thinner.
- **Coastal pooling.** At 1°, τ = 3.5 days, a conservative Natural Earth 50m land-proximity filter removes four offshore candidates from the original 103 coastal cells, leaving 99 cells grouped into 44 contiguous coastal segments, with at most four neighbouring cells per segment. Before pooling, 35.0% of active drogued coastal rows and 48.9% of active undrogued coastal rows in this filtered set were reliable (≥ 10 distinct drifters). After pooling, 60.6% / 72.7% of the 99 coastal cells belong to a reliable pooled segment for drogued / undrogued drifters, and 29 of 44 segments (65.9%) are reliable for both types. The median segment contains two coastal cells; 15 segments remain thin and explicitly flagged. The geographic filter removes unsupported candidates but does not add potentially missing near-land cells. The before/after percentages use different denominators and do not demonstrate improved predictive accuracy. Pooling is used for coastal-segment summaries and future coastal-risk analysis only; it does not alter or smooth the main transition matrix.
- **Stranding by coast.** Stranding is split into six named coasts by where drifters ran aground. Material travels far before grounding: from the lanes south and east of Madagascar, the largest share of oil strands on southern Mozambique (14–22%), not on Madagascar. The added windage brings oil ashore sooner rather than much more often: from the 47.5°E lane, oil strands 49% (42–55%) in 95 days (81–111) and undrogued material 39% (32–47%) in 151 days (127–178). Every drogued coast rests on fewer than 10 grounded drifters, so drogued results by coast are indicative only and a lower bound.
- **Traffic-weighted exposure.** If releases follow ship traffic, 27% of oil parcels eventually strand, in 87 days (79–97) on average: southern Mozambique 10% (6–12), central and northern Mozambique 8% (5–10), the South African south and east coast 4% (2–6). The Mozambique Channel route is the most dangerous (69% strands, in 38 days), while the Cape route west of Cape Agulhas carries 27% of traffic but strands only 2%, since 97% of its oil leaves into the Atlantic. Half of southern Mozambique's stranding comes from the route south of Madagascar. With a 14-day mass loss, only 3.6% of the released oil mass reaches a coast. 32% of the weight sits on unreliable cells and moves to the nearest reliable one, which changes each coast by under 1 point; weighting by lane pixels lowers the Mozambique coasts by about 3 points but keeps the same three coasts on top, so the ranking is firmer than the percentages.

## Second region: Gulf Stream

The core pipeline also runs on the Gulf Stream, box 80–40°W, 25–45°N, with no region-specific code changes (1°, τ = 3.5 days, drogued / undrogued; details in [Research.md](Research.md)). Retention peaks in the Sargasso Sea at about 58–61°W, 30–33°N, undrogued drifters slip west under the trades and east under the westerlies, and held-out exits and stranding are close to calibrated. Three things differ from R, so settings and skill checked in one region should be rechecked in another:

- **Point accuracy.** At 14 days the operator beats advection in no fold, and at 28 days the drogued operator beats persistence in only 2 of 5 folds. The undrogued operator beats both in all 5 folds at 28 days.
- **Grid.** Mean time in the box falls from 608 / 427 days at 0.5° to 350 / 305 days at 2°, against differences of about 40 days in R, so long-run results here depend on the grid.
- **Seasonality.** Drogued transport is seasonal beyond sampling noise in 5 of 6 season pairs, so the annual matrix is a weaker assumption here.

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

The coast operators from `23` (`data/P_{name}_coast_…npz`) are the same matrices with `stranded` split into six named coasts (`transport.COAST_LABELS`), so they have five more absorbing states; read them through `exit_labels`.

At τ = 3.5 days: 1 week = 2 steps, 1 month ≈ 9 steps, 1 year ≈ 104 steps.

## Status

- [x] Data subset, exploration, grid coverage
- [x] Time step: τ = 3.5 days (the velocity decorrelation time over the core region 10–40°E, 45–25°S is about 1.2–1.4 days)
- [x] Transition matrices (drogued and undrogued)
- [x] Region, grid and lag are parameters; matrices at 0.5°, 1° and 2° for both drogue types
- [x] Shipping lanes over currents; candidate release points chosen and ranked
- [x] Stranded state from GDP's record of drifters that ran aground in R (139 drifters, 18 of them drogued when they reached the coast); a stranding move takes the drogue status of its start, since the drogue flag often flips in a record's final hours (`22`)
- [x] Drifter bootstrap: 95% intervals on release-point fates, drogued vs undrogued differences, row stability
- [x] Release-point maps at 1 week, 1 month, 1 year
- [x] Held-out validation (5-fold by drifter): calibrated exits and stranding, beats persistence and mean-current advection by 28 days
- [x] Sensitivity to τ and grid size: τ = 2–5 days gives the same results; 1° is used (0.5° adds no skill and has 50% thin drogued rows, 2° loses short-range skill and shifts one-year fates)
- [x] Seasonal matrices, with a permutation test: no drogued seasonality beyond sampling noise, a small but real undrogued one
- [x] Chapman–Kolmogorov check: `P(3.5 d)^2` against direct 7-day transitions; median row TVD 0.243 / 0.173 for drogued / undrogued, almost all sampling noise (1.8% / 3.7% of rows above their 95% noise level)
- [x] Coastal pooling: Natural Earth 50m conservative geographic filter reduces 103 candidates to 99 coastal cells, grouped into 44 short segments; 29 segments are reliable for both drogue types, while 15 remain thin and flagged
- [x] ERA5 winds at every drifter position; oil matrix = undrogued + the missing 1.5% windage (undrogued carries 2.0%), with a drogued + 3.5% check
- [x] Whole-box analysis: accumulation zones, exit zones, residence time, almost-invariant regions
- [x] Stranding by coast: six named coasts, with intervals and days to strand, for drogued, undrogued, oil and the oil check
- [x] Traffic-weighted exposure by coast and route, with sensitivity to the weights
- [x] Release-risk atlas: every release cell and horizon, with intervals
- [x] Second region: the core pipeline on the Gulf Stream box, with no region-specific code changes
