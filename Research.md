Something goes into the water at a chosen point in R: a container off a ship or a slick of oil. Where has the surface flow taken it a week later? A month later? A year? At the scale of the whole box, which parts of R feed which others, and which are cut off from the rest?

Product. A surface-transport operator for R: transition matrices that someone else can load and iterate forward, together with what they say about where chosen release points end up, and about where R as a whole gathers material and where it loses it.


Region:
Box R: **10°E–55°E, 45°S–15°S** (`BOX` in `agulhas/config.py`)

Why the Agulhas:
- The Agulhas Current is one of the strongest western boundary currents (up to ~2 m/s), flowing south-west along the South African east coast.
- At the retroflection south of Africa (~16–20°E) most of the flow turns back east as the Agulhas Return Current (~38–40°S); some leaks into the South Atlantic via Agulhas rings. This gives distinct "feed" and "cut-off" structure for the whole-box question.
- The shipping route around the Cape of Good Hope is busy (more so since the 2024 Red Sea diversions), so container loss and oil spills are realistic scenarios.
- GDP drifters sample it well: 5.95M hourly observations from 1,503 drifters in R.

Why this box (`agulhas/exploration/box_selection.py` scores boxes on data, shipping risk and retention):
- Oil-like undrogued transport stays in R 136 days on average; 6.5% is still inside after a year in the box-selection calculation.
- Contains 97% of the Cape Basin eddy energy (Agulhas rings) and the Mozambique Channel eddies.
- Covers the coastal route around the Cape, the Mozambique Channel routes and the route south of Madagascar.
- Passes the data rules with margin: 4.12M undrogued hourly observations, 92% of ocean cells seen by ≥ 10 drifters.


What we track:

Shipping container:
- Modelled with drogued drifters, which follow ~15 m currents with almost no wind effect (windage), matching a mostly submerged or flooded container.

Oil spill:
- Surface oil drifts at roughly current + 3–3.5% of wind speed.
- Modelled with undrogued drifters, which ride at the surface and slip downwind of drogued drifters by ~10–12 cm/s in the westerly and trade-wind bands (`agulhas/exploration/wind_slip.py`). Against ERA5 winds that slip is 2.0% of the 10 m wind (`agulhas/operators/oil_wind.py`), a little over half of oil's windage.
- ERA5 10 m winds (Copernicus Climate Data Store, cds.climate.copernicus.eu) supply the wind term. Undrogued motion already contains 2.0% windage, so the oil matrix adds only the missing 1.5% (3.5% − 2.0%) along each drifter's path; adding the full 3.5% would count the wind twice. A check matrix (drogued + 3.5% wind) tests that choice.
- The measured 2.0% is everything that moves the surface differently from 15 m: direct windage on the drifter, wave-driven Stokes drift and wind-driven shear in the top 15 m. Subtracting it from 3.5% therefore treats oil's 3.5% as drift relative to the 15 m current. If the 3.5% is instead read as drift relative to the true surface current, the oil matrix adds too little wind, and the drogued + 3.5% check bounds that case.
- The model represents transport of surface oil parcels only. Weathering, evaporation, emulsification, dissolution and sinking are not modelled, so results describe where surface material is transported rather than how much oil mass remains.

Drogued and undrogued matrices are built and reported side by side and never pooled: an "all drifters" matrix is 46–84% undrogued depending on the cell, so its wind effect would follow sampling, not physics.


Data:
- NOAA GDP hourly drifter data (v2.01, 1995–2022 in R), streamed from the public AWS S3 zarr store (the CloudDrift `gdp1h` source) by `agulhas/sources/fetch_gdp_subset.py` into `data/agulhas_gdp1h_subset.nc`.
  - Native hourly resolution; positions interpolated across > 6 h between satellite fixes are dropped (0.3%).
  - The saved area (5°W–65°E, 60–5°S) covers R plus a margin, so positions just after a drifter leaves R are kept.
  - Key fields: `lon`, `lat`, `time`, `ve`/`vn` (velocity), `drogue_status`, `gap`, `rowsize`, `id`.
  - Split by `drogue_status`: drogued → container matrix; undrogued → oil-like surface matrix.
- Commercial shipping density: World Bank / IMF Global Shipping Traffic Density, Commercial layer (AIS, Jan 2015–Feb 2021, CC-BY 4.0).
- ERA5 10 m wind (u10, v10), 6-hourly on a 0.5° grid, R plus a 5° margin, for the oil wind term.


Method (Ulam's method / transition matrix):
1. Grid R into cells at 0.5°, 1° and 2°; 1° is the working grid. Land cells are masked.
2. Transition time τ = 3.5 days, longer than the Lagrangian decorrelation time (integral timescale 1.40 d drogued, 95% interval 1.34–1.47, and 1.20 d undrogued, 1.16–1.23, over 10–40°E, 45–25°S; `agulhas/exploration/decorrelation_time.py`); drifters move about one cell per step. The Markov approximation is checked separately with a Chapman–Kolmogorov comparison between `P(τ)^2` and transitions estimated directly at `2τ`.
3. For every drifter position at time t in cell i, find where it is at t + τ (cell j). Count transitions C[i, j].
4. Five absorbing states: four "exited R" states (W, E, S, N) record which edge material leaves by, and a "stranded" state records material that runs aground inside R. A drifter strands when GDP records its death as "ran aground" and its last observation lies in R; every position in its final τ moves to the stranded state. A stranding move takes the drogue status of its start, since the drogue flag often flips to undrogued in a record's final hours.
5. Row-normalise: P[i, j] = C[i, j] / Σ_j C[i, j]. Each row is "where does material in cell i go in τ days". A cell with no outgoing moves of its own (its drifters' records ended there for other reasons, such as failed transmitters) takes the pooled moves of the nearest ring of ocean cells around it.
6. Iterate: distribution after n steps = p₀ · Pⁿ. 1 week = 2 steps, 1 month ≈ 9 steps, 1 year ≈ 104 steps.
7. Oil version: each undrogued move's end point is shifted by the missing windage (3.5% − the measured undrogued 2.0%) × the mean ERA5 wind along that move × τ, and the shifted moves are counted as in step 3. Moves into the stranded state are kept as they are. The matrix conserves probability and models transport only; oil weathering and mass loss are outside the scope of the model.


Reliability:
- Rows built from < 10 distinct drifters are flagged; cells with no outgoing moves of their own (rows pooled from neighbouring cells) are flagged separately.
- `agulhas/validation/bootstrap.py` resamples whole drifters (never observations) to give 95% intervals for release-point results and a row-stability check (median bootstrap TVD > 0.2 = unstable).
- Hourly positions are strongly correlated; the independent sample size is the number of drifters, not the observation count.
- Coastal data are much thinner than offshore data. `agulhas/coast/coastal_pooling.py` therefore groups neighbouring coastal cells into short contiguous segments for coastal summaries, while leaving the main transition matrix unchanged. Segments that remain thin are retained and flagged.
- At 1 year most material has left R, so answers are "probability still in R" plus where it exited or stranded.


Analysis:

Release points: 5 hotspots plus the busiest reliable shipping cells (`agulhas/release_points/shipping_overlay.py`):
- Richards Bay, Durban (major ports, in the core of the current)
- Algoa Bay (ship-to-ship bunkering, off Gqeberha)
- Cape Town, Saldanha Bay (Atlantic side)
- Busiest shipping cells along the coastal route, in the Mozambique Channel and south of Madagascar

For each: map of the probability distribution after 1 week, 1 month and 1 year, plus the fraction left in R, where it exited and how much stranded, with bootstrap intervals.

Whole-box structure:
- Where R holds material: the share released in each cell still afloat in R after a year; the top 10% of reliable cells are retention hotspots, checked across grids and with a drifter bootstrap.
- Where R loses material: residence time (expected time before leaving or stranding) and the eventual fate (which edge, or stranded) of each cell, both exact from the absorbing chain's fundamental matrix N = (I − Q)⁻¹.
- Which parts keep material to themselves: spectral clustering of the symmetrised transition matrix → almost-invariant regions, checked with the directional P against longitude bands of the same sizes.
- Source vs sink cells: every row of P sums to 1, so the column sums show which cells gather or shed material over one step (`agulhas/whole_box/source_sink.py`).
- Coastal support: neighbouring thin coastal cells are grouped into short coastal segments for coastal summaries and near-shore exposure.
- Stranding by coast: the stranded state is split into six named coasts by where drifters ran aground, so stranding can be reported per coast with intervals.
- Traffic-weighted exposure: releases spread over R in proportion to shipping density, for all traffic and six routes; which coasts are exposed, from which routes and how soon.
- Release-risk atlas: every ocean cell as a release cell, with stranding, exit and retention probabilities at each horizon and 95% intervals.


Validation:
- Held-out drifters (whole trajectories): their actual positions after 1 week / 1 month against the matrix predictions.
- Held-out fates out to one year, allowing for records that end early (`agulhas/validation/held_out_year.py`).
- Sensitivity to grid size (0.5°, 1°, 2°) and τ (2 and 5 days).
- Four seasonal operators (DJF, MAM, JJA and SON) against the annual operator, with 2° used as a robustness check where 1° drogued seasonal support is sparse.
- Chapman–Kolmogorov consistency: compare `P(3.5 d)^2` with transitions estimated directly at 7 days.
- Drifter coverage over time (temporal plot), so no few years dominate the matrix.
- Known features: current path, retroflection, rings heading into the Atlantic, set against ranges fixed before the script first ran (`agulhas/validation/known_features.py`).


Final output:
- Transition matrices saved as `data/P_{drogued|undrogued}_{res}deg_{tau}d_{box}.npz` (P, counts, grid edges, τ, flags), with a usage example in the README.
- Two working transport versions: container (drogued) and oil (undrogued plus the missing ERA5 windage).
- Maps for the release points at 1 week / 1 month / 1 year.
- Maps of accumulation zones, exit zones, residence time and almost-invariant regions.
- Seasonal transport operators and seasonal sensitivity summaries.
- Chapman–Kolmogorov Markov-consistency diagnostics.
- Pooled coastal segments for coastal support summaries.
- Coast operators, stranding by coast from every cell, and traffic-weighted exposure by coast and route.
- A release-risk atlas: outcomes for every release cell and horizon, with 95% intervals.


Progress:
- [x] Data subset — `agulhas/sources/fetch_gdp_subset.py`: 17.0M hourly positions from 2,942 drifters in the saved area; in R 5.95M (1.83M drogued, 4.12M undrogued) from 1,503 drifters.
- [x] Exploration and grid coverage — `agulhas/exploration/gdp_exploration.py`, `agulhas/exploration/grid_coverage.py`.
  - Ocean cells in R: 3,780 / 970 / 263 at 0.5° / 1° / 2° (the 45°-wide box at 2° ends in a partial 1° column).
  - Cells with ≥ 10 distinct drifters: drogued 52% / 78% / 90%, undrogued 90% / 92% / 93%.
  - Drogued data is thin on the Agulhas Bank shelf, off Namibia and in parts of the Mozambique Channel.

- [x] Region, grid and lag are parameters (`agulhas/config.py`, `--box --res --tau --data`); matrix file names include res, τ and box. Another region gets its own subset from `fetch_gdp_subset` (see the second-region step below).

- [x] Transition matrices, both drogue types, 0.5° / 1° / 2° — `agulhas/operators/transition_matrix.py`, helpers in `agulhas/transport.py`.
  - Transitions: 1.77M drogued, 4.07M undrogued.
  - Stranding: 139 drifters ran aground in R (GDP `typedeath` 1, last observation in R and within a day of the record's end date). A move into the stranded state counts for the drogue status at its start (`transport.same_drogue`).
  - Drogue loss before grounding — `agulhas/exploration/drogue_loss.py`: of the 139, 2 were still drogued at the last fix and 16 lost the drogue in their final day (median 4 km from the grounding point; none between 1 and 3.5 days before it). The other 121 were undrogued for more than τ before grounding: 63 throughout their time in the saved area and 58 for a median of 172 days. Drifters that stopped transmitting in R lose the drogue in their final day as often (17%, against 12% for groundings), so the late flip marks the end of a record rather than the grounding, and these 18 drifters reached the coast drogued. The drogued matrix has 1,240 moves into the stranded state from 18 drifters, the undrogued matrix 10,159 from 137 (the 16 late-flip drifters contribute to both: drogued starts before the flip, undrogued after). Most grounded drifters lost their drogue long before, so drogued stranding is a lower bound.
  - Flagged rows (< 10 drifters): drogued 50% / 24% / 11%, undrogued 11% / 8% / 7%. At 1°, 20 (drogued) and 8 (undrogued) cells have no outgoing moves and take their rows from neighbouring cells.
  - Per step at 1°: exit 2.4% drogued, 3.5% undrogued; stranding 0.23% drogued (from 29 cells), 1.95% undrogued (from 89 cells) (mean over cells).
  - Release off Durban (31.5°E, 30.5°S), 1°, after 1 year: drogued 18% still in R, 49% exited E, 18% W, 11% stranded; undrogued 17% in R, 56% E, 19% W, 6% stranded. Across grids 16–20% stays in R, 45–57% exits east, 15–22% west.

- [x] Release points from shipping traffic — `agulhas/release_points/shipping_overlay.py`.
  - Shipping values are a relative density only (units undocumented and implausibly large; pixel coverage drops about 8-fold at exactly 40°S, a source artifact). The data predate the 2024 Red Sea diversions, so Cape traffic is higher today.
  - Mean surface current from drifters (1° cells with ≥ 300 hourly obs): max 1.23 m/s drogued, 1.34 m/s undrogued, in the Agulhas Current.
  - Candidates sit in cells reliable in both matrices (≥ 10 drifters, has outgoing data). Algoa Bay, Cape Town and Saldanha Bay sit 62–76 km offshore in the nearest reliable cell, since their own cells have too few drifters.
  - Near-shore exposure over 30 days (material-days afloat in ocean cells next to land; stranding is reported separately, by `bootstrap`):

    | Release point | Drogued | Undrogued |
    |---|---|---|
    | Saldanha Bay | 11.0 | 7.7 |
    | Cape Town | 9.5 | 6.7 |
    | Algoa Bay | 7.5 | 2.3 |
    | Richards Bay | 7.3 | 6.6 |
    | Durban | 6.7 | 4.5 |
    | Busiest lanes | 0.0–5.1 | 0.4–5.3 |

  - Algoa Bay's offshore cell sits in the Agulhas Current, so its undrogued score describes oil released there, not in the bay itself.

- [x] Box scoring — `agulhas/exploration/box_selection.py` (undrogued data): boxes containing the Agulhas core (18–33°E, 40–28°S), ≥ 5° inside the saved area so exits stay observable; data rules ≥ 1M undrogued hourly observations and ≥ 90% of ocean cells with ≥ 10 drifters; scored on shipping density per ocean cell and one-step retention → mean residence time. Box R's scores are listed under "Why this box". Pixel coverage in the shipping raster drops about 8-fold at exactly 40°S, so shipping density south of that line is less trustworthy.

- [x] Drifter bootstrap — `agulhas/validation/bootstrap.py` (1°, τ = 3.5 d, 1,000 resamples of whole drifters: 800 drogued, 1,100 undrogued; the two types are resampled independently).
  - One-year fate with 95% intervals (drogued / undrogued):

    | Release point | Still in R | Exited west (Atlantic) | Exited east | Stranded |
    |---|---|---|---|---|
    | Durban | 18% (13–24) / 11% (6–17) | 19% (13–25) / 14% (7–23) | 49% (41–56) / 37% (21–58) | 10% (4–18) / 38% (5–64) |
    | Richards Bay | 22% (15–31) / 16% (13–20) | 17% (12–23) / 19% (14–24) | 45% (38–54) / 53% (47–59) | 11% (4–19) / 10% (5–18) |
    | Algoa Bay | 13% (9–18) / 13% (10–15) | 25% (17–33) / 27% (21–34) | 41% (32–51) / 47% (40–54) | 15% (5–28) / 11% (5–18) |
    | Cape Town | 2% (1–4) / 0% (0–1) | 86% (69–97) / 62% (42–84) | 2% (1–4) / 1% (0–3) | 9% (0–27) / 36% (14–57) |
    | Saldanha Bay | 1% (0–1) / 0% (0–0) | 99% (97–100) / 97% (92–100) | 0% (0–1) / 0% (0–0) | 0% (0–0) / 3% (0–8) |

  - Offshore, undrogued material strands far more than drogued material after a year: 24–39% against 1–9% at the lanes south of Madagascar, and 66% (50–82%) against 27% (10–47%) at the Mozambique Channel lane (40.5°E, 16.5°S); the intervals do not overlap at 7 of the 8 lanes (all but 27.5°E). Along the South African coast the two are alike: drogued 10–19% at Durban, Richards Bay, Algoa Bay and the 27.5°E lane against undrogued 10–38%, with overlapping intervals. Of the 18 drifters that reached the coast drogued, 11 grounded on the South African south and east coast (19–29°E), 4 in southern Mozambique, 2 on Madagascar and 1 in Namibia, so drogued stranding is concentrated there. The drogued figure is a lower bound (most grounded drifters lost their drogue long before grounding, see `transition_matrix`).
  - Exits agree between the versions along the South African coast (Algoa Bay, Richards Bay, Durban, the 27.5°E lane): the 95% interval of the difference includes 0 for still in R and for both exit edges at all four. At the eastern lanes, drogued material stays in R more at 43–48°E (45–57% vs 27–32%) and leaves east more at 47–55°E (46–78% vs 23–43%); undrogued material strands on Madagascar instead.
  - Durban's and Cape Town's undrogued stranding intervals are wide (5–64%, 14–57%), since their coastal rows rest on few stranded drifters.
  - The intervals are not corrected for the number of points and metrics compared, so a single "differ" flag is weak evidence on its own.
  - Row stability (median bootstrap TVD > 0.2 = unstable): drogued 258 of 950 rows (27%), undrogued 54 of 962 (6%). Unstable drogued rows are concentrated in the Mozambique Channel and south of Madagascar. The two reliability checks pick different rows (drogued: 174 unstable rows pass the 10-drifter flag, 124 flagged rows are stable), so both are reported. The TVD understates the uncertainty of rows with very few drifters (a single-drifter row is either resampled whole or dropped), so the 10-drifter flag is the floor.
  - Outputs: `data/bootstrap_release_1deg_3p5d_10E-55E_45S-15S.csv`, `data/bootstrap_difference_…csv`, `data/bootstrap_rows_…npz`, `figures/agulhas_bootstrap_fates_…png`, `figures/agulhas_bootstrap_rows_…png`.

- [x] Undrogued wind slip — `agulhas/exploration/wind_slip.py` (1°, undrogued − drogued mean velocity, per-drifter means so no drifter dominates a cell; 746 cells with ≥ 10 drifters of each type; 95% intervals from 500 resamples of whole drifters).

    | Band | Cells | East (cm/s) | North (cm/s) | Speed | Points toward |
    |---|---|---|---|---|---|
    | 40–45°S | 219 | 11.1 (9.5, 12.6) | 0.1 (−0.6, 0.9) | 11.1 | 90° (east) |
    | 35–40°S | 199 | 1.6 (−0.5, 4.2) | 3.1 (1.5, 4.7) | 3.5 | 28° |
    | 30–35°S | 124 | −4.4 (−7.4, −1.9) | 1.4 (−0.3, 4.2) | 4.6 | 287° |
    | 25–30°S | 112 | −10.2 (−13.1, −7.1) | 3.0 (0.4, 5.7) | 10.6 | 287° (WNW) |
    | 20–25°S | 47 | −11.4 (−16.0, −7.6) | 4.2 (0.9, 9.7) | 12.1 | 290° (WNW) |
    | 15–20°S | 45 | −5.2 (−9.1, −0.7) | 11.2 (3.9, 15.3) | 12.3 | 335° |

  - The slip follows the winds: eastward under the westerlies south of ~38°S, west-north-westward under the south-east trades at 20–33°S, northward along the west coast. It is weakest at 30–40°S, under the subtropical high.
  - Single cells are noisy (drogued and undrogued drifters pass at different times; median per-cell slip 12.9 cm/s against a median drogued current of 21.4 cm/s), so band and block means are the robust result.
  - Outputs: `data/wind_slip_cells_1deg_10E-55E_45S-15S.csv`, `data/wind_slip_bands_…csv`, `figures/agulhas_wind_slip_…png`.

- [x] Release-point maps at 1 week / 1 month / 1 year — `agulhas/release_points/release_point_maps.py` (1°, τ = 3.5 d): one figure per point from `shipping_overlay`, drogued and undrogued, with the share afloat in R, each exit and stranded, and 95% intervals from `bootstrap`.
  - Durban (drogued) after a year: 18% (13–24%) still in R, 49% (41–56%) left east, 19% (13–25%) west and 10% (4–18%) stranded, as expected.
  - Undrogued material strands far more than drogued offshore: after a year 24–66% from the Mozambique Channel lane and the lanes south of Madagascar (drogued 1–27%). Along the South African coast the two are alike (drogued 9–19%, undrogued 10–38%), and Cape Town and Durban's undrogued shares have wide intervals (5–64%), since coastal stranding rests on a few grounded drifters (see the oil version).
  - Outputs: `data/release_point_maps_1deg_3p5d_10E-55E_45S-15S.csv`, `figures/agulhas_release_{nn}_{point}_…png`.

- [x] Held-out validation — `agulhas/validation/held_out.py` (1°, τ = 3.5 d, 5-fold by drifter: 817 drogued / 1,170 undrogued drifters each held out once; daily test starts followed at whole τ steps, exits and stranding absorbing).

    | Type | Horizon | Test starts | Markov error | Advection (cell centre) | Persistence (cell centre) | Log score Markov / climatology | Exit pred / obs | Stranded pred / obs |
    |---|---|---|---|---|---|---|---|---|
    | Drogued | 3.5 d | 73,902 | 73 km | 73 km | 73 km | −2.89 / −6.80 | 3.2% / 3.1% | 0.06% / 0.06% |
    | Drogued | 7 d | 71,664 | 110 km | 113 km | 115 km | −3.09 / −6.73 | 5.9% / 5.9% | 0.12% / 0.12% |
    | Drogued | 28 d | 62,405 | 252 km | 269 km | 295 km | −3.89 / −6.25 | 18.6% / 19.0% | 0.45% / 0.46% |
    | Undrogued | 3.5 d | 169,750 | 82 km | 82 km | 89 km | −2.52 / −6.74 | 2.7% / 2.6% | 0.26% / 0.26% |
    | Undrogued | 7 d | 168,296 | 126 km | 128 km | 142 km | −3.03 / −6.67 | 4.9% / 4.9% | 0.51% / 0.50% |
    | Undrogued | 28 d | 161,075 | 289 km | 303 km | 373 km | −4.04 / −6.19 | 15.2% / 15.6% | 1.93% / 1.97% |

  - Errors are median distances from the forecast (Markov: centre of mass of the predicted distribution still in R) to the actual position. Baselines start from the cell centre, the same information the matrix has; from the exact start position they are ~10 km better at 3.5 d, which is the 1° grid's own error.
  - The matrix ties the baselines at 3.5 d and beats them by 28 d (5–6% vs advection, 15–23% vs persistence). Its main value is the distribution: the log score is far above climatology at every horizon, the predicted share leaving R matches the observed share (calibration within a few points across the full 0–1 range at 28 d), and the predicted share stranding matches what held-out drifters do at every horizon.

- [x] Held-out validation to one year — `agulhas/validation/held_out_year.py` (1°, τ = 3.5 d; the folds and test starts of `held_out`, followed for 104 steps). A fate is censored when a record ends without grounding or a drogued start loses its drogue, and observed shares are Aalen–Johansen estimates, since keeping only known fates over-counts exits: drogued "still in R" at a year would read 4.1% instead of 17.1%. On simulated fates with about two-thirds censored the estimator stays within 0.8 points of the truth. 95% intervals resample whole held-out drifters with the fold matrices fixed.

    | Type | Fate | Day 182, predicted / observed | Day 364, predicted / observed (95%) | Predicted − observed at day 364 (95%) |
    |---|---|---|---|---|
    | Drogued | Still in R | 38.2 / 35.6% | 17.3 / 17.1% (9.5–25.0) | +0.2 (−7.0 to +7.2) |
    | Drogued | Exited east | 37.1 / 39.1% | 53.8 / 55.0% (46.9–63.3) | −1.2 (−8.4 to +6.0) |
    | Drogued | Exited west | 14.9 / 18.2% | 17.6 / 19.6% (15.3–24.3) | −2.1 (−4.9 to +0.6) |
    | Drogued | Stranded | 1.8 / 1.5% | 2.5 / 1.5% (0.6–2.7) | +1.0 (−0.1 to +2.0) |
    | Undrogued | Still in R | 41.0 / 38.6% | 21.0 / 18.8% (15.3–22.6) | +2.2 (−1.3 to +5.2) |
    | Undrogued | Exited east | 36.9 / 41.4% | 51.6 / 55.3% (51.4–59.4) | −3.7 (−7.3 to −0.3) |
    | Undrogued | Exited west | 10.5 / 10.5% | 12.7 / 13.2% (10.7–16.0) | −0.5 (−2.5 to +1.4) |
    | Undrogued | Stranded | 7.4 / 6.8% | 10.2 / 9.7% (7.4–12.3) | +0.5 (−1.7 to +2.7) |

  - Across R the matrices hold up to a year: every fate's predicted share is within 3.7 points of the observed one at day 364, and within 4.4 points at any horizon. The start cell keeps its value, with a multi-class Brier skill against the training folds' mean fate of 0.27 (0.24–0.31) undrogued and 0.47 (0.38–0.58) drogued at day 364.
  - Support falls quickly. In R at day 364 the estimate rests on 105 undrogued drifters but only 14 drogued ones, since most lose the drogue first, so the drogued year is weak evidence; its calibration can be binned only to about day 182, where it holds.
  - Undrogued exits east are under-predicted at every horizon, by 0.7, 2.9, 4.4, 4.3 and 3.7 points at days 28, 91, 182, 273 and 364, with intervals below zero throughout.
  - The highest undrogued retention is overstated: in the top two deciles of predicted "still in R" at day 364, 42% and 51% are predicted against 33% (26–41) and 40% (30–50) observed. Stranding is well calibrated up to about 25%; the top decile reads 49% against 43% (35–51).
  - The largest regional gap is undrogued starts at 50–55°E: 26% against 19% still in R (difference +2.9 to +11.6) and 53% against 64% exited east (−15.2 to −5.6). Bands from 10°E to 50°E agree within their intervals for still in R and stranded.
  - Single release points cannot be tested at a year: at most 11 undrogued and 1 drogued drifters are still followed in R at day 364 from any release cell, and most have 0–4. Their one-year figures rest on the whole-box result, not on a test of their own.
  - Estimating within start-longitude bands or before and after 2010 moves each share by at most 1.8 points. This does not rule out censoring tied to the fate itself. Stranding is judged by the same GDP labels the matrix is built from, so a mislabelled grounding cannot be detected. The comparisons are not corrected for their number, and the oil matrix has no direct test, since no drifter moves as it does.
  - Outputs: `data/validation_year_summary_…csv`, `data/validation_year_support_…csv`, `data/validation_year_calibration_…csv`, `data/validation_year_bands_…csv`, `data/validation_year_points_…csv`, `figures/agulhas_validation_year_…png`.

- [x] Sensitivity to τ and grid size — `agulhas/validation/sensitivity.py`: reference 1°, τ = 3.5 d against τ = 2 and 5 d at 1°, and 0.5° and 2° at τ = 3.5 d, each with its own held-out validation from `held_out`; the 13 release points' one-year fates (still in R, exited W, exited E, stranded) are checked against the reference's 95% bootstrap interval from `bootstrap`.

    | Setting | Flagged rows | 28-day error | Skill vs persistence at 7 / 28 d | Exit pred / obs at 28 d | Stranded pred / obs at 28 d (undrogued) | One-year fates inside the reference interval |
    |---|---|---|---|---|---|---|
    | 1°, τ = 3.5 d | 24% / 8% | 252 / 289 km | −2% / 14%, 8% / 22% | 18.6 / 19.0%, 15.2 / 15.6% | 1.93 / 1.97% | reference |
    | 1°, τ = 2 d | 23% / 8% | 252 / 289 km | −2% / 14%, 8% / 22% | 18.8 / 19.5%, 15.6 / 15.9% | 1.92 / 1.97% | 52 / 52, 52 / 52 |
    | 1°, τ = 5 d | 24% / 8% | 251 / 288 km | −4% / 14%, 8% / 23% | 18.4 / 18.5%, 14.9 / 15.3% | 2.05 / 2.08% (30 d) | 51 / 52, 52 / 52 |
    | 0.5°, τ = 3.5 d | 50% / 11% | 251 / 287 km | 0% / 14%, 11% / 23% | 18.1 / 19.0%, 14.6 / 15.6% | 1.96 / 1.97% | 51 / 52, 48 / 52 |
    | 2°, τ = 3.5 d | 11% / 7% | 262 / 299 km | −19% / 11%, −4% / 20% | 19.4 / 19.0%, 15.8 / 15.6% | 1.85 / 1.97% | 47 / 52, 36 / 52 |

    Pairs are drogued, undrogued. Skill is against persistence from the exact start, a baseline that does not depend on the grid; horizons between whole steps are interpolated linearly.

  - τ does not change the results. From 2 to 5 days the forecast errors move by ≤ 1 km, exit and stranding calibration by < 1 point, and 207 of 208 one-year fates stay inside the reference interval (the exception is drogued stranding at one point at τ = 5 d, 8 points off); 28-day centres of mass move 11–21 km (median), well under one cell. τ = 3.5 d is used for all versions.
  - 1° is used for all versions. 0.5° matches 1° on skill (28-day error within 2 km) and on one-year fates (99 of 104 inside the reference interval; undrogued stranding moves by up to 24 points at single points), but under-predicts exits by about 1 point at 28 days and has 50% flagged drogued rows. 2° loses short-range skill (worse than persistence at 7 days), shifts 28-day centres of mass by 55–82 km (median) and moves one-year fates outside the reference interval at 21 of 104 checks, mostly undrogued.
  - Outputs: `data/sensitivity_skill_10E-55E_45S-15S.csv`, `data/sensitivity_fates_…csv`, `figures/agulhas_sensitivity_…png`, and `data/validation_summary_{res}deg_{tau}d_…csv` for every setting.

- [x] Seasonal matrices — `agulhas/operators/seasonal_matrices.py`.
  - Four operators per drogue type: DJF, MAM, JJA and SON, with each move assigned by the season of its start observation.
  - At 1°, undrogued seasonal support remains good: 70–75% of ocean rows are reliable per season, and 551 of 970 cells are reliable in all four seasons. Pairwise median TVD over common reliable cells is 0.35–0.38.
  - Drogued seasonal support is sparse at 1°: 12–25% of rows are reliable per season and only 26 cells are reliable in all four seasons.
  - At 2°, drogued support improves to 41–51% per season and 72 of 263 cells are reliable in all four seasons; pairwise median TVD remains 0.20–0.24.
  - Undrogued support at 2° is 84–88%, with pairwise median TVD 0.15–0.16.
  - Permutation test (200 shuffles of the season labels of 4,629 blocks, each one drifter's moves in one season of one year; median row TVD per season pair over cells reliable in both seasons):

    | | 1° observed / shuffled | Pairs with p ≤ 0.05 (1°) | 2° observed / shuffled | Pairs with p ≤ 0.05 (2°) |
    |---|---|---|---|---|
    | Drogued | 0.44–0.54 / 0.50–0.54 | 0 of 6 | 0.20–0.24 / 0.20–0.21 | 1 of 6 (DJF–SON) |
    | Undrogued | 0.35–0.37 / 0.35 | 4 of 6 | 0.15–0.16 / 0.14–0.15 | 5 of 6 |

  - Drogued (15 m) transport shows no seasonal change beyond sampling noise: its large pairwise TVD is what four matrices built from a quarter of the data each differ by anyway. Undrogued transport does change with season, consistent with seasonal winds acting through windage, but only by 0.01–0.02 TVD above the shuffled level. DJF–MAM is the one undrogued pair alike at both grids.
  - The p-values are per pair and not corrected for the 24 tests; with 200 shuffles the smallest possible p is 0.005. The annual operator stays the baseline, with the seasonal operators as a sensitivity check.
     
- [x] Calendar-ordered seasonal forecast — `agulhas/release_points/calendar_forecast.py`, following the season-aware chain of Miron et al. (arXiv 1903.06165).
  - Each of the 13 release points is released in each of the twelve months; every τ step uses the matrix of the season of its start day, and a seasonal row that is flagged or empty takes the annual row. One-year fates are compared with the annual estimate and its 95% drifter-bootstrap interval.
  - Seasonal rows replaced by the annual row: 75–88% drogued and 25–30% undrogued at 1°; 49–59% and 12–16% at 2°.
  - Median departure from the annual one-year fate (drogued / undrogued): 0.2 / 0.4 points at 1° and 0.4 / 0.7 at 2°. Estimates outside the annual 95% interval: 1.0% / 5.3% at 1° and 1.6% / 8.5% at 2°.
  - Largest departures: 23.7 points drogued (Lane 27.5°E, stranded, June release) and 16.4 undrogued (Cape Town, exit west, November release) at 1°; 10.5 (Lane 27.5°E, exit east, December) and 16.2 (Lane 54.5°E, exit east, September) at 2°. At 1°, undrogued estimates outside the interval cluster in June–July releases, with smaller groups in March–April and September–November.
  - One-year fates change little with the release season on average. Drogued seasonal rows are too thin for this step to say more. The seasonal matrices' own sampling noise is not in the annual intervals, so the comparison is a guide, not a test.
  - Run `python -m agulhas calendar_forecast` (and `--res 2`); outputs `data/calendar_forecast_{tag}.csv`, `figures/agulhas_calendar_forecast_{tag}.png`.

- [x] Chapman–Kolmogorov consistency — `agulhas/validation/chapman_kolmogorov.py`.
  - Compares `P(3.5 d)^2` with an operator estimated directly from 7-day drifter transitions.
  - Common reliable rows: 731 drogued and 892 undrogued.
  - Median row-wise TVD is 0.243 for drogued and 0.173 for undrogued drifters; the 90th percentiles are 0.374 / 0.261.
  - TVD is below 0.25 for 53.2% / 87.1% of common reliable rows and below 0.50 for 98.9% / 99.8%.
  - Almost all of that TVD is sampling noise in the direct 7-day rows. Resampling whole drifters 1,000 times (782 drogued, 1,092 undrogued) gives a median noise TVD of 0.224 / 0.169 per row, so the TVD beyond noise is 0.015 / 0.002 (median). Only 1.8% / 3.7% of rows exceed the 95th percentile of their own noise, about the share expected by chance.
  - The noise level leaves out the smaller noise in `P(3.5 d)^2`, which would raise it further, so the comparison leans towards flagging rows. Within sampling noise the 3.5-day operator is Markov at 1°: the drogued rows' larger TVD comes from their thinner data, not from memory beyond τ.

- [x] Coastal pooling — `agulhas/coast/coastal_pooling.py`.
  - At 1°, τ = 3.5 d, a conservative Natural Earth 50m land-proximity filter removes four offshore candidates from the original 103 coastal cells, leaving 99 cells.
  - The 99 coastal cells are grouped into 44 contiguous coastal segments, capped at four neighbouring cells per segment.
  - Before pooling, 35.0% of active drogued coastal rows and 48.9% of active undrogued coastal rows are reliable (≥ 10 distinct drifters).
  - After pooling, 60.6% / 72.7% of coastal cells belong to a reliable pooled segment for drogued / undrogued drifters.
  - 29 of 44 segments (65.9%) are reliable for both drogue types.
  - Median segment size is two coastal cells; 15 segments remain thin and explicitly flagged.
  - The geographic filter only removes candidates from the existing coastal-cell set; it does not add potentially missing near-land cells.
  - `shipping_overlay`, `bootstrap`, `oil_wind` and `coastal_coverage` use the model-neighbour coastal definition.
  - Pooling does not alter or smooth the main transition matrix.

- [x] ERA5 winds — `agulhas/sources/fetch_era5.py`: 10 m wind (0.5°, 6-hourly, 1995–2022) from the Copernicus CDS API, interpolated to 10.03M drifter positions (99.9% of positions in R).

- [x] Oil version — `agulhas/operators/oil_wind.py` (1°, τ = 3.5 d).
  - Undrogued windage: 2.0% of the 10 m wind (95% interval 1.81–2.19%, resampling whole drifters over the 746 cells with ≥ 10 drifters of each type), from regressing the per-cell slip (undrogued − drogued velocity) on the ERA5 wind at undrogued positions. The estimate holds without the fast-current cells (2.2% without cells whose mean drogued current exceeds 0.3 m/s, and 2.0–2.3% for cutoffs from 0.2 to 1 m/s) and east of 35°E (2.1%).
  - The oil matrix adds 1.5%: a median shift of 25 km per step (90th percentile 42 km); 0.01% of shifted ends land on land or unvisited cells and keep their drifter end.
  - Share of released parcels stranded within 30 days, with 95% intervals from resampling whole drifters (oil in `oil_wind`, undrogued from `bootstrap`). "Grounded" is the number of drifters that ran aground starting from the release cell itself; stranding also arrives from neighbouring cells, so a point with none can still strand.

    | Release point | Undrogued | Oil | Check (drogued + 3.5%) | Grounded |
    |---|---|---|---|---|
    | Durban | 35% (1–62) | 39% (2–66) | 6% | 2 |
    | Cape Town | 35% (14–55) | 39% (15–58) | 11% | 4 |
    | Mozambique Channel lane (40.5°E, 16.5°S) | 35% (16–56) | 41% (22–62) | 10% | 3 |
    | Saldanha Bay | 3% (0–8) | 11% (0–20) | 0% | 0 |
    | Lane 45.5°E, 26.5°S | 10% (4–18) | 10% (5–18) | 1% | 3 |
    | Lane 47.5°E, 25.5°S | 8% (4–14) | 9% (4–15) | 0% | 1 |
    | Richards Bay | 6% (2–14) | 7% (2–16) | 6% | 1 |
    | Algoa Bay | 6% (2–13) | 7% (2–13) | 9% | 2 |
    | Lane 27.5°E, 33.5°S | 6% (2–11) | 7% (3–12) | 14% | 3 |
    | Lanes 43.5°E and 50.5–54.5°E | 0–1% (0–2) | 2–3% (1–5) | 0% | 0 |

  - The high coastal shares rest on few groundings: Durban's 35% comes from 2 drifters in its cell, and its interval runs from 1% to 62%. Treat the three ~35% points as "a large share may strand within a month", not as a precise rate; the three agree to the percent by coincidence. The oil–undrogued differences at single points sit well inside these intervals.
  - The added windage moves oil out of R faster than undrogued material: after a year 1–21% is still afloat in R from the lanes and 0–8% from the hotspots (undrogued 3–43% and 0–16%). These one-year shares are much better constrained than stranding (intervals about ±5 points, in `oil_compare_…csv`), oil is below undrogued wherever any remains, and the two intervals do not overlap at Algoa Bay, Richards Bay and every lane except 40.5°E, 16.5°S (they overlap there and at Durban).
  - The oil and check matrices differ by TVD 0.19–0.42 at 30 days. Most of the gap at Durban, Cape Town and the Mozambique Channel lane is stranding: the check matrix strands 6–11% there within 30 days against 39–41% for oil, since few drifters were still drogued when they reached the coast (see `drogue_loss`). Along the South African south coast the check strands as much as oil or more (Algoa Bay 9%, the 27.5°E lane 14%), where most drogued groundings happened. For oil still afloat the two agree within 0.16–0.28 everywhere except the Mozambique Channel lane (0.35), where drogued rows are thin. The oil matrix is the working version.
  - Outputs: `data/P_oil_1deg_3p5d_10E-55E_45S-15S.npz`, `data/P_oil_check_…npz`, `data/oil_compare_…csv`, `figures/agulhas_oil_durban_30d_…png`.

- [x] Whole-box structure — `agulhas/whole_box/` (all grids at τ = 3.5 d, both drogue types; numbers below at 1°, drogued / undrogued, over reliable cells).
  - Support (`whole_box_analysis`, `coastal_coverage`): 78% / 93% of active rows reliable (0.5°: 52% / 91%; 2°: 90% / 94%). Coastal cells (land neighbour on the grid): 33% / 47% reliable, against 82% / 98% offshore.
  - Retention hotspots (`trap_stability`): drogued at 41–44°E, 30–33°S, 63–65% still in R after a year (95% intervals about 48–74%); 70 of 75 hotspot cells stay in the top 10% in ≥ 50% of 1,000 drifter resamples. Undrogued at 48–51°E, 30–33°S, 55–56% (49–60%), 88 of 90 stable. Where both grids are reliable, 1° and 2° hotspots agree (Jaccard 76% / 73%); 0.5° drogued agrees less (37–41%). Held-out drifters show the highest undrogued retention overstated by about 10 points (95% intervals +1 to +21; `held_out_year`).
  - Residence time (`residence_time`): mean 223 / 203 days, median 223 / 213, longest in the hotspots (518 / 459 days); grids within about 40 days.
  - Exit zones (`exit_zones`): eventual fate E 69% / 62%, W 18% / 17%, S 6% / 4%, N 2% / 3%, stranded 4% / 15%; east is the most likely fate in 83% / 75% of cells, stranding in 11% of undrogued cells (Mozambique Channel coast, east of Madagascar) and in no drogued cell.
  - Almost-invariant regions (`almost_invariant_regions`): four for both drogue types (retroflection and Atlantic side; Natal coast and Mozambique Channel; south of Madagascar and Return Current; east of Madagascar), keeping 81–94% over one step, 54–70% over 30 days and 1–8% over a year. Three keep 13–23 points more over 30 days than longitude bands of the same sizes; the western one matches its band. The eigenvalues fall off smoothly, so four regions is a choice.
  - Outputs: `data/whole_box_support_…csv`, `data/coastal_support_…csv`, `data/retention_*_…csv`, `data/residence_time_*…csv`, `data/exit_zones_*…csv`, `data/almost_invariant_*…csv` and the matching `figures/agulhas_*` maps.


- [x] Stranding by coast — `agulhas/coast/coastal_stranding.py` (1°, τ = 3.5 d; drogued, undrogued, oil and the oil check; 1,000 resamples of whole drifters).
  - Six named coasts (`transport.coast_of`), each stranding move taking the coast of its drifter's last observation. The coast columns sum to each operator's stranded column exactly (checked on every build).

    | Coast | Grounded drifters: drogued / undrogued | Mean eventual stranding over reliable cells: drogued / undrogued / oil |
    |---|---|---|
    | Atlantic and Cape (Namibia to Cape Agulhas, 20°E) | 5 / 12 | 0.9% / 1.1% / 1.2% |
    | South African south and east coast (to 26.9°S) | 7 / 17 | 2.0% / 2.1% / 2.1% |
    | Southern Mozambique (to 21°S) | 4 / 28 | 1.1% / 5.3% / 6.0% |
    | Central and northern Mozambique | 0 / 18 | 0.0% / 3.6% / 4.6% |
    | West and south Madagascar (to 47°E) | 1 / 18 | 0.2% / 1.5% / 1.5% |
    | East Madagascar | 1 / 44 | 0.1% / 1.5% / 1.7% |

  - Every drogued coast is thin (< 10 grounded drifters), so drogued stranding by coast is indicative only and a lower bound (`drogue_loss`).
  - Material travels far before grounding. From the lanes south and east of Madagascar the largest share strands on southern Mozambique (oil 14–22%, undrogued 8–17%), not on Madagascar: from the 47.5°E lane oil strands 20% (13–26%) there against 11% (6–17%) on west and south Madagascar.
  - The added windage brings oil ashore sooner rather than much more often. From the 47.5°E lane oil strands 49% (42–55%) in 95 days (81–111) and undrogued material 39% (32–47%) in 151 days (127–178): the stranding intervals overlap and the time intervals do not.
  - Mozambique Channel lane: oil strands 71% (53–84%), 53% (33–70%) on central and northern Mozambique, in 35 days (22–56). Drogued material strands 29% (12–48%), mostly on southern Mozambique, in 136 days (88–237).
  - Cape Town and Durban rest on few groundings: undrogued Durban 39% (7–65%), Cape Town 36% (14–57%). These say a large share may strand, not at what rate.
  - A resample can leave a few thin cells (1–6 drifters, off Angola, Namibia and western Madagascar) with no way out; material entering them stays in R (`transport.stranding_chain`), so every resample is kept.
  - Outputs: `data/P_{drogued,undrogued,oil,oil_check}_coast_…npz`, `data/coastal_stranding_points_…csv`, `data/coastal_stranding_cells_…csv`, `data/coastal_stranding_support_…csv`, `data/coast_counts_…npz`, `figures/agulhas_coastal_stranding_…png`.

- [x] Traffic-weighted exposure — `agulhas/coast/traffic_exposure.py` (1°, τ = 3.5 d, 1,000 resamples of whole drifters).
  - Raster values fall into two groups with almost nothing between: about 1–30 and 1e4–1e7 per pixel. The high group traces the shipping lanes (14% of pixels at 34.5°S, 0.4% at 39.5°S). Weights are the mean density per cell, capped at the 99th percentile of ocean cells; the data predate the Red Sea diversions of late 2023.
  - 31.6% of the weight sits on cells not reliable in both matrices and moves to the nearest reliable cell (weighted median 148 km, 90th percentile 672 km). 0.04% of the weight lies south of 40°S, so the raster's step there does not matter.
  - Share of all traffic-weighted releases that eventually strands, with 95% intervals:

    | Coast | Drogued | Undrogued | Oil |
    |---|---|---|---|
    | Southern Mozambique | 3% (0–5) | 8% (5–11) | 10% (6–12) |
    | Central and northern Mozambique | 0% (0–0) | 6% (4–8) | 8% (5–10) |
    | South African south and east coast | 4% (1–7) | 4% (2–6) | 4% (2–6) |
    | West and south Madagascar | 0% (0–1) | 2% (1–4) | 3% (1–4) |
    | Atlantic and Cape | 1% (0–4) | 2% (1–3) | 2% (1–3) |
    | East Madagascar | 0% (0–0) | 2% (1–2) | 2% (1–2) |

  - By route, for oil:

    | Route | Share of traffic | Stranded within 30 days | Eventually stranded | Days to strand (95% interval) | Main receiving coast |
    |---|---|---|---|---|---|
    | Mozambique Channel | 10% | 41% | 69% | 38 (28–51) | Central and northern Mozambique, 40% |
    | East of Madagascar | 17% | 9% | 43% | 122 (109–135) | Southern Mozambique, 14% |
    | South of Madagascar | 31% | 6% | 33% | 104 (94–116) | Southern Mozambique, 16% |
    | Natal coast | 9% | 9% | 14% | 73 (55–112) | South African south and east coast, 9% |
    | Cape route, east of Cape Agulhas | 7% | 7% | 13% | 71 (54–111) | South African south and east coast, 7% |
    | Cape route, west of Cape Agulhas (and west coast) | 27% | 2% | 2% | 17 (13–25) | Atlantic and Cape, 2% |
    | All traffic | 100% | 9% | 27% | 87 (79–97) | Southern Mozambique, 10% |

  - Source of each coast's oil stranding: half of southern Mozambique's comes from the route south of Madagascar, 54% of central and northern Mozambique's from the Mozambique Channel, and 97% of east Madagascar's from the route east of Madagascar. The South African south and east coast takes 45% of its stranding from the route south of Madagascar, against 32% from its own routes (Natal, Cape east).
  - All traffic strands 27% as oil in 87 days (79–97), 23% as undrogued material in 134 days (118–150) and 8% as drogued material in 202 days (151–263). Oil released on the Cape west route leaves west into the Atlantic (97%, 96–99), so that route strands little despite carrying 27% of traffic.
  - With oil's mass loss (exp(−t / 14 d)), 3.6% of the released mass reaches a coast, against 27% of the parcels: most oil that strands has weathered by the time it arrives. The 14-day rate is a working value and the matrix itself models transport only.
  - Sensitivity. Dropping the moved weight changes each coast by under 1 point. Weighting by the share of lane pixels lowers the two Mozambique coasts by about 3 points for oil (6.8% and 4.7%, at the low end of their intervals) and keeps the same three coasts most exposed in the same order, so the ranking is firmer than the percentages.
  - Near-shore presence over 30 days, all traffic, is highest on the South African south and east coast for drogued, undrogued and oil material (0.83 / 0.63 / 0.68 material-days); per-segment values are in `traffic_nearshore_segments_…csv`.
  - Outputs: `data/traffic_exposure_…csv`, `data/traffic_nearshore_segments_…csv`, `data/traffic_weights_…csv`, `figures/agulhas_traffic_exposure_…png`.

- [x] Release-risk atlas — `agulhas/coast/risk_atlas.py` (1°, τ = 3.5 d; drogued, undrogued and oil at 3%, 3.5% and 4% of the wind; 1,000 resamples of whole drifters).
  - All 970 ocean cells as release cells: the share afloat in R, exited across each edge and stranded on each coast at 7, 30 and 365 days and eventually, and days to strand, each with a 95% interval and the cell's reliability flags (`flagged`, `empty`, `row_drifters`). Outcomes sum to 1 for every cell and horizon.
  - Stranding from a typical reliable cell is low even after a year (median 0.8% drogued, 3.1% undrogued, 4.0% oil), with a long tail: the top 10% of cells reach 9% / 45% / 56%.
  - The highest-risk cells for oil at 30 days lie on the Mozambique coast at 35–36°E, 19–22°S (78–97%, intervals from 58%) and off north-east Madagascar at 49.5°E, 16.5°S (80%, 55–100%).
  - Outputs: `data/risk_atlas_…csv` (232,800 rows), `figures/agulhas_risk_atlas_…png`.

- [x] Packaging — `python -m agulhas all` runs every step in order, with logs, resuming with `--from`, and the core steps on another region with `--box` and `--data`. `agulhas/tools/worked_example.py` answers the project's question for any release point, with the cell's reliability, intervals, coast and method flag.
  - Reproducibility test: a fresh clone in a fresh environment built from `requirements.txt`, given only the three raw inputs (drifter subset, ERA5 winds, shipping zip), ran the 34 steps that are not downloads and reproduced all 115 output tables and matrices exactly (numeric values within 1e-9; text in the tables identical). The downloads themselves were not repeated in this test. `source_sink` and `known_features` were checked by running them twice in place, which gave identical tables. Results depend on the GDP store at fetch time, so the README records each subset's counts and SHA-256.
  - Oil explorer: `agulhas/tools/oil_explorer.py` builds a web page where a click in R releases oil and a slider steps it through the year, with the reliability of the cell, the atlas's intervals and the coasts reached. The page's steps were checked against Python on 139 cells (largest difference 2e-15) and against the atlas's estimates on all 970 cells at 7 days, 30 days and 1 year (largest difference 5e-6, the atlas's rounding). It only reads pipeline outputs, so a full run skips it.
  - The Gulf Stream run through `python -m agulhas all` also reproduced its 45 tables.

- [x] Spill hindcasts — `agulhas/validation/spill_hindcasts.py` (oil at 3.5%, 1°, τ = 3.5 d; 1,000 resamples of whole drifters). A check of coast and timescale: the matrix holds average conditions over 1995–2022, so it cannot reproduce one event's winds, and one event can neither confirm nor reject a probability.
  - Records: MV Treasure sank on 23 June 2000 at 33.67°S, 18.33°E (400–1,000 t of heavy fuel oil, sources differ), and oil reached Robben and Dassen islands, drifting towards Dassen by 2 July. Castillo de Bellver broke up on 6 August 1983 at 33.28°S, 17.50°E (light crude; 50–60,000 t or 145–170,000 t, sources differ); the oil first drifted towards the coast, then a wind shift took it offshore into the north-west flowing Benguela Current, with little coastal pollution.
  - The rule, set before the script was run: more than 50% stranded within 30 days for Treasure and less than 50% for Castillo de Bellver; Atlantic and Cape receiving the most; north-westward drift of material in R after 14 days. "Consistent" needs the whole interval on the side of what happened. The stranding estimates for both cells had already been seen in the atlas when the rule was written, so it is not blind.
  - Castillo de Bellver is consistent on all three at 3%, 3.5% and 4%: 11% (0–19%) stranded within 30 days, Atlantic and Cape, and a move of 168 km west (129–235) and 264 km north (216–341).
  - Treasure is inconclusive. The spill cell gives 90% (0–100%) stranded from 4 undrogued drifters, and its direction is unclear too. The nearest reliable cell (17.5°E, 33.5°S, about 80 km away and offshore) gives 11% (0–19%), inconsistent, and the reliable neighbours 27% (12–39%). Only the coast rule passes, which any nearby release would. The wreck, Robben Island and Dassen Island all lie in one 1° cell, and the nearest reliable cell is the Castillo de Bellver cell, so the matrix gives one answer there to two spills that went different ways.
  - This shows the limit of the 1° operator: it represents offshore transport, but not a spill about 10 km from a thin coastal cell. The winter undrogued matrix does not change this (3% to 4% at the Castillo de Bellver cell; the Treasure cell stays thin). The oil and check matrices also disagree at the Treasure cell (90% against 0%), where the check rests on 1 drogued drifter.
  - Outputs: `data/spill_hindcasts_…csv`, `data/spill_hindcast_verdicts_…csv`, `figures/agulhas_spill_hindcasts_…png`.

- [x] Oil windage range — `agulhas/operators/oil_wind.py` builds oil (undrogued + the missing windage) and its drogued check at 3%, 3.5% and 4% of the wind; `agulhas/operators/oil_windage.py` compares them (1°, τ = 3.5 d; 1,000 resamples of whole drifters, the same draws for all three oil matrices). The range adds 1.0–2.0% to undrogued moves (a median shift of 16–33 km per step), which also covers the uncertainty in measured windage (1.81–2.19%).
  - How much more often, if releases follow ship traffic: undrogued 23% (20–26%) eventually strands; oil 26% (22–29%), 27% (24–30%) and 29% (25–32%) at 3%, 3.5% and 4%. From 3% to 4% adds 2.5 points (1.6–3.1), beyond sampling noise.
  - How much sooner: 134 days (118–151) undrogued; 98 (89–110), 87 (79–97) and 78 (71–87) for oil. From 3% to 4% is 20 days sooner (17–24). Windage changes timing more than amount: the 3% and 4% intervals overlap on amount but not on timing. Within 30 days, 7% undrogued and 8% / 9% / 10% oil have stranded.
  - Which coasts change: mainly Mozambique. Southern Mozambique goes from 7.9% undrogued to 9.2% / 9.6% / 10.0% and central and northern Mozambique from 5.7% to 6.9% / 7.5% / 8.2%, while the other four coasts move by 0.2 points or less across 3–4%. The order of the coasts is the same at all three windages.
  - Which routes respond: east of Madagascar most (40% / 43% / 46%; 3–4% difference 5.7 points, 4.4–6.8), then south of Madagascar (3.4, 2.1–4.4) and the Mozambique Channel (2.9, 1.3–4.6). The Cape routes and Natal do not change in amount beyond noise, though stranding there still comes sooner.
  - Oil mass stranded within a year with the 14-day decay: 3.3% (2.7–4.0%), 3.6% (2.9–4.3%) and 3.9% (3.2–4.5%), against 2.8% for undrogued material treated the same way.
  - Method flag (oil against its drogued check, TVD > 0.3 at 30 days): the same 4 of 13 release points at every windage, Cape Town (0.31–0.33), Durban (0.34–0.37), the lane at 40.5°E, 16.5°S (0.41–0.44) and the lane at 45.5°E, 26.5°S (0.31–0.32). Their oil results depend on the method. The flag is saved in `oil_compare_…csv` and at each windage in `oil_windage_method_flags_…csv`.
  - Outputs: `data/oil_windage_…csv`, `data/oil_windage_difference_…csv`, `data/oil_windage_method_flags_…csv`, `figures/agulhas_oil_windage_…png`.

- [x] Hypothesis tests — `agulhas/validation/hypothesis_tests.py` (1°, τ = 3.5 d; 1,000 resamples of whole drifters), each hypothesis under its stated decision rule.
  - H1 supported. P(exit E) − P(exit W), eventual fate averaged over reliable cells, is 0.51 (0.45–0.57) drogued and 0.45 (0.41–0.49) undrogued. Cell by cell, east is the clear majority (85% / 88% of cells have an interval above zero), west holds in 12% / 11%, around the Cape and the west coast, and 4% / 2% are unclear.
  - H2 supported, with a caveat. Within a year, undrogued stranding exceeds drogued with non-overlapping intervals in 7 of 8 lane cells; the result is the same for eventual stranding. Examples: 66% (50–82%) against 27% (10–47%) at 40.5°E, 16.5°S and 24% (17–30%) against 0.9% (0.3–1.9%) at 54.5°E, 22.5°S. The exception is the South African coast at 27.5°E, 33.5°S, where drogued is higher but uncertain (19%, 5–36%, against 10%, 5–16%). Drogued stranding is a lower bound, since few drifters still have a drogue at the coast (`drogue_loss`), so the rule cannot separate how much of the gap is windage.
  - H3 supported from 17.5 days. Folds won out of 5 against mean-current advection, the harder baseline: 14 d 4 drogued / 3 undrogued, 17.5 d 4 / 5, 21–28 d 5 / 5. The operator beats persistence in all folds at every one of these horizons. At 14 days the undrogued operator misses the rule, but with a mean margin of 1–2 km. Margins over advection stay small at 28 days (252 against 267 km, 289 against 299 km), so its main advantage remains calibrated probabilities.
  - Outputs: `data/hypothesis_h{1,2,3}_…csv`, `figures/agulhas_hypothesis_tests_…png`.

- [x] Second region — the core pipeline on the Gulf Stream, box 80–40°W, 25–45°N, to test that region, grid and lag really are inputs. `python -m agulhas fetch_gdp_subset --fetch-box -90 -30 15 55 --out data/gulfstream_gdp1h_subset.nc` saves 23.6M hourly positions from 3,022 drifters (11.8M positions in the box); the steps not marked Agulhas-only, up to `drogue_loss`, then run with `--box -80 -40 25 45 --data …` and no region-specific code changes. Not run: `box_selection` (searches around the Agulhas core), `fetch_era5` and `oil_wind` (would need ERA5 for this area) and `coastal_stranding`, `traffic_exposure` and `risk_atlas` (the named coasts and routes describe the Agulhas).
  - Support (1°, drogued / undrogued): 94% / 92% of the 732 ocean cells have ≥ 10 drifters; coastal cells are thin (33% / 26% reliable, against 96% / 95% offshore).
  - Fates over reliable cells: east 42% / 63%, south 35% / 21%, north 22% / 11%, west 0.2% / 0.4% (the west edge is mostly land), stranded 1.5% / 4.3%. East is the largest fate but a majority only for undrogued material. Stranding rests on 25 / 81 drifters.
  - Physics matches what is known of the region. One-year retention peaks in the Sargasso Sea (58–61°W, 30–33°N, 76–77% / 70–71%), and 67 of 70 / 67 of 68 hotspot cells hold up in at least half of 1,000 resamples. Undrogued slip is westward under the trades (6.8 cm/s at 25–30°N) and eastward under the westerlies (6.4 cm/s at 40–45°N), the same pattern as R.
  - Validation: held-out exits at 28 days are 10.1% / 9.8% predicted against 10.8% / 10.5% observed, stranding 0.24% / 0.45% against 0.26% / 0.45%, and the log score beats climatology in all folds. Point accuracy is much weaker than in R. Folds out of 5 won against advection and persistence:

    | Horizon | Drogued vs advection | Drogued vs persistence | Undrogued vs advection | Undrogued vs persistence |
    |---|---|---|---|---|
    | 14 d | 0 | 0 | 0 | 1 |
    | 21 d | 3 | 1 | 4 | 5 |
    | 28 d | 3 | 2 | 5 | 5 |

    Started from the cell centre, the same information the matrix has, advection loses to the operator in all folds at these horizons. This fits the operator's main strength being calibrated probabilities, not point accuracy.
  - Chapman–Kolmogorov: 5.1% / 2.1% of rows are above their 95% noise level, so the Markov assumption holds as in R. The decorrelation time is 1.37 d drogued and 1.40 d undrogued, so τ = 3.5 d suits this box too.
  - Windage and stranding (`hypothesis_tests`): undrogued stranding exceeds drogued with non-overlapping intervals in only 1 of 8 lane cells, against 7 of 8 in R. The result in R is therefore not general.
  - Differences from R: long-run results depend on the grid (mean time in the box 608 → 478 → 350 days drogued and 427 → 379 → 305 days undrogued from 0.5° to 1° to 2°, against about 40 days of spread in R), and drogued transport is seasonal beyond sampling noise in 5 of 6 season pairs (undrogued 6 of 6; drogued 0 of 6 in R). The 1° annual baseline chosen for R therefore should not be assumed elsewhere without rechecking.

- [x] Known features — `agulhas/validation/known_features.py` (1°, τ = 3.5 d; drogued / undrogued, 95% intervals from 1,000 resamples of whole drifters). The drogued result is the test, since wind also moves undrogued drifters. The rules were fixed before the script first ran, though the speed threshold was set knowing the maximum mean current from `shipping_overlay` (1.23 m/s drogued).

    | Feature | Rule | Drogued | Undrogued |
    |---|---|---|---|
    | Fastest mean current on the core rows (27–34°S) | at least 1 m/s | 1.11 m/s (0.96–1.53) | 1.34 m/s (1.22–1.67) |
    | Direction of those cells | south-west, 180–270° | 219° (215–221) | 220° (218–222) |
    | Drift from Durban over 7 days | south-west, 180–270° | 215° (210–220) | 216° (213–219) |
    | End of the westward jet (retroflection) | 15–21°E | 18.5°E (17.5–20.5) | 18.5°E (17.5–18.5) |
    | Fate divide, P(E) = P(W), at 34–45°S | 15–21°E | **14.5°E (11.5–18.5), outside** | 15.5°E (13.5–17.5) |
    | Return Current latitude | 37–41°S | 38.5°S (38.5–39.5) | 40.0°S (39.5–40.5) |
    | Cape Basin mean flow | north-west, 270–360° | 342° (331–352) | 356° (352–359) |
    | Leakage share from the core rows | interval overlaps 21–30% | 27% (20–36) | 26% (20–32) |

  - Seven of the eight rules hold for drogued material. The fate divide falls one cell west of its range, though its interval reaches into it. Exits are counted at 10°E, so the divide also depends on where R was drawn.
  - The leakage share compares surface material with a transport over the upper 1,000 m (15–21.3 Sv of 70 Sv in the literature), so it is a check of size only. The Cape Basin mean holds the Benguela Current as well as the rings, so its direction cannot separate them.
- [x] Source and sink cells — `agulhas/whole_box/source_sink.py` (1°, τ = 3.5 d): with one unit of material in every ocean cell, the column sum of P is what a cell holds after one step. Reliable cells off the box edge are judged by whether the 95% interval excludes 1. Cells on the edge are not judged: inflow from outside R is not in the matrix, so they look like sources whatever the flow does.
  - Drogued: 28 sinks and 36 sources of 647 cells. Undrogued: 30 sinks and 48 sources of 789 cells. That is about 10% of cells, against the 5% expected by chance, since the intervals are not corrected for the number of cells.
  - The two drogue types barely agree over the 642 cells reliable in both (correlation 0.21). Only 13 cells are judged in both, all with the same kind. On the 2° grid the correlation is 0.52, and 17 of the 18 cells judged in both have the same kind.
  - The one-step pattern is therefore weak: about half of the judged cells are what chance alone would give. Some of the rest may come from the 1° grid, since a fast current crosses about three cells per step, but that has not been tested. Retention (`trap_stability`) and fates (`exit_zones`) are the better guide to where R gathers and loses material.
  - Gulf Stream: 28 sinks and 30 sources of 609 cells drogued, 26 and 21 of 599 undrogued, with correlation 0.15.

Open questions:
- Drogued stranding rests on the few drifters still drogued when they reached the coast, since 121 of the 139 grounded drifters lost their drogue more than τ before grounding (`drogue_loss`). The drogued stranding estimate is therefore treated as a lower bound rather than borrowing the undrogued rate.
- Coastal sparsity is mitigated with short pooled coastal segments, while the main transition matrix remains unchanged and thin segments remain flagged.
- The annual 1°, τ = 3.5 d operator remains the baseline. Drogued transport shows no seasonality beyond sampling noise and undrogued transport only a small one (permutation test in `seasonal_matrices`), so the seasonal operators serve as a sensitivity check.
- Next: the report and presentation.
