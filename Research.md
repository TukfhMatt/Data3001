Something goes into the water at a chosen point in R: a container off a ship or a slick of oil. Where has the surface flow taken it a week later? A month later? A year? At the scale of the whole box, which parts of R feed which others, and which are cut off from the rest?

Product. A surface-transport operator for R: transition matrices that someone else can load and iterate forward, together with what they say about where chosen release points end up, and about where R as a whole gathers material and where it loses it.


Region:
Box R: **10°E–55°E, 45°S–15°S** (`BOX` in `scripts/config.py`)

Why the Agulhas:
- The Agulhas Current is one of the strongest western boundary currents (up to ~2 m/s), flowing south-west along the South African east coast.
- At the retroflection south of Africa (~16–20°E) most of the flow turns back east as the Agulhas Return Current (~38–40°S); some leaks into the South Atlantic via Agulhas rings. This gives distinct "feed" and "cut-off" structure for the whole-box question.
- The shipping route around the Cape of Good Hope is busy (more so since the 2024 Red Sea diversions), so container loss and oil spills are realistic scenarios.
- GDP drifters sample it well: 5.95M hourly observations from 1,503 drifters in R.

Why this box (`scripts/05_box_selection.py` scores boxes on data, shipping risk and retention):
- Oil-like undrogued transport stays in R 136 days on average; 6.5% is still inside after a year in the box-selection calculation.
- Contains 97% of the Cape Basin eddy energy (Agulhas rings) and the Mozambique Channel eddies.
- Covers the coastal route around the Cape, the Mozambique Channel routes and the route south of Madagascar.
- Passes the data rules with margin: 4.12M undrogued hourly observations, 92% of ocean cells seen by ≥ 10 drifters.


What we track:

Shipping container:
- Modelled with drogued drifters, which follow ~15 m currents with almost no wind effect (windage), matching a mostly submerged or flooded container.

Oil spill:
- Surface oil drifts at roughly current + 3–3.5% of wind speed.
- Modelled with undrogued drifters, which ride at the surface and slip downwind of drogued drifters by ~10–12 cm/s in the westerly and trade-wind bands (`scripts/07_wind_slip.py`). Against ERA5 winds that slip is 2.0% of the 10 m wind (`scripts/09_oil_wind.py`), a little over half of oil's windage.
- ERA5 10 m winds (Copernicus Climate Data Store, cds.climate.copernicus.eu) supply the wind term. Undrogued motion already contains 2.0% windage, so the oil matrix adds only the missing 1.5% (3.5% − 2.0%) along each drifter's path; adding the full 3.5% would count the wind twice. A check matrix (drogued + 3.5% wind) tests that choice.
- The measured 2.0% is everything that moves the surface differently from 15 m: direct windage on the drifter, wave-driven Stokes drift and wind-driven shear in the top 15 m. Subtracting it from 3.5% therefore treats oil's 3.5% as drift relative to the 15 m current. If the 3.5% is instead read as drift relative to the true surface current, the oil matrix adds too little wind, and the drogued + 3.5% check bounds that case.
- The model represents transport of surface oil parcels only. Weathering, evaporation, emulsification, dissolution and sinking are not modelled, so results describe where surface material is transported rather than how much oil mass remains.

Drogued and undrogued matrices are built and reported side by side and never pooled: an "all drifters" matrix is 46–84% undrogued depending on the cell, so its wind effect would follow sampling, not physics.


Data:
- NOAA GDP hourly drifter data (v2.01, 1995–2022 in R), streamed from the public AWS S3 zarr store (the CloudDrift `gdp1h` source) by `scripts/00_fetch_agulhas_subset.py` into `data/agulhas_gdp1h_subset.nc`.
  - Native hourly resolution; positions interpolated across > 6 h between satellite fixes are dropped (0.3%).
  - The saved area (5°W–65°E, 60–5°S) covers R plus a margin, so positions just after a drifter leaves R are kept.
  - Key fields: `lon`, `lat`, `time`, `ve`/`vn` (velocity), `drogue_status`, `gap`, `rowsize`, `id`.
  - Split by `drogue_status`: drogued → container matrix; undrogued → oil-like surface matrix.
- Commercial shipping density: World Bank / IMF Global Shipping Traffic Density, Commercial layer (AIS, Jan 2015–Feb 2021, CC-BY 4.0).
- ERA5 10 m wind (u10, v10), 6-hourly on a 0.5° grid, R plus a 5° margin, for the oil wind term.


Method (Ulam's method / transition matrix):
1. Grid R into cells at 0.5°, 1° and 2°; 1° is the working grid. Land cells are masked.
2. Transition time τ = 3.5 days, longer than the Lagrangian decorrelation time (integral timescale 1.40 d drogued, 95% interval 1.34–1.47, and 1.20 d undrogued, 1.16–1.23, over 10–40°E, 45–25°S; `scripts/27_decorrelation_time.py`); drifters move about one cell per step. The Markov approximation is checked separately with a Chapman–Kolmogorov comparison between `P(τ)^2` and transitions estimated directly at `2τ`.
3. For every drifter position at time t in cell i, find where it is at t + τ (cell j). Count transitions C[i, j].
4. Five absorbing states: four "exited R" states (W, E, S, N) record which edge material leaves by, and a "stranded" state records material that runs aground inside R. A drifter strands when GDP records its death as "ran aground" and its last observation lies in R; every position in its final τ moves to the stranded state. A stranding move takes the drogue status of its start, since the drogue flag often flips to undrogued in a record's final hours.
5. Row-normalise: P[i, j] = C[i, j] / Σ_j C[i, j]. Each row is "where does material in cell i go in τ days". A cell with no outgoing moves of its own (its drifters' records ended there for other reasons, such as failed transmitters) takes the pooled moves of the nearest ring of ocean cells around it.
6. Iterate: distribution after n steps = p₀ · Pⁿ. 1 week = 2 steps, 1 month ≈ 9 steps, 1 year ≈ 104 steps.
7. Oil version: each undrogued move's end point is shifted by the missing windage (3.5% − the measured undrogued 2.0%) × the mean ERA5 wind along that move × τ, and the shifted moves are counted as in step 3. Moves into the stranded state are kept as they are. The matrix conserves probability and models transport only; oil weathering and mass loss are outside the scope of the model.


Reliability:
- Rows built from < 10 distinct drifters are flagged; cells with no outgoing moves of their own (rows pooled from neighbouring cells) are flagged separately.
- `scripts/06_bootstrap.py` resamples whole drifters (never observations) to give 95% intervals for release-point results and a row-stability check (median bootstrap TVD > 0.2 = unstable).
- Hourly positions are strongly correlated; the independent sample size is the number of drifters, not the observation count.
- Coastal data are much thinner than offshore data. `scripts/21_coastal_pooling.py` therefore groups neighbouring coastal cells into short contiguous segments for coastal summaries, while leaving the main transition matrix unchanged. Segments that remain thin are retained and flagged.
- At 1 year most material has left R, so answers are "probability still in R" plus where it exited or stranded.


Analysis:

Release points: 5 hotspots plus the busiest reliable shipping cells (`scripts/04_shipping_overlay.py`):
- Richards Bay, Durban (major ports, in the core of the current)
- Algoa Bay (ship-to-ship bunkering, off Gqeberha)
- Cape Town, Saldanha Bay (Atlantic side)
- Busiest shipping cells along the coastal route, in the Mozambique Channel and south of Madagascar

For each: map of the probability distribution after 1 week, 1 month and 1 year, plus the fraction left in R, where it exited and how much stranded, with bootstrap intervals.

Whole-box structure:
- Where R holds material: the share released in each cell still afloat in R after a year; the top 10% of reliable cells are retention hotspots, checked across grids and with a drifter bootstrap.
- Where R loses material: residence time (expected time before leaving or stranding) and the eventual fate (which edge, or stranded) of each cell, both exact from the absorbing chain's fundamental matrix N = (I − Q)⁻¹.
- Which parts keep material to themselves: spectral clustering of the symmetrised transition matrix → almost-invariant regions, checked with the directional P against longitude bands of the same sizes.
- Source vs sink cells: row and column sums of P.
- Coastal support: neighbouring thin coastal cells are grouped into short coastal segments for coastal summaries and near-shore exposure.
- Stranding by coast: the stranded state is split into six named coasts by where drifters ran aground, so stranding can be reported per coast with intervals.
- Traffic-weighted exposure: releases spread over R in proportion to shipping density, for all traffic and six routes; which coasts are exposed, from which routes and how soon.
- Release-risk atlas: every ocean cell as a release cell, with stranding, exit and retention probabilities at each horizon and 95% intervals.


Validation:
- Held-out drifters (whole trajectories): their actual positions after 1 week / 1 month against the matrix predictions.
- Sensitivity to grid size (0.5°, 1°, 2°) and τ (2 and 5 days).
- Four seasonal operators (DJF, MAM, JJA and SON) against the annual operator, with 2° used as a robustness check where 1° drogued seasonal support is sparse.
- Chapman–Kolmogorov consistency: compare `P(3.5 d)^2` with transitions estimated directly at 7 days.
- Drifter coverage over time (temporal plot), so no few years dominate the matrix.
- Known features: current path, retroflection, rings heading into the Atlantic.


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
- [x] Data subset — `scripts/00_fetch_agulhas_subset.py`: 17.0M hourly positions from 2,942 drifters in the saved area; in R 5.95M (1.83M drogued, 4.12M undrogued) from 1,503 drifters.
- [x] Exploration and grid coverage — `scripts/01_agulhas_gdp_exploration.py`, `scripts/02_grid_coverage.py`.
  - Ocean cells in R: 3,780 / 970 / 263 at 0.5° / 1° / 2° (the 45°-wide box at 2° ends in a partial 1° column).
  - Cells with ≥ 10 distinct drifters: drogued 52% / 78% / 90%, undrogued 90% / 92% / 93%.
  - Drogued data is thin on the Agulhas Bank shelf, off Namibia and in parts of the Mozambique Channel.

- [x] Region, grid and lag are parameters (`scripts/config.py`, `--box --res --tau --data`); matrix file names include res, τ and box. Another region gets its own subset from `00` (see the second-region step below).

- [x] Transition matrices, both drogue types, 0.5° / 1° / 2° — `scripts/03_transition_matrix.py`, helpers in `scripts/transport.py`.
  - Transitions: 1.77M drogued, 4.07M undrogued.
  - Stranding: 139 drifters ran aground in R (GDP `typedeath` 1, last observation in R and within a day of the record's end date). A move into the stranded state counts for the drogue status at its start (`transport.same_drogue`).
  - Drogue loss before grounding — `scripts/22_drogue_loss.py`: of the 139, 2 were still drogued at the last fix and 16 lost the drogue in their final day (median 4 km from the grounding point; none between 1 and 3.5 days before it). The other 121 were undrogued for more than τ before grounding: 63 throughout their time in the saved area and 58 for a median of 172 days. Drifters that stopped transmitting in R lose the drogue in their final day as often (17%, against 12% for groundings), so the late flip marks the end of a record rather than the grounding, and these 18 drifters reached the coast drogued. The drogued matrix has 1,240 moves into the stranded state from 18 drifters, the undrogued matrix 10,159 from 137 (the 16 late-flip drifters contribute to both: drogued starts before the flip, undrogued after). Most grounded drifters lost their drogue long before, so drogued stranding is a lower bound.
  - Flagged rows (< 10 drifters): drogued 50% / 24% / 11%, undrogued 11% / 8% / 7%. At 1°, 20 (drogued) and 8 (undrogued) cells have no outgoing moves and take their rows from neighbouring cells.
  - Per step at 1°: exit 2.4% drogued, 3.5% undrogued; stranding 0.23% drogued (from 29 cells), 1.95% undrogued (from 89 cells) (mean over cells).
  - Release off Durban (31.5°E, 30.5°S), 1°, after 1 year: drogued 18% still in R, 49% exited E, 18% W, 11% stranded; undrogued 17% in R, 56% E, 19% W, 6% stranded. Across grids 16–20% stays in R, 45–57% exits east, 15–22% west.

- [x] Release points from shipping traffic — `scripts/04_shipping_overlay.py`.
  - Shipping values are a relative density only (units undocumented and implausibly large; pixel coverage drops about 8-fold at exactly 40°S, a source artifact). The data predate the 2024 Red Sea diversions, so Cape traffic is higher today.
  - Mean surface current from drifters (1° cells with ≥ 300 hourly obs): max 1.23 m/s drogued, 1.34 m/s undrogued, in the Agulhas Current.
  - Candidates sit in cells reliable in both matrices (≥ 10 drifters, has outgoing data). Algoa Bay, Cape Town and Saldanha Bay sit 62–76 km offshore in the nearest reliable cell, since their own cells have too few drifters.
  - Near-shore exposure over 30 days (material-days afloat in ocean cells next to land; stranding is reported separately, by `06`):

    | Release point | Drogued | Undrogued |
    |---|---|---|
    | Saldanha Bay | 11.0 | 7.7 |
    | Cape Town | 9.5 | 6.7 |
    | Algoa Bay | 7.5 | 2.3 |
    | Richards Bay | 7.3 | 6.6 |
    | Durban | 6.7 | 4.5 |
    | Busiest lanes | 0.0–5.1 | 0.4–5.3 |

  - Algoa Bay's offshore cell sits in the Agulhas Current, so its undrogued score describes oil released there, not in the bay itself.

- [x] Box scoring — `scripts/05_box_selection.py` (undrogued data): boxes containing the Agulhas core (18–33°E, 40–28°S), ≥ 5° inside the saved area so exits stay observable; data rules ≥ 1M undrogued hourly observations and ≥ 90% of ocean cells with ≥ 10 drifters; scored on shipping density per ocean cell and one-step retention → mean residence time. Box R's scores are listed under "Why this box". Pixel coverage in the shipping raster drops about 8-fold at exactly 40°S, so shipping density south of that line is less trustworthy.

- [x] Drifter bootstrap — `scripts/06_bootstrap.py` (1°, τ = 3.5 d, 1,000 resamples of whole drifters: 800 drogued, 1,100 undrogued; the two types are resampled independently).
  - One-year fate with 95% intervals (drogued / undrogued):

    | Release point | Still in R | Exited west (Atlantic) | Exited east | Stranded |
    |---|---|---|---|---|
    | Durban | 18% (13–24) / 11% (6–17) | 19% (13–25) / 14% (7–23) | 49% (41–56) / 37% (21–58) | 10% (4–18) / 38% (5–64) |
    | Richards Bay | 22% (15–31) / 16% (13–20) | 17% (12–23) / 19% (14–24) | 45% (38–54) / 53% (47–59) | 11% (4–19) / 10% (5–18) |
    | Algoa Bay | 13% (9–18) / 13% (10–15) | 25% (17–33) / 27% (21–34) | 41% (32–51) / 47% (40–54) | 15% (5–28) / 11% (5–18) |
    | Cape Town | 2% (1–4) / 0% (0–1) | 86% (69–97) / 62% (42–84) | 2% (1–4) / 1% (0–3) | 9% (0–27) / 36% (14–57) |
    | Saldanha Bay | 1% (0–1) / 0% (0–0) | 99% (97–100) / 97% (92–100) | 0% (0–1) / 0% (0–0) | 0% (0–0) / 3% (0–8) |

  - Offshore, undrogued material strands far more than drogued material after a year: 24–39% against 1–9% at the lanes south of Madagascar, and 66% (50–82%) against 27% (10–47%) at the Mozambique Channel lane (40.5°E, 16.5°S); the intervals do not overlap at 7 of the 8 lanes (all but 27.5°E). Along the South African coast the two are alike: drogued 10–19% at Durban, Richards Bay, Algoa Bay and the 27.5°E lane against undrogued 10–38%, with overlapping intervals. Of the 18 drifters that reached the coast drogued, 11 grounded on the South African south and east coast (19–29°E), 4 in southern Mozambique, 2 on Madagascar and 1 in Namibia, so drogued stranding is concentrated there. The drogued figure is a lower bound (most grounded drifters lost their drogue long before grounding, see `03`).
  - Exits agree between the versions along the South African coast (Algoa Bay, Richards Bay, Durban, the 27.5°E lane): the 95% interval of the difference includes 0 for still in R and for both exit edges at all four. At the eastern lanes, drogued material stays in R more at 43–48°E (45–57% vs 27–32%) and leaves east more at 47–55°E (46–78% vs 23–43%); undrogued material strands on Madagascar instead.
  - Durban's and Cape Town's undrogued stranding intervals are wide (5–64%, 14–57%), since their coastal rows rest on few stranded drifters.
  - The intervals are not corrected for the number of points and metrics compared, so a single "differ" flag is weak evidence on its own.
  - Row stability (median bootstrap TVD > 0.2 = unstable): drogued 258 of 950 rows (27%), undrogued 54 of 962 (6%). Unstable drogued rows are concentrated in the Mozambique Channel and south of Madagascar. The two reliability checks pick different rows (drogued: 174 unstable rows pass the 10-drifter flag, 124 flagged rows are stable), so both are reported. The TVD understates the uncertainty of rows with very few drifters (a single-drifter row is either resampled whole or dropped), so the 10-drifter flag is the floor.
  - Outputs: `data/bootstrap_release_1deg_3p5d_10E-55E_45S-15S.csv`, `data/bootstrap_difference_…csv`, `data/bootstrap_rows_…npz`, `figures/agulhas_bootstrap_fates_…png`, `figures/agulhas_bootstrap_rows_…png`.

- [x] Undrogued wind slip — `scripts/07_wind_slip.py` (1°, undrogued − drogued mean velocity, per-drifter means so no drifter dominates a cell; 746 cells with ≥ 10 drifters of each type; 95% intervals from 500 resamples of whole drifters).

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

- [x] Release-point maps at 1 week / 1 month / 1 year — `scripts/17_release_point_maps.py` (1°, τ = 3.5 d): one figure per point from `04`, drogued and undrogued, with the share afloat in R, each exit and stranded, and 95% intervals from `06`.
  - Durban (drogued) after a year: 18% (13–24%) still in R, 49% (41–56%) left east, 19% (13–25%) west and 10% (4–18%) stranded, as expected.
  - Undrogued material strands far more than drogued offshore: after a year 24–66% from the Mozambique Channel lane and the lanes south of Madagascar (drogued 1–27%). Along the South African coast the two are alike (drogued 9–19%, undrogued 10–38%), and Cape Town and Durban's undrogued shares have wide intervals (5–64%), since coastal stranding rests on a few grounded drifters (see the oil version).
  - Outputs: `data/release_point_maps_1deg_3p5d_10E-55E_45S-15S.csv`, `figures/agulhas_release_{nn}_{point}_…png`.

- [x] Held-out validation — `scripts/10_validation.py` (1°, τ = 3.5 d, 5-fold by drifter: 817 drogued / 1,170 undrogued drifters each held out once; daily test starts followed at whole τ steps, exits and stranding absorbing).

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

- [x] Sensitivity to τ and grid size — `scripts/18_sensitivity.py`: reference 1°, τ = 3.5 d against τ = 2 and 5 d at 1°, and 0.5° and 2° at τ = 3.5 d, each with its own held-out validation from `10`; the 13 release points' one-year fates (still in R, exited W, exited E, stranded) are checked against the reference's 95% bootstrap interval from `06`.

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

- [x] Seasonal matrices — `scripts/19_seasonal_matrices.py`.
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

- [x] Chapman–Kolmogorov consistency — `scripts/20_chapman_kolmogorov.py`.
  - Compares `P(3.5 d)^2` with an operator estimated directly from 7-day drifter transitions.
  - Common reliable rows: 731 drogued and 892 undrogued.
  - Median row-wise TVD is 0.243 for drogued and 0.173 for undrogued drifters; the 90th percentiles are 0.374 / 0.261.
  - TVD is below 0.25 for 53.2% / 87.1% of common reliable rows and below 0.50 for 98.9% / 99.8%.
  - Almost all of that TVD is sampling noise in the direct 7-day rows. Resampling whole drifters 1,000 times (782 drogued, 1,092 undrogued) gives a median noise TVD of 0.224 / 0.169 per row, so the TVD beyond noise is 0.015 / 0.002 (median). Only 1.8% / 3.7% of rows exceed the 95th percentile of their own noise, about the share expected by chance.
  - The noise level leaves out the smaller noise in `P(3.5 d)^2`, which would raise it further, so the comparison leans towards flagging rows. Within sampling noise the 3.5-day operator is Markov at 1°: the drogued rows' larger TVD comes from their thinner data, not from memory beyond τ.

- [x] Coastal pooling — `scripts/21_coastal_pooling.py`.
  - At 1°, τ = 3.5 d, a conservative Natural Earth 50m land-proximity filter removes four offshore candidates from the original 103 coastal cells, leaving 99 cells.
  - The 99 coastal cells are grouped into 44 contiguous coastal segments, capped at four neighbouring cells per segment.
  - Before pooling, 35.0% of active drogued coastal rows and 48.9% of active undrogued coastal rows are reliable (≥ 10 distinct drifters).
  - After pooling, 60.6% / 72.7% of coastal cells belong to a reliable pooled segment for drogued / undrogued drifters.
  - 29 of 44 segments (65.9%) are reliable for both drogue types.
  - Median segment size is two coastal cells; 15 segments remain thin and explicitly flagged.
  - The geographic filter only removes candidates from the existing coastal-cell set; it does not add potentially missing near-land cells.
  - Scripts `04`, `06`, `09` and `12` retain the original coastal-cell definition.
  - Pooling does not alter or smooth the main transition matrix.

- [x] ERA5 winds — `scripts/08_fetch_era5.py`: 10 m wind (0.5°, 6-hourly, 1995–2022) from the Copernicus CDS API, interpolated to 10.03M drifter positions (99.9% of positions in R).

- [x] Oil version — `scripts/09_oil_wind.py` (1°, τ = 3.5 d).
  - Undrogued windage: 2.0% of the 10 m wind (95% interval 1.81–2.19%, resampling whole drifters over the 746 cells with ≥ 10 drifters of each type), from regressing the per-cell slip (undrogued − drogued velocity) on the ERA5 wind at undrogued positions. The estimate holds without the fast-current cells (2.2% without cells whose mean drogued current exceeds 0.3 m/s, and 2.0–2.3% for cutoffs from 0.2 to 1 m/s) and east of 35°E (2.1%).
  - The oil matrix adds 1.5%: a median shift of 25 km per step (90th percentile 42 km); 0.01% of shifted ends land on land or unvisited cells and keep their drifter end.
  - Share of released parcels stranded within 30 days, with 95% intervals from resampling whole drifters (oil in `09`, undrogued from `06`). "Grounded" is the number of drifters that ran aground starting from the release cell itself; stranding also arrives from neighbouring cells, so a point with none can still strand.

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
  - The oil and check matrices differ by TVD 0.19–0.42 at 30 days. Most of the gap at Durban, Cape Town and the Mozambique Channel lane is stranding: the check matrix strands 6–11% there within 30 days against 39–41% for oil, since few drifters were still drogued when they reached the coast (see `22`). Along the South African south coast the check strands as much as oil or more (Algoa Bay 9%, the 27.5°E lane 14%), where most drogued groundings happened. For oil still afloat the two agree within 0.16–0.28 everywhere except the Mozambique Channel lane (0.35), where drogued rows are thin. The oil matrix is the working version.
  - Outputs: `data/P_oil_1deg_3p5d_10E-55E_45S-15S.npz`, `data/P_oil_check_…npz`, `data/oil_compare_…csv`, `figures/agulhas_oil_durban_30d_…png`.

- [x] Whole-box structure — `scripts/11`–`16` (all grids at τ = 3.5 d, both drogue types; numbers below at 1°, drogued / undrogued, over reliable cells).
  - Support (`11`, `12`): 78% / 93% of active rows reliable (0.5°: 52% / 91%; 2°: 90% / 94%). Coastal cells (land neighbour on the grid): 33% / 47% reliable, against 82% / 98% offshore.
  - Retention hotspots (`13`): drogued at 41–44°E, 30–33°S, 63–65% still in R after a year (95% intervals about 48–74%); 70 of 75 hotspot cells stay in the top 10% in ≥ 50% of 1,000 drifter resamples. Undrogued at 48–51°E, 30–33°S, 55–56% (49–60%), 88 of 90 stable. Where both grids are reliable, 1° and 2° hotspots agree (Jaccard 76% / 73%); 0.5° drogued agrees less (37–41%).
  - Residence time (`14`): mean 223 / 203 days, median 223 / 213, longest in the hotspots (518 / 459 days); grids within about 40 days.
  - Exit zones (`15`): eventual fate E 69% / 62%, W 18% / 17%, S 6% / 4%, N 2% / 3%, stranded 4% / 15%; east is the most likely fate in 83% / 75% of cells, stranding in 11% of undrogued cells (Mozambique Channel coast, east of Madagascar) and in no drogued cell.
  - Almost-invariant regions (`16`): four for both drogue types (retroflection and Atlantic side; Natal coast and Mozambique Channel; south of Madagascar and Return Current; east of Madagascar), keeping 81–94% over one step, 54–70% over 30 days and 1–8% over a year. Three keep 13–23 points more over 30 days than longitude bands of the same sizes; the western one matches its band. The eigenvalues fall off smoothly, so four regions is a choice.
  - Outputs: `data/whole_box_support_…csv`, `data/coastal_support_…csv`, `data/retention_*_…csv`, `data/residence_time_*…csv`, `data/exit_zones_*…csv`, `data/almost_invariant_*…csv` and the matching `figures/agulhas_*` maps.


- [x] Stranding by coast — `scripts/23_coastal_stranding.py` (1°, τ = 3.5 d; drogued, undrogued, oil and the oil check; 1,000 resamples of whole drifters).
  - Six named coasts (`transport.coast_of`), each stranding move taking the coast of its drifter's last observation. The coast columns sum to each operator's stranded column exactly (checked on every build).

    | Coast | Grounded drifters: drogued / undrogued | Mean eventual stranding over reliable cells: drogued / undrogued / oil |
    |---|---|---|
    | Atlantic and Cape (Namibia to Cape Agulhas, 20°E) | 5 / 12 | 0.9% / 1.1% / 1.2% |
    | South African south and east coast (to 26.9°S) | 7 / 17 | 2.0% / 2.1% / 2.1% |
    | Southern Mozambique (to 21°S) | 4 / 28 | 1.1% / 5.3% / 6.0% |
    | Central and northern Mozambique | 0 / 18 | 0.0% / 3.6% / 4.6% |
    | West and south Madagascar (to 47°E) | 1 / 18 | 0.2% / 1.5% / 1.5% |
    | East Madagascar | 1 / 44 | 0.1% / 1.5% / 1.7% |

  - Every drogued coast is thin (< 10 grounded drifters), so drogued stranding by coast is indicative only and a lower bound (`22`).
  - Material travels far before grounding. From the lanes south and east of Madagascar the largest share strands on southern Mozambique (oil 14–22%, undrogued 8–17%), not on Madagascar: from the 47.5°E lane oil strands 20% (13–26%) there against 11% (6–17%) on west and south Madagascar.
  - The added windage brings oil ashore sooner rather than much more often. From the 47.5°E lane oil strands 49% (42–55%) in 95 days (81–111) and undrogued material 39% (32–47%) in 151 days (127–178): the stranding intervals overlap and the time intervals do not.
  - Mozambique Channel lane: oil strands 71% (53–84%), 53% (33–70%) on central and northern Mozambique, in 35 days (22–56). Drogued material strands 29% (12–48%), mostly on southern Mozambique, in 136 days (88–237).
  - Cape Town and Durban rest on few groundings: undrogued Durban 39% (7–65%), Cape Town 36% (14–57%). These say a large share may strand, not at what rate.
  - A resample can leave a few thin cells (1–6 drifters, off Angola, Namibia and western Madagascar) with no way out; material entering them stays in R (`transport.stranding_chain`), so every resample is kept.
  - Outputs: `data/P_{drogued,undrogued,oil,oil_check}_coast_…npz`, `data/coastal_stranding_points_…csv`, `data/coastal_stranding_cells_…csv`, `data/coastal_stranding_support_…csv`, `data/coast_counts_…npz`, `figures/agulhas_coastal_stranding_…png`.

- [x] Traffic-weighted exposure — `scripts/24_traffic_exposure.py` (1°, τ = 3.5 d, 1,000 resamples of whole drifters).
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

- [x] Release-risk atlas — `scripts/25_risk_atlas.py` (1°, τ = 3.5 d; drogued, undrogued and oil; 1,000 resamples of whole drifters).
  - All 970 ocean cells as release cells: the share afloat in R, exited across each edge and stranded on each coast at 7, 30 and 365 days and eventually, and days to strand, each with a 95% interval and the cell's reliability flags (`flagged`, `empty`, `row_drifters`). Outcomes sum to 1 for every cell and horizon.
  - Stranding from a typical reliable cell is low even after a year (median 0.8% drogued, 3.1% undrogued, 4.0% oil), with a long tail: the top 10% of cells reach 9% / 45% / 56%.
  - The highest-risk cells for oil at 30 days lie on the Mozambique coast at 35–36°E, 19–22°S (78–97%, intervals from 58%) and off north-east Madagascar at 49.5°E, 16.5°S (80%, 55–100%).
  - Outputs: `data/risk_atlas_…csv` (139,680 rows), `figures/agulhas_risk_atlas_…png`.

- [x] Hypothesis tests — `scripts/26_hypothesis_tests.py` (1°, τ = 3.5 d; 1,000 resamples of whole drifters), each hypothesis under its stated decision rule.
  - H1 supported. P(exit E) − P(exit W), eventual fate averaged over reliable cells, is 0.51 (0.45–0.57) drogued and 0.45 (0.41–0.49) undrogued. Cell by cell, east is the clear majority (85% / 88% of cells have an interval above zero), west holds in 12% / 11%, around the Cape and the west coast, and 4% / 2% are unclear.
  - H2 supported, with a caveat. Within a year, undrogued stranding exceeds drogued with non-overlapping intervals in 7 of 8 lane cells; the result is the same for eventual stranding. Examples: 66% (50–82%) against 27% (10–47%) at 40.5°E, 16.5°S and 24% (17–30%) against 0.9% (0.3–1.9%) at 54.5°E, 22.5°S. The exception is the South African coast at 27.5°E, 33.5°S, where drogued is higher but uncertain (19%, 5–36%, against 10%, 5–16%). Drogued stranding is a lower bound, since few drifters still have a drogue at the coast (`22`), so the rule cannot separate how much of the gap is windage.
  - H3 supported from 17.5 days. Folds won out of 5 against mean-current advection, the harder baseline: 14 d 4 drogued / 3 undrogued, 17.5 d 4 / 5, 21–28 d 5 / 5. The operator beats persistence in all folds at every one of these horizons. At 14 days the undrogued operator misses the rule, but with a mean margin of 1–2 km. Margins over advection stay small at 28 days (252 against 267 km, 289 against 299 km), so its main advantage remains calibrated probabilities.
  - Outputs: `data/hypothesis_h{1,2,3}_…csv`, `figures/agulhas_hypothesis_tests_…png`.

- [x] Second region — the core pipeline on the Gulf Stream, box 80–40°W, 25–45°N, to test that region, grid and lag really are inputs. `00 --fetch-box -90 -30 15 55 --out data/gulfstream_gdp1h_subset.nc` saves 23.6M hourly positions from 3,022 drifters (11.8M positions in the box); `01`–`04`, `06`, `07` and `10`–`22` then run with `--box -80 -40 25 45 --data …` and no region-specific code changes. Not run: `05` (searches around the Agulhas core), `08`–`09` (would need ERA5 for this area) and `23`–`25` (the named coasts and routes describe the Agulhas).
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
  - Chapman–Kolmogorov: 5.1% / 2.1% of rows are above their 95% noise level, so the Markov assumption holds as in R.
  - Differences from R: long-run results depend on the grid (mean time in the box 608 → 478 → 350 days drogued and 427 → 379 → 305 days undrogued from 0.5° to 1° to 2°, against about 40 days of spread in R), and drogued transport is seasonal beyond sampling noise in 5 of 6 season pairs (undrogued 6 of 6; drogued 0 of 6 in R). The 1° annual baseline chosen for R therefore should not be assumed elsewhere without rechecking.

Open questions:
- Drogued stranding rests on the few drifters still drogued when they reached the coast, since 121 of the 139 grounded drifters lost their drogue more than τ before grounding (`22`). The drogued stranding estimate is therefore treated as a lower bound rather than borrowing the undrogued rate.
- Coastal sparsity is mitigated with short pooled coastal segments, while the main transition matrix remains unchanged and thin segments remain flagged.
- The annual 1°, τ = 3.5 d operator remains the baseline. Drogued transport shows no seasonality beyond sampling noise and undrogued transport only a small one (permutation test in `19`), so the seasonal operators serve as a sensitivity check.
- Next: rebuild the oil matrix at 3%, 3.5% and 4% wind, and spill hindcasts.
