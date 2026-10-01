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
- Oil (undrogued) stays in R 136 days on average; 6.5% is still inside after a year.
- Contains 97% of the Cape Basin eddy energy (Agulhas rings) and the Mozambique Channel eddies.
- Covers the coastal route around the Cape, the Mozambique Channel routes and the route south of Madagascar.
- Passes the data rules with margin: 4.12M undrogued hourly observations, 92% of ocean cells seen by ≥ 10 drifters.


What we track:
Shipping container:
- Modelled with drogued drifters, which follow ~15 m currents with almost no wind effect (windage), matching a mostly submerged or flooded container.

Oil spill:
- Surface oil drifts at roughly current + 3–3.5% of wind speed.
- Modelled with undrogued drifters, which ride at the surface and slip downwind of drogued drifters by ~10–12 cm/s in the westerly and trade-wind bands (`scripts/07_wind_slip.py`), roughly 1% of typical wind speeds and about a third of oil's windage. The exact ratio to wind comes from ERA5.
- Oil mass decays as m(t) = m0 · exp(−λt) with λ ≈ 1/14 per day (evaporation + emulsification), so results answer both "where is it?" and "how much surface oil remains after X days?".
- ERA5 10 m winds (Copernicus Climate Data Store, cds.climate.copernicus.eu) supply the wind term. Undrogued motion already contains ~1% windage, so the full 3–3.5% is never added on top of it (see open questions for how the term is applied).

Drogued and undrogued matrices are built and reported side by side and never pooled: an "all drifters" matrix is 46–84% undrogued depending on the cell, so its wind effect would follow sampling, not physics.


Data:
- NOAA GDP hourly drifter data (v2.01, 1995–2022 in R), streamed from the public AWS S3 zarr store (the CloudDrift `gdp1h` source) by `scripts/00_fetch_agulhas_subset.py` into `data/agulhas_gdp1h_subset.nc`.
  - Native hourly resolution; positions interpolated across > 6 h between satellite fixes are dropped (0.3%).
  - The saved area (5°W–65°E, 60–5°S) covers R plus a margin, so positions just after a drifter leaves R are kept.
  - Key fields: `lon`, `lat`, `time`, `ve`/`vn` (velocity), `drogue_status`, `gap`, `rowsize`, `id`.
  - Split by `drogue_status`: drogued → container matrix; undrogued → oil matrix.
- Commercial shipping density: World Bank / IMF Global Shipping Traffic Density, Commercial layer (AIS, Jan 2015–Feb 2021, CC-BY 4.0).
- ERA5 10 m wind (u10, v10), hourly, R plus a small margin, for the oil wind term.


Method (Ulam's method / transition matrix):
1. Grid R into cells at 0.5°, 1° and 2°; 1° is the working grid. Land cells are masked.
2. Transition time τ = 3.5 days, longer than the Lagrangian decorrelation time (~1.2–1.4 days) so the Markov assumption holds; drifters move about one cell per step.
3. For every drifter position at time t in cell i, find where it is at t + τ (cell j). Count transitions C[i, j].
4. Four absorbing "exited R" states (W, E, S, N) record which edge material leaves by.
5. Row-normalise: P[i, j] = C[i, j] / Σ_j C[i, j]. Each row is "where does material in cell i go in τ days".
6. Iterate: distribution after n steps = p₀ · Pⁿ. 1 week = 2 steps, 1 month ≈ 9 steps, 1 year ≈ 104 steps.
7. Oil version: the mass decay factor applied each step, plus an ERA5 wind term (see open questions).

Reliability:
- Rows built from < 10 distinct drifters are flagged; cells with no outgoing data hold mass in place and are flagged separately.
- `scripts/06_bootstrap.py` resamples whole drifters (never observations) to give 95% intervals for release-point results and a row-stability check (median bootstrap TVD > 0.2 = unstable).
- Hourly positions are strongly correlated; the independent sample size is the number of drifters, not the observation count.
- At 1 year most material has left R, so answers are "probability still in R" plus where it exited.


Analysis:
Release points: 5 hotspots plus the busiest reliable shipping cells (`scripts/04_shipping_overlay.py`):
- Richards Bay, Durban (major ports, in the core of the current)
- Algoa Bay (ship-to-ship bunkering, off Gqeberha)
- Cape Town, Saldanha Bay (Atlantic side)
- Busiest shipping cells along the coastal route, in the Mozambique Channel and south of Madagascar

For each: map of the probability distribution after 1 week, 1 month and 1 year, plus the fraction left in R and where it exited, with bootstrap intervals.

Whole-box structure:
- Where R gathers material: stationary / quasi-stationary distribution (leading left eigenvector of P, conditioned on staying in R); high values = accumulation zones.
- Where R loses material: exit rate per cell (the exit columns), and residence time (expected steps before leaving).
- Which parts feed which: eigenvectors near eigenvalue 1 / spectral clustering → almost-invariant sets (regions that mostly keep material to themselves = "cut off"); flows between clusters show who feeds whom.
- Source vs sink cells: row and column sums of P.


Validation:
- Held-out drifters (whole trajectories): their actual positions after 1 week / 1 month against the matrix predictions.
- Sensitivity to grid size (0.5°, 1°, 2°) and τ (2 and 5 days).
- Seasonal (summer / winter) matrices against the all-year matrix.
- Drifter coverage over time (temporal plot), so no few years dominate the matrix.
- Known features: current path, retroflection, rings heading into the Atlantic.


Final output:
- Transition matrices saved as `data/P_{drogued|undrogued}_{res}deg_{tau}d_{box}.npz` (P, counts, grid edges, τ, flags), with a usage example in the README.
- Two versions: container (drogued) and oil (undrogued, with mass decay and wind term).
- Maps for the release points at 1 week / 1 month / 1 year.
- Maps of accumulation zones, exit zones, residence time and almost-invariant regions.


Progress:
- [x] Data subset — `scripts/00_fetch_agulhas_subset.py`: 17.0M hourly positions from 2,942 drifters in the saved area; in R 5.95M (1.83M drogued, 4.12M undrogued) from 1,503 drifters.
- [x] Exploration and grid coverage — `scripts/01_agulhas_gdp_exploration.py`, `scripts/02_grid_coverage.py`.
  - Ocean cells in R: 3,780 / 970 / 263 at 0.5° / 1° / 2° (the 45°-wide box at 2° ends in a partial 1° column).
  - Cells with ≥ 10 distinct drifters: drogued 52% / 78% / 90%, undrogued 90% / 92% / 93%.
  - Drogued data is thin on the Agulhas Bank shelf, off Namibia and in parts of the Mozambique Channel.
- [x] Region, grid and lag are parameters (`scripts/config.py`, `--box --res --tau --data` on `01`–`04`); matrix file names include res, τ and box.
- [x] Transition matrices, both drogue types, 0.5° / 1° / 2° — `scripts/03_transition_matrix.py`, helpers in `scripts/transport.py`.
  - Transitions: 1.77M drogued, 4.06M undrogued.
  - Flagged rows (< 10 drifters): drogued 50% / 24% / 12%, undrogued 11% / 9% / 7%. At 1°, 20 (drogued) and 15 (undrogued) cells have no outgoing data.
  - Exit per step at 1°: 2.4% drogued, 3.3% undrogued.
  - Durban release, 1°, after 1 year: drogued 20% still in R, 53% exited E, 22% W; undrogued 18% in R, 57% E, 23% W. Across grids 18–22% stays in R, 49–57% exits east, 18–24% west.
- [x] Release points from shipping traffic — `scripts/04_shipping_overlay.py`.
  - Shipping values are a relative density only (units undocumented and implausibly large; source artifact at ~40°S). The data predate the 2024 Red Sea diversions, so Cape traffic is higher today.
  - Mean surface current from drifters (1° cells with ≥ 300 hourly obs): max 1.23 m/s drogued, 1.34 m/s undrogued, in the Agulhas Current.
  - Candidates sit in cells reliable in both matrices (≥ 10 drifters, has outgoing data). Algoa Bay, Cape Town and Saldanha Bay sit 60–76 km offshore in the nearest reliable cell, since their own cells have too few drifters.
  - Near-shore exposure over 30 days (material-days in ocean cells next to land; the matrix has no beaching state, so this is a proxy):

    | Release point | Drogued | Undrogued |
    |---|---|---|
    | Saldanha Bay | 11.0 | 8.2 |
    | Cape Town | 10.7 | 11.1 |
    | Algoa Bay | 8.3 | 2.7 |
    | Richards Bay | 7.6 | 7.1 |
    | Durban | 7.0 | 7.6 |
    | Busiest lanes | 0.0–5.2 | 0.4–7.7 |

  - Algoa Bay's offshore cell sits in the Agulhas Current, so its undrogued score describes oil released there, not in the bay itself.
- [x] Box scoring — `scripts/05_box_selection.py` (undrogued data): boxes containing the Agulhas core (18–33°E, 40–28°S), ≥ 5° inside the saved area so exits stay observable; data rules ≥ 1M undrogued hourly observations and ≥ 90% of ocean cells with ≥ 10 drifters; scored on shipping density per ocean cell and one-step retention → mean residence time. Box R's scores are listed under "Why this box". The 40–41°S band coincides with the shipping raster artifact, so shipping density near that latitude is less trustworthy.
- [x] Drifter bootstrap — `scripts/06_bootstrap.py` (1°, τ = 3.5 d, 1,000 resamples of whole drifters: 799 drogued, 1,082 undrogued; the two types are resampled independently).
  - One-year fate with 95% intervals (drogued / undrogued):

    | Release point | Still in R | Exited west (Atlantic) | Exited east |
    |---|---|---|---|
    | Durban | 20% (15–27) / 20% (15–28) | 23% (16–31) / 24% (18–30) | 52% (44–60) / 55% (48–61) |
    | Richards Bay | 25% (18–34) / 18% (15–23) | 21% (14–28) / 23% (17–29) | 49% (41–57) / 57% (51–62) |
    | Algoa Bay | 16% (12–23) / 14% (11–19) | 32% (21–41) / 34% (26–42) | 46% (38–54) / 50% (43–56) |
    | Cape Town | 2% (1–18) / 1% (0–33) | 95% (80–98) / 98% (65–100) | 2% (1–5) / 1% (0–5) |
    | Saldanha Bay | 1% (0–1) / 0% (0–4) | 99% (97–100) / 100% (96–100) | 0% (0–1) / 0% (0–0) |

  - Along the South African coast (the five hotspots and the 27.5°E lane), drogued and undrogued give the same one-year fate: the 95% interval of the difference includes 0 for every metric. 30-day coastal exposure differs only at Algoa Bay (8.3 vs 2.7 material-days).
  - The two versions differ at the eastern lanes (43–55°E, south of Madagascar). At 50–55°E, drogued material leaves east more (69–78% vs 29–46%) and stays in R less (22–30% vs 48–63%); at 43–46°E, drogued material stays in R more (61–62% vs 46–49%).
  - Cape Town's intervals are wide (still in R 0–33% undrogued), since its cell sits next to thin coastal rows.
  - The intervals are not corrected for the number of points and metrics compared, so a single "differ" flag is weak evidence on its own.
  - Row stability (median bootstrap TVD > 0.2 = unstable): drogued 246 of 950 rows (26%), undrogued 60 of 955 (6%). Unstable drogued rows are concentrated in the Mozambique Channel and south of Madagascar. The two reliability checks pick different rows (drogued: 166 unstable rows pass the 10-drifter flag, 128 flagged rows are stable), so both are reported. The TVD understates the uncertainty of rows with very few drifters (a single-drifter row is either resampled whole or dropped), so the 10-drifter flag is the floor.
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
- [ ] Grid size for the drogued version (0.5° has 50% flagged rows, so 1° or 2°).
- [ ] Release-point maps at 1 week / 1 month / 1 year.
- [x] Held-out validation — `scripts/10_validation.py` (1°, τ = 3.5 d, 5-fold by drifter: 817 drogued / 1,170 undrogued drifters each held out once; daily test starts followed at whole τ steps, exits absorbing).

    | Type | Horizon | Test starts | Markov error | Advection (cell centre) | Persistence (cell centre) | Log score Markov / climatology | Exit pred / obs |
    |---|---|---|---|---|---|---|---|
    | Drogued | 3.5 d | 73,856 | 73 km | 73 km | 73 km | −2.89 / −6.80 | 3.2% / 3.1% |
    | Drogued | 7 d | 71,573 | 110 km | 113 km | 115 km | −3.08 / −6.73 | 5.9% / 5.9% |
    | Drogued | 28 d | 62,113 | 252 km | 269 km | 295 km | −3.90 / −6.25 | 18.6% / 19.1% |
    | Undrogued | 3.5 d | 169,318 | 82 km | 82 km | 89 km | −2.52 / −6.73 | 2.7% / 2.6% |
    | Undrogued | 7 d | 167,456 | 126 km | 128 km | 142 km | −3.04 / −6.67 | 4.9% / 4.9% |
    | Undrogued | 28 d | 157,914 | 289 km | 303 km | 373 km | −4.08 / −6.22 | 15.4% / 15.9% |

  - Errors are median distances from the forecast (Markov: centre of mass of the predicted distribution still in R) to the actual position. Baselines start from the cell centre, the same information the matrix has; from the exact start position they are ~10 km better at 3.5 d, which is the 1° grid's own error.
  - The matrix ties the baselines at 3.5 d and beats them by 28 d (5–6% vs advection, 15–23% vs persistence). Its main value is the distribution: the log score is far above climatology at every horizon, and the predicted share leaving R matches the observed share (calibration within a few points across the full 0–1 range at 28 d).
- [ ] Sensitivity to τ (2 and 5 days) and grid size; seasonal matrices.
- [ ] ERA5 winds and the oil version (mass decay + wind term).
- [ ] Eigen / clustering analysis for whole-box structure.

Open questions:
- At 1° the drogued matrix has 24% of rows flagged (< 10 drifters) and 26% unstable under the bootstrap, mostly in the Mozambique Channel and south of Madagascar. Merge or smooth thin cells there, or use 2° for the drogued version?
- Oil wind term: add the missing windage (~2–2.5% of ERA5 wind) to the undrogued matrix, or simulate particles with drogued currents + 3–3.5% wind?
- Is τ the same for both versions?
- How much weight goes on seasonal matrices vs one all-year matrix?
