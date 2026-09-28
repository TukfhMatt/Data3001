Something goes into the water at a chosen point in R: a container off a ship, a life raft, a slick of oil. Where has the surface flow taken it a week later? A month later? A year? At the scale of the whole box, which parts of R feed which others, and which are cut off from the rest?

Product. A surface-transport operator for R. E.g. a transition matrix (or similar) someone else could pick up and iterate forward, together with what it says about where a few chosen release points end up, and about where R as a whole gathers material and where it loses it.


Regions:

---
Agulhas region:
Box R: **10°E–55°E, 45°S–15°S** — chosen by the group on 28 Sep 2026 (set in `scripts/config.py`). History: 10–40°E, 45–25°S → 5–50°E, 50–20°S → this box. The notes below record the 5–50°E box that preceded it.

- box: 1,147,386 obs at 6-hourly (6,883,838 hourly) from 1,631 drifters, 1995–2022.
- Also brings in more of the inflow (southern Mozambique Channel, south of Madagascar), the Return Current down to 50°S, and the Atlantic leakage region west to 5°E.

Why here:
- Agulhas Current is one of the strongest western boundary currents (up to ~2 m/s), flowing south-west along the South African east coast.
- Retroflection south of Africa (~16–20°E): most of the flow turns back east as the Agulhas Return Current (~38–40°S); some leaks into the South Atlantic via Agulhas rings. Expect distinct "feed" and "cut-off" structure — good for the whole-box question.
- Busy shipping route around the Cape of Good Hope (more traffic since Red Sea diversions in 2024), so container loss and oil spills are realistic scenarios.
- Well sampled by GDP drifters (check counts from exploration script).

What to track?
Shipping container:
- Would be easiest to track. The wind has almost no effect on the movement of the container (windage), and hence the current will be the only impact. 
- Caveat: this holds for a mostly submerged / flooded container. A container floating high has non-trivial leeway (~1–3% of wind). Could treat as a sensitivity case.
- Data: drogued GDP drifters (follow ~15 m currents) are the closest match.

Life raft:
- High windage — leeway roughly 3–4% of wind speed, plus a crosswind component. Needs ERA5 wind like oil.
- Probably out of scope unless time permits; mention as extension.

Oil Spill:
-  Oil at the surface drifts at roughly current + 0.035 × wind. You'll need ERA5 10 m wind fields alongside your current data.
- ERA5 is freely available via the Copernicus Climate Data Store (cds.climate.copernicus.eu)
- assign each particle a mass that decreases over time (exponential decay approximating evaporation + emulsification). This lets you answer "how much surface oil remains after X days?" as well as "where is it?"
- Undrogued GDP drifters already feel some wind + Stokes drift at the surface, so they are a partial oil proxy. Be careful not to double count wind if adding 0.035 × wind on top of undrogued motion.


Data:
- NOAA GDP 6-hourly ragged array (`data/gdp6h_ragged_dec24.nc`), via CloudDrift format.
  - Key fields: `lon`, `lat`, `time`, `ve`/`vn` (velocity), `drogue_status`, `rowsize`, `id`.
  - Split by `drogue_status`: drogued → container/current matrix; undrogued → surface/oil matrix.
- ERA5 10 m wind (u10, v10), hourly or 6-hourly, same box + small margin. Only needed for oil / life raft.
- Optional gridded currents to fill gaps / compare: Copernicus Marine (CMEMS) surface currents or OSCAR.


Method (Ulam's method / transition matrix):
1. Grid R into cells (start with 1°, try 0.5°). Mask land cells.
2. Pick a transition time τ (e.g. 2–5 days). Should be longer than the Lagrangian decorrelation time (~1–3 days) so the Markov assumption is reasonable.
3. For every drifter position at time t in cell i, find where it is at t + τ (cell j). Count transitions C[i, j].
4. Add an "outside R" state for drifters that leave the box (absorbing). Keep track of what fraction leaves through each edge.
5. Row-normalise: P[i, j] = C[i, j] / Σ_j C[i, j]. Each row is "where does material in cell i go in τ days".
6. Iterate: distribution after n steps = p₀ · Pⁿ. So 1 week ≈ 7/τ steps, 1 month ≈ 30/τ, 1 year ≈ 365/τ.
7. Oil version: build matrix from undrogued drifters (or currents + 0.035 × ERA5 wind with simulated particles), then multiply by the mass decay factor each step.

Things to watch:
- Cells with few observations → noisy rows. Set a minimum count threshold, merge cells, or flag them.
- Seasonality: Agulhas varies through the year. Compare an all-year matrix against summer/winter matrices.
- Drifter coverage changes over time (see temporal plot) — check matrix isn't dominated by a few years.
- Long horizons (1 year): most material will have left R, so answers become "probability still in R" plus where it exited.


Analysis:
Release points (pick 3–5), e.g.
- Off Durban (major port, in the core of the current)
- Off Port Elizabeth / Gqeberha
- Agulhas Bank / Cape Agulhas (retroflection)
- Off Cape Town (shipping lane, Atlantic side)
- Offshore in the Agulhas Return Current

For each: map of probability distribution after 1 week, 1 month, 1 year, plus fraction left in R and where it exited.

Whole-box structure:
- Where R gathers material: stationary / quasi-stationary distribution (leading left eigenvector of P, conditioned on staying in R) — high values = accumulation zones.
- Where R loses material: exit rate per cell (the "outside" column), and residence time (expected steps before leaving).
- Which parts feed which: eigenvectors near eigenvalue 1 / spectral clustering → almost-invariant sets (regions that mostly keep material to themselves = "cut off"); flows between clusters show who feeds whom.
- Source vs sink cells: compare row and column sums of P.


Validation:
- Hold out some drifters (e.g. random 20% of trajectories, or recent years). Compare their actual positions after 1 week / 1 month to the matrix predictions.
- Sensitivity to grid size and τ — results should not change a lot.
- Sanity check against known features (current path, retroflection, rings heading into the Atlantic).


Final Output:
- Transition matrix iterating through time to show where the output ends up at specific times (e.g 1 week, 1 month, 1 year)
- Saved as a file others can load (e.g. `.npz` / NetCDF with P, grid edges, τ, and which drifters it was built from) + short usage example.
- Two versions: current-only (container) and surface/wind (oil), with mass decay for oil.
- Maps for chosen release points at 1 week / 1 month / 1 year.
- Maps of accumulation zones, exit zones, residence time, and almost-invariant regions.


Next steps:
- [x] Finish exploration: drifter counts per cell, drogued vs undrogued split, temporal coverage.
  - Data: GDP hourly via CloudDrift (S3 zarr), thinned to 6-hourly, 1995–2022 (`scripts/00_fetch_agulhas_subset.py`).
  - In R (5–50°E, 50–20°S): 1,631 drifters, 1,147k obs (393k drogued, 754k undrogued).
  - Grid coverage (`scripts/02_grid_coverage.py`), % of ocean cells with ≥10 distinct drifters:
    drogued 80% at 1°, 58% at 0.5°; undrogued 95% at 1°, 93% at 0.5°.
  - Drogued data is thin in the north (southern Mozambique Channel, off Namibia), the far north-east near Madagascar, on the Agulhas Bank shelf, and in a band at ~37–42°S west of 15°E.
- [x] Drogued vs undrogued vs all drifters (6-hourly subset, 1°, τ = 3.5 d):
  - Undrogued slip (undrogued − drogued mean velocity, 999 cells): median 9.9 cm/s vs median drogued current 12 cm/s; +9.8 cm/s eastward at 50–40°S (westerlies), −6.9 E / +5.3 N cm/s at 30–20°S (SE trades). Downwind, ≈ 1% of wind speed, about a third of oil's ~3–3.5%.
  - All drifters vs drogued at the 23 release candidates: TVD 0.20 / 0.19 / 0.10 and centre-of-mass gap 39 / 95 / 393 km at 7 / 30 / 365 d; after 1 year 14% (all) vs 21% (drogued) still in R. The pooled matrix is 46–84% undrogued depending on the cell, so its wind effect follows sampling, not physics.
  - Decision: build and report drogued (passive transport) and undrogued (surface + partial wind) side by side; never pool.
- [x] Region, grid and lag are parameters (`scripts/config.py`, `--box --res --tau --data` on `01`–`04`); matrix file names include res, τ and box.
- [x] Grid coverage and matrices at 0.5°, 1° and 2°, both drogue types (hourly data, τ = 3.5 d, default box):
  - Cells ≥ 10 drifters: drogued 59% / 81% / 93%, undrogued 93% / 95% / 96% at 0.5° / 1° / 2°. Ocean cells 4,384 / 1,112 / 293.
  - Flagged rows: drogued 42% / 20% / 7%, undrogued 7% / 5% / 4%. 45° box at 2° → last column is a partial 1° cell.
  - Durban sanity check, still in R after 1 year: drogued 22% / 22% / 28%, undrogued 13% / 12% / 16%; exits east 56–68%, west 13–21% at all grids.
  - `04` now places release points in cells reliable in both matrices and ranks both (drogued 30-day coastal exposure: Saldanha 11.0, Cape Town 10.7, Algoa Bay 8.3, Richards Bay 7.6, Durban 7.0 material-days).
- [ ] Confirm grid size (drogued at 0.5° has 42% flagged rows, so 1° or 2° for the drogued version).
- [x] Choose τ: **3.5 days** (1 week = 2 steps). Velocity decorrelation time ≈ 1.4 d (drogued), 1.2 d (undrogued) at τ = 3.5 d drifters move ~1 cell per step. Sensitivity check later with τ = 2 and 5 d.
- [x] Build first transition matrices (1°, τ = 3.5 d, all years) — `scripts/03_transition_matrix.py`, helpers in `scripts/transport.py`.
  - Saved: `data/P_drogued_1deg_3p5d.npz`, `data/P_undrogued_1deg_3p5d.npz` (1,112 ocean cells + 4 exit states W/E/S/N). Superseded by the box-tagged files below.
  - 383k drogued / 746k undrogued transitions; ~2.3–3.2% of mass leaves R per step.
  - Cells with < 10 drifters flagged: 21% drogued, 5% undrogued. 9–13 cells with no outgoing data hold mass in place.
  - Sanity check (release off Durban): moves SW along the coast; after 1 year 22% (drogued) / 12% (undrogued) still in R, ~60–65% exited east (Return Current), ~1/5 west (leakage into the Atlantic). 
- [x] Choose release points from shipping traffic — `scripts/04_shipping_overlay.py`.
  - Shipping: World Bank / IMF Global Shipping Traffic Density, Commercial layer (AIS, Jan 2015–Feb 2021, CC-BY 4.0). Units undocumented and implausibly large, so used as relative density only; source artifact at ~40°S. Predates the 2024 Red Sea diversions (Cape traffic now higher).
  - Mean surface current from undrogued drifters (1° cells with ≥ 50 obs), max 1.33 m/s in the Agulhas Current.
  - Candidates: 5 hotspots (Richards Bay, Durban, Algoa Bay STS bunkering, Cape Town, Saldanha Bay) + 8 busiest reliable shipping cells. Only reliable matrix cells are used (≥ 10 drifters, has outgoing data); Algoa Bay, Cape Town and Saldanha were moved ~60–76 km offshore to the nearest reliable cell.
  - Ranked by near-shore exposure (oil-days in ocean cells next to land over 30 days; no beaching state in the matrix, so this is a proxy): Cape Town 11.2, Saldanha 8.3, Durban 7.5, Richards Bay 7.0, Algoa Bay 2.6; open-ocean lane cells 0.3–2.5.
  - Algoa Bay's low score reflects the move offshore into the Agulhas Current, not the bay itself (the bay cell has no drifter data). Treat as a known limitation.
  - Outputs: `figures/agulhas_shipping_currents_{drogued,undrogued}.png`, `data/release_candidates.csv` (see the both-matrix update above).
- [x] Bounding-box search — `scripts/05_box_selection.py` (oil / undrogued only).
  - Every 1°-aligned box containing the Agulhas core (18–33°E, 40–28°S) and ≥ 5° inside the subset edge (0–55°E, 55–15°S) — ~98k boxes. Margin needed because exits beyond the subset aren't observed, which inflated retention for edge boxes in a first run.
  - Constraints: ≥ 1M undrogued obs and ≥ 90% of ocean cells with ≥ 10 drifters. Scored on shipping density per ocean cell (risk) and 1-step retention → mean residence time (coherence). Pareto set, then best normalised shipping × retention.
  - 1M at 6-hourly: only very large boxes pass; recommended 0–55°E, 52–15°S (1.11M obs, 165 d residence) hits the search limits — the data requirement just pushes the box to "as big as possible".
  - 1M at hourly (≈ 6 × 6-hourly): real choice. Recommended **10–55°E, 41–15°S**: ~3.4M hourly obs (est.), 90% reliable cells, shipping density 0.116 vs 0.066 for current R (+75%), residence 97 d vs 124 d.
  - Most compact high-shipping option: 14–39°E, 40–18°S (~1.0M hourly obs, 38 d residence).
  - Caveats: the southern edge settles at ~40–41°S, exactly where the shipping raster has a sharp artifact, and it cuts off most of the Agulhas Return Current. Retention always favours bigger boxes; the shipping × retention weighting is a choice, not a statistical test.
- [x] Resolution: **hourly** (native GDP; 6-hourly was only a thinning choice). `00` now saves `data/agulhas_gdp1h_subset.nc` (17.0M obs, 2,942 drifters, 5°W–65°E, 60–5°S — covers every candidate box, so changing the box needs no refetch). Positions with gap > 6 h dropped (43,683, 0.3%).
  - Current R, hourly: 6.86M obs, **4.51M undrogued** from ~1,300 drifters; oil matrix built from 4.46M transitions; 5% of cells flagged. Results unchanged vs 6-hourly (Durban: 12% in R after 1 yr, 65% exit E, 21% W).
  - Box search rerun on hourly data (search area 0–60°E, 55–10°S). The "recommended" box keeps growing to the search limits (now 10–60°E, 41–10°S) because retention always rewards size, so it is not a stable optimum. Named options (shipping density relative to busiest cell in the search area):

    | Option | Box | Oil obs (hourly) | Reliable | Shipping | Residence | Left after 1 yr |
    |---|---|---|---|---|---|---|
    | Current R | 5–50°E, 50–20°S | 4.51M | 95% | 0.020 | 124 d | 5.1% |
    | Shipping-focused | 10–55°E, 41–15°S | 3.39M | 90% | 0.036 | 97 d | 2.2% |
    | Shipping-focused + Return Current | 10–55°E, 45–15°S | 4.12M | 92% | 0.029 | 136 d | 6.5% |
    | Compact core | 14–39°E, 40–18°S | 1.02M | 90% | 0.039 | 38 d | 0.0% |
- [x] Final box: **10–55°E, 45–15°S** (group decision, 28 Sep 2026). Best balance: longest residence (136 d), most oil left after 1 yr (6.5%), 97% of Cape Basin eddy energy (Agulhas rings; current R 100%), most shipping routes (coastal, Mozambique Channel, south of Madagascar), though not the highest shipping density (0.029 vs 0.036–0.039 for the tighter boxes).
  - Rerun `01`–`04`: 5.95M hourly obs in R, **4.12M undrogued**, 1,503 drifters; oil matrix at 1° has 970 ocean cells, 4.06M transitions, 9% of cells flagged (< 10 drifters), 3.3% exit per step.
  - Durban release (oil, 1°): after 1 yr 18% still in R, 57% exited E, 23% W (previous box: 12% / 65% / 21%).
- [ ] Propagate release points and plot (1 week / 1 month / 1 year).
- [ ] Validation with held-out drifters.
- [ ] Download ERA5 winds and build oil version.
- [ ] Eigen / clustering analysis for whole-box structure.

Open questions:
- ~~Box size: does 10–40°E, 45–25°S include enough of the Return Current and leakage region?~~ Resolved: final box 10–55°E, 45–15°S (via 5–50°E, 50–20°S).
- The drogued (container) matrix now has 21% of cells flagged (< 10 drifters). Is that acceptable, or should thin cells be merged / smoothed?
- Is τ fixed across both versions?
- How much weight to put on seasonal matrices vs one all-year matrix?
