Something goes into the water at a chosen point in R: a container off a ship, a life raft, a slick of oil. Where has the surface flow taken it a week later? A month later? A year? At the scale of the whole box, which parts of R feed which others, and which are cut off from the rest?

Product. A surface-transport operator for R. E.g. a transition matrix (or similar) someone else could pick up and iterate forward, together with what it says about where a few chosen release points end up, and about where R as a whole gathers material and where it loses it.


Regions:

---
Agulhas region:
Box R: 10°E–40°E, 45°S–25°S (as used in `scripts/01_agulhas_gdp_exploration.py`).

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
  - In R: 1,048 drifters, 503k obs (179k drogued, 325k undrogued).
  - Grid coverage (`scripts/02_grid_coverage.py`), % of ocean cells with ≥10 distinct drifters:
    drogued 88% at 1°, 68% at 0.5°; undrogued 95% at 1°, 93% at 0.5°.
  - Drogued data is thin in the north-east (Mozambique side / current source region) and on the Agulhas Bank shelf.
- [ ] Confirm grid size. Suggested: 1° for both versions (0.5° possible for undrogued only).
- [x] Choose τ: **3.5 days** (1 week = 2 steps). Velocity decorrelation time in R ≈ 1.4 d (drogued), 1.2 d (undrogued); at τ = 3.5 d drifters move ~1 cell per step. Sensitivity check later with τ = 2 and 5 d.
- [x] Build first transition matrices (1°, τ = 3.5 d, all years) — `scripts/03_transition_matrix.py`, helpers in `scripts/transport.py`.
  - Saved: `data/P_drogued_1deg_3p5d.npz`, `data/P_undrogued_1deg_3p5d.npz` (483 ocean cells + 4 exit states W/E/S/N).
  - 173k drogued / 321k undrogued transitions; ~3.5–4% of mass leaves R per step.
  - Cells with < 10 drifters flagged: 13% drogued, 5% undrogued. 3–4 cells with no outgoing data hold mass in place.
  - Sanity check (release off Durban): moves SW along the coast; after 1 year ~2/3 exits east (Return Current), ~1/5 west (leakage into the Atlantic).
- [ ] Propagate release points and plot.
- [ ] Validation with held-out drifters.
- [ ] Download ERA5 winds and build oil version.
- [ ] Eigen / clustering analysis for whole-box structure.

Open questions:
- Box size: does 10–40°E, 45–25°S include enough of the Return Current and leakage region?
- Is τ fixed across both versions?
- How much weight to put on seasonal matrices vs one all-year matrix?
