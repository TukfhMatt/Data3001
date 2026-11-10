# Pipeline map

Every step in the order `python -m agulhas all` runs it (`agulhas/pipeline.py`), with what it reads and writes. Run one step with `python -m agulhas <step>`; `python -m agulhas list` prints the same order. For the code layout see [architecture.md](architecture.md), and for the fields of each file see [data.md](data.md).

**Names.** `{tag}` is the setting a step ran on, from `config.setting_tag`, e.g. `1deg_3p5d_10E-55E_45S-15S` (or `0p5-1-2deg_3p5d_…` for a step over several grids); `{box}` is the box alone, e.g. `10E-55E_45S-15S`. `{region}` is the first word of the subset's file name (`agulhas`, `gulfstream`). **Matrices** are `data/P_{name}_{res}deg_{tau}d_{box}.npz`, always named through `config.operator_path`. **Release points** are `data/release_candidates_{tag}.csv` at the default grid and lag, read through `config.release_points_path`. **Subset** is `data/agulhas_gdp1h_subset.nc`, or the file given with `--data`.

**Agulhas only** steps need the Agulhas coasts, shipping routes, winds or features, and are skipped by `all` in another region.

## Sources

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `fetch_gdp_subset` | Streams the GDP hourly store chunk by chunk and keeps 5°W–65°E, 60–5°S at hourly resolution, dropping positions across gaps > 6 h; stops rather than drop a chunk that keeps failing. Agulhas only in `all`; `--fetch-box` and `--out` save another region. | GDP hourly zarr store on AWS S3 | subset |
| `fetch_era5` | ERA5 10 m wind at every drifter position. For the default subset, downloads the published file (`data-v1` release) after checking the subset's fingerprint, then checks its SHA-256; otherwise, or with `--cds`, requests it one year at a time from the CDS API. Agulhas only in `all`. | subset; the data release, or the CDS API (`~/.cdsapirc`) | `data/drifter_wind10m.nc` (another subset: `config.wind_paths`) |

## Exploration

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `gdp_exploration` | Observations and drifters in R by drogue type; coverage map; observations per year. | subset | `figures/{region}_gdp_coverage_{box}.png`, `figures/{region}_gdp_temporal_{box}.png` |
| `grid_coverage` | Distinct drifters per cell at each grid, to choose the grid. | subset | `data/grid_coverage_{tag}.npz`, `figures/{region}_grid_coverage_{tag}.png` |
| `box_selection` | Scores candidate boxes on data, shipping and oil retention; the Pareto set and named options. Agulhas only. | subset, shipping zip | `data/box_pareto.csv`, `data/box_options.csv`, `figures/agulhas_box_selection.png` |
| `wind_slip` | Undrogued − drogued mean velocity per cell and by latitude band. | subset | `data/wind_slip_cells_{tag}.csv`, `data/wind_slip_bands_{tag}.csv`, figure |
| `drogue_loss` | Time from drogue loss to the end of the record, grounded against other deaths; stranding moves under each drogue rule. | subset | `data/drogue_loss_{tag}.csv`, `…_summary_…`, `…_stranding_moves_…`, figure |
| `decorrelation_time` | Velocity autocorrelation and integral timescale per drogue type, with drifter-bootstrap intervals; sets τ. | subset | `data/decorrelation_{tag}.csv`, `data/decorrelation_curve_{tag}.csv`, figure |

## Operators

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `transition_matrix` | Drogued and undrogued matrices at 0.5°, 1° and 2°, τ = 3.5 d, with exit and stranded states; a test release off Durban. | subset | `data/P_{drogued,undrogued}_…npz` |
| `transition_matrix_lags` | The same at 1° for τ = 2, 5 and 7 days (sensitivity, Chapman–Kolmogorov). | subset | `data/P_{drogued,undrogued}_1deg_{2,5,7}d_…npz` |
| `oil_wind` | Stops if the winds were built for another subset (`subset_fingerprint`). Measures undrogued windage against ERA5; builds oil (undrogued + the missing windage) and its drogued check at 3%, 3.5% and 4%; compares them at the release points. Agulhas only. | subset, winds, undrogued matrix, release points, `bootstrap_release_{tag}.csv` (if present) | `data/P_oil*_…npz`, `data/P_oil_check*_…npz`, `data/oil_moves_{tag}.npz`, `data/oil_compare_{tag}.csv`, `figures/{region}_oil_durban_30d_{tag}.png` |
| `seasonal_matrices` | DJF, MAM, JJA and SON matrices; TVD against the annual matrix and between seasons; permutation test; 30-day release scenarios. | subset, annual matrices, release points | `data/P_{type}_{season}_…npz`, `data/seasonal_{support,tvd_cells,pairwise_tvd,permutation,release_30d}_{tag}.csv`, `figures/{region}_seasonal_summary_{tag}.png` |
| `seasonal_matrices_res2` | The same at 2°, where drogued seasonal support at 1° is thin. | as above | as above at 2° |
| `oil_windage` | Oil at 3%, 3.5% and 4% of the wind against undrogued, for traffic, routes and release points, with the 4% − 3% difference on the same drifters. Agulhas only. | coast operators, `coast_counts_{tag}.npz`, `traffic_releases_{tag}.npz`, oil matrices, release points | `data/oil_windage{,_difference,_method_flags}_{tag}.csv`, `figures/agulhas_oil_windage_{tag}.png` |

## Release points

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `shipping_overlay` | Shipping density over the mean current; picks release points (busiest lanes, known hotspots) in cells reliable in both matrices and ranks them by near-shore exposure. | subset, shipping zip, drogued and undrogued matrices | release points, `data/shipping_{tag}.npz`, `figures/{region}_shipping_currents_{drogued,undrogued}_{tag}.png` |
| `release_point_maps` | Where material from each release point is after 7, 30 and 365 days, with bootstrap intervals. | matrices, release points, `bootstrap_release_{tag}.csv` | `data/release_point_maps_{tag}.csv`, `figures/{region}_release_{nn}_{point}_{tag}.png` |

## Whole box

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `whole_box_analysis` | Row support for every grid, lag and drogue type. | matrices | `data/whole_box_support_{tag}.csv`, figure |
| `coastal_coverage` | The same for coastal cells against offshore cells. | matrices | `data/coastal_support_{tag}.csv`, figure |
| `trap_stability` | Retention hotspots (share still in R after a year), agreement across grids, drifter bootstrap at 1°. | subset, matrices | `data/retention_{hotspots,hotspot_agreement,cells,bootstrap}_{tag}.csv`, figure |
| `residence_time` | Expected time afloat in R from each cell, from the absorbing chain. | matrices | `data/residence_time{,_cells}_{tag}.csv`, figure |
| `exit_zones` | Eventual fate from each cell (each edge or stranded) and the fate after a year. | matrices | `data/exit_zones{,_cells}_{tag}.csv`, figure |
| `almost_invariant_regions` | Four regions per drogue type that keep material, from spectral clustering, against longitude bands. | matrices | `data/almost_invariant_{regions,cells}_{tag}.csv`, figure |
| `source_sink` | Cells that gather or shed material over one step, with a drifter bootstrap. | subset, matrices | `data/source_sink_{tag}.csv`, figure |

## Coast

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `coastal_pooling` | Pools thin neighbouring coastal cells (filtered by Natural Earth land) into short segments. Does not change P. | subset, matrices, Natural Earth 50m land | `data/coastal_pooling_{cells,segments,summary}_{tag}.csv`, `data/coastal_segments_{tag}.npz`, figure |
| `coastal_stranding` | Splits stranding into six named coasts and checks the split against the saved matrix; stranding by coast for every cell and the release points. Agulhas only. | subset, drogued, undrogued and oil matrices, `oil_moves_{tag}.npz`, release points | `data/P_{name}_coast_…npz`, `data/coast_counts_{tag}.npz`, `data/coastal_stranding_{points,cells,support}_{tag}.csv`, `figures/agulhas_coastal_stranding_{tag}.png` |
| `traffic_exposure` | Releases spread by shipping density, for all traffic and six routes: stranding by coast, days to strand, near-shore days, oil mass; weight sensitivity. Agulhas only. | shipping zip, coast operators, `coast_counts_{tag}.npz`, `coastal_segments_{tag}.npz` | `data/traffic_{exposure,nearshore_segments,weights}_{tag}.csv`, `data/traffic_releases_{tag}.npz`, `figures/agulhas_traffic_exposure_{tag}.png` |
| `risk_atlas` | Every release cell × horizon × outcome, with intervals and reliability flags. Agulhas only. | coast operators, `coast_counts_{tag}.npz` | `data/risk_atlas_{tag}.csv`, `figures/agulhas_risk_atlas_{tag}.png` |

## Validation

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `bootstrap` | Resamples whole drifters: intervals on each release point's fate, drogued against undrogued, row stability. | subset, matrices, release points | `data/bootstrap_{release,difference}_{tag}.csv`, `data/bootstrap_rows_{tag}.npz`, `figures/{region}_bootstrap_{fates,rows}_{tag}.png` |
| `held_out` | 5-fold validation by drifter against persistence, advection and climatology. | subset | `data/validation_{folds,summary}_{tag}.csv`, `figures/{region}_validation_{tag}.png` |
| `held_out_tau2` | The same at 1°, τ = 2 d. | subset | as above |
| `held_out_tau5` | The same at 1°, τ = 5 d. | subset | as above |
| `held_out_res05` | The same at 0.5°, τ = 3.5 d. | subset | as above |
| `held_out_res2` | The same at 2°, τ = 3.5 d. | subset | as above |
| `held_out_year` | Fates of held-out drifters out to 364 days, in `held_out`'s folds: Aalen–Johansen estimates that allow for records ending early, with drifter-bootstrap intervals, Brier skill, calibration, and results by start longitude and release cell. | subset, release points | `data/validation_year_{summary,support,calibration,bands,points}_{tag}.csv`, `figures/{region}_validation_year_{tag}.png` |
| `sensitivity` | The reference setting against τ = 2, 5 d and the 0.5°, 2° grids: held-out skill and one-year fates against the reference's bootstrap interval. | matrices at each setting, release points, `bootstrap_release_…`, `validation_summary_…` at each setting | `data/sensitivity_{skill,fates}_{box}.csv`, `figures/{region}_sensitivity_{box}.png` |
| `chapman_kolmogorov` | P(τ)² against the direct 2τ matrix, row by row, against each row's sampling noise. | subset, matrices at τ and 2τ | `data/chapman_kolmogorov_{cells,summary}_{tag}.csv`, two figures |
| `hypothesis_tests` | The proposal's decision rules H1 (east against west), H2 (windage and stranding), H3 (skill). | subset, matrices, release points, `validation_folds_{tag}.csv` | `data/hypothesis_h{1,2,3}_{tag}.csv`, figure |
| `spill_hindcasts` | The oil matrix against two recorded Cape spills under a fixed rule. Agulhas only. | coast operators, `coast_counts_{tag}.npz`, `P_undrogued_JJA_…npz` | `data/spill_hindcast{s,_verdicts}_{tag}.csv`, `figures/agulhas_spill_hindcasts_{tag}.png` |
| `known_features` | Mean current and matrices against the known circulation under fixed rules. Agulhas only. | subset, matrices, release points | `data/known_features_{tag}.csv`, figure |

## Tools

These only read pipeline outputs, so `all` skips them.

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `worked_example` | Where material released at one point goes after a week, a month and a year, with intervals, coast and method check. | the chosen matrix, its coast operator and `risk_atlas_{tag}.csv` if present | `figures/{region}_worked_example_{matrix}_{point}_{tag}.png` |
| `oil_explorer` | A single web page that steps the oil, undrogued or drogued matrix from any clicked point. | `P_{oil,undrogued,drogued}_coast_…npz`, oil and check matrices, `risk_atlas_{tag}.csv`, release points, `agulhas/tools/oil_explorer.html` | `figures/agulhas_oil_explorer_{tag}.html` |

## OpenDrift comparison

The oil matrix against OpenDrift's `OceanDrift`, run on each date's GLORYS12 currents and ERA5 wind. OpenDrift needs Python 3.10 or newer, so these steps run in their own environment (`requirements-opendrift.txt`, see the README) and `all` skips them. Their forcing and results stay in `data/opendrift/`. Agulhas only.

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `opendrift_forcing` | Downloads GLORYS12 daily surface currents (Copernicus Marine) and ERA5 10 m wind for 2000, 2005, 2010, 2015 and 2020; finished files are skipped. | Copernicus Marine login, CDS API (`~/.cdsapirc`) | `data/opendrift/forcing/glorys_{year}.nc`, `…/era5_{year}.nc`, `…/era5_{year+1}01.nc` |
| `opendrift_points` | One run per start date (the 1st of each month), 500 particles over the ocean part of each release point's cell, sorted into the matrix's states at day 7 and 31.5; finished starts are skipped. | forcing, `P_oil_coast_…npz`, release points | `data/opendrift/points/{yyyymm}.npz` |
| `opendrift_treasure` | The Treasure spill of June 2000 from the wreck, judged by `spill_hindcasts`' three rules, and the islands reached; `--leak-days 5` is a sensitivity. | forcing | `data/opendrift/treasure{,_leak5d}.json` |
| `opendrift_compare` | The matrix against the OpenDrift runs at each release point, under a rule fixed before the results, with OpenDrift's own year-to-year distance as a reference. | `data/opendrift/points/*.npz`, `P_oil_coast_…npz`, `risk_atlas_{tag}.csv`, release points | `data/opendrift/compare_{tag}.csv`, `figures/agulhas_opendrift_compare_{tag}.png` |
| `opendrift_moves` | Repeats `opendrift_points`' runs with the same particles and seeds and counts every particle's moves between the matrix's states over τ, from every hour; stops if its day-7 and day-31.5 counts differ from `opendrift_points`'. | forcing, `P_oil_coast_…npz`, release points, `data/opendrift/points/*.npz` | `data/opendrift/moves/{yyyymm}.npz` |
| `opendrift_checks` | Two checks added after `opendrift_compare`'s result: the distance over what is still afloat, and a matrix built from OpenDrift's own moves, which separates the matrix method's error from the difference in data. | `data/opendrift/points/*.npz`, `…/moves/*.npz`, `P_oil_coast_…npz`, release points | `data/opendrift/checks_{tag}.csv`, `figures/agulhas_opendrift_checks_{tag}.png` |
| `opendrift_drifters` | `held_out`'s undrogued test starts in the forcing years, each with its fold matrix's forecast, and OpenDrift at 2% of the wind from the drifter's position and from over its cell; one run per half month, finished runs skipped. | the subset, forcing, `P_oil_coast_…npz` | `data/opendrift/drifters/starts_{tag}.npz`, `…/drifters/{yyyymm}_{half}.npz` |
| `opendrift_score` | Distance from each method's forecast centre (matrix, OpenDrift from the exact position and from over the cell, persistence) to the held-out drifter, at every step to 28 days; difference in median error with 95% intervals from resampling whole drifters. The rule was fixed before scoring. | `data/opendrift/drifters/*.npz` | `data/opendrift/drifters_score_{tag}.csv`, `figures/agulhas_opendrift_drifters_{tag}.png` |
| `opendrift_coverage` | How often OpenDrift's mean share at each release point lies inside the oil matrix's 95% interval from `risk_atlas`, by outcome and horizon; `--matrix oil_3pct` or `oil_4pct` uses the runs at that windage. The rule is in `PLAN.md`, fixed before it ran. | `data/opendrift/points*/`, `P_{matrix}_coast_…npz`, `risk_atlas_{tag}.csv`, release points | `data/opendrift/coverage_{tag}.csv` |
