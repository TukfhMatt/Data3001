# Data dictionary

The inputs, the matrix files every analysis reads, and the intermediate files steps pass to each other. Which step writes and reads each file is in [pipeline.md](pipeline.md). `data/` and `figures/` are not in git; `python -m agulhas all` rebuilds them.

## Inputs

### Drifter subset — `data/agulhas_gdp1h_subset.nc`

NOAA Global Drifter Program hourly product v2.01, from the public AWS S3 zarr store, cut by `fetch_gdp_subset` to 5°W–65°E, 60–5°S. Ragged arrays: observations of one drifter are contiguous, sorted by (drifter, time), every time on the hour; positions interpolated across more than 6 h between satellite fixes are dropped.

| Variable | Per | Meaning |
|---|---|---|
| `lon`, `lat` | observation | Position (degrees) |
| `ve`, `vn` | observation | East and north velocity (m/s) |
| `drogue_status` | observation | True while the 15 m drogue is attached |
| `time` | observation | UTC, on the hour |
| `gap` | observation | Time between the previous and next satellite fix |
| `rowsize` | drifter | Observations of each drifter in the subset |
| `id` | drifter | GDP drifter number |
| `typedeath` | drifter | How the full record ended (1 = ran aground) |
| `end_date` | drifter | Last good date of the full record; with `typedeath`, it marks a grounding inside the box (`transport.strand_times`) |

The SHA-256 of the subsets behind the reported results is in the README. Another region has its own subset, passed with `--data`.

### ERA5 winds — `data/drifter_wind10m.nc`

`u10`, `v10`: ERA5 10 m wind (m/s) interpolated to every observation of the subset, in the same order (`fetch_era5`). Another subset has its own file (`config.wind_paths`).

### Shipping density — `data/shipping/shipdensity_commercial.zip`

World Bank / IMF "Global Shipping Traffic Density", Commercial layer, CC-BY 4.0: one GeoTIFF of about 0.005° pixels, read for one box straight from the zip (`agulhas/shipping.py`). The values are undocumented in scale and bimodal (about 1–30 and 1e4–1e7 per pixel), and pixel coverage drops about 8-fold at exactly 40°S, so they are used only as a relative density or ranking.

## Matrices — `data/P_{name}_{res}deg_{tau}d_{box}.npz`

Always named through `config.operator_path(name, res, tau, box)`. `P` is square, `n_cells` ocean states then the absorbing states of `exit_labels`. Load with `transport.load_operator`.

| Field | Meaning |
|---|---|
| `P` | `P[i, j]`: probability that material in state `i` is in state `j` one lag later; rows sum to 1 |
| `C` | Moves behind `P` |
| `row_drifters`, `row_obs` | Distinct drifters and moves behind each ocean row |
| `flagged` | Rows from fewer than `MIN_DRIFTERS` (10) drifters |
| `empty` | Cells with no moves of their own, whose rows pool the moves of the nearest ring of ocean cells |
| `cell_flat` | Grid index (`j * n_lon + i`) of each ocean state |
| `lon_edges`, `lat_edges`, `res` | The grid; a partial last cell is clipped to the box |
| `exit_labels` | Names of the absorbing states, in order: `W, E, S, N, stranded` |
| `box`, `tau_days`, `drogued`, `years`, `source` | Region, lag (days), drogue type, years of the moves, data source |

Names and their extra fields:

| `name` | What it is | Extra fields |
|---|---|---|
| `drogued`, `undrogued` | From `transition_matrix` | — |
| `oil`, `oil_3pct`, `oil_4pct` | Undrogued moves shifted by the windage they lack, for oil at 3.5%, 3% and 4% of the wind (`oil_wind`) | `alpha_added`, `alpha_undrogued`, `alpha_oil`, `decay_days` |
| `oil_check`, `oil_check_3pct`, `oil_check_4pct` | Drogued moves shifted by the full oil windage, a check of the method | as above |
| `{type}_{DJF,MAM,JJA,SON}` | Moves starting in one season (`seasonal_matrices`) | `season`, `season_name`, `season_months`, `season_rule` |
| `{name}_coast` | The same matrix with `stranded` split into six named coasts (`coastal_stranding`, `transport.COAST_LABELS`), so `n_cells + 10` states | `coast_names` |

```python
from agulhas.transport import load_operator, release, propagate
op = load_operator("data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz")
p = propagate(op, release(op, lon=31.5, lat=-30.5), days=30)
n = len(op["cell_flat"])
print("in R:", p[:n].sum(), dict(zip(op["exit_labels"], p[n:])))
```

## Files steps pass to each other

| File | Written by | Fields |
|---|---|---|
| `release_candidates_{tag}.csv` | `shipping_overlay` | `kind` (`hotspot` or `traffic`), `name`, `lon`, `lat` (centre of a cell reliable in both matrices), shipping density, and each matrix's coastal exposure, one-year fate and row drifters |
| `shipping_{tag}.npz` | `shipping_overlay` | `ship` (per grid cell), `ship_fine` (0.1°), `u_{type}`, `v_{type}`, `current_obs_{type}` (mean current per cell), `coastal`, `lon_min`, `lat_min`, `res`, `fine` |
| `grid_coverage_{tag}.npz` | `grid_coverage` | `{type}_{res}_obs`, `_drifters`, `_ocean` per grid; the box edges |
| `bootstrap_release_{tag}.csv` | `bootstrap` | Per release point, drogue type, metric and horizon: `estimate`, `ci_low`, `ci_high` |
| `bootstrap_rows_{tag}.npz` | `bootstrap` | `row_tvd_{type}`: median bootstrap TVD per row (unstable above `unstable_tvd`); `cell_flat`, `n_boot` |
| `validation_folds_{tag}.csv`, `validation_summary_{tag}.csv` | `held_out` | Per drogue type, fold and horizon: position errors of the matrix and baselines, log scores, exit and stranding calibration; the summary averages over folds |
| `oil_moves_{tag}.npz` | `oil_wind` | Per oil matrix `{name}_traj`, `_ft`, `_n`: moves per (drifter, from × to state); `n_states` |
| `coast_counts_{tag}.npz` | `coastal_stranding` | Per coast operator `{name}_d_idx`, `_ft`, `_n`, `_n_drifters`: per-drifter move counts for `transport.resampled_counts`; `n_states`. Read with `transport.load_coast_counts` |
| `coastal_segments_{tag}.npz` | `coastal_pooling` | `segment_of_state` (−1 off the coast), `coastal`, support per segment and drogue type, `segment_size`, `reliable_both` |
| `traffic_releases_{tag}.npz` | `traffic_exposure` | `W`: release weights over ocean states, all traffic then each route (rows sum to 1); `route_share`; `releases` |
| `risk_atlas_{tag}.csv` | `risk_atlas` | Per matrix, release state, horizon and outcome: `estimate`, `lo`, `hi`, with the cell's `reliable`, `flagged`, `empty` and `row_drifters` |
| `opendrift/points/{yyyymm}.npz` | `opendrift_points` | `counts`: particles per (release point, horizon, state), with horizons day 7 and 31.5 and states as in the oil coast matrix, then one for a position it has no state for (`opendrift_common.classify`) |

The other CSVs are result tables; their columns are named in full (`median_tvd`, `ci_low`, `reliable_both`, …) and each step's docstring says what it reports.
