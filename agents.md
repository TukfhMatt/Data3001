# Agent guide

Instructions for AI coding agents working in this repository. Read [README.md](README.md) for the project overview and [Research.md](Research.md) for the research plan and current progress.

## Project in one paragraph

A university data science project (Data3001) that builds a surface-transport operator for the Agulhas region, box R = 10–55°E, 45–15°S (scored on data, shipping risk and retention in `05_box_selection.py`). NOAA Global Drifter Program tracks are binned on a grid (0.5°, 1° and 2°), and cell-to-cell moves over a lag τ (default 3.5 days) are counted into a row-stochastic transition matrix (Ulam's method). Two versions are tracked side by side: drogued drifters (15 m current, passive transport) and undrogued drifters (surface current plus about 1% wind slip, the closer match to oil). Never pool them into an "all drifters" matrix. The matrices are iterated to answer where material released at a point is after 1 week, 1 month and 1 year, and to analyse where R gathers and loses material.

## Environment

- Use the project virtual environment: `.venv/bin/python` (Python 3.9).
- Run scripts **from the repository root**. They use relative paths (`data/...`, `figures/...`), and they import `config` and `transport` from `scripts/`.
- The Python is built against LibreSSL, so urllib3 prints a `NotOpenSSLWarning`; it is harmless. Pass `-W ignore` for cleaner output.
- `zarr` is pinned to 2.x (`zarr.open_group` on an HTTPS URL). Do not upgrade to zarr 3 without checking `00_fetch_agulhas_subset.py`.
- No dask is installed and the machine has 8 GB RAM. Never load the full global GDP arrays (197M observations) into memory; stream them chunk by chunk as `00` does.

## Data rules

- The data source is the GDP hourly zarr store on AWS S3 (`gdp1h`), subset by `00` into `data/agulhas_gdp1h_subset.nc` at the native **hourly** resolution (positions interpolated across > 6 h between satellite fixes are dropped; `gap` is kept). All later scripts read that file. Rerun `00` only if the region or margin changes.
- The subset covers 5°W–65°E, 60–5°S: every candidate box from `05` (search area 0–60°E, 55–10°S) plus a margin, so positions just after a drifter leaves a box are available. Changing the box within that area does not need a new fetch. Always re-apply the R mask in analysis.
- The subset is sorted by (trajectory, time), and every time is on the hour. `03` relies on this for pairing.
- `drogue_status` is boolean (True = drogued). A transition counts for a drogue type only if both ends have that status.
- Stranding comes from GDP's per-drifter `typedeath` (1 = ran aground) and `end_date`, kept in the subset by `00`. `transport.strand_times` marks a drifter as stranded when its last observation is in R and within a day of `end_date`; `move_indices` then sends its final τ of positions to the stranded state. Other deaths (stopped transmitting, picked up) are missing data, not stranding. Almost every grounding is undrogued (drogues are lost before or at grounding), so drogued stranding is a lower bound.
- Shipping traffic: World Bank / IMF "Global Shipping Traffic Density", Commercial layer (CC-BY 4.0), zip at `data/shipping/shipdensity_commercial.zip` (458 MB, 9.8 GB GeoTIFF inside). `04` reads the box straight from the zip with `rasterio` (`zip://…!/ShipDensity_Commercial1.tif`); never extract it. Values are undocumented in scale (bimodal: ~1–5 and ~1e5–1e6 per pixel, far above plausible AIS counts) and there is a source artifact at ~40°S. Use them only as a relative density or ranking, never as counts.
- `data/` and `figures/` are gitignored, and so are `*.nc`, `*.zarr` and `*.csv`. Never commit data or figures.

## Code conventions

- Scripts are numbered in run order (`00_`, `01_`, …). New pipeline steps get the next number, e.g. `05_release_points.py`.
- `04_shipping_overlay.py` picks candidate release points only in cells that are reliable in both the drogued and undrogued matrices (not `flagged`, not `empty`), so both versions are compared at the same points; hotspots in unreliable cells are moved to the nearest such cell and the shift is printed. Keep that rule for any new release points.
- Each script starts with a docstring (title line "Agulhas Region — …", then what it does) and is split into numbered sections with headers such as `# ── 2. Pair each obs … ─────`.
- Plots use `matplotlib.use("Agg")`, cartopy with a plain-matplotlib fallback, and are saved to `figures/` at 150 dpi. Scripts print a short summary and the saved paths.
- Region, grid, lag and data path live in `scripts/config.py` (defaults: box 10–55°E, 45–15°S; grids 0.5°, 1°, 2°; τ = 3.5 d) and are overridden with `--box`, `--res`, `--tau` and `--data` (`parse_args`). Scripts `01`–`04` unpack them into `LON_MIN, LON_MAX, LAT_MIN, LAT_MAX`, `RES`, `TAU_DAYS`; do not hard-code them again. `00` keeps its own padded fetch box and `05` its own search settings.
- Use `grid_shape(box, res)` for the number of cells. If a box side is not a multiple of `res`, the last row/column is a partial cell clipped at the box edge (edges are clipped to the box).
- Grid cells use half-open intervals `[min, max)`, with index `j * n_lon + i`, where `j` is the latitude row from the box's southern edge and `i` the longitude column from its western edge.
- Use `scripts/transport.py` (`load_operator`, `release`, `propagate`, `to_grid`, `coastal_states`) to work with saved matrices rather than re-implementing the logic. Its building helpers (`strand_times`, `move_indices`, `transition_states`, `row_normalise`) are shared by `03`, `06`, `09` and `10`; anything that builds a matrix uses them, so every build matches `03` exactly.
- Uncertainty comes from `06_bootstrap.py`: resample whole drifters (never observations), with drogued and undrogued resampled independently. Report the 95% interval with any release-point number, and treat a single "differ" flag with care, since the intervals are not corrected for the number of points and metrics compared.

## Transition matrix format

Files are `data/P_{drogued|undrogued}_{res}deg_{tau}d_{box}.npz`, built by `config.operator_path` (e.g. `P_drogued_1deg_3p5d_10E-55E_45S-15S.npz`). Only read matrices through `operator_path`; any other `P_*.npz` in `data/` is not a pipeline output.

- `P` is (n_cells + 5) square: 975 × 975 at 1°, 3785 at 0.5°, 268 at 2° for the default box. The first `n_cells` states are ocean cells (`cell_flat` maps a state to its grid index); the last five are absorbing, in the order of `exit_labels`: exits `W, E, S, N` and `stranded` (`transport.STRANDED` is its offset after the ocean cells). Both drogue types share the same ocean cells. Read exits and stranding through `exit_labels`, never by assuming four exit columns.
- A row holds the probabilities of moving from that state to each state in one τ step. Distributions are row vectors, advanced with `p @ P`.
- `flagged` marks rows built from fewer than 10 distinct drifters, and `empty` marks cells with no outgoing moves of their own, whose rows pool the moves of the nearest ring of ocean cells (`row_normalise`). Mention both when reporting results for those cells. `06` adds a second check, `row_tvd_{drogued,undrogued}` in `data/bootstrap_rows_*.npz` (unstable if > 0.2). It understates the uncertainty of rows with only a few drifters, so the 10-drifter flag stays as the floor.
- If τ, the grid or the region changes, the file name changes with it; always get paths from `operator_path`.

## Domain notes

- Drogued = passive water transport (little windage, e.g. a mostly submerged container). Undrogued drifters slip downwind by ~10 cm/s (~1% of wind speed), measured as undrogued − drogued mean velocity per cell. Oil moves at current + about 3–3.5% of wind speed, so undrogued captures only about a third of oil's windage. Do not add the full 3.5% wind on top of undrogued motion; that counts the wind twice.
- Pooling drogued and undrogued ("all drifters") gives a matrix whose wind effect varies with the local undrogued share (46–84% per cell), so it represents neither; do not use it.
- Oil mass loss is modelled as `m(t) = m0 · exp(−λt)`, with λ ≈ 1/14 per day as the working value.
- Velocity decorrelation time over the core region (10–40°E, 45–25°S) is about 1.4 days (drogued) and 1.2 days (undrogued), which is why τ = 3.5 days. Sensitivity checks use τ = 2 and 5 days.
- Expected behaviour for sanity checks: material released off Durban moves south-west along the coast. After a year (default box, 1°) about 50–57% has left R to the east (Return Current), about a fifth to the west (leakage into the Atlantic), and 18–22% is still in R.

## Workflow

- After finishing a pipeline step, update the checklist in `Research.md` with the key numbers, and the Status section of `README.md`.
- Notes, docs, docstrings and code comments describe only the current approach and current results. No change history ("previously…", "instead of…", "was X, now Y", old numbers kept for comparison): when something changes, rewrite the text as if the current approach had always been the plan. Git history holds the evolution.
- Work on the `alex` branch (tracks `origin/alex`). Commit or push only when asked.
- The notebook `notebooks/01_agulhas_gdp_exploration.ipynb` is not part of the pipeline (it imports `gdp_6h`, loads the full dataset and assumes a 0/1 drogue flag). Do not use it as a reference.
