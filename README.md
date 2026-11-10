# Data3001 — Agulhas Surface Transport

Something goes into the water in the Agulhas region off South Africa: a container lost from a ship or an oil slick. Where has the surface flow taken it after a week, a month, a year? Which parts of the region feed which others, and where does material collect or leave?

This project builds a **surface-transport operator** for that region: a transition matrix, estimated from real ocean drifter tracks, that anyone can load and step forward in time.


---

## Region

Box **R**: 10°E–55°E, 45°S–15°S, gridded at 0.5°, 1° and 2° (3,780 / 970 / 263 ocean cells). The box, grid and lag are command-line options (see below), so the same code runs on any box.

It covers the inflow to the Agulhas Current from the southern Mozambique Channel and south of Madagascar, the current along the South African east coast, the retroflection south of Africa where most of the flow turns back east, leakage into the South Atlantic, and the Agulhas Return Current.

The box balances shipping risk against oil retention (scored in `box_selection`): oil stays in it 136 days on average and 6.5% is still inside after a year, it contains 97% of the Cape Basin eddy activity (Agulhas rings), and it covers the coastal, Mozambique Channel and south-of-Madagascar shipping routes.

## Data

- **NOAA Global Drifter Program (GDP)**, hourly product v2.01, read from the public AWS S3 zarr store (the same source as `clouddrift.datasets.gdp1h()`), at the native hourly resolution (positions interpolated across > 6 h between satellite fixes are dropped, 0.3% of the data). Coverage in R, 1995–2022:

  | | Hourly observations |
  |---|---|
  | All | 5,952,447 |
  | … drogued | 1,834,513 |
  | … undrogued | 4,117,934 |
  | Distinct drifters | 1,503 |

  Consecutive hourly positions are strongly correlated; the independent sample size is the number of drifters (~1,170 undrogued), not the observation count.
- GDP's `latest` store can change, so results depend on the fetch. The subsets behind the results in this repository were fetched in November 2026:

  | File | Drifters | Hourly positions | SHA-256 |
  |---|---|---|---|
  | `data/agulhas_gdp1h_subset.nc` | 2,942 | 16,951,519 | `d11e21bb359e0575bc381554308bcfa4dedc8e7ebfc9808f000302f84c2528af` |
  | `data/gulfstream_gdp1h_subset.nc` | 3,022 | 23,595,748 | `4ae3273624c7f162cf01f5e17b83056a2d743b879fd207d9aa751c2e11408d22` |

  Check a copy with `shasum -a 256 data/*_gdp1h_subset.nc`. If a new fetch gives another hash, compare the counts first, since the file's bytes can differ with the same data.

- Drifters are split by **drogue status**, and a matrix is built for each:
  - **Drogued**: a sea anchor at 15 m is attached, so the drifter follows the near-surface current with little wind effect. This is passive water transport (e.g. a submerged container).
  - **Undrogued**: the drogue has been lost, so the drifter rides at the surface and slips downwind by about 10 cm/s (2.0% of the ERA5 10 m wind: eastward under the westerlies, north-westward under the trades). Closest match to oil, but oil drifts at about 3.5% of wind speed, so the oil matrix adds the missing 1.5%.
  - The two are not pooled: an "all drifters" matrix mixes 46–84% undrogued data depending on the cell, so its wind effect would follow sampling history rather than physics.
- ERA5 10 m winds (Copernicus Climate Data Store) at every drifter position supply the oil matrix's wind term (`fetch_era5`, `oil_wind`). The winds at the Agulhas subset's positions are published as a [data release](https://github.com/TukfhMatt/Data3001/releases/tag/data-v1) of this repository (`drifter_wind10m.nc`, 136 MB; generated using Copernicus Climate Change Service information, 2026), so a run needs no CDS account. `fetch_era5` downloads them and checks their SHA-256, but only if the local subset holds the same positions and times as the one they were built for (`Drifters.fingerprint`, which compares content rather than file bytes); otherwise, and for any other subset, it requests them from the CDS.

---

## Setup

Python 3.9, with a virtual environment in `.venv/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Or with Docker

Docker gives the same environment without installing Python or any libraries. The image holds the code and its pinned dependencies only: `data/` and `figures/` are mounted from the repository, so downloads and outputs stay on disk between runs. Build and run it for x86-64 (`linux/amd64`), where every pinned package has a prebuilt wheel; on Apple silicon Docker emulates x86-64, which is slower but gives the same matrices.

```bash
docker build --platform linux/amd64 -t agulhas .
docker run --rm --platform linux/amd64 -v "$PWD/data:/app/data" -v "$PWD/figures:/app/figures" agulhas                                  # python -m agulhas all
docker run --rm --platform linux/amd64 -v "$PWD/data:/app/data" -v "$PWD/figures:/app/figures" agulhas python -m agulhas exit_zones  # one step
docker run --rm --platform linux/amd64 agulhas python -m pytest
```

The ERA5 winds download from the data release, so the container needs no CDS credentials; only another subset, or `fetch_era5 --cds`, needs them: add `-v "$HOME/.cdsapirc:/root/.cdsapirc:ro"` to the run command once `~/.cdsapirc` exists (if it does not, Docker mounts an empty folder in its place). Each new container downloads Cartopy's Natural Earth coastlines again, so runs need an internet connection. A full run (`python -m agulhas all`, starting from an empty `data/`) takes about 3 hours under emulation on an Apple-silicon laptop, within Docker Desktop's 8 GB memory limit; the drifter and shipping downloads take about 10 minutes of that. Its tables match a native run to floating-point rounding, which shows only in near-zero bootstrap bounds.

`cartopy` is optional for most plots, which can fall back to plain matplotlib without coastlines. However, `coastal_pooling` requires both `cartopy` and `shapely` to apply its Natural Earth 50m geographic filter.

`coastal_pooling` uses Natural Earth 50m land polygons to check whether an existing coastal-cell candidate intersects mapped land within its own grid cell or one of its eight neighbouring cells. The geographic filter excludes unsupported offshore candidates but does not add potentially missing coastal cells.

On the first run, Cartopy may need an internet connection to download the Natural Earth 50m land dataset. Subsequent runs can use the locally cached data.
  

## Project layout

```text
agulhas/
  pipeline.py      the steps in run order, and the `python -m agulhas` command
  config.py        default region, grids and lag, command-line options, file names
  drifters.py      the drifter subset, loaded array by array
  grid.py          grid cells of a box and the cell of any position
  moves.py         one-step moves on a grid and lag: what every matrix build starts from
  transport.py     load, step and analyse saved matrices; the helpers every matrix build shares
  plotting.py      headless figures, maps with or without cartopy, saving at 150 dpi
  shipping.py      the shipping density raster, read for one box from the zip
  stats.py         distances and bootstrap intervals
  opendrift_common.py  what the OpenDrift comparison steps share
  sources/         download the drifter subset and the ERA5 winds
  exploration/     look at the data and set the choices the matrices rest on
  operators/       build the transition matrices: drogued, undrogued, oil, seasonal
  release_points/  pick release points and map where material goes from them
  whole_box/       where R gathers, keeps and loses material
  coast/           stranding and exposure by coast, and the release-risk atlas
  validation/      how far the matrices can be trusted
  tools/           the worked example and the oil explorer; these only read the outputs
  comparison/      the oil matrix against OpenDrift, in its own environment
tests/             tests of the shared modules, the command and the layout rules; they need no data
docs/              architecture, pipeline map, data dictionary
```

Each step is one module with a `main()`, named as its step: `python -m agulhas bootstrap` runs `agulhas/validation/bootstrap.py`.

| Document | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How the package is organised, the shared modules, and the shape every step follows |
| [docs/pipeline.md](docs/pipeline.md) | Every step in run order, with what it reads and writes |
| [docs/data.md](docs/data.md) | The inputs, the matrix files and the files steps pass to each other |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, checks before committing, adding a step, changing code without changing results |
| [Research.md](Research.md) | The research plan and the results step by step |

## Testing

```bash
pip install -r requirements-dev.txt
python -m pyflakes agulhas tests
python -m pytest
```

## Running the pipeline

One command runs every step in order, with one log per step in `data/logs/`. It stops at the first failure and tells you how to resume:

```bash
python -m agulhas all                          # every step: roughly 30 minutes on an idle 8 GB laptop once the downloads are in place
python -m agulhas list                         # print the steps
python -m agulhas all --from coastal_stranding  # resume at a step
```

Downloads run only when their file is missing: `fetch_gdp_subset` (~1.2 GB streamed, 492 MB saved), the shipping data (458 MB, read directly from the zip) and `fetch_era5` (the ERA5 winds, 136 MB from the data release; only another subset or `--cds` needs `~/.cdsapirc`, a CDS account and the ERA5 licence, and many hours). No account or login is needed for a full run. Run these commands from the repository root; outputs go to `data/` and `figures/` there. Any single step can also be run by name, e.g. `python -m agulhas exit_zones`. [docs/pipeline.md](docs/pipeline.md) says what each one does, reads and writes.

Most steps take the region, grid and lag as options; add `--help` after a step's name to see which (defaults in `agulhas/config.py`: R = 10–55°E, 45–15°S; grids 0.5°, 1° and 2°; τ = 3.5 days):

```bash
python -m agulhas grid_coverage --res 0.5 1 2
python -m agulhas transition_matrix --box 10 55 -41 -15 --res 1 --tau 2 3.5 5   # example: custom box and several lags
python -m agulhas shipping_overlay --res 1 --tau 3.5                           # needs that matrix from transition_matrix
python -m agulhas seasonal_matrices --res 1 --tau 3.5
python -m agulhas seasonal_matrices --res 2 --tau 3.5                         # coarser seasonal robustness check
python -m agulhas calendar_forecast --res 1 --tau 3.5                         # one-year fates by release month; needs seasonal_matrices and bootstrap at the same setting
python -m agulhas calendar_forecast --res 2 --tau 3.5                         # the same at 2° (also needs bootstrap --res 2)
python -m agulhas transition_matrix --res 1 --tau 7                           # direct 2τ matrix needed by chapman_kolmogorov
python -m agulhas chapman_kolmogorov --res 1 --tau 3.5                        # P(τ)^2 vs direct P(2τ)
python -m agulhas coastal_pooling --res 1 --tau 3.5                           # needs the annual matrices from transition_matrix
```

Outputs of the whole-box, release-point-map, seasonal, Chapman–Kolmogorov and coastal-pooling steps carry the grids and lags they were run on in their names (`{tag}`, e.g. `0p5-1-2deg_3p5d_10E-55E_45S-15S`), so runs at other settings do not overwrite each other. So do the coverage files from `grid_coverage`, the release points and shipping grid from `shipping_overlay`, and the maps from `gdp_exploration` and `shipping_overlay`. Later steps read the release points for their box picked at the default grid and lag (`config.release_points_path`), whatever grid and lag they run on, so every setting is compared at the same points.

τ must be a whole number of hours. If a box side is not a multiple of the grid size (e.g. 45° at 2°), the last row or column is a partial cell cut off at the box edge.

`data/` and `figures/` are not in git. After cloning, `python -m agulhas all` rebuilds them.

### Another region

A box outside the area saved by `fetch_gdp_subset` needs its own subset, with a margin of about 10° so moves leaving the box are seen. Then pass `--box` and `--data` together to every step; a box outside the data file stops with a message. The Gulf Stream run:

```bash
python -m agulhas fetch_gdp_subset --fetch-box -90 -30 15 55 --out data/gulfstream_gdp1h_subset.nc
python -m agulhas all --box -80 -40 25 45 --data data/gulfstream_gdp1h_subset.nc
```

`python -m agulhas all --box -80 -40 25 45 --data data/gulfstream_gdp1h_subset.nc` runs the steps that apply, which are those not marked Agulhas-only in `python -m agulhas list`. `shipping_overlay` picks release points by traffic alone outside R, since its named ports are all in R. `box_selection` searches around the Agulhas core only, and `coastal_stranding`, `traffic_exposure`, `risk_atlas` and `known_features` stop with a message outside the Agulhas area, because the named coasts, the routes and the expected features describe that region. `fetch_era5` and `oil_wind` keep separate wind files for each subset, and need ERA5 for the extra years and area. Output names carry the box, and figure names start with the first word of the subset's file name (here `gulfstream_`).

---
## Whole-box results

At 1°, τ = 3.5 days, over reliable cells (≥ 10 drifters), drogued / undrogued unless stated. From the whole-box, release-point-map, seasonal, Chapman–Kolmogorov and coastal-pooling steps; seasonal robustness also uses 2°.

- **Data support.** 78% of active drogued rows and 93% of undrogued rows are reliable at 1° (52% / 91% at 0.5°, 90% / 94% at 2°). Coastal cells are much thinner: 33% / 47% reliable at 1°, against 82% / 98% offshore, so coastal and stranding results carry wide intervals.
- **Retention hotspots.** Drogued material is held longest at 41–44°E, 30–33°S, south-west of Madagascar: about 63–65% is still in R after a year (95% intervals about 48–74%), and 70 of the 75 hotspot cells stay in the top 10% in at least half of 1,000 drifter resamples. Undrogued material is held longest further east, at 48–51°E, 30–33°S (55–56%, intervals 49–60%). Where both grids are reliable, the 1° and 2° hotspots agree (Jaccard 76% drogued, 73% undrogued); 0.5° agrees less for drogued (37–41%), whose rows are thin. Retention after a year also reflects distance from the open edges of R, so a hotspot is a place R holds material, not necessarily a closed eddy. Held-out drifters show the highest undrogued retention overstated by about 10 points (95% intervals +1 to +21; `held_out_year`), so read the undrogued shares as upper values.
- **Residence time.** Material stays afloat in R for 223 / 203 days on average (median 223 / 213), with the longest stays in the hotspots (up to 518 / 459 days). The grids agree within about 40 days.
- **Exit zones.** Most material eventually leaves east with the Return Current: 69% / 62% on average, and east is the most likely fate in 83% / 75% of cells. The west (Atlantic leakage) takes 18% / 17%, the most likely fate in a strip along the west coast and the retroflection. 15% of undrogued material strands, and stranding is the most likely undrogued fate along the Mozambique Channel coast and east of Madagascar; drogued stranding is 4%, a lower bound that comes from the 18 drifters still drogued when they reached the coast (see `drogue_loss`).
- **Almost-invariant regions.** Four regions for both drogue types: the retroflection and the Atlantic side, the Natal coast and Mozambique Channel, south of Madagascar and the Return Current, and east of Madagascar. They keep 81–94% of their material over one step and 54–70% over 30 days, and by a year only 1–8% remains in its own region. Three of the four keep 13–23 points more over 30 days than longitude bands of the same sizes; the western one matches its band, whose edge falls close to the retroflection anyway. The leading eigenvalues fall off smoothly, so four is a choice rather than a gap in the spectrum.
- **Release points.** After a year, 18% (13–24%) of drogued material from Durban is still in R, 49% has left east, 19% west and 10% has stranded, the expected behaviour. Undrogued material from the Mozambique Channel lane (40.5°E, 16.5°S) and the lanes south of Madagascar strands far more: 24–66% after a year, against 1–27% drogued. Along the South African coast drogued and undrogued stranding are alike (9–19% against 10–38%), since most of the drifters that reached the coast drogued grounded there.
- **Seasonality.** Four seasonal operators (DJF, MAM, JJA and SON) are compared with each other and with a permutation test that shuffles season labels between blocks of one drifter's moves in one season of one year. At 1°, undrogued seasonal support remains good (70–75% of ocean rows reliable), with 551 of 970 cells reliable in all four seasons; pairwise median TVD on those common cells is 0.35–0.38. Drogued seasonal coverage is much thinner at 1° (12–25% reliable; only 26 cells reliable in all four seasons), so those differences are treated cautiously. At 2°, drogued support improves to 41–51% per season and 72 of 263 cells are reliable in all four seasons, while broad seasonal differences remain (pairwise median TVD 0.20–0.24). Undrogued support at 2° is 84–88%, with pairwise median TVD 0.15–0.16. Against shuffled seasons, drogued transport shows no seasonal change beyond sampling noise (no season pair at 1°, one of six at 2°), while undrogued transport does change with season (4 of 6 pairs at 1°, 5 of 6 at 2°, p ≤ 0.05), consistent with seasonal winds acting through windage, but only by 0.01–0.02 TVD above the shuffled level. The annual operator therefore stays the baseline.
- **Markov consistency.** A Chapman–Kolmogorov check compares `P(3.5 d)^2` with transitions estimated directly at 7 days. Over cells reliable at both lags, median row-wise TVD is 0.243 for drogued and 0.173 for undrogued drifters (731 / 892 common reliable cells); the 90th percentiles are 0.374 / 0.261. Nearly all of it is sampling noise in the direct 7-day rows: resampling whole drifters puts the noise at 0.224 / 0.169 per row, and only 1.8% / 3.7% of rows exceed the 95th percentile of their own noise, about the share expected by chance. Within sampling noise the 3.5-day operator is Markov; the drogued rows differ more only because their data are thinner.
- **Coastal pooling.** At 1°, τ = 3.5 days, a conservative Natural Earth 50m land-proximity filter removes four offshore candidates from the original 103 coastal cells, leaving 99 cells grouped into 44 contiguous coastal segments, with at most four neighbouring cells per segment. Before pooling, 35.0% of active drogued coastal rows and 48.9% of active undrogued coastal rows in this filtered set were reliable (≥ 10 distinct drifters). After pooling, 60.6% / 72.7% of the 99 coastal cells belong to a reliable pooled segment for drogued / undrogued drifters, and 29 of 44 segments (65.9%) are reliable for both types. The median segment contains two coastal cells; 15 segments remain thin and explicitly flagged. The geographic filter removes unsupported candidates but does not add potentially missing near-land cells. The before/after percentages use different denominators and do not demonstrate improved predictive accuracy. Pooling is used for coastal-segment summaries and future coastal-risk analysis only; it does not alter or smooth the main transition matrix.
- **Oil windage range.** If releases follow ship traffic, oil at 3% / 3.5% / 4% of the wind eventually strands 26% / 27% / 29% (undrogued 23%), in 98 / 87 / 78 days (undrogued 134). Going from 3% to 4% adds 2.5 points (1.6–3.1) and brings stranding 20 days sooner (17–24), both beyond sampling noise. Most of the extra stranding (84%) falls on Mozambique, and the order of the coasts does not change. Four of the 13 release points depend on the method at every windage (Cape Town, Durban and the lanes at 40.5°E and 45.5°E): oil and its drogued check differ by TVD > 0.3 at 30 days.
- **Spill hindcasts.** Two recorded spills off the Cape test the oil matrix under a rule fixed in advance. Castillo de Bellver (1983), which went offshore with little coastal pollution, is consistent: 11% (0–19%) strands within 30 days, moving north-west. Treasure (2000), which reached Robben and Dassen islands, is inconclusive: its own cell gives 90% with an interval of 0–100% from 4 drifters, and the nearest reliable cell, about 80 km away, gives 11% (0–19%). That cell is also the Castillo de Bellver cell, so the matrix gives one answer there to two spills that went different ways. These are checks of coast and timescale, since the matrix holds average conditions.
- **Stranding by coast.** Stranding is split into six named coasts by where drifters ran aground. Material travels far before grounding: from the lanes south and east of Madagascar, the largest share of oil strands on southern Mozambique (14–22%), not on Madagascar. The added windage brings oil ashore sooner rather than much more often: from the 47.5°E lane, oil strands 49% (42–55%) in 95 days (81–111) and undrogued material 39% (32–47%) in 151 days (127–178). Every drogued coast rests on fewer than 10 grounded drifters, so drogued results by coast are indicative only and a lower bound.
- **Traffic-weighted exposure.** If releases follow ship traffic, 27% of oil parcels eventually strand, in 87 days (79–97) on average: southern Mozambique 10% (6–12), central and northern Mozambique 8% (5–10), the South African south and east coast 4% (2–6). The Mozambique Channel route is the most dangerous (69% strands, in 38 days), while the Cape route west of Cape Agulhas carries 27% of traffic but strands only 2%, since 97% of its oil leaves into the Atlantic. Half of southern Mozambique's stranding comes from the route south of Madagascar. With a 14-day mass loss, only 3.6% of the released oil mass reaches a coast. 32% of the weight sits on unreliable cells and moves to the nearest reliable one, which changes each coast by under 1 point; weighting by lane pixels lowers the Mozambique coasts by about 3 points but keeps the same three coasts on top, so the ranking is firmer than the percentages.

## Second region: Gulf Stream

The core pipeline also runs on the Gulf Stream, box 80–40°W, 25–45°N, with no region-specific code changes (1°, τ = 3.5 days, drogued / undrogued; details in [Research.md](Research.md)). Retention peaks in the Sargasso Sea at about 58–61°W, 30–33°N, undrogued drifters slip west under the trades and east under the westerlies, and held-out exits and stranding are close to calibrated. Three things differ from R, so settings and skill checked in one region should be rechecked in another:

- **Point accuracy.** At 14 days the operator beats advection in no fold, and at 28 days the drogued operator beats persistence in only 2 of 5 folds. The undrogued operator beats both in all 5 folds at 28 days.
- **Grid.** Mean time in the box falls from 608 / 427 days at 0.5° to 350 / 305 days at 2°, against differences of about 40 days in R, so long-run results here depend on the grid.
- **Seasonality.** Drogued transport is seasonal beyond sampling noise in 5 of 6 season pairs, so the annual matrix is a weaker assumption here.
- **Windage and stranding.** Undrogued stranding exceeds drogued with non-overlapping intervals in only 1 of 8 lane cells, against 7 of 8 in R, since the busiest lanes here mostly strand little either way.

## Oil explorer

`agulhas/tools/oil_explorer.py` builds a single web page, `figures/agulhas_oil_explorer_….html`: click anywhere in R and play the year to see where oil released there goes, in the matrix's own 3.5-day steps. It shows where the oil drifts and how much is left under the 14-day decay, and reports the drifters behind the cell, the atlas's 95% intervals at 1 week, 1 month and 1 year, the coast that receives the stranding and whether the result depends on the method. The page does the matrix steps itself, so it needs no server. Like the worked example, it only reads pipeline outputs. The time axis is days since release under average conditions, so it is not a forecast.

```bash
python -m agulhas oil_explorer
```

## OpenDrift comparison

The matrix is also compared with OpenDrift, an open-source particle model used for spill work. `OceanDrift` runs on the real currents and winds of each date (GLORYS12 at 1/12° and ERA5), with oil at 3.5% of the wind and no weathering. It answers what the matrix's averaging costs and whether the matrix still gets the broad risks right; it does not show which is better, since the matrix only knows the average. These steps are not part of a full run, because OpenDrift needs Python 3.10 or newer:

```bash
python3.12 -m venv .venv-opendrift
.venv-opendrift/bin/pip install -r requirements-opendrift.txt
.venv-opendrift/bin/copernicusmarine login          # a free Copernicus Marine account

.venv-opendrift/bin/python -m agulhas opendrift_forcing    # about 3 GB, into data/opendrift/forcing
.venv-opendrift/bin/python -m agulhas opendrift_points     # 60 start dates at the 13 release points
.venv-opendrift/bin/python -m agulhas opendrift_treasure   # the Treasure spill; --leak-days 5 for the sensitivity
.venv-opendrift/bin/python -m agulhas opendrift_compare    # the table and figure
```

If `roaring_landmask` cannot load `liblzma` on macOS, install `xz`, or point the library at the system copy with `install_name_tool -change /opt/homebrew/opt/xz/lib/liblzma.5.dylib /usr/lib/liblzma.5.dylib` on its `.so` and re-sign it with `codesign -f -s -`.

## Worked example

`agulhas/tools/worked_example.py` answers the project's question for any point: where does material released here go? It reads the pipeline's outputs and reports what is needed to judge the answer, namely the data behind the cell, 95% intervals, the coast and, for oil, whether the result depends on the method. It also saves a map.

```bash
python -m agulhas worked_example --lon 40.5 --lat -16.5 --matrix oil
```

```text
Release: 40.5°E 16.5°S, in the 1° cell at 40.5°E 16.5°S; oil matrix, τ = 3.5 days
Data behind this cell: 25 drifters.

            afloat in R         exit W            exit E            exit S            exit N            stranded
   7 days   75.9% (62%–94%)      0.0% (0%–0%)      0.0% (0%–0%)      0.0% (0%–0%)      3.4% (1%–8%)     20.6% (0%–35%)
  30 days   37.7% (23%–55%)      0.0% (0%–0%)      0.0% (0%–0%)      0.0% (0%–0%)     21.1% (9%–36%)    41.2% (20%–61%)
 365 days    0.7% (0%–2%)        0.7% (0%–1%)      2.4% (1%–5%)      0.1% (0%–0%)     25.6% (12%–42%)   70.5% (53%–84%)

Eventually: exit W 1%, exit E 3%, exit S 0%, exit N 26%, stranded 71%. Mean time afloat in R: 41 days.

Where it strands, eventually:
  52.5%  Central and northern Mozambique
  10.0%  Southern Mozambique (to 21°S)
   7.1%  West and south Madagascar
   0.7%  South African south and east coast
  Among material that strands, it takes 35 days on average (22–56).

Method check at 30 days: oil against its drogued check differ by TVD 0.42, above 0.3, so this oil result depends on the method.
```

Read this as: oil released on the Mozambique Channel lane mostly comes ashore, about 41% within a month and 71% within a year, chiefly on central and northern Mozambique, but the amount depends on how oil is modelled. A thin release cell gets a warning and the nearest reliable cell to compare with. `--matrix` also takes `drogued`, `undrogued`, `oil_3pct` and `oil_4pct`, and `--box` and `--data` run it on another region.

## Using the transition matrix

Each `.npz` file is a self-contained operator. `P[i, j]` is the probability that material in state `i` is in state `j` one lag τ later (`tau_days`).

- **States `0 … n_cells − 1`**: ocean cells of R (970 at 1°; the mapping to the grid is in `cell_flat`).
- **Last 5 states**: absorbing, in the order of `exit_labels`: `W`, `E`, `S`, `N` (exited R across that edge) and `stranded` (ran aground on a coast inside R).

```python
from agulhas.transport import load_operator, release, propagate, to_grid

op = load_operator("data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz")
p0 = release(op, lon=31.5, lat=-30.5)      # all mass in the cell off Durban
p  = propagate(op, p0, days=30)            # rounded to whole 3.5-day steps
grid = to_grid(op, p)                      # 30 × 45 lat/lon array, NaN off-ocean
n_cells = len(op["cell_flat"])
print("still in R:", p[:n_cells].sum(), "exited / stranded:", dict(zip(op["exit_labels"], p[n_cells:])))
```

Every field of the file, and the oil, seasonal and coast variants, are in [docs/data.md](docs/data.md).

The coast operators from `coastal_stranding` (`data/P_{name}_coast_…npz`) are the same matrices with `stranded` split into six named coasts (`transport.COAST_LABELS`), so they have five more absorbing states; read them through `exit_labels`.

At τ = 3.5 days: 1 week = 2 steps, 1 month ≈ 9 steps, 1 year ≈ 104 steps.

## Status

- [x] Data subset, exploration, grid coverage
- [x] Time step: τ = 3.5 days (the velocity decorrelation time over the core region 10–40°E, 45–25°S is about 1.2–1.4 days)
- [x] Transition matrices (drogued and undrogued)
- [x] Region, grid and lag are parameters; matrices at 0.5°, 1° and 2° for both drogue types
- [x] Shipping lanes over currents; candidate release points chosen and ranked
- [x] Stranded state from GDP's record of drifters that ran aground in R (139 drifters, 18 of them drogued when they reached the coast); a stranding move takes the drogue status of its start, since the drogue flag often flips in a record's final hours (`drogue_loss`)
- [x] Drifter bootstrap: 95% intervals on release-point fates, drogued vs undrogued differences, row stability
- [x] Release-point maps at 1 week, 1 month, 1 year
- [x] Held-out validation (5-fold by drifter): calibrated exits and stranding, beats persistence and mean-current advection by 28 days
- [x] Held-out validation to one year: fates across R agree within 3.7 points at day 364 (undrogued from 105 drifters, drogued from only 14). Undrogued exits east are under-predicted by up to 4.4 points, the highest undrogued retention is overstated by about 10 points, and single release points cannot be tested at a year
- [x] Sensitivity to τ and grid size: τ = 2–5 days gives the same results; 1° is used (0.5° adds no skill and has 50% thin drogued rows, 2° loses short-range skill and shifts one-year fates)
- [x] Seasonal matrices, with a permutation test: no drogued seasonality beyond sampling noise, a small but real undrogued one
- [x] Calendar-ordered seasonal forecast: for a release in each month, every step uses its season's matrix; one-year fates differ from the annual matrix by a median of 0.2 / 0.4 points (drogued / undrogued) at 1° and 0.4 / 0.7 at 2°, with 1.0% / 5.3% and 1.6% / 8.5% of estimates outside the annual 95% interval. Drogued seasonal rows are mostly replaced by the annual row (49–88%), so only the undrogued result is informative
- [x] Chapman–Kolmogorov check: `P(3.5 d)^2` against direct 7-day transitions; median row TVD 0.243 / 0.173 for drogued / undrogued, almost all sampling noise (1.8% / 3.7% of rows above their 95% noise level)
- [x] Coastal pooling: Natural Earth 50m conservative geographic filter reduces 103 candidates to 99 coastal cells, grouped into 44 short segments; 29 segments are reliable for both drogue types, while 15 remain thin and flagged
- [x] ERA5 winds at every drifter position; oil matrix = undrogued + the missing 1.5% windage (undrogued carries 2.0%), with a drogued + 3.5% check
- [x] Whole-box analysis: accumulation zones, exit zones, residence time, almost-invariant regions
- [x] Stranding by coast: six named coasts, with intervals and days to strand, for drogued, undrogued, oil and the oil check
- [x] Traffic-weighted exposure by coast and route, with sensitivity to the weights
- [x] Release-risk atlas: every release cell and horizon, with intervals
- [x] Spill hindcasts: Castillo de Bellver consistent on all three rules; Treasure inconclusive, because its cell has only 4 undrogued drifters and 1° is too coarse for a spill about 10 km offshore
- [x] Oil windage range: 3%, 3.5% and 4% of the wind; windage changes how soon oil strands more than how much, and the coast ranking holds
- [x] Packaging: one command runs every step (`python -m agulhas all`); a fresh clone in a fresh environment from `requirements.txt` reproduces every output table and matrix exactly; worked example for any release point
- [x] Second region: the core pipeline on the Gulf Stream box, with no region-specific code changes
- [x] Known features: the current's speed and path, the end of the westward jet (18.5°E), the Return Current's latitude (38.5°S), Cape Basin flow and the leakage share (27%) all agree with the fixed rules; the drogued fate divide (14.5°E) falls just west of its range
- [x] Source and sink cells: about 10% of cells off the box edge are judged either, against 5% by chance, and the two drogue types barely agree (correlation 0.21), so retention and fates are the better guide
- [x] Hypothesis tests: H1 supported, H2 supported in 7 of 8 lane cells (drogued is a lower bound), H3 supported from 17.5 days (at 14 days the undrogued operator beats advection in 3 of 5 folds)
