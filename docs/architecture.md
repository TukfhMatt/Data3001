# Architecture

How the code is organised, what each shared module is for, and the shape every step follows. For what each step reads and writes, see [pipeline.md](pipeline.md); for the fields of the saved files, see [data.md](data.md).

## Layers

```text
              sources/            fetch_gdp_subset, fetch_era5        → data/*_gdp1h_subset.nc, ERA5 winds
                 │
              exploration/        look at the data, choose box, grid, lag
                 │
              operators/          transition_matrix, oil_wind, seasonal_matrices
                 │                                                    → data/P_*.npz
     ┌───────────┼──────────────┬──────────────┐
release_points/  whole_box/     coast/         validation/            read the matrices
     │                                                                → data/*.csv, figures/*.png
     └───────────────── tools/ (worked_example, oil_explorer): read the outputs only
                        comparison/ (opendrift_*): OpenDrift against the oil matrix, in its own environment
```

Each folder of `agulhas/` is one topic, and each module in it is one step, named by its file: `python -m agulhas bootstrap` runs `agulhas/validation/bootstrap.py`. `agulhas/pipeline.py` lists the steps in the order they run (`STEPS`), with the options each variant passes, so a step always comes after the steps whose files it reads. `tests/test_pipeline.py` checks that order for the dependencies that matter.

## Shared modules

Steps hold only what is particular to them. Everything two steps would otherwise both write lives in one of these:

| Module | What it holds |
|---|---|
| `config.py` | Defaults (box, grids, lag, `MIN_DRIFTERS`, oil windage range), the command-line options every step takes (`parse_args`), and every file name: `operator_path`, `release_points_path`, `wind_paths`, `setting_tag` and the other tags. |
| `drifters.py` | `Drifters(path)`: the GDP subset, loading each array the first time a step asks for it; `in_box` with half-open edges. |
| `grid.py` | `Grid(box, res)`: cell edges and centres (a partial last cell is clipped to the box), and the cell index of any position. |
| `moves.py` | `MoveSet(drifters, box, res, tau)`: every one-step move from the box on one grid and lag, filtered by drogue type, turned into matrix states, or grouped per drifter for a bootstrap and checked against a saved matrix (`counts_for`). |
| `transport.py` | Working with saved matrices (`load_operator`, `release`, `propagate`, `to_grid`, `absorption`, `stranding_chain`, `reliable`, `coastal_states`) and the helpers every matrix build shares (`strand_times`, `move_indices`, `same_drogue`, `transition_states`, `row_normalise`, `drifter_counts`, `resampled_counts`, `load_coast_counts`). Named coasts of the Agulhas area (`COAST_LABELS`, `coast_of`). |
| `plotting.py` | The headless backend, cartopy with a plain-matplotlib fallback (`MAP_KW`, `TRANSFORM_KW`), `map_axes`, `add_gridlines`, `save_figure` (150 dpi) and the drogue-type colours. |
| `shipping.py` | The shipping density raster: `read_window` reads one box straight from the zip, `blocks` groups its pixels into grid cells. |
| `stats.py` | `haversine_km` and `interval` (the 95% interval of resampled values). |
| `opendrift_common.py` | What the OpenDrift comparison steps share: their settings, one `OceanDrift` run (`simulate`), and `classify` and `hourly_states`, which sort particles into the matrix's states at one hour or at every hour. OpenDrift is imported only inside `simulate`, so the rest also works in the main environment. |
| `pipeline.py` | The step list and the `python -m agulhas` command (`all`, `list`, one step by name). |

A step that builds or resamples a matrix starts from `MoveSet`, so every matrix in the project is built by the same code as `transition_matrix`; `counts_for` stops a step whose rebuilt counts differ from the saved matrix.

## Anatomy of a step

```python
"""
Agulhas Region — Title
What the step does, what it needs, and how to run it:

    python -m agulhas step_name --res 1 --tau 3.5
"""

import ...                                  # standard library, third party, then agulhas

CONSTANT = ...                              # the step's fixed choices, with a comment each


# ── Section name ─────────────────────────────────────────────────────────────

def load_something(...):                    # small functions, grouped under section headers:
def build_or_compute_something(...):        # load_*, build_* / compute, print_*, plot_*, save_*
def print_summary(...):
def plot_something(..., path):              # ends in save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_csv = f"data/name_{tag}.csv"        # every output path, up front
    out_png = f"figures/{region_prefix(args.data)}_name_{tag}.png"

    table = compute(load(...))              # main reads like a table of contents
    table.to_csv(out_csv, index=False)
    print_summary(table)
    plot_something(table, args, out_png)
    print(f"Saved → {out_csv}, {out_png}")


if __name__ == "__main__":
    main()
```

Rules that follow from it, and that `tests/test_structure.py` checks:

- Importing a step does nothing; the run is `main(argv=None)`, which passes `argv` to `parse_args`.
- `main()` stays under 40 lines and no function over 80; a step that needs shared state across many functions puts it in a small class (`HeldOut` in `held_out`, which `held_out_year` extends; `SegmentPool` in `coastal_pooling`; `OilMoves` in `oil_wind`).
- Any module that uses pyplot imports `agulhas.plotting`, which alone sets the backend.
- `docs/pipeline.md` lists every step and this page every shared module.

## Conventions that hold across the code

- **Grid cells** are half-open, `[min, max)`, indexed `j * n_lon + i` from the box's south-west corner. A box side that is not a multiple of the grid ends in a partial cell clipped at the box edge.
- **States** of a matrix: the ocean cells of the box (any cell a drifter of either type visited, so both types share them), then the absorbing states in the order of `exit_labels` (`W, E, S, N, stranded`, or six named coasts in a coast operator). Distributions are row vectors, stepped with `p @ P`.
- **Drogue types** are never pooled. A move counts for a type if both ends have that status, except a move into stranding, which needs it at the start only (`same_drogue`).
- **Reliability**: a row from fewer than `MIN_DRIFTERS` distinct drifters is `flagged`; a cell with no moves of its own is `empty` and takes its neighbours' moves. Release points and reported results use reliable cells.
- **Uncertainty** comes from resampling whole drifters, each drogue type independently. Each step that resamples seeds its own generator (`SEED`) and draws in a fixed order, so reruns give the same intervals.
- **File names** come from `config.py` and carry the box, grid and lag, so runs at different settings never overwrite each other.
