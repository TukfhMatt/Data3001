# Contributing

How to set up, change and check the code. Read [docs/architecture.md](docs/architecture.md) first for how the package is laid out, and [docs/pipeline.md](docs/pipeline.md) for what each step reads and writes.

## Setup

Python 3.9, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt     # requirements.txt plus pytest and pyflakes
```

Run everything from the repository root: steps read and write `data/` and `figures/` there. Versions in `requirements.txt` are the ones the results were built with; keep `zarr` on 2.x, which `fetch_gdp_subset` relies on.

## Before every commit

```bash
python -m pyflakes agulhas tests
python -m pytest
```

The tests need no data and finish in seconds. Besides the shared helpers, they check the layout rules below (`tests/test_structure.py`) and the step order (`tests/test_pipeline.py`). CI (`.github/workflows/tests.yml`) runs both on every push and pull request.

## Branches and commits

- Work on a branch and merge through a pull request; the template asks how results were checked.
- Never commit `data/`, `figures/`, `*.nc`, `*.csv`, `*.zarr` or `~/.cdsapirc`. `.gitignore` covers the first five; the CDS key lives outside the repository.
- Commit messages say what the change does, in the imperative ("Split held_out into functions").

## Adding a step

1. Write `agulhas/<topic>/<step>.py` in the shape of [the anatomy of a step](docs/architecture.md#anatomy-of-a-step): a docstring whose first line is `Agulhas Region — Title`, constants, small functions under `# ── Section ──` headers, and a short `main(argv=None)`.
2. Build on the shared modules: `parse_args` and the path helpers in `config.py`; `Drifters`, `Grid` and `MoveSet` for anything that reads the subset or builds a matrix; `transport` for anything that uses a saved matrix; `plotting` for figures; `stats` for distances and intervals.
3. Add it to `STEPS` in `agulhas/pipeline.py` after every step whose output it reads (or to `TOOLS` if it only reads outputs), with `agulhas_only=True` if it needs the Agulhas coasts, routes or winds. Add the dependency to `tests/test_pipeline.py` if the order matters.
4. Add a row to `docs/pipeline.md`; the tests fail until it is there.
5. Name every output with `setting_tag` (or the narrower tag the step depends on), so runs at other settings never overwrite each other.
6. After the step runs, add its key numbers to the checklist in `Research.md` and the Status section of the README.

## Rules the code keeps

- A module does nothing when imported. Steps print a short summary and the paths they saved.
- `main()` stays under 40 lines, no function over 80. Shared state across many functions goes in a small class rather than long parameter lists.
- Figures go through `agulhas.plotting` (`save_figure`, `map_axes`), which alone sets the headless backend.
- Never pool drogued and undrogued drifters. Release points and results use cells reliable in both matrices; report `flagged` and `empty` for any other cell.
- Uncertainty comes from resampling whole drifters, never observations. A step that resamples seeds its own generator; keep the order of draws when changing it, or the intervals change.
- Never load the full global GDP arrays (the machine the project runs on has 8 GB); stream them as `fetch_gdp_subset` does.
- Docs, docstrings and comments describe the current approach only. Git history holds how it got there.

## Changing code without changing results

A refactor must leave every output identical. To check one:

1. Before the change, run the affected steps (and each variant in `STEPS`) on both regions and keep `data/`, `figures/` and the printed output.
2. Make the change, rerun the same steps in a copy of that folder, and compare every file byte for byte (`cmp`) and the printed output line for line.
3. A random draw that moves, an array saved in another order or a figure saved with other settings all show up as differences; fix them rather than accept them, unless the change is meant to alter an output, in which case say so in the pull request.

The second region is the Gulf Stream: `--box -80 -40 25 45 --data data/gulfstream_gdp1h_subset.nc`. Agulhas-only steps stop outside the Agulhas area by design.
