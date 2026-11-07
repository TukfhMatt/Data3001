"""The shared building blocks (drifters, grid, moves, stats, shipping, plotting) on
small inputs whose answers can be worked out by hand."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from agulhas import transport as tr
from agulhas.drifters import Drifters, in_box
from agulhas.grid import Grid
from agulhas.moves import MoveSet
from agulhas.shipping import blocks
from agulhas.stats import haversine_km, interval

BOX = (10.0, 12.0, -2.0, 0.0)


@pytest.fixture
def subset(tmp_path):
    """Two drifters, hourly, in a 2° × 2° box. Drifter 0 (drogued) moves east along
    0.5°S from 10.1°E, 0.1° an hour for 10 hours, and runs aground at its last fix.
    Drifter 1 (undrogued) sits at 11.5°E, 1.5°S for 6 hours and stops transmitting."""
    t0 = np.datetime64("2001-03-01T00:00")
    hours = np.r_[np.arange(10), np.arange(6)].astype("timedelta64[h]")
    ds = xr.Dataset(
        {"lon": ("obs", np.r_[10.1 + 0.1 * np.arange(10), [11.5] * 6]),
         "lat": ("obs", np.r_[[-0.5] * 10, [-1.5] * 6]),
         "ve": ("obs", np.r_[[0.3] * 10, [0.0] * 6]), "vn": ("obs", np.zeros(16)),
         "drogue_status": ("obs", np.r_[[True] * 10, [False] * 6]),
         "time": ("obs", t0 + hours),
         "rowsize": ("traj", [10, 6]), "typedeath": ("traj", [1, 3]),
         "end_date": ("traj", [t0 + np.timedelta64(9, "h"), t0 + np.timedelta64(5, "h")])})
    path = tmp_path / "subset.nc"
    ds.to_netcdf(path)
    return str(path)


def test_in_box_uses_half_open_edges():
    assert in_box(np.array([10.0, 11.999, 12.0]), np.array([-2.0, -0.001, -1.0]), BOX).tolist() == [True, True, False]


def test_grid_clips_a_partial_last_cell_at_the_box_edge():
    g = Grid((10, 15, -2, 0), 2.0)
    assert (g.n_lon, g.n_lat, g.n_cells) == (3, 1, 3)
    assert g.lon_edges.tolist() == [10, 12, 14, 15]
    assert g.lon_centres.tolist() == [11, 13, 14.5]
    assert g.cell_of(np.array([10.0, 14.9, 15.0]), np.array([-1.0, -0.1, -1.0])).tolist() == [0, 2, -1]


def test_grid_cell_index_is_row_from_the_south_times_columns_plus_column():
    g = Grid(BOX, 1.0)
    assert g.cell_in_box(np.array([10.5, 11.5, 10.5]), np.array([-1.5, -1.5, -0.5])).tolist() == [0, 1, 2]


def test_drifters_reads_the_subset(subset):
    d = Drifters(subset)
    assert d.traj_idx.tolist() == [0] * 10 + [1] * 6
    assert d.year[0] == 2001 and d.month[0] == 3
    assert d.t[1] - d.t[0] == 3600
    strand_t = d.strand_times(in_box(d.lon, d.lat, BOX))
    assert (strand_t[:10] == d.t[9]).all() and (strand_t[10:] == -1).all()


def test_moveset_pairs_each_drifter_tau_apart_and_ends_in_stranding(subset):
    moves = MoveSet(Drifters(subset), BOX, 1.0, 3 / 24)
    assert moves.grid.n_cells == 4
    # Drifter 0 pairs hours 0–6 with 3–9; hours 7–9 end in stranding. Drifter 1 pairs hours 0–2.
    assert moves.start.tolist() == list(range(10)) + [10, 11, 12]
    assert moves.stranded.tolist() == [False] * 7 + [True] * 3 + [False] * 3
    s, e, st = moves.of_type(True)
    assert s.tolist() == list(range(10)) and st.sum() == 3
    assert moves.of_type(False)[0].tolist() == [10, 11, 12]


def test_counts_for_checks_the_saved_counts(subset):
    moves = MoveSet(Drifters(subset), BOX, 1.0, 3 / 24)
    cells = np.unique(moves.cell_flat[moves.in_R])
    n_states = len(cells) + len(tr.EXIT_LABELS)
    f, to = moves.states(cells, *moves.of_type(True))
    C = np.zeros((n_states, n_states))
    np.add.at(C, (f, to), 1)
    op = {"cell_flat": cells, "P": np.eye(n_states), "C": C}
    counts = moves.counts_for(op, "drogued", True)
    assert counts[3] == 1 and counts[2].sum() == 10
    op["C"] = C + np.eye(n_states)
    with pytest.raises(AssertionError):
        moves.counts_for(op, "drogued", True)


def test_haversine_gives_one_degree_of_latitude():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.01)
    assert haversine_km(30, -40, 30, -40) == 0


def test_interval_ignores_nan_draws():
    draws = np.r_[np.arange(1, 101, dtype=float), [np.nan] * 10]
    lo, hi = interval(draws)
    assert lo == pytest.approx(3.475) and hi == pytest.approx(97.525)


def test_blocks_keeps_the_pixels_of_a_partial_last_cell():
    a = np.ones((3, 5))
    sums = blocks(a, 0.2, 0.1).sum(axis=(1, 3))
    assert sums.tolist() == [[4, 4, 2], [2, 2, 1]]


def test_save_figure_makes_the_folder_and_closes_the_figure(tmp_path):
    import matplotlib.pyplot as plt
    from agulhas.plotting import save_figure
    plt.plot([0, 1])
    path = tmp_path / "figures" / "x.png"
    save_figure(str(path))
    assert path.exists() and plt.get_fignums() == []


def test_coast_counts_must_match_the_operator(tmp_path):
    path = tmp_path / "coast_counts.npz"
    np.savez(path, n_states=7, drogued_d_idx=[0], drogued_ft=[1], drogued_n=[2], drogued_n_drifters=1)
    counts = tr.load_coast_counts(path, ["drogued"], 7)
    assert [c.tolist() if hasattr(c, "tolist") else c for c in counts["drogued"]] == [[0], [1], [2], 1]
    with pytest.raises(SystemExit):
        tr.load_coast_counts(path, ["drogued"], 8)


def test_each_operator_loops_lags_then_grids_then_drogue_types(monkeypatch):
    seen = []
    monkeypatch.setattr(tr, "load_operator", lambda path: seen.append(path) or {})
    args = SimpleNamespace(res=[1.0, 2.0], tau=[3.5], box=BOX)
    out = [(res, tau, name) for res, tau, name, _ in tr.each_operator(args)]
    assert out == [(1.0, 3.5, "drogued"), (1.0, 3.5, "undrogued"), (2.0, 3.5, "drogued"), (2.0, 3.5, "undrogued")]
