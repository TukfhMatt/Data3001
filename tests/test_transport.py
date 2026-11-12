"""The transport helpers on small matrices whose answers can be worked out by hand."""

import numpy as np
import pytest
import xarray as xr

from agulhas import transport as tr


@pytest.fixture
def op():
    """Two ocean cells side by side (10–11°E and 11–12°E, 1–0°S) on a 3 × 2 grid
    with a third column of land: cell 0 keeps half and sends half to cell 1, cell
    1 sends half back and half out across the east edge."""
    n = 2
    P = np.zeros((n + 5, n + 5))
    P[0, [0, 1]] = 0.5
    P[1, 0] = 0.5
    P[1, n + 1] = 0.5
    P[n:, n:] = np.eye(5)
    return {"P": P, "cell_flat": np.array([3, 4]), "res": np.array(1.0), "tau_days": np.array(3.5),
            "lon_edges": np.array([10.0, 11.0, 12.0, 13.0]), "lat_edges": np.array([-2.0, -1.0, 0.0]),
            "exit_labels": tr.EXIT_LABELS, "flagged": np.array([False, True]), "empty": np.array([False, False])}


def test_state_of_uses_half_open_cells(op):
    assert tr.state_of(op, 10.0, -1.0) == 0
    assert tr.state_of(op, 10.999, -0.001) == 0
    assert tr.state_of(op, 11.0, -1.0) == 1


def test_state_of_refuses_a_cell_that_is_not_ocean(op):
    with pytest.raises(ValueError):
        tr.state_of(op, 12.5, -0.5)


def test_propagate_rounds_days_to_whole_steps(op):
    p0 = tr.release(op, 10.5, -0.5)
    assert p0.sum() == 1 and p0[0] == 1
    assert np.allclose(tr.propagate(op, p0, days=7)[:2], [0.5, 0.25])
    assert np.allclose(tr.propagate(op, p0, days=7), tr.propagate(op, p0, steps=2))
    assert np.isclose(tr.propagate(op, p0, days=7).sum(), 1)


def test_to_grid_marks_non_ocean_cells(op):
    grid = tr.to_grid(op, np.array([0.25, 0.75, 0, 0, 0, 0, 0]))
    assert grid.shape == (2, 3)
    assert grid[1, 0] == 0.25 and grid[1, 1] == 0.75
    assert np.isnan(grid[0]).all() and np.isnan(grid[1, 2])


def test_cell_centres_and_reliable(op):
    lon, lat = tr.cell_centres(op)
    assert lon.tolist() == [10.5, 11.5] and lat.tolist() == [-0.5, -0.5]
    assert tr.reliable(op).tolist() == [True, False]


def test_absorption_matches_the_hand_answer(op):
    steps, fate = tr.absorption(op)
    # N = (I − Q)⁻¹ with Q = [[.5, .5], [.5, 0]]: N·1 = [6, 4]
    assert np.allclose(steps, [6, 4])
    assert np.allclose(fate[:, 1], 1)          # everything leaves east
    assert np.allclose(fate.sum(axis=1), 1)


def test_absorption_refuses_a_closed_cell(op):
    op["P"][1] = 0
    op["P"][1, 1] = 1
    op["P"][0] = 0
    op["P"][0, 0] = 1
    with pytest.raises(ValueError):
        tr.absorption(op)


def test_stranding_chain_leaves_closed_cells_out():
    # state 0 strands for sure in 2 steps through state 1; state 2 is closed
    n = 3
    P = np.zeros((n + 2, n + 2))
    P[0, 1] = 1
    P[1, n + 1] = 1
    P[2, 2] = 1
    P[n:, n:] = np.eye(2)
    fate, visits = tr.stranding_chain(P, n, strand_from=1)
    assert fate[:, 1].tolist() == [1, 1, 0]
    assert visits[0] == 2 and visits[2] == 0


def test_days_to_strand_is_empty_below_min_stranding():
    # 2 steps to strand where half strands; rounding-level stranding gets no time
    days = tr.days_to_strand(np.array([1.0, 2e-17, 0.0]), np.array([0.5, 1e-17, 0.0]), tau=3.5)
    assert days[0] == 7.0
    assert np.isnan(days[1:]).all()

def test_coast_of_splits_at_the_stated_lines():
    lon = [19.9, 20.0, 33.0, 36.0, 44.0, 47.0]
    lat = [-34.0, -34.0, -25.0, -18.0, -22.0, -18.0]
    assert tr.coast_of(lon, lat).tolist() == [0, 1, 2, 3, 4, 5]
    assert tr.coast_of(33.0, -26.9).tolist() == 2
    assert tr.coast_of(33.0, -26.91).tolist() == 1


def test_check_coast_box_stops_outside_the_agulhas():
    tr.check_coast_box((10.0, 55.0, -45.0, -15.0))
    with pytest.raises(SystemExit):
        tr.check_coast_box((-80.0, -40.0, 25.0, 45.0))


def test_pair_indices_pairs_only_within_one_drifter():
    traj = np.array([0, 0, 0, 1, 1])
    t = np.array([0, 3600, 7200, 3600, 7200])
    start, end = tr.pair_indices(traj, t, np.ones(5, bool), 1 / 24)
    assert start.tolist() == [0, 1, 3] and end.tolist() == [1, 2, 4]


def test_pair_indices_starts_must_be_inside_the_box():
    traj = np.zeros(3, int)
    t = np.array([0, 3600, 7200])
    start, end = tr.pair_indices(traj, t, np.array([False, True, True]), 1 / 24)
    assert start.tolist() == [1] and end.tolist() == [2]


def strand_data():
    """Drifter 0 ran aground inside the box at its last fix, 6 h after it began;
    drifter 1 stopped transmitting, and drifter 2 ran aground outside the box."""
    traj = np.repeat([0, 1, 2], 3)
    t = np.tile([0, 3 * 3600, 6 * 3600], 3)
    in_box = np.array([True] * 6 + [True, True, False])
    end = np.datetime64("1970-01-01T06:00:00", "ns")
    ds = xr.Dataset({"end_date": ("traj", [end] * 3), "typedeath": ("traj", [1, 3, 1])})
    return ds, traj, t, in_box


def test_strand_times_only_marks_groundings_inside_the_box():
    ds, traj, t, in_box = strand_data()
    assert tr.strand_times(ds, traj, t, in_box).tolist() == [21600] * 3 + [-1] * 6


def test_strand_times_needs_the_record_to_reach_the_end_date():
    ds, traj, t, in_box = strand_data()
    ds["end_date"] = ("traj", [np.datetime64("1970-01-05T00:00:00", "ns")] * 3)
    assert (tr.strand_times(ds, traj, t, in_box) == -1).all()


def test_move_indices_sends_a_drifter_s_final_lag_to_stranded():
    ds, traj, t, in_box = strand_data()
    start, end, stranded = tr.move_indices(traj, t, in_box, 3 / 24, tr.strand_times(ds, traj, t, in_box))
    # Drifter 0: 0→1 and 1→2 are ordinary, and its last fix has no pair so it strands
    assert start[:3].tolist() == [0, 1, 2] and end[:3].tolist() == [1, 2, 2]
    assert stranded[:3].tolist() == [False, False, True]
    assert stranded[3:].sum() == 0


def test_same_drogue_needs_both_ends_except_into_stranded():
    drogue = np.array([True, True, False])
    start, end = np.array([0, 0]), np.array([1, 2])
    assert tr.same_drogue(drogue, start, end, np.array([False, False]), True).tolist() == [True, False]
    assert tr.same_drogue(drogue, start, end, np.array([False, True]), True).tolist() == [True, True]
    assert tr.same_drogue(drogue, start, end, np.array([False, True]), False).tolist() == [False, False]


def test_transition_states_names_the_exit_side_and_stranding():
    box = (10.0, 12.0, -1.0, 0.0)
    cell_flat = np.array([0, 1, -1, -1, 0])
    lon = np.array([10.5, 11.5, 9.0, 12.5, 10.5])
    lat = np.array([-0.5, -0.5, -0.5, -0.5, -0.5])
    start, end = np.array([0, 1, 1, 0]), np.array([1, 2, 3, 4])
    stranded = np.array([False, False, False, True])
    f, to = tr.transition_states(cell_flat, np.array([0, 1]), start, end, lon, lat, box, stranded)
    assert f.tolist() == [0, 1, 1, 0]
    assert to.tolist() == [1, 2, 3, 2 + tr.STRANDED]   # cell 1, W, E, stranded


def test_row_normalise_pools_an_empty_cell_from_its_nearest_ring():
    # Three cells in a row and one absorbing state; the middle cell has no moves
    C = np.array([[2.0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
    P, empty = tr.row_normalise(C, [0, 1, 2], n_lon=3)
    assert empty.tolist() == [1]
    assert np.allclose(P.sum(axis=1), 1)
    assert np.allclose(P[1], [0.25, 0, 0.125, 0.625])
    assert P[3, 3] == 1


def test_row_normalise_holds_mass_when_no_neighbour_has_data():
    C = np.zeros((3, 3))
    C[0, 2] = 1
    P, empty = tr.row_normalise(C, [0, 90], n_lon=10, max_ring=1)
    assert empty.tolist() == [1] and P[1, 1] == 1


def test_a_bootstrap_draw_keeps_whole_drifters():
    traj = np.array([7, 7, 9])
    f, to = np.array([0, 0, 1]), np.array([1, 1, 0])
    counts = tr.drifter_counts(traj, f, to, 2)
    assert counts[3] == 2
    C = tr.resampled_counts(counts, 2, np.random.default_rng(0))
    assert C.sum() in (2, 3, 4, 6) and C[0, 1] % 2 == 0    # drifter 7's two moves stay together
