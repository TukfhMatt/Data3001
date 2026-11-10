import numpy as np

from agulhas.opendrift_common import classify, hourly_states

BOX = (10.0, 55.0, -45.0, -15.0)
N_LON = 45
CELLS = {0: 0, 46: 1}     # the south-west corner cell, and the cell one up and one along from it
N = 2


def sort(x, y, stranded=False, last=None, hour=2):
    lon, lat = np.array([x], float), np.array([y], float)
    last = np.array([lon.shape[1] - 1 if last is None else last])
    return classify(lon, lat, np.array([stranded]), last, hour, CELLS, N, BOX, N_LON)[0]


def test_a_particle_in_the_box_gets_its_cell():
    assert sort([10.5, 10.6, 11.5], [-44.5, -44.4, -43.5]) == 1


def test_a_position_with_no_matrix_state_is_kept_apart():
    assert sort([30.5, 30.5, 30.5], [-30.5, -30.5, -30.5]) == N + 4 + 6


def test_the_first_edge_crossed_is_the_exit_and_it_does_not_return():
    assert sort([10.5, 9.9, 10.5], [-44.5, -44.5, -44.5]) == N + 0      # west, then back inside
    assert sort([54.9, 55.0, 54.0], [-30.0, -30.0, -30.0]) == N + 1     # the east edge belongs outside
    assert sort([20.0, 20.0, 20.0], [-44.9, -45.1, -44.9]) == N + 2     # south
    assert sort([20.0, 20.0, 20.0], [-15.1, -15.0, -15.1]) == N + 3     # north


def test_stranding_counts_only_once_it_has_happened():
    x, y = [18.0, 18.2, np.nan, np.nan], [-33.0, -33.2, np.nan, np.nan]     # lands at hour 1, at 18.2°E: Atlantic and Cape
    assert sort(x, y, stranded=True, last=1, hour=3) == N + 4
    x, y = [31.0, 31.5, np.nan], [-30.0, -30.0, np.nan]                     # South Africa's south and east coast
    assert sort(x, y, stranded=True, last=1, hour=2) == N + 4 + 1
    x, y = [10.5, 10.5, 10.5], [-44.5, -44.5, -44.5]                        # lands later than the hour asked for
    assert sort(x, y, stranded=True, last=2, hour=1) == 0


def test_a_particle_lost_inside_the_box_is_kept_apart():
    assert sort([20.0, np.nan, np.nan], [-30.0, np.nan, np.nan], last=0, hour=2) == N + 4 + 6


def test_every_hour_at_once_matches_one_hour_at_a_time():
    rng = np.random.default_rng(0)
    n, t = 300, 12
    lon = np.cumsum(np.c_[rng.uniform(9.5, 12.0, n), rng.normal(0, 0.4, (n, t - 1))], axis=1)
    lat = np.cumsum(np.c_[rng.uniform(-45.5, -43.0, n), rng.normal(0, 0.4, (n, t - 1))], axis=1)
    last = rng.integers(0, t, n)
    stranded = rng.random(n) < 0.5
    for i in range(n):
        lon[i, last[i] + 1:] = lat[i, last[i] + 1:] = np.nan
    every = hourly_states(lon, lat, stranded, last, CELLS, N, BOX, N_LON)
    for hour in range(t):
        assert (every[:, hour] == classify(lon, lat, stranded, last, hour, CELLS, N, BOX, N_LON)).all()
