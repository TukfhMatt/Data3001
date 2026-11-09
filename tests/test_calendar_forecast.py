"""The calendar-ordered seasonal forecast: which season each step takes, how a
thin seasonal matrix falls back to the annual one, and that chaining the
matrices reduces to the annual forecast when the seasons do not differ."""

import numpy as np
import pytest

from agulhas.operators.seasonal_matrices import SEASONS
from agulhas.release_points.calendar_forecast import (blend, fate_shares, propagate_calendar, season_of,
                                                      season_sequence)


def stochastic(seed, n=4, absorbing=True):
    """A random row-stochastic matrix, with its last state absorbing unless told otherwise."""
    P = np.random.default_rng(seed).random((n, n))
    if absorbing:
        P[-1] = 0
        P[-1, -1] = 1
    return P / P.sum(axis=1, keepdims=True)


def test_every_month_has_one_season():
    assert {season_of(m) for m in range(1, 13)} == set(SEASONS)
    assert [season_of(m) for m in (12, 1, 2)] == ["DJF"] * 3
    assert [season_of(m) for m in (6, 7, 8)] == ["JJA"] * 3


def test_a_release_starts_in_its_own_season_and_moves_on_with_the_calendar():
    seq = season_sequence(7, 365, 3.5)
    assert len(seq) == 104
    assert seq[:9] == ["JJA"] * 9          # days 0-28, July
    assert seq[9] == "JJA"                  # day 31.5, August
    assert seq[18] == "SON"                 # day 63, September


@pytest.mark.parametrize("start", range(1, 13))
def test_a_year_passes_through_all_four_seasons(start):
    assert set(season_sequence(start, 365, 3.5)) == set(SEASONS)


@pytest.mark.parametrize("tau", [2, 3.5, 7])
def test_the_number_of_steps_follows_the_lag(tau):
    assert len(season_sequence(1, 365, tau)) == int(round(365 / tau))


def test_equal_seasonal_matrices_give_the_annual_forecast():
    P = stochastic(1)
    p0 = np.array([1.0, 0, 0, 0])
    mats = {s: P for s in SEASONS}
    got = propagate_calendar(p0, mats, season_sequence(4, 365, 3.5))
    assert np.allclose(got, p0 @ np.linalg.matrix_power(P, 104))


def test_the_matrices_are_applied_in_calendar_order():
    A, B = stochastic(2, absorbing=False), stochastic(3, absorbing=False)
    mats = {"DJF": A, "MAM": B, "JJA": A, "SON": B}
    p0 = np.array([1.0, 0, 0, 0])
    seq = season_sequence(5, 90, 3.5)             # May: MAM first, then JJA
    assert seq[0] == "MAM" and seq[-1] == "JJA"
    manual = p0
    for season in seq:
        manual = manual @ mats[season]
    assert np.allclose(propagate_calendar(p0, mats, seq), manual)
    assert not np.allclose(propagate_calendar(p0, mats, seq[::-1]), manual)
    assert np.isclose(manual.sum(), 1)


def test_a_thin_seasonal_row_takes_the_annual_row():
    annual = {"P": stochastic(4), "cell_flat": np.array([5, 6, 9])[:2]}
    seasonal = {"P": stochastic(5), "cell_flat": annual["cell_flat"].copy(),
                "flagged": np.array([True, False]), "empty": np.array([False, False])}
    P, share = blend(annual, seasonal)
    assert np.allclose(P[0], annual["P"][0]) and np.allclose(P[1], seasonal["P"][1])
    assert np.allclose(P[2:], seasonal["P"][2:])       # absorbing rows are untouched
    assert share == 0.5
    assert np.allclose(P.sum(axis=1), 1)


def test_an_empty_seasonal_row_also_takes_the_annual_row():
    annual = {"P": stochastic(6), "cell_flat": np.array([1, 2])}
    seasonal = {"P": stochastic(7), "cell_flat": np.array([1, 2]),
                "flagged": np.array([False, False]), "empty": np.array([False, True])}
    P, share = blend(annual, seasonal)
    assert np.allclose(P[1], annual["P"][1]) and share == 0.5


def test_seasonal_matrices_with_other_states_are_refused():
    annual = {"P": stochastic(8), "cell_flat": np.array([1, 2])}
    other = {"P": stochastic(9), "cell_flat": np.array([1, 3]),
             "flagged": np.array([False, False]), "empty": np.array([False, False])}
    with pytest.raises(SystemExit):
        blend(annual, other)


def test_fate_shares_account_for_all_the_mass():
    p = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
    shares = fate_shares(p, ["W", "E", "stranded"], 2)
    assert shares["in_R"] == pytest.approx(0.5)
    assert shares["exit_W"] == pytest.approx(0.1) and shares["exit_stranded"] == pytest.approx(0.15)
    assert sum(shares.values()) == pytest.approx(1)
