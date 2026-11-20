"""traffic_exposure reports the share stranding on any coast with its own interval.

The coast shares move together across drifter resamples, so the interval of their sum has
to come from summing inside each resample; adding per-coast interval ends would give the
wrong range. The totals go to their own table so the per-coast table is unchanged. These
tests use synthetic outputs, so they need no data download.
"""
import re

import numpy as np
import pandas as pd
import pytest

from agulhas.coast import traffic_exposure as te
from agulhas.transport import COAST_LABELS

R, K, E = len(te.RELEASES), len(COAST_LABELS), len(te.EXIT_NAMES)


def _out(rng):
    out = {"eventual": 0.05 + rng.random((R, K)) / K, "exit": rng.random((R, E)),
           "days": 50 + rng.random(R) * 100, "nearshore": rng.random((R, K)),
           "source": rng.random((len(te.ROUTES), K))}
    out["total"] = out["eventual"].sum(axis=1)
    for d in te.HORIZONS:
        out[d] = 0.05 + rng.random((R, K)) / K
        out[f"total_{d}"] = out[d].sum(axis=1)
        out[f"in_R_{d}"] = rng.random(R)
    return out


def _ci(out, width):
    return {key: np.stack([value - width, value + width]) for key, value in out.items()}


def _boot(width=0.01, seed=0):
    out = _out(np.random.default_rng(seed))
    return {name: out for name in te.BOOT_MATRICES}, {name: _ci(out, width) for name in te.BOOT_MATRICES}


def test_totals_table_is_the_sum_of_the_coast_rows():
    full, intervals = _boot()
    totals = te.totals_table(full, intervals)
    coasts = pd.DataFrame(te.rows_for("oil", "density", full["oil"], intervals["oil"]))
    coasts = coasts[coasts.metric == "stranded"]
    for horizon in ["eventual"] + te.HORIZONS:
        for release in te.RELEASES:
            total = totals[(totals.matrix == "oil") & (totals.horizon == horizon) & (totals.release == release)]
            assert len(total) == 1
            part = coasts[(coasts.horizon == horizon) & (coasts.release == release)]
            assert total.estimate.item() == pytest.approx(part.estimate.sum())


def test_totals_table_carries_its_own_interval():
    full, intervals = _boot(width=0.02, seed=1)
    totals = te.totals_table(full, intervals)
    row = totals[(totals.matrix == "drogued") & (totals.horizon == 30) & (totals.release == "all")].iloc[0]
    assert row.lo == pytest.approx(full["drogued"]["total_30"][0] - 0.02)
    assert row.hi == pytest.approx(full["drogued"]["total_30"][0] + 0.02)


def test_per_coast_table_is_unchanged_by_the_totals():
    out = _out(np.random.default_rng(2))
    table = pd.DataFrame(te.rows_for("oil", "density", out, _ci(out, 0.01)))
    assert "all coasts" not in set(table.target)
    stranded = table[(table.metric == "stranded") & (table.horizon == "eventual") & (table.release == "all")]
    assert list(stranded.target) == list(COAST_LABELS)


def test_printed_any_coast_table_shows_an_interval_for_every_cell(capsys):
    full, intervals = _boot(seed=3)
    te.print_stranding(full, intervals)
    section = capsys.readouterr().out.split("Stranding (any coast)")[1]
    shown = re.findall(r"\d+% \(\d+%\u2013\d+%\)", section)
    assert len(shown) == R * len(te.BOOT_MATRICES) * 2
