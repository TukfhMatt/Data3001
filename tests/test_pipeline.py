import importlib
import pathlib

import pytest

from agulhas import pipeline

ALL = pipeline.STEPS + pipeline.TOOLS


def order(name):
    return [s.name for s in pipeline.STEPS].index(name)


def test_step_names_are_unique():
    names = [s.name for s in ALL]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("step", ALL, ids=lambda s: s.name)
def test_a_step_imports_without_running(step):
    """Importing does no work; everything happens in main()."""
    assert callable(importlib.import_module(f"agulhas.{step.module}").main)


def test_every_module_is_a_step():
    root = pathlib.Path(pipeline.__file__).parent
    found = {".".join(p.relative_to(root).with_suffix("").parts) for p in root.glob("*/*.py") if p.stem != "__init__"}
    assert found == {s.module for s in ALL}


@pytest.mark.parametrize("before, after", [
    ("fetch_gdp_subset", "transition_matrix"),
    ("transition_matrix", "shipping_overlay"),
    ("shipping_overlay", "bootstrap"),
    ("transition_matrix_lags", "chapman_kolmogorov"),
    ("fetch_era5", "oil_wind"),
    ("bootstrap", "release_point_maps"),
    ("held_out_res2", "sensitivity"),
    ("coastal_pooling", "traffic_exposure"),
    ("oil_wind", "coastal_stranding"),
    ("coastal_stranding", "traffic_exposure"),
    ("traffic_exposure", "oil_windage"),
    ("coastal_stranding", "risk_atlas"),
    ("seasonal_matrices", "spill_hindcasts"),
    ("seasonal_matrices", "calendar_forecast"),
    ("bootstrap", "calendar_forecast"),
    ("seasonal_matrices_res2", "calendar_forecast_res2"),
    ("bootstrap_res2", "calendar_forecast_res2"),
])
def test_steps_run_after_what_they_read(before, after):
    assert order(before) < order(after)


def test_list_prints_every_step(capsys, monkeypatch):
    monkeypatch.setattr(pipeline.os, "chdir", lambda _: None)
    pipeline.main(["list"])
    out = capsys.readouterr().out
    assert all(s.name in out for s in ALL)


@pytest.mark.parametrize("argv", [["no_such_step"], ["all", "--from", "no_such_step"],
                                  ["all", "--box", "1", "2", "3", "4"], ["all", "--data", "x.nc"]])
def test_bad_commands_stop_before_doing_anything(argv, monkeypatch):
    monkeypatch.setattr(pipeline.os, "chdir", lambda _: None)
    monkeypatch.setattr(pipeline, "ready", lambda step: pytest.fail("a step started"))
    with pytest.raises(SystemExit):
        pipeline.main(argv)


def test_one_step_gets_its_own_options_then_the_callers(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.importlib, "import_module",
                        lambda name: type("M", (), {"main": staticmethod(lambda argv: seen.append((name, argv)))}))
    pipeline.main(["held_out_tau2", "--data", "x.nc"])
    assert seen == [("agulhas.validation.held_out", ["--res", "1", "--tau", "2", "--data", "x.nc"])]
