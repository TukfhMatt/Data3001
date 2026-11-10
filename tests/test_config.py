import numpy as np
import pytest

from agulhas.config import (BOX, box_tag, grid_shape, grid_tag, operator_path, parse_args, place_label,
                            release_points_path, setting_tag, wind_paths)


def test_grid_shape_counts_a_partial_last_cell():
    assert grid_shape(BOX, 1.0) == (45, 30)
    assert grid_shape(BOX, 2.0) == (23, 15)
    assert grid_shape(BOX, 0.5) == (90, 60)


def test_tags_name_every_setting():
    assert box_tag(BOX) == "10E-55E_45S-15S"
    assert box_tag((-80, -40, 25, 45)) == "80W-40W_25N-45N"
    assert setting_tag(1.0, 3.5, BOX) == "1deg_3p5d_10E-55E_45S-15S"
    assert setting_tag([0.5, 1, 2], 3.5, BOX) == "0p5-1-2deg_3p5d_10E-55E_45S-15S"
    assert grid_tag([0.5, 1, 2], BOX) == "0p5-1-2deg_10E-55E_45S-15S"
    assert operator_path("drogued", 1.0, 3.5, BOX) == "data/P_drogued_1deg_3p5d_10E-55E_45S-15S.npz"


def test_release_points_default_to_the_reference_setting():
    assert release_points_path(BOX) == "data/release_candidates_1deg_3p5d_10E-55E_45S-15S.csv"
    assert release_points_path(BOX, 2.0, 5.0) == "data/release_candidates_2deg_5d_10E-55E_45S-15S.csv"


def test_each_subset_keeps_its_own_winds():
    assert wind_paths("data/agulhas_gdp1h_subset.nc") == ("data/era5", "data/drifter_wind10m.nc")
    assert wind_paths("data/g.nc") == ("data/era5_g", "data/drifter_wind10m_g.nc")


def test_place_label_handles_both_hemispheres():
    assert place_label(-70.5, 35.5) == "70.5°W 35.5°N"
    assert place_label(31.5, -30.5) == "31.5°E 30.5°S"


def test_parse_args_reads_the_list_it_is_given(tmp_path):
    missing = str(tmp_path / "none.nc")
    args = parse_args("Title\nBody", grid=True, lag=True, argv=["--res", "2", "--tau", "5", "--data", missing])
    assert (args.res, args.tau, args.data) == (2.0, 5.0, missing)
    assert args.box == list(BOX)


def test_parse_args_rejects_options_a_step_does_not_take(tmp_path):
    with pytest.raises(SystemExit):
        parse_args("Title", argv=["--res", "2", "--data", str(tmp_path / "none.nc")])


def test_oil_mass_left_loses_the_evaporating_share_in_days_and_the_residue_over_months():
    from agulhas.config import OIL_RESIDUE_DAYS, oil_mass_left

    assert oil_mass_left(0) == 1.0
    assert abs(oil_mass_left(14) - 0.85 * np.exp(-14 / 60)) < 1e-3     # evaporation done, residue still mostly there
    assert abs(oil_mass_left(60) - 0.85 * np.exp(-1)) < 1e-12          # 60-day e-folding of the residue
    assert oil_mass_left(365) < 0.01
    assert oil_mass_left(90, OIL_RESIDUE_DAYS["mass_30d"]) < oil_mass_left(90) < oil_mass_left(90, OIL_RESIDUE_DAYS["mass_120d"])
    assert (np.diff(oil_mass_left(np.arange(0, 400, 7))) <= 0).all()
