"""Synthetic pupil geometry, calibration direction and omitted donor controls."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pupil_ladder as pl
import label_examples as le
import label_ladder as ll


def test_identical_pupil_has_zero_boundary_and_area_error():
    eye = pl.people(1, np.random.default_rng(0))[0][0]
    image = pl.render(eye, np.random.default_rng(1))
    v = pl.shape_scores(image, image)
    assert v == {"pupil_dice": 1., "pupil_area_error": 0., "pupil_boundary_distance": 0., "pupil_components_error": 0.}


def test_dilation_changes_shape_but_not_component_count():
    eye = pl.people(1, np.random.default_rng(0))[0][0]
    baseline = pl.render(eye, np.random.default_rng(1))
    dilated = pl.render(eye, np.random.default_rng(1), 1.3)
    a, b = pl.segment(baseline), pl.segment(dilated)
    assert b.sum() > 1.5 * a.sum()
    scores = pl.shape_scores(dilated, baseline)
    assert scores["pupil_components_error"] == 0
    assert scores["pupil_boundary_distance"] > 2
    assert 0 < scores["pupil_dice"] < .85


def test_tied_topology_is_not_perfect_identity():
    assert pl.c_score(0, [0] * 16, "pupil_components_error") == .5
    assert pl.c_score(.9, [.2, .5, .8, .9], "pupil_dice") == .875
    assert pl.c_score(.9, [.2, .5, .8, .9], "rmse") == .125


def test_other_hemisphere_figure_uses_other_side_not_reflected_target():
    imgs = le.example(.25)
    acq, _, _ = ll.people(ll.BASE, 60, np.random.default_rng(101), prior_scale=.25)
    assert len(imgs) == 7
    np.testing.assert_array_equal(imgs[3][ll.MASK], acq[(0, 0, "R", "S0", 0)][ll.MASK])
    np.testing.assert_array_equal(imgs[3][~ll.MASK], imgs[0][~ll.MASK])
    assert not np.array_equal(imgs[3][ll.MASK], imgs[0][:, ::-1][ll.MASK])
