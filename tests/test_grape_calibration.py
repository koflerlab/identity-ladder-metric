"""Tests for the random-pair calibrated score on toy distances and a toy image table."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import grape_calibration as gc  # noqa: E402


def test_calibrated_score_extremes_ties_and_monotonicity():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    c = gc.calibrated_score(np.array([0.0, 2.0, 5.0, 2.5]), ref)
    np.testing.assert_allclose(c, [1.0, 0.625, 0.0, 0.5])  # 2.0: two above, one tie
    d = np.linspace(0, 5, 50)
    assert np.all(np.diff(gc.calibrated_score(d, ref)) <= 0)


def test_mean_c_equals_auc_against_reference():
    rng = np.random.default_rng(0)
    x, ref = rng.normal(0, 1, 200), rng.normal(1, 1, 300)
    auc = (x[:, None] < ref[None, :]).mean()
    assert gc.calibrated_score(x, ref).mean() == pytest.approx(auc)


def test_pairs_use_flipped_distance_only_for_opposite_eyes():
    t = pd.DataFrame({"filename": ["a", "b", "c"], "patient": ["1", "1", "2"], "eye": ["OD", "OS", "OD"],
                      "visit": ["1", "1", "1"], "device": ["x", "x", "x"], "interval_years": [0, 0, 0]})
    f = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]), "c": np.array([1.0, 0.0])}
    flip = {k: v[::-1].copy() for k, v in f.items()}
    p = gc.pairs(t, f, flip).set_index("rung")
    assert p.loc["fellow_eye_same_visit", "distance"] == pytest.approx(0.0)  # b flipped equals a
    same_eye_other_patient = gc.pairs(t, f, flip)
    row = same_eye_other_patient[(same_eye_other_patient.rung == "other_patient") & ~same_eye_other_patient.opposite_eye]
    assert row.distance.iloc[0] == pytest.approx(0.0)  # a vs c: same eye, plain distance


def test_calibrate_refuses_thin_reference():
    p = pd.DataFrame({"rung": ["other_patient"] * 10, "same_camera": True, "device_pair": "x + x", "opposite_eye": False,
                      "patient": "", "gap_years": 0.0, "distance": np.arange(10.0)})
    with pytest.raises(ValueError):
        gc.calibrate(p)


def test_same_visit_is_defined_by_date_not_visit_number():
    # GRAPE counts Visit Number per eye: the fellow eye on the same date can carry another visit number
    t = pd.DataFrame({"filename": ["a", "b"], "patient": ["1", "1"], "eye": ["OD", "OS"], "visit": ["1", "2"],
                      "device": ["x", "x"], "interval_years": [0.5, 0.5]})
    f = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    assert gc.pairs(t, f).rung.iloc[0] == "fellow_eye_same_visit"


def test_reference_is_stratified_by_camera_pair():
    rows = []
    for i in range(150):
        rows.append({"rung": "other_patient", "same_camera": False, "device_pair": "A + B", "opposite_eye": False, "patient": "", "gap_years": 0.0, "distance": 10.0 + i * 0.01})
        rows.append({"rung": "other_patient", "same_camera": False, "device_pair": "A + C", "opposite_eye": False, "patient": "", "gap_years": 0.0, "distance": 5.0 + i * 0.01})
    rows.append({"rung": "same_eye_other_visit", "same_camera": False, "device_pair": "A + B", "opposite_eye": False, "patient": "1", "gap_years": 1.0, "distance": 7.0})
    p = gc.calibrate(pd.DataFrame(rows))
    assert p.c.iloc[-1] == 1.0  # closer than every A+B stranger, although farther than every A+C stranger
