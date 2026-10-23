"""Tests for the re-identification reimplementation: safe loading and the original protocol on toy data."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reid  # noqa: E402


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo should-not-run",))


def test_restricted_unpickler_blocks_non_numpy_globals(tmp_path):
    (tmp_path / "f.pkl").write_bytes(pickle.dumps([{"path": "a.jpg", "features": _Evil()}]))
    with pytest.raises(pickle.UnpicklingError):
        reid.load_features_safe(str(tmp_path))


def test_restricted_unpickler_loads_numpy_arrays(tmp_path):
    recs = [{"path": "/x/a.jpg", "features": np.arange(3.0)}, {"path": "b.jpg", "features": np.ones(3)}]
    (tmp_path / "f.pkl").write_bytes(pickle.dumps(recs))
    out = reid.load_features_safe(str(tmp_path))
    assert set(out) == {"a.jpg", "b.jpg"} and np.allclose(out["a.jpg"], [0, 1, 2])


def _toy():
    # patient 1: two visits, both eyes at visit 1; patient 2: two visits; patient 3: one visit only (dropped)
    rows = [("1", "OD", "1", "p1v1od.jpg"), ("1", "OS", "1", "p1v1os.jpg"), ("1", "OD", "2", "p1v2od.jpg"),
            ("2", "OD", "1", "p2v1.jpg"), ("2", "OD", "2", "p2v2.jpg"), ("3", "OD", "1", "p3v1.jpg")]
    t = pd.DataFrame(rows, columns=["patient", "eye", "visit", "filename"]).assign(device="cam")
    f = {"p1v1od.jpg": [0.0], "p1v1os.jpg": [0.1], "p1v2od.jpg": [5.0],
         "p2v1.jpg": [10.0], "p2v2.jpg": [10.2], "p3v1.jpg": [0.05]}
    return t, {k: np.array(v) for k, v in f.items()}


def test_same_visit_exclusion_and_single_visit_patients():
    t, f = _toy()
    res = reid.evaluate(t, f, exclude_same_visit=True)
    m = res.matches.set_index("query")
    # p3 has one visit and is not in the gallery, so p1v1od cannot match it; its fellow eye (same visit) is excluded
    assert m.at["p1v1od.jpg", "result"] == "p1v2od.jpg"
    assert res.metrics["n_images"] == 5 and res.metrics["n_patients"] == 2
    # all five queries find their own patient at another visit
    assert res.metrics["image_R@1"] == 1.0 and res.metrics["patient_R@1"] == 1.0


def test_without_exclusion_fellow_eye_wins():
    t, f = _toy()
    res = reid.evaluate(t, f, exclude_same_visit=False)
    assert res.matches.set_index("query").at["p1v1od.jpg", "result"] == "p1v1os.jpg"


def test_pair_rungs():
    t, f = _toy()
    p = reid.ladder_pairs(t, f)
    names = t["filename"].tolist()
    lab = {(names[i], names[j]): r for i, j, r in zip(p["i"], p["j"], p["rung"])}
    assert lab[("p1v1od.jpg", "p1v1os.jpg")] == "fellow_eye_same_visit"
    assert lab[("p1v1od.jpg", "p1v2od.jpg")] == "same_eye_other_visit"
    assert lab[("p1v1os.jpg", "p1v2od.jpg")] == "fellow_eye_other_visit"
    assert lab[("p1v1od.jpg", "p2v1.jpg")] == "other_patient"


def test_decidability_known_value():
    x, y = np.array([0.0, 2.0]), np.array([4.0, 6.0])
    assert reid.decidability(x, y) == pytest.approx(4 / np.sqrt(2))
