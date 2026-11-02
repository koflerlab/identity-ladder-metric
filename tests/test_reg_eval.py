"""Decision rules of the registration protocol on synthetic per-pair records (no real data)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reg_eval as E  # noqa: E402


def records(n=120, null_shift=0.0, stranger_fail=0.0, own_q1a=1.0, stranger_q1a=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        own = float(rng.uniform(0.6, 1.0))
        rec = {"pair": i, "person": f"p{i // 2}", "rung": "fellow_eye_same_visit" if i % 4 == 3 else "same_camera_<1y",
               "dropped": "",
               "fail": {"anchor": False, "candidate": False,
                        "strangers": [int(rng.binomial(60, stranger_fail)), 60], "members": [0, 16]},
               "c": {"own_P": own, "own_N": 0.5, "mean_P": 0.4, "mean_N": 0.5,
                     "null_P": float(np.clip(rng.uniform(0, 1) + null_shift, 0, 1)),
                     "swap_P": own + float(rng.normal(0, 0.02)), "reflect_P": own},
               "q1a": {"candidate": own_q1a + float(rng.normal(0, 0.1)),
                       "strangers": stranger_q1a + float(rng.normal(0, 0.1)), "members": 1.0, "null": 1.0},
               "q1b": {r: {"dice": 0.5} for r in ("candidate", "strangers", "members", "null")},
               "q1c": {"candidate": 0.9}, "coverage": {"candidate": 1.0}, "coverage_lt95": {"candidate": 0.0},
               "q1d": {}, "q1e": {}}
        out.append(rec)
    return out


def verdict(recs, method="G1"):
    return E.eligibility(E.failure_rates(recs), E.fairness(recs, method), method)


def test_symmetric_method_is_eligible():
    v = verdict(records())
    assert v["eligible"], v["reasons"]


def test_candidate_slot_bias_fails_q3c():
    v = verdict(records(null_shift=0.3))
    assert not v["eligible"] and v["biased"]
    assert any("Q3c" in r for r in v["reasons"])


def test_stranger_failures_are_bias_and_failure():
    v = verdict(records(stranger_fail=0.3))
    assert v["biased"] and any("Q3a" in r for r in v["reasons"]) and any("failure strangers" in r for r in v["reasons"])


def test_geometric_accuracy_gap_is_bias():
    v = verdict(records(own_q1a=1.0, stranger_q1a=3.5))
    assert v["biased"] and any("Q3b q1a" in r for r in v["reasons"])


def test_vessel_dice_gap_is_not_a_criterion():
    recs = records()
    for r in recs:
        r["q1b"]["candidate"]["dice"] = 0.9
        r["q1b"]["strangers"]["dice"] = 0.3
    assert verdict(recs)["eligible"]


def test_choose_prefers_fewer_dof_when_tied():
    rows = [{"method": "G3", "eligible": True, "rank_key": "q1b_dice_candidate", "higher": True, "q1a": 0.5,
             "rank": {"mean": 0.62, "ci95": [0.58, 0.66]}},
            {"method": "G1", "eligible": True, "rank_key": "q1b_dice_candidate", "higher": True, "q1a": 0.9,
             "rank": {"mean": 0.60, "ci95": [0.56, 0.64]}},
            {"method": "G4", "eligible": False, "rank_key": "q1b_dice_candidate", "higher": True, "q1a": 0.1,
             "rank": {"mean": 0.9, "ci95": [0.88, 0.92]}}]
    c = E.choose(rows)
    assert c["recommended"] == "G1" and c["tied"] == ["G1", "G3"]


def test_choose_none_eligible():
    assert E.choose([])["recommended"] is None


def test_lower_is_better_for_distances():
    rows = [{"method": "I1", "eligible": True, "rank_key": "q1b_pupil_candidate", "higher": False, "q1a": 1,
             "rank": {"mean": 1.0, "ci95": [0.8, 1.2]}},
            {"method": "I0", "eligible": True, "rank_key": "q1b_pupil_candidate", "higher": False, "q1a": 9,
             "rank": {"mean": 9.0, "ci95": [8.0, 10.0]}}]
    assert E.choose(rows)["recommended"] == "I1"
