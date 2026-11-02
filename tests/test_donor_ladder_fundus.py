"""Logic checks of the fundus donor ladder on synthetic inputs (no real images)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_fundus as F


def test_units_drop_side_conflicts_and_keep_earliest_visit():
    t = pd.DataFrame({"person": ["a", "a", "a", "b"], "side": ["R", "R", "L", "R"], "visit": [2, 1, 1, 1],
                      "time": [1., 0., 0., 0.], "path": list("wxyz"), "age": 50, "sex": 0, "device": "d"})
    ok = np.array([True, True, True, True])
    disc_right = np.array([True, True, False, False])  # b: right eye with the disc on the left -> conflict
    u = F.units(t, ok, disc_right)
    assert set(zip(u.person, u.side)) == {("a", "R"), ("a", "L")}
    assert u[(u.person == "a") & (u.side == "R")].visit.item() == 1


def test_patch_positions_respect_support():
    fov = np.zeros((20, F.G, F.G), bool)
    fov[:, 40:200, 40:200] = True
    pos, n_cand = F.patch_positions(fov, np.random.default_rng(0))
    assert n_cand > 0 and len(pos) == min(F.N_PATCH, n_cand)
    for y, x, s in pos:
        assert y - F.RING >= 40 and x - F.RING >= 40 and y + s + F.RING <= 200 and x + s + F.RING <= 200


def test_ring_mask_excludes_patch():
    m = F.ring_mask(10)
    assert m.shape == (10 + 2 * F.RING,) * 2 and m.sum() == m.size - 100


def test_summary_recovers_r2_from_theory_rows():
    rows = []
    for person in range(30):
        for c, a in (("population", 1.0), ("cohort", 0.7)):
            for k in F.KS:
                rows.append((person, 0, c, str(k), a * (1 + 1 / k), a / k))
            rows.append((person, 0, c, "all", a, 0.0))
        rows.append((person, 0, "fellow_eye", "1", 2 * (1 - 0.4), 0.5))
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread"])
    r["person"] = r.target.astype(str)
    r["region"] = "random_patches"
    s = F.summarize(r)["regions"]["random_patches"]
    assert np.isclose(s["V_mean_error_of_population_mean"], 1.0)
    assert np.isclose(s["cohorts"]["cohort"]["fit"]["R2"], 0.3)
    assert np.isclose(s["cohorts"]["fellow_eye"]["r_implied"], 0.4)
    assert np.isclose(s["k_eff_population"]["8"], 8)
