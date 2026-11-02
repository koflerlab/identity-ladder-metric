"""Logic checks of the donor-ladder robustness runner on synthetic inputs (no real images or scans)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_fundus as F
import donor_ladder_robustness as RB
import ladder2d_register as R


def test_exact_k1_is_the_mean_single_donor_error():
    rng = np.random.default_rng(0)
    pool, xi = rng.standard_normal((50, 30)), rng.standard_normal(30)
    mu, rows = pool.mean(0), []
    RB.exact_rows(0, 0, xi, pool, mu, rows)
    direct = np.mean([np.mean((xi - p) ** 2) for p in pool])
    assert np.isclose(rows[0][4], direct)
    assert rows[1][2] == "pool_size" and rows[1][4] == 50


def test_fundus_positions_follow_box_parameters():
    fov = np.zeros((10, F.G, F.G), bool)
    fov[:, 40:200, 40:200] = True
    for patch in (32, 64):
        pos, n = RB.fundus_positions(fov, np.random.default_rng(1), patch, 16, 0.9)
        assert n > 0 and all(s == patch and y - 16 >= 40 and y + s + 16 <= 200 for y, x, s in pos)
    small = np.zeros_like(fov)
    small[:, 40:120, 40:120] = True  # 80 px of support: no 96 px window
    assert RB.fundus_positions(small, np.random.default_rng(1), 64, 16, 0.9) == ([], 0)


def test_disc_frame_puts_the_disc_on_the_canonical_position():
    yy, xx = np.mgrid[:512, :512]
    img = np.zeros((512, 512, 3), np.float32)
    fov = np.hypot(xx - 260, yy - 250) < 230
    img[fov] = 0.4
    disc, fovea = (300, 240), (160, 250)  # disc right of the fovea
    img[..., 0] += 0.5 * np.exp(-np.hypot(xx - disc[0], yy - disc[1]) ** 2 / 400) * fov
    img[..., 1] += 0.5 * np.exp(-np.hypot(xx - disc[0], yy - disc[1]) ** 2 / 400) * fov
    img[..., 1] -= 0.3 * np.exp(-np.hypot(xx - fovea[0], yy - fovea[1]) ** 2 / 400) * fov
    lm = R.fundus_landmarks(img)
    assert lm is not None and np.allclose(lm[0], disc, atol=3)
    r = np.sqrt(fov.sum() / np.pi)
    out, disc_right = RB.canonical(img, "disc", r_target=r)
    assert disc_right
    bright = out[..., 0] + out[..., 1]
    y, x = np.unravel_index(np.argmax(bright), bright.shape)
    assert abs(x - R.DISC_R[0]) <= 3 and abs(y - R.DISC_R[1]) <= 3


def test_headline_finite_pool_and_knn_shape():
    reg = {"cohorts": {
        "population": {"1": {"err_over_V": 2.0, "ci95": [1.9, 2.1]}, "fit": {"R2": 0.0, "rms_residual": 0.01}},
        "knn": {"1": {"err_over_V": 1.2}, "8": {"err_over_V": 0.7}, "64": {"err_over_V": 0.9}},
        "sibling": {"1": {"err_over_V": 1.8, "ci95": [1.6, 2.0], "n_targets": 40}}},
        "k_eff_population": {"1": 1.0, "32": 32 * 99 / 68, "64": 500.0}}
    h = RB.headline(reg, n_pool=100.0)
    assert np.isclose(h["keff_over_pred_min"], 1.0) and np.isclose(h["keff_over_pred_max"], 1.0)  # k = 64 > 32 skipped
    assert h["knn_best_k"] == 8 and h["knn_u_shape"]
    assert np.isclose(h["r_sibling"], 0.1) and np.allclose(h["r_sibling_ci"], [0.0, 0.2])


def test_verdict_rules_hold_sensitive_break():
    tab = pd.DataFrame({"dataset": "D", "variant": ["a", "b", "c"], "family": "seed", "region": "random_patches",
                        "metric": "stranger_k1", "value": [1.95, 2.05, 2.0]})
    assert RB.verdicts(tab)[0]["verdict"] == "holds"
    tab.loc[2, "value"] = 1.6
    assert RB.verdicts(tab)[0]["verdict"] == "sensitive"
    tab.loc[2, "value"] = 0.9
    v = RB.verdicts(tab)[0]
    assert v["verdict"] == "breaks" and v["out_of_band"] == ["D/c"]


def test_syn_subset_keeps_repeat_sessions_and_two_members_per_family():
    rows = []
    for f in range(30):
        for m in range(3):
            rows.append({"person": f"p{f}_{m}", "session": "ses-01", "family": f"f{f}", "path": "", "sex": "F", "age": 20.0})
    rows.append({"person": "p40_0", "session": "ses-01", "family": "f40", "path": "", "sex": "F", "age": 20.0})
    rows.append({"person": "p40_0", "session": "ses-02", "family": "f40", "path": "", "sex": "F", "age": 21.0})
    s = RB.syn_subset(pd.DataFrame(rows))
    assert set(s[s.person == "p40_0"].session) == {"ses-01", "ses-02"}
    fam = s[s.family != "f40"].groupby("family").size()
    assert len(fam) == RB.SYN_FAMILIES and (fam == 2).all()
