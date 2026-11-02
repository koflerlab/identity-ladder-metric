import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ladder2d as L  # noqa: E402


def test_guard_refuses_a_changed_input(tmp_path, monkeypatch):
    f = tmp_path / "x.txt"
    f.write_text("a")
    monkeypatch.setattr(L, "EXPECTED", {"x": (str(f), L.sha256(str(f)))})
    assert L.guard() == {"x": L.sha256(str(f))}
    f.write_text("b")
    with pytest.raises(SystemExit):
        L.guard()


def test_load_base_crops_square_and_scales(tmp_path):
    a = np.zeros((100, 300, 3), np.uint8)
    a[:, 100:200] = 255  # the centre square is white, the sides black
    p = tmp_path / "x.png"
    Image.fromarray(a).save(p)
    b = L.load_base(str(p), 32, rgb=True)
    assert b.shape == (32, 32, 3) and np.allclose(b, 1.0)
    g = L.load_base(str(p), 32, rgb=False)
    assert g.shape == (32, 32) and np.allclose(g, 1.0)


def test_pixel_and_nuisance_representations():
    rng = np.random.default_rng(0)
    b = rng.random((224, 224)).astype(np.float32)
    p = L.zscore(L.rep_pixels_raw(b))
    assert p.shape == (64 * 64,) and abs(p.mean()) < 1e-9 and abs(p.std() - 1) < 1e-9
    # z-scoring makes P blind to a global gain and offset; N is not
    assert np.allclose(p, L.zscore(L.rep_pixels_raw(0.5 * b + 0.2)), atol=1e-5)  # float32 resize
    assert not np.allclose(L.rep_nuisance(b), L.rep_nuisance(0.5 * b + 0.2))
    assert L.cosine_distance(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(0)


def test_vessel_masks_find_dark_lines_and_topology_counts():
    img = np.full((512, 512, 3), 0.6, np.float32)
    img[:, 250:254, 1] = 0.2  # one vertical vessel
    img[200:204, 100:400, 1] = 0.2  # crossing horizontal vessel
    masks = L.vessel_masks(img, fractions=(0.02,))
    m = masks[0]
    assert m[:, 248:256].mean() > m.mean() * 5  # vessel pixels are over-represented on the line
    assert L.topology(m)[0] >= 1
    assert L.dice_distance(m, m) == 0 and L.dice_distance(m, ~m) == 1


def test_topology_of_a_plus_sign():
    m = np.zeros((41, 41), bool)
    m[20, 5:36] = True
    m[5:36, 20] = True
    comp, branches = L.topology(m)
    assert comp == 1 and branches >= 1


def test_bootstrap_mean_and_difference():
    people = np.array(["a", "b", "c", "d"] * 25)
    idx, counts = L.person_weights(np.unique(people), n_boot=500)
    pid = {p: i for i, p in enumerate(np.unique(people))}
    v = np.where(people == "a", 1.0, 0.0)
    r = L.boot_mean(v, np.array([pid[p] for p in people]), counts)
    assert r["mean"] == pytest.approx(0.25) and r["ci95"][0] <= 0.25 <= r["ci95"][1] and r["ci95"][1] > r["ci95"][0]
    d = L.boot_diff(np.ones(4), np.arange(4), np.zeros(4), np.arange(4), counts)
    assert d["diff"] == 1 and d["ci95"] == [1.0, 1.0]


def test_mean_labels():
    assert L.label_mean([0.01, 0.2]) == "rewards the mean"
    assert L.label_mean([-0.2, -0.01]) == "places the mean below the person"
    assert L.label_mean([-0.1, 0.1]) == "not separated"


class FakeDino:
    def __call__(self, bases):
        return np.stack([np.r_[np.asarray(b, float).mean(), np.asarray(b, float).std(), 1.0] for b in bases])


def test_score_dataset_end_to_end_on_toy_images(tmp_path):
    """Person t's two images share a pattern; strangers are independent noise; the mean is smooth."""
    rng = np.random.default_rng(3)
    base = rng.random((64, 64))
    paths = {}
    for name, arr in [("t_a", base), ("t_b", base + 0.05 * rng.random((64, 64)))] + \
            [(f"s{i}", rng.random((64, 64))) for i in range(40)]:
        p = tmp_path / f"{name}.png"
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(p)
        paths[name] = str(p)
    imgs = L.Images(paths, rgb=False, v=False)
    pair = {"rung": "same_eye_same_sitting", "person": "t", "anchor": "t_a", "candidate": "t_b",
            "mean_members": [f"s{i}" for i in range(8)], "reference": [f"s{i}" for i in range(8, 40)]}
    df = L.score_dataset("TOY", [pair], imgs, FakeDino(), None, np.random.default_rng(0), None)
    pp = df[(df.rep == "P") & (df.variant == "primary")].set_index("lam")
    assert pp.loc[1.0, "c"] == 1.0  # own image closer than every stranger
    assert set(pp.index) == set(L.LAMBDAS) and (pp.k == 32).all()
    diag = df[df.variant == "diagnostic"].set_index("lam")
    assert diag.loc[1.0, "residual_ratio"] == pytest.approx(1) and diag.loc[0.0, "residual_ratio"] == pytest.approx(0)
    assert diag.loc[0.5, "target_projection"] == pytest.approx(0.5 * diag.loc[1.0, "target_projection"])
    agg = L.aggregate(df)
    assert agg["TOY"]["rungs"]["P | primary"]["same_eye_same_sitting"]["lambda=1"]["mean"] == 1.0


def test_interpret_labels_only_the_time_and_camera_contrasts():
    ci = {"mean": 0.9, "ci95": [0.8, 0.95]}
    rungs = {f"{r} | primary": {"same_camera_<1y": {"lambda=1": ci}} for r in ("P", "D", "N")}
    contrasts = {"same_camera_<1y": {f"{r} minus N (own)": ci for r in ("P", "D")} |
                 {f"{r} mean minus own": {"mean": -0.1, "ci95": [-0.2, -0.05]} for r in ("P", "D")}}
    diffs = {"same_camera_<1y minus other_camera | P": {"diff": 0.1, "ci95": [0.02, 0.2]},
             "fellow_eye_same_visit minus same_camera_<1y | P": {"diff": -0.2, "ci95": [-0.3, -0.1]}}
    lab = L.interpret({"GRAPE": {"rungs": rungs, "contrasts": contrasts, "rung_differences": diffs}})["GRAPE"]
    assert lab["same_camera_<1y minus other_camera | P"] == "lowers the score"
    assert "fellow_eye_same_visit minus same_camera_<1y | P" not in lab
    assert lab["same_camera_<1y | P identity persists"] and lab["same_camera_<1y | nuisance-driven"]
    assert lab["same_camera_<1y | P mean"] == "places the mean below the person"
