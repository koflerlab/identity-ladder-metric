import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ladder2d as L  # noqa: E402
import ladder2d_brats as B  # noqa: E402


def toy_images(tmp_path):
    """Target pattern t_a; t_b is t_a plus noise; t_m is t_a mirrored; strangers are independent noise."""
    rng = np.random.default_rng(5)
    base = rng.random((224, 224))
    arrays = {"t_a": base, "t_b": base + 0.03 * rng.random((224, 224)), "t_m": base[:, ::-1]}
    arrays.update({f"s{i}": rng.random((224, 224)) for i in range(30)})
    paths = {}
    for k, a in arrays.items():
        p = tmp_path / f"{k}.png"
        Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(p)
        paths[k] = str(p)
    return L.Images(paths, rgb=False, v=False)


def pair(rung, cand):
    return {"rung": rung, "person": "t", "anchor": "t_a", "candidate": cand,
            "mean_members": [f"s{i}" for i in range(10)], "reference": [f"s{i}" for i in range(10, 30)]}


def test_mask_is_fixed_centred_square():
    assert B.MASK.sum() == 64 * 64 and B.MASK[80, 80] and B.MASK[143, 143] and not B.MASK[79, 80] and not B.MASK[144, 143]


def test_own_repeat_beats_every_stranger_on_all_nine_metrics(tmp_path):
    r = B.score_pair(pair("same_eye_same_sitting", "t_b"), toy_images(tmp_path))
    assert set(r["raw"]["own"]) == set(B.METRICS) and len(r["strangers"]) == 20
    assert all(r["c"]["own"][m] == 1.0 for m in B.METRICS)


def test_fellow_eye_is_flipped_before_filling(tmp_path):
    r = B.score_pair(pair("fellow_eye_same_sitting", "t_m"), toy_images(tmp_path))
    assert r["raw"]["own"]["mse"] < 1e-4 and r["c"]["own"]["ssim"] == 1.0  # the mirror image restored by the flip


def test_summary_and_table_shape(tmp_path):
    imgs = toy_images(tmp_path)
    per = [{"dataset": "UPOL", "rung": "same_eye_same_sitting", "person": f"t{i}", **B.score_pair(pair("same_eye_same_sitting", "t_b"), imgs)}
           for i in range(3)]
    s = B.summarise(per)
    assert set(s["UPOL"]) == {"same-sitting repeat", "population mean", "stranger (matched)"}
    t = B.table(s)
    assert len(t) == len(B.ROW_ORDER) * 9 and "QTIM MRI (from its own run) median [IQR]" in t.columns
    assert (t["QTIM MRI (from its own run) median [IQR]"] == "").all()
