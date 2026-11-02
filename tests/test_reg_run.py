"""reg_run end to end on a synthetic fundus store (no real data): one pair through G0 and G1."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import reg_methods as M  # noqa: E402
import reg_run as RR  # noqa: E402
from test_reg_methods import fundus  # noqa: E402


@pytest.fixture()
def synthetic_store(tmp_path, monkeypatch):
    monkeypatch.setattr(RR, "CACHE_DIR", str(tmp_path))
    rng = np.random.default_rng(1)
    names = [f"img{i:02d}" for i in range(44)]
    raws = []
    for i, n in enumerate(names):
        disc = (380 + rng.integers(-15, 15), 250 + rng.integers(-15, 15))
        raws.append((fundus(disc=disc, fovea=(disc[0] - 150, disc[1] + 8), seed=i // 2) * 255).astype(np.uint8))
    d = RR.store("GRAPE")
    os.makedirs(d)
    np.save(os.path.join(d, "raw.npy"), np.stack(raws))
    np.save(os.path.join(d, "vessel.npy"), np.stack([np.packbits(r[..., 1] < 60) for r in raws]))
    with open(os.path.join(d, "index.json"), "w") as fh:
        json.dump({"names": names, "camera": {n: "cam" for n in names}}, fh)
    pair = {"anchor": names[0], "candidate": names[1], "rung": "same_camera_<1y", "person": "p0",
            "reference": names[2:36], "mean_members": names[36:44]}
    return pair


def test_pair_record_has_every_prespecified_field(synthetic_store):
    p = synthetic_store
    aux = {"href": [M.quantile_reference(np.linspace(0, 1, 100))] * 3, "cam": {"cam": {"mean": [0.5] * 3, "sd": [0.2] * 3}}}
    for m in ("G0", "G1"):
        RR.init_worker("GRAPE", [m], {**aux, "n_stats": [[0.3, 0.2], [0.05, 0.05]]})
        rec = RR.process((0, p))[0]
        assert rec["dropped"] == "" and not rec["fail"]["anchor"]
        assert set(rec["c"]) == {"own_P", "own_N", "mean_P", "mean_N", "null_P", "swap_P", "reflect_P"}
        assert all(0 <= v <= 1 for v in rec["c"].values() if np.isfinite(v))
        assert rec["c"]["reflect_P"] == pytest.approx(rec["c"]["own_P"])  # exact index reversal
        assert "dice" in rec["q1b"]["candidate"] and 0 <= rec["coverage"]["candidate"] <= 1
        assert rec["fail"]["strangers"][1] == 34


def test_q1a_image_frame_beats_none(synthetic_store):
    RR.init_worker("GRAPE", ["G0", "G1"], {"n_stats": [[0, 0], [1, 1]]})
    out = RR.q1a_image((0, "img00"))["q1a"]
    assert out["G1"] < 4 < out["G0"]
