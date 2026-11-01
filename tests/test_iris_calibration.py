"""Tests for the iris pair builder and label parsing on synthetic images."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import iris_calibration as ic  # noqa: E402


def test_pair_frame_rungs_and_flip():
    t = pd.DataFrame({"person": ["1", "1", "1", "2"], "eye": ["L", "L", "R", "L"], "sensor": ["NIR", "VIS", "NIR", "NIR"]})
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    Xf = X[:, ::-1].copy()
    p = ic.pair_frame(t, X, Xf)
    rungs = dict(zip(zip(*np.triu_indices(4, 1)), p.rung))
    assert rungs[(0, 1)] == "same_eye_other_sensor"
    assert rungs[(0, 2)] == "fellow_eye_same_sensor"
    assert rungs[(1, 2)] == "fellow_eye_other_sensor"
    assert rungs[(0, 3)] == "other_person"
    assert p.distance.iloc[1] == pytest.approx(0.0)  # fellow eye flipped matches exactly


def test_utiris_table_drops_folder_filename_eye_conflicts(tmp_path):
    nir = tmp_path / "Infrared Images" / "001" / "001_L"
    nir.mkdir(parents=True)
    for name in ("Img_001_L_1_1.bmp", "Img_001_R_2_1.bmp"):
        Image.new("L", (8, 8), 100).save(nir / name)
    rgb = tmp_path / "RGB Images" / "001"
    rgb.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(rgb / "IMG_001_R_1.JPG")
    t, conflicts = ic.utiris_table(str(tmp_path))
    assert conflicts == 1
    assert sorted(zip(t.sensor, t.eye)) == [("NIR", "L"), ("VIS", "R")]


def test_upol_table_parses_names(tmp_path):
    for name in ("001L_1.png", "001R_3.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    t = ic.upol_table(str(tmp_path))
    assert list(zip(t.person, t.eye)) == [("001", "L"), ("001", "R")]


def test_features_flip_is_mirror(tmp_path):
    a = np.zeros((16, 16), dtype=np.uint8)
    a[:, :4] = 255
    Image.fromarray(a).save(tmp_path / "a.png")
    f, ff = ic.features([str(tmp_path / "a.png")]), ic.features([str(tmp_path / "a.png")], flip=True)
    np.testing.assert_allclose(f.reshape(16, 16)[:, ::-1], ff.reshape(16, 16))


def test_zscore_rejects_constant_image():
    with pytest.raises(FloatingPointError):
        ic.zscore_rows(np.ones((2, 4)))
