"""Tests for the classical iris-code pipeline on synthetic eyes."""
import os
import sys

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import iris_codes as icd  # noqa: E402


def synthetic_eye(texture_seed: int, rotate_deg: float = 0.0, noise: float = 2.0, rp: float = 20, ri: float = 60,
                  cx: float = 160, cy: float = 120) -> np.ndarray:
    """Sclera 200, iris with an angular texture (mean 100), pupil 20, on a 320 x 240 grid."""
    h, w = 240, 320
    y, x = np.indices((h, w)).astype(float)
    r = np.hypot(x - cx, y - cy)
    th = np.arctan2(-(y - cy), x - cx) + np.radians(rotate_deg)
    rng = np.random.default_rng(texture_seed)
    k = rng.integers(3, 20, 8)
    ph = rng.uniform(0, 2 * np.pi, 8)
    tex = sum(np.cos(kk * th + p + 0.05 * kk * r) for kk, p in zip(k, ph)) * 4  # about +-30 grey levels
    img = np.where(r < rp, 20.0, np.where(r < ri, 100.0 + tex, 200.0))
    return img + np.random.default_rng(texture_seed + 1000).normal(0, noise, img.shape)


def test_segmentation_finds_pupil_and_limbus():
    s = icd.segment(ndimage.median_filter(synthetic_eye(1), size=5))
    (px, py, rp), (ix, iy, ri) = s["pupil"], s["iris"]
    assert abs(px - 160) <= 3 and abs(py - 120) <= 3 and abs(rp - 20) <= 3
    assert abs(ri - 60) <= 4 and abs(ix - 160) <= 4 and abs(iy - 120) <= 4


def _code(img):
    img = ndimage.median_filter(img, size=5)
    return icd.log_gabor_code(icd.normalise(img, icd.segment(img)))


def test_hamming_same_eye_low_other_eye_near_half_and_rotation_tolerated():
    a = _code(synthetic_eye(1))
    b = _code(synthetic_eye(1, rotate_deg=5, noise=4))  # same iris, rotated and noisier
    c = _code(synthetic_eye(2))
    C = np.stack([a[0], b[0], c[0]])
    M = np.stack([a[1], b[1], c[1]])
    D = icd.hamming_matrix(C, M)
    assert D[0, 0] == pytest.approx(0.0)
    assert D[0, 1] < 0.25
    assert 0.35 < D[0, 2] < 0.65


def test_failed_images_give_nan_distances():
    a = _code(synthetic_eye(1))
    C = np.stack([a[0], a[0]])
    M = np.stack([a[1], np.zeros_like(a[1])])
    assert np.isnan(icd.hamming_matrix(C, M)[0, 1])


def test_flip_of_image_maps_eyelid_sectors_onto_themselves():
    th = np.degrees(np.linspace(0, 2 * np.pi, icd.ANGULAR, endpoint=False))
    masked = np.zeros_like(th, bool)
    for a, b in icd.EYELID_SECTORS_DEG:
        masked |= (th >= a) & (th <= b)
    mirrored = (180 - th) % 360
    for a, b in icd.EYELID_SECTORS_DEG:
        inside = (th >= a) & (th <= b)
        assert (((mirrored[inside] >= a) & (mirrored[inside] <= b))).mean() > 0.9


def test_load_grey_resizes_to_work_width(tmp_path):
    Image.new("RGB", (640, 480), (90, 90, 90)).save(tmp_path / "e.png")
    assert icd.load_grey(str(tmp_path / "e.png")).shape == (240, icd.WORK_WIDTH)
