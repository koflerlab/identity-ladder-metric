"""Registration methods on synthetic images with known geometry (no real data)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reg_methods as M  # noqa: E402


def fundus(disc=(380, 250), fovea=(230, 260), size=512, seed=0):
    """Round field of view, bright disc, dark fovea, a few vessels and smooth texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    f = np.hypot(xx - size / 2, yy - size / 2) < 0.46 * size
    tex = M.ndimage.gaussian_filter(rng.standard_normal((size, size)), 6) * 0.6
    g = 0.45 + 0.04 * tex + 0.5 * np.exp(-((xx - disc[0]) ** 2 + (yy - disc[1]) ** 2) / (2 * 18 ** 2)) \
        - 0.25 * np.exp(-((xx - fovea[0]) ** 2 + (yy - fovea[1]) ** 2) / (2 * 30 ** 2))
    for k in range(6):
        ang = k * np.pi / 3 + 0.3
        d = np.abs((xx - disc[0]) * np.sin(ang) - (yy - disc[1]) * np.cos(ang))
        along = (xx - disc[0]) * np.cos(ang) + (yy - disc[1]) * np.sin(ang)
        g -= 0.15 * ((d < 2.5) & (along > 0))
    rgb = np.stack([np.clip(g + 0.25, 0, 1), np.clip(g, 0, 1), np.clip(g * 0.5, 0, 1)], -1) * f[..., None]
    return rgb.astype(np.float32)


def test_similarity_from_two_is_exact():
    o = [[0, 0], [10, 0]]
    r = [[5, 5], [5, 25]]
    A = M.similarity_from_two(o, r)
    assert np.allclose(A(np.array(o, float)), r)
    s, rot = M.scale_rotation(A.M)
    assert s == pytest.approx(2.0) and rot == pytest.approx(90.0)


def test_affine_then_and_inverse():
    a = M.Affine([[1, 0, 3], [0, 1, -2], [0, 0, 1]])
    b = M.Affine([[0, -2, 0], [2, 0, 0], [0, 0, 1]])
    p = np.array([[1.0, 2.0], [-3.0, 4.0]])
    assert np.allclose(a.then(b)(p), b(a(p)))
    assert np.allclose(a.inverse()(a(p)), p)


def test_g1_places_disc_and_fovea_on_targets():
    r = M.g1(fundus())
    assert r.ok
    lm = M.R.fundus_landmarks(r.out)
    assert lm is not None
    assert np.linalg.norm(lm[0] - M.R.DISC_R) < 4 and np.linalg.norm(lm[1] - M.R.FOVEA_R) < 6


def test_g1_frame_recovers_known_motion():
    raw = fundus()
    S, _ = M.random_motion(np.random.default_rng(3), raw.shape[:2])
    a, b = M.g1(raw), M.g1(M.move(raw, S))
    assert a.ok and b.ok
    err = M.recovery_error(a.phi, b.phi, S, M.grid16(a.valid))
    assert np.median(err) < 4


def test_g0_does_not_recover_motion():
    raw = fundus()
    S, p = M.random_motion(np.random.default_rng(3), raw.shape[:2])
    a, b = M.g0(raw), M.g0(M.move(raw, S))
    assert np.median(M.recovery_error(a.phi, b.phi, S, M.grid16(a.valid))) > 5


def test_failure_is_not_the_unregistered_image():
    r = M.g1(np.zeros((512, 512, 3), np.float32))
    assert not r.ok and r.out is None and r.phi is None


def test_ecc_recovers_small_shift_after_frame():
    raw = fundus()
    a = M.g1(raw)
    S = M.Affine([[1, 0, 3.0], [0, 1, -2.0], [0, 0, 1]])
    moved = M.move(raw, S)
    b0 = M.Result(M.remap(moved, a.phi, (512, 512)), a.phi, None)  # same frame as the anchor: residual = S
    b0.valid = M.fov(b0.out)
    r = M.g3(a, b0, moved)
    assert r.ok
    err = M.recovery_error(a.phi, r.phi, S, M.grid16(a.valid))
    assert np.median(err) < 1.0


def test_g4_has_no_folding_on_recovered_pair():
    raw = fundus()
    a = M.g1(raw)
    r3 = M.g3(a, a, raw)
    r4 = M.g4(a, r3, raw)
    assert r4.ok and r4.info["jac_nonpos"] == 0


def test_g4_propagates_g3_failure():
    assert M.g4(M.g1(fundus()), M.failed("x"), fundus()).fail == "G3 failed"


def test_histogram_matching_hits_reference_quantiles():
    rng = np.random.default_rng(0)
    ref = M.quantile_reference(rng.normal(5, 2, 10000))
    v = M.match_to_reference(rng.uniform(0, 1, 5000), ref)
    assert v.mean() == pytest.approx(5, abs=0.1) and v.std() == pytest.approx(2, abs=0.1)


def eye(px=160, py=118, rp=18, ri=55, w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(xx - px, yy - py)
    tex = M.ndimage.gaussian_filter(rng.standard_normal((h, w)), 2)
    g = np.where(d < rp, 20, np.where(d < ri, 110 + 12 * tex, 200)).astype(float)
    g = M.ndimage.gaussian_filter(g, 1)
    return M.IrisRaw(g, g / 255, (w, h))


def test_i2_centres_the_pupil():
    r = M.i2(eye())
    assert r.ok
    c = (M.BASE - 1) / 2
    assert np.allclose(r.phi(np.array([[c, c]]))[0], [160, 118], atol=2)


def test_i0_maps_the_centre_crop():
    raw = eye(w=320, h=240)
    r = M.i0(raw)
    # the base crop covers the central 240 x 240 square of the original, i.e. working x 40..280
    lo, hi = r.phi(np.array([[0, 0], [M.BASE - 1, M.BASE - 1]]))
    assert lo[0] == pytest.approx(40, abs=1) and hi[0] == pytest.approx(279, abs=1)


def test_i4_finds_angular_shift():
    raw = eye()
    seg = {"pupil": (160, 118, 18), "iris": (160, 118, 55)}
    a = M._iris_result(raw, M.RubberSheet(seg["pupil"], seg["iris"]), M.POLAR)
    b = M.Result(np.roll(a.out, 7, axis=1), a.phi, a.valid)
    assert M.i4(a, b).info["shift"] == 7


def test_reflection_reversal_is_exact():
    raw = fundus()
    assert np.array_equal(raw[:, ::-1][:, ::-1], raw)


def test_grid16_respects_mask():
    v = np.zeros((224, 224), bool)
    v[:, :112] = True
    pts = M.grid16(v)
    assert len(pts) == 16 * 8 and (pts[:, 0] < 112).all()
