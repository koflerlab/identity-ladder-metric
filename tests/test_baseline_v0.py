"""Tests for the baseline v0 helpers on synthetic scores, distance matrices and strips."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import baseline_v0 as bv  # noqa: E402


def test_eer_separable_overlapping_and_symmetric():
    assert bv.eer([0.0, 1.0], [2.0, 3.0]) == 0.0
    x = np.arange(10.0)
    assert bv.eer(x, x) == pytest.approx(0.5, abs=0.06)
    rng = np.random.default_rng(0)
    g, i = rng.normal(0, 1, 4000), rng.normal(2, 1, 4000)
    assert bv.eer(g, i) == pytest.approx(0.159, abs=0.02)  # Phi(-1) for unit-variance normals 2 SD apart


def test_dprime_sign():
    assert bv.dprime_signed([0.0, 1.0], [2.0, 3.0]) > 0 > bv.dprime_signed([2.0, 3.0], [0.0, 1.0])


def test_discriminability_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 9
    X = rng.normal(size=(n, 3))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    item = np.array(list("aaabbbccc"))
    person = np.array(["p1"] * 3 + ["p1"] * 3 + ["p2"] * 3)  # items a and b are two eyes of one person
    vals = []
    for i in range(n):
        for j in range(n):
            if i != j and item[i] == item[j]:
                others = [k for k in range(n) if person[k] != person[i]]
                vals.append(np.mean([1.0 if D[i, k] > D[i, j] else 0.5 if D[i, k] == D[i, j] else 0.0 for k in others]))
    assert bv.discriminability(D, item, person) == pytest.approx(np.mean(vals))


def test_differential_identifiability_sign():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    D = np.abs(X - X.T)
    item = np.array(["a", "a", "b", "b"])
    assert bv.differential_identifiability(D, item, item) > 0
    assert bv.differential_identifiability(D, np.array(["a", "b", "a", "b"]), np.array(["a", "b", "a", "b"])) < 0


def test_two_anchor_s():
    assert bv.two_anchor_s(2.0, 2.0, 10.0) == pytest.approx(1.0)
    assert bv.two_anchor_s(10.0, 2.0, 10.0) == pytest.approx(0.0)
    assert bv.two_anchor_s(12.0, 2.0, 10.0) < 0


def test_icc1_extremes():
    groups = np.repeat(np.arange(20), 3)
    assert bv.icc1(np.repeat(np.arange(20.0), 3), groups) == pytest.approx(1.0)
    noise = np.random.default_rng(0).normal(size=60)
    assert abs(bv.icc1(noise, groups)) < 0.3


def test_rank1_with_gallery():
    D = np.array([[0, 1, 9, 9], [1, 0, 9, 9], [9, 9, 0, 1], [9, 9, 1, 0]], float)
    label = np.array(["a", "a", "b", "b"])
    r = bv.rank1(D, label, label, np.array([0, 2]), np.array([1, 3]))
    assert r["rank1_same_item"] == 1.0 and r["chance_item"] == 0.5


def test_rank1_person_chance_counts_the_fellow_eye():
    # two persons, both eyes enrolled: a probe's person owns 2 of 4 gallery entries, so chance_person = 0.5, chance_item = 0.25
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    item = np.array(["p1L", "p1L", "p1R", "p1R", "p2L", "p2L", "p2R", "p2R"])
    person = np.array(["p1"] * 4 + ["p2"] * 4)
    r = bv.rank1(D, item, person, np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7]))
    assert r["chance_item"] == 0.25 and r["chance_person"] == pytest.approx(0.5)


def test_gallery_draws_are_common_across_cells_and_averaged():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    D1 = np.linalg.norm(X[:, None] - X[None], axis=-1)
    D2 = D1 * 3.0 + 1.0  # another cell with the same usable images
    label = np.repeat(np.arange(4).astype(str), 3)
    U = rng.random((30, 12))
    g1, g2 = bv.gallery_draws(D1, label, U), bv.gallery_draws(D2, label, U)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(g1, g2))
    assert len({tuple(g) for g, _ in g1}) > 1  # the draws vary
    r = bv.rank1_averaged(D1, label, label, U)
    assert r["n_draws"] == 30 and 0 <= r["rank1_same_item_draw_interval95"][0] <= r["rank1_same_item_mean_over_draws"] <= r["rank1_same_item_draw_interval95"][1] <= 1


def test_panel_wrapper_on_a_2d_strip():
    rng = np.random.default_rng(0)
    target = rng.uniform(40, 200, (16, 128))
    mask = np.zeros((16, 128), bool)
    mask[4:12, 50:82] = True
    same = bv.panel(target.copy(), target, mask)
    assert same["mse"] == pytest.approx(0.0, abs=1e-12) and same["ssim"] == pytest.approx(1.0, abs=1e-6)
    noisy = target + mask * rng.normal(0, 20, target.shape)
    worse = bv.panel(noisy, target, mask)
    assert worse["rmse"] > 0 and worse["ssim"] < 1 and set(bv.PANEL) <= set(worse)


def test_place_recovers_rotation_and_intensity():
    rng = np.random.default_rng(0)
    target = rng.uniform(40, 200, (16, 128))
    mask = np.zeros((16, 128), bool)
    mask[4:12, 50:82] = True
    donor = 0.5 * np.roll(target, -3, axis=1) + 10  # rotated by three samples, darker
    fill = bv.place(target, donor, mask)
    np.testing.assert_allclose(fill, target, atol=1e-6)
    assert np.abs(bv.place(target, donor, mask, match=False)[mask] - target[mask]).mean() > 1


def test_donor_plan_uses_only_qc_passed_strips():
    t = pd.DataFrame({"person": ["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4, "eye": ["L", "L", "R", "R"] * 3})
    ok = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1], bool)
    ok_flip = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], bool)
    plan = bv.donor_plan(t, ok, ok_flip, np.random.default_rng(0), n_same=5, n_fellow=5, n_strangers=5)
    assert plan
    for i, kind, j in plan:
        assert ok[i] and i != j
        assert (ok_flip if kind.endswith("flipped") else ok)[j]
        same_person = t.person[i] == t.person[j]
        assert same_person == (kind in (bv.SAME, "fellow_eye_flipped"))
        assert (t.eye[i] == t.eye[j]) == (kind in (bv.SAME, "stranger_same_side"))


def test_ladder_c_and_win_rate_directions():
    R = pd.DataFrame({"target": 0, "mask": "A", "person": "p", "kind": [bv.SAME, "stranger_same_side", "stranger_same_side"],
                      "rmse": [0.1, 0.3, 0.5], "ssim": [0.9, 0.5, 0.95]})
    c = bv.ladder_c(R, "rmse", "stranger_same_side")
    assert c.iloc[0] == 1.0
    assert bv.ladder_c(R, "ssim", "stranger_same_side").iloc[0] == 0.5
    assert bv.win_rate(R, "rmse", bv.SAME, "stranger_same_side") == 1.0
    assert bv.win_rate(R, "rmse", "stranger_same_side", bv.SAME) == 0.0  # the direction is in the name: share_a_beats_b


# ---------- regression tests for the review of 2026-11-01 ----------

def _brute_eer(g, im):
    g, im = np.sort(np.asarray(g, float)), np.sort(np.asarray(im, float))
    thr = np.unique(np.concatenate([g, im]))
    fnmr = 1 - np.searchsorted(g, thr, side="right") / len(g)
    fmr = np.searchsorted(im, thr, side="right") / len(im)
    k = int(np.argmin(np.abs(fnmr - fmr)))
    return (fnmr[k] + fmr[k]) / 2


def test_weighted_helpers_reduce_to_unweighted_and_equal_expansion():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 37), rng.normal(1, 1, 52)
    assert bv.wauc(a, None, b, None) == pytest.approx(bv.auc_above(a, b))
    assert bv.weer(a, None, b, None) == pytest.approx(_brute_eer(a, b))
    assert bv.wdprime(a, None, b, None) == pytest.approx((b.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    for x in (a, a[:-1]):
        assert bv.wmedian(x, None) == pytest.approx(np.median(x))
    wa, wb = rng.integers(0, 4, len(a)).astype(float), rng.integers(0, 4, len(b)).astype(float)
    ea, eb = np.repeat(a, wa.astype(int)), np.repeat(b, wb.astype(int))
    assert bv.wmedian(a, wa) == pytest.approx(np.median(ea))
    assert bv.wauc(a, wa, b, wb) == pytest.approx(bv.auc_above(ea, eb))
    assert bv.weer(a, wa, b, wb) == pytest.approx(_brute_eer(ea, eb))
    assert bv.wdprime(a, wa, b, wb) == pytest.approx((eb.mean() - ea.mean()) / np.sqrt((ea.var(ddof=1) + eb.var(ddof=1)) / 2))


def test_calibrator_matches_calibrated_score_and_rebuilds_the_reference():
    rng = np.random.default_rng(0)
    q = rng.normal(0, 1, 50)
    r = rng.normal(2, 1, 300)
    pa, pb = rng.integers(1, 10, 300), rng.integers(10, 20, 300)
    cal = bv.Calibrator(q, np.zeros(50), r, np.zeros(300), pa, pb)
    np.testing.assert_allclose(cal.c(), bv.calibrated_score(q, r))
    w = np.ones(20)
    w[[3, 12]] = 0  # persons left out of a replicate leave the reference too
    keep = (w[pa] * w[pb]) > 0
    np.testing.assert_allclose(cal.c(w), bv.calibrated_score(q, r[keep]))
    w2 = np.ones(20)
    w2[5] = 3.0  # a person drawn three times counts three times in the reference
    np.testing.assert_allclose(cal.c(w2), bv.calibrated_score(q, np.repeat(r, (w2[pa] * w2[pb]).astype(int))))


def test_person_bootstrap_contains_point_and_reports_replicates():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 20)
    out = bv.person_bootstrap(lambda w: {"m": bv.wmean(vals, w)}, 20, 200, np.random.default_rng(1))["m"]
    assert out["ci95"][0] <= out["value"] <= out["ci95"][1] and out["n_finite_replicates"] == 200
    assert bv.pairs_needed([0.4, 0.6], 50, 400)["persons_needed"] == 200  # half-width 0.1 -> 4x the persons for 0.05


def test_t3b_c_scale_scores_l1_against_the_stated_l4_anchor():
    # a device shift k adds to cross-device pairs, genuine and stranger alike: on the residual scale it cancels, on the
    # stated anchor (same-device strangers) it shows, which is what the raw ratio measures
    rng = np.random.default_rng(0)
    k = 2.5
    l0, l1 = rng.normal(1, 0.3, 400), rng.normal(1, 0.3, 400) + k
    same_dev, cross_dev = rng.normal(3, 0.5, 4000), rng.normal(3, 0.5, 4000) + k
    pa, pb = np.arange(4000) % 50, 50 + np.arange(4000) % 50
    c0s, c1s = bv.c_against(l0, same_dev, pa, pb), bv.c_against(l1, same_dev, pa, pb)
    c1x, c0x = bv.c_against(l1, cross_dev, pa, pb), bv.c_against(l0, same_dev, pa, pb)
    same_anchor = bv.t3b_c_scale(c0s, None, c1s, None)
    residual = bv.t3b_c_scale(c0x, None, c1x, None)
    assert same_anchor > 1.0 and abs(residual) < 0.2  # the device change looks like a change of person only on the stated anchor


def test_icc_sensor_offset_is_not_within_eye_error():
    rng = np.random.default_rng(0)
    eyes = np.repeat(np.arange(40), 6)
    sensor = np.tile(np.array(["NIR"] * 3 + ["VIS"] * 3), 40)
    v = rng.normal(0, 1, 40)[eyes] + np.where(sensor == "VIS", 5.0, 0.0) + rng.normal(0, 0.3, len(eyes))
    assert bv.icc1(v, eyes) < 0.3
    assert bv.icc1_sensor_adjusted(v, eyes, sensor) > 0.8
    assert bv.icc1(v[sensor == "NIR"], eyes[sensor == "NIR"]) > 0.8


def test_stratified_eer_thresholds_within_strata():
    rng = np.random.default_rng(0)
    g = np.concatenate([rng.normal(1, 0.2, 300), rng.normal(10, 2, 300)])
    i = np.concatenate([rng.normal(2, 0.2, 3000), rng.normal(20, 2, 3000)])
    sg, si = np.repeat(["a", "b"], 300), np.repeat(["a", "b"], 3000)
    one = np.ones
    strat = bv.weer_stratified(g, one(600), sg, i, one(6000), si)
    assert strat < 0.05 < bv.weer(g, None, i, None)


def _strip_case(seed=0):
    rng = np.random.default_rng(seed)
    target = ndimage_smooth(rng.uniform(40, 200, (16, 128)))
    mask = np.zeros((16, 128), bool)
    mask[4:12, 50:82] = True
    return rng, target, mask


def ndimage_smooth(x):
    from scipy import ndimage
    return ndimage.gaussian_filter(x, 1.0, mode="wrap")


def test_template_fill_modes():
    rng, target, mask = _strip_case()
    template = 0.3 * target + 70 + rng.normal(0, 0.1, target.shape)  # low-variance, offset template
    raw = bv.place(target, template, mask, mode="raw", shifts=0)
    np.testing.assert_allclose(raw[mask], template[mask])
    mean = bv.place(target, template, mask, mode="mean", shifts=0)
    assert (mean[~mask] == target[~mask]).all() and np.std(mean[mask]) == pytest.approx(np.std(template[mask]))
    aff = bv.place(target, template, mask, mode="affine", shifts=0)
    a, b = np.polyfit(template[~mask], target[~mask], 1)
    np.testing.assert_allclose(aff[mask], a * template[mask] + b)
    sd = bv.place(target, template, mask, mode="sd", shifts=0)
    assert np.std(sd[mask]) > 2 * np.std(raw[mask])  # SD matching inflates the template: the H12 confound


def test_clipped_range_matches_the_package_normalisation():
    import torch
    from inpainting.challenge_metrics_2023 import _normalize_with_percentiles
    rng, target, mask = _strip_case(1)
    target[mask.nonzero()[0][0], mask.nonzero()[1][0]] = 5 * target.max()  # an in-mask peak above the clipping level
    T, M = torch.tensor(target, dtype=torch.float32)[None, None], torch.tensor(mask)[None, None]
    g = _normalize_with_percentiles((T * M).float(), normalization_tensor=T * ~M)[M]
    assert bv.clipped_range(target, mask) == pytest.approx(float(g.max() - g.min()), abs=1e-5)
    v_min, v_max = bv.scaling(target, mask)
    assert (target[mask].max() - target[mask].min()) / (v_max - v_min) > bv.clipped_range(target, mask) + 0.5  # the old proxy


def test_overshoot_has_a_fixed_pixel_count_and_never_equals_truth():
    rng = np.random.default_rng(0)
    tiny = np.zeros((16, 128), bool)
    tiny[7:9, 60:62] = True  # 4 pixels: a Binomial draw would often select none
    assert all(bv.overshoot_pixels(tiny, rng).sum() == 1 for _ in range(50))
    mask_b = np.zeros((16, 128), bool)
    mask_b[6:10, 60:72] = True
    assert bv.overshoot_pixels(mask_b, rng).sum() == 5
    _, target, _ = _strip_case(2)
    fills, dropped = bv.truth_fills(target, tiny, target * 0 + target.mean(), target[::-1].copy(), rng)
    assert "overshoot_above_clip" in fills and not np.array_equal(fills["overshoot_above_clip"][tiny], target[tiny])
    assert all(not np.array_equal(f[tiny], target[tiny]) for f in fills.values()) and isinstance(dropped, list)


def test_nonfinite_panel_values_are_counted():
    R = pd.DataFrame({"kind": ["a", "a", "b"], **{m: [0.1, 0.2, 0.3] for m in bv.PANEL}})
    R.loc[0, "psnr"] = np.inf
    R.loc[1, "rmse"] = np.nan
    assert bv.nonfinite_counts(R) == {"a": {"rmse": 1, "psnr": 1}}


def test_guarded_loader_error_hides_the_path():
    def boom(path):
        raise FileNotFoundError(f"cannot open {path}")
    with pytest.raises(RuntimeError) as e:
        bv.guarded(boom, r"C:\data\upol\raw\001L_1.png", what="image #3")
    assert "001L" not in str(e.value) and "upol" not in str(e.value) and e.value.__suppress_context__


def test_h14_fills_share_the_same_noise_inside_the_small_mask():
    rng, target, mask_a = _strip_case(3)
    mask_b = np.zeros_like(mask_a)
    mask_b[6:10, 60:72] = True
    f = bv.h14_fills(target, {"A": mask_a, "B": mask_b}, rng)
    np.testing.assert_allclose(f["A"][mask_b], f["B"][mask_b])
    assert (f["B"][mask_a & ~mask_b] == target[mask_a & ~mask_b]).all() and not np.allclose(f["A"][mask_a], target[mask_a])


def test_weighted_discriminability_matches_unweighted_and_expansion():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 3))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    item = np.array(list("aabbccddeeff"))
    person = np.array(["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4)
    code = pd.factorize(person)[0]
    disc = bv.Discriminability(D, item, code)
    assert disc.value() == pytest.approx(bv.discriminability(D, item, person))
    sub = person != "p3"  # 0/1 weights equal the statistic on the kept persons
    assert disc.value(np.array([1.0, 1.0, 0.0])) == pytest.approx(bv.discriminability(D[np.ix_(sub, sub)], item[sub], person[sub]))
    w = np.array([2.0, 1.0, 3.0])  # general weights against a direct weighted count
    num = den = 0.0
    for i in range(12):
        for j in range(12):
            if i != j and item[i] == item[j]:
                ks = [k for k in range(12) if code[k] != code[i]]
                wk = np.array([w[code[k]] for k in ks])
                hit = np.array([1.0 if D[i, k] > D[i, j] else 0.5 if D[i, k] == D[i, j] else 0.0 for k in ks])
                num += w[code[i]] * (wk * hit).sum() / wk.sum()
                den += w[code[i]]
    assert disc.value(w) == pytest.approx(num / den)


def test_h12_verdict_reads_the_raw_and_affine_templates():
    shares = {(t, m): 0.3 for t in bv.TEMPLATES for m in bv.PANEL}
    cis = {(t, m): [0.2, 0.4] for t in bv.TEMPLATES for m in bv.PANEL}
    assert bv.h12_verdict(shares, cis).startswith("not supported")
    shares[("template_affine", "mse")] = shares[("template_affine", "psnr")] = 0.7
    cis[("template_affine", "mse")] = cis[("template_affine", "psnr")] = [0.6, 0.8]
    assert bv.h12_verdict(shares, cis) == "supported (template_affine)"
    shares[("template_sd_matched", "mse")] = 0.9  # the SD-matched template never decides the verdict
    shares[("template_affine", "psnr")] = 0.45
    cis[("template_affine", "psnr")] = [0.4, 0.5]
    assert bv.h12_verdict(shares, cis) == "inconclusive"
