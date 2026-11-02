import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import real_cases as rc  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def test_box_and_mirror_on_a_symmetric_grid():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-10, -5, -5]
    shape = (21, 11, 11)
    box = rc.box_mask(shape, affine, ((-8, -4), (-2, 2), (0, 3)))
    mirror = rc.box_mask(shape, affine, ((4, 8), (-2, 2), (0, 3)))
    assert box.sum() == mirror.sum() == 4 * 4 * 3
    vol = np.zeros(shape)
    vol[affine[0, 3] + np.arange(shape[0]) == 4] = 1.0  # x = 4: the first mirror column
    other = rc.mirrored_values(vol, box, mirror).reshape(4, 4, 3)
    assert other[-1].all() and not other[:-1].any()  # x = 4 lands at the inner edge of the left box (x = -5)


def test_wm_peak_is_the_upper_mode():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(0.3, 0.05, 2000), rng.normal(0.65, 0.05, 6000), rng.normal(1.0, 0.04, 5000)])
    assert abs(rc.wm_peak(v) - 1.0) < 0.03


def test_fills_hold_their_definitions():
    rng = np.random.default_rng(1)
    x, mu, s1 = rng.normal(1, 0.2, 500), rng.normal(1, 0.05, 500), rng.normal(1, 0.2, 500)
    brain = np.ones(500, bool)
    brain[:20] = False
    x[~brain] = mu[~brain] = s1[~brain] = 0
    f = rc.fills(x, mu, s1, brain=brain)
    assert np.array_equal(f["mean"], mu)
    np.testing.assert_allclose(f["mean_restored"][brain].std(), x[brain].std())
    assert not f["mean_restored"][~brain].any()
    for lam in rc.LAMBDAS:
        d = rc.diagnostics(x, f[f"shrink_{lam:g}"], mu, np.ones(500, bool))
        assert abs(d["target_projection"] - lam) < 1e-9 and abs(d["residual_ratio"] - lam) < 1e-9
    d = rc.diagnostics(x, f["wrong_detail"], mu, np.ones(500, bool))
    assert abs(d["target_projection"]) < 1e-9 and abs(d["residual_ratio"] - 1) < 1e-9


def toy_volume(seed=2):
    rng = np.random.default_rng(seed)
    x = rng.random((12, 24, 24)).astype(np.float32)
    box = np.zeros(x.shape, bool)
    box[3:9, 8:16, 8:16] = True
    return x, box, rng


def test_metrics_reward_the_target_and_calibrate_against_strangers():
    x, box, rng = toy_volume()
    n = int(box.sum())
    strangers = {j: rng.random(n).astype(np.float32) for j in range(5)}
    named, refs = rc.score_volume(x, box, {"oracle": x[box], "near": x[box] + 0.01}, strangers)
    assert named["oracle"]["rmse"] == 0 and set(named["oracle"]) == set(rc.METRICS)
    c = rc.c_scores(named["near"], list(refs.values()))
    assert c["rmse"] == 1 and c["ssim"] == 1 and c["psnr"] == 1  # higher-is-better metrics oriented correctly
    assert rc.rho(named["near"], list(refs.values())) < 0.01


def test_bootstrap_is_seeded_and_labels_follow_the_protocol():
    v = np.random.default_rng(3).normal(0.1, 0.05, 40)
    a, b = rc.summary(v, rc.boot_index(40)), rc.summary(v, rc.boot_index(40))
    assert a == b and a["ci95"][0] < a["mean"] < a["ci95"][1]
    assert rc.label_c1([0.01, 0.2]).startswith("the population mean")
    assert rc.label_c1([-0.2, -0.01]).startswith("the sibling outranks")
    assert rc.label_c1([-0.1, 0.1]) == "no difference resolved"
    assert rc.label_c2([0.51, 0.7]) == "the sibling is closer than a stranger"
    assert rc.label_c2([0.49, 0.7]).startswith("label not met")


def test_label_measures():
    oracle = np.zeros((10, 10, 10), np.int32)
    oracle[2:8, 2:8, 2:5] = 3
    oracle[2:8, 2:8, 5:8] = 2
    same = rc.label_measures(oracle.copy(), oracle)
    assert same["dice_mean"] == 1 and same["accuracy"] == 1 and same["euler_error"] == same["components_error"] == 0
    split = oracle.copy()
    split[:, 5, :] = 0  # cuts the cortex into two pieces
    moved = rc.label_measures(split, oracle)
    assert moved["dice_mean"] < 1 and moved["components_error"] == 1


def test_label_analysis_gates_on_the_blur_sweep():
    rng = np.random.default_rng(6)

    def meas(d):
        return {"dice_mean": d, "accuracy": d, "euler_error": 0, "components_error": 0}

    def rows(drop):
        out = []
        for t in range(6):
            m = {f"stranger_{j}": meas(0.6 + rng.normal(0, 0.02)) for j in range(16)}
            m.update(mean=meas(0.8), sibling=meas(0.65), other_hemisphere=meas(0.62))
            m.update({f"blur_{s}": meas(1 - drop * s + rng.normal(0, 0.01)) for s in rc.BLUR_SIGMAS})
            out.append({"id": f"S{t:02d}", "measures": m, "failed": []})
        return out
    seen = rc.analyse_labels(rows(0.05))
    assert seen["segmenter_sees_blur"] and seen["c"]["mean"]["dice_mean"]["mean"] == 1
    blind = rc.analyse_labels(rows(0.0))
    assert not blind["segmenter_sees_blur"] and blind["interpretation"].startswith("uninformative")
    assert rc.analyse_labels([]) == {"run": False}


def test_robust_flags_and_qc_summary():
    assert rc.robust_flags([1.0, 1.1, 0.9, 1.0, 5.0, None]) == [4]
    assert rc.robust_flags([2.3, 2.3, 2.3, 2.4]) == [3]  # zero MAD: any departure is flagged
    rows = []
    for i in range(20):
        r = {"scan": f"scan_{i:03d}", "tr_s": 1.9, "te_s": 0.0029, "ti_s": 0.9, "flip_deg": 9, "field_t": 4,
             "scanner": "Bruker", "software": "v1" if i < 15 else "v2", "acq_orientation": "1", "voxel_mm": "0.9x0.94x0.94",
             "matrix": "256x256x256", "orientation": "RAS", "dtype": "int16", "nonfinite": 0, "prep_error": None}
        r.update({k: 1.0 + 0.01 * (i % 5) for k in rc.QC_CONTINUOUS})
        rows.append(r)
    rows[3]["snr_proxy"] = 9.0
    rows[4]["box_in_own_brain"] = 0.90
    s, flagged = rc.qc_summary(rows)
    assert s["continuous"]["snr_proxy"]["flagged_3mad"] == 1 and flagged["snr_proxy"] == ["scan_003"]
    assert s["categorical"]["software"]["departing_from_mode"] == 5 and s["box_coverage_below_95"] == 1
    assert "participant_id" not in str(s)


def test_native_qc_on_a_synthetic_head():
    rng = np.random.default_rng(7)
    img = np.zeros((60, 60, 60))
    zz, yy, xx = np.indices(img.shape)
    head = (zz - 30) ** 2 + (yy - 30) ** 2 + (xx - 30) ** 2 < 25 ** 2
    brain = (zz - 30) ** 2 + (yy - 30) ** 2 + (xx - 30) ** 2 < 15 ** 2
    img[head] = 300
    img[brain] = 100 + rng.normal(0, 3, brain.sum())
    img[~head] = np.abs(rng.normal(0, 5, (~head).sum()))
    q = rc.native_qc(img, brain, (1.0, 1.0, 1.0))
    assert abs(q["wm_peak_native"] - 100) < 3 and abs(q["mask_ml"] - brain.sum() / 1000) < 1e-9
    assert 10 < q["snr_proxy"] < 40  # 100 over the SD of a half-normal background with sigma 5 (about 3)


def design():
    S = [{"target": i, "donor": 10 + i, "sex": "F" if i % 2 else "M"} for i in range(4)]
    R = [{"target": 20 + i, "donor": 30 + i, "sex": "M"} for i in range(3)]
    strangers = list(range(40, 80))
    return {"n_scans": 120, "S": S, "R": R, "strangers": strangers,
            "stranger_sex": ["F" if j % 2 else "M" for j in strangers], "mean_pool": list(range(80, 120))}


def test_assign_drops_without_replacement_and_draws_from_qc_passing_pool():
    excluded = {"scan_011": "x", "scan_045": "x", "scan_081": "x"}
    a, b = rc.assign(design(), excluded), rc.assign(design(), excluded)
    assert a == b
    assert [t["target"] for t in a["S"]] == [0, 2, 3] and a["dropped"] == {"S": 1, "R": 0}
    for t in a["S"] + a["R"]:
        assert len(set(t["strangers"])) == rc.K and 45 not in t["strangers"]
        assert all((j % 2 == 1) == (t["target"] % 2 == 1) for j in t["strangers_sex"]) or t["target"] >= 20
    assert 81 not in a["mean_pool"] and len(a["mean_pool"]) == 39


def fake_row(set_name, order, rng, relative_gap):
    """Synthetic metric rows: strangers around RMSE 0.2, the mean at 0.15, the relative at 0.15 + gap."""
    def raw(r):
        return {"rmse": r, "mse": r * r, "mae": r * 0.8, "msle": r * r / 2, "psnr": -20 * np.log10(r),
                "psnr_eps": -20 * np.log10(r), "psnr_01": -20 * np.log10(r), "psnr_01_eps": -20 * np.log10(r),
                "ssim": 1 - r}
    rel = "sibling" if set_name == "S" else "repeat"
    base = {rel: 0.15 + relative_gap, "other_hemisphere": 0.17, "mean": 0.15, "mean_restored": 0.19,
            "shrink_0.75": 0.05, "shrink_0.5": 0.08, "shrink_0.25": 0.12, "wrong_detail": 0.2, "mean20": 0.155}
    fills = {k: raw(v + rng.normal(0, 0.003)) for k, v in base.items()}
    pool = list(range(40))
    strangers = {str(j): raw(0.2 + rng.normal(0, 0.02)) for j in pool}
    diag = {"residual_ratio": 1.0, "target_projection": 0.5}
    st = [int(j) for j in rng.choice(pool, rc.K, replace=False)]
    return {"id": f"{set_name}{order:02d}", "set": set_name, "order": order,
            "wm": {"fills": fills, "strangers": strangers},
            "z": {"fills": {k: v for k, v in fills.items() if k != "mean20"}, "strangers": strangers},
            "diagnostics": {**{k: diag for k in base}, **{f"stranger_{j}": diag for j in st}},
            "strangers": st, "strangers_sex": st[:10]}


def test_analysis_labels_c1_and_c2_on_synthetic_rows():
    rng = np.random.default_rng(4)
    rows = [fake_row("S", i, rng, 0.035) for i in range(30)] + [fake_row("R", i, rng, -0.1) for i in range(12)]
    res, claims = rc.analyse(rows)
    assert claims["C1"]["label"] == "the population mean outranks the sibling under RMSE"
    assert claims["C2"]["label"] == "the sibling is closer than a stranger"
    assert claims["kate_question"].startswith("the sibling carries individual anatomy")
    assert claims["C4"]["mean_minus_repeat_c_rmse"]["mean"] < 0  # the repeat beats the mean in these rows
    assert res["S"]["n_targets"] == 30 and set(claims["C8"]["S"]["kendall_tau_vs_rmse"]) == set(rc.METRICS)
    assert claims["C8"]["S"]["kendall_tau_vs_rmse"]["rmse"]["mean"] == 1


def test_score_target_and_analysis_run_end_to_end_on_a_toy_grid(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    shape = (12, 24, 24)
    box, mirror = np.zeros(shape, bool), np.zeros(shape, bool)
    box[1:4, 8:16, 8:16] = True
    mirror[8:11, 8:16, 8:16] = True
    brain = np.ones(shape, bool)
    brain[1, 8, 8] = False
    vols = {i: rng.random(shape).astype(np.float32) + 0.5 for i in range(30)}
    monkeypatch.setattr(rc, "RUN", str(tmp_path))
    monkeypatch.setattr(rc, "space", lambda: (None, brain, box, mirror))
    monkeypatch.setattr(rc, "normalized", lambda i, meta, kind="wm", raw=None: vols[i] * brain)
    rc._CACHE.update({k: np.stack([vols[i][box] * brain[box] for i in range(30)]) for k in ("box_wm", "box_z")})
    rows = []
    for t in range(4):  # targets 0-3, siblings 4-7, stranger pool 8-23, mean pool 24-29
        st = [int(j) for j in rng.permutation(range(8, 24))]
        job = {"id": f"S{t:02d}", "order": t, "set": "S", "metas": [{}] * 30, "mean_pool": list(range(24, 30)),
               "stranger_pool": list(range(8, 24)),
               "target": {"target": t, "donor": 4 + t, "strangers": st, "strangers_sex": st[:8]}}
        assert rc.score_target(job) == f"S{t:02d}"
        rows.append(rc.read_json(str(tmp_path / "scores" / f"S{t:02d}.json")))
    row = rows[0]
    assert set(row["wm"]["fills"]) == {"sibling", "other_hemisphere", "mean", "mean_restored", "shrink_0.75",
                                       "shrink_0.5", "shrink_0.25", "wrong_detail", "mean20"}
    assert len(row["wm"]["strangers"]) == len(row["z"]["strangers"]) == rc.K
    assert abs(row["diagnostics"]["shrink_0.5"]["target_projection"] - 0.5) < 1e-6
    rows[0]["wm"]["strangers"][str(st[0])]["psnr"] = "Infinity"  # an exact copy, as JSON spells it
    res, claims = rc.analyse(rows)
    assert res["S"]["n_targets"] == 4 and "C1" in claims and "C4" not in claims


def test_expanded_design_drops_rows_not_targets_and_labels_each_relative():
    design = {"n_scans": 60, "mean_pool": list(range(40, 60)),
              "targets": [{"id": f"T{k:03d}", "order": k, "target": k, "relatives": {"cotwin": 10 + k, "sibling": 20 + k},
                           "strangers": list(range(24, 40))} for k in range(3)]}
    plan = rc.assign_expanded(design, {"scan_000": "x", "scan_011": "x", "scan_025": "x", "scan_041": "x"})
    assert [t["id"] for t in plan["targets"]] == ["T001", "T002"] and plan["dropped_targets"] == 1
    assert plan["dropped_rows"] == {"cotwin": 1} and "cotwin" not in plan["targets"][0]["relatives"]
    assert 25 not in plan["targets"][0]["strangers"] and 41 not in plan["mean_pool"]
    rng = np.random.default_rng(8)
    rows = []
    for k in range(30):
        r = fake_row("S", k, rng, 0.035)
        r["wm"]["fills"]["cotwin"] = r["wm"]["fills"]["mean"]  # a co-twin that ties the mean
        r["diagnostics"]["cotwin"] = r["diagnostics"]["mean"]
        del r["wm"]["fills"]["mean20"]
        rows.append(r)
    res, claims = rc.analyse_expanded(rows)
    assert claims["C1-all"]["label_mean_vs_relative"] == "the population mean outranks the sibling under RMSE"
    assert claims["C11"]["label_mean_vs_relative"] == "no difference resolved"
    assert claims["C11"]["label_relative_vs_strangers"] == "the co-twin is closer than a stranger"
    assert "repeat" not in claims and res["n_targets"] == 30


def test_guard_refuses_changed_inputs_and_missing_amendment(tmp_path, monkeypatch):
    data = tmp_path / "input.txt"
    data.write_text("frozen input")
    manifest = tmp_path / "scan_manifest.json"
    manifest.write_text("{}")
    msha, dsha = rc.sha256(str(manifest)), rc.sha256(str(data))
    monkeypatch.setattr(rc, "SCAN_MANIFEST", str(manifest))
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text(f"| input | `{dsha}` |\n\n### Amendment 2 (planned, before any fill is scored)\n\n"
                        f"Scan manifest hash, e.g. `{msha}`, goes here.\n")
    inputs = {"input": (str(data), dsha)}
    rc.guard("prepare", inputs, msha, str(protocol))  # preparation may run before the amendment
    with pytest.raises(SystemExit, match="dated amendment"):
        rc.guard("score", inputs, msha, str(protocol))  # a hash in the planned section does not count
    protocol.write_text(protocol.read_text() + f"\n### Amendment 2: frozen design (2026-11-01)\n\nManifest `{msha}`.\n")
    rc.guard("score", inputs, msha, str(protocol))
    with pytest.raises(SystemExit, match="input hash"):
        rc.guard("prepare", {"input": (str(data), "0" * 64)}, msha, str(protocol))
    with pytest.raises(SystemExit, match="scan manifest"):
        rc.guard("prepare", inputs, "1" * 64, str(protocol))


def test_run_constants_match_the_frozen_protocol():
    text = open(rc.PROTOCOL, encoding="utf-8").read()
    for _, want in rc.INPUTS.values():
        assert want in text
    assert "x [-45, -13), y [-65, -33), z [43, 67)" in text and "x [13, 45)" in text
    assert rc.PARIETAL_BOX == ((-45, -13), (-65, -33), (43, 67)) and rc.PARIETAL_MIRROR[0] == (13, 45)
    assert rc.MIRROR[0] == (1 - rc.BOX[0][1], 1 - rc.BOX[0][0]) and rc.MIRROR[1:] == rc.BOX[1:]  # exact reflection
    assert rc.SEED == 20261101 and rc.N_BOOT == 2000 and rc.K == 16 and rc.LAMBDAS == (0.75, 0.5, 0.25)
    for image in rc.SEGMENTERS.values():
        assert image.split("@")[1] in text


def test_parietal_box_gets_its_own_outputs():
    try:
        rc.use_box("parietal")
        assert rc.BOX == rc.PARIETAL_BOX and rc.boxed("scores") == "scores_parietal"
        assert rc.boxed("real_cases.json") == "real_cases_parietal.json"
    finally:
        rc.use_box("deep")
    assert rc.boxed("plan.json") == "plan.json" and rc.BOX[0] == (-32, 0)
