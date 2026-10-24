"""Tests for the SIMON E0 estimator, session QC and scanner-cluster bootstrap on synthetic data."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import simon_e0  # noqa: E402


def _simulated(seed=0, sd_u=0.03, sd_e=0.02, n_groups=30):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, n_groups)
    g = np.repeat(np.arange(n_groups), sizes)
    t = rng.uniform(0, 15, len(g))
    y = 1.0 - 0.002 * t + rng.normal(0, sd_u, n_groups)[g] + rng.normal(0, sd_e, len(g))
    return pd.DataFrame({"y": y, "t": t - t.mean(), "g": g.astype(str)})


def test_reml_agrees_with_statsmodels_default_optimiser():
    df = _simulated()
    X = np.column_stack([np.ones(len(df)), df["t"]])
    problem = simon_e0.RandomInterceptREML(df["y"], X, df["g"])
    mine = problem.fit()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the reference fit itself must be clean...
        # ...except for statsmodels' boundary heuristic, an absolute threshold (variance < 0.01) that fires for any
        # log-scale volume model regardless of where the optimum is
        warnings.filterwarnings("ignore", message="The MLE may be on the boundary")
        ref = smf.mixedlm("y ~ t", df, groups=df["g"]).fit(reml=True)
    assert ref.converged
    assert abs(mine["scanner_sd"] - np.sqrt(ref.cov_re.iloc[0, 0])) < 1e-4
    assert abs(mine["residual_sd"] - np.sqrt(ref.scale)) < 1e-4
    assert abs(mine["beta"][1] - ref.params["t"]) < 1e-6
    assert abs(mine["se"][1] - ref.bse["t"]) < 1e-5
    # the profile search is never beaten by the statsmodels solution
    assert problem.criterion(ref.cov_re.iloc[0, 0] / ref.scale)[0] >= mine["criterion"] - 1e-9


def test_zero_group_variance_returns_ols_residual_sd():
    # Within every group the residuals are +e and -e at the same time point, so group means carry no extra
    # variance and the REML optimum is the boundary. The first version of the script reported a zero scanner SD
    # together with a residual SD below OLS; at the boundary the residual SD must equal the OLS SD exactly.
    rng = np.random.default_rng(1)
    t = np.repeat(rng.uniform(0, 10, 20), 2)
    e = np.repeat(rng.normal(0, 0.02, 20), 2) * np.tile([1, -1], 20)
    y = 0.5 + 0.01 * t + e
    g = np.repeat(np.arange(20), 2)
    X = np.column_stack([np.ones(len(y)), t])
    r = simon_e0.RandomInterceptREML(y, X, g).fit()
    rss = np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)
    assert r["variance_ratio"] == 0.0 and r["scanner_sd"] == 0.0
    assert np.isfinite(r["criterion"])
    assert abs(r["residual_sd"] - np.sqrt(rss / (len(y) - 2))) < 1e-12


def test_recovers_known_variance_components():
    est = np.array([[simon_e0.RandomInterceptREML(d["y"], np.column_stack([np.ones(len(d)), d["t"]]), d["g"]).fit()[k]
                     for k in ("scanner_sd", "residual_sd")] for d in (_simulated(s, n_groups=200) for s in range(5))])
    assert abs(est[:, 0].mean() - 0.03) < 0.004 and abs(est[:, 1].mean() - 0.02) < 0.002


def _pairs(dice_by_pair):
    rows = []
    for (a, b), dice in dice_by_pair.items():
        for label in ("x_L", "x_R", "y_L"):
            rows.append({"session1": a, "session2": b, "label_name": label, "volume1": 1.0, "volume2": 1.0, "dice": dice})
    return pd.DataFrame(rows)


def test_session_qc_flags_whole_session_failure_only():
    pairs = _pairs({("s1", "s2"): 0.9, ("s2", "s3"): 0.1, ("s3", "s4"): 0.1, ("s4", "s5"): 0.9})
    assert simon_e0.failed_sessions(pairs, 0.5) == ["s3"]


def test_single_failed_label_does_not_flag_a_session():
    pairs = _pairs({("s1", "s2"): 0.9, ("s2", "s3"): 0.9})
    pairs.loc[(pairs.session1 == "s2") & (pairs.label_name == "x_L"), "dice"] = 0.0
    assert simon_e0.failed_sessions(pairs, 0.5) == []
    vol, n_drop = simon_e0.session_volumes(pairs)
    assert n_drop == 2 and vol.loc["s2", "x_L"] != vol.loc["s2", "x_L"]  # dropped cell is NaN


def test_scanner_bootstrap_keeps_whole_scanners_and_splits_duplicates():
    scanners = np.array(["a", "a", "a", "b", "c", "c"])
    rng = np.random.default_rng(3)
    for _ in range(50):
        rows, groups = simon_e0.resample_scanners(scanners, rng)
        for k in np.unique(groups):
            members = rows[groups == k]
            assert len(set(scanners[members])) == 1  # a cluster is one scanner
            assert len(members) == (scanners == scanners[members[0]]).sum()  # with all of its sessions
        assert len(np.unique(groups)) == 3  # one cluster per draw, duplicates included


def test_lr_swap_check_detects_mirrored_session():
    rng = np.random.default_rng(4)
    profile = rng.normal(0, 0.05, 12)
    ai = pd.DataFrame(profile + rng.normal(0, 0.005, (8, 12)), index=[f"s{i}" for i in range(8)])
    ai.loc["s7"] = -profile
    out = simon_e0.lr_swap_check(ai, ["s7", "s0"])
    assert out["s7"]["spearman_rho"] < -0.8 and out["s0"]["spearman_rho"] > 0.8
