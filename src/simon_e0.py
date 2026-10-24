"""E0 on SIMON (one healthy person scanned repeatedly on many scanners): separate scanner change from time, and test
whether the mirror (left-right) axis cancels global nuisance.

Per structure, on per-session FastSurfer volumes reconstructed from the MIDL 2026 pair table:
  log(volume) ~ years + (1 | scanner)   -> annual change (L2), scanner SD (L1), residual SD (L0 + state)
The same model on the asymmetry index AI = log(left) - log(right) tests the mirror axis. If left and right carried
independent noise, AI would vary sqrt(2) times more than one hemisphere; it varies less only if scanner gain,
hydration or time-of-day effects scale both hemispheres alike and cancel in the difference.

Estimation: exact profile REML for one random intercept (search over the variance ratio, boundary at zero evaluated
explicitly), cross-checked against statsmodels MixedLM with its default optimiser chain. The first version of this
script used statsmodels' lbfgs with warnings silenced; it returned spurious zero scanner variances (infinite
log-likelihood, residual SD below the OLS SD), so the estimator is now explicit and covered by tests.

Time base: years since the first session, from acquisition dates (the Age column is rounded and is not used).
QC: a session whose median Dice is below SESSION_QC_DICE in every pair it appears in is a whole-session failure
and is left out of the primary analysis (a sensitivity analysis keeps it); label cells with Dice == 0 are dropped.
Uncertainty: percentile intervals from a scanner-cluster bootstrap; structures share sessions, so the bootstrap
interval of the across-structure median, not the Wilcoxon test, is the primary inference.
The annual change is aliased with scanner era (each scanner covers a limited span of dates); it is reported with a
manufacturer-adjusted sensitivity and is not an ageing rate.

Inputs belong to the MIDL repository (brain-mri-segmentation/data); outputs are aggregate per structure.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from scipy import optimize, stats

MIDL = r"C:\Projects\02_academia\studies\brain-mri-segmentation\data"
PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
N_BOOT = 1000
SESSION_QC_DICE = 0.5
LAMBDA_GRID = np.r_[0.0, np.logspace(-4, 4, 161)]
# Possible spellings of one physical scanner (institution name variants) or a hardware upgrade of it. Not verified;
# used only in a sensitivity analysis.
SCANNER_ALIASES = {"Mc Gill University | TrioTim": "Mc Connell Brain Imaging Centre | TrioTim",
                   "UBC | Intera": "UBC MRI Research Centre | Intera",
                   "IRM Quebec - Mailloux | Achieva dStream": "IRM Quebec - Mailloux | Achieva"}
KINDS = ("left", "right", "asymmetry")


class RandomInterceptREML:
    """Exact REML for y = X b + u[group] + e with u ~ N(0, s2_u) and e ~ N(0, s2_e), profiled over s2_e.

    With lam = s2_u / s2_e the covariance is s2_e H, with H block-diagonal and H_g = I + lam 11'. Hence
    H_g^-1 = I - c_g 11' with c_g = lam / (1 + lam n_g), and log|H_g| = log(1 + lam n_g). The profiled criterion
        (n - p) log(q / (n - p)) + sum_g log(1 + lam n_g) + log|X'H^-1 X|,   q = y'H^-1 y - (X'H^-1 y)' b_gls
    is minimised over lam >= 0 on a log grid that includes lam = 0, then by a bounded scalar search next to the best
    grid point. At lam = 0 the fit is OLS, so a zero group variance always comes with the OLS residual SD.
    """

    def __init__(self, y, X, groups):
        self.y, self.X = np.asarray(y, float), np.asarray(X, float)
        codes, self.levels = pd.factorize(np.asarray(groups))
        self.n, self.p = self.X.shape
        Z = np.zeros((len(self.levels), self.n))
        Z[codes, np.arange(self.n)] = 1.0
        self.ng = Z.sum(1)
        self.S, self.T = Z @ self.X, Z @ self.y
        self.XtX, self.Xty, self.yty = self.X.T @ self.X, self.X.T @ self.y, self.y @ self.y

    def _parts(self, lam):
        lam = np.atleast_1d(np.asarray(lam, float))
        c = lam[:, None] / (1.0 + lam[:, None] * self.ng[None, :])
        A = self.XtX[None] - np.einsum("lg,gi,gj->lij", c, self.S, self.S)
        b = self.Xty[None] - np.einsum("lg,gi,g->li", c, self.S, self.T)
        beta = np.linalg.solve(A, b[..., None])[..., 0]
        q = self.yty - (c * self.T[None] ** 2).sum(1) - (b * beta).sum(1)
        return A, beta, q

    def criterion(self, lam) -> np.ndarray:
        lam = np.atleast_1d(np.asarray(lam, float))
        A, _, q = self._parts(lam)
        dof = self.n - self.p
        return dof * np.log(q / dof) + np.log1p(lam[:, None] * self.ng[None, :]).sum(1) + np.linalg.slogdet(A)[1]

    def fit(self) -> dict:
        if self.n - self.p < 2 or len(self.levels) < 2:
            raise ValueError(f"REML needs more data: n={self.n}, p={self.p}, groups={len(self.levels)}")
        crit = self.criterion(LAMBDA_GRID)
        if not np.isfinite(crit).all():
            raise FloatingPointError("non-finite REML criterion on the grid (perfect fit or singular design)")
        i = int(np.argmin(crit))
        lam, best = float(LAMBDA_GRID[i]), float(crit[i])
        lo, hi = LAMBDA_GRID[max(i - 1, 0)], LAMBDA_GRID[min(i + 1, len(LAMBDA_GRID) - 1)]
        r = optimize.minimize_scalar(lambda x: self.criterion(x)[0], bounds=(lo, hi), method="bounded",
                                     options={"xatol": 1e-12})
        if r.fun < best:
            lam, best = float(r.x), float(r.fun)
        A, beta, q = self._parts(lam)
        s2e = float(q[0]) / (self.n - self.p)
        se = np.sqrt(np.diag(s2e * np.linalg.inv(A[0])))
        return {"n": self.n, "n_groups": len(self.levels), "variance_ratio": lam, "criterion": best,
                "beta": beta[0], "se": se, "scanner_sd": float(np.sqrt(lam * s2e)), "residual_sd": float(np.sqrt(s2e)),
                "at_upper_bound": bool(i == len(LAMBDA_GRID) - 1)}


def session_volumes(pairs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Per-session volumes from the pair table; a label with Dice == 0 in a pair is dropped for both sessions."""
    long = pd.concat([pairs[["session1", "label_name", "volume1"]].set_axis(["session", "label", "volume"], axis=1),
                      pairs[["session2", "label_name", "volume2"]].set_axis(["session", "label", "volume"], axis=1)])
    failed = pairs.loc[pairs["dice"] == 0, ["session1", "session2", "label_name"]]
    drop = set(zip(failed.session1, failed.label_name)) | set(zip(failed.session2, failed.label_name))
    long = long[[(s, l) not in drop for s, l in zip(long.session, long.label)]]
    g = long.groupby(["session", "label"])["volume"]
    if ((g.max() - g.min()) / g.mean()).max() > 1e-9:
        raise ValueError("a session's volume differs between pairs")
    return g.first().unstack("label"), len(drop)


def failed_sessions(pairs: pd.DataFrame, threshold: float = SESSION_QC_DICE) -> list[str]:
    """Sessions whose median Dice over labels is below threshold in every pair they appear in.

    A poor pair alone does not flag a session: its other session may be at fault. A session that is poor against
    all of its partners has a whole-session problem (registration, orientation or segmentation). A chain endpoint
    has one partner only, so a poor end pair flags both of its sessions.
    """
    med = pairs.groupby(["session1", "session2"])["dice"].median().reset_index()
    long = pd.concat([med[["session1", "dice"]].set_axis(["session", "dice"], axis=1),
                      med[["session2", "dice"]].set_axis(["session", "dice"], axis=1)])
    best = long.groupby("session")["dice"].max()
    return sorted(best[best < threshold].index)


def lr_swap_check(ai: pd.DataFrame, sessions: list[str]) -> dict:
    """Spearman correlation of a session's asymmetry profile (over structures) with the median profile of the other
    sessions. A strongly negative value points to swapped left/right labels rather than to random segmentation error."""
    out = {}
    for s in sessions:
        ref = ai.drop(index=s).median()
        ok = ai.loc[s].notna() & ref.notna()
        rho = stats.spearmanr(ai.loc[s][ok], ref[ok])
        out[s] = {"spearman_rho": float(rho.statistic), "n_structures": int(ok.sum())}
    return out


def pheno() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(MIDL, "SIMON_pheno.csv"))
    p["session"] = p["Session"].map(lambda s: f"ses-{int(s):03d}")
    p["scanner"] = p.institution_name.astype(str).str.strip() + " | " + p.man_model_name.astype(str).str.strip()
    p["date"] = pd.to_datetime(p.Acquisition_date, format="%m/%d/%y")
    p["manufacturer"] = p["manufacturer"].astype(str).str.strip()
    return p.set_index("session")[["scanner", "date", "manufacturer"]]


def design(meta: pd.DataFrame, manufacturer: bool = False) -> np.ndarray:
    cols = [np.ones(len(meta)), meta["years"].to_numpy() - meta["years"].mean()]
    if manufacturer:
        cols += list(pd.get_dummies(meta["manufacturer"], drop_first=True, dtype=float).to_numpy().T)
    return np.column_stack(cols)


def fit(y: pd.Series, meta: pd.DataFrame, manufacturer: bool = False) -> dict:
    y = y.dropna()
    m = meta.loc[y.index]
    r = RandomInterceptREML(y.to_numpy(), design(m, manufacturer), m["scanner"].to_numpy()).fit()
    return {"n": r["n"], "n_scanners": r["n_groups"], "slope_per_year": float(r["beta"][1]),
            "slope_se": float(r["se"][1]), "scanner_sd": r["scanner_sd"], "residual_sd": r["residual_sd"],
            "variance_ratio": r["variance_ratio"], "at_upper_bound": r["at_upper_bound"]}


def outcome_table(logv: pd.DataFrame) -> pd.DataFrame:
    """Columns (structure, kind): log left, log right and AI = log left - log right."""
    bases = sorted({c[:-2] for c in logv.columns if c.endswith(("_L", "_R"))})
    cols = {}
    for b in bases:
        cols[(b, "left")], cols[(b, "right")] = logv[b + "_L"], logv[b + "_R"]
        cols[(b, "asymmetry")] = logv[b + "_L"] - logv[b + "_R"]
    return pd.DataFrame(cols)


def estimates(Y: pd.DataFrame, meta: pd.DataFrame, manufacturer: bool = False) -> tuple[np.ndarray, dict]:
    """Array [structure, kind, (scanner_sd, residual_sd, slope)] and the full fits keyed by (structure, kind)."""
    bases = list(dict.fromkeys(b for b, _ in Y.columns))
    arr = np.full((len(bases), 3, 3), np.nan)
    fits = {}
    for i, b in enumerate(bases):
        for k, kind in enumerate(KINDS):
            f = fit(Y[(b, kind)], meta, manufacturer)
            fits[(b, kind)] = f
            arr[i, k] = f["scanner_sd"], f["residual_sd"], f["slope_per_year"]
    return arr, fits


def derived(arr: np.ndarray) -> dict:
    """Per-structure quantities from an estimates array (used identically for point estimates and bootstrap draws)."""
    sc, rs, sl = arr[..., 0], arr[..., 1], arr[..., 2]
    tot = np.sqrt(sc ** 2 + rs ** 2)
    return {"scanner_sd_hemi": sc[:, :2].mean(1), "residual_sd_hemi": rs[:, :2].mean(1),
            "scanner_sd_AI": sc[:, 2], "residual_sd_AI": rs[:, 2],
            "total_sd_hemi": tot[:, :2].mean(1), "total_sd_AI": tot[:, 2],
            "ratio_total_AI_to_hemi": tot[:, 2] / tot[:, :2].mean(1),
            # 1 - var(L - R) / (var L + var R) = cov(L, R) / mean variance: the share of nuisance the hemispheres share
            "shared_nuisance_fraction": 1 - tot[:, 2] ** 2 / (tot[:, 0] ** 2 + tot[:, 1] ** 2),
            "annual_change": sl[:, :2].mean(1)}


def summarize(arr: np.ndarray) -> dict:
    d = derived(arr)
    out = {f"median_{k}": float(np.median(v)) for k, v in d.items()}
    out["structures_with_AI_total_below_hemisphere"] = int((d["ratio_total_AI_to_hemi"] < 1).sum())
    return out


def resample_scanners(scanners: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One scanner-cluster bootstrap draw: scanners with replacement; a scanner drawn twice becomes two clusters."""
    levels = np.unique(scanners)
    rows, groups = [], []
    for k, s in enumerate(rng.choice(levels, size=len(levels), replace=True)):
        r = np.flatnonzero(scanners == s)
        rows.append(r)
        groups.append(np.full(len(r), k))
    return np.concatenate(rows), np.concatenate(groups)


def bootstrap_draw(Yv: np.ndarray, X: np.ndarray, scanners: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rows, groups = resample_scanners(scanners, rng)
    Yr, Xr = Yv[rows], X[rows]
    est = np.full((Yv.shape[1], 3), np.nan)
    for j in range(Yv.shape[1]):
        ok = np.isfinite(Yr[:, j])
        r = RandomInterceptREML(Yr[ok, j], Xr[ok], groups[ok]).fit()
        est[j] = r["scanner_sd"], r["residual_sd"], r["beta"][1]
    return est


def scanner_bootstrap(Y: pd.DataFrame, meta: pd.DataFrame, n_boot: int, seed: int = 0) -> tuple[np.ndarray, int]:
    """Bootstrap draws reshaped to [draw, structure, kind, stat]; failed refits are counted and must stay rare."""
    rng = np.random.default_rng(seed)
    Yv, X, scanners = Y.to_numpy(float), design(meta), meta["scanner"].to_numpy()
    t0 = time.time()
    first = bootstrap_draw(Yv, X, scanners, rng)
    per = time.time() - t0
    assert np.isfinite(first).all(), "bootstrap smoke test produced non-finite estimates"
    print(f"bootstrap smoke test: {per:.2f}s/draw; estimated total {per * n_boot / 60:.1f} min for {n_boot} draws")
    draws, failed = [first], 0
    while len(draws) < n_boot:
        try:
            draws.append(bootstrap_draw(Yv, X, scanners, rng))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            failed += 1  # degenerate resample; counted and reported
            if failed > 0.05 * n_boot:
                raise RuntimeError(f"scanner bootstrap: {failed} failed draws")
    return np.stack(draws).reshape(n_boot, -1, 3, 3), failed


def ci(a: np.ndarray) -> list:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def statsmodels_crosscheck(Y: pd.DataFrame, meta: pd.DataFrame, fits: dict) -> dict:
    """Refit every primary model with statsmodels and compare REML criteria at the two solutions.

    If statsmodels reaches a lower criterion than the profile search, the search missed the optimum: that is an
    error. If statsmodels lands higher, it stopped early (the failure of the first version with method='lbfgs').
    Warnings are recorded and counted, not silenced.
    """
    rows = []
    for b, kind in fits:
        y = Y[(b, kind)].dropna()
        df = meta.loc[y.index].assign(y=y.to_numpy(), t=lambda d: d["years"] - d["years"].mean())
        problem = RandomInterceptREML(df["y"], design(df), df["scanner"])
        opt = problem.fit()
        for method in (None, "lbfgs"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m = smf.mixedlm("y ~ t", df, groups=df["scanner"]).fit(reml=True, method=method)
            var_u = max(float(m.cov_re.iloc[0, 0]), 0.0)
            # statsmodels' boundary warning uses an absolute threshold (variance < 0.01), so it fires for every
            # log-scale model; it is counted apart from the warnings that carry information
            boundary = sum("may be on the boundary" in str(w.message) for w in caught)
            rows.append({"method": method or "default", "converged": bool(m.converged), "llf_finite": bool(np.isfinite(m.llf)),
                         "n_warnings": len(caught) - boundary, "n_boundary_heuristic": boundary,
                         "criterion_gap": float(problem.criterion(var_u / float(m.scale))[0] - opt["criterion"]),
                         "scanner_sd_diff": abs(np.sqrt(var_u) - opt["scanner_sd"])})
    r = pd.DataFrame(rows)
    if (r["criterion_gap"] < -1e-6).any():
        raise RuntimeError("statsmodels found a better REML optimum than the profile search")
    out = {}
    for method, g in r.groupby("method"):
        out[method] = {"n_fits": int(len(g)), "n_converged": int(g.converged.sum()), "n_llf_finite": int(g.llf_finite.sum()),
                       "n_with_other_warnings": int((g.n_warnings > 0).sum()),
                       "n_with_boundary_heuristic_warning": int((g.n_boundary_heuristic > 0).sum()),
                       "n_worse_than_optimum_by_1e-3": int((g.criterion_gap > 1e-3).sum()),
                       "max_scanner_sd_diff_when_at_optimum": float(g.loc[g.criterion_gap <= 1e-3, "scanner_sd_diff"].max())}
    return out


def pair_summary(vol: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Descriptive E0 on all session pairs: median |volume change| over labels, by scanner relation and gap."""
    s = list(vol.index)
    rows = []
    for i, a in enumerate(s):
        for b in s[i + 1:]:
            gap = abs(meta.at[b, "years"] - meta.at[a, "years"])
            same = meta.at[a, "scanner"] == meta.at[b, "scanner"]
            pct = (100 * (vol.loc[b] - vol.loc[a]).abs() / ((vol.loc[a] + vol.loc[b]) / 2)).median()
            rows.append((same, meta.at[a, "scanner"] if same else None, gap, pct))
    p = pd.DataFrame(rows, columns=["same_scanner", "scanner", "gap_years", "median_abs_pct_change"])
    p["gap_bin"] = pd.cut(p.gap_years, [-0.01, 1, 3, 10, 20], labels=["<1y", "1-3y", "3-10y", "10-20y"]).astype(str)
    tab = p.groupby(["gap_bin", "same_scanner"])["median_abs_pct_change"].agg(["size", "median"]).reset_index()
    near = p[(p.gap_bin == "<1y") & p.same_scanner]
    top = near["scanner"].value_counts()
    return {"n_pairs": int(len(p)), "by_gap_and_scanner": tab.to_dict(orient="records"),
            "same_scanner_pairs_under_1y": {"n": int(len(near)), "distinct_scanners": int(len(top)),
                                            "share_from_most_frequent_scanner": float(top.iloc[0] / len(near)) if len(near) else None}}


def main():
    meta_all = pheno()
    pairs = pd.read_csv(os.path.join(MIDL, "simon_fastserfer.csv"))
    vol_all, n_drop = session_volumes(pairs)
    vol_all = vol_all.loc[[s for s in vol_all.index if s in meta_all.index]]
    meta_all = meta_all.loc[vol_all.index].copy()
    meta_all["years"] = (meta_all["date"] - meta_all["date"].min()).dt.days / 365.25
    Y_all = outcome_table(np.log(vol_all))
    ai_all = Y_all.xs("asymmetry", axis=1, level=1)

    bad = failed_sessions(pairs)
    keep = [s for s in vol_all.index if s not in bad]
    vol, meta, Y = vol_all.loc[keep], meta_all.loc[keep], Y_all.loc[keep]
    bases = list(dict.fromkeys(b for b, _ in Y.columns))

    arr, fits = estimates(Y, meta)
    draws, failed = scanner_bootstrap(Y, meta, N_BOOT)
    point, boot = derived(arr), [derived(d) for d in draws]
    boot_summary = [summarize(d) for d in draws]

    out = {
        "source": "brain-mri-segmentation/data/simon_fastserfer.csv + SIMON_pheno.csv",
        "supersedes": "first run of 2026-10-23 (statsmodels lbfgs with warnings silenced): its scanner SDs, residual SDs "
                      "and mirror-axis conclusion were wrong",
        "model": "log(volume) ~ years + (1|scanner), profile REML; same model on AI = log(L) - log(R)",
        "time_base": "years since first session from acquisition dates",
        "units": "SDs and slopes on the log scale; x100 = approximate percent",
        "qc": {"dice0_session_label_cells_dropped": n_drop, "session_rule": f"median Dice < {SESSION_QC_DICE} in every pair",
               "sessions_excluded": bad, "lr_swap_check": lr_swap_check(ai_all, bad),
               "excluded_session_scanners_have_no_other_session": bool(all(
                   (meta_all["scanner"] == meta_all.at[s, "scanner"]).sum() == 1 for s in bad))},
        "n_sessions": int(len(Y)), "n_scanners": int(meta["scanner"].nunique()),
        "bootstrap": {"type": "scanner clusters", "n_boot": N_BOOT, "failed_draws": failed},
        "structures": {},
    }
    rows = []
    for i, b in enumerate(bases):
        ai = Y[(b, "asymmetry")].dropna()
        ai_full = Y_all[(b, "asymmetry")].dropna()
        res = {kind: fits[(b, kind)] for kind in KINDS}
        res["asymmetry_mean"] = float(ai.mean())
        res["asymmetry_sign_consistency"] = float(max((ai > 0).mean(), (ai < 0).mean()))
        res["asymmetry_sign_consistency_with_excluded"] = float(max((ai_full > 0).mean(), (ai_full < 0).mean()))
        res["ci95"] = {k: ci(np.array([d[k][i] for d in boot])) for k in point}
        out["structures"][b] = res
        row = {"structure": b, **{k: float(point[k][i]) for k in point}}
        row.update({f"{k}_ci_lo": res["ci95"][k][0] for k in point})
        row.update({f"{k}_ci_hi": res["ci95"][k][1] for k in point})
        row.update({"AI_mean": res["asymmetry_mean"], "AI_sign_consistency": res["asymmetry_sign_consistency"]})
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(PROJECT, "results", "simon_e0_per_structure.csv"), index=False)

    s = summarize(arr)
    s_ci = {k: ci(np.array([d[k] for d in boot_summary])) for k in s if k.startswith("median_")}
    w = stats.wilcoxon(point["total_sd_AI"], point["total_sd_hemi"])
    out["summary"] = {**s, "n_structures": len(bases), "ci95": s_ci,
                      "wilcoxon_AI_vs_hemisphere_total_sd_descriptive": {"statistic": float(w.statistic), "p": float(w.pvalue)},
                      "structures_with_sign_consistency_1": [b for b in bases if out["structures"][b]["asymmetry_sign_consistency"] == 1.0]}

    Y_alias_meta = meta.assign(scanner=meta["scanner"].replace(SCANNER_ALIASES))
    out["sensitivity"] = {
        "keep_qc_failed_sessions": summarize(estimates(Y_all, meta_all)[0]),
        "merge_possible_scanner_aliases": {**summarize(estimates(Y, Y_alias_meta)[0]),
                                           "n_scanners": int(Y_alias_meta["scanner"].nunique())},
        "manufacturer_fixed_effect": summarize(estimates(Y, meta, manufacturer=True)[0]),
    }
    out["statsmodels_crosscheck"] = statsmodels_crosscheck(Y, meta, fits)
    out["pairs"] = pair_summary(vol, meta)
    out["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                          "scipy": __import__("scipy").__version__, "statsmodels": statsmodels.__version__}
    with open(os.path.join(PROJECT, "results", "simon_e0.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("qc", "n_sessions", "n_scanners", "bootstrap", "summary", "sensitivity",
                                          "statsmodels_crosscheck", "pairs")}, indent=1, default=str))
    print(tab[["structure", "scanner_sd_hemi", "residual_sd_hemi", "scanner_sd_AI", "residual_sd_AI",
               "ratio_total_AI_to_hemi", "shared_nuisance_fraction", "AI_sign_consistency"]].round(4).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
