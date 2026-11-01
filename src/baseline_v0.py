"""Baseline v0 on the 2D pilots: feature space x comparison metric, scored on the ladder (a partial Section 9.3).

Iris (UTIRIS NIR primary; UPOL secondary; UTIRIS NIR + VIS for a cross-spectral device change within one sitting):
  cells F0 exposure-only (mean and SD of the 16 x 16 grey image), F1 16 x 16 pixels (raw and per-image z-scored),
  F2 classical iris code (Hamming distance, minimum over angular shifts). For every cell: mean c per rung against
  strangers matched on sensor pair and eye relation (fellow eye plain and flipped), AUC between adjacent rungs on c,
  EER and signed d' for same eye vs strangers (pooled, support-matched and sensor-stratified raw distances, and c),
  rank-1 identification within the sitting (averaged over common gallery draws), discriminability, differential
  identifiability, median z against the matched reference, the two-anchor S, and the cross-spectral T3b ratio on raw
  medians and on the c scale. ICC(1) of the F0 scalars per eye (per sensor where two sensors are pooled).
  UTIRIS same-eye pairs are captures from one sitting; UPOL same-eye pairs are repeat captures with the session
  undocumented. Neither is a re-test over days.
Intervals: person-cluster bootstrap. Persons are redrawn with replacement and every pair is weighted by the
  multiplicities of its persons (w_p for a same-person pair, w_p * w_q for a stranger pair), so the stranger reference
  of every replicate is rebuilt from the resampled persons. Rank-1 intervals hold the gallery fixed and resample probe
  persons. Replicate counts are written to the JSON.
2D donor-swap with the official BraTS 2023 inpainting metrics (inpainting.challenge_metrics_2023.generate_metrics):
  on unrolled iris strips (rubber sheet, 16 x 128, aligned by construction up to rotation), a patch is masked and
  filled from the same eye in another capture, the fellow eye (flipped image), strangers of the same and of the
  opposite eye side, the leave-one-person-out mean strip (raw, mean-shifted, affine least-squares fit, SD-matched, and
  blurred), a blurred stranger, blurred copies of the truth (these keep the target's own identity), a gain of 1.1
  and bright overshoot artefacts of a fixed pixel count. Donors are aligned by the circular angular shift (+-SHIFTS)
  that best fits the visible tissue, and by default intensity-matched (mean and SD) to it; raw-intensity donors are
  scored as well. Every panel metric is placed on the ladder: per case, c of a fill is the share of that case's
  stranger fills that score worse (strangers themselves score 0.5 by construction: an anchor, not a result). Only
  strips passing the iris-code segmentation QC are used, as targets and as donors. Masks A (8 x 32) and B (4 x 12).
GRAPE fundus: F0 exposure, F1 pixels (fellow eye also flipped), F4 released RETFound features (Euclidean distance),
  reference matched on camera pair and eye relation as in grape_calibration.py. No pixel BraTS panel: GRAPE image
  pairs are not registered.

Not covered from Section 9.3 (listed in the JSON): normalised cross-correlation, mutual information, cosine distance,
T1 inversion counts, T5 noise / compression / sub-pixel shift degradations, the BraTS panel on the F0-F2 cells.

Only aggregate numbers are written: results/baseline_v0.json. No image, file name or participant identifier is
written or printed; loader errors are re-raised with an index or a count only.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from grape_calibration import calibrated_score, cluster_ci, distance_matrix  # noqa: E402
from iris_calibration import (calibrate, exposure_features, features, pairs_from_matrices, upol_table,  # noqa: E402
                              utiris_table, zscore_rows)
from iris_codes import MIN_VALID_FRACTION, SHIFTS, encode, hamming_matrix, load_grey, log_gabor_code, normalise, segment  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 0
MASKS = {"A_rows4-12_cols50-82": (slice(4, 12), slice(50, 82)),  # 8 x 32 samples at 141-231 deg, clear of the eyelid sectors
         "B_rows6-10_cols60-72": (slice(6, 10), slice(60, 72))}  # 4 x 12 samples inside A
N_SAME, N_FELLOW, N_STRANGERS = 3, 3, 8
BLUR_SIGMAS = (1.0, 2.0, 4.0)
OVERSHOOT_SHARE = 0.10
H14_NOISE_SHARE = 0.05  # additive noise SD as a share of the clipped intensity range, identical values in A and B
N_BOOT = {"default": 200, "UTIRIS_NIR_VIS_cross_spectral": 100}
N_GALLERY_DRAWS = 200
CI_HALF_WIDTH_TARGET = 0.05
HIGHER_IS_BETTER = ("ssim", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps")
PANEL = ("ssim", "mse", "rmse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps")
IRIS_L0, IRIS_L1, IRIS_MIRROR, IRIS_L4 = "same_eye_same_sensor", "same_eye_other_sensor", "fellow_eye_same_sensor", "other_person"
SAME = "same_eye_repeat_capture"
TEMPLATES = ("template_raw", "template_mean_shift", "template_affine", "template_sd_matched", "template_blur_sigma2", "stranger_blur_sigma2")


def guarded(fn, *args, what: str = "image loading", **kw):
    """Call a loader; on failure re-raise without the original message, which can hold a file name."""
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001 - re-raised below without the path-bearing message
        raise RuntimeError(f"{what} failed ({type(e).__name__}); file names suppressed") from None


# ---------- generic metrics (unweighted) ----------

def eer(genuine, impostor) -> float:
    """Equal error rate for distances (genuine expected smaller): FNMR = P(genuine > t) meets FMR = P(impostor <= t)."""
    return weer(genuine, None, impostor, None)


def dprime_signed(genuine, impostor) -> float:
    """Daugman decidability with a sign: positive when genuine distances are smaller than impostor distances."""
    return wdprime(genuine, None, impostor, None)


def auc_above(a, b) -> float:
    """P(a > b) + half the ties, for two samples of calibrated scores c."""
    return float(calibrated_score(np.asarray(b, float), np.asarray(a, float)).mean())


def discriminability(D: np.ndarray, item: np.ndarray, person: np.ndarray) -> float:
    """Bridgeford et al. 2021: for every ordered within-item pair (anchor i, partner j), the share of distances from the
    same anchor to measurements of other persons that exceed d(i, j), ties counted half; averaged over all such pairs.
    Here an item is one eye, and the between-item set excludes the anchor person's other eye."""
    vals = []
    for i in range(len(D)):
        within = item == item[i]
        within[i] = False
        w, b = D[i, within], D[i, person != person[i]]
        w, b = w[np.isfinite(w)], b[np.isfinite(b)]
        if len(w) and len(b):
            vals.append(calibrated_score(w, b))
    return float(np.concatenate(vals).mean()) if vals else float("nan")


def differential_identifiability(D: np.ndarray, item: np.ndarray, person: np.ndarray) -> float:
    """Mean similarity within item minus mean similarity between persons, with similarity = -distance (cell units)."""
    iu, ju = np.triu_indices(len(D), 1)
    d = D[iu, ju]
    ok = np.isfinite(d)
    return float(d[(person[iu] != person[ju]) & ok].mean() - d[(item[iu] == item[ju]) & ok].mean())


def icc1(values, groups) -> float:
    """One-way random-effects ICC(1) for unbalanced groups; groups with one value are dropped."""
    df = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)})
    df = df[df.g.map(df.g.value_counts()) > 1]
    a, n = df.g.nunique(), len(df)
    n_i, means = df.groupby("g").size(), df.groupby("g").v.mean()
    msb = (n_i * (means - df.v.mean()) ** 2).sum() / (a - 1)
    msw = ((df.v - df.g.map(means)) ** 2).sum() / (n - a)
    k0 = (n - (n_i ** 2).sum() / n) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def icc1_sensor_adjusted(values, groups, sensor) -> float:
    """ICC(1) after removing each sensor's mean (sensor as a fixed effect), so a sensor offset is not within-eye error."""
    v = pd.Series(np.asarray(values, float))
    s = pd.Series(np.asarray(sensor))
    return icc1((v - v.groupby(s).transform("mean")).to_numpy(), groups)


def two_anchor_s(d, l0, l4):
    """S = (d_L4 - d) / (d_L4 - d_L0): 1 at the L0 anchor, 0 at the L4 anchor."""
    return (l4 - np.asarray(d, float)) / (l4 - l0)


def rank1(D: np.ndarray, label: np.ndarray, person: np.ndarray, gallery: np.ndarray, probes: np.ndarray) -> dict:
    """Closed-set identification of probes against a gallery; NaN distances count as infinite. The item-level chance is
    1 / gallery size; the person-level chance is, per probe, that person's share of the gallery (fellow eye included)."""
    sub = np.where(np.isfinite(D[np.ix_(probes, gallery)]), D[np.ix_(probes, gallery)], np.inf)
    nearest = gallery[np.argmin(sub, axis=1)]
    per_person = pd.Series(person[gallery]).value_counts()
    return {"rank1_same_item": float((label[nearest] == label[probes]).mean()),
            "rank1_same_person": float((person[nearest] == person[probes]).mean()),
            "gallery_size": int(len(gallery)), "n_probes": int(len(probes)), "chance_item": 1.0 / len(gallery),
            "chance_person": float(np.mean(per_person.reindex(person[probes]).fillna(0).to_numpy() / len(gallery))),
            "hit_item": label[nearest] == label[probes], "hit_person": person[nearest] == person[probes]}


def gallery_draws(D: np.ndarray, label: np.ndarray, U: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Common random numbers: per draw, the usable image with the smallest key U[draw, image] enrols each label with at
    least two usable images; all other usable images are probes. Cells with the same usable images get the same draws."""
    usable = np.isfinite(D).sum(1) > 1
    groups = [g for g in (np.flatnonzero(usable & (label == lab)) for lab in np.unique(label[usable])) if len(g) >= 2]
    out = []
    for u in U:
        gallery = np.array([g[np.argmin(u[g])] for g in groups])
        probes = np.concatenate([g[g != gi] for g, gi in zip(groups, gallery)])
        out.append((gallery, probes))
    return out


def rank1_averaged(D: np.ndarray, label: np.ndarray, person: np.ndarray, U: np.ndarray) -> dict:
    """Rank-1 averaged over common gallery draws, with per-probe hit rates for the person bootstrap."""
    hits, seen, draws = np.zeros(len(D)), np.zeros(len(D)), []
    for gallery, probes in gallery_draws(D, label, U):
        r = rank1(D, label, person, gallery, probes)
        hits[probes] += r.pop("hit_item")
        seen[probes] += 1
        r.pop("hit_person")
        draws.append(r)
    rates = pd.DataFrame(draws)
    probe = seen > 0
    return {"rank1_same_item_mean_over_draws": float(rates.rank1_same_item.mean()),
            "rank1_same_item_draw_interval95": [float(rates.rank1_same_item.quantile(q)) for q in (0.025, 0.975)],
            "rank1_same_person_mean_over_draws": float(rates.rank1_same_person.mean()),
            "chance_item": float(rates.chance_item.mean()), "chance_person": float(rates.chance_person.mean()),
            "gallery_size": int(rates.gallery_size.iloc[0]), "n_draws": int(len(rates)),
            "_probe_rate": np.where(probe, hits / np.maximum(seen, 1), np.nan)}


# ---------- weighted statistics for the person-cluster bootstrap ----------

def _ok(x, w):
    x = np.asarray(x, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    m = np.isfinite(x) & (w > 0)
    return x[m], w[m]


def wmean(x, w=None) -> float:
    x, w = _ok(x, w)
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def wmedian(x, w=None) -> float:
    """Median of the sample expanded by integer weights (equals np.median for unit weights)."""
    x, w = _ok(x, w)
    if not len(x):
        return float("nan")
    o = np.argsort(x, kind="stable")
    xs, cum = x[o], np.cumsum(np.round(w[o]))
    W = cum[-1]
    lo, hi = np.floor((W - 1) / 2), np.floor(W / 2)
    return float((xs[np.searchsorted(cum, lo + 1)] + xs[np.searchsorted(cum, hi + 1)]) / 2)


def wauc(a, wa, b, wb) -> float:
    """Weighted P(a > b) + half the ties."""
    a, wa = _ok(a, wa)
    b, wb = _ok(b, wb)
    if not len(a) or not len(b):
        return float("nan")
    o = np.argsort(b, kind="stable")
    bs, cum = b[o], np.concatenate([[0.0], np.cumsum(wb[o])])
    left, right = np.searchsorted(bs, a, "left"), np.searchsorted(bs, a, "right")
    return float((wa * (cum[left] + 0.5 * (cum[right] - cum[left]))).sum() / (wa.sum() * cum[-1]))


def weer(g, wg, im, wi) -> float:
    """Weighted equal error rate for distances (genuine expected smaller)."""
    g, wg = _ok(g, wg)
    im, wi = _ok(im, wi)
    if not len(g) or not len(im):
        return float("nan")
    og, oi = np.argsort(g, kind="stable"), np.argsort(im, kind="stable")
    gs, cg = g[og], np.concatenate([[0.0], np.cumsum(wg[og])])
    is_, ci = im[oi], np.concatenate([[0.0], np.cumsum(wi[oi])])
    thr = np.unique(np.concatenate([gs, is_]))
    fnmr = 1 - cg[np.searchsorted(gs, thr, "right")] / cg[-1]
    fmr = ci[np.searchsorted(is_, thr, "right")] / ci[-1]
    k = int(np.argmin(np.abs(fnmr - fmr)))
    return float((fnmr[k] + fmr[k]) / 2)


def wdprime(g, wg, im, wi) -> float:
    """Weighted signed d' with frequency-weight variances (equals the ddof=1 version for unit weights)."""
    g, wg = _ok(g, wg)
    im, wi = _ok(im, wi)
    if wg.sum() <= 1 or wi.sum() <= 1:
        return float("nan")
    mg, mi = (g * wg).sum() / wg.sum(), (im * wi).sum() / wi.sum()
    vg, vi = (wg * (g - mg) ** 2).sum() / (wg.sum() - 1), (wi * (im - mi) ** 2).sum() / (wi.sum() - 1)
    return float((mi - mg) / np.sqrt((vg + vi) / 2))


def weer_stratified(d_g, w_g, s_g, d_i, w_i, s_i) -> float:
    """EER within each stratum (genuine and impostor pairs of the same stratum), averaged with the genuine weights."""
    vals, ws = [], []
    for s in np.unique(s_g):
        g, i = s_g == s, s_i == s
        if g.any() and i.any() and np.nansum(w_g[g]) > 0 and np.nansum(w_i[i]) > 0:
            vals.append(weer(d_g[g], w_g[g], d_i[i], w_i[i]))
            ws.append(np.nansum(w_g[g]))
    return wmean(vals, ws)


class Calibrator:
    """c of query distances against a weighted stranger reference, per stratum; reference weights w[pa] * w[pb].
    With unit weights this is calibrated_score per stratum (the reference contains its own pairs, self-ties count half)."""

    def __init__(self, q_dist, q_stratum, r_dist, r_stratum, r_pa, r_pb, min_ref: int = 100):
        self.n, self.parts = len(q_dist), []
        q_dist, r_dist = np.asarray(q_dist, float), np.asarray(r_dist, float)
        for s in np.unique(q_stratum):
            qi, ri = np.flatnonzero(q_stratum == s), np.flatnonzero(r_stratum == s)
            if len(ri) < min_ref:
                raise ValueError(f"too few stranger pairs for a calibration reference: {len(ri)}")
            order = ri[np.argsort(r_dist[ri], kind="stable")]
            rs = r_dist[order]
            self.parts.append((qi, r_pa[order], r_pb[order], np.searchsorted(rs, q_dist[qi], "left"), np.searchsorted(rs, q_dist[qi], "right")))

    def c(self, w=None) -> np.ndarray:
        out = np.full(self.n, np.nan)
        for qi, pa, pb, left, right in self.parts:
            rw = np.ones(len(pa)) if w is None else w[pa] * w[pb]
            cum = np.concatenate([[0.0], np.cumsum(rw)])
            if cum[-1] > 0:
                out[qi] = (cum[-1] - cum[right] + 0.5 * (cum[right] - cum[left])) / cum[-1]
        return out


def c_against(d_query, r_dist, r_pa, r_pb, w=None) -> np.ndarray:
    """c of query distances against one fixed reference (a single stratum)."""
    return Calibrator(d_query, np.zeros(len(d_query)), r_dist, np.zeros(len(r_dist)), r_pa, r_pb, min_ref=1).c(w)


def t3b_c_scale(c_l0, w_l0, c_l1, w_l1) -> float:
    """(c_L0 - c_L1) / (c_L0 - 0.5), both scored against the same L4 reference, whose own mean c is 0.5."""
    m0 = wmean(c_l0, w_l0)
    return float((m0 - wmean(c_l1, w_l1)) / (m0 - 0.5))


class Discriminability:
    """Weighted discriminability: anchors and partners weighted by w[person of anchor], between distances by w[other person]."""

    def __init__(self, D: np.ndarray, item: np.ndarray, pcode: np.ndarray):
        n = len(D)
        rows, ai, left, right = [], [], [], []
        for i in range(n):
            b = np.flatnonzero((pcode != pcode[i]) & np.isfinite(D[i]))
            o = b[np.argsort(D[i, b], kind="stable")]
            rows.append(o)
            within = np.flatnonzero((item == item[i]) & np.isfinite(D[i]))
            within = within[within != i]
            if len(within) and len(o):
                bs = D[i, o]
                ai.extend([i] * len(within))
                left.extend(np.searchsorted(bs, D[i, within], "left"))
                right.extend(np.searchsorted(bs, D[i, within], "right"))
        m = max(len(r) for r in rows)
        self.person = np.full((n, m), -1)
        for i, o in enumerate(rows):
            self.person[i, :len(o)] = pcode[o]
        self.ai, self.left, self.right, self.pcode = np.array(ai), np.array(left), np.array(right), pcode

    def value(self, w=None) -> float:
        if not len(self.ai):
            return float("nan")
        w = np.ones(self.pcode.max() + 1) if w is None else w
        wext = np.append(w, 0.0)
        cum = np.concatenate([np.zeros((len(self.person), 1)), np.cumsum(wext[self.person], axis=1)], axis=1)
        tot = cum[self.ai, -1]
        cr, cl = cum[self.ai, self.right], cum[self.ai, self.left]
        with np.errstate(invalid="ignore", divide="ignore"):
            place = (tot - cr + 0.5 * (cr - cl)) / tot
        return wmean(place, w[self.pcode[self.ai]])


def person_bootstrap(stat_fn, n_persons: int, n_boot: int, rng: np.random.Generator) -> dict:
    """Point estimates with unit weights and percentile intervals over person-resampling replicates."""
    point = stat_fn(np.ones(n_persons))
    reps = {k: [] for k in point}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_persons, n_persons), minlength=n_persons).astype(float)
        s = stat_fn(w)
        for k in point:
            reps[k].append(s.get(k, np.nan))
    out = {}
    for k, v in point.items():
        r = np.asarray(reps[k], float)
        fin = r[np.isfinite(r)]
        out[k] = {"value": float(v), "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) else [None, None],
                  "n_finite_replicates": int(len(fin))}
    return out


def pairs_needed(ci: list, n_persons: int, n_pairs: int, target: float = CI_HALF_WIDTH_TARGET) -> dict:
    """Persons and pairs needed for a 95% CI half-width of `target`, scaling the bootstrap half-width by 1/sqrt(n)."""
    if ci[0] is None:
        return {"persons_needed": None, "pairs_needed": None}
    factor = ((ci[1] - ci[0]) / 2 / target) ** 2
    return {"current_half_width": float((ci[1] - ci[0]) / 2), "persons_needed": int(np.ceil(n_persons * factor)),
            "pairs_needed": int(np.ceil(n_pairs * factor))}


# ---------- iris cells ----------

def iris_pairs(t: pd.DataFrame, D: np.ndarray, Dm: np.ndarray | None, pcode: np.ndarray) -> pd.DataFrame:
    """The pair table of pairs_from_matrices (same rows, same order) plus both persons' codes and a same-item flag."""
    iu, ju = np.triu_indices(len(t), 1)
    eye = t.eye.to_numpy()
    p = pairs_from_matrices(t, D, Dm)
    keep = np.isfinite(D[iu, ju] if Dm is None else np.where(eye[iu] != eye[ju], Dm[iu, ju], D[iu, ju]))
    p["pa"], p["pb"] = pcode[iu[keep]], pcode[ju[keep]]
    p["same_item"] = (p.pa == p.pb).to_numpy() & ~p.opposite_eye.to_numpy()
    p["stratum"] = p.sensor_pair + " | " + p.opposite_eye.astype(str)
    return p


def matched_z(p: pd.DataFrame, strata: list[str], stranger: str) -> pd.Series:
    z = pd.Series(np.nan, index=p.index)
    for _, g in p.groupby(strata):
        ref = g.loc[g.rung == stranger, "distance"]
        if len(ref) > 1:
            z.loc[g.index] = (ref.mean() - g.distance) / ref.std(ddof=1)
    return z


def iris_cell(t: pd.DataFrame, D: np.ndarray, Dm: np.ndarray, U: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    pcode = pd.factorize(t.person)[0]
    p = iris_pairs(t, D, Dm, pcode)
    ref = (p.rung == IRIS_L4).to_numpy()
    d, rung, pa, pb = p.distance.to_numpy(), p.rung.to_numpy(), p.pa.to_numpy(), p.pb.to_numpy()
    strat, opp, sp = p.stratum.to_numpy(), p.opposite_eye.to_numpy(), p.sensor_pair.to_numpy()
    cal = Calibrator(d, strat, d[ref], strat[ref], pa[ref], pb[ref])
    c0 = cal.c()
    assert np.allclose(c0, calibrate(pairs_from_matrices(t, D, Dm)).c.to_numpy()), "weighted calibration departs from calibrate()"
    p["z"] = matched_z(p, ["sensor_pair", "opposite_eye"], IRIS_L4)
    plain = calibrate(pairs_from_matrices(t, D, None))
    present = [r for r in (IRIS_L0, IRIS_L1, IRIS_MIRROR, IRIS_L4) if (rung == r).any()]
    adj = list(zip(present, present[1:])) + [(IRIS_L0, IRIS_L4)]
    R_ = {r: rung == r for r in present}
    gen = R_[IRIS_L0]
    imp = (rung == IRIS_L4) & ~opp
    imp_matched = imp & np.isin(sp, np.unique(sp[gen]))
    item = (t.person + "_" + t.eye).to_numpy()
    disc = Discriminability(D, item, pcode)
    iu, ju = np.triu_indices(len(D), 1)  # differential identifiability on the unflipped matrix, as in the unweighted version
    dd = D[iu, ju]
    fin = np.isfinite(dd)
    da, db = pcode[iu][fin], pcode[ju][fin]
    dd, same_it = dd[fin], (item[iu] == item[ju])[fin]
    betw = da != db
    r1 = rank1_averaged(D, item, t.person.to_numpy(), U)
    probe_rate, probe_w = r1.pop("_probe_rate"), pcode
    strata_of = {r: p.loc[rung == r, ["sensor_pair", "opposite_eye"]].drop_duplicates() for r in present if r != IRIS_L4}
    in_strata = {r: p[["sensor_pair", "opposite_eye"]].merge(s.assign(_k=1), how="left", on=["sensor_pair", "opposite_eye"])._k.notna().to_numpy() & ref
                 for r, s in strata_of.items()}
    cross = IRIS_L1 in present
    if cross:
        nn = (sp == "NIR + NIR")
        anchor = ref & ~opp & nn
        l0x, l1x = gen & nn, rung == IRIS_L1
        q = l0x | l1x
        cal_anchor = Calibrator(d[q], np.zeros(q.sum()), d[anchor], np.zeros(anchor.sum()), pa[anchor], pb[anchor])
        q_l0 = l0x[q]

    def stats(w):
        c = cal.c(w)
        wp = np.where(pa == pb, w[pa], w[pa] * w[pb])
        s = {f"mean_c|{r}": wmean(c[R_[r]], wp[R_[r]]) for r in present if r != IRIS_L4}
        for a, b in adj:
            s[f"adjacent_auc|{a} > {b}"] = wauc(c[R_[a]], wp[R_[a]], c[R_[b]], wp[R_[b]])
        s["eer_raw_pooled"] = weer(d[gen], wp[gen], d[imp], wp[imp])
        s["eer_raw_support_matched_single_threshold"] = weer(d[gen], wp[gen], d[imp_matched], wp[imp_matched])
        s["eer_raw_sensor_stratified"] = weer_stratified(d[gen], wp[gen], sp[gen], d[imp_matched], wp[imp_matched], sp[imp_matched])
        s["eer_on_c"] = weer(1 - c[gen], wp[gen], 1 - c[imp_matched], wp[imp_matched])
        s["dprime_raw_pooled"] = wdprime(d[gen], wp[gen], d[imp], wp[imp])
        s["dprime_raw_support_matched"] = wdprime(d[gen], wp[gen], d[imp_matched], wp[imp_matched])
        s["dprime_on_c"] = wdprime(1 - c[gen], wp[gen], 1 - c[imp_matched], wp[imp_matched])
        s["rank1_same_item_within_sitting"] = wmean(probe_rate, w[probe_w])
        s["discriminability_unmatched"] = disc.value(w)
        s["differential_identifiability_distance_units"] = wmean(dd[betw], w[da[betw]] * w[db[betw]]) - wmean(dd[same_it], w[da[same_it]])
        l0 = wmedian(d[gen], wp[gen])
        for r in strata_of:
            l4 = wmedian(d[in_strata[r]], wp[in_strata[r]])
            s[f"S_two_anchor|{r}"] = float(two_anchor_s(wmedian(d[R_[r]], wp[R_[r]]), l0, l4))
        if cross:
            l0n, l1m, l4n = wmedian(d[l0x], wp[l0x]), wmedian(d[l1x], wp[l1x]), wmedian(d[anchor], wp[anchor])
            s["T3b_raw_medians"] = (l1m - l0n) / (l4n - l0n)
            ca = cal_anchor.c(w)
            wq = wp[q]
            s["T3b_c_scale_same_anchor"] = t3b_c_scale(ca[q_l0], wq[q_l0], ca[~q_l0], wq[~q_l0])
            s["T3b_residual_on_nuisance_matched_scale"] = t3b_c_scale(c[l0x], wp[l0x], c[l1x], wp[l1x])
        return s

    boot = person_bootstrap(stats, int(pcode.max()) + 1, n_boot, rng)
    n_persons = int(pcode.max()) + 1
    out = {"rungs": {r: {"mean_c": boot[f"mean_c|{r}"], "n_pairs": int(R_[r].sum()), "n_people": int(len(np.unique(pa[R_[r]]))),
                         "median_z": float(p.loc[p.rung == r, "z"].median())} for r in present if r != IRIS_L4},
           "stranger_anchor": "strangers score 0.5 against their own matched reference by construction",
           "adjacent_auc_on_c": {f"{a} > {b}": {**boot[f"adjacent_auc|{a} > {b}"],
                                                 "T4_for_half_width_0.05": pairs_needed(boot[f"adjacent_auc|{a} > {b}"]["ci95"], n_persons, int(min((rung == a).sum(), (rung == b).sum())))}
                                 for a, b in adj},
           "same_eye_vs_strangers": {k: boot[k] for k in ("eer_raw_pooled", "eer_raw_support_matched_single_threshold", "eer_raw_sensor_stratified", "eer_on_c",
                                                           "dprime_raw_pooled", "dprime_raw_support_matched", "dprime_on_c")},
           "rank1_within_sitting": {**r1, "rank1_same_item_person_bootstrap": boot["rank1_same_item_within_sitting"],
                                    "note": "gallery and probes come from the same captures; the bootstrap interval holds the gallery fixed"},
           "discriminability_unmatched": boot["discriminability_unmatched"],
           "differential_identifiability_distance_units": boot["differential_identifiability_distance_units"],
           "S_two_anchor": {r: boot[f"S_two_anchor|{r}"] for r in strata_of}}
    if IRIS_MIRROR in present:
        out["rungs"][IRIS_MIRROR]["mean_c_plain_unflipped_point"] = float(plain.loc[plain.rung == IRIS_MIRROR, "c"].mean())
    if len(np.unique(sp[gen])) == 1:
        out["same_eye_vs_strangers"]["note"] = "one sensor: the pooled, support-matched and stratified raw EERs coincide by construction"
    else:
        out["same_eye_vs_strangers"]["note"] = "support-matched uses one raw threshold across the sensor strata present; the stratified EER thresholds within each stratum"
    if cross:
        out["T3b_cross_spectral"] = {"raw_medians": boot["T3b_raw_medians"], "c_scale_same_anchor": boot["T3b_c_scale_same_anchor"],
                                     "residual_on_nuisance_matched_scale": boot["T3b_residual_on_nuisance_matched_scale"],
                                     "L0": "same eye, NIR + NIR, same sitting", "L1": "same eye, NIR vs VIS, same sitting (cross-spectral device change)",
                                     "L4": "strangers, NIR + NIR, same eye side (raw medians and c_scale_same_anchor)",
                                     "L4_residual": "strangers on the pair's own sensor pair and eye relation (NIR + VIS for L1)"}
    return out


def iris_dataset(t: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> dict:
    t = t.reset_index(drop=True)
    X, Xf = guarded(features, t.path, what="feature extraction"), guarded(features, t.path, flip=True, what="feature extraction")
    E = exposure_features(X)
    Z, Zf = zscore_rows(X), zscore_rows(Xf)
    C, M, qc = guarded(encode, t.path, what="iris encoding")
    Cf, Mf, qcf = guarded(encode, t.path, flip=True, what="iris encoding")
    De = distance_matrix(E, E)
    cells = {"F0_exposure_only": (De, De),  # mean and SD are unchanged by a horizontal flip
             "F1_pixels_16x16": (distance_matrix(X, X), distance_matrix(X, Xf)),
             "F1_pixels_16x16_zscored": (distance_matrix(Z, Z), distance_matrix(Z, Zf)),
             "F2_iris_code": (hamming_matrix(C, M), hamming_matrix(C, M, Cf, Mf))}
    eye_id = (t.person + "_" + t.eye).to_numpy()
    U = np.random.default_rng(SEED + 1).random((N_GALLERY_DRAWS, len(t)))  # common gallery draws for every cell
    out = {"n_images": int(len(t)), "n_people": int(t.person.nunique()), "images_by_sensor": t.sensor.value_counts().to_dict(),
           "iris_code_qc_pass": float(qc.ok.mean()), "iris_code_qc_pass_flipped": float(qcf.ok.mean()), "bootstrap_replicates": n_boot,
           "cells": {k: iris_cell(t, D, Dm, U, n_boot, rng) for k, (D, Dm) in cells.items()}}
    sensors = t.sensor.unique()
    if len(sensors) == 1:
        out["icc1_F0_by_eye"] = {"mean_intensity": icc1(E[:, 0], eye_id), "sd_intensity": icc1(E[:, 1], eye_id)}
    else:
        out["icc1_F0_by_eye_per_sensor"] = {s: {"mean_intensity": icc1(E[t.sensor == s, 0], eye_id[t.sensor == s]),
                                                 "sd_intensity": icc1(E[t.sensor == s, 1], eye_id[t.sensor == s])} for s in sensors}
        out["icc1_F0_by_eye_sensor_adjusted"] = {"mean_intensity": icc1_sensor_adjusted(E[:, 0], eye_id, t.sensor),
                                                 "sd_intensity": icc1_sensor_adjusted(E[:, 1], eye_id, t.sensor),
                                                 "note": "sensor means removed before the one-way ICC; not comparable with single-sensor ICCs"}
    return out


# ---------- 2D donor-swap ----------

def strips(paths, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Rubber-sheet strips (n, 16, 128) and the iris-code QC flag, with the same QC rule as iris_codes.encode."""
    out, ok = [], []
    for k, p in enumerate(paths):
        img = guarded(load_grey, p, flip, what=f"image #{k}")
        try:
            s = segment(img)
            polar = normalise(img, s)
            _, v = log_gabor_code(polar)
            good = 0.15 <= s["ratio"] <= 0.75 and s["offset"] <= 0.25 and v.mean() >= MIN_VALID_FRACTION
        except ValueError:
            polar, good = np.zeros((16, 128)), False
        out.append(polar)
        ok.append(bool(good))
    return np.stack(out), np.array(ok)


def donor_plan(t: pd.DataFrame, ok: np.ndarray, ok_flip: np.ndarray, rng: np.random.Generator,
               n_same: int = N_SAME, n_fellow: int = N_FELLOW, n_strangers: int = N_STRANGERS) -> list[tuple[int, str, int]]:
    """(target, kind, donor) triples. Targets and donors all pass QC; flipped donors pass QC on the flipped image."""
    person, eye = t.person.to_numpy(), t.eye.to_numpy()
    plan = []
    for i in np.flatnonzero(ok):
        pools = {SAME: (np.flatnonzero(ok & (person == person[i]) & (eye == eye[i])), n_same),
                 "fellow_eye_flipped": (np.flatnonzero(ok_flip & (person == person[i]) & (eye != eye[i])), n_fellow),
                 "stranger_same_side": (np.flatnonzero(ok & (person != person[i]) & (eye == eye[i])), n_strangers),
                 "stranger_opposite_flipped": (np.flatnonzero(ok_flip & (person != person[i]) & (eye != eye[i])), n_strangers)}
        for kind, (pool, n) in pools.items():
            pool = pool[pool != i]
            for j in rng.choice(pool, min(n, len(pool)), replace=False):
                plan.append((int(i), kind, int(j)))
    return plan


def place(target: np.ndarray, donor: np.ndarray, mask: np.ndarray, match: bool = True, shifts: int = SHIFTS, mode: str | None = None) -> np.ndarray:
    """Fill the mask with donor tissue, rotated by the angular shift that best fits the visible tissue, then adapted to it:
    'sd' rescales to the visible mean and SD (match=True), 'raw' keeps the donor as it is (match=False), 'mean' shifts
    to the visible mean only, 'affine' applies the least-squares a * donor + b fit on the visible tissue."""
    mode = mode or ("sd" if match else "raw")
    vis = ~mask
    best, best_err = None, np.inf
    for s in range(-shifts, shifts + 1):
        d = np.roll(donor, s, axis=1)
        if mode == "sd":
            d = (d - d[vis].mean()) / max(d[vis].std(), 1e-9) * target[vis].std() + target[vis].mean()
        elif mode == "mean":
            d = d - d[vis].mean() + target[vis].mean()
        elif mode == "affine":  # least squares a * d + b on the visible tissue; a constant donor gets the mean shift
            dv, tv = d[vis], target[vis]
            var = dv.var()
            a = ((dv - dv.mean()) * (tv - tv.mean())).mean() / var if var > 1e-12 else 0.0
            d = a * (d - dv.mean()) + tv.mean() if var > 1e-12 else d - dv.mean() + tv.mean()
        err = np.mean((d[vis] - target[vis]) ** 2)
        if err < best_err:
            best, best_err = d, err
    fill = target.copy()
    fill[mask] = best[mask]
    return fill


def panel(fill: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """Official BraTS 2023 inpainting panel on a 2D strip: tensors [1, 1, H, W], normalisation from the voided target."""
    from inpainting.challenge_metrics_2023 import generate_metrics
    T = torch.tensor(target, dtype=torch.float32)[None, None]
    M = torch.tensor(mask)[None, None]
    return generate_metrics(prediction=torch.tensor(fill, dtype=torch.float32)[None, None], target=T, normalization_tensor=T * ~M, mask=M)


def scaling(target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(v_min, v_max) of the package's percentile normalisation (0.5th and 99.5th percentile of the voided target)."""
    v_min, v_max = np.percentile(target * ~mask, [0.5, 99.5])
    return max(float(v_min), 0.0), float(v_max)


def clipped_range(target: np.ndarray, mask: np.ndarray) -> float:
    """Range of the reference inside the mask after the package's clipping, on its [0, 1] scale: PSNR's data range."""
    v_min, v_max = scaling(target, mask)
    t = target[mask]
    return float(max(min(t.max(), v_max) - max(t.min(), v_min), 0.0) / (v_max - v_min))


def overshoot_pixels(mask: np.ndarray, rng: np.random.Generator, share: float = OVERSHOOT_SHARE) -> np.ndarray:
    """A fixed number of mask pixels, k = max(1, round(share * mask size)), as a boolean image."""
    idx = np.flatnonzero(mask)
    k = max(1, int(round(share * len(idx))))
    sel = np.zeros(mask.size, bool)
    sel[rng.choice(idx, k, replace=False)] = True
    return sel.reshape(mask.shape)


def truth_fills(target: np.ndarray, mask: np.ndarray, mean_strip: np.ndarray, stranger: np.ndarray | None, rng: np.random.Generator) -> dict:
    """Fills built from the truth (blur, gain, overshoot) or from the population template and a blurred stranger."""
    blur = lambda x, s: ndimage.gaussian_filter(x, s, mode=("nearest", "wrap"))  # noqa: E731
    out = {f"blur_truth_sigma{s:g}": np.where(mask, blur(target, s), target) for s in BLUR_SIGMAS}
    out["template_sd_matched"] = place(target, mean_strip, mask, mode="sd")
    out["template_raw"] = place(target, mean_strip, mask, mode="raw")
    out["template_mean_shift"] = place(target, mean_strip, mask, mode="mean")
    out["template_affine"] = place(target, mean_strip, mask, mode="affine")
    out["template_blur_sigma2"] = place(target, blur(mean_strip, 2.0), mask, mode="mean")
    if stranger is not None:
        out["stranger_blur_sigma2"] = place(target, blur(stranger, 2.0), mask, mode="mean")
    out["gain_1.1"] = np.where(mask, 1.1 * target, target)
    v_min, v_max = scaling(target, mask)
    hi = target[mask].max()
    sel = overshoot_pixels(mask, rng)
    if v_max > hi:  # room between the reference's in-mask maximum and the clipping level
        f = target.copy()
        f[sel] = (hi + v_max) / 2
        out["overshoot_within_clip"] = f
    f = target.copy()
    f[sel] = max(v_max, float(target.max())) + 0.5 * (v_max - v_min)
    out["overshoot_above_clip"] = f
    kept = {k: v for k, v in out.items() if not np.array_equal(v[mask], target[mask])}
    return kept, sorted(set(out) - set(kept))


def unclipped_rmse(fill: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    v_min, v_max = scaling(target, mask)
    return float(np.sqrt(np.mean(((fill[mask] - target[mask]) / (v_max - v_min)) ** 2)))


def h14_fills(target: np.ndarray, masks: dict, rng: np.random.Generator) -> dict:
    """One additive noise field (SD = H14_NOISE_SHARE of mask A's clipped intensity range) on the union of the masks;
    each mask's fill carries exactly these noise values inside it, so fill quality is the same where the masks overlap."""
    union = np.any(list(masks.values()), axis=0)
    first = next(iter(masks.values()))
    v_min, v_max = scaling(target, first)
    noise = rng.normal(0, H14_NOISE_SHARE * (v_max - v_min), target.shape)
    return {k: np.where(m & union, target + noise, target) for k, m in masks.items()}


def donor_swap(t: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = t.reset_index(drop=True)
    S, ok = strips(t.path)
    Sf, ok_flip = strips(t.path, flip=True)
    person = t.person.to_numpy()
    tot, n_tot = S[ok].sum(0), ok.sum()
    per_person = {p: (S[ok & (person == p)].sum(0), (ok & (person == p)).sum()) for p in np.unique(person)}
    plan = donor_plan(t, ok, ok_flip, rng)
    by_target = {}
    for i, kind, j in plan:
        by_target.setdefault(i, []).append((kind, j))
    masks = {}
    for mname, (rs, cs) in MASKS.items():
        m = np.zeros((16, 128), bool)
        m[rs, cs] = True
        masks[mname] = m
    rows, h14, dropped = [], [], {}
    for mname, mask in masks.items():
        for i, donors in by_target.items():
            target = S[i]
            ps, pn = per_person[person[i]]
            mean_strip = (tot - ps) / (n_tot - pn)
            base = {"target": i, "person": person[i], "mask": mname, "ref_range_clipped": clipped_range(target, mask)}
            for kind, j in donors:
                donor = Sf[j] if kind.endswith("flipped") else S[j]
                for match in (True, False):
                    rows.append({**base, "kind": kind if match else kind + "_raw_intensity", **panel(place(target, donor, mask, match=match), target, mask)})
            stranger = next((S[j] for kind, j in donors if kind == "stranger_same_side"), None)
            fills, gone = truth_fills(target, mask, mean_strip, stranger, rng)
            for k in gone:
                dropped[k] = dropped.get(k, 0) + 1
            for kind, fill in fills.items():
                r = {**base, "kind": kind, **panel(fill, target, mask)}
                if kind == "overshoot_above_clip":
                    r["rmse_unclipped"] = unclipped_rmse(fill, target, mask)
                rows.append(r)
    for i in by_target:
        for mname, fill in h14_fills(S[i], masks, rng).items():
            h14.append({"target": i, "person": person[i], "mask": mname, **panel(fill, S[i], masks[mname])})
    R = pd.DataFrame(rows)
    R.attrs.update(n_images=int(len(t)), strip_qc_pass=float(ok.mean()), strip_qc_pass_flipped=float(ok_flip.mean()),
                   n_targets=int(R.target.nunique()), n_people=int(R.person.nunique()), fills_dropped_equal_to_truth=dropped)
    return R, pd.DataFrame(h14)


def better(a, b, metric: str) -> np.ndarray:
    """1 where a scores better than b, 0.5 for ties, 0 otherwise."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(a == b, 0.5, (a > b) if metric in HIGHER_IS_BETTER else (a < b)).astype(float)


def ladder_c(R: pd.DataFrame, metric: str, ref_kind: str) -> pd.Series:
    """Per case (target and mask): the share of that case's ref_kind fills that score worse than each fill."""
    c = pd.Series(np.nan, index=R.index)
    for _, g in R.groupby(["target", "mask"]):
        ref = g.loc[g.kind == ref_kind, metric].to_numpy()
        ref = ref[np.isfinite(ref)]
        if len(ref):
            c.loc[g.index] = better(g[metric].to_numpy()[:, None], ref[None, :], metric).mean(1)
    return c


def win_rate(R: pd.DataFrame, metric: str, a: str, b: str) -> float:
    """Per case, the share of (a fill, b fill) pairs in which the a fill scores better; mean over cases with both."""
    rates = []
    for _, g in R.groupby(["target", "mask"]):
        va, vb = g.loc[g.kind == a, metric].to_numpy(), g.loc[g.kind == b, metric].to_numpy()
        if len(va) and len(vb):
            rates.append(better(va[:, None], vb[None, :], metric).mean())
    return float(np.mean(rates)) if rates else float("nan")


def nonfinite_counts(R: pd.DataFrame) -> dict:
    """Panel values that are NaN or infinite, per fill kind and metric (only non-zero counts)."""
    out = {}
    for k, g in R.groupby("kind"):
        n = {m: int((~np.isfinite(g[m].to_numpy(float))).sum()) for m in PANEL if m in g}
        n = {m: v for m, v in n.items() if v}
        if n:
            out[k] = n
    return out


def h12_verdict(shares: dict, cis: dict) -> str:
    """Supported if the raw or affine template beats stranger tissue on MSE and PSNR with c intervals above 0.5;
    not supported if it loses on both with intervals below 0.5; otherwise inconclusive."""
    for t in ("template_raw", "template_affine"):
        if all(shares[(t, m)] > 0.5 and cis[(t, m)][0] > 0.5 for m in ("mse", "psnr")):
            return f"supported ({t})"
    if all(shares[(t, m)] < 0.5 and cis[(t, m)][1] < 0.5 for t in ("template_raw", "template_affine") for m in ("mse", "psnr")):
        return "not supported: both templates lose to stranger tissue on MSE and PSNR"
    return "inconclusive"


def summarize_donor_swap(R: pd.DataFrame, H: pd.DataFrame, same_label: str) -> dict:
    out = {"n_images": R.attrs["n_images"], "strip_qc_pass": R.attrs["strip_qc_pass"], "strip_qc_pass_flipped": R.attrs["strip_qc_pass_flipped"],
           "n_targets": R.attrs["n_targets"], "n_people": R.attrs["n_people"], "same_eye_repeat_is": same_label,
           "fills_dropped_equal_to_truth": R.attrs["fills_dropped_equal_to_truth"],
           "nonfinite_panel_values": nonfinite_counts(R), "masks": {}}
    opposite = ("fellow_eye_flipped", "stranger_opposite_flipped")  # calibrated against opposite-side, flipped strangers
    ref_of = {"c_same": "stranger_same_side", "c_opp": "stranger_opposite_flipped",
              "c_same_raw": "stranger_same_side_raw_intensity", "c_opp_raw": "stranger_opposite_flipped_raw_intensity"}
    challengers = ("blur_truth_sigma2", *TEMPLATES)
    for mname, M in R.groupby("mask"):
        M = M.copy()
        res = {"panel_median_iqr": {k: {m: [float(np.nanquantile(g[m], q)) for q in (0.25, 0.5, 0.75)] for m in PANEL}
                                    for k, g in M.groupby("kind")}}
        ladder, shares, cis = {}, {}, {}
        for m in PANEL:
            for col, ref in ref_of.items():
                M[col] = ladder_c(M, m, ref)
            per_kind = {}
            for k, g in M.groupby("kind"):
                raw = k.endswith("_raw_intensity")
                col = ("c_opp" if k.replace("_raw_intensity", "") in opposite else "c_same") + ("_raw" if raw else "")
                vals = g[col].to_numpy()
                ci = cluster_ci(vals[np.isfinite(vals)], g.person.to_numpy()[np.isfinite(vals)], n_boot=300)
                per_kind[k] = {"mean_c": float(np.nanmean(vals)), "ci95": ci, "reference": ref_of[col],
                               **({"anchor_by_construction": True} if k == ref_of[col] else {})}
                if k in TEMPLATES:
                    cis[(k, m)] = ci
            wr = {f"share_{SAME}_beats_fellow_eye_flipped": win_rate(M, m, SAME, "fellow_eye_flipped"),
                  "share_fellow_eye_flipped_beats_stranger_opposite_flipped": win_rate(M, m, "fellow_eye_flipped", "stranger_opposite_flipped"),
                  f"share_{SAME}_beats_stranger_same_side": win_rate(M, m, SAME, "stranger_same_side")}
            for b in challengers:
                for a in ("stranger_same_side", "fellow_eye_flipped", SAME):
                    wr[f"share_{b}_beats_{a}"] = win_rate(M, m, b, a)
            for t_ in TEMPLATES:
                shares[(t_, m)] = wr.get(f"share_{t_}_beats_stranger_same_side", float("nan"))
            ladder[m] = {"mean_c_by_fill": per_kind, "win_rates": wr}
        res["ladder"] = ladder
        res["H12"] = {"verdict": h12_verdict(shares, cis),
                      "evidence": {t_: {m: {"share_beats_stranger_same_side": shares[(t_, m)], "mean_c_ci95": cis[(t_, m)]} for m in ("mse", "psnr", "ssim")}
                                   for t_ in TEMPLATES if (t_, "mse") in cis},
                      "note": "blur_truth fills keep the target's own identity and are not H12 evidence; template_sd_matched inflates the template to the target's SD"}
        taus = {m: [] for m in PANEL if m != "rmse"}
        identical = {m: [] for m in taus}
        for _, g in M.groupby("target"):
            for m in taus:
                v = g[m].to_numpy() if m in HIGHER_IS_BETTER else -g[m].to_numpy()
                ok = np.isfinite(v) & np.isfinite(g.rmse.to_numpy())
                tau = kendalltau(-g.rmse.to_numpy()[ok], v[ok]).statistic
                taus[m].append(tau)
                identical[m].append(bool(np.isclose(tau, 1.0)))
        res["H11_per_case_kendall_tau_with_rmse"] = {m: {"median": float(np.nanmedian(v)), "share_identical_order": float(np.mean(identical[m]))}
                                                      for m, v in taus.items()}
        g = M[M.kind == "gain_1.1"]
        res["H13_gain_fill_spearman_with_clipped_in_mask_range"] = {m: float(spearmanr(g.ref_range_clipped, g[m]).statistic) for m in ("psnr", "psnr_eps", "psnr_01", "rmse")}
        w = M[M.kind == "overshoot_within_clip"]
        a = M[M.kind == "overshoot_above_clip"]
        ratio = (a.rmse / a.rmse_unclipped).to_numpy()
        res["H15_overshoot"] = {"within_clip_n_cases": int(len(w)),
                                "within_clip_psnr_minus_psnr_eps_db_median": float((w.psnr - w.psnr_eps).median()) if len(w) else None,
                                "above_clip_rmse_reported_over_unclipped_median": float(np.nanmedian(ratio)),
                                "above_clip_ratio_nonfinite": int((~np.isfinite(ratio)).sum()),
                                "above_clip_mean_c_rmse_vs_strangers": ladder["rmse"]["mean_c_by_fill"]["overshoot_above_clip"]["mean_c"],
                                "overshoot_pixels": "fixed k = max(1, round(0.10 * mask size)) per case"}
        res["T2_anchor_medians"] = {k: {m: float(np.nanmedian(M.loc[M.kind == k, m])) for m in ("ssim", "rmse", "psnr", "psnr_01", "mae")}
                                    for k in (SAME, "stranger_same_side", "template_raw", "blur_truth_sigma2")}
        out["masks"][mname] = res
    a, b = (out["masks"][k] for k in MASKS)
    out["H14_mask_size_unpaired_medians"] = {k: {"ssim_median_A": a["T2_anchor_medians"][k]["ssim"], "ssim_median_B": b["T2_anchor_medians"][k]["ssim"]}
                                             for k in (SAME, "stranger_same_side", "template_raw", "blur_truth_sigma2")}
    kA, kB = list(MASKS)
    P = H.pivot_table(index=["target", "person"], columns="mask", values=["ssim", "rmse"]).dropna()
    persons = P.index.get_level_values("person").to_numpy()
    dif = {m: (P[(m, kB)] - P[(m, kA)]).to_numpy() for m in ("ssim", "rmse")}
    out["H14_paired_constant_noise"] = {"noise_sd_share_of_clipped_range": H14_NOISE_SHARE, "n_cases": int(len(P)),
                                        **{f"{m}_B_minus_A": {"median": float(np.median(v)), "mean_ci95": cluster_ci(v, persons, n_boot=300),
                                                              "share_B_higher": float((v > 0).mean())} for m, v in dif.items()},
                                        "note": "identical noise values where the masks overlap; SSIM changes while RMSE stays at the same noise level"}
    return out


# ---------- GRAPE ----------

def grape_pairs(table: pd.DataFrame, feats: dict, flipped: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """grape_calibration.pairs (same rows, same order) plus both patients' codes; also returns the image table used."""
    t = table[table["filename"].isin(feats)].reset_index(drop=True)
    X = np.stack([feats[f] for f in t["filename"]])
    D = distance_matrix(X, X)
    Dm = distance_matrix(X, np.stack([flipped[f] for f in t["filename"]])) if flipped is not None else None
    iu, ju = np.triu_indices(len(t), 1)
    pat, eye, cam = (t[c].to_numpy() for c in ("patient", "eye", "device"))
    code = pd.factorize(t.patient)[0]
    yrs = t["interval_years"].astype(float).to_numpy()
    same_pat, same_eye = pat[iu] == pat[ju], eye[iu] == eye[ju]
    same_vis = np.abs(yrs[iu] - yrs[ju]) < 1e-6
    rung = np.where(~same_pat, "other_patient",
                    np.where(same_eye, np.where(same_vis, "same_eye_same_visit", "same_eye_other_visit"),
                             np.where(same_vis, "fellow_eye_same_visit", "fellow_eye_other_visit")))
    dist = D[iu, ju] if Dm is None else np.where(~same_eye, Dm[iu, ju], D[iu, ju])
    device_pair = np.array([" + ".join(sorted((a, b))) for a, b in zip(cam[iu], cam[ju])])
    p = pd.DataFrame({"rung": rung, "same_camera": cam[iu] == cam[ju], "device_pair": device_pair, "opposite_eye": ~same_eye,
                      "patient": np.where(same_pat, pat[iu], ""), "pa": code[iu], "pb": code[ju], "distance": dist})
    p["stratum"] = p.device_pair + " | " + p.opposite_eye.astype(str)
    return p, t.assign(code=code)


def grape(rng: np.random.Generator, n_boot: int) -> dict:
    import reid
    from grape_calibration import calibrate as grape_calibrate, pairs as reference_pairs
    from mirror_and_interval import flipped_pixel_features
    from run_grape import EXT, IMG_DIR
    table = guarded(reid.load_table, os.path.join(EXT, "data", "GRAPE", "grape_img_info_w_file.csv"), what="GRAPE table")
    names = table.filename.tolist()
    px = guarded(reid.pixel_features, IMG_DIR, names, what="GRAPE pixel features")
    pxf = guarded(flipped_pixel_features, names, what="GRAPE flipped pixel features")
    rf = guarded(reid.load_features_safe, os.path.join(EXT, "features", "GRAPE"), what="RETFound features")
    expo = {n: exposure_features(v[None, :])[0] for n, v in px.items()}
    out = {"n_images": int(len(table)), "bootstrap_replicates": n_boot, "cells": {}}
    A, B, F = "same_eye_other_visit | same camera", "same_eye_other_visit | different camera", "fellow_eye_same_visit | same camera"
    for cell, feats, flipped in (("F0_exposure_only", expo, None), ("F1_pixels_16x16", px, None),
                                 ("F1_pixels_16x16_fellow_eye_flipped", px, pxf), ("F4_retfound_released", rf, None)):
        p, t = grape_pairs(table, feats, flipped)
        d, pa, pb, strat = p.distance.to_numpy(), p.pa.to_numpy(), p.pb.to_numpy(), p.stratum.to_numpy()
        ref = (p.rung == "other_patient").to_numpy()
        cal = Calibrator(d, strat, d[ref], strat[ref], pa[ref], pb[ref])
        assert np.allclose(cal.c(), grape_calibrate(reference_pairs(table, feats, flipped)).c.to_numpy()), "weighted calibration departs from grape_calibration"
        key = (p.rung + " | " + np.where(p.same_camera, "same", "different") + " camera").to_numpy()
        mA, mB, mF = key == A, key == B, key == F
        strangers = ref
        imp = ref & ~p.opposite_eye.to_numpy()
        gen = (p.rung == "same_eye_other_visit").to_numpy()
        anchor = imp & p.same_camera.to_numpy()  # the stated L4: strangers, same eye side, same camera (pooled over cameras)
        qAB = mA | mB
        cal_anchor = Calibrator(d[qAB], np.zeros(qAB.sum()), d[anchor], np.zeros(anchor.sum()), pa[anchor], pb[anchor])
        qA = mA[qAB]
        X = np.stack([feats[f] for f in t.filename])
        D = distance_matrix(X, X)
        item, person = (t.patient + "_" + t.eye).to_numpy(), t.patient.to_numpy()
        disc = Discriminability(D, item, t.code.to_numpy())
        yrs = t.interval_years.astype(float).to_numpy()
        gallery = np.array([g.index[np.argmin(yrs[g.index])] for _, g in t.groupby(item)])
        first = dict(zip(item[gallery], yrs[gallery]))
        probes = np.array([i for i in range(len(t)) if yrs[i] > first[item[i]] + 1e-6])
        r1 = rank1(D, item, person, gallery, probes)
        hit = r1.pop("hit_item").astype(float)
        r1.pop("hit_person")
        probe_code = t.code.to_numpy()[probes]
        rungs = [k for k in (A, B, F) if (key == k).any()]
        adj = [(A, B), (B, "other_patient"), (A, "other_patient")] + ([(F, "other_patient"), (A, F)] if F in rungs else [])
        KM = {k: key == k for k in rungs} | {"other_patient": strangers}

        def stats(w):
            c = cal.c(w)
            wp = np.where(pa == pb, w[pa], w[pa] * w[pb])
            s = {f"mean_c|{k}": wmean(c[KM[k]], wp[KM[k]]) for k in rungs}
            sel = KM.__getitem__
            for a, b in adj:
                s[f"adjacent_auc|{a} > {b}"] = wauc(c[sel(a)], wp[sel(a)], c[sel(b)], wp[sel(b)])
            s["eer_raw_pooled"] = weer(d[gen], wp[gen], d[imp], wp[imp])
            s["eer_raw_camera_pair_stratified"] = weer_stratified(d[gen], wp[gen], strat[gen], d[imp], wp[imp], strat[imp])
            s["eer_on_c"] = weer(1 - c[gen], wp[gen], 1 - c[imp], wp[imp])
            s["dprime_raw_pooled"] = wdprime(d[gen], wp[gen], d[imp], wp[imp])
            s["dprime_on_c"] = wdprime(1 - c[gen], wp[gen], 1 - c[imp], wp[imp])
            s["rank1_same_item_cross_visit"] = wmean(hit, w[probe_code])
            s["discriminability_unmatched_across_visits"] = disc.value(w)
            base_d, l1_d, l4_d = wmedian(d[mA], wp[mA]), wmedian(d[mB], wp[mB]), wmedian(d[anchor], wp[anchor])
            s["T3b_raw_medians"] = (l1_d - base_d) / (l4_d - base_d)
            ca, wq = cal_anchor.c(w), wp[qAB]
            s["T3b_c_scale_same_anchor"] = t3b_c_scale(ca[qA], wq[qA], ca[~qA], wq[~qA])
            s["T3b_residual_camera_cost_on_nuisance_matched_scale"] = t3b_c_scale(c[mA], wp[mA], c[mB], wp[mB])
            return s

        boot = person_bootstrap(stats, int(t.code.max()) + 1, n_boot, rng)
        n_persons = int(t.code.max()) + 1
        out["cells"][cell] = {
            "rungs": {k: {"mean_c": boot[f"mean_c|{k}"], "n_pairs": int(KM[k].sum())} for k in rungs},
            "stranger_anchor": "strangers score 0.5 against their own matched reference by construction",
            "adjacent_auc_on_c": {f"{a} > {b}": {**boot[f"adjacent_auc|{a} > {b}"],
                                                  "T4_for_half_width_0.05": pairs_needed(boot[f"adjacent_auc|{a} > {b}"]["ci95"], n_persons,
                                                                                         int(min(KM[a].sum(), KM[b].sum())))}
                                  for a, b in adj},
            "same_eye_other_visit_vs_strangers": {k: boot[k] for k in ("eer_raw_pooled", "eer_raw_camera_pair_stratified", "eer_on_c", "dprime_raw_pooled", "dprime_on_c")},
            "rank1_cross_visit": {**r1, "rank1_same_item_person_bootstrap": boot["rank1_same_item_cross_visit"],
                                  "note": "gallery = earliest visit of each eye, probes = later visits; the interval holds the gallery fixed"},
            "discriminability_unmatched_across_visits": boot["discriminability_unmatched_across_visits"],
            "T3b_camera": {"raw_medians": boot["T3b_raw_medians"], "c_scale_same_anchor": boot["T3b_c_scale_same_anchor"],
                           "residual_camera_cost_on_nuisance_matched_scale": boot["T3b_residual_camera_cost_on_nuisance_matched_scale"],
                           "base": "same eye, other visit, same camera (GRAPE has no same-session repeats)", "L1": "same eye, other visit, other camera",
                           "L4": "strangers, same eye side, same camera, pooled over cameras (raw medians and c_scale_same_anchor)",
                           "L4_residual": "strangers on the pair's own camera pair and eye relation"}}
        if cell == "F0_exposure_only":
            out["icc1_F0_by_eye_across_visits"] = {"mean_intensity": icc1([expo[f][0] for f in t.filename], item),
                                                   "sd_intensity": icc1([expo[f][1] for f in t.filename], item)}
    out["limitation"] = "No pixel BraTS panel for GRAPE: image pairs are not registered, so a voxel-wise panel would measure misalignment."
    return out


HYPOTHESES = {
    "H7_T2_anchors": "raw panel and distance values at L0 and L4 across masks and datasets (donor_swap T2_anchor_medians; iris anchors; differential identifiability in cell units)",
    "H8_T3a_negative_control": "F0 exposure-only cell against F1 pixels (iris cells, GRAPE cells)",
    "H9_T3c_matched_vs_pooled": "pooled, support-matched, stratified and on-c EER and d' (informative only in UTIRIS NIR + VIS and GRAPE; in one-sensor sets the raw EERs coincide)",
    "H10_T3b_ratio": "T3b_cross_spectral (UTIRIS NIR + VIS, a cross-spectral device change within one sitting) and GRAPE T3b_camera, raw and on the c scale",
    "H11_redundancy": "donor_swap H11_per_case_kendall_tau_with_rmse",
    "H12_mean_fill_beats_stranger": "donor_swap H12 verdict from template_raw and template_affine against stranger tissue (MSE, PSNR)",
    "H13_psnr_range": "donor_swap H13_gain_fill_spearman_with_clipped_in_mask_range",
    "H14_mask_size": "donor_swap H14_paired_constant_noise (primary) and H14_mask_size_unpaired_medians",
    "H15_clipping": "donor_swap H15_overshoot",
}
NOT_COVERED = ["normalised cross-correlation", "mutual information", "cosine distance (RETFound uses Euclidean distance)",
               "T1 inversion counts", "T5 noise, compression and sub-pixel shift degradations", "BraTS panel on the F0-F2 cells",
               "T6 cross-modality comparison", "T7 method rankings"]


def main():
    t0 = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    rng = np.random.default_rng(SEED)
    up = upol_table()
    ut, conflicts = utiris_table()
    nir = ut[ut.sensor == "NIR"]
    nb = N_BOOT["default"]
    from importlib.metadata import version
    out = {"seed": SEED, "versions": {p: version(p) for p in ("torchmetrics", "torch", "inpainting")},
           "coverage": {"section": "partial Section 9.3", "not_covered": NOT_COVERED},
           "bootstrap": {"method": "person-cluster; pairs weighted by person multiplicities so the stranger reference is rebuilt in every replicate",
                         "replicates": N_BOOT, "gallery_draws": N_GALLERY_DRAWS},
           "masks_rows_cols": {k: [v[0].start, v[0].stop, v[1].start, v[1].stop] for k, v in MASKS.items()},
           "donors_per_target": {SAME: N_SAME, "fellow_eye_flipped": N_FELLOW, "stranger_same_side": N_STRANGERS, "stranger_opposite_flipped": N_STRANGERS,
                                 "note": "donors sampled without replacement with the fixed seed; strangers are a sample, not all strangers"},
           "iris": {"UTIRIS_NIR": {"excluded_eye_label_conflicts": conflicts, "same_eye_pairs_are": "captures from one sitting", **iris_dataset(nir, rng, nb)},
                    "UPOL_secondary": {"same_eye_pairs_are": "repeat captures, session undocumented", **iris_dataset(up, rng, nb)},
                    "UTIRIS_NIR_VIS_cross_spectral": {"same_eye_pairs_are": "captures from one sitting", **iris_dataset(ut, rng, N_BOOT["UTIRIS_NIR_VIS_cross_spectral"])}},
           "donor_swap": {"UTIRIS_NIR": summarize_donor_swap(*donor_swap(nir, rng), "captures from one sitting (not a re-scan)"),
                          "UPOL_secondary": summarize_donor_swap(*donor_swap(up, rng), "repeat captures, session undocumented (not a re-scan)")},
           "grape": grape(rng, nb),
           "hypotheses": HYPOTHESES,
           "caveats": ["UTIRIS same-eye pairs are captures from one sitting; its NIR vs VIS rung is a cross-spectral device change within that sitting, not a re-test over days.",
                       "UPOL same-eye pairs are repeat captures with the session undocumented; UPOL iris-code QC passes for about a fifth of images; UPOL results are secondary.",
                       "Rank-1 for iris is within one sitting; for GRAPE it is across visits.",
                       "Strangers score c = 0.5 against their own reference by construction; that value is an anchor, not a result.",
                       "Fellow-eye fills come from the flipped image of the fellow eye; flipped strips are re-segmented, so QC is applied to them separately.",
                       "Donor-swap donors are aligned by angular shift on the visible tissue and, unless marked raw_intensity, rescaled to its mean and SD.",
                       "blur_truth fills blur the true strip and keep the target's identity; templates and the blurred stranger carry none of it.",
                       "Per case, c of a fill is computed against that case's 8 stranger fills, so per-case c is coarse; means over cases are not.",
                       "Iris code has about 35 degrees of freedom (audit 2026-11-01), far below Daugman's systems.",
                       "GRAPE has no same-session repeats; its base rung is the same eye at another visit, and camera change is confounded with time."]}
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(PROJECT, "results", "baseline_v0.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("written results/baseline_v0.json in", out["runtime_s"], "s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
