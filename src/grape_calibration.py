"""Random-pair calibration of identity distances on GRAPE fundus photographs (2D pilot of the metric).

A raw distance has no meaning across feature spaces: pixels and RETFound live on different scales. Following the
random-pair calibration of TM-score, every pair distance d is mapped to
    c(d) = P(d_ref > d),   d_ref drawn from unrelated-person pairs that share the pair's nuisance condition
(the same unordered pair of cameras and the same eye relation; for opposite eyes, also the same mirroring).
c = 1: closer than every stranger;
c = 0.5: as close as a typical stranger. The mean of c over a rung is the AUC of that rung against strangers, so
rungs and feature spaces can be compared on one unitless scale. Tolerance curve: the share of a rung with c >= tau.

Rungs: same eye at another visit (L2, with or without a camera change = L1), fellow eye at the same visit and at
another visit (mirror half), other patient (L4, the reference). GRAPE has no same-session repeats, so L0 is absent.
A visit is an acquisition date (Interval Years), not GRAPE's Visit Number, which is counted per eye. The audit of
2026-11-01 showed that a binary same/different-camera reference pools camera pairs whose stranger distances differ
widely, and that camera change is confounded with time (cross-camera pairs are about twice as far apart in time);
the reference is therefore stratified by the camera pair, and same-eye results are also reported by time gap.
Mirror: pixel features of the fellow eye are also computed after a horizontal flip; RETFound features are the
released ones and cannot be recomputed for flipped images.

Aggregate output only: results/grape_calibration.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import reid  # noqa: E402
from run_grape import EXT, IMG_DIR, PROJECT  # noqa: E402

TAUS = (0.5, 0.9, 0.95, 0.99)
N_BOOT = 1000


def calibrated_score(d: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """c(d) = share of reference distances strictly greater than d, plus half the ties."""
    ref = np.sort(np.asarray(ref, float))
    d = np.asarray(d, float)
    above = len(ref) - np.searchsorted(ref, d, side="right")
    ties = np.searchsorted(ref, d, side="right") - np.searchsorted(ref, d, side="left")
    return (above + 0.5 * ties) / len(ref)


def distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    sa, sb = (A * A).sum(1), (B * B).sum(1)
    return np.sqrt(np.maximum(sa[:, None] + sb[None, :] - 2 * A @ B.T, 0))


def pairs(table: pd.DataFrame, feats: dict, flipped: dict | None = None) -> pd.DataFrame:
    """Every unordered image pair with rung, camera relation, eye relation, gap and distance.

    With flipped features, opposite-eye pairs use the distance between one image and the other one flipped
    (an isometry, so the choice of which image to flip does not matter).
    """
    t = table[table["filename"].isin(feats)].reset_index(drop=True)
    X = np.stack([feats[f] for f in t["filename"]])
    D = distance_matrix(X, X)
    if flipped is not None:
        Dm = distance_matrix(X, np.stack([flipped[f] for f in t["filename"]]))
    iu, ju = np.triu_indices(len(t), 1)
    pat, eye, cam = (t[c].to_numpy() for c in ("patient", "eye", "device"))
    yrs = t["interval_years"].astype(float).to_numpy()
    same_pat, same_eye = pat[iu] == pat[ju], eye[iu] == eye[ju]
    same_vis = np.abs(yrs[iu] - yrs[ju]) < 1e-6  # same acquisition date
    rung = np.where(~same_pat, "other_patient",
                    np.where(same_eye, np.where(same_vis, "same_eye_same_visit", "same_eye_other_visit"),
                             np.where(same_vis, "fellow_eye_same_visit", "fellow_eye_other_visit")))
    dist = D[iu, ju]
    if flipped is not None:
        dist = np.where(~same_eye, Dm[iu, ju], dist)
    device_pair = np.array([" + ".join(sorted((a, b))) for a, b in zip(cam[iu], cam[ju])])
    return pd.DataFrame({"rung": rung, "same_camera": cam[iu] == cam[ju], "device_pair": device_pair, "opposite_eye": ~same_eye,
                         "patient": np.where(same_pat, pat[iu], ""), "gap_years": np.abs(yrs[iu] - yrs[ju]),
                         "distance": dist})


def calibrate(p: pd.DataFrame) -> pd.DataFrame:
    """Add c to every pair, with the reference drawn from strangers with the same camera pair and eye relation."""
    p = p.copy()
    p["c"] = np.nan
    for (cam, opp), g in p.groupby(["device_pair", "opposite_eye"]):
        ref = g.loc[g.rung == "other_patient", "distance"].to_numpy()
        if len(ref) < 100:
            raise ValueError(f"too few reference pairs for cameras={cam}, opposite_eye={opp}: {len(ref)}")
        p.loc[g.index, "c"] = calibrated_score(g["distance"].to_numpy(), ref)
    return p


def cluster_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    by = {c: values[clusters == c] for c in ids}
    stats = [np.concatenate([by[c] for c in rng.choice(ids, len(ids))]).mean() for _ in range(n_boot)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def summarize(p: pd.DataFrame) -> dict:
    out = {}
    for (rung, cam), g in p[p.rung != "other_patient"].groupby(["rung", "same_camera"]):
        if len(g) < 5:
            continue
        key = f"{rung} | {'same' if cam else 'different'} camera"
        out[key] = {"n_pairs": int(len(g)), "n_patients": int(g.patient.nunique()),
                    "mean_c_auc": float(g.c.mean()), "mean_c_ci95_patient_bootstrap": cluster_ci(g.c.to_numpy(), g.patient.to_numpy()),
                    "median_c": float(g.c.median()),
                    "tolerance_curve": {f"c>={t}": float((g.c >= t).mean()) for t in TAUS}}
    se = p[p.rung == "same_eye_other_visit"].copy()
    se["gap_bin"] = pd.cut(se.gap_years, [0, 1, 2, 10], labels=["<1y", "1-2y", ">2y"], include_lowest=True).astype(str)
    out["same_eye_c_by_gap_and_camera"] = {f"{gb} | {'same' if cam else 'different'} camera": {"n_pairs": int(len(g)), "mean_c": float(g.c.mean())}
                                           for (gb, cam), g in se.groupby(["gap_bin", "same_camera"]) if len(g) >= 10}
    out["same_eye_c_vs_gap_spearman"] = {("same" if cam else "different") + " camera": float(g[["gap_years", "c"]].corr("spearman").iloc[0, 1])
                                         for cam, g in se.groupby("same_camera")}
    return out


def main():
    table = reid.load_table(os.path.join(EXT, "data", "GRAPE", "grape_img_info_w_file.csv"))
    names = table["filename"].tolist()
    feats_rf = reid.load_features_safe(os.path.join(EXT, "features", "GRAPE"))
    feats_px = reid.pixel_features(IMG_DIR, names)
    from mirror_and_interval import flipped_pixel_features
    flipped_px = flipped_pixel_features(names)
    out = {"reference": "unrelated-patient pairs with the same unordered camera pair and eye relation",
           "visit_definition": "acquisition date (Interval Years)",
           "note": "all GRAPE images (631), not only the patients with more than one visit; no L0 rung in GRAPE",
           "feature_spaces": {}}
    for name, feats, flipped in (("retfound_released", feats_rf, None), ("pixels_16x16", feats_px, None),
                                 ("pixels_16x16_fellow_eye_flipped", feats_px, flipped_px)):
        p = calibrate(pairs(table, feats, flipped))
        out["feature_spaces"][name] = {"n_images": int(table.filename.isin(feats).sum()), **summarize(p)}
        print(name, json.dumps({k: (v["mean_c_auc"] if isinstance(v, dict) and "mean_c_auc" in v else v)
                                for k, v in out["feature_spaces"][name].items()}, indent=1))
    with open(os.path.join(PROJECT, "results", "grape_calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
