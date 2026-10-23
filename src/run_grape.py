"""Reproduce GRAPE re-identification (Nebbia et al. 2025; Engelmann et al. 2026) and split it by ladder rung.

Aggregate results -> results/grape_repro.json (versioned with the project).
Per-image matches (patient-level derived data) -> C:/Projects/data/projects/identity-ladder-repro/<run>/ (not versioned).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import reid  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
EXT = os.path.join(PROJECT, "external", "re_identification-50bf2ea", "src", "re_identification-50bf2ead64acf63f5071a4d947a11f6d4056070a")
DATA = r"C:\Projects\data\datasets\ophthalmology\grape\figshare-c6406319-v1"
IMG_DIR = os.path.join(DATA, "raw", "CFPs")
XLSX = os.path.join(DATA, "downloads", "VF and clinical information.xlsx")
RUN = "run-2026-10-23-grape"
PRIVATE = os.path.join(r"C:\Projects\data\projects\identity-ladder-repro", RUN)

PUBLISHED = {  # Table 2 of Nebbia et al. (RETFound, GRAPE) and Table 1 of Engelmann et al. (16x16 pixels), in %
    "retfound_released_features": {"image_R@1": 38.9, "image_R@10": 56.1, "image_AP": 83.5,
                                   "patient_R@1": 54.7, "patient_R@10": 72.6, "patient_AP": 83.6},
    "pixels_16x16": {"image_R@1": 25.13, "image_R@10": 36.46, "image_AP": 78.36,
                     "patient_R@1": 42.45, "patient_R@10": 59.43, "patient_AP": 70.55},
}


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list:
    """95% percentile interval for a mean, resampling patients (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    by = {c: values[clusters == c] for c in ids}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        stats.append(np.concatenate([by[c] for c in pick]).mean())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def check_table_against_spreadsheet(table: pd.DataFrame) -> dict:
    """Consistency of the authors' CSV with the official GRAPE spreadsheet (aggregate counts only)."""
    fu = pd.read_excel(XLSX, sheet_name="Follow-up", header=0, skiprows=[1]).rename(columns=lambda c: str(c).strip())
    fu = fu[fu["Corresponding CFP"].astype(str).str.endswith(".jpg")].copy()
    fu["filename"] = fu["Corresponding CFP"].astype(str).str.strip()
    merged = table.merge(fu, on="filename", how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"]
    return {
        "csv_rows": int(len(table)), "spreadsheet_images": int(len(fu)),
        "matched": int(len(both)), "only_csv": int((merged["_merge"] == "left_only").sum()), "only_spreadsheet": int((merged["_merge"] == "right_only").sum()),
        "patient_agrees": bool((both["patient"] == both["Subject Number"].astype(str)).all()),
        "eye_agrees": bool((both["eye"] == both["Laterality"].astype(str).str.strip()).all()),
        "visit_agrees": bool((both["visit"] == both["Visit Number"].astype(str)).all()),
        "device_agrees": bool((both["device"] == both["Acquisition Device"].astype(str).str.strip()).all()),
        "images_on_disk": int(table["filename"].isin(set(os.listdir(IMG_DIR))).sum()),
    }


def ladder_summary(table: pd.DataFrame, feats: dict, res: reid.EvalResult) -> dict:
    """Rung and camera of every top-1 match, pair-distance distributions per rung and camera, and d' contrasts."""
    info = table.set_index("filename")
    m = res.matches.copy()
    q, r = info.loc[m["query"]], info.loc[m["result"]]
    rel = []
    for (_, a), (_, b) in zip(q.reset_index().iterrows(), r.reset_index().iterrows()):
        rel.append(reid.pair_rung(a, b))
    m["rung"] = rel
    m["same_camera"] = q["device"].to_numpy() == r["device"].to_numpy()
    top1 = m.groupby(["rung", "same_camera"]).size().rename("n").reset_index()
    errors = m[~m["same_patient"]]
    pairs = reid.ladder_pairs(table, feats)
    dist = (pairs.groupby(["rung", "same_camera"])["distance"]
            .agg(n="size", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index())

    def sel(rung, cam=None):
        s = pairs[pairs["rung"] == rung]
        return (s if cam is None else s[s["same_camera"] == cam])["distance"].to_numpy()

    contrasts = {
        "same_eye_other_visit vs other_patient (same camera)": reid.decidability(sel("same_eye_other_visit", True), sel("other_patient", True)),
        "same_eye_other_visit vs other_patient (different camera)": reid.decidability(sel("same_eye_other_visit", False), sel("other_patient", False)),
        "fellow_eye_same_visit vs other_patient (same camera)": reid.decidability(sel("fellow_eye_same_visit", True), sel("other_patient", True)),
        "fellow_eye_same_visit vs same_eye_other_visit (same camera)": reid.decidability(sel("fellow_eye_same_visit", True), sel("same_eye_other_visit", True)),
        "other_patient same camera vs other_patient different camera": reid.decidability(sel("other_patient", True), sel("other_patient", False)),
    }
    confound = {
        "median other_patient same camera": float(np.median(sel("other_patient", True))),
        "median same_eye_other_visit different camera": float(np.median(sel("same_eye_other_visit", False))) if len(sel("same_eye_other_visit", False)) else None,
        "median same_eye_other_visit same camera": float(np.median(sel("same_eye_other_visit", True))),
        "n same_eye_other_visit different camera pairs": int(len(sel("same_eye_other_visit", False))),
    }
    return {
        "top1_by_rung_and_camera": top1.to_dict(orient="records"),
        "correct_top1_fellow_eye": int(((m["rung"] == "fellow_eye_other_visit") & m["same_patient"]).sum()),
        "errors": int(len(errors)), "errors_same_camera": int(errors["same_camera"].sum()),
        "pair_distance_by_rung_and_camera": dist.to_dict(orient="records"),
        "decidability": contrasts, "camera_confound": confound,
    }, m


def camera_restricted_r1(table: pd.DataFrame, feats: dict, same_camera: bool) -> dict:
    """Different-visit protocol with the gallery limited to images from the same (or a different) camera as the query."""
    t = table[table["filename"].isin(feats)].copy()
    v = t.groupby("patient")["visit"].nunique()
    t = t[t["patient"].isin(v[v > 1].index)].reset_index(drop=True)
    X = np.stack([feats[f] for f in t["filename"]])
    sq = (X * X).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(D, np.inf)
    pat, vis, cam = t["patient"].to_numpy(), t["visit"].to_numpy(), t["device"].to_numpy()
    correct, has_target, clusters = [], [], []
    for i in range(len(t)):
        d = D[i].copy()
        d[(pat == pat[i]) & (vis == vis[i])] = np.inf
        d[(cam == cam[i]) != same_camera] = np.inf
        if not np.isfinite(d).any():
            continue
        target_available = bool(((pat == pat[i]) & np.isfinite(d)).any())
        j = int(np.argmin(d))
        correct.append(pat[j] == pat[i]); has_target.append(target_available); clusters.append(pat[i])
    correct, has_target, clusters = map(np.asarray, (correct, has_target, clusters))
    ok = has_target
    return {"queries": int(len(correct)), "queries_with_own_patient_in_gallery": int(ok.sum()),
            "R@1_among_queries_with_target": float(correct[ok].mean()) if ok.any() else None,
            "R@1_ci95_patient_bootstrap": cluster_bootstrap_ci(correct[ok].astype(float), clusters[ok]) if ok.sum() > 10 else None}


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    table = reid.load_table(os.path.join(EXT, "data", "GRAPE", "grape_img_info_w_file.csv"))
    out = {"run": RUN, "table_check": check_table_against_spreadsheet(table), "configs": {}}
    print("table check:", out["table_check"])

    feats_rf = reid.load_features_safe(os.path.join(EXT, "features", "GRAPE"))
    released = pd.read_csv(os.path.join(EXT, "results", "GRAPE", "matches_diff_dates.csv"))

    # Smoke test before the image loop: one image, output size, time per image, projected total.
    t0 = time.time()
    one = reid.pixel_features(IMG_DIR, [table["filename"].iloc[0]])
    per = time.time() - t0
    vec = next(iter(one.values()))
    assert vec.shape == (768,) and np.isfinite(vec).all(), "pixel smoke test failed"
    print(f"smoke test: {per:.2f}s/item; estimated total {per * len(table) / 60:.1f} min for {len(table)} items; device: CPU only")
    feats_px = reid.pixel_features(IMG_DIR, table["filename"].tolist())

    for name, feats in (("retfound_released_features", feats_rf), ("pixels_16x16", feats_px)):
        res = reid.evaluate(table, feats, exclude_same_visit=True)
        per_query = res.matches["same_patient"].to_numpy().astype(float)
        clusters = res.matches["query"].map(dict(zip(table["filename"], table["patient"]))).to_numpy()
        lad, m = ladder_summary(table, feats, res)
        cfg = {
            "n_features": len(feats),
            "metrics_pct": {k: round(100 * v, 2) for k, v in res.metrics.items() if k.endswith(("R@1", "R@10", "AP"))},
            "published_pct": PUBLISHED[name],
            "counts": {k: v for k, v in res.metrics.items() if k.startswith("n_")},
            "image_R@1_ci95_patient_bootstrap": cluster_bootstrap_ci(per_query, clusters),
            "ladder": lad,
            "camera_restricted": {"same_camera_gallery": camera_restricted_r1(table, feats, True),
                                  "different_camera_gallery": camera_restricted_r1(table, feats, False)},
        }
        if name == "retfound_released_features":
            rel = released.set_index("query")["result"]
            mine = res.matches.set_index("query")["result"]
            common = rel.index.intersection(mine.index)
            cfg["agreement_with_released_matches"] = {"queries_compared": int(len(common)),
                                                      "same_top1": int((rel.loc[common] == mine.loc[common]).sum())}
        date_res = reid.evaluate(table, feats, exclude_same_visit=True, date_col="interval_years")
        cfg["sensitivity_exclude_by_date_pct"] = {k: round(100 * v, 2) for k, v in date_res.metrics.items() if k.endswith(("R@1", "R@10", "AP"))}
        out["configs"][name] = cfg
        m.to_csv(os.path.join(PRIVATE, f"matches_{name}.csv"), index=False)
        print(name, cfg["metrics_pct"], "| published", PUBLISHED[name])

    import sklearn, torch, torchvision
    out["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                          "scikit-learn": sklearn.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__}
    out["not_attempted"] = ["Engelmann et al. 2026 ResNet10t (ImageNet) row: needs timm and pretrained weights, not downloaded",
                            "Nebbia et al. 2025 RETFound feature extraction from images: gated weights; released features used instead",
                            "Nebbia et al. 2025 fine-tuned (SimCLR) results: reported only on private datasets"]
    with open(os.path.join(PROJECT, "results", "grape_repro.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("written results/grape_repro.json and private matches to", PRIVATE)


if __name__ == "__main__":
    main()
