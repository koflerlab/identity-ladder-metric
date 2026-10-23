"""Re-identification on GRAPE fundus photographs: reproduction of Nebbia et al. 2025 and Engelmann et al. 2026,
plus a split of every pair and every top-1 match by identity-ladder rung and camera.

Protocol of the original find_nearest_neighbor.py (QTIM-Lab/re_identification @ 50bf2ea), reimplemented with
exact nearest-neighbour search instead of chromadb's approximate HNSW index:
  * keep only patients with more than one visit;
  * L2 distance between feature vectors (chromadb "l2" is squared L2; the ranking is identical);
  * a query never matches itself, and in the "different date" protocol all images of the same patient taken at
    the same visit (including the fellow eye) are removed from its gallery;
  * image-level R@1 = correct top-1 / number of query images; R@10 over queries with at least 10 candidates;
  * patient-level R@1 = patients with at least one correctly matched image / number of patients;
  * image-level AP on the top-1 distance; patient-level AP keeps, per patient, the closest correct match if any,
    otherwise the closest match.
"""
from __future__ import annotations

import io
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler for third-party feature files: only numpy array reconstruction may be resolved.

    The released GRAPE feature pickles are untrusted input. An opcode scan (pickletools, no execution) showed that
    they reference only numpy.core.multiarray._reconstruct, numpy.ndarray and numpy.dtype. Every other global is
    refused here, so no callable outside numpy array reconstruction can run while loading them.
    """

    def find_class(self, module, name):
        if (module, name) in ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_features_safe(folder: str) -> dict[str, np.ndarray]:
    """Read every .pkl in folder (list of {'path', 'features'} dicts) without executing arbitrary code."""
    out: dict[str, np.ndarray] = {}
    files = sorted(f for f in os.listdir(folder) if f.endswith(".pkl"))
    if not files:
        raise FileNotFoundError(f"no .pkl files in {folder}")
    for f in files:
        with open(os.path.join(folder, f), "rb") as fh:
            records = RestrictedUnpickler(io.BytesIO(fh.read())).load()
        for r in records:
            name = os.path.basename(str(r["path"]).replace("\\", "/"))
            if name in out:
                raise ValueError(f"duplicate feature for {name}")
            out[name] = np.asarray(r["features"], dtype=np.float64).ravel()
    return out


def load_table(csv_path: str) -> pd.DataFrame:
    """Image table used by the original run (one row per CFP)."""
    t = pd.read_csv(csv_path)
    t = t.rename(columns={"Subject Number": "patient", "Laterality": "eye", "Visit Number": "visit",
                          "Corresponding CFP": "filename", "Acquisition Device": "device", "Interval Years": "interval_years"})
    t["filename"] = t["filename"].astype(str).map(os.path.basename)
    t["patient"] = t["patient"].astype(str)
    t["visit"] = t["visit"].astype(str)
    t["device"] = t["device"].astype(str).str.strip()
    if t["filename"].duplicated().any():
        raise ValueError("duplicate filenames in image table")
    return t


def pixel_features(img_dir: str, filenames: list[str], size: int = 16) -> dict[str, np.ndarray]:
    """Engelmann et al. baseline: torchvision read_image, Resize((size, size)), flatten, divide by 256."""
    import torchvision.transforms.v2 as T
    from torchvision.io import read_image

    resize = T.Resize((size, size))
    out = {}
    for name in filenames:
        img = read_image(os.path.join(img_dir, name))
        out[name] = (resize(img).flatten() / 256.0).numpy().astype(np.float64)
    return out


@dataclass
class EvalResult:
    metrics: dict
    matches: pd.DataFrame  # one row per query: query, result, distance, flags


def evaluate(table: pd.DataFrame, feats: dict[str, np.ndarray], exclude_same_visit: bool = True,
             date_col: str = "visit") -> EvalResult:
    """Replicates the original protocol on the images that have features.

    date_col="visit" follows the original code (Visit Number, counted per eye in GRAPE); date_col="interval_years"
    excludes by acquisition date instead, as the paper's text describes ("same day").
    """
    t = table[table["filename"].isin(feats)].copy()
    visits = t.groupby("patient")["visit"].nunique()  # patient selection always follows the original code
    t = t[t["patient"].isin(visits[visits > 1].index)].reset_index(drop=True)
    X = np.stack([feats[f] for f in t["filename"]])
    sq = (X * X).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)  # squared L2, as chromadb "l2"
    np.fill_diagonal(D, np.inf)
    pat, vis = t["patient"].to_numpy(), t[date_col].astype(str).to_numpy()
    rows, top1_pat, top10_pat, top10_support = [], set(), set(), set()
    n_top1 = n_top10 = n_with10 = 0
    for i in range(len(t)):
        d = D[i].copy()
        if exclude_same_visit:
            d[(pat == pat[i]) & (vis == vis[i])] = np.inf
        order = np.argsort(d, kind="stable")
        order = order[np.isfinite(d[order])]
        if len(order) == 0:
            continue
        j = order[0]
        correct = pat[j] == pat[i]
        n_top1 += correct
        if correct:
            top1_pat.add(pat[i])
        if len(order) >= 10:
            n_with10 += 1
            top10_support.add(pat[i])
            if (pat[order[:10]] == pat[i]).any():
                n_top10 += 1
                top10_pat.add(pat[i])
        rows.append({"query": t.at[i, "filename"], "result": t.at[j, "filename"], "top_distance": d[j], "same_patient": bool(correct)})
    m = pd.DataFrame(rows)
    n_img, n_pat = len(t), t["patient"].nunique()
    score = m["top_distance"].max() - m["top_distance"]
    per_pat = []
    q_pat = m["query"].map(dict(zip(t["filename"], t["patient"])))
    for p, g in m.assign(score=score, qp=q_pat).groupby("qp"):
        same = g[g["same_patient"]]
        best = (same if len(same) else g).sort_values("score", ascending=False).iloc[0]
        per_pat.append({"same": bool(best["same_patient"]), "score": best["score"]})
    per_pat = pd.DataFrame(per_pat)
    metrics = {
        "n_images": int(n_img), "n_patients": int(n_pat),
        "image_R@1": n_top1 / n_img, "image_R@10": n_top10 / n_with10 if n_with10 else float("nan"),
        "image_AP": float(average_precision_score(m["same_patient"], score)),
        "patient_R@1": len(top1_pat) / n_pat, "patient_R@10": len(top10_pat) / len(top10_support) if top10_support else float("nan"),
        "patient_AP": float(average_precision_score(per_pat["same"], per_pat["score"])),
        "n_correct_top1": int(n_top1),
    }
    return EvalResult(metrics, m)


RUNGS = ["same_eye_other_visit", "fellow_eye_same_visit", "fellow_eye_other_visit", "other_patient"]


def pair_rung(a: pd.Series, b: pd.Series) -> str:
    if a["patient"] != b["patient"]:
        return "other_patient"
    if a["eye"] == b["eye"]:
        return "same_eye_same_visit" if a["visit"] == b["visit"] else "same_eye_other_visit"
    return "fellow_eye_same_visit" if a["visit"] == b["visit"] else "fellow_eye_other_visit"


def decidability(x: np.ndarray, y: np.ndarray) -> float:
    """Daugman's d' between two distance distributions."""
    return float(abs(x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))


def ladder_pairs(table: pd.DataFrame, feats: dict[str, np.ndarray]) -> pd.DataFrame:
    """Every unordered image pair with its rung, camera relation and Euclidean distance (all images with features)."""
    t = table[table["filename"].isin(feats)].reset_index(drop=True)
    X = np.stack([feats[f] for f in t["filename"]])
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    iu, ju = np.triu_indices(len(t), k=1)
    same_pat = t["patient"].to_numpy()[iu] == t["patient"].to_numpy()[ju]
    same_eye = t["eye"].to_numpy()[iu] == t["eye"].to_numpy()[ju]
    same_vis = t["visit"].to_numpy()[iu] == t["visit"].to_numpy()[ju]
    rung = np.where(~same_pat, "other_patient",
                    np.where(same_eye, np.where(same_vis, "same_eye_same_visit", "same_eye_other_visit"),
                             np.where(same_vis, "fellow_eye_same_visit", "fellow_eye_other_visit")))
    same_cam = t["device"].to_numpy()[iu] == t["device"].to_numpy()[ju]
    return pd.DataFrame({"i": iu, "j": ju, "rung": rung, "same_camera": same_cam, "distance": D[iu, ju]})
