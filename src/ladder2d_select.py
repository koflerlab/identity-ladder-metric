"""Selection of pairs, stranger references and mean pools for the 2D identity ladder (docs/PROTOCOL_2d_ladder.md).

Reads metadata only (no image). Writes the selection, which holds image names and patient IDs, to the private project
folder of the data root and prints counts and the file's SHA-256 only.

Every pair has an anchor x (the earlier image) and a candidate y (the later image, or the fellow eye). The stranger
reference of a pair substitutes the candidate: one image from each of up to K other patients, matched to y's nuisance
(the match keys exactly, age within a tolerance that widens once if too few patients qualify). The population-mean
rung keeps the anchor: M other patients matched to x form the mean image, and a disjoint set matched to x is its
reference. One pair per unit (patient for ChestX-ray14, eye for GRAPE) per rung, drawn with a fixed seed.
"""
import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

SEED = 20261101
DATA_ROOT = os.environ.get("DATA_ROOT", r"C:\Projects\data")
CXR_META = os.path.join(DATA_ROOT, "datasets", "radiology", "nih-chestxray14", "version-unverified", "downloads",
                        "Data_Entry_2017_v2020.csv")
GRAPE_META = os.path.join(os.path.dirname(__file__), "..", "external", "re_identification-50bf2ea", "src",
                          "re_identification-50bf2ead64acf63f5071a4d947a11f6d4056070a", "data", "GRAPE",
                          "grape_img_info_w_file.csv")
OUT_DIR = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "run-2026-11-02-2d-ladder")

CXR_N_IMAGES = 14_999  # images_001 + images_002 in sorted Image Index order (4,999 + 10,000)
CXR_ARCHIVE_1 = 4_999
CXR_RUNGS = (("short_0y", 0, 0), ("mid_2-4y", 2, 4), ("long_5y+", 5, 200))  # age difference in whole years
CXR = {"keys": ("sex", "device"), "age_tols": (2, 5), "k": 200, "k_min": 50, "m": 32}
GRAPE_RUNGS = ("same_camera_<1y", "same_camera_1y+", "other_camera", "fellow_eye_same_visit")
GRAPE = {"keys": ("device", "side"), "age_tols": (None,), "k": 200, "k_min": 30, "m": 16}


def cxr_universe(meta: pd.DataFrame, n_images: int = CXR_N_IMAGES) -> pd.DataFrame:
    """Adult PA images of the downloaded archives; device proxy = original pixel spacing."""
    m = meta.sort_values("Image Index").reset_index(drop=True)
    m["archive"] = np.where(m.index < CXR_ARCHIVE_1, 1, 2)
    m = m.iloc[:n_images]
    u = pd.DataFrame({"image": m["Image Index"], "patient": m["Patient ID"].astype(str),
                      "order": m["Follow-up #"].astype(int), "age": m["Patient Age"].astype(int),
                      "sex": m["Patient Sex"].astype(str), "view": m["View Position"].astype(str),
                      "device": m["OriginalImagePixelSpacing[x"].round(4).astype(str),
                      "no_finding": m["Finding Labels"] == "No Finding", "archive": m["archive"]})
    return u[(u.view == "PA") & (u.age >= 18)].reset_index(drop=True)


def cxr_rung(gap: int) -> str | None:
    for name, lo, hi in CXR_RUNGS:
        if lo <= gap <= hi:
            return name
    return None


def cxr_pairs(u: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, Counter]:
    """One (earlier, later) pair per patient per rung. A gap of one year is ambiguous (0 to 2 years) and unused."""
    rows, dropped = [], Counter()
    for pid, g in u.sort_values(["patient", "order"]).groupby("patient", sort=True):
        recs = g.to_dict("records")
        by_rung = {}
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                gap = recs[j]["age"] - recs[i]["age"]
                if gap < 0:
                    dropped["age_decreases_with_follow_up"] += 1
                    continue
                rung = cxr_rung(gap)
                if rung:
                    by_rung.setdefault(rung, []).append((recs[i], recs[j], gap))
        for rung, _, _ in CXR_RUNGS:
            if rung in by_rung:
                x, y, gap = by_rung[rung][rng.integers(len(by_rung[rung]))]
                rows.append({"rung": rung, "patient": pid, "unit": pid, "anchor": x["image"], "candidate": y["image"],
                             "gap": gap})
    return pd.DataFrame(rows), dropped


def grape_universe(table: pd.DataFrame) -> pd.DataFrame:
    t = table.rename(columns={"Subject Number": "patient", "Laterality": "side", "Corresponding CFP": "image",
                              "Acquisition Device": "device", "Interval Years": "time", "Age": "age", "Gender": "sex"})
    u = t[["image", "patient", "side", "device", "time", "age", "sex"]].copy()
    u["image"] = u["image"].astype(str).map(os.path.basename)
    u["patient"] = u["patient"].astype(str)
    u["device"] = u["device"].astype(str).str.strip()
    u["time"] = u["time"].astype(float)
    u["unit"] = u["patient"] + "-" + u["side"]
    return u.reset_index(drop=True)


def grape_rung(x: dict, y: dict) -> str | None:
    gap = y["time"] - x["time"]
    if x["unit"] != y["unit"]:
        same_visit = abs(gap) < 1e-6
        return "fellow_eye_same_visit" if same_visit and x["device"] == y["device"] else None
    if gap < 1e-6:
        return None  # GRAPE has no same-session repeats; a same-date pair would be one
    if x["device"] != y["device"]:
        return "other_camera"
    return "same_camera_<1y" if gap < 1 else "same_camera_1y+"


def grape_pairs(u: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, Counter]:
    """Same-eye pairs: one per eye per rung, anchor earlier. Fellow-eye pairs: one per patient, anchor OD."""
    rows, dropped = [], Counter()
    for pid, g in u.sort_values(["patient", "time", "side", "image"]).groupby("patient", sort=True):
        recs = g.to_dict("records")
        by_rung = {}
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                x, y = recs[i], recs[j]
                if x["unit"] != y["unit"] and x["side"] == "OS":
                    x, y = y, x
                rung = grape_rung(x, y)
                if rung is None:
                    dropped["pairs_outside_rungs"] += 1
                    continue
                unit = pid if rung == "fellow_eye_same_visit" else x["unit"]
                by_rung.setdefault((rung, unit), []).append((x, y, y["time"] - x["time"]))
        for rung in GRAPE_RUNGS:
            for (r, unit) in sorted(k for k in by_rung if k[0] == rung):
                x, y, gap = by_rung[(r, unit)][rng.integers(len(by_rung[(r, unit)]))]
                rows.append({"rung": rung, "patient": pid, "unit": unit, "anchor": x["image"],
                             "candidate": y["image"], "gap": float(gap)})
    return pd.DataFrame(rows), dropped


class StrangerPool:
    """Images of the reference universe grouped by the exact match keys."""

    def __init__(self, u: pd.DataFrame, keys: tuple):
        self.keys = keys
        self.groups = {k if isinstance(k, tuple) else (k,): g.reset_index(drop=True)
                       for k, g in u.groupby(list(keys), sort=True)}

    def draw(self, r: dict, exclude: set, n: int, n_min: int, age_tols: tuple, rng: np.random.Generator):
        """One image from each of n other patients matched to image row r; (images, tol) or (None, None) if < n_min."""
        g = self.groups.get(tuple(r[k] for k in self.keys))
        if g is None:
            return None, None
        g = g[~g.patient.isin(exclude)]
        for tol in age_tols:
            pool = g if tol is None else g[(g.age - r["age"]).abs() <= tol]
            patients = np.array(sorted(pool.patient.unique()))
            if len(patients) >= n_min:
                chosen = rng.choice(patients, size=min(n, len(patients)), replace=False)
                by_patient = pool.groupby("patient").image.apply(sorted)
                return [by_patient[p][rng.integers(len(by_patient[p]))] for p in chosen], tol
        return None, None


def attach(pairs: pd.DataFrame, u_pairs: pd.DataFrame, u_ref: pd.DataFrame, cfg: dict,
           rng: np.random.Generator) -> tuple[list, Counter]:
    """Stranger reference for each pair's candidate; mean members and their reference for each anchor.

    The nuisance of x and y is read from the pair universe; strangers come from the reference universe, which a
    sensitivity analysis may restrict (for example to a random half of the patients).
    """
    rows = u_pairs.set_index("image")
    pool = StrangerPool(u_ref, cfg["keys"])
    m = cfg["m"]
    out, dropped = [], Counter()
    for p in pairs.to_dict("records"):
        exclude = {p["patient"]}
        ref, tol_c = pool.draw(rows.loc[p["candidate"]], exclude, cfg["k"], cfg["k_min"], cfg["age_tols"], rng)
        if ref is None:
            dropped[f"{p['rung']}: too few matched strangers for the candidate"] += 1
            continue
        mean_draw, tol_m = pool.draw(rows.loc[p["anchor"]], exclude, m + cfg["k"], m + cfg["k_min"], cfg["age_tols"],
                                     rng)
        rec = {**p, "ref_candidate": ref, "age_tol_candidate": tol_c}
        if mean_draw is None:
            dropped[f"{p['rung']}: too few matched strangers for the mean (pair kept)"] += 1
            rec.update({"mean_members": None, "ref_mean": None, "age_tol_mean": None})
        else:
            rec.update({"mean_members": mean_draw[:m], "ref_mean": mean_draw[m:], "age_tol_mean": tol_m})
        out.append(rec)
    return out, dropped


def select_cxr(meta: pd.DataFrame, seed: int = SEED) -> dict:
    u = cxr_universe(meta)
    patients = np.array(sorted(u.patient.unique()))
    half_a = set(np.random.default_rng([seed, 99]).permutation(patients)[: len(patients) // 2])
    variants = {"primary": (u, u, CXR),
                "S1_no_finding": (u[u.no_finding], u[u.no_finding], CXR),
                "S2_age_tol_5": (u, u, {**CXR, "age_tols": (5,)}),
                "S3a_reference_half_A": (u, u[u.patient.isin(half_a)], CXR),
                "S3b_reference_half_B": (u, u[~u.patient.isin(half_a)], CXR),
                "S5_unmatched_on_age": (u, u, {**CXR, "age_tols": (None,)})}
    return _select(u, variants, cxr_pairs, seed)


def select_grape(table: pd.DataFrame, seed: int = SEED) -> dict:
    u = grape_universe(table)
    return _select(u, {"primary": (u, u, GRAPE)}, grape_pairs, seed)


def _select(u, variants, pair_fn, seed):
    out = {"seed": seed, "n_universe_images": int(len(u)), "n_universe_patients": int(u.patient.nunique()),
           "variants": {}}
    for i, (name, (u_pairs, u_ref, cfg)) in enumerate(variants.items()):
        rng = np.random.default_rng([seed, i])
        pairs, dropped_pairs = pair_fn(u_pairs, rng)
        recs, dropped_refs = attach(pairs, u_pairs, u_ref, cfg, rng)
        out["variants"][name] = {"config": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                                 "dropped": {**dropped_pairs, **dropped_refs}, "pairs": recs}
    return out


def summary(sel: dict) -> dict:
    """Counts only: no image name, no patient ID."""
    s = {"n_universe_images": sel["n_universe_images"], "n_universe_patients": sel["n_universe_patients"],
         "variants": {}}
    for name, v in sel["variants"].items():
        df = pd.DataFrame(v["pairs"])
        rungs = {}
        for rung, g in df.groupby("rung", sort=False):
            k = g.ref_candidate.map(len)
            rungs[rung] = {"pairs": int(len(g)), "patients": int(g.patient.nunique()),
                           "k_median": float(k.median()), "k_min": int(k.min()),
                           "widened_age_tol_share": float(g.age_tol_candidate.map(lambda t: t != v["config"]["age_tols"][0]).mean()),
                           "with_mean_rung": int(g.mean_members.notna().sum()),
                           "gap_median": float(g.gap.median())}
        s["variants"][name] = {"rungs": rungs, "dropped": dict(v["dropped"])}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    report = {}
    for name, path, fn in (("cxr", CXR_META, lambda p: select_cxr(pd.read_csv(p))),
                           ("grape", GRAPE_META, lambda p: select_grape(pd.read_csv(p)))):
        sel = fn(path)
        blob = json.dumps(sel, sort_keys=True).encode()
        with open(os.path.join(a.out_dir, f"selection_{name}.json"), "wb") as fh:
            fh.write(blob)
        report[name] = {"metadata_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                        "selection_sha256": hashlib.sha256(blob).hexdigest(), **summary(sel)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
