"""Selection of pairs, stranger references and mean pools for the 2D identity ladder (docs/PROTOCOL_2d_ladder.md).

Reads metadata and file names only (no pixel). Writes the selection, which holds image names and person IDs, to the
private project folder of the data root and prints counts and the file's SHA-256 only.

Every pair has an anchor x and a candidate y: for a same-eye pair the earlier image (GRAPE) or the first in file
order (iris, same sitting); for a fellow-eye pair the right eye is the anchor. The candidate slot is the one that
strangers and the mean replace: for each pair, one image from each of up to M + K other people matched to y's
nuisance (exact match keys) is drawn; the first M form the population-mean image, the other K are the stranger
reference shared by the own rung, the mean rung and the shrinkage path between them. One pair per unit (eye, or
person for the fellow eye) per rung, drawn with a fixed seed.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from iris_calibration import UPOL, UTIRIS, upol_table, utiris_table  # noqa: E402

SEED = 20261101
DATA_ROOT = os.environ.get("DATA_ROOT", r"C:\Projects\data")
GRAPE_META = os.path.join(os.path.dirname(__file__), "..", "external", "re_identification-50bf2ea", "src",
                          "re_identification-50bf2ead64acf63f5071a4d947a11f6d4056070a", "data", "GRAPE",
                          "grape_img_info_w_file.csv")
OUT_DIR = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "run-2026-11-02-2d-ladder")

GRAPE_RUNGS = ("same_camera_<1y", "same_camera_1y+", "other_camera", "fellow_eye_same_visit")
IRIS_RUNGS = ("same_eye_same_sitting", "fellow_eye_same_sitting")
RIGHT = {"OD", "R"}
CFG = {"grape": {"keys": ("device", "side"), "k": 200, "k_min": 30, "m": 16},
       "iris": {"keys": ("dataset", "device", "side"), "k": 200, "k_min": 30, "m": 16}}


def grape_universe(table: pd.DataFrame) -> pd.DataFrame:
    t = table.rename(columns={"Subject Number": "person", "Laterality": "side", "Corresponding CFP": "image",
                              "Acquisition Device": "device", "Interval Years": "time"})
    u = t[["image", "person", "side", "device", "time"]].copy()
    u["image"] = u["image"].astype(str).map(os.path.basename)
    u["person"] = u["person"].astype(str)
    u["device"] = u["device"].astype(str).str.strip()
    u["time"] = u["time"].astype(float)
    u["unit"] = u["person"] + "-" + u["side"]
    u["dataset"] = "GRAPE"
    return u.reset_index(drop=True)


def iris_universe(upol: pd.DataFrame, utiris: pd.DataFrame) -> pd.DataFrame:
    """UPOL and UTIRIS near-infrared images; image = path relative to the dataset root (no sitting dates exist)."""
    nir = utiris[utiris.sensor == "NIR"]
    parts = [pd.DataFrame({"image": [os.path.relpath(p, root) for p in t.path], "person": t.dataset + "-" + t.person,
                           "side": t.eye, "device": t.sensor, "dataset": t.dataset, "time": 0.0})
             for t, root in ((upol, UPOL), (nir, UTIRIS))]
    u = pd.concat(parts, ignore_index=True)
    u["unit"] = u["person"] + "-" + u["side"]
    return u.sort_values(["dataset", "image"]).reset_index(drop=True)


def grape_rung(x: dict, y: dict) -> str | None:
    gap = y["time"] - x["time"]
    if x["unit"] != y["unit"]:
        return "fellow_eye_same_visit" if abs(gap) < 1e-6 and x["device"] == y["device"] else None
    if gap < 1e-6:
        return None  # GRAPE has no same-session repeats; a same-date pair would be one
    if x["device"] != y["device"]:
        return "other_camera"
    return "same_camera_<1y" if gap < 1 else "same_camera_1y+"


def iris_rung(x: dict, y: dict) -> str | None:
    if x["unit"] == y["unit"]:
        return "same_eye_same_sitting"
    return "fellow_eye_same_sitting" if x["device"] == y["device"] else None


def pairs(u: pd.DataFrame, rung_fn, rung_order: tuple, rng: np.random.Generator) -> tuple[pd.DataFrame, Counter]:
    """One pair per unit per rung. Same-eye anchor: earlier in (time, file) order; fellow-eye anchor: right eye."""
    rows, dropped = [], Counter()
    for pid, g in u.sort_values(["person", "time", "image"]).groupby("person", sort=True):
        recs = g.to_dict("records")
        by_rung = {}
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                x, y = recs[i], recs[j]
                if x["unit"] != y["unit"] and x["side"] not in RIGHT:
                    x, y = y, x
                rung = rung_fn(x, y)
                if rung is None:
                    dropped["pairs_outside_rungs"] += 1
                    continue
                unit = pid if rung.startswith("fellow") else x["unit"]
                by_rung.setdefault((rung, unit), []).append((x, y))
        for rung in rung_order:
            for key in sorted(k for k in by_rung if k[0] == rung):
                x, y = by_rung[key][rng.integers(len(by_rung[key]))]
                rows.append({"rung": rung, "dataset": x["dataset"], "person": pid, "unit": key[1],
                             "anchor": x["image"], "candidate": y["image"], "gap": float(y["time"] - x["time"])})
    return pd.DataFrame(rows), dropped


class StrangerPool:
    """Images grouped by the exact match keys."""

    def __init__(self, u: pd.DataFrame, keys: tuple):
        self.keys = keys
        self.groups = {k if isinstance(k, tuple) else (k,): g.reset_index(drop=True)
                       for k, g in u.groupby(list(keys), sort=True)}

    def draw(self, r, exclude: set, n: int, n_min: int, rng: np.random.Generator) -> list | None:
        """One image from each of n other people matched to image row r, in random order; None if fewer than n_min."""
        g = self.groups.get(tuple(r[k] for k in self.keys))
        if g is None:
            return None
        g = g[~g.person.isin(exclude)]
        people = np.array(sorted(g.person.unique()))
        if len(people) < n_min:
            return None
        chosen = rng.choice(people, size=min(n, len(people)), replace=False)
        by_person = g.groupby("person").image.apply(sorted)
        return [by_person[p][rng.integers(len(by_person[p]))] for p in chosen]


def attach(prs: pd.DataFrame, u: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> tuple[list, Counter]:
    """M mean members and K reference strangers, disjoint people, all matched to the candidate."""
    rows = u.set_index("image")
    pool = StrangerPool(u, cfg["keys"])
    m = cfg["m"]
    out, dropped = [], Counter()
    for p in prs.to_dict("records"):
        draw = pool.draw(rows.loc[p["candidate"]], {p["person"]}, m + cfg["k"], m + cfg["k_min"], rng)
        if draw is None:
            dropped[f"{p['rung']}: fewer than M + k_min matched strangers"] += 1
            continue
        out.append({**p, "mean_members": draw[:m], "reference": draw[m:]})
    return out, dropped


def select(u: pd.DataFrame, rung_fn, rung_order: tuple, cfg: dict, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    prs, dropped_pairs = pairs(u, rung_fn, rung_order, rng)
    recs, dropped_refs = attach(prs, u, cfg, rng)
    return {"seed": seed, "config": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
            "n_universe_images": int(len(u)), "n_universe_people": int(u.person.nunique()),
            "dropped": {**dropped_pairs, **dropped_refs}, "pairs": recs}


def summary(sel: dict) -> dict:
    """Counts only: no image name, no person ID."""
    df = pd.DataFrame(sel["pairs"])
    rungs = {f"{ds} | {rung}": {"pairs": int(len(g)), "people": int(g.person.nunique()),
                                "k_median": float(g.reference.map(len).median()),
                                "k_min": int(g.reference.map(len).min()), "gap_median": float(g.gap.median())}
             for (ds, rung), g in df.groupby(["dataset", "rung"], sort=False)}
    return {"n_universe_images": sel["n_universe_images"], "n_universe_people": sel["n_universe_people"],
            "rungs": rungs, "dropped": sel["dropped"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    utiris, _ = utiris_table()
    jobs = {"grape": (grape_universe(pd.read_csv(GRAPE_META)), grape_rung, GRAPE_RUNGS),
            "iris": (iris_universe(upol_table(), utiris), iris_rung, IRIS_RUNGS)}
    report = {"grape_metadata_sha256": hashlib.sha256(open(GRAPE_META, "rb").read()).hexdigest()}
    for name, (u, fn, order) in jobs.items():
        sel = select(u, fn, order, CFG[name])
        blob = json.dumps(sel, sort_keys=True).encode()
        with open(os.path.join(a.out_dir, f"selection_{name}.json"), "wb") as fh:
            fh.write(blob)
        report[name] = {"selection_sha256": hashlib.sha256(blob).hexdigest(), **summary(sel)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
