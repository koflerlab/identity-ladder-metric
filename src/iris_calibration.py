"""Random-pair calibrated identity scores on two open iris sets (UPOL, UTIRIS): the 2D iris pilot of the metric.

Same calibration as grape_calibration.py: a pair distance d becomes c(d) = P(d_ref > d) with d_ref drawn from
unrelated-person pairs in the same nuisance condition (same or different sensor; same or opposite eye; mirrored or
not). Feature: the Engelmann et al. pixel baseline (greyscale, 16x16, /256), plus a per-image z-scored variant that
removes global brightness, which matters for the visible-versus-NIR comparison in UTIRIS.

Rungs: same eye, same sensor (L0); same eye, other sensor (UTIRIS only: visible light vs NIR; archive and EXIF times
show both were taken in one sitting about a minute apart once a constant 11.2 h clock offset is removed, so this is a
cross-spectral L0, not a change of session); fellow eye (mirror), plain and after a horizontal flip; other person
(L4, the reference). The reference is stratified by the unordered sensor pair and the eye relation.
Negative control: a feature made only of the image's mean and SD of grey level. The audit of 2026-11-01 found that
global exposure alone reproduces most of the pixel-baseline scores, so pixel rungs mostly measure capture conditions.
Labels: UPOL names <subject><L|R>_<k>.png; UTIRIS NIR folders <subject>/<subject>_<L|R>/ and file names, RGB file
names. NIR images whose file-name eye disagrees with their folder eye are excluded and counted.

Images are read locally and only aggregate numbers are written: results/iris_calibration.json.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from grape_calibration import TAUS, calibrated_score, cluster_ci, distance_matrix  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
UPOL = r"C:\Projects\data\datasets\biometrics\upol\version-unverified\raw"
UTIRIS = r"C:\Projects\data\datasets\biometrics\utiris\v1\raw"
SIZE = 16


def upol_table(root: str = UPOL) -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(root)):
        m = re.fullmatch(r"(\d{3})([LR])_(\d)\.png", f)
        if m:
            rows.append({"dataset": "UPOL", "person": m.group(1), "eye": m.group(2), "sensor": "TOPCON-SONY", "path": os.path.join(root, f)})
    return pd.DataFrame(rows)


def utiris_table(root: str = UTIRIS) -> tuple[pd.DataFrame, int]:
    rows, conflicts = [], 0
    for sensor, folder in (("NIR", "Infrared Images"), ("VIS", "RGB Images")):
        for d, _, files in os.walk(os.path.join(root, folder)):
            folder_eye = re.search(r"_([LR])$", os.path.basename(d))
            for f in files:
                m = re.search(r"_(\d{3})_([LR])_", f)
                if not m or not f.lower().endswith((".bmp", ".jpg")):
                    continue
                if folder_eye and folder_eye.group(1) != m.group(2):
                    conflicts += 1
                    continue
                rows.append({"dataset": "UTIRIS", "person": m.group(1), "eye": m.group(2), "sensor": sensor, "path": os.path.join(d, f)})
    return pd.DataFrame(rows), conflicts


def features(paths, size: int = SIZE, flip: bool = False) -> np.ndarray:
    out = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("L")
            if flip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            out.append(np.asarray(im.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64).ravel() / 256.0)
    return np.stack(out)


def zscore_rows(X: np.ndarray) -> np.ndarray:
    sd = X.std(1, keepdims=True)
    if (sd == 0).any():
        raise FloatingPointError("constant image in feature matrix")
    return (X - X.mean(1, keepdims=True)) / sd


def pair_frame(t: pd.DataFrame, X: np.ndarray, Xflip: np.ndarray | None = None) -> pd.DataFrame:
    """All unordered pairs within one dataset; with Xflip, opposite-eye pairs use one image flipped."""
    return pairs_from_matrices(t, distance_matrix(X, X), distance_matrix(X, Xflip) if Xflip is not None else None)


def pairs_from_matrices(t: pd.DataFrame, D: np.ndarray, Dm: np.ndarray | None = None) -> pd.DataFrame:
    """Pair table from a distance matrix D (and Dm, image i against flipped image j, for opposite-eye pairs).

    Pairs whose distance is NaN (for example an iris code with too few valid bits) are dropped."""
    iu, ju = np.triu_indices(len(t), 1)
    person, eye, sensor = (t[c].to_numpy() for c in ("person", "eye", "sensor"))
    same_p, same_e, same_s = person[iu] == person[ju], eye[iu] == eye[ju], sensor[iu] == sensor[ju]
    sensor_pair = np.array([" + ".join(sorted((a, b))) for a, b in zip(sensor[iu], sensor[ju])])
    rung = np.where(~same_p, "other_person",
                    np.where(same_e, np.where(same_s, "same_eye_same_sensor", "same_eye_other_sensor"),
                             np.where(same_s, "fellow_eye_same_sensor", "fellow_eye_other_sensor")))
    dist = D[iu, ju] if Dm is None else np.where(~same_e, Dm[iu, ju], D[iu, ju])
    p = pd.DataFrame({"rung": rung, "same_sensor": same_s, "sensor_pair": sensor_pair, "opposite_eye": ~same_e,
                      "person": np.where(same_p, person[iu], ""), "distance": dist})
    return p[np.isfinite(p.distance)].reset_index(drop=True)


def calibrate(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["c"] = np.nan
    for _, g in p.groupby(["sensor_pair", "opposite_eye"]):
        ref = g.loc[g.rung == "other_person", "distance"].to_numpy()
        if len(ref) < 100:
            raise ValueError("too few stranger pairs for a calibration reference")
        p.loc[g.index, "c"] = calibrated_score(g["distance"].to_numpy(), ref)
    return p


def summarize(p: pd.DataFrame) -> dict:
    out = {}
    for rung, g in p[p.rung != "other_person"].groupby("rung"):
        out[rung] = {"n_pairs": int(len(g)), "n_people": int(g.person.nunique()), "mean_c_auc": float(g.c.mean()),
                     "mean_c_ci95_person_bootstrap": cluster_ci(g.c.to_numpy(), g.person.to_numpy()),
                     "tolerance_curve": {f"c>={t}": float((g.c >= t).mean()) for t in TAUS}}
    return out


def exposure_features(X: np.ndarray) -> np.ndarray:
    """Negative control: only global mean and SD of the 16x16 grey image (no spatial information)."""
    return np.column_stack([X.mean(1), X.std(1)])


def run(t: pd.DataFrame) -> dict:
    X = features(t.path)
    Xf = features(t.path, flip=True)
    res = {"exposure_only_negative_control": summarize(calibrate(pair_frame(t.reset_index(drop=True), exposure_features(X))))}
    for name, A, B in (("pixels_16x16", X, None), ("pixels_16x16_fellow_flipped", X, Xf),
                       ("pixels_16x16_zscored", zscore_rows(X), None), ("pixels_16x16_zscored_fellow_flipped", zscore_rows(X), zscore_rows(Xf))):
        res[name] = summarize(calibrate(pair_frame(t.reset_index(drop=True), A, B)))
    return res


def main():
    up = upol_table()
    ut, conflicts = utiris_table()
    for name, t in (("UPOL", up), ("UTIRIS", ut)):
        if t.empty:
            raise FileNotFoundError(f"no images found for {name}")
    out = {"reference": "unrelated-person pairs with the same unordered sensor pair and eye relation",
           "UPOL": {"n_images": int(len(up)), "n_people": int(up.person.nunique()), "feature_spaces": run(up)},
           "UTIRIS": {"n_images": int(len(ut)), "n_people": int(ut.person.nunique()),
                      "excluded_eye_label_conflicts": conflicts, "images_by_sensor": ut.sensor.value_counts().to_dict(),
                      "feature_spaces": run(ut)},
           "UTIRIS_NIR": {"feature_spaces": run(ut[ut.sensor == "NIR"])},
           "UTIRIS_VIS": {"feature_spaces": run(ut[ut.sensor == "VIS"])}}
    with open(os.path.join(PROJECT, "results", "iris_calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for ds in ("UPOL", "UTIRIS", "UTIRIS_NIR", "UTIRIS_VIS"):
        for fs, rungs in out[ds]["feature_spaces"].items():
            print(ds, fs, {r: (round(v["mean_c_auc"], 3), [round(x, 3) for x in v["mean_c_ci95_person_bootstrap"]]) for r, v in rungs.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
