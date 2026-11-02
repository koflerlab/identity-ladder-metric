"""Donor-averaging ladder on QTIM brain MRI (docs/PROTOCOL_donor_ladder_brain.md). Pre-QC; affine only.

Scans prepared by real_cases.prepare (MNI 2009a symmetric, 1 mm) are z-scored in their own brain mask; 16 mm cubes
with a 4 mm context shell are cut at fixed template positions, plus the two frozen boxes. Fills average k donor
cubes; the summary is the one of the fundus ladder, clustered by family.
"""
import csv
import glob
import json
import os
import re
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_fundus as F  # noqa: E402

SEED = 20261102
DATA = os.environ.get("DATA_ROOT", r"C:\Projects\data")
PROJ = os.path.join(DATA, "projects", "identity-ladder-metric")
PRIVATE = os.path.join(PROJ, "donor-ladder-brain-20261102-v1")
PARTICIPANTS = os.path.join(DATA, "brain_images", "siblings", "participants.tsv")
ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = {"real-cases-run": "selection.json", "real-cases-run-expanded": "selection_expanded.json"}
PATCH, SHELL, N_PATCH = 16, 4, 32
BOXES_MM = {"deep": ((-32, 0), (-19, 13), (-2, 22)), "parietal": ((-45, -13), (-65, -33), (43, 67))}
DRAWN = ("population", "population_both", "cohort")
SINGLE = ("other_hemisphere", "sibling", "cotwin", "repeat")


def scans() -> pd.DataFrame:
    """One row per unique (participant, session) with its prepared skull-stripped volume."""
    rows = {}
    for run, sel in RUNS.items():
        listing = json.load(open(os.path.join(PROJ, "real-cases-protocol", sel)))["scans"]
        for f in glob.glob(os.path.join(PROJ, run, "prep", "scan_*_brain.nii.gz")):
            i = int(re.findall(r"\d+", os.path.basename(f))[0])
            rows.setdefault(tuple(listing[i]), f)
    p = {r["participant_id"]: r for r in csv.DictReader(open(PARTICIPANTS), delimiter="\t")}
    t = pd.DataFrame([{"person": s, "session": ses, "path": f, "family": p[s]["family_id"], "sex": p[s]["sex"],
                       "age": float(p[s]["age"] if ses == "ses-01" or p[s]["age_ses02"] == "n/a" else p[s]["age_ses02"])}
                      for (s, ses), f in sorted(rows.items())])
    return t


def mirror_index(affine, n):
    """Voxel index along x that holds the mirrored (x -> -x mm) position, assuming a diagonal x axis."""
    i = np.arange(n)
    x = affine[0, 0] * i + affine[0, 3]
    return np.rint((-x - affine[0, 3]) / affine[0, 0]).astype(int)


def positions(rng):
    """Cube centres where >= 90% of the window lies inside the template brain mask, plus the frozen boxes."""
    from nilearn.datasets import load_mni152_brain_mask
    m = load_mni152_brain_mask(resolution=1)
    mask = m.get_fdata() > 0
    w = PATCH + 2 * SHELL
    frac = ndimage.uniform_filter(mask.astype(np.float32), size=w, mode="constant")
    half = w // 2
    ok = np.zeros_like(mask)
    ok[half:-half, half:-half, half:-half] = frac[half:-half, half:-half, half:-half] >= 0.9
    cand = np.argwhere(ok)
    centres = cand[rng.choice(len(cand), N_PATCH, replace=False)]
    pos = [("random_patches", tuple(int(c - PATCH // 2) for c in ctr), (PATCH,) * 3) for ctr in centres]
    inv = np.linalg.inv(m.affine)
    for name, b in BOXES_MM.items():
        lo = np.rint(inv @ np.array([b[0][0], b[1][0], b[2][0], 1]))[:3].astype(int)
        hi = np.rint(inv @ np.array([b[0][1], b[1][1], b[2][1], 1]))[:3].astype(int)
        pos.append((name, tuple(np.minimum(lo, hi)), tuple(np.abs(hi - lo))))
    return pos, m.affine, len(cand)


def cut(vol, start, size, shell=True):
    s = SHELL if shell else 0
    sl = tuple(slice(a - s, a + n + s) for a, n in zip(start, size))
    return vol[sl]


def extract(t, pos, affine):
    """Per patch: arrays (scans, voxels) of cube, shell and mirrored cube (z-scored), and cube validity."""
    out = {f"{key}{pi}": [] for pi in range(len(pos)) for key in ("cube", "shell", "mcube", "valid", "mvalid")}
    for k, f in enumerate(t.path):
        img = nib.load(f.replace("_brain.nii.gz", ".nii.gz"))  # N4-corrected, registered intensities
        if not np.allclose(img.affine, affine, atol=1e-3):
            raise RuntimeError("scan grid differs from the template grid")
        v = img.get_fdata(dtype=np.float32)
        m = nib.load(f).get_fdata(dtype=np.float32) > 0.5     # *_brain.nii.gz is the binary brain mask
        if v[m].std() <= 0:
            raise RuntimeError("degenerate scan: no intensity variation inside the brain mask")
        z = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), 0).astype(np.float32)
        mi = mirror_index(affine, v.shape[0])
        zm, mm = z[mi], m[mi]
        for pi, (_, start, size) in enumerate(pos):
            win = cut(z, start, size)
            inner = tuple(slice(SHELL, SHELL + n) for n in size)
            sh = np.ones(win.shape, bool)
            sh[inner] = False
            out[f"cube{pi}"].append(win[inner].ravel())
            out[f"shell{pi}"].append(win[sh])
            out[f"mcube{pi}"].append(cut(zm, start, size, False).ravel())
            out[f"valid{pi}"].append(cut(m, start, size, False).mean() >= 0.95)
            out[f"mvalid{pi}"].append(cut(mm, start, size, False).mean() >= 0.95)
        if k % 25 == 0:
            print(f"extract {k + 1}/{len(t)}", flush=True)
    return {key: np.asarray(v) for key, v in out.items()}


def run(t, pos, data):
    tg = t[t.session == "ses-01"].reset_index()           # targets and donor pool
    n = len(tg)
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    row_of = {(r.person, r.session): r.Index for r in t.itertuples()}
    rows = []
    for pi, (region, _, size) in enumerate(pos):
        ix = tg["index"].to_numpy()
        X, M = data[f"cube{pi}"][ix], data[f"mcube{pi}"][ix]
        C = data[f"shell{pi}"][ix].astype(np.float64)
        valid, mvalid = data[f"valid{pi}"][ix], data[f"mvalid{pi}"][ix]
        sq = (C ** 2).sum(1)
        D = sq[:, None] + sq[None] - 2 * C @ C.T
        for i in np.flatnonzero(valid):
            others = np.flatnonzero((fam != fam[i]) & valid)
            if others.size < 10:
                continue
            mu = X[others].mean(0)
            xi = X[i]
            pools = {"population": others, "cohort": others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]}
            for c in DRAWN:
                if c == "population_both":
                    mo = np.flatnonzero((fam != fam[i]) & mvalid)
                    allp = np.concatenate([X[others], M[mo]])
                    order = np.random.default_rng([SEED, i, pi, 1]).permutation(len(allp))
                    src = allp
                else:
                    order = np.random.default_rng([SEED, i, pi, DRAWN.index(c)]).permutation(pools[c])
                    src = X
                for k in F.KS + ("all",):
                    if k == "all" and c == "population_both":
                        continue
                    sel = order if k == "all" else order[:k]
                    if k != "all" and sel.size < k:
                        continue
                    f = src[sel].mean(0)
                    rows.append((i, pi, c, k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
            nn = others[np.argsort(D[i, others])]
            for k in F.KS:
                if nn.size >= k:
                    f = X[nn[:k]].mean(0)
                    rows.append((i, pi, "knn", k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
            single = {}
            if mvalid[i]:
                single["other_hemisphere"] = M[i]
            rel = np.flatnonzero((fam == fam[i]) & (np.arange(n) != i) & valid)
            sib = [j for j in rel if age[j] != age[i]]
            twin = [j for j in rel if age[j] == age[i]]
            if sib:
                single["sibling"] = X[sib[0]]
            if twin:
                single["cotwin"] = X[twin[0]]
            j2 = row_of.get((tg.person.iloc[i], "ses-02"))
            if j2 is not None and data[f"valid{pi}"][j2]:
                single["repeat"] = data[f"cube{pi}"][j2]
            for c, f in single.items():
                rows.append((i, pi, c, 1, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
        print(f"patch {pi + 1}/{len(pos)}", flush=True)
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread"])
    r["k"] = r["k"].astype(str)
    r["person"] = fam[r.target]                           # bootstrap clusters are families
    r["region"] = [pos[p][0] for p in r.patch]
    return r, n


def finite(o):
    """Non-finite summary values (too few targets) become null instead of breaking the JSON."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [finite(v) for v in o]
    return None if isinstance(o, float) and not np.isfinite(o) else o


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    t = scans()
    pos, affine, n_cand = positions(np.random.default_rng(SEED))
    cache = os.path.join(PRIVATE, "patches.npz")
    if os.path.exists(cache):
        with np.load(cache) as z:  # plain numeric arrays only; no pickle
            data = {k: z[k] for k in z.files}
    else:
        data = extract(t, pos, affine)
        np.savez(cache, **data)
    r, n = run(t, pos, data)
    r.to_csv(os.path.join(PRIVATE, "rows.csv.gz"), index=False)
    out = {"status": "pre-QC: affine registration only, MRI registration QC not met",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md", "seed": SEED, "patch_mm": PATCH, "shell_mm": SHELL,
           "QTIM": F.summarize(r, n_targets=n, n_scans=len(t), n_people=int(t.person.nunique()),
                               n_families=int(t.family.nunique()), n_candidates=n_cand,
                               n_repeat=int((t.session == "ses-02").sum()))}
    out["QTIM"]["counts"]["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(ROOT, "results", "donor_ladder_brain.json"), "w", encoding="utf-8") as fh:
        json.dump(finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["QTIM"]["counts"]))


if __name__ == "__main__":
    main()
