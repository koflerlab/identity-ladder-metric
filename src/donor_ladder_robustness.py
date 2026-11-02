"""Robustness of the donor-averaging ladders (docs/PROTOCOL_donor_ladder_robustness.md). Pre-QC; no identity score.

Re-runs the fundus and brain ladders with one thing changed at a time: the seed (patch positions, donor draws,
bootstrap), the box (patch size, support thresholds) or the registration (fundus: disc-only or jittered
disc-fovea frame; brain: SyN on top of the affine scans, on a fixed subset). The fill rules, the cohorts and the
summary are the ones of donor_ladder_fundus / donor_ladder_brain; the baseline variant must reproduce their JSON.
Rows and SyN outputs stay in the private folder; the public table holds aggregates only.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402
import ladder2d as L  # noqa: E402
import ladder2d_register as R  # noqa: E402

SEED = F.SEED
PRIVATE = os.path.join(F.DATA, "projects", "identity-ladder-metric", "donor-ladder-robustness-20261102-v1")
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_JSON = os.path.join(ROOT, "results", "donor_ladder_robustness.json")
SEEDS = (1, 2, 3, 4, 5)
JITTER_PX = 10.0          # landmark jitter sd on the 512 image
MEM_BYTES = 6e8           # brain cube arrays held at once
SYN_FAMILIES = 15

FUNDUS_VARIANTS = {"baseline": {}, **{f"seed{s}": {"seed": s} for s in SEEDS},
                   "patch32": {"patch": 32}, "patch64": {"patch": 64},
                   "support85": {"support": 0.85}, "support95": {"support": 0.95},
                   "reg_disc": {"frame": "disc"}, "reg_jitter": {"frame": "jitter"}}
BRAIN_VARIANTS = {"baseline": {}, **{f"seed{s}": {"seed": s} for s in SEEDS},
                  "cube12": {"patch": 12}, "cube24": {"patch": 24}, "cube32": {"patch": 32},
                  "tsupport80": {"tsupport": 0.80}, "tsupport95": {"tsupport": 0.95},
                  "valid90": {"valid": 0.90}, "valid99": {"valid": 0.99}}


def fill_rows(i, pi, xi, X, order, mu, cohort, ks, rows, src=None):
    """Rows (target, patch, cohort, k, err, spread) of the k-averages of the first k donors in order."""
    src = X if src is None else src
    for k in ks:
        sel = order if k == "all" else order[:k]
        if k != "all" and sel.size < k:
            continue
        f = src[sel].mean(0)
        rows.append((i, pi, cohort, k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))


def exact_rows(i, pi, xi, pool_X, mu, rows):
    """Seed-free k = 1 expectation over the whole pool: |x - mu|^2 + pool variance; also the pool size N."""
    e = float(np.mean((xi - mu) ** 2) + np.mean((pool_X - mu) ** 2))
    rows.append((i, pi, "population_exact", "1", e, float(np.mean((pool_X - mu) ** 2))))
    rows.append((i, pi, "pool_size", "1", float(len(pool_X)), 0.0))


# ---------- fundus ----------

def canonical(rgb512, frame, r_target=None, rng=None):
    """The frame variants of ladder2d_register.fundus_canonical (same landmarks, different transform)."""
    from skimage.transform import SimilarityTransform, warp
    lm = R.fundus_landmarks(rgb512)
    if lm is None:
        return None, None
    disc, fovea = lm
    disc_right = disc[0] > fovea[0]
    if frame == "disc_fovea":
        return R.fundus_canonical(rgb512), disc_right
    if frame == "jitter":
        disc, fovea = disc + rng.normal(0, JITTER_PX, 2), fovea + rng.normal(0, JITTER_PX, 2)
        targets = (R.DISC_R, R.FOVEA_R) if disc_right else (R.FOVEA_R, R.DISC_R)
        t = SimilarityTransform()
        t.estimate(np.array(targets), np.array([disc, fovea]))
    elif frame == "disc":
        r = np.sqrt((rgb512[..., 0] > 0.1).sum() / np.pi)
        target = R.DISC_R if disc_right else np.array([rgb512.shape[1] - 1 - R.DISC_R[0], R.DISC_R[1]])
        s = r / r_target                                        # output -> input scale
        t = SimilarityTransform(scale=s, translation=disc - s * target)
    else:
        raise ValueError(frame)
    return warp(rgb512, t, order=1, mode="constant", cval=0.0).astype(np.float32), disc_right


def to_grid(c, disc_right):
    """Green channel on the 256 grid, disc right, z-scored inside the field of view (as donor_ladder_fundus.load)."""
    if not disc_right:
        c = c[:, ::-1]
    G = F.G
    g = c[..., 1].reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = (c[..., 0] > 0.1).reshape(G, 2, G, 2).all(axis=(1, 3))
    v = g[fov]
    return np.where(fov, (g - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32), fov


def fundus_images(name, t, frame):
    """(img, fov, ok, disc_right); the disc-fovea frame reuses the parent experiment's cache read-only."""
    base = os.path.join(F.PRIVATE, f"{name}_canonical.npz")
    with np.load(base, allow_pickle=False) as z:
        img, fov, ok, dr = z["img"], z["fov"], z["ok"], z["disc_right"]
    if frame == "disc_fovea":
        return img, fov, ok, dr
    cache = os.path.join(PRIVATE, f"{name}_{frame}.npz")
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            return z["img"], z["fov"], z["ok"], z["disc_right"]
    r_target = float(np.median(2 * np.sqrt(fov[ok].sum(axis=(1, 2)) / np.pi)))  # FOV radius at 512, baseline frame
    img2, fov2, ok2, dr2 = np.zeros_like(img), np.zeros_like(fov), np.zeros_like(ok), np.zeros_like(dr)
    for i, p in enumerate(t.path):
        c, d = canonical(L.load_base(p, 512, True), frame, r_target, np.random.default_rng([SEED, i]))
        if c is not None:
            img2[i], fov2[i] = to_grid(c, d)
            ok2[i], dr2[i] = True, d
    if not (np.array_equal(ok2, ok) and np.array_equal(dr2, dr)):
        raise RuntimeError("landmark detection differs from the parent run")
    np.savez_compressed(cache, img=img2, fov=fov2, ok=ok2, disc_right=dr2)
    return img2, fov2, ok2, dr2


def fundus_positions(fov, rng, patch, ring, support):
    """As donor_ladder_fundus.patch_positions, with the box parameters explicit."""
    G, w = F.G, patch + 2 * ring
    ii = np.pad(fov.astype(np.int32).cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    cand = []
    for y in range(0, G - w + 1, 4):
        for x in range(0, G - w + 1, 4):
            s = ii[:, y + w, x + w] - ii[:, y, x + w] - ii[:, y + w, x] + ii[:, y, x]
            if np.mean(s == w * w) >= support:
                cand.append((y + ring, x + ring, patch))
    if not cand:
        return [], 0
    pick = rng.choice(len(cand), size=min(F.N_PATCH, len(cand)), replace=False)
    return [cand[i] for i in sorted(pick)], len(cand)


def ring_mask(size, ring):
    m = np.ones((size + 2 * ring, size + 2 * ring), bool)
    m[ring:ring + size, ring:ring + size] = False
    return m


def fundus_rows(name, t, img, fov, ok, dr, seed=SEED, patch=F.PATCH, ring=F.RING, support=F.SUPPORT):
    """donor_ladder_fundus.run with explicit seed and box; adds the seed-free k = 1 expectation and pool sizes."""
    rng = np.random.default_rng(seed)
    u = F.units(t, ok, dr)
    full = t.assign(idx=np.arange(len(t)), ok=ok, side_conflict=ok & (dr != (t.side == "R")))
    idx = u.idx.to_numpy()
    patches, n_cand = fundus_positions(fov[idx], rng, patch, ring, support)
    if not patches:
        return None, {"n_candidates": 0}
    patches = patches + [F.METRIC]
    n = len(u)
    person, side = u.person.to_numpy(), u.side.to_numpy()
    age, sex, dev = u.age.to_numpy(float), u.sex.astype(str).to_numpy(), u.device.to_numpy()

    def pool(i, both=False, cohort=False):
        m = person != person[i]
        if not both:
            m &= side == side[i]
        if cohort:
            m &= (sex == sex[i]) & (np.abs(age - age[i]) <= 5)
            if name == "GRAPE":
                m &= dev == dev[i]
        return np.flatnonzero(m)

    fellow = np.full(n, -1)
    for i in range(n):
        j = np.flatnonzero((person == person[i]) & (side != side[i]))
        if j.size:
            fellow[i] = j[0]
    other = np.full(n, -1)
    if name == "GRAPE":
        for i in range(n):
            c = full[(full.person == person[i]) & (full.side == side[i]) & full.ok & ~full.side_conflict
                     & (full.visit != u.visit.iloc[i])]
            if len(c):
                other[i] = c.iloc[np.argmin(np.abs(c.time.to_numpy() - u.time.iloc[i]))].idx
    draws = {(c, i): np.random.default_rng([seed, i, F.DRAWN.index(c)]).permutation(
        pool(i, both=c == "population_both", cohort=c == "cohort")) for c in F.DRAWN for i in range(n)}

    rows = []
    for pi, (y, x, s) in enumerate(patches):
        rg = ring
        win = (slice(y, y + s), slice(x, x + s))
        X = img[idx][:, win[0], win[1]].reshape(n, -1)
        valid = fov[idx][:, win[0], win[1]].all(axis=(1, 2))
        ry, rx = slice(y - rg, y + s + rg), slice(x - rg, x + s + rg)
        C = img[idx][:, ry, rx][:, ring_mask(s, rg)]
        ring_valid = fov[idx][:, ry, rx][:, ring_mask(s, rg)].all(axis=1)
        sq = (C ** 2).sum(1)
        D = sq[:, None] + sq[None] - 2 * C @ C.T
        Xo = img[:, win[0], win[1]].reshape(len(img), -1)
        vo = fov[:, win[0], win[1]].all(axis=(1, 2))
        for i in range(n):
            if not valid[i]:
                continue
            same = pool(i)
            mu_set = same[valid[same]]
            if mu_set.size < 2:
                continue
            mu, xi = X[mu_set].mean(0), X[i]
            exact_rows(i, pi, xi, X[mu_set], mu, rows)
            for c in F.DRAWN:
                order = draws[c, i]
                order = order[valid[order]]
                ks = F.KS if c == "population_both" else F.KS + ("all",)
                fill_rows(i, pi, xi, X, order, mu, c, ks, rows)
            if ring_valid[i]:
                cand = same[valid[same] & ring_valid[same]]
                fill_rows(i, pi, xi, X, cand[np.argsort(D[i, cand])], mu, "knn", F.KS, rows)
            j = fellow[i]
            if j >= 0 and valid[j]:
                fill_rows(i, pi, xi, X, np.array([j]), mu, "fellow_eye", (1,), rows)
            j = other[i]
            if j >= 0 and vo[j]:
                fill_rows(i, pi, xi, Xo, np.array([j]), mu, "other_visit", (1,), rows)
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread"])
    r["k"] = r["k"].astype(str)
    r["person"] = person[r.target]
    r["region"] = np.where(r.patch == len(patches) - 1, "metric_box", "random_patches")
    return r, {"n_units": n, "n_candidates": n_cand, "n_patches": len(patches) - 1}


# ---------- brain ----------

def brain_positions(rng, patch=B.PATCH, tsupport=0.9):
    """As donor_ladder_brain.positions, with the cube side and template support explicit."""
    from nilearn.datasets import load_mni152_brain_mask
    from scipy import ndimage
    m = load_mni152_brain_mask(resolution=1)
    mask = m.get_fdata() > 0
    w = patch + 2 * B.SHELL
    frac = ndimage.uniform_filter(mask.astype(np.float32), size=w, mode="constant")
    half = w // 2
    ok = np.zeros_like(mask)
    ok[half:-half, half:-half, half:-half] = frac[half:-half, half:-half, half:-half] >= tsupport
    cand = np.argwhere(ok)
    centres = cand[rng.choice(len(cand), B.N_PATCH, replace=False)]
    pos = [("random_patches", tuple(int(c - patch // 2) for c in ctr), (patch,) * 3) for ctr in centres]
    inv = np.linalg.inv(m.affine)
    for name, b in B.BOXES_MM.items():
        lo = np.rint(inv @ np.array([b[0][0], b[1][0], b[2][0], 1]))[:3].astype(int)
        hi = np.rint(inv @ np.array([b[0][1], b[1][1], b[2][1], 1]))[:3].astype(int)
        pos.append((name, tuple(np.minimum(lo, hi)), tuple(np.abs(hi - lo))))
    return pos, m.affine, len(cand)


def load_scan(path_img, path_mask, affine):
    import nibabel as nib
    img = nib.load(path_img)
    if not np.allclose(img.affine, affine, atol=1e-3):
        raise RuntimeError("scan grid differs from the template grid")
    v = img.get_fdata(dtype=np.float32)
    m = nib.load(path_mask).get_fdata(dtype=np.float32) > 0.5
    if v[m].std() <= 0:
        raise RuntimeError("degenerate scan: no intensity variation inside the brain mask")
    return np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), 0).astype(np.float32), m


def brain_extract(paths, pos, affine, valid_thr):
    """donor_ladder_brain.extract for a subset of positions; the shell only feeds the kNN distance matrix."""
    out = {f"{key}{pi}": [] for pi in range(len(pos)) for key in ("cube", "shell", "mcube", "valid", "mvalid")}
    for img_path, mask_path in paths:
        z, m = load_scan(img_path, mask_path, affine)
        mi = B.mirror_index(affine, z.shape[0])
        zm, mm = z[mi], m[mi]
        for pi, (_, start, size) in enumerate(pos):
            win = B.cut(z, start, size)
            inner = tuple(slice(B.SHELL, B.SHELL + n) for n in size)
            sh = np.ones(win.shape, bool)
            sh[inner] = False
            out[f"cube{pi}"].append(win[inner].ravel())
            out[f"shell{pi}"].append(win[sh])
            out[f"mcube{pi}"].append(B.cut(zm, start, size, False).ravel())
            out[f"valid{pi}"].append(B.cut(m, start, size, False).mean() >= valid_thr)
            out[f"mvalid{pi}"].append(B.cut(mm, start, size, False).mean() >= valid_thr)
    return {key: np.asarray(v) for key, v in out.items()}


def brain_patch_rows(pi, region, d, tg, row_of, seed, rows):
    """donor_ladder_brain.run for one position, from that position's arrays d (all scans of the table)."""
    n = len(tg)
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    ix = tg["index"].to_numpy()
    X, M = d["cube"][ix], d["mcube"][ix]
    C = d["shell"][ix].astype(np.float64)
    valid, mvalid = d["valid"][ix], d["mvalid"][ix]
    sq = (C ** 2).sum(1)
    D = sq[:, None] + sq[None] - 2 * C @ C.T
    for i in np.flatnonzero(valid):
        others = np.flatnonzero((fam != fam[i]) & valid)
        if others.size < 10:
            continue
        mu, xi = X[others].mean(0), X[i]
        exact_rows(i, pi, xi, X[others], mu, rows)
        pools = {"population": others,
                 "cohort": others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]}
        for c in F.DRAWN:
            if c == "population_both":
                mo = np.flatnonzero((fam != fam[i]) & mvalid)
                allp = np.concatenate([X[others], M[mo]])
                order = np.random.default_rng([seed, i, pi, 1]).permutation(len(allp))
                fill_rows(i, pi, xi, allp, order, mu, c, F.KS, rows)
            else:
                order = np.random.default_rng([seed, i, pi, F.DRAWN.index(c)]).permutation(pools[c])
                fill_rows(i, pi, xi, X, order, mu, c, F.KS + ("all",), rows)
        fill_rows(i, pi, xi, X, others[np.argsort(D[i, others])], mu, "knn", F.KS, rows)
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
        if j2 is not None and d["valid"][j2]:
            single["repeat"] = d["cube"][j2]
        for c, f in single.items():
            rows.append((i, pi, c, "1", float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))


def brain_rows(t, paths, pos, affine, seed=SEED, valid=0.95):
    """All rows of one brain variant, extracting the positions in chunks that fit in MEM_BYTES."""
    tg = t[t.session == "ses-01"].reset_index()
    row_of = {(r.person, r.session): r.Index for r in t.itertuples()}
    per_pos = [len(paths) * 4 * (np.prod(s) + np.prod(np.add(s, 2 * B.SHELL))) for _, _, s in pos]  # float32 bytes
    rows, chunk, used = [], [], 0.0
    for pi in range(len(pos) + 1):
        if pi == len(pos) or (chunk and used + per_pos[pi] > MEM_BYTES):
            data = brain_extract(paths, [pos[j] for j in chunk], affine, valid)
            for local, pj in enumerate(chunk):
                d = {key: data[f"{key}{local}"] for key in ("cube", "shell", "mcube", "valid", "mvalid")}
                brain_patch_rows(pj, pos[pj][0], d, tg, row_of, seed, rows)
            print(f"  positions {chunk[0] + 1}-{chunk[-1] + 1}/{len(pos)}", flush=True)
            chunk, used = [], 0.0
        if pi < len(pos):
            chunk.append(pi)
            used += per_pos[pi]
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread"])
    r["k"] = r["k"].astype(str)
    r["person"] = tg.family.to_numpy()[r.target]       # bootstrap clusters are families
    r["region"] = [pos[p][0] for p in r.patch]
    return r, len(tg)


# ---------- SyN subset ----------

def syn_subset(t):
    """Every person with a second session (both sessions) plus SYN_FAMILIES families with >= 2 session-1 members."""
    rep = set(t.person[t.session == "ses-02"])
    s1 = t[t.session == "ses-01"]
    fams = sorted(f for f, g in s1.groupby("family") if len(g) >= 2)
    pick = np.random.default_rng(SEED).choice(len(fams), SYN_FAMILIES, replace=False)
    keep_people = set(rep)
    for f in (fams[i] for i in sorted(pick)):
        keep_people |= set(s1[s1.family == f].sort_values("person").person.iloc[:2])
    return t[t.person.isin(keep_people)].reset_index(drop=True)


def syn_one(job):
    """SyN on the 2 mm template grid from the affine scan, applied to the 1 mm intensities and mask."""
    i, img_path, mask_path = job
    dest = os.path.join(PRIVATE, "syn", f"s{i:03d}")
    if os.path.exists(dest + "_mask.nii.gz"):
        return 0.0
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
    import tempfile
    import ants
    tdir = os.path.join(B.PROJ, "real-cases-run", "template")
    tpl1 = ants.image_read(os.path.join(tdir, "template_1mm.nii.gz"))
    tpl2 = ants.image_read(os.path.join(tdir, "template_2mm.nii.gz"))
    br2 = ants.image_read(os.path.join(tdir, "brain_2mm.nii.gz"))
    mov, msk = ants.image_read(img_path), ants.image_read(mask_path)
    mov2 = ants.resample_image_to_target(mov * msk, tpl2, interp_type="linear")
    start, work = time.monotonic(), tempfile.mkdtemp(prefix="dlsyn_")
    reg = ants.registration(fixed=tpl2 * br2, moving=mov2, type_of_transform="SyNOnly", initial_transform="Identity",
                            syn_metric="CC", syn_sampling=2, reg_iterations=(60, 30, 10), grad_step=0.2,
                            flow_sigma=3, total_sigma=0, random_seed=123, outprefix=os.path.join(work, "s_"))
    w = ants.apply_transforms(tpl1, mov, reg["fwdtransforms"], interpolator="linear")
    wm = ants.apply_transforms(tpl1, msk, reg["fwdtransforms"], interpolator="nearestNeighbor")
    ants.image_write(w, dest + ".nii.gz")
    ants.image_write(wm, dest + "_mask.nii.gz")
    return time.monotonic() - start


def run_syn(t, workers=2):
    os.makedirs(os.path.join(PRIVATE, "syn"), exist_ok=True)
    jobs = [(i, p.replace("_brain.nii.gz", ".nii.gz"), p) for i, p in enumerate(t.path)]
    with ProcessPoolExecutor(workers) as ex:
        secs = list(ex.map(syn_one, jobs))
    print({"syn_scans": len(jobs), "median_seconds": float(np.median(secs))}, flush=True)


# ---------- headline numbers ----------

def headline(reg: dict, n_pool: float) -> dict:
    """The protocol's headline numbers for one region summary of donor_ladder_fundus.summarize."""
    c = reg["cohorts"]
    h = {}
    pop = c.get("population", {})
    if "1" in pop:
        h["stranger_k1"] = pop["1"]["err_over_V"]
        h["stranger_k1_ci"] = pop["1"]["ci95"]
    if "1" in c.get("population_exact", {}):
        h["stranger_k1_exact"] = c["population_exact"]["1"]["err_over_V"]
    if "fit" in pop:
        h["population_R2"], h["population_rms"] = pop["fit"]["R2"], pop["fit"]["rms_residual"]
    keff = reg.get("k_eff_population", {})
    ratios = {k: keff[k] / (int(k) * (n_pool - 1) / (n_pool - int(k))) for k in keff
              if keff[k] and int(k) < n_pool and int(k) <= 32}
    if ratios:
        h["keff_over_pred_min"], h["keff_over_pred_max"] = min(ratios.values()), max(ratios.values())
        h["keff_over_pred"] = ratios
    if "fit" in c.get("cohort", {}):
        h["cohort_R2"] = c["cohort"]["fit"]["R2"]
    knn = {int(k): v["err_over_V"] for k, v in c.get("knn", {}).items() if k.isdigit()}
    if len(knn) >= 3:
        best = min(knn, key=knn.get)
        h["knn_best_k"], h["knn_best"] = best, knn[best]
        h["knn_u_shape"] = bool(min(knn) < best < max(knn))
    for rung in ("fellow_eye", "other_hemisphere", "other_visit", "repeat", "sibling", "cotwin"):
        if "1" in c.get(rung, {}):
            v = c[rung]["1"]
            h[f"r_{rung}"] = 1 - v["err_over_V"] / 2
            h[f"r_{rung}_ci"] = [1 - v["ci95"][1] / 2, 1 - v["ci95"][0] / 2]
            h[f"n_{rung}"] = v["n_targets"]
    return h


def summarize_variant(r: pd.DataFrame, seed: int, **counts) -> dict:
    pool = r[r.cohort == "pool_size"].groupby("region").err.mean()
    s = F.summarize(r[r.cohort != "pool_size"], seed=seed, **counts)
    return {"counts": counts, "regions": {reg: {"V": v["V_mean_error_of_population_mean"],
                                                "n_pool_mean": float(pool[reg]),
                                                "headline": headline(v, float(pool[reg])),
                                                "summary": v} for reg, v in s["regions"].items()}}


# ---------- drivers ----------

def partial_path(part):
    """Each driver writes its own aggregate file, so parallel drivers never race; assemble() merges them."""
    return os.path.join(PRIVATE, f"partial_{part}.json")


def store(part, keys, value):
    p = partial_path(part)
    out = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    d = out
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(B.finite(out), fh, indent=1, allow_nan=False)


def fundus(variants):
    for name in ("PAPILA", "GRAPE"):
        t = F.papila_table() if name == "PAPILA" else F.grape_table()
        for v in variants:
            p = FUNDUS_VARIANTS[v]
            t0 = time.time()
            img, fov, ok, dr = fundus_images(name, t, {"disc": "disc", "jitter": "jitter"}.get(p.get("frame"), "disc_fovea"))
            seed = p.get("seed", SEED)
            r, counts = fundus_rows(name, t, img, fov, ok, dr, seed=seed, patch=p.get("patch", F.PATCH),
                                    support=p.get("support", F.SUPPORT))
            if r is None:
                store("fundus", (name, v), {"counts": counts, "no_window": True})
            else:
                r.to_csv(os.path.join(PRIVATE, f"fundus_{name}_{v}.csv.gz"), index=False)
                store("fundus", (name, v), summarize_variant(r, seed, **counts))
            print(f"{name} {v}: {time.time() - t0:.0f}s", flush=True)


def brain(variants):
    t = B.scans()
    for v in variants:
        p = BRAIN_VARIANTS[v]
        t0 = time.time()
        seed = p.get("seed", SEED)
        pos, affine, n_cand = brain_positions(np.random.default_rng(seed), p.get("patch", B.PATCH), p.get("tsupport", 0.9))
        paths = [(f.replace("_brain.nii.gz", ".nii.gz"), f) for f in t.path]
        r, n = brain_rows(t, paths, pos, affine, seed=seed, valid=p.get("valid", 0.95))
        r.to_csv(os.path.join(PRIVATE, f"brain_{v}.csv.gz"), index=False)
        store("brain", ("QTIM", v), summarize_variant(r, seed, n_targets=n, n_candidates=n_cand))
        print(f"brain {v}: {time.time() - t0:.0f}s", flush=True)


def brain_syn(workers):
    t = syn_subset(B.scans())
    run_syn(t, workers)
    pos, affine, _ = brain_positions(np.random.default_rng(SEED))
    arms = {"reg_affine_subset": [(f.replace("_brain.nii.gz", ".nii.gz"), f) for f in t.path],
            "reg_syn_subset": [(os.path.join(PRIVATE, "syn", f"s{i:03d}.nii.gz"),
                                os.path.join(PRIVATE, "syn", f"s{i:03d}_mask.nii.gz")) for i in range(len(t))]}
    for v, paths in arms.items():
        r, n = brain_rows(t, paths, pos, affine)
        r.to_csv(os.path.join(PRIVATE, f"brain_{v}.csv.gz"), index=False)
        store("syn", ("QTIM", v), summarize_variant(r, SEED, n_targets=n, n_scans=len(t),
                                                    n_families=int(t.family.nunique()),
                                                    n_repeat=int((t.session == "ses-02").sum())))
        print(f"brain {v} done", flush=True)


# ---------- assembly ----------

FAMILY = {"baseline": "baseline", "reg_disc": "registration", "reg_jitter": "registration",
          "reg_affine_subset": "registration", "reg_syn_subset": "registration"}
METRICS = ("stranger_k1", "stranger_k1_exact", "population_R2", "population_rms", "keff_over_pred_min",
           "keff_over_pred_max", "cohort_R2", "knn_best_k", "knn_best", "knn_u_shape", "r_fellow_eye",
           "r_other_hemisphere", "r_other_visit", "r_repeat", "r_sibling", "r_cotwin")
# (conclusion, metric, "holds" band of the protocol, direction that must not flip)
RULES = (("one stranger costs about twice the mean", "stranger_k1", lambda v: 1.8 <= v <= 2.2, lambda v: v > 1),
         ("population rung follows 1 + 1/k (R2)", "population_R2", lambda v: abs(v) < 0.05, lambda v: abs(v) < 0.5),
         ("population rung follows 1 + 1/k (rms)", "population_rms", lambda v: v < 0.05, lambda v: v < 0.5),
         ("finite-pool k_eff, lowest ratio", "keff_over_pred_min", lambda v: 0.85 <= v <= 1.15, lambda v: v > 0.5),
         ("finite-pool k_eff, highest ratio", "keff_over_pred_max", lambda v: 0.85 <= v <= 1.15, lambda v: v < 2),
         ("cohort explains little", "cohort_R2", lambda v: v < 0.15, lambda v: v < 0.5),
         ("kNN best k in [4, 32]", "knn_best_k", lambda v: 4 <= v <= 32, lambda v: 2 <= v <= 64),
         ("kNN U-shape", "knn_u_shape", lambda v: bool(v), lambda v: bool(v)),
         ("fellow eye loses to the mean", "r_fellow_eye", lambda v: v < 0.5, lambda v: v < 0.5),
         ("other hemisphere vs the mean (per region)", "r_other_hemisphere", lambda v: v < 0.5, lambda v: v < 0.5),
         ("other visit beats the mean", "r_other_visit", lambda v: v > 0.5, lambda v: v > 0.5),
         ("repeat scan beats the mean", "r_repeat", lambda v: v > 0.5, lambda v: v > 0.5),
         ("sibling loses to the mean", "r_sibling", lambda v: v < 0.5, lambda v: v < 0.5))


def table(merged: dict) -> pd.DataFrame:
    """Long table: one row per dataset, variant, region and headline number (aggregates only)."""
    rows = []
    for part in ("fundus", "brain", "syn"):
        for ds, variants in merged.get(part, {}).items():
            for v, res in variants.items():
                fam = FAMILY.get(v, "seed" if v.startswith("seed") else "box")
                for region, reg in res.get("regions", {}).items():
                    h = reg["headline"]
                    for m in METRICS:
                        if m in h:
                            ci = h.get(f"{m}_ci", [None, None])
                            rows.append({"dataset": ds, "variant": v, "family": fam, "region": region, "metric": m,
                                         "value": float(h[m]), "ci_lo": ci[0], "ci_hi": ci[1]})
    return pd.DataFrame(rows)


def verdicts(tab: pd.DataFrame) -> list:
    """Rule counts per conclusion and region, over every variant and dataset where the number is defined."""
    out = []
    for name, m, band, direction in RULES:
        for region, g in tab[tab.metric == m].groupby("region"):
            inb, kept = g.value.map(band), g.value.map(direction)
            verdict = "holds" if inb.all() else ("sensitive" if kept.all() else "breaks")
            out.append({"conclusion": name, "metric": m, "region": region, "verdict": verdict, "n": int(len(g)),
                        "n_in_band": int(inb.sum()), "n_direction_kept": int(kept.sum()),
                        "min": float(g.value.min()), "max": float(g.value.max()),
                        "out_of_band": sorted(f"{d}/{v}" for d, v in zip(g.dataset[~inb], g.variant[~inb]))})
    return out


def figure(tab: pd.DataFrame, dest: str):
    """One panel per headline number: every variant as a dot, coloured by perturbation family, per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"baseline": "#0b0b0b", "seed": "#8a8986", "box": "#2a78d6", "registration": "#e8590c"}
    panels = (("stranger_k1", "stranger k=1, error/V", (2.0,)), ("population_R2", "population R2", (0.0,)),
              ("keff_over_pred_max", "max k_eff / finite-pool pred.", (1.0,)), ("cohort_R2", "cohort R2", (0.0,)),
              ("knn_best_k", "kNN best k", ()), ("r_fellow_eye", "r, own mirror side", (0.5,)),
              ("r_other_visit", "r, other visit / repeat", (0.5,)), ("r_sibling", "r, sibling", (0.5,)))
    t = tab.copy()
    t.loc[t.metric == "r_other_hemisphere", "metric"] = "r_fellow_eye"
    t.loc[t.metric == "r_repeat", "metric"] = "r_other_visit"
    groups = [("PAPILA", "random_patches"), ("PAPILA", "metric_box"), ("GRAPE", "random_patches"),
              ("GRAPE", "metric_box"), ("QTIM", "random_patches"), ("QTIM", "deep"), ("QTIM", "parietal")]
    labels = ["PAPILA", "PAP box", "GRAPE", "GRA box", "QTIM", "deep", "parietal"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6))
    rng = np.random.default_rng(0)
    for ax, (m, title, refs) in zip(axes.flat, panels):
        for r in refs:
            ax.axhline(r, color="#c9c8c4", lw=0.7, zorder=0)
        for gi, (ds, region) in enumerate(groups):
            g = t[(t.metric == m) & (t.dataset == ds) & (t.region == region)]
            for fam, gg in g.groupby("family"):
                x = gi + rng.uniform(-0.2, 0.2, len(gg)) * (fam != "baseline")
                ax.scatter(x, gg.value, s=10 if fam == "baseline" else 6, c=colours[fam], lw=0,
                           marker="D" if fam == "baseline" else "o", zorder=3 if fam == "baseline" else 2)
        ax.set_xticks(range(len(groups)), labels, rotation=60)
        ax.set_title(title, fontsize=7)
        if m == "knn_best_k":
            ax.set_yscale("log", base=2)
    handles = [plt.Line2D([], [], ls="", marker="D" if f == "baseline" else "o", color=c, label=f)
               for f, c in colours.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Donor ladders under seed, box and registration perturbations (pre-QC)", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dest, f"fig_donor_ladder_robustness.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def assemble():
    merged = {"status": "pre-QC: registration visual QC not met; metric pause lifted for donor ladders only",
              "protocol": "docs/PROTOCOL_donor_ladder_robustness.md"}
    for part in ("fundus", "brain", "syn"):
        if os.path.exists(partial_path(part)):
            merged[part] = json.load(open(partial_path(part), encoding="utf-8"))
    tab = table(merged)
    merged["verdicts"] = verdicts(tab)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(B.finite(merged), fh, indent=1, allow_nan=False)
    tab.to_csv(OUT_JSON.replace(".json", ".csv"), index=False, float_format="%.4f")
    figure(tab, os.path.join(ROOT, "results", "figures"))
    print(pd.DataFrame(merged["verdicts"]).drop(columns="out_of_band").to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("fundus", "brain", "syn", "assemble"))
    ap.add_argument("--variants", default="")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(PRIVATE, exist_ok=True)
    if a.part == "assemble":
        assemble()
        return
    if a.part == "syn":
        brain_syn(a.workers)
        return
    allv = FUNDUS_VARIANTS if a.part == "fundus" else BRAIN_VARIANTS
    vs = a.variants.split(",") if a.variants else list(allv)
    (fundus if a.part == "fundus" else brain)(vs)


if __name__ == "__main__":
    main()
