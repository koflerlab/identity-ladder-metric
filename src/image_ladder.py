"""Image-space identity ladder pilot on data already on disk: L0 to L4 plus the mirror half, in two feature spaces.

Scans: SIMON (one person; within-session repeats = L0, other sessions on the same or another scanner = L1/L2),
QTIM local subset (adolescents; second visit about 3.5 years later = L2, siblings = related rung, others = L4)
and IXI (unrelated adults from three centres = L4).

Every T1w is resampled to 2 mm and bias-corrected (N4). The symmetric MNI ICBM152 2009a template bundled with
nilearn is skull-stripped, so each scan is skull-stripped first (Otsu threshold, erosion, largest component,
dilation; several erosion settings are tried and the one whose registration best matches the template is kept),
then registered brain-to-brain (affine + SyN, ANTs defaults, 2 mm). A first version registered whole heads to
this brain-only template; its affine stage failed and its results were discarded (run-2026-10-31, flagged INVALID).
Scans whose brain correlation with the template stays below QC_MIN_CORR are excluded and counted.
Two features are read on a 6 mm grid that is symmetric about x = 0 and restricted to the template brain mask:
  * intensity: the warped image, scaled by its median and interquartile range inside the brain mask;
  * geometry: the log-Jacobian of the non-linear warp (local volume relative to the template, after the affine).
Intensity carries scanner contrast directly; geometry was expected to be less sensitive to it, but the audit of
2026-11-01 showed that geometry also depends strongly on protocol and centre, so neither space is protocol-free.
Mirror half: the left half of one scan against the reflected right half of ANOTHER scan of the same person, versus the
same half on that other scan and versus another person's reflected half. Within one scan both halves share every
scan-level nuisance, so the within-scan mirror distance is reported only as a descriptive number.
SIMON L0 counts only within-session repeats acquired with the same protocol (TR, TI, flip angle, slice thickness from
the BIDS sidecars); within-session pairs of different T1 sequences form their own rung.

Per-scan features (derived, person-level) go to C:/Projects/data/projects/identity-ladder-metric/<run>/;
aggregate rung statistics go to results/image_ladder.json. No image, filename or participant ID is printed.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA = r"C:\Projects\data"
SIMON_DIR = os.path.join(DATA, "brain_images", "SIMON_data", "SIMON_BIDS")
QTIM_DIR = os.path.join(DATA, "brain_images", "siblings")
IXI_DIR = os.path.join(DATA, "datasets", "neuroimaging", "ixi", "version-unverified", "raw")
RUN = "run-2026-11-01-image-ladder-v2"
PRIVATE = os.path.join(DATA, "projects", "identity-ladder-metric", RUN)
IXI_PER_CENTRE = 33
BLOCK = 3  # 2 mm template voxels per feature voxel along each axis -> 6 mm
MIN_BRAIN_FRACTION = 0.5
FEATURES = ("intensity", "geometry")
STRIP_SETTINGS = ({"erode": 3, "frac": 1.0}, {"erode": 5, "frac": 0.8}, {"erode": 4, "frac": 0.9})
QC_MIN_CORR = 0.5


# ---------------------------------------------------------------------------------------------------- grid helpers

def block_mean(vol: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Average non-overlapping block^3 cubes; trailing slices that do not fill a block are dropped."""
    sx, sy, sz = (s // block * block for s in vol.shape)
    v = vol[:sx, :sy, :sz]
    return v.reshape(sx // block, block, sy // block, block, sz // block, block).mean(axis=(1, 3, 5))


def mirror_halves(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(left half, reflected right half) of a volume whose x axis is symmetric about its central index.

    On the 2 mm MNI grid x runs from -98 to +98 mm over 99 voxels, so index 49 is x = 0; after 3-voxel blocks the
    central block 16 is centred on x = 0, and flipping axis 0 reflects across the midsagittal plane exactly.
    The central column is excluded from both halves.
    """
    nx = vol.shape[0]
    if nx % 2 == 0:
        raise ValueError("mirror_halves needs an odd number of x columns centred on the midline")
    half = nx // 2
    return vol[:half], vol[::-1][:half]


# ---------------------------------------------------------------------------------------------------- manifest

def protocol_key(sidecar: dict) -> str:
    """Acquisition signature of a T1w series: TR, TI, TE (seconds), flip angle and slice thickness.

    BIDS asks for seconds, but some sidecars give TR/TI/TE in ms; values above 100 (TR, TI) or above 1 (TE) are
    treated as ms. Missing values are kept as 'na' (Philips sidecars often lack InversionTime)."""
    def sec(v, ms_above):
        if v is None:
            return None
        v = float(v)
        return v / 1000.0 if v > ms_above else v

    def fmt(v, nd):
        return "na" if v is None else f"{round(float(v), nd)}"
    return "|".join([fmt(sec(sidecar.get("RepetitionTime"), 100), 4), fmt(sec(sidecar.get("InversionTime"), 100), 3),
                     fmt(sec(sidecar.get("EchoTime"), 1), 4), fmt(sidecar.get("FlipAngle"), 1),
                     fmt(sidecar.get("SliceThickness"), 2)])


def simon_manifest() -> pd.DataFrame:
    import simon_e0

    meta = simon_e0.pheno()
    rows = []
    for f in sorted(glob.glob(os.path.join(SIMON_DIR, "sub-*", "ses-*", "anat", "*_T1w.nii.gz"))):
        ses = re.search(r"(ses-\d+)", os.path.basename(f)).group(1)
        run = re.search(r"run-(\d+)", os.path.basename(f))
        side = f.replace(".nii.gz", ".json")
        sidecar = json.load(open(side, encoding="utf-8")) if os.path.exists(side) else {}
        rows.append({"dataset": "SIMON", "person": "simon", "family": "simon", "session": ses,
                     "run": int(run.group(1)) if run else 1, "path": f, "protocol": protocol_key(sidecar),
                     "series": sidecar.get("SeriesDescription", ""),
                     "scanner": meta.at[ses, "scanner"] if ses in meta.index else None,
                     "device": str(sidecar.get("DeviceSerialNumber") or (meta.at[ses, "scanner"] if ses in meta.index else "")),
                     "date": meta.at[ses, "date"] if ses in meta.index else pd.NaT})
    m = pd.DataFrame(rows)
    m["years"] = (m["date"] - m["date"].min()).dt.days / 365.25
    return m


def qtim_manifest() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(QTIM_DIR, "participants.tsv"), sep="\t").set_index("participant_id")
    rows = []
    for f in sorted(glob.glob(os.path.join(QTIM_DIR, "sub-*", "ses-*", "anat", "*_T1w.nii.gz"))):
        sub, ses = f.split(os.sep)[-4], f.split(os.sep)[-3]
        age = p.at[sub, "age"] if ses == "ses-01" else p.at[sub, "age_ses02"]
        rows.append({"dataset": "QTIM", "person": sub, "family": str(p.at[sub, "family_id"]), "session": ses, "run": 1,
                     "path": f, "scanner": "QTIM", "years": float(age)})
    return pd.DataFrame(rows)


def ixi_manifest(per_centre: int = IXI_PER_CENTRE, seed: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(IXI_DIR, "**", "IXI*-T1.nii.gz"), recursive=True))
    t = pd.DataFrame({"path": files})
    t["person"] = t.path.map(lambda f: os.path.basename(f).split("-")[0])
    t["scanner"] = t.path.map(lambda f: os.path.basename(f).split("-")[1])
    t = t.drop_duplicates("person")
    if t.groupby("scanner").size().min() < per_centre:
        raise ValueError("an IXI centre has fewer subjects than the requested sample")
    t = t.groupby("scanner").sample(n=per_centre, random_state=seed)
    return t.assign(dataset="IXI", family=t.person, session="ses-1", run=1, years=np.nan)


def manifest() -> pd.DataFrame:
    m = pd.concat([simon_manifest(), qtim_manifest(), ixi_manifest()], ignore_index=True)
    m["scan"] = [f"scan{i:04d}" for i in range(len(m))]  # pseudonymous key used everywhere downstream
    return m


# ---------------------------------------------------------------------------------------------------- features

class Extractor:
    """Registration and feature extraction against the bundled symmetric MNI template."""

    def __init__(self, workdir: str):
        import ants
        import nibabel as nib
        from nilearn import datasets

        ants.config._random_seed = 123  # ANTs sampling seed; multi-threading can still add tiny differences
        self.ants = ants
        os.makedirs(workdir, exist_ok=True)
        tpl_path, mask_path = os.path.join(workdir, "mni2009a_sym_2mm.nii.gz"), os.path.join(workdir, "mni2009a_sym_2mm_mask.nii.gz")
        nib.save(datasets.load_mni152_template(resolution=2), tpl_path)
        nib.save(datasets.load_mni152_brain_mask(resolution=2), mask_path)
        self.t2 = ants.image_read(tpl_path)
        brain2 = ants.image_read(mask_path).numpy() > 0
        if brain2.shape[0] != 99:
            raise ValueError("unexpected template grid; mirror_halves assumes 99 x columns at 2 mm")
        outside = self.t2.numpy()[~brain2]
        if np.percentile(outside, 99) > 0.05 * np.percentile(self.t2.numpy()[brain2], 50):
            raise ValueError("template has signal outside the brain mask; the brain-to-brain registration assumes it is skull-stripped")
        self.mask6 = block_mean(brain2.astype(float)) >= MIN_BRAIN_FRACTION
        self.brain2 = brain2

    def _register(self, img) -> tuple[dict, np.ndarray, float]:
        r = self.ants.registration(fixed=self.t2, moving=img, type_of_transform="SyN")
        warped = self.ants.apply_transforms(self.t2, img, r["fwdtransforms"], interpolator="linear").numpy()
        corr = float(np.corrcoef(warped[self.brain2], self.t2.numpy()[self.brain2])[0, 1])
        return r, warped, corr

    def __call__(self, path: str) -> dict:
        ants = self.ants
        img = ants.resample_image(ants.image_read(path), (2, 2, 2), use_voxels=False, interp_type=0)
        img = ants.n4_bias_field_correction(img, shrink_factor=4)
        best = None
        for k, s in enumerate(STRIP_SETTINGS):
            try:
                stripped, _ = strip_skull(img.numpy(), **s)
            except ValueError:
                continue
            r, warped, corr = self._register(img.new_image_like(stripped))
            if best is None or corr > best[2]:
                best = (r, warped, corr, k)
        if best is None:
            raise ValueError("skull stripping failed for every setting")
        r, warped, corr, k = best
        logjac = ants.create_jacobian_determinant_image(self.t2, r["fwdtransforms"][0], do_log=True).numpy()
        inside = warped[self.brain2]
        med, iqr = np.median(inside), np.subtract(*np.percentile(inside, [75, 25]))
        if not np.isfinite(iqr) or iqr <= 0:
            raise FloatingPointError("degenerate intensity range inside the brain mask")
        affine = ants.read_transform(r["fwdtransforms"][1]).parameters[:9].reshape(3, 3)
        return {"intensity": block_mean((warped - med) / iqr).astype(np.float32),
                "geometry": block_mean(logjac).astype(np.float32),
                "qc_template_corr": corr, "strip_setting": k,
                "log_affine_scale": float(np.log(abs(np.linalg.det(affine))))}


def strip_skull(a: np.ndarray, erode: int = 3, frac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Crude brain extraction for 2 mm T1: Otsu threshold (scaled by frac), erosion to cut the brain from the scalp,
    largest connected component, dilation back by erode + 1 voxels, hole filling. Returns (stripped image, mask)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    pos = a[a > 0]
    if pos.size == 0:
        raise ValueError("empty image")
    m = ndimage.binary_erosion(a > threshold_otsu(pos) * frac, iterations=erode)
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("erosion removed the whole head")
    m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    m = ndimage.binary_fill_holes(ndimage.binary_dilation(m, iterations=erode + 1))
    return (a * m).astype(a.dtype), m


def extract_all(m: pd.DataFrame, out_dir: str) -> tuple[dict, pd.DataFrame]:
    """Features for every scan, cached per scan in out_dir; a smoke test on the first scan runs before the loop."""
    ex = Extractor(os.path.join(out_dir, "template"))
    feats, qc = {}, []

    def one(row):
        cache = os.path.join(out_dir, "features", f"{row.scan}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            return {k: z[k] for k in z.files}
        f = ex(row.path)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **{k: np.asarray(v) for k, v in f.items()})
        return f

    first = m.iloc[0]
    t0 = time.time()
    f0 = one(first)
    per = time.time() - t0
    for k in FEATURES:
        assert f0[k].shape == ex.mask6.shape and np.isfinite(f0[k][ex.mask6]).all(), f"smoke test failed for {k}"
    print(f"smoke test: {per:.0f}s/item; estimated total {per * len(m) / 60:.0f} min ({per * len(m) / 3600:.1f} h) "
          f"for {len(m)} items; device: CPU only (no GPU)")
    failed = []
    for i, row in enumerate(m.itertuples()):
        try:
            f = f0 if i == 0 else one(row)
        except Exception as e:  # recorded with the scan key only, then the run stops if failures are not rare
            failed.append((row.scan, type(e).__name__))
            if len(failed) > 0.05 * len(m):
                raise RuntimeError(f"feature extraction failed for {len(failed)} scans: {failed[:5]}")
            continue
        feats[row.scan] = {k: f[k] for k in FEATURES}
        qc.append({"scan": row.scan, "qc_template_corr": float(f["qc_template_corr"]),
                   "strip_setting": int(f["strip_setting"]), "log_affine_scale": float(f["log_affine_scale"])})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(m)} scans")
    return {"features": feats, "mask6": ex.mask6, "failed": failed}, pd.DataFrame(qc)


# ---------------------------------------------------------------------------------------------------- pairs and rungs

def pair_rung(a: pd.Series, b: pd.Series) -> str:
    """Ladder rung of two scans (a and b differ)."""
    if a["dataset"] != b["dataset"]:
        return "L4_cross_dataset"
    if a["person"] == b["person"]:
        if a["dataset"] == "SIMON":
            if a["session"] == b["session"]:
                return "L0_same_session" if a.get("protocol") == b.get("protocol") else "L0x_same_session_other_sequence"
            dev = "same_scanner" if a.get("device", a["scanner"]) == b.get("device", b["scanner"]) else "other_scanner"
            prot = "same_protocol" if a.get("protocol") == b.get("protocol") else "other_protocol"
            return f"L1L2_{dev}_{prot}"
        return "L2_same_person_later_visit"
    if a["family"] == b["family"]:
        return "L3_cotwin_zygosity_unknown"  # QTIM family pairs here are same-sex, same-age co-twins
    if a["dataset"] == "IXI":
        return "L4_same_centre" if a["scanner"] == b["scanner"] else "L4_other_centre"
    return "L4_unrelated"


def pair_table(m: pd.DataFrame, X: np.ndarray, X_mirror: np.ndarray | None = None) -> pd.DataFrame:
    """All scan pairs with rung, time gap and Euclidean distance; X rows follow m."""
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    iu, ju = np.triu_indices(len(m), 1)
    recs = m.reset_index(drop=True)
    rung = [pair_rung(recs.iloc[i], recs.iloc[j]) for i, j in zip(iu, ju)]
    gap = np.abs(recs["years"].to_numpy()[iu] - recs["years"].to_numpy()[ju])
    return pd.DataFrame({"i": iu, "j": ju, "rung": rung, "gap_years": gap, "distance": D[iu, ju]})


def dprime(x: np.ndarray, y: np.ndarray) -> float:
    """Signed d': positive when y lies above x, so a reversed ordering shows as negative."""
    return float((np.mean(y) - np.mean(x)) / np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2))


def auc_lower(x: np.ndarray, y: np.ndarray) -> float:
    """P(a draw from x is below a draw from y), ties counted half."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float((x < y).mean() + 0.5 * (x == y).mean())


def rung_summary(p: pd.DataFrame) -> dict:
    g = p.groupby("rung")["distance"]
    return {r: {"n_pairs": int(len(s)), "median": float(s.median()), "q25": float(s.quantile(.25)),
                "q75": float(s.quantile(.75))} for r, s in g}


def contrasts(p: pd.DataFrame) -> dict:
    s = {r: g["distance"].to_numpy() for r, g in p.groupby("rung")}
    pairs = [("L0_same_session", "L0x_same_session_other_sequence"),
             ("L0_same_session", "L1L2_same_scanner_same_protocol"),
             ("L1L2_same_scanner_same_protocol", "L1L2_same_scanner_other_protocol"),
             ("L1L2_same_scanner_same_protocol", "L1L2_other_scanner_same_protocol"),
             ("L1L2_other_scanner_same_protocol", "L1L2_other_scanner_other_protocol"),
             ("L1L2_other_scanner_other_protocol", "L4_other_centre"),
             ("L2_same_person_later_visit", "L4_unrelated"), ("L3_cotwin_zygosity_unknown", "L4_unrelated"),
             ("L2_same_person_later_visit", "L3_cotwin_zygosity_unknown"), ("L4_same_centre", "L4_other_centre")]
    return {f"{a} < {b}": {"auc": auc_lower(s[a], s[b]), "dprime": dprime(s[a], s[b])}
            for a, b in pairs if a in s and b in s and len(s[a]) > 1 and len(s[b]) > 1}


def qtim_identification(m: pd.DataFrame, X: np.ndarray) -> dict:
    """Rank-1 re-identification of QTIM second-visit scans among all QTIM first-visit scans (same scanner)."""
    q = m.reset_index(drop=True)
    later = q.index[(q.dataset == "QTIM") & (q.session == "ses-02")]
    gallery = q.index[(q.dataset == "QTIM") & (q.session == "ses-01")]
    hits, n = 0, 0
    for i in later:
        if not (q.loc[gallery, "person"] == q.at[i, "person"]).any():
            continue
        d = np.linalg.norm(X[gallery] - X[i], axis=1)
        hits += q.at[gallery[int(np.argmin(d))], "person"] == q.at[i, "person"]
        n += 1
    return {"queries": int(n), "gallery_size": int(len(gallery)), "rank1": hits / n if n else None,
            "chance": 1 / len(gallery) if len(gallery) else None}


def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    sa, sb = (A * A).sum(1), (B * B).sum(1)
    return np.sqrt(np.maximum(sa[:, None] + sb[None, :] - 2 * A @ B.T, 0))


def mirror_test(m: pd.DataFrame, vols: list[np.ndarray], mask6: np.ndarray) -> dict:
    """Mirror half across scans: d(left half of scan i, reflected right half of scan j) for i != j.

    same person, other scan  -> does the reflected half still carry the person's identity?
    other person             -> the stranger reference for that question
    same half, other scan    -> the ordinary within-person comparison it is measured against
    The within-scan distance d(left_i, reflected right_i) shares every scan-level nuisance and is descriptive only.
    """
    lmask, rmask = mirror_halves(mask6)
    hm = lmask & rmask
    L = np.stack([mirror_halves(v)[0][hm] for v in vols])
    R = np.stack([mirror_halves(v)[1][hm] for v in vols])
    q = m.reset_index(drop=True)
    person, ds = q["person"].to_numpy(), q["dataset"].to_numpy()
    DLR, DLL = _dist(L, R), _dist(L, L)
    out = {}
    for name in ("QTIM", "IXI", "SIMON"):
        idx = np.flatnonzero(ds == name)
        if len(idx) < 3:
            continue
        sub_p = person[idx]
        same = sub_p[:, None] == sub_p[None, :]
        off = ~np.eye(len(idx), dtype=bool)
        lr, ll = DLR[np.ix_(idx, idx)], DLL[np.ix_(idx, idx)]
        res = {"n_scans": int(len(idx)), "median_within_scan_mirror_descriptive": float(np.median(np.diag(lr)))}
        if (same & off).any():
            res.update({"median_mirror_same_person_other_scan": float(np.median(lr[same & off])),
                        "median_same_half_same_person_other_scan": float(np.median(ll[same & off])),
                        "n_same_person_scan_pairs": int((same & off).sum())})
        if (~same).any():
            res.update({"median_mirror_other_person": float(np.median(lr[~same])),
                        "median_same_half_other_person": float(np.median(ll[~same]))})
        if (same & off).any() and (~same).any():
            res.update({"auc_mirror_same_person_below_other_person": auc_lower(lr[same & off], lr[~same]),
                        "auc_same_half_same_person_below_other_person": auc_lower(ll[same & off], ll[~same])})
        out[name] = res
    if "SIMON" in out:  # one person: strangers come from IXI, a different protocol (flagged)
        si, xi = np.flatnonzero(ds == "SIMON"), np.flatnonzero(ds == "IXI")
        lr_x, ll_x = DLR[np.ix_(si, xi)].ravel(), DLL[np.ix_(si, xi)].ravel()
        same = DLR[np.ix_(si, si)][~np.eye(len(si), dtype=bool)]
        same_ll = DLL[np.ix_(si, si)][~np.eye(len(si), dtype=bool)]
        out["SIMON"].update({"cross_protocol_auc_mirror_same_person_below_IXI": auc_lower(same, lr_x),
                             "cross_protocol_auc_same_half_same_person_below_IXI": auc_lower(same_ll, ll_x)})
    return out


# ---------------------------------------------------------------------------------------------------- main

def qc_outliers(qc: pd.DataFrame, k: float = 4.0, floor: float = QC_MIN_CORR) -> list[str]:
    """Registration failures: brain correlation with the template below an absolute floor or median - k MAD."""
    c = qc["qc_template_corr"]
    mad = np.median(np.abs(c - c.median())) * 1.4826
    return qc.loc[(c < floor) | (c < c.median() - k * mad), "scan"].tolist()


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    m = manifest()
    m.to_csv(os.path.join(PRIVATE, "manifest.csv"), index=False)
    print("scans per dataset:", m.dataset.value_counts().to_dict())
    store, qc = extract_all(m, PRIVATE)
    qc.to_csv(os.path.join(PRIVATE, "qc.csv"), index=False)
    outliers = set(qc_outliers(qc))
    bad = outliers | {s for s, _ in store["failed"]}
    qcd = qc.merge(m[["scan", "dataset"]], on="scan")
    excluded = m[m.scan.isin(bad)]
    m = m[m.scan.isin(store["features"]) & ~m.scan.isin(bad)].reset_index(drop=True)
    mask6 = store["mask6"]
    out = {"run": RUN, "n_scans": m.dataset.value_counts().to_dict(), "qc": {
        "rule": f"brain correlation with the template below {QC_MIN_CORR} (absolute floor) or below median - 4 MAD "
                f"(the MAD rule is reported but binds only if it is stricter than the floor)",
        "excluded_registration_failures": int(len(outliers)), "failed_extraction": len(store["failed"]),
        "failure_types": sorted({t for _, t in store["failed"]}),
        "excluded_by_dataset": excluded.dataset.value_counts().to_dict(),
        "excluded_simon_series": excluded.loc[excluded.dataset == "SIMON", "series"].value_counts().to_dict()
        if "series" in excluded else {},
        "median_template_corr_by_dataset": qcd.groupby("dataset").qc_template_corr.median().round(3).to_dict(),
        "strip_setting_counts": qc.strip_setting.value_counts().sort_index().to_dict()},
        "simon_within_session_pairs": {"same_protocol": int(0), "other_sequence": int(0)},
        "feature_grid": {"voxel_mm": 2 * BLOCK, "n_brain_voxels": int(mask6.sum())}, "features": {}}
    for feat in FEATURES:
        vols = [store["features"][s][feat] for s in m.scan]
        X = np.stack([v[mask6] for v in vols]).astype(np.float64)
        p = pair_table(m, X)
        p.to_csv(os.path.join(PRIVATE, f"pairs_{feat}.csv.gz"), index=False)
        out["simon_within_session_pairs"] = {"same_protocol": int((p.rung == "L0_same_session").sum()),
                                             "other_sequence": int((p.rung == "L0x_same_session_other_sequence").sum())}
        si = p.rung.str.startswith("L1L2_other_scanner")
        out["features"][feat] = {
            "rungs": rung_summary(p), "contrasts": contrasts(p),
            "simon_distance_vs_gap_spearman": {r: float(p[p.rung == r][["gap_years", "distance"]].corr("spearman").iloc[0, 1])
                                               for r in sorted(p.rung.unique()) if r.startswith("L1L2_")},
            "separation_ratio_L4_other_centre_over_simon_other_scanner": float(
                p[p.rung == "L4_other_centre"].distance.median() / p[si].distance.median())
            if si.any() else None,
            "qtim_identification": qtim_identification(m, X),
            "mirror": mirror_test(m, vols, mask6)}
    import ants, nilearn, scipy
    out["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                          "antspyx": ants.__version__, "nilearn": nilearn.__version__, "scipy": scipy.__version__}
    out["caveats"] = ["SIMON is one person: its rungs are within-person; L4 comes from IXI and QTIM",
                      "L4_cross_dataset mixes person and protocol differences and is reported only for completeness",
                      "QTIM family pairs are 2 families of same-sex, same-age co-twins with unknown zygosity: descriptive only",
                      "SIMON rungs cover IR-GRE contrasts only: QC removed every CUBE/DIR/Core-3D T1 scan",
                      "pairwise AUCs treat dependent pairs as independent and have no confidence intervals",
                      "QTIM is adolescent (12-16 y, then about 3.5 y later): its L2 includes development"]
    with open(os.path.join(PROJECT, "results", "image_ladder.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()
