"""Run the frozen real-case protocol (docs/PROTOCOL_real_cases.md) on QTIM T1w scans.

Stages, each resumable; private outputs go to the data root, aggregates to results/:
  strip    SynthStrip brain mask of every native scan (pinned FreeSurfer 8.0.0 image)
  prepare  N4, affine ANTs registration of the skull-stripped scan to the nilearn MNI 2009a symmetric template,
           linear resampling of the N4-corrected scan to its 1 mm grid; own brain mask, template correlation,
           white-matter peak
  qc       D7: template correlation below median - 3 MAD flagged with three-slice views; automatic exclusions
  segtime  D8: wall time of both segmenters on one registered scan
  score    every fill of every target: the nine package metrics and the distance-to-mean diagnostics (private)
  label    D8/D9: label space and blur sweep on the first n_L targets of set S (private)
  report   c, rho, bootstrap, claims C1 to C9, sensitivity analyses; results/real_cases.json and the manifest
Refuses to run if an input hash, the scan manifest hash or a package version differs from the protocol; the scoring
stages also need a dated amendment with the scan manifest hash (Amendment 2) committed in the protocol. Scans are keyed by their index
in selection.json; no participant ID is printed or written outside the private folder.
"""
import argparse
import csv
import datetime
import functools
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_common import HIGHER, METRICS, calibrate  # noqa: E402
from population_null import diagnostics, serializable, shrinkage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_real_cases.md")
DATA_ROOT = os.environ.get("DATA_ROOT", r"C:\Projects\data")
BIDS = os.path.join(DATA_ROOT, "brain_images", "siblings")
PRIVATE = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "real-cases-protocol")
RUN = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "real-cases-run")
NILEARN_DATA = os.path.join(os.path.dirname(importlib.util.find_spec("nilearn").origin), "datasets", "data")

SEED, N_BOOT, K = 20261101, 2000, 16
LAMBDAS = (0.75, 0.5, 0.25)
BLUR_SIGMAS = (1, 2, 3, 4)
BOX = ((-45, -13), (-65, -33), (43, 67))
MIRROR = ((13, 45), (-65, -33), (43, 67))
MEAN20 = 20
Z_OFFSET = 10.0  # z-scores shifted so the package's clip at zero never acts; the constant cancels in every difference
MODEL_RHO_MEAN = 0.5 * (1 + 1 / 60)
INPUTS = {
    "participants.tsv": (os.path.join(BIDS, "participants.tsv"),
                         "a013daf4672804d4f5a837ce75f2b0a52b67270c7ce4d09d1d01dc2c5aeff582"),
    "dataset_description.json": (os.path.join(BIDS, "dataset_description.json"),
                                 "0f1dc04f0d8e777ae3ce9e264fb621c4227c9b6c842d5a4898e85a8a9f5ecc61"),
    "selection.json": (os.path.join(PRIVATE, "selection.json"),
                       "cb264436a1346f6a83b17e7d3abd8f3bc27ac10174ad89c8d7339eec421f7f31"),
    "template_t1": (os.path.join(NILEARN_DATA, "mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz"),
                    "421a10e872fd6cadae7f61d358dffbcc1795a497d61ee76c5dda2503e1a1e9e6"),
    "template_gm": (os.path.join(NILEARN_DATA, "mni_icbm152_gm_tal_nlin_sym_09a_converted.nii.gz"),
                    "97a5ca69bd24db37a9cb7b32525e1733a209af904129bf1cd36da06d24243bed"),
    "template_wm": (os.path.join(NILEARN_DATA, "mni_icbm152_wm_tal_nlin_sym_09a_converted.nii.gz"),
                    "382d92812de4744f9c86c7a0e4f680dc317a0a50e4da1f0153618a6798c7b7db"),
}
DESIGNS = {  # private run folder, selection (file, SHA-256), scan manifest (file, SHA-256)
    "frozen": ("real-cases-run", ("selection.json", INPUTS["selection.json"][1]),
               ("scan_manifest.json", "4016dcf4a30ab95c76170ba714598c435792a260d70a39780dd6701928b0dcb8")),
    "expanded": ("real-cases-run-expanded",  # Amendment 1, A1.1 and A1.5
                 ("selection_expanded.json", "fcfca9875f9b895b1d96ac11b357248ab950a972eed7b7b125e878e83b8d3dc6"),
                 ("scan_manifest_expanded.json", "2a1d907c575e6a9c2120dbcc3deb67ad9901761275fbef4c4dd75cbf52a17cd7")),
}


def use_design(name):
    """Point the run folder, selection and scan manifest at one design (workers inherit it via the environment)."""
    global DESIGN, RUN, SCAN_MANIFEST, SCAN_MANIFEST_SHA256
    run, (sel, sel_sha), (man, man_sha) = DESIGNS[name]
    DESIGN, RUN = name, os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", run)
    INPUTS["selection.json"] = (os.path.join(PRIVATE, sel), sel_sha)
    SCAN_MANIFEST, SCAN_MANIFEST_SHA256 = os.path.join(PRIVATE, man), man_sha
    os.environ["REAL_CASES_DESIGN"] = name


use_design(os.environ.get("REAL_CASES_DESIGN", "frozen"))
VERSIONS = {"inpainting": "0.0.12", "torchmetrics": "1.9.0", "antspyx": "0.6.3", "nilearn": "0.13.1",
            "nibabel": "5.4.2", "numpy": "1.26.4"}
SEGMENTERS = {
    "fastsurfer": "deepmi/fastsurfer@sha256:8aa18bae0a287a8925cb26c8903e77043880f21911f2e08821253e41a54f8e7f",
    "synthseg": "freesurfer/freesurfer@sha256:f3e22b472afd9ca2b269d1b40c061ac891efaec070d09e387f4b024ea681e266",
}
LABEL_BUDGET_S, LABEL_VOLUMES = 8 * 3600, 24
LABEL_HIGHER = {"dice_mean", "accuracy"}
SCORING_STAGES = {"score", "label", "report"}


# ---------------------------------------------------------------------------------------------------- provenance

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def guard(stage, inputs=None, manifest_sha=None, protocol=PROTOCOL):
    """Refuse to run on any difference from the protocol; returns the verified hashes."""
    inputs = INPUTS if inputs is None else inputs
    manifest_sha = SCAN_MANIFEST_SHA256 if manifest_sha is None else manifest_sha
    bad = [p for p, v in VERSIONS.items() if version(p) != v]
    if bad or sys.version_info[:2] != (3, 11):
        raise SystemExit(f"package versions differ from the protocol: {bad or 'python'}")
    got = {name: sha256(path) for name, (path, _) in inputs.items()}
    bad = [name for name, (_, want) in inputs.items() if got[name] != want]
    if bad:
        raise SystemExit(f"input hash differs from the protocol: {bad}")
    if not manifest_sha or sha256(SCAN_MANIFEST) != manifest_sha:
        raise SystemExit("scan manifest hash differs from the one recorded for Amendment 1")
    got["scan_manifest"] = manifest_sha
    if stage in SCORING_STAGES:
        text = open(protocol, encoding="utf-8").read()
        dated = [s for s in text.split("\n### Amendment ")[1:] if "(planned" not in s.split("\n", 1)[0]]
        if not any(manifest_sha in s for s in dated):
            raise SystemExit("no dated amendment records the scan manifest hash: no fill is scored")
        missing = [name for name, (_, want) in inputs.items() if want not in text]
        if missing:
            raise SystemExit(f"hashes not listed in the protocol: {missing}")
        if protocol == PROTOCOL and git("status", "--porcelain", "--", protocol):
            raise SystemExit("the protocol has uncommitted changes: commit the amendment before scoring")
    return got


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout.strip()


def write_json(path, obj, exclusive=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = json.dumps(serializable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n"
    with open(path, "x" if exclusive else "w", encoding="utf-8") as f:
        f.write(blob)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------- design

def load_design():
    """Targets, donors and pools as scan indices; participant IDs never leave this function except as indices."""
    sel = read_json(INPUTS["selection.json"][0])
    key = {tuple(s): i for i, s in enumerate(sel["scans"])}
    with open(INPUTS["participants.tsv"][0], encoding="utf-8") as f:
        sex = {r["participant_id"]: r["sex"] for r in csv.DictReader(f, delimiter="\t")}
    first = lambda p: key[(p, "ses-01")]  # noqa: E731
    return {"n_scans": len(sel["scans"]),
            "S": [{"target": first(e["target"]), "donor": first(e["sibling"]), "sex": sex[e["target"]]}
                  for e in sel["set_S"]],
            "R": [{"target": first(e["target"]), "donor": key[(e["target"], "ses-02")], "sex": sex[e["target"]]}
                  for e in sel["set_R"]],
            "strangers": [first(p) for p in sel["stranger_pool"]],
            "stranger_sex": [sex[p] for p in sel["stranger_pool"]],
            "mean_pool": [first(p) for p in sel["mean_pool"]]}


def assign(design, excluded):
    """Apply QC (D7: no replacement) and draw the strangers with the seed, set S first, then set R."""
    ok = lambda i: f"scan_{i:03d}" not in excluded  # noqa: E731
    pool = [(j, s) for j, s in zip(design["strangers"], design["stranger_sex"]) if ok(j)]
    rng, rng_sex = np.random.default_rng(SEED), np.random.default_rng([SEED, 2])
    out = {"S": [], "R": [], "dropped": {"S": 0, "R": 0}}
    for name in ("S", "R"):
        for order, e in enumerate(design[name]):
            if not (ok(e["target"]) and ok(e["donor"])):
                out["dropped"][name] += 1
                continue
            same = [j for j, s in pool if s == e["sex"]]
            out[name].append({"id": f"{name}{order:02d}", "order": order, "target": e["target"], "donor": e["donor"],
                              "strangers": [int(j) for j in rng.choice([j for j, _ in pool], K, replace=False)],
                              "strangers_sex": [int(j) for j in rng_sex.choice(same, min(K, len(same)), replace=False)]})
    out["stranger_pool"] = [j for j, _ in pool]
    out["mean_pool"] = [j for j in design["mean_pool"] if ok(j)]
    return out


# ---------------------------------------------------------------------------------------------------- space

def box_mask(shape, affine, bounds):
    """Voxels whose centre lies in the half-open mm bounds of every axis."""
    ijk = np.indices(shape).reshape(3, -1)
    xyz = (affine[:3, :3] @ ijk + affine[:3, 3:4]).reshape((3,) + tuple(shape))
    m = np.ones(shape, bool)
    for a, (lo, hi) in zip(xyz, bounds):
        m &= (a >= lo) & (a < hi)
    return m


def box_slices(mask):
    idx = np.argwhere(mask)
    sl = tuple(slice(a, b + 1) for a, b in zip(idx.min(0), idx.max(0)))
    if not mask[sl].all():
        raise ValueError("mask is not an axis-aligned box")
    return sl


def mirrored_values(vol, box, mirror):
    """The mirrored box flipped left to right, in the voxel order of vol[box] (array axis 0 runs along x)."""
    return vol[box_slices(mirror)][::-1].ravel()


def wm_peak(values, bins=256, min_height=0.1, min_prominence=0.05):
    """Upper mode of the intensity histogram: the highest-intensity peak of the smoothed histogram with height and
    prominence of at least min_height and min_prominence of the tallest peak (so tail noise is not a mode)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    lo, hi = np.percentile(values, [0.5, 99.5])
    h, edges = np.histogram(values, bins=bins, range=(lo, hi))
    hs = gaussian_filter1d(h.astype(float), 2)
    peaks, _ = find_peaks(np.r_[0, hs, 0], height=min_height * hs.max(), prominence=min_prominence * hs.max())
    if not len(peaks):
        raise ValueError("no histogram mode")
    i = int(peaks.max()) - 1
    return float((edges[i] + edges[i + 1]) / 2)


@functools.lru_cache(1)
def space():
    """Template grid, template brain mask, box and mirrored box (1 mm)."""
    import nibabel as nib
    ni = nib.load(os.path.join(RUN, "template", "template_1mm.nii.gz"))
    brain = nib.load(os.path.join(RUN, "template", "brain_1mm.nii.gz")).get_fdata() > 0
    affine = ni.affine
    if not (np.allclose(affine[:3, :3], np.eye(3))):
        raise ValueError("template grid is not 1 mm RAS")
    return ni, brain, box_mask(ni.shape, affine, BOX), box_mask(ni.shape, affine, MIRROR)


def ensure_templates():
    import nibabel as nib
    from nilearn import datasets
    d = os.path.join(RUN, "template")
    os.makedirs(d, exist_ok=True)
    for res in (1, 2):
        for name, load in (("template", datasets.load_mni152_template), ("brain", datasets.load_mni152_brain_mask)):
            path = os.path.join(d, f"{name}_{res}mm.nii.gz")
            if not os.path.exists(path):
                nib.save(load(resolution=res), path)


def normalized(i, meta, kind="wm", raw=None):
    """Registered scan i on the template grid, scaled (white-matter peak or offset z-score), zero outside the brain."""
    import nibabel as nib
    _, brain, _, _ = space()
    v = nib.load(os.path.join(RUN, "prep", f"scan_{i:03d}.nii.gz")).get_fdata(dtype=np.float32) if raw is None else raw
    v = v / meta["wm_peak"] if kind == "wm" else (v - meta["brain_mean"]) / meta["brain_sd"] + Z_OFFSET
    v[~brain] = 0
    return v.astype(np.float32)


# ---------------------------------------------------------------------------------------------------- prepare

def native_qc(img, mask, spacing):
    """Descriptive data QC of one N4-corrected native scan inside its brain mask (never used to exclude): mask
    volume, intensity percentiles, white-matter peak and an SNR proxy (mean of voxels within 10% of the peak over the
    SD of the background: non-zero voxels below the Otsu threshold, more than 10 mm outside the brain)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    inside = img[mask]
    peak = wm_peak(inside)
    wm = inside[np.abs(inside - peak) < 0.1 * peak]
    far = ~ndimage.binary_dilation(mask, iterations=max(1, int(round(10 / min(spacing)))))
    bg = img[far & (img > 0) & (img < threshold_otsu(img[img > 0]))]
    pct = np.percentile(inside, [1, 5, 50, 95, 99])
    return {"mask_ml": float(mask.sum() * np.prod(spacing) / 1000), "wm_peak_native": peak,
            **{f"p{q:02d}": float(v) for q, v in zip((1, 5, 50, 95, 99), pct)},
            "snr_proxy": float(wm.mean() / bg.std()) if bg.size > 100 and bg.std() > 0 else None}


def scan_jobs(limit=None):
    """(index, BIDS-relative path, manifest SHA-256) of every scan in selection order."""
    files = {r["path"]: r["sha256"] for r in read_json(SCAN_MANIFEST)["files"]}
    sel = read_json(INPUTS["selection.json"][0])
    rels = [f"{p}/{s}/anat/{p}_{s}_T1w.nii.gz" for p, s in sel["scans"]][:limit]
    return [(i, rel, files[rel]) for i, rel in enumerate(rels)]


def strip(limit=None, threads=8):
    """SynthStrip brain masks of the native scans, one container looping over the scans (masks keyed by index)."""
    out = os.path.join(RUN, "strip")
    os.makedirs(out, exist_ok=True)
    todo = [(i, rel, sha) for i, rel, sha in scan_jobs(limit)
            if not os.path.exists(os.path.join(out, f"scan_{i:03d}_mask.nii.gz"))]
    bad = [f"scan_{i:03d}" for i, rel, sha in todo if sha256(os.path.join(BIDS, *rel.split("/"))) != sha]
    if bad:
        raise SystemExit(f"files differ from the scan manifest: {bad}")
    listing = os.path.join(out, "strip_list.txt")  # private: holds BIDS paths, removed after the run
    with open(listing, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{rel} scan_{i:03d}\n" for i, rel, _ in todo)
    loop = ('while read rel key; do mri_synthstrip -i "/bids/$rel" -m "/out/${key}_mask.nii.gz" -t %d >/dev/null 2>&1'
            ' || echo "$key failed"; done < /out/strip_list.txt' % threads)
    start = time.monotonic()
    run = subprocess.run(["docker", "run", "--rm", "-v", f"{BIDS}:/bids:ro", "-v", f"{out}:/out", SEGMENTERS["synthseg"],
                          "bash", "-c", loop], capture_output=True, text=True, check=False)
    os.remove(listing)
    failed = [w for w in run.stdout.split() if w.startswith("scan_")]
    print({"stripped": len(todo) - len(failed), "failed": failed, "seconds_per_scan":
           round((time.monotonic() - start) / max(len(todo), 1), 1)})


def prepare_scan(job):
    i, rel, expected = job
    out = os.path.join(RUN, "prep", f"scan_{i:03d}")
    if os.path.exists(out + ".json"):
        return read_json(out + ".json")
    import ants
    import nibabel as nib
    start = time.monotonic()
    meta = {"scan": f"scan_{i:03d}", "source_sha256": expected}
    try:
        path = os.path.join(BIDS, *rel.split("/"))
        if sha256(path) != expected:
            raise RuntimeError("file differs from the scan manifest")
        tdir = os.path.join(RUN, "template")
        tpl1, tpl2 = (ants.image_read(os.path.join(tdir, f"template_{r}mm.nii.gz")) for r in (1, 2))
        brain1, brain2 = (nib.load(os.path.join(tdir, f"brain_{r}mm.nii.gz")).get_fdata() > 0 for r in (1, 2))
        native = ants.image_read(path)
        mask = ants.image_read(os.path.join(RUN, "strip", f"scan_{i:03d}_mask.nii.gz"))
        small = ants.resample_image(native, (2, 2, 2), use_voxels=False, interp_type=0)
        bias = ants.n4_bias_field_correction(small, shrink_factor=4, return_bias_field=True)
        corrected2 = small / bias
        mask2 = ants.resample_image_to_target(mask, small, interp_type="nearestNeighbor").numpy() > 0.5
        work = tempfile.mkdtemp(prefix=f"scan_{i:03d}_")
        reg = ants.registration(fixed=tpl2, moving=corrected2.new_image_like(corrected2.numpy() * mask2),
                                type_of_transform="Affine", mask=tpl2.new_image_like(brain2.astype(np.float32)),
                                mask_all_stages=True, aff_iterations=(1000, 500, 250, 50), random_seed=123,
                                outprefix=os.path.join(work, "a_"), verbose=False)
        corr2 = float(np.corrcoef(reg["warpedmovout"].numpy()[brain2], tpl2.numpy()[brain2])[0, 1])
        shutil.copy(reg["fwdtransforms"][0], out + "_affine.mat")
        native_bias = ants.resample_image_to_target(bias, native, interp_type="linear")
        corrected = native.new_image_like(native.numpy() / np.maximum(native_bias.numpy(), 1e-6))
        meta["native_qc"] = native_qc(corrected.numpy(), mask.numpy() > 0.5, native.spacing)
        warped = ants.apply_transforms(tpl1, corrected, reg["fwdtransforms"], interpolator="linear").numpy()
        own = ants.apply_transforms(tpl1, mask, reg["fwdtransforms"], interpolator="nearestNeighbor").numpy() > 0.5
        shutil.rmtree(work, ignore_errors=True)
        _, _, box, mirror = space()
        visible = warped[brain1 & ~box]
        affine = nib.load(os.path.join(tdir, "template_1mm.nii.gz")).affine
        nib.save(nib.Nifti1Image(warped.astype(np.float32), affine), out + ".nii.gz")
        nib.save(nib.Nifti1Image(own.astype(np.uint8), affine), out + "_brain.nii.gz")
        meta.update(template_corr=float(np.corrcoef(warped[brain1], tpl1.numpy()[brain1])[0, 1]),
                    template_corr_2mm=corr2, wm_peak=wm_peak(visible),
                    brain_mean=float(visible.mean()), brain_sd=float(visible.std()),
                    box_in_own_brain=float(own[box].mean()), mirror_in_own_brain=float(own[mirror].mean()),
                    registration_seconds=time.monotonic() - start)
    except Exception as e:  # noqa: BLE001 - recorded as an exclusion reason (D7), never dropped silently
        meta["error"] = f"{type(e).__name__}: {e}"
    write_json(out + ".json", meta)
    return meta


def prepare(limit, workers):
    ensure_templates()
    space()
    jobs = scan_jobs(limit)
    def ready(i):  # a mask younger than two minutes may still be being written
        p = os.path.join(RUN, "strip", f"scan_{i:03d}_mask.nii.gz")
        return os.path.exists(p) and time.time() - os.path.getmtime(p) > 120
    waiting = [i for i, _, _ in jobs if not ready(i)]
    jobs = [j for j in jobs if j[0] not in waiting]  # masks still being made: prepared on a later call
    os.makedirs(os.path.join(RUN, "prep"), exist_ok=True)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(max(1, 8 // workers))
    with ProcessPoolExecutor(workers) as ex:
        metas = list(ex.map(prepare_scan, jobs))
    secs = [m["registration_seconds"] for m in metas if "registration_seconds" in m]
    print({"prepared": len(metas), "waiting_for_mask": len(waiting), "errors": sum("error" in m for m in metas),
           "registration_seconds_median": float(np.median(secs)) if secs else None})


# ---------------------------------------------------------------------------------------------------- QC (D7)

def qc_flags(metas):
    """Automatic exclusions (unreadable, box < 95% inside the scan's brain mask) and correlation flags for review."""
    auto = {m["scan"]: m["error"] for m in metas if "error" in m}
    ok = [m for m in metas if "error" not in m]
    corr = np.array([m["template_corr"] for m in ok])
    med = float(np.median(corr))
    mad = float(np.median(np.abs(corr - med)))
    auto.update({m["scan"]: f"box {100 * m['box_in_own_brain']:.1f}% inside own brain mask" for m in ok
                 if m["box_in_own_brain"] < 0.95})
    flagged = [m["scan"] for m in ok if m["template_corr"] < med - 3 * mad]
    return {"median": med, "mad": mad, "threshold": med - 3 * mad, "flagged_for_review": flagged, "automatic": auto}


def qc(visual_fail):
    n = len(read_json(INPUTS["selection.json"][0])["scans"])
    metas = [read_json(os.path.join(RUN, "prep", f"scan_{i:03d}.json")) for i in range(n)]
    flags = qc_flags(metas)
    views = os.path.join(RUN, "qc")
    os.makedirs(views, exist_ok=True)
    for scan in flags["flagged_for_review"]:
        qc_view(int(scan[5:]), metas[int(scan[5:])], os.path.join(views, f"{scan}.png"))
    for scan in visual_fail:
        if scan not in flags["flagged_for_review"]:
            raise SystemExit(f"{scan} was not flagged: D7 excludes only flagged scans with a visible failure")
    excluded = {**flags["automatic"], **{s: "visible registration failure on review" for s in visual_fail}}
    write_json(os.path.join(views, "qc_summary.json"), {**flags, "visual_fail": sorted(visual_fail),
                                                         "excluded": excluded})
    print({"scans": len(metas), "flagged_for_review": len(flags["flagged_for_review"]),
           "excluded": len(excluded), "threshold": round(flags["threshold"], 3)})


def qc_view(i, meta, path):
    """Three orthogonal slices through the box centre, scan and template side by side (private)."""
    import matplotlib.pyplot as plt
    import nibabel as nib
    ni, brain, box, _ = space()
    v = normalized(i, meta) if "error" not in meta else np.zeros(ni.shape)
    t = ni.get_fdata()
    c = [int(np.mean(a)) for a in np.nonzero(box)]
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    for row, img in enumerate((v, t)):
        for col, sl in enumerate((img[c[0]], img[:, c[1]], img[:, :, c[2]])):
            ax[row, col].imshow(np.rot90(sl), cmap="gray")
            ax[row, col].axis("off")
    fig.suptitle(f"scan_{i:03d}: template correlation {meta.get('template_corr', float('nan')):.3f}")
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- data QC report

ACQ_FIELDS = {"RepetitionTime": "tr_s", "EchoTime": "te_s", "InversionTime": "ti_s", "FlipAngle": "flip_deg",
              "MagneticFieldStrength": "field_t", "ManufacturersModelName": "scanner", "SoftwareVersions": "software"}
QC_CONTINUOUS = ("mask_ml", "wm_peak_native", "p01", "p05", "p50", "p95", "p99", "p01_rel", "p05_rel", "p50_rel",
                 "p95_rel", "p99_rel", "snr_proxy", "template_corr", "box_in_own_brain", "mirror_in_own_brain")
QC_CATEGORICAL = ("tr_s", "te_s", "ti_s", "flip_deg", "field_t", "scanner", "software", "acq_orientation",
                  "voxel_mm", "matrix", "orientation", "dtype")


def robust_flags(values, k=3.0):
    """Positions of finite values more than k MAD from the median (any departure when the MAD is zero)."""
    v = np.array([np.nan if x is None else x for x in values], float)
    ok = np.isfinite(v)
    if not ok.any():
        return []
    med = np.median(v[ok])
    mad = np.median(np.abs(v[ok] - med))
    dev = np.abs(v - med)
    return [i for i in range(len(v)) if ok[i] and (dev[i] > k * mad if mad > 0 else dev[i] > 0)]


def qc_table():
    """One descriptive row per scan of the current design's selection (private: holds participant IDs)."""
    import nibabel as nib
    sel = read_json(INPUTS["selection.json"][0])
    with open(INPUTS["participants.tsv"][0], encoding="utf-8") as f:
        part = {r["participant_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    rows = []
    for i, (p, s) in enumerate(sel["scans"]):
        base = os.path.join(BIDS, p, s, "anat", f"{p}_{s}_T1w")
        row = {"scan": f"scan_{i:03d}", "participant_id": p, "session": s,
               "acq_orientation": part[p]["ses01_T1w_acq" if s == "ses-01" else "ses02_T1w_acq"]}
        side = read_json(base + ".json") if os.path.exists(base + ".json") else {}
        row.update({short: side.get(key) for key, short in ACQ_FIELDS.items()})
        if isinstance(row["software"], list):
            row["software"] = " ".join(row["software"])
        ni = nib.load(base + ".nii.gz")
        data = np.asanyarray(ni.dataobj)
        row.update(voxel_mm="x".join(f"{z:.4g}" for z in ni.header.get_zooms()[:3]),
                   matrix="x".join(str(n) for n in ni.shape[:3]), orientation="".join(nib.aff2axcodes(ni.affine)),
                   dtype=str(ni.get_data_dtype()), nonfinite=int(np.size(data) - np.isfinite(data).sum()))
        meta_path = os.path.join(RUN, "prep", f"scan_{i:03d}.json")
        meta = read_json(meta_path) if os.path.exists(meta_path) else {}
        nq = meta.get("native_qc", {})
        row.update(nq)
        if nq:
            row.update({f"{q}_rel": nq[q] / nq["wm_peak_native"] for q in ("p01", "p05", "p50", "p95", "p99")})
        row.update({k: meta.get(k) for k in ("template_corr", "box_in_own_brain", "mirror_in_own_brain")})
        row["prep_error"] = meta.get("error")
        rows.append(row)
    return rows


def qc_summary(rows):
    """Aggregate QC without IDs: per variable n, median, range and robust-outlier count; value counts of the
    acquisition and header fields and how many scans depart from the most common value."""
    out = {"scans": len(rows), "sidecars_found": sum(r["tr_s"] is not None for r in rows),
           "nonfinite_scans": sum(r["nonfinite"] > 0 for r in rows),
           "prep_errors": sum(r["prep_error"] is not None for r in rows), "continuous": {}, "categorical": {}}
    flagged = {}
    for k in QC_CONTINUOUS:
        vals = [r.get(k) for r in rows]
        fin = [v for v in vals if v is not None]
        idx = robust_flags(vals)
        flagged[k] = [rows[i]["scan"] for i in idx]
        out["continuous"][k] = ({"n": len(fin), "median": float(np.median(fin)), "min": float(min(fin)),
                                 "max": float(max(fin)), "flagged_3mad": len(idx)} if fin else {"n": 0})
    for k in QC_CATEGORICAL:
        counts = {}
        for r in rows:
            counts[str(r.get(k))] = counts.get(str(r.get(k)), 0) + 1
        mode = max(counts, key=counts.get)
        flagged[k] = [r["scan"] for r in rows if str(r.get(k)) != mode]
        out["categorical"][k] = {"values": dict(sorted(counts.items())), "departing_from_mode": len(flagged[k])}
    out["box_coverage_below_95"] = sum((r.get("box_in_own_brain") or 1) < 0.95 for r in rows)
    out["scans_with_any_flag"] = len({s for v in flagged.values() for s in v})
    return out, flagged


def dataqc():
    """Kate's descriptive data-QC report (2026-11-01): never excludes; suggestions go to Kate as amendments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = qc_table()
    summary_, flagged = qc_summary(rows)
    d = os.path.join(RUN, "data_qc")
    os.makedirs(os.path.join(d, "thumbs"), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(d, "qc_table.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    write_json(os.path.join(d, "qc_flags.json"), flagged)
    any_flag = sorted({s for k, v in flagged.items() if k in QC_CONTINUOUS for s in v})
    metas = {r["scan"]: read_json(os.path.join(RUN, "prep", f"{r['scan']}.json")) for r in rows
             if os.path.exists(os.path.join(RUN, "prep", f"{r['scan']}.json"))}
    _, brain, _, _ = space()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1.6, 161)
    for scan, meta in metas.items():
        if "error" in meta:
            continue
        h, _ = np.histogram(normalized(int(scan[5:]), meta)[brain], bins=bins, density=True)
        bad = scan in any_flag
        ax.plot(bins[:-1], h, color="tab:red" if bad else "0.6", lw=0.8 if bad else 0.3, alpha=1 if bad else 0.5)
    ax.set_xlabel("intensity / white-matter peak, inside the template brain mask")
    ax.set_ylabel("density")
    fig.savefig(os.path.join(d, "histograms.png"), dpi=110)
    plt.close(fig)
    for scan in any_flag:
        if scan in metas:
            qc_view(int(scan[5:]), metas[scan], os.path.join(d, "thumbs", f"{scan}.png"))
    summary_.update(generated=datetime.date.today().isoformat(), design=f"{DESIGN} selection ({len(rows)} scans)",
                    note="descriptive only; no exclusion beyond D7; flags are 3 MAD from the median or departures "
                         "from the most common acquisition value")
    name = "real_cases_data_qc_summary" + ("" if DESIGN == "frozen" else f"_{DESIGN}") + ".json"
    write_json(os.path.join(ROOT, "results", name), summary_)
    with open(os.path.join(RUN, "run_log.md"), "a", encoding="utf-8") as f:
        f.write(f"- {datetime.datetime.now().isoformat(timespec='minutes')}: data-QC report (descriptive) on "
                f"{len(rows)} scans; {summary_['scans_with_any_flag']} scans carry at least one flag; table, flags, "
                "histograms and thumbnails in data_qc/; summary in results/real_cases_data_qc_summary.json.\n")
    print({k: summary_[k] for k in ("scans", "sidecars_found", "nonfinite_scans", "prep_errors",
                                     "box_coverage_below_95", "scans_with_any_flag")})


# ---------------------------------------------------------------------------------------------------- fills

def fills(x, mu, s1, donor=None, other=None, brain=None):
    """Box-voxel fills of one target. x, mu, s1, donor, other: values in box order; brain: in-brain box voxels."""
    brain = np.ones(x.shape, bool) if brain is None else brain
    out = {}
    if donor is not None:
        out["relative"] = donor
    if other is not None:
        out["other_hemisphere"] = other
    out["mean"] = mu
    m = mu[brain]
    restored = np.zeros_like(mu)
    restored[brain] = m.mean() + (m - m.mean()) / m.std() * x[brain].std()
    out["mean_restored"] = restored
    for lam in LAMBDAS:
        out[f"shrink_{lam:g}"] = shrinkage(x, mu, lam)
    a, r = x - mu, s1 - mu
    aa = float(a @ a)
    r = r - (r @ a) / aa * a
    out["wrong_detail"] = mu + r * np.sqrt(aa) / np.linalg.norm(r)
    return out


def metrics_on(x_t, box_t, norm_t, values):
    """The nine package metrics of one fill on the 3D volume: target outside the box, the fill inside it."""
    import torch
    from inpainting.challenge_metrics_2023 import generate_metrics
    f = x_t.clone()
    f[box_t] = torch.as_tensor(values, dtype=torch.float32)
    return generate_metrics(prediction=f[None], target=x_t[None], normalization_tensor=norm_t[None], mask=box_t[None])


def score_volume(x, box, named, strangers):
    """Metrics of every named fill and every stranger fill (box values) against target volume x."""
    import torch
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    box_t = torch.from_numpy(box)
    norm_t = x_t * ~box_t
    return ({k: metrics_on(x_t, box_t, norm_t, v) for k, v in named.items()},
            {str(j): metrics_on(x_t, box_t, norm_t, v) for j, v in strangers.items()})


# ---------------------------------------------------------------------------------------------------- score

_CACHE = {}


def _init_worker(cache_path, threads):
    import torch
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(threads)
    with np.load(cache_path) as f:
        _CACHE.update({k: f[k] for k in f.files})


def score_target(job):
    t, set_name, metas = job["target"], job["set"], job["metas"]
    out = os.path.join(RUN, "scores", f"{job['id']}.json")
    if os.path.exists(out):
        return job["id"]
    _, brain, box, mirror = space()
    bb = brain[box]
    row = {"id": job["id"], "set": set_name, "order": job["order"], "box_voxels": int(box.sum()),
           "box_in_template_brain": float(bb.mean())}
    for kind in ("wm", "z"):
        c = _CACHE[f"box_{kind}"]
        x = normalized(t["target"], metas[t["target"]], kind)
        xb = x[box]
        mu = c[job["mean_pool"]].mean(0)
        named = fills(xb, mu, c[t["strangers"][0]], c[t["donor"]], mirrored_values(x, box, mirror), bb)
        named = {("sibling" if set_name == "S" else "repeat") if k == "relative" else k: v for k, v in named.items()}
        if kind == "wm":
            named["mean20"] = c[job["mean_pool"][:MEAN20]].mean(0)
            pool = sorted(set(job["stranger_pool"]))
        else:
            pool = t["strangers"]
        raw, strangers = score_volume(x, box, named, {j: c[j] for j in pool})
        row[kind] = {"fills": raw, "strangers": strangers}
        if kind == "wm":
            row["diagnostics"] = {k: diagnostics(xb, v, mu, np.ones(xb.shape, bool)) for k, v in named.items()}
            row["diagnostics"].update({f"stranger_{j}": diagnostics(xb, c[j], mu, np.ones(xb.shape, bool))
                                       for j in t["strangers"]})
    row.update(strangers=t["strangers"], strangers_sex=t["strangers_sex"])
    write_json(out, row)
    return job["id"]


def excluded_scans():
    return read_json(os.path.join(RUN, "qc", "qc_summary.json"))["excluded"]


def score(workers):
    design = load_design()
    plan = assign(design, excluded_scans())
    metas = [read_json(os.path.join(RUN, "prep", f"scan_{i:03d}.json")) for i in range(design["n_scans"])]
    cache_path = os.path.join(RUN, "box_cache.npz")
    if not os.path.exists(cache_path):
        import nibabel as nib
        _, _, box, _ = space()
        arrays = {f"box_{k}": np.zeros((design["n_scans"], int(box.sum())), np.float32) for k in ("wm", "z")}
        for i, m in enumerate(metas):
            if "error" not in m:
                raw = nib.load(os.path.join(RUN, "prep", f"scan_{i:03d}.nii.gz")).get_fdata(dtype=np.float32)
                for k in ("wm", "z"):
                    arrays[f"box_{k}"][i] = normalized(i, m, k, raw)[box]
        np.savez(cache_path, **arrays)
    jobs = [{"id": t["id"], "order": t["order"], "set": s, "target": t, "metas": metas,
             "mean_pool": plan["mean_pool"], "stranger_pool": plan["stranger_pool"]}
            for s in ("S", "R") for t in plan[s]]
    os.makedirs(os.path.join(RUN, "scores"), exist_ok=True)
    start = time.monotonic()
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cache_path, max(1, 8 // workers))) as ex:
        for n, done in enumerate(ex.map(score_target, jobs), 1):
            if n % 10 == 0:
                print({"scored_targets": n, "of": len(jobs), "minutes": round((time.monotonic() - start) / 60, 1)},
                      flush=True)
    write_json(os.path.join(RUN, "plan.json"), plan)
    print({"scored_targets": len(jobs), "dropped_by_qc": plan["dropped"]})


# ---------------------------------------------------------------------------------------------------- A1.1 expanded

A11_BUDGET_S = 12 * 3600


def seed_from_frozen():
    """Reuse the frozen run's masks and registrations for scans shared with the expanded selection (hard links;
    the JSON is rewritten with the expanded scan key). The same files, code and parameters, so nothing is redone."""
    frozen_run = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", DESIGNS["frozen"][0])
    frozen = {tuple(s): i for i, s in enumerate(read_json(os.path.join(PRIVATE, DESIGNS["frozen"][1][0]))["scans"])}
    for sub in ("strip", "prep"):
        os.makedirs(os.path.join(RUN, sub), exist_ok=True)
    linked = 0
    for j, s in enumerate(read_json(INPUTS["selection.json"][0])["scans"]):
        i = frozen.get(tuple(s))
        if i is None:
            continue
        a, b = f"scan_{i:03d}", f"scan_{j:03d}"
        pairs = [("strip", "_mask.nii.gz")] + [("prep", x) for x in (".nii.gz", "_brain.nii.gz", "_affine.mat")]
        for sub, suf in pairs:
            src, dst = os.path.join(frozen_run, sub, a + suf), os.path.join(RUN, sub, b + suf)
            if os.path.exists(src) and not os.path.exists(dst):
                os.link(src, dst)
        meta = os.path.join(frozen_run, "prep", a + ".json")
        if os.path.exists(meta) and not os.path.exists(os.path.join(RUN, "prep", b + ".json")):
            m = read_json(meta)
            m.update(scan=b, linked_from_frozen=a)
            write_json(os.path.join(RUN, "prep", b + ".json"), m)
            linked += 1
    print({"linked_from_frozen": linked})


def load_expanded():
    """A1.1 targets in the seeded family processing order, as scan indices (participant IDs stay inside)."""
    sel = read_json(INPUTS["selection.json"][0])
    key = {tuple(s): i for i, s in enumerate(sel["scans"])}
    first = lambda p: key[(p, "ses-01")]  # noqa: E731
    rank = {f: r for r, f in enumerate(sel["processing_order"])}
    targets = []
    for e in sel["targets"]:
        rel = {}
        if "cotwin" in e:
            rel["cotwin"] = first(e["cotwin"])
        if "sibling" in e:
            rel["sibling"] = first(e["sibling"])
        if e["repeat"]:
            rel["repeat"] = key[(e["target"], "ses-02")]
        targets.append({"id": f"T{rank[e['family']]:03d}", "order": rank[e["family"]], "target": first(e["target"]),
                        "relatives": rel, "strangers": [first(p) for p in e["strangers"]]})
    return {"n_scans": len(sel["scans"]), "targets": sorted(targets, key=lambda t: t["order"]),
            "mean_pool": [first(p) for p in sel["mean_pool"]]}


def assign_expanded(design, excluded):
    """D7 on the expanded design: an excluded target drops its family; an excluded relative drops its row; excluded
    strangers or mean-pool members are left out (counts reported)."""
    ok = lambda i: f"scan_{i:03d}" not in excluded  # noqa: E731
    out = {"targets": [], "dropped_targets": 0, "dropped_rows": {}, "mean_pool": [j for j in design["mean_pool"] if ok(j)]}
    for t in design["targets"]:
        if not ok(t["target"]):
            out["dropped_targets"] += 1
            continue
        rel = {k: v for k, v in t["relatives"].items() if ok(v)}
        for k in set(t["relatives"]) - set(rel):
            out["dropped_rows"][k] = out["dropped_rows"].get(k, 0) + 1
        out["targets"].append({**t, "relatives": rel, "strangers": [j for j in t["strangers"] if ok(j)]})
    return out


def score_target_expanded(job):
    t, metas = job["target"], job["metas"]
    out = os.path.join(RUN, "scores", f"{t['id']}.json")
    if os.path.exists(out):
        return t["id"]
    _, brain, box, mirror = space()
    c = _CACHE["box_wm"]
    x = normalized(t["target"], metas[t["target"]])
    xb = x[box]
    mu = c[job["mean_pool"]].mean(0)
    named = fills(xb, mu, c[t["strangers"][0]], other=mirrored_values(x, box, mirror), brain=brain[box])
    named.update({k: c[v] for k, v in t["relatives"].items()})
    raw, strangers = score_volume(x, box, named, {j: c[j] for j in t["strangers"]})
    ones = np.ones(xb.shape, bool)
    diag = {k: diagnostics(xb, v, mu, ones) for k, v in named.items()}
    diag.update({f"stranger_{j}": diagnostics(xb, c[j], mu, ones) for j in t["strangers"]})
    write_json(out, {"id": t["id"], "order": t["order"], "relatives": sorted(t["relatives"]),
                     "wm": {"fills": raw, "strangers": strangers}, "diagnostics": diag, "strangers": t["strangers"]})
    return t["id"]


def score_expanded(workers):
    """Score A1.1 targets in the seeded family order until all are done or the 12-hour wall budget (counted from
    the first A1.1 preparation step, recorded in a11_start.json) runs out."""
    from concurrent.futures import FIRST_COMPLETED, wait
    import nibabel as nib
    design = load_expanded()
    plan = assign_expanded(design, excluded_scans())
    start = read_json(os.path.join(RUN, "a11_start.json"))["epoch"]
    metas = [read_json(os.path.join(RUN, "prep", f"scan_{i:03d}.json")) for i in range(design["n_scans"])]
    cache_path = os.path.join(RUN, "box_cache.npz")
    if not os.path.exists(cache_path):
        _, _, box, _ = space()
        arr = np.zeros((design["n_scans"], int(box.sum())), np.float32)
        for i, m in enumerate(metas):
            if "error" not in m:
                raw = nib.load(os.path.join(RUN, "prep", f"scan_{i:03d}.nii.gz")).get_fdata(dtype=np.float32)
                arr[i] = normalized(i, m, "wm", raw)[box]
        np.savez(cache_path, box_wm=arr, box_z=arr[:1])
    os.makedirs(os.path.join(RUN, "scores"), exist_ok=True)
    todo = iter(plan["targets"])
    done, stopped = 0, False
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cache_path, max(1, 8 // workers))) as ex:
        running = set()
        while True:
            while len(running) < workers and not stopped:
                if time.time() - start > A11_BUDGET_S:
                    stopped = True
                    break
                t = next(todo, None)
                if t is None:
                    break
                running.add(ex.submit(score_target_expanded, {"target": t, "metas": metas, "mean_pool": plan["mean_pool"]}))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            done += len(finished)
            for f in finished:
                f.result()
            if done % 20 == 0:
                print({"scored_targets": done, "hours": round((time.time() - start) / 3600, 2)}, flush=True)
    write_json(os.path.join(RUN, "plan.json"), {**plan, "budget_stopped": stopped, "scored": done})
    print({"scored_targets": done, "of": len(plan["targets"]), "budget_stopped": stopped,
           "dropped_targets": plan["dropped_targets"], "dropped_rows": plan["dropped_rows"]})


def analyse_expanded(rows):
    """C1-all and C2-all (sibling rows), C11 and C12 (co-twin rows), the repeat rows and every fill, as in the
    frozen analysis; each relative row is summarised over the targets that have it."""
    rows = restore_inf(rows)
    per = []
    for r in rows:
        refs = [r["wm"]["strangers"][str(j)] for j in r["strangers"]]
        per.append({k: {"c": c_scores(v, refs), "rho": rho(v, refs)} for k, v in r["wm"]["fills"].items()})
    res = {"n_targets": len(rows), "strangers_min": min(len(r["strangers"]) for r in rows)}
    fills_ = sorted({k for p in per for k in p})
    res["c"], res["rho"], res["distance_to_mean"] = {}, {}, {}
    for k in fills_:
        sub = [(p, r) for p, r in zip(per, rows) if k in p]
        idx = boot_index(len(sub))
        res["c"][k] = {m: summary([p[k]["c"][m] for p, _ in sub], idx) for m in METRICS}
        res["rho"][k] = summary([p[k]["rho"] for p, _ in sub], idx)
        res["distance_to_mean"][k] = {d: summary([r["diagnostics"][k][d] for _, r in sub], idx)
                                      for d in ("residual_ratio", "target_projection")}
    claims = {}
    for rel, (cm, cs) in {"sibling": ("C1-all", "C2-all"), "cotwin": ("C11", "C12"), "repeat": (None, None)}.items():
        sub = [(p, r) for p, r in zip(per, rows) if rel in p]
        if not sub:
            continue
        idx = boot_index(len(sub))
        d = summary([p["mean"]["c"]["rmse"] - p[rel]["c"]["rmse"] for p, _ in sub], idx)
        crel = summary([p[rel]["c"]["rmse"] for p, _ in sub], idx)
        block = {"n_targets": len(sub), "delta_c_rmse_mean_minus_relative": d, "c_relative_rmse": crel,
                 "rho_relative": summary([p[rel]["rho"] for p, _ in sub], idx),
                 "share_mean_rmse_below_relative": float(np.mean(
                     [r["wm"]["fills"]["mean"]["rmse"] < r["wm"]["fills"][rel]["rmse"] for _, r in sub]))}
        if cm:
            name = "the co-twin" if rel == "cotwin" else "the sibling"
            block["label_mean_vs_relative"] = label_c1(d["ci95"]).replace("the sibling", name)
            block["label_relative_vs_strangers"] = label_c2(crel["ci95"]).replace("the sibling", name)
            claims[cm] = block
            claims[cs] = {"see": cm}
        else:
            claims["repeat"] = block
    claims["rho_mean_vs_model"] = {"rho_mean": res["rho"]["mean"], "model_value": 0.5 * (1 + 1 / 179)}
    return res, claims


# ---------------------------------------------------------------------------------------------------- statistics

def c_scores(raw, refs):
    """c per metric against the stranger fills, oriented so that higher means closer to the target."""
    return calibrate({m: raw[m] for m in METRICS}, [{m: r[m] for m in METRICS} for r in refs])


def rho(raw, refs):
    return raw["mse"] / float(np.mean([r["mse"] for r in refs]))


def kendall_by_metric(fill_raws):
    """Kendall tau-b of the within-target ranking of all fills under each metric against RMSE."""
    from scipy.stats import kendalltau
    ref = [r["rmse"] for r in fill_raws]
    out = {}
    for m in METRICS:
        v = [(-r[m] if m in HIGHER else r[m]) for r in fill_raws]
        out[m] = float(kendalltau(v, ref)[0])
    return out


def boot_index(n, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, (N_BOOT, n))


def summary(values, idx):
    v = np.asarray(values, float)
    bs = v[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci95": [float(lo), float(hi)], "median": float(np.median(v)), "n": int(v.size)}


def label_c1(ci):
    if ci[0] > 0:
        return "the population mean outranks the sibling under RMSE"
    if ci[1] < 0:
        return "the sibling outranks the mean"
    return "no difference resolved"


def label_c2(ci):
    return "the sibling is closer than a stranger" if ci[0] > 0.5 else "label not met (interval does not lie above 0.5)"


def per_target(row, refs_key="strangers"):
    """c, rho and RMSE ranks of every fill of one scored target (primary: white-matter scale, 16 strangers)."""
    wm = row["wm"]
    ids = [str(j) for j in row[refs_key]]
    refs = [wm["strangers"][j] for j in ids]
    out = {k: {"c": c_scores(v, refs), "rho": rho(v, refs)} for k, v in wm["fills"].items()}
    primary = [str(j) for j in row["strangers"]]
    raws = [v for k, v in wm["fills"].items() if k != "mean20"] + [wm["strangers"][j] for j in primary]
    return out, kendall_by_metric(raws)


def restore_inf(obj):
    """Undo the JSON spelling of infinite metrics (an exact fill has infinite PSNR)."""
    if isinstance(obj, dict):
        return {k: restore_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [restore_inf(v) for v in obj]
    return float(obj) if obj in ("Infinity", "-Infinity") else obj


def analyse(rows):
    """Claims C1 to C8 and sensitivity analyses 2, 3 and 5 from the private per-target rows."""
    rows = restore_inf(rows)
    res = {}
    for set_name, rel in (("S", "sibling"), ("R", "repeat")):
        rs = sorted((r for r in rows if r["set"] == set_name), key=lambda r: r["order"])
        if not rs:
            continue
        idx = boot_index(len(rs))
        tabs = [per_target(r) for r in rs]
        fills_ = [k for k in tabs[0][0]]
        block = {"n_targets": len(rs),
                 "c": {k: {m: summary([t[0][k]["c"][m] for t in tabs], idx) for m in METRICS} for k in fills_},
                 "rho": {k: summary([t[0][k]["rho"] for t in tabs], idx) for k in fills_},
                 "kendall_tau_vs_rmse": {m: summary([t[1][m] for t in tabs], idx) for m in METRICS}}
        diag = {k: {d: summary([r["diagnostics"][k][d] for r in rs], idx) for d in ("residual_ratio", "target_projection")}
                for k in rs[0]["diagnostics"] if not k.startswith("stranger_")}
        diag["strangers"] = {d: summary([np.mean([r["diagnostics"][f"stranger_{j}"][d] for j in r["strangers"]])
                                         for r in rs], idx) for d in ("residual_ratio", "target_projection")}
        block["distance_to_mean"] = diag
        rel_c = [t[0][rel]["c"]["rmse"] for t in tabs]
        mean_c = [t[0]["mean"]["c"]["rmse"] for t in tabs]
        block["mean_minus_relative_c_rmse"] = summary(np.subtract(mean_c, rel_c), idx)
        block["share_mean_rmse_below_relative"] = float(np.mean(
            [r["wm"]["fills"]["mean"]["rmse"] < r["wm"]["fills"][rel]["rmse"] for r in rs]))
        block["restored_minus_mean_c_rmse"] = summary(
            [t[0]["mean_restored"]["c"]["rmse"] - t[0]["mean"]["c"]["rmse"] for t in tabs], idx)
        sens = {}
        tabs_sex = [per_target(r, "strangers_sex")[0] for r in rs]
        sens["2_sex_matched_strangers"] = {k: summary([t[k]["c"]["rmse"] for t in tabs_sex], idx)
                                           for k in (rel, "mean", "other_hemisphere")}
        sens["2_sex_matched_strangers"]["n_strangers_min"] = min(len(r["strangers_sex"]) for r in rs)
        sens["3_mean_pool_20"] = {"c_rmse": summary([t[0]["mean20"]["c"]["rmse"] for t in tabs], idx),
                                  "rho": summary([t[0]["mean20"]["rho"] for t in tabs], idx)}
        ztabs = []
        for r in rs:
            z = r["z"]
            refs = [z["strangers"][str(j)] for j in r["strangers"]]
            ztabs.append({k: c_scores(v, refs)["rmse"] for k, v in z["fills"].items()})
        sens["5_z_scored"] = {k: summary([t[k] for t in ztabs], idx) for k in ztabs[0]}
        sens["5_z_scored"]["mean_minus_relative"] = summary([t["mean"] - t[rel] for t in ztabs], idx)
        block["sensitivity"] = sens
        res[set_name] = block
    s = res.get("S")
    claims = {}
    if s:
        d = s["mean_minus_relative_c_rmse"]
        c_sib = s["c"]["sibling"]["rmse"]
        claims["C1"] = {"delta_c_rmse_mean_minus_sibling": d, "share_mean_rmse_below_sibling":
                        s["share_mean_rmse_below_relative"], "label": label_c1(d["ci95"])}
        claims["C2"] = {"c_sibling_rmse": c_sib, "rho_sibling": s["rho"]["sibling"],
                        "r_sibling": {"mean": 1 - s["rho"]["sibling"]["mean"],
                                      "ci95": [1 - s["rho"]["sibling"]["ci95"][1], 1 - s["rho"]["sibling"]["ci95"][0]]},
                        "label": label_c2(c_sib["ci95"])}
        both = claims["C1"]["label"].startswith("the population mean") and claims["C2"]["label"].startswith("the sibling")
        claims["kate_question"] = ("the sibling carries individual anatomy, yet RMSE ranks it behind the mean" if both
                                   else "not answered with the prespecified label: both C1 and C2 labels are needed")
    claims["C3"] = {k: {"rho_mean": v["rho"]["mean"], "model_value": MODEL_RHO_MEAN} for k, v in res.items()}
    if "R" in res:
        claims["C4"] = {"c_repeat_rmse": res["R"]["c"]["repeat"]["rmse"], "c_mean_rmse": res["R"]["c"]["mean"]["rmse"],
                        "mean_minus_repeat_c_rmse": res["R"]["mean_minus_relative_c_rmse"]}
    claims["C5"] = {k: {"c_rmse": v["c"]["other_hemisphere"]["rmse"], "rho": v["rho"]["other_hemisphere"]}
                    for k, v in res.items()}
    claims["C6"] = {k: v["restored_minus_mean_c_rmse"] for k, v in res.items()}
    claims["C7"] = {k: {f: {"c_rmse": v["c"][f]["rmse"], **v["distance_to_mean"][f]}
                        for f in [f"shrink_{lam:g}" for lam in LAMBDAS] + ["wrong_detail"]} for k, v in res.items()}
    claims["C8"] = {k: {"mean_c": {f: {m: v["c"][f][m]["mean"] for m in METRICS} for f in v["c"]},
                        "kendall_tau_vs_rmse": v["kendall_tau_vs_rmse"]} for k, v in res.items()}
    return res, claims


# ---------------------------------------------------------------------------------------------------- label space

def cortical_gm(seg):
    return (seg == 3) | (seg == 42) | ((seg >= 1000) & (seg < 3000))


def label_measures(pred, oracle, min_voxels=50):
    """Inside the box: Dice per oracle label with >= min_voxels, their mean, accuracy, cortical topology errors."""
    from scipy import ndimage
    from skimage.measure import euler_number
    labels = [int(v) for v, n in zip(*np.unique(oracle, return_counts=True)) if n >= min_voxels and v != 0]
    dice = {}
    for v in labels:
        a, b = pred == v, oracle == v
        dice[v] = float(2 * np.sum(a & b) / (a.sum() + b.sum()))
    gp, go = cortical_gm(pred), cortical_gm(oracle)
    comp = lambda m: int(ndimage.label(m, structure=np.ones((3, 3, 3)))[1])  # noqa: E731
    return {"dice_mean": float(np.mean(list(dice.values()))) if dice else float("nan"), "dice": dice,
            "accuracy": float(np.mean(pred == oracle)),
            "euler_error": abs(int(euler_number(gp, connectivity=3)) - int(euler_number(go, connectivity=3))),
            "components_error": abs(comp(gp) - comp(go))}


def run_segmenter(tool, in_dir, out_dir, names, threads=8, parallel=1):
    """Segment in_dir/<name>.nii.gz for every name with the pinned image, in `parallel` containers sharing the
    threads; returns ({name: output path or None}, wall seconds)."""
    os.makedirs(out_dir, exist_ok=True)
    each = str(max(1, threads // parallel))
    start = time.monotonic()
    if tool == "synthseg":  # one container per group, the model loaded once per container
        procs = []
        for k in range(parallel):
            group = os.path.join(in_dir, f"_group{k}")
            os.makedirs(group, exist_ok=True)
            for n in names[k::parallel]:
                dst = os.path.join(group, f"{n}.nii.gz")
                if not os.path.exists(dst):
                    os.link(os.path.join(in_dir, f"{n}.nii.gz"), dst)
            procs.append(subprocess.Popen(
                ["docker", "run", "--rm", "-v", f"{group}:/in:ro", "-v", f"{out_dir}:/out", SEGMENTERS[tool],
                 "mri_synthseg", "--i", "/in", "--o", "/out", "--cpu", "--threads", each, "--keepgeom"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        for k in range(parallel):
            shutil.rmtree(os.path.join(in_dir, f"_group{k}"), ignore_errors=True)
        out = {n: os.path.join(out_dir, f"{n}_synthseg.nii.gz") for n in names}
    else:
        def one(n):
            subprocess.run(["docker", "run", "--rm", "-v", f"{in_dir}:/in:ro", "-v", f"{out_dir}:/out",
                            SEGMENTERS[tool], "--t1", f"/in/{n}.nii.gz", "--sid", n, "--sd", "/out", "--seg_only",
                            "--no_cereb", "--no_hypothal", "--no_biasfield", "--device", "cpu", "--viewagg_device",
                            "cpu", "--threads", each], capture_output=True, check=False)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(parallel) as ex:
            list(ex.map(one, names))
        out = {n: os.path.join(out_dir, n, "mri", "aparc.DKTatlas+aseg.deep.mgz") for n in names}
    return {n: p if os.path.exists(p) else None for n, p in out.items()}, time.monotonic() - start


def to_template(path):
    """A segmentation resampled to the template grid with nearest neighbour."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    ni, _, _, _ = space()
    seg = nib.load(path)
    if seg.shape[:3] != ni.shape or not np.allclose(seg.affine, ni.affine):
        seg = resample_from_to(seg, ni, order=0)
    return np.asarray(seg.dataobj).astype(np.int32)


def segtime(batch=8):
    """D8: wall time of each segmenter on the first set S target's registered scan; D9 (A1.5): throughput of the
    chosen one on `batch` copies, one container with all threads or two with half each (timing, not a result)."""
    import nibabel as nib
    design = load_design()
    i = design["S"][0]["target"]
    meta = read_json(os.path.join(RUN, "prep", f"scan_{i:03d}.json"))
    ni, _, _, _ = space()
    d = os.path.join(RUN, "segtime")
    os.makedirs(os.path.join(d, "in"), exist_ok=True)
    src = os.path.join(d, "in", "timing.nii.gz")
    nib.save(nib.Nifti1Image(normalized(i, meta), ni.affine), src)
    out = {"scan": f"scan_{i:03d}", "batch": batch}
    for tool in ("fastsurfer", "synthseg"):
        res, secs = run_segmenter(tool, os.path.join(d, "in"), os.path.join(d, tool), ["timing"])
        out[tool] = {"seconds_single": secs, "ok": res["timing"] is not None, "image": SEGMENTERS[tool]}
    fast = out["fastsurfer"]
    tool = "fastsurfer" if fast["ok"] and fast["seconds_single"] <= 300 else "synthseg"
    out["d8_choice"] = tool
    for k in range(batch):
        dst = os.path.join(d, "in", f"copy{k}.nii.gz")
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    names = [f"copy{k}" for k in range(batch)]
    for parallel in (1, 2):
        res, secs = run_segmenter(tool, os.path.join(d, "in"), os.path.join(d, f"{tool}_p{parallel}"), names,
                                  parallel=parallel)
        out[tool][f"throughput_seconds_per_volume_p{parallel}"] = secs / batch
        out[tool][f"failed_p{parallel}"] = sum(p is None for p in res.values())
    best = min((1, 2), key=lambda p: out[tool][f"throughput_seconds_per_volume_p{p}"])
    t = out[tool][f"throughput_seconds_per_volume_p{best}"]
    out.update(parallel=best, seconds_per_volume=t, n_L=int(LABEL_BUDGET_S // (LABEL_VOLUMES * t)))
    out["label_space_runs"] = out["n_L"] >= 5
    write_json(os.path.join(d, "segtime.json"), out)
    print(out)


def label(tool, n_l, parallel=1):
    """C9: label space for the first n_L scored targets of set S, plus the D8 blur sweep (private)."""
    import nibabel as nib
    from scipy.ndimage import gaussian_filter
    if n_l < 5:
        raise SystemExit("n_L below 5: label space is reported as not run (D9)")
    plan = read_json(os.path.join(RUN, "plan.json"))
    metas = [read_json(os.path.join(RUN, "prep", f"scan_{i:03d}.json")) for i in range(load_design()["n_scans"])]
    ni, brain, box, mirror = space()
    with np.load(os.path.join(RUN, "box_cache.npz")) as f:
        cache = f["box_wm"]
    sl = box_slices(box)
    for t in plan["S"][:n_l]:
        d = os.path.join(RUN, "label", t["id"])
        out_path = os.path.join(d, "label.json")
        if os.path.exists(out_path):
            continue
        x = normalized(t["target"], metas[t["target"]])
        xb = x[box]
        mu = cache[plan["mean_pool"]].mean(0)
        named = fills(xb, mu, cache[t["strangers"][0]], cache[t["donor"]], mirrored_values(x, box, mirror), brain[box])
        named = {("sibling" if k == "relative" else k): v for k, v in named.items()}
        named["oracle"] = xb
        named.update({f"stranger_{j}": cache[j] for j in t["strangers"]})
        for s in BLUR_SIGMAS:
            named[f"blur_{s}"] = gaussian_filter(x, s)[box]
        os.makedirs(os.path.join(d, "in"), exist_ok=True)
        for k, v in named.items():
            f = x.copy()
            f[box] = v
            nib.save(nib.Nifti1Image(f, ni.affine), os.path.join(d, "in", f"{k}.nii.gz"))
        segs, secs = run_segmenter(tool, os.path.join(d, "in"), os.path.join(d, "seg"), list(named), parallel=parallel)
        failed = [k for k, p in segs.items() if p is None]
        if failed:  # D8: one retry, then counted as failed for its rung
            retry, _ = run_segmenter(tool, os.path.join(d, "in"), os.path.join(d, "seg_retry"), failed)
            segs.update({k: p for k, p in retry.items() if p is not None})
        if segs["oracle"] is None:
            write_json(out_path, {"id": t["id"], "failed": sorted(k for k, p in segs.items() if p is None)})
            continue
        oracle = to_template(segs["oracle"])[sl]
        meas = {k: label_measures(to_template(p)[sl], oracle) for k, p in segs.items() if p is not None and k != "oracle"}
        write_json(out_path, {"id": t["id"], "tool": tool, "seconds": secs, "measures": meas,
                              "failed": sorted(k for k, p in segs.items() if p is None)})
        shutil.rmtree(os.path.join(d, "in"), ignore_errors=True)
        print({"label_target": t["id"], "minutes": round(secs / 60, 1), "failed": len(meas) < len(named) - 1}, flush=True)


def analyse_labels(rows):
    """C9 aggregates: blur sweep gate, then c of mean, sibling and other hemisphere under label measures."""
    rows = [r for r in rows if "measures" in r]
    if not rows:
        return {"run": False}
    idx = boot_index(len(rows))
    n_fail = sum(len(r["failed"]) for r in rows)
    drop = {s: summary([1 - r["measures"][f"blur_{s}"]["dice_mean"] for r in rows], idx) for s in BLUR_SIGMAS}
    sees_blur = drop[3]["ci95"][0] > 0
    out = {"run": True, "n_targets": len(rows), "failed_segmentations": n_fail, "blur_dice_drop": drop,
           "segmenter_sees_blur": sees_blur}
    keys = ("dice_mean", "accuracy", "euler_error", "components_error")
    cs = {}
    for k in ("mean", "sibling", "other_hemisphere", "mean_restored", "wrong_detail"):
        per = []
        for r in rows:
            refs = [v for n, v in r["measures"].items() if n.startswith("stranger_")]
            if k not in r["measures"] or not refs:
                continue
            v = r["measures"][k]
            per.append({m: float(np.mean([(ref[m] < v[m]) if m in LABEL_HIGHER else (ref[m] > v[m]) for ref in refs])
                                 + 0.5 * np.mean([ref[m] == v[m] for ref in refs])) for m in keys})
        if per:
            i2 = boot_index(len(per))
            cs[k] = {m: summary([p[m] for p in per], i2) for m in keys}
    out["c"] = cs
    out["interpretation"] = ("interpreted: the segmenter sees blur" if sees_blur else
                             "uninformative for this segmenter: Dice does not fall with blur")
    return out


# ---------------------------------------------------------------------------------------------------- report

def report():
    hashes = guard("report")
    rows = [read_json(os.path.join(RUN, "scores", f)) for f in sorted(os.listdir(os.path.join(RUN, "scores")))]
    plan = read_json(os.path.join(RUN, "plan.json"))
    qcs = read_json(os.path.join(RUN, "qc", "qc_summary.json"))
    res, claims = analyse(rows)
    lab_dir = os.path.join(RUN, "label")
    lab_rows = [read_json(os.path.join(lab_dir, d, "label.json")) for d in sorted(os.listdir(lab_dir))
                if os.path.exists(os.path.join(lab_dir, d, "label.json"))] if os.path.isdir(lab_dir) else []
    claims["C9"] = analyse_labels(lab_rows)
    reasons = {}
    for r in qcs["excluded"].values():
        key = "unreadable or failed preparation" if ":" in r and "inside own brain" not in r else r.split(" ")[0]
        reasons[key] = reasons.get(key, 0) + 1
    out = {"protocol": "docs/PROTOCOL_real_cases.md", "dataset": "QTIM, OpenNeuro ds004169 v1.0.8 (CC0)",
           "space": "affine template space, 1 mm, white-matter peak scale, template brain mask",
           "qc": {"scans": load_design()["n_scans"], "excluded": len(qcs["excluded"]), "excluded_by_reason": reasons,
                  "flagged_for_review": len(qcs["flagged_for_review"]), "dropped_targets": plan["dropped"]},
           "pools": {"strangers": len(plan["stranger_pool"]), "mean": len(plan["mean_pool"]), "K": K},
           "claims": claims, "sets": res,
           "sensitivity_not_run": {"1_orientation_unmatched": "no sagittal scans in the frozen download list",
                                   "4_syn": "not run in this pass", "6_other_boxes": "coordinates not yet amended",
                                   "7_other_segmenter": "not run in this pass"}}
    write_json(os.path.join(ROOT, "results", "real_cases.json"), out)
    code = {f: sha256(os.path.join(ROOT, "src", f)) for f in ("real_cases.py", "atlas_common.py", "population_null.py",
                                                            "toy_ladder.py")}
    write_json(os.path.join(ROOT, "results", "real_cases_manifest.json"), {
        "protocol_sha256": sha256(PROTOCOL), "protocol_commit": git("log", "-1", "--format=%H", "--", PROTOCOL),
        "git_commit": git("rev-parse", "HEAD"), "code_sha256": code, "input_sha256": hashes, "seed": SEED,
        "n_bootstrap": N_BOOT, "versions": {**{p: version(p) for p in VERSIONS}, "python": platform.python_version(),
                                            "torch": version("torch"), "scipy": version("scipy")},
        "segmenters": SEGMENTERS, "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    print(json.dumps({"C1": claims.get("C1", {}).get("label"), "C2": claims.get("C2", {}).get("label")}))


def report_expanded():
    hashes = guard("report")
    rows = [read_json(os.path.join(RUN, "scores", f)) for f in sorted(os.listdir(os.path.join(RUN, "scores")))]
    plan = read_json(os.path.join(RUN, "plan.json"))
    qcs = read_json(os.path.join(RUN, "qc", "qc_summary.json"))
    res, claims = analyse_expanded(rows)
    claims["rho_mean_vs_model"]["model_value"] = 0.5 * (1 + 1 / len(plan["mean_pool"]))
    write_json(os.path.join(ROOT, "results", "real_cases_expanded.json"), {
        "protocol": "docs/PROTOCOL_real_cases.md, Amendment 1 (A1.1)", "dataset": "QTIM, OpenNeuro ds004169 v1.0.8 (CC0)",
        "qc": {"scans": load_expanded()["n_scans"], "excluded": len(qcs["excluded"]),
               "dropped_targets": plan["dropped_targets"], "dropped_rows": plan["dropped_rows"]},
        "budget": {"hours": A11_BUDGET_S / 3600, "stopped": plan["budget_stopped"], "scored_targets": plan["scored"],
                   "eligible_targets": len(plan["targets"])},
        "pools": {"mean": len(plan["mean_pool"]), "strangers_per_target_min": res["strangers_min"]},
        "claims": claims, "fills": res})
    write_json(os.path.join(ROOT, "results", "real_cases_expanded_manifest.json"), {
        "protocol_sha256": sha256(PROTOCOL), "protocol_commit": git("log", "-1", "--format=%H", "--", PROTOCOL),
        "git_commit": git("rev-parse", "HEAD"), "input_sha256": hashes, "seed": SEED, "n_bootstrap": N_BOOT,
        "code_sha256": {f: sha256(os.path.join(ROOT, "src", f)) for f in ("real_cases.py", "atlas_common.py",
                                                                         "population_null.py", "toy_ladder.py")},
        "versions": {**{p: version(p) for p in VERSIONS}, "python": platform.python_version()},
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith("label")} for k, v in claims.items()
                      if k in ("C1-all", "C11")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["seed", "strip", "prepare", "qc", "dataqc", "segtime", "score", "label",
                                      "report"])
    ap.add_argument("--design", choices=list(DESIGNS), default="frozen")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--visual-fail", default="", help="comma-separated flagged scans with a visible failure (D7)")
    ap.add_argument("--tool", choices=list(SEGMENTERS))
    ap.add_argument("--n-l", type=int)
    a = ap.parse_args()
    use_design(a.design)
    guard(a.stage)
    if DESIGN == "expanded" and a.stage in ("seed", "strip", "prepare"):
        start = os.path.join(RUN, "a11_start.json")  # the 12-hour A1.1 budget runs from the first preparation step
        if not os.path.exists(start):
            write_json(start, {"epoch": time.time(), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if a.stage == "seed":
        seed_from_frozen()
    elif a.stage == "strip":
        strip(a.limit)
    elif a.stage == "prepare":
        prepare(a.limit, a.workers)
    elif a.stage == "qc":
        qc([s for s in a.visual_fail.split(",") if s])
    elif a.stage == "dataqc":
        dataqc()
    elif a.stage == "segtime":
        segtime()
    elif a.stage == "score":
        score_expanded(a.workers) if DESIGN == "expanded" else score(a.workers)
    elif a.stage == "label":
        timing = read_json(os.path.join(RUN, "segtime", "segtime.json"))
        label(a.tool or timing["d8_choice"], a.n_l or timing["n_L"], timing["parallel"])
    else:
        report_expanded() if DESIGN == "expanded" else report()


if __name__ == "__main__":
    main()
