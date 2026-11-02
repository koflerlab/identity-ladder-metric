"""Amendment 4 preparation, isolated from the historical affine run.

Only counts are printed. Images, transforms and per-scan metadata stay private.
No scoring is permitted by this module before a separate visual-QC review.
"""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import real_cases as rc


def reflect(volume, affine):
    """Exact physical x -> -x on the frozen symmetric 1 mm RAS grid."""
    if not np.allclose(affine[:3, :3], np.eye(3)):
        raise ValueError("reflection requires the frozen 1 mm RAS grid")
    if not np.isclose(2 * affine[0, 3] + volume.shape[0] - 1, 0):
        raise ValueError("grid is not centred on the midsagittal plane")
    return np.flip(volume, axis=0)


def scaled(volume, brain, box, kind):
    visible = volume[brain & ~box]
    p90, mean, sd = float(np.percentile(visible, 90)), float(visible.mean()), float(visible.std())
    if not np.isfinite([p90, mean, sd]).all() or p90 <= 0 or sd <= 0:
        raise ValueError("invalid intensity scale")
    result = volume / p90 if kind == "p90" else (volume - mean) / sd + rc.Z_OFFSET
    if kind not in ("p90", "z"):
        raise ValueError("unknown scale")
    if np.any(result[brain] < 0):
        raise ValueError("negative in-brain values would be clipped by the package")
    return np.where(brain, result, 0).astype(np.float32), {"p90": p90, "mean": mean, "sd": sd}


def corr(a, b, mask):
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def prepare_one(job, output):
    import ants
    import nibabel as nib
    i, relative, expected = job
    dest = output / f"scan_{i:03d}"
    dest.mkdir(exist_ok=True)
    complete = dest / "complete.json"
    fingerprint = {"source_sha256": expected, "protocol_sha256": rc.sha256(rc.PROTOCOL),
                   "code_sha256": rc.sha256(__file__), "core_sha256": rc.sha256(rc.__file__)}
    if complete.exists():
        prior = rc.read_json(complete)
        if prior["inputs"] != fingerprint:
            raise RuntimeError("completed preparation belongs to a different configuration")
        for name, digest in prior["outputs"].items():
            if rc.sha256(dest / name) != digest:
                raise RuntimeError("prepared output hash mismatch")
        return
    start = time.monotonic()
    source = Path(rc.BIDS) / relative
    if rc.sha256(source) != expected:
        raise RuntimeError("source differs from the scan manifest")
    ants.config.set_ants_deterministic(False, seed_value=123)
    native = ants.image_read(str(source))
    small = ants.resample_image(native, (2, 2, 2), use_voxels=False, interp_type=0)
    bias = ants.n4_bias_field_correction(small, shrink_factor=4, return_bias_field=True)
    native_bias = ants.resample_image_to_target(bias, native, interp_type="linear")
    corrected = native.new_image_like(native.numpy() / np.maximum(native_bias.numpy(), 1e-6))
    ants.image_write(corrected, str(dest / "n4.nii.gz"))
    # SynthStrip must see the corrected scan, rather than reuse the raw-image mask.
    command = ["docker", "run", "--rm", "--cpus=2", "-v", f"{dest.resolve()}:/work",
               rc.SEGMENTERS["synthseg"], "mri_synthstrip", "-i", "/work/n4.nii.gz",
               "-m", "/work/mask.nii.gz", "-t", "2"]
    run = subprocess.run(command, capture_output=True, text=True)
    (dest / "synthstrip.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError("SynthStrip failed; see private log")
    mask = ants.image_read(str(dest / "mask.nii.gz"))
    templates = Path(rc.RUN) / "template"
    fixed = ants.image_read(str(templates / "template_2mm.nii.gz"))
    fixed_mask = ants.image_read(str(templates / "brain_2mm.nii.gz"))
    fixed = fixed * fixed_mask
    moving = ants.resample_image_to_target(corrected * mask, small)
    moving_mask = ants.resample_image_to_target(mask, small, interp_type="nearestNeighbor")
    common = dict(fixed=fixed, moving=moving, mask=fixed_mask, moving_mask=moving_mask,
                  mask_all_stages=True, aff_iterations=(1000, 500, 250, 50),
                  aff_shrink_factors=(8, 4, 2, 1), aff_smoothing_sigmas=(3, 2, 1, 0),
                  aff_metric="mattes", verbose=False)
    rigid = ants.registration(**common, type_of_transform="Rigid", outprefix=str(dest / "rigid_"))
    affine = ants.registration(**common, type_of_transform="Affine", initial_transform=rigid["fwdtransforms"],
                               outprefix=str(dest / "affine_"))
    syn = ants.registration(**common, type_of_transform="SyNOnly", initial_transform=affine["fwdtransforms"],
                            syn_metric="CC", syn_sampling=4, reg_iterations=(100, 70, 50),
                            grad_step=0.2, flow_sigma=3, total_sigma=0, outprefix=str(dest / "syn_"))
    template1 = ants.image_read(str(templates / "template_1mm.nii.gz"))
    ni = nib.load(templates / "template_1mm.nii.gz")
    brain = nib.load(templates / "brain_1mm.nii.gz").get_fdata() > 0
    reflect(brain, ni.affine)
    stats = {}
    for name, reg in (("affine", affine), ("syn", syn)):
        warped = ants.apply_transforms(template1, corrected, reg["fwdtransforms"], interpolator="linear")
        own = ants.apply_transforms(template1, mask, reg["fwdtransforms"], interpolator="nearestNeighbor")
        ants.image_write(warped, str(dest / f"{name}.nii.gz"))
        ants.image_write(own, str(dest / f"{name}_mask.nii.gz"))
        v, b = warped.numpy(), own.numpy() > 0.5
        stats[name] = {"template_corr": corr(v, template1.numpy(), brain), "boxes": {}}
        for box_name, (bounds, _) in rc.BOXES.items():
            box = rc.box_mask(ni.shape, ni.affine, bounds)
            _, scale = scaled(v, brain, box, "p90")
            stats[name]["boxes"][box_name] = {**scale,
                "template_corr": corr(v, template1.numpy(), box & brain),
                "coverage": float(b[box].mean()),
                "mirror_coverage": float(reflect(b, ni.affine)[box].mean())}
    outputs = {p.name: rc.sha256(p) for p in dest.iterdir() if p.suffix in (".gz", ".mat")}
    rc.write_json(complete, {"inputs": fingerprint, "outputs": outputs, "stats": stats,
                            "seconds": time.monotonic() - start,
                            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "visual_qc": "pending", "metric_scoring": "not_started"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1, help="bounded preparation pilot")
    ap.add_argument("--design", choices=list(rc.DESIGNS), default="frozen")
    ap.add_argument("--wait-for-storage", action="store_true")
    a = ap.parse_args()
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    rc.use_design(a.design)
    rc.guard("score")  # includes committed-protocol and package/hash checks
    lock = Path(rc.DATA_ROOT) / "_registry" / "storage-maintenance.lock"
    announced = False
    while True:
        try:
            handle = lock.open("x", encoding="utf-8")
            break
        except FileExistsError:
            if not a.wait_for_storage:
                raise SystemExit("storage maintenance lock held by another session")
            if not announced:
                print({"state": "waiting_for_storage_lock", "metrics": "not_started"}, flush=True)
                announced = True
            time.sleep(10)
    try:
        import json
        with handle:
            json.dump({"owner": "identity-ladder-metric SyN pilot", "pid": os.getpid()}, handle)
        output = Path(rc.RUN) / "syn_amendment4"
        output.mkdir(exist_ok=True)
        for n, job in enumerate(rc.scan_jobs(a.limit), 1):
            prepare_one(job, output)
            print({"prepared": n, "qc": "pending", "metrics": "not_started"}, flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
