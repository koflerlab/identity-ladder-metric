"""Claim C10 (protocol D11, A1.7): the rank-1 BraTS challenge models BraTS23_1 and BraTS24_1 as fills, scored in
the target's BraTS space next to the sibling, the population mean and 16 strangers.

Per target of set S (seeded order, QC-passing), in the frozen design's private run folder:
  1. the N4-corrected native T1 (same N4 as `real_cases.prepare_scan`) goes through BraTS-style preprocessing
     (`brats_helper.py preprocess`, separate venv): rigid registration to SRI24 with atlas correction, HD-BET;
  2. the saved transforms are checked: they must reproduce the preprocessed T1 from the native one;
  3. the template box is mapped into BraTS space (nearest neighbour); the input is the target with the box voided;
  4. every donor (sibling, 16 strangers, the 60 mean-pool scans) is brought from its native N4-corrected scan to
     the target's BraTS space with ONE resampling (donor affine, inverse target affine, target's BraTS transforms),
     scaled by the ratio of P90 scales (Amendment 3) and masked with the target's brain mask;
  5. each model fills the box (`brats_helper.py infer`, force_cpu, at most 20 minutes, one retry);
  6. all fills are scored with the nine package metrics on the 240 x 240 x 155 volume, c against the 16 strangers,
     rho, distance to the mean and target projection.
The first target times each model: n_B = min(49, floor(4 h / time per fill)), at least 5, else not feasible.
Prints scan keys and counts only.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_cases as rc  # noqa: E402

ALGOS = {"BraTS23_1": "brainles/brats23_inpainting_ying_weng", "BraTS24_1": "brainles/brats24_inpainting_ying_weng"}
FILL_LIMIT_S, MODEL_BUDGET_S, N_MAX = 20 * 60, 4 * 3600, 49
VENV_PY = os.path.join(rc.ROOT, "external", "venv-brats", "Scripts", "python.exe")
HELPER = os.path.join(rc.ROOT, "src", "brats_helper.py")


def bdir(*parts):
    return os.path.join(rc.RUN, "brats", *parts)


def native_n4(i):
    """N4-corrected native scan i as in prepare_scan (cached, private)."""
    import ants
    path = bdir("n4", f"scan_{i:03d}.nii.gz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rel = rc.scan_jobs()[i][1]
        native = ants.image_read(os.path.join(rc.BIDS, *rel.split("/")))
        small = ants.resample_image(native, (2, 2, 2), use_voxels=False, interp_type=0)
        bias = ants.n4_bias_field_correction(small, shrink_factor=4, return_bias_field=True)
        nb = ants.resample_image_to_target(bias, native, interp_type="linear")
        ants.image_write(native.new_image_like(native.numpy() / np.maximum(nb.numpy(), 1e-6)), path)
    return path


def helper(*args, timeout=None):
    out = subprocess.run([VENV_PY, HELPER, *args], capture_output=True, text=True, timeout=timeout, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def brats_transforms(d):
    """Native-to-atlas transforms in ANTs order for apply_transforms (atlas correction, then atlas registration)."""
    files = sorted(glob.glob(os.path.join(d, "transforms", "t1n", "*.mat")))
    corr = [f for f in files if "correction" in os.path.basename(f)]
    reg = [f for f in files if "correction" not in os.path.basename(f)]
    if len(reg) != 1 or len(corr) > 1:
        raise RuntimeError(f"unexpected transforms: {[os.path.basename(f) for f in files]}")
    return corr + reg


def kill_model_containers():
    for image in ALGOS.values():
        ids = subprocess.run(["docker", "ps", "-q", "--filter", f"ancestor={image}"], capture_output=True,
                             text=True).stdout.split()
        if ids:
            subprocess.run(["docker", "kill", *ids], capture_output=True)


def run_model(algo, d):
    """One inference within the 20-minute limit, retried once on failure; returns (seconds or None, status)."""
    out = os.path.join(d, f"fill_{algo}.nii.gz")
    for attempt in (1, 2):
        start = time.monotonic()
        try:
            helper("infer", algo, os.path.join(d, "t1n_voided.nii.gz"), os.path.join(d, "mask.nii.gz"), out,
                   timeout=FILL_LIMIT_S)
            if os.path.exists(out):
                return time.monotonic() - start, "ok"
        except subprocess.TimeoutExpired:
            kill_model_containers()
            return None, "timed out"
        except subprocess.CalledProcessError:
            kill_model_containers()
    return None, "failed"


def prepare_target(t, metas):
    """Steps 1 to 4 for one target: BraTS-space target, box, voided input and every donor's box values."""
    import ants
    d = bdir(t["id"])
    if not os.path.exists(os.path.join(d, "t1n_bet.nii.gz")):
        helper("preprocess", native_n4(t["target"]), d)
    atlas = ants.image_read(os.path.join(d, "t1n_bet.nii.gz"))
    bet = ants.image_read(os.path.join(d, "bet_mask.nii.gz")).numpy() > 0.5
    tr = brats_transforms(d)
    check = ants.apply_transforms(atlas, ants.image_read(native_n4(t["target"])), tr, interpolator="linear").numpy()
    corr = float(np.corrcoef(check[bet], atlas.numpy()[bet])[0, 1])
    if corr < 0.99:
        raise RuntimeError(f"{t['id']}: saved transforms do not reproduce the BraTS-space T1 (r={corr:.3f})")
    prep = lambda i: os.path.join(rc.RUN, "prep", f"scan_{i:03d}_affine.mat")  # noqa: E731
    ni, _, box_t, _ = rc.space()
    tpl = ants.image_read(os.path.join(rc.RUN, "template", "template_1mm.nii.gz"))
    box = ants.apply_transforms(atlas, tpl.new_image_like(box_t.astype(np.float32)), tr + [prep(t["target"])],
                                whichtoinvert=[False] * len(tr) + [True], interpolator="nearestNeighbor").numpy() > 0.5
    x = atlas.numpy().astype(np.float32)
    ants.image_write(atlas.new_image_like((x * ~box).astype(np.float32)), os.path.join(d, "t1n_voided.nii.gz"))
    ants.image_write(atlas.new_image_like(box.astype(np.float32)), os.path.join(d, "mask.nii.gz"))
    peak_t = metas[t["target"]]["scale"]["deep"]["p90"]  # Amendment 3 P90 scale

    def donor(j):
        moved = ants.apply_transforms(atlas, ants.image_read(native_n4(j)), tr + [prep(t["target"]), prep(j)],
                                      whichtoinvert=[False] * len(tr) + [True, False], interpolator="linear").numpy()
        return (moved * (peak_t / metas[j]["scale"]["deep"]["p90"]) * bet)[box].astype(np.float32)
    return x, box, bet, corr, donor


def score_target(t, plan, metas, algos):
    out = bdir(t["id"], "scores.json")
    if os.path.exists(out):
        return rc.read_json(out)
    x, box, bet, corr, donor = prepare_target(t, metas)
    xb = x[box]
    strangers = {j: donor(j) for j in t["strangers"]}
    mu = np.mean([donor(j) for j in plan["mean_pool"]], axis=0)
    named = {"sibling": donor(t["donor"]), "mean": mu}
    named.update({f"shrink_{lam:g}": rc.shrinkage(xb, mu, lam) for lam in rc.LAMBDAS})
    timing = {}
    import nibabel as nib
    for algo in algos:
        secs, status = run_model(algo, bdir(t["id"]))
        timing[algo] = {"seconds": secs, "status": status}
        if status == "ok":
            named[algo] = np.asarray(nib.load(bdir(t["id"], f"fill_{algo}.nii.gz")).dataobj, np.float32)[box]
    raw, refs = rc.score_volume(x, box, named, strangers)
    ones = np.ones(xb.shape, bool)
    row = {"id": t["id"], "order": t["order"], "transform_check_r": corr, "box_voxels": int(box.sum()),
           "timing": timing, "wm": {"fills": raw, "strangers": refs}, "strangers": t["strangers"],
           "diagnostics": {k: rc.diagnostics(xb, v, mu, ones) for k, v in named.items()}}
    rc.write_json(out, row)
    return row


def run(n_override=None):
    rc.guard("score")
    plan = rc.read_json(os.path.join(rc.RUN, "plan.json"))
    metas = [rc.read_json(os.path.join(rc.RUN, "prep", f"scan_{i:03d}.json")) for i in range(rc.load_design()["n_scans"])]
    targets = plan["S"]
    first = score_target(targets[0], plan, metas, list(ALGOS))
    n_b = {}
    for algo, tm in first["timing"].items():
        s = tm["seconds"]
        n = 0 if s is None or s > FILL_LIMIT_S else min(N_MAX, int(MODEL_BUDGET_S // s))
        n_b[algo] = n if n >= 5 else 0
    rc.write_json(bdir("n_b.json"), {"first_target_timing": first["timing"], "n_B": n_b})
    print({"n_B": n_b, "first_timing": first["timing"]}, flush=True)
    for k, t in enumerate(targets[1:], 1):
        algos = [a for a, n in n_b.items() if k < (n_override or n)]
        if not algos:
            break
        row = score_target(t, plan, metas, algos)
        print({"target": t["id"], "timing": row["timing"]}, flush=True)


def analyse(rows):
    """C10 aggregates: c and rho of each model fill, the sibling and the mean on the same targets."""
    rows = rc.restore_inf(rows)
    out = {}
    for algo in ALGOS:
        sub = [r for r in rows if algo in r["wm"]["fills"]]
        if not sub:
            out[algo] = {"run": False}
            continue
        idx = rc.boot_index(len(sub))
        per = []
        for r in sub:
            refs = [r["wm"]["strangers"][str(j)] for j in r["strangers"]]
            per.append({k: {"c": rc.c_scores(v, refs), "rho": rc.rho(v, refs)} for k, v in r["wm"]["fills"].items()})
        out[algo] = {"n_targets": len(sub),
                     "c": {k: {m: rc.summary([p[k]["c"][m] for p in per], idx) for m in rc.METRICS}
                           for k in (algo, "sibling", "mean")},
                     "rho": {k: rc.summary([p[k]["rho"] for p in per], idx) for k in (algo, "sibling", "mean")},
                     "distance_to_mean": {d: rc.summary([r["diagnostics"][algo][d] for r in sub], idx)
                                          for d in ("residual_ratio", "target_projection")},
                     "seconds_per_fill": rc.summary([r["timing"][algo]["seconds"] for r in sub], idx)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="cap n_B for a smoke run")
    run(ap.parse_args().n)
