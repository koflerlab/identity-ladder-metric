"""Exploratory H19 sensitivity experiment; independent simulated cohorts, paired priors.

No real data. Five seeds x two variance settings x three prior strengths.
Seed ranges quantify simulation variability, not confidence intervals. Twin contrasts
are paired within each target and restricted to the appropriate MZ or DZ families.
Existing results are preserved; resumable checkpoints reject configuration drift.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import torch

import label_ladder as ll

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "label_robustness.json"
METRICS = ("rmse", "mse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim",
           "dice", "accuracy") + tuple(f"topo_err_{q}" for q in ll.TOPO)
LOWER = {"rmse", "mse", "mae", "msle"} | {f"topo_err_{q}" for q in ll.TOPO}


def calibrated(row, fill, metric):
    ref = np.array([v[metric] for k, v in row.items() if k.startswith("stranger_")])
    x = row[fill][metric]
    wins = ref > x if metric in LOWER else ref < x
    return float(np.mean(wins) + .5 * np.mean(ref == x))


def paired_twin_contrasts(rows):
    out = {}
    for z in ("MZ", "DZ"):
        selected = [r for r in rows if r["_zygosity"] == z]
        out[z] = {"n_targets": len(selected), "n_families": len({r["_family"] for r in selected}), "metrics": {}}
        for m in METRICS:
            a = [calibrated(r, "population_mean", m) for r in selected]
            b = [calibrated(r, z, m) for r in selected]
            out[z]["metrics"][m] = {"mean_c_population": float(np.mean(a)), "mean_c_twin": float(np.mean(b)),
                                      "delta_mean_c": float(np.mean(np.array(a) - b))}
    return out


def variation(values):
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "sd_across_seeds": float(a.std(ddof=1)) if len(a) > 1 else None, "n_seeds": len(a)}


def summarize(runs):
    out = {}
    for config in sorted({r["config"] for r in runs}):
        group = [r for r in runs if r["config"] == config]
        c = {"n_seeds": len(group), "metrics": {}, "paired_twin_delta": {}, "oracle_topology_mean": {}}
        for m in METRICS:
            c["metrics"][m] = {f: variation([r["scores"][m][f]["mean_c"] for r in group])
                                for f in group[0]["scores"][m]}
        for z in ("MZ", "DZ"):
            c["paired_twin_delta"][z] = {m: variation([r["paired_twin_contrasts"][z]["metrics"][m]["delta_mean_c"]
                                                       for r in group]) for m in METRICS}
        for q in ll.TOPO:
            c["oracle_topology_mean"][q] = variation([r["scores"]["oracle_topology_mean"][q] for r in group])
        out[config] = c
    return out


def main():
    torch.set_num_threads(1)
    spec = {"seeds": [101, 102, 103, 104, 105], "prior_scales": [0.0, 0.25, 1.0],
            "settings": {"base": ll.BASE, "low_h": ll.LOW_H}, "n_fam_per_seed": 60,
            "n_targets_per_seed": 120, "n_strangers_per_target": 16,
            "source_sha256": {p: hashlib.sha256((ROOT / "src" / p).read_bytes()).hexdigest()
                              for p in ("label_ladder.py", "toy_ladder.py", "label_robustness.py")},
            "versions": {p: version(p) for p in ("numpy", "scipy", "scikit-image", "torch", "torchmetrics", "inpainting")}}
    out = {"design": spec, "status": "in_progress", "runs": [],
           "interpretation_limits": ["Exploratory sensitivity analysis, not confirmatory validation.",
             "Seeds are independent simulated cohorts; priors and variance settings reuse random draws.",
             "Targets share family effects, donors and population means within each seed.",
             "Min/max are ranges over five seed-level means, not confidence intervals.",
             "Genetic-share labels refer to input variance parameters, not heritability of nonlinear images or masks.",
             "Topology is computed after intersecting with the mask; border cuts affect it. Perimeter is geometric, not a topological invariant.",
             "Oracle is the segmentation of the noisy true image, not the latent label map. No real MRI or FastSurfer."]}
    if DEST.exists():
        out = json.loads(DEST.read_text())
        if out["design"] != spec:
            raise ValueError("Checkpoint design changed; use a new output path")
    done = {(r["config"], r["seed"]) for r in out["runs"]}
    for seed in spec["seeds"]:
        for name, params in spec["settings"].items():
            for prior in spec["prior_scales"]:
                key = f"{name}_prior{prior:g}"
                if (key, seed) in done:
                    continue
                start = time.perf_counter()
                scores, rows = ll.run(params, np.random.default_rng(seed), prior_scale=prior, return_rows=True)
                out["runs"].append({"config": key, "seed": seed, "prior_scale": prior,
                                    "scores": scores, "paired_twin_contrasts": paired_twin_contrasts(rows),
                                    "runtime_s": round(time.perf_counter() - start, 3)})
                out["summary"] = summarize(out["runs"])
                tmp = DEST.with_suffix(".tmp")
                tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
                tmp.replace(DEST)
                print(f"{len(out['runs'])}/30 {key} seed={seed} {time.perf_counter()-start:.1f}s", flush=True)
    out["status"] = "complete"
    DEST.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
