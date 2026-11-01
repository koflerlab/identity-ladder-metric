"""H19: does a pointwise downstream metric (Dice / accuracy against the oracle segmentation) reward an identity-free
average fill the way squared error does? Synthetic 2D tissue with two tissue classes; only synthetic data.

Anatomy: label(x) = 1[prior_logit(x) + a * F_p(x) > 0], with a population prior (strong in some places, ambiguous in
others) and an individual smooth field F = g + e + b (variance components as in toy_ladder, shared by co-twins and
across sides as there). Image: 100 + 30 * tanh(2 * (prior + a F)) + sitting offset + noise. Segmenter: threshold at 100
after a light Gaussian smoothing (a stand-in for FastSurfer). Oracle: the segmentation of the true image.
Bayes view: per pixel, the mode label agrees with the truth with probability max(p, 1 - p), an independent stranger
with p^2 + (1 - p)^2, which is never larger; so the average fill should beat a stranger in Dice, and a co-twin when the
shared part is small. Topology of the masked segmentation (components, holes, Euler number, boundary length) is
scored as the absolute error against the oracle's and calibrated into c against the stranger fills like any metric.
'prior_scale' weakens the population prior: at 0.25 the individual field dominates and the anatomy is folding-like,
so an average fill can no longer copy the topology from the prior.

Fills: same sitting, other day, other side, co-twin, stranger (16 per case), population mean (other families),
blurred stranger, blurred population mean. Intensity metrics from the official BraTS code; c per case against the
stranger fills. Output: results/label_ladder.json.
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import euler_number, label, perimeter

sys.path.insert(0, os.path.dirname(__file__))
from toy_ladder import BASE, LOW_H, field, panel  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
H = W = 64
SEED = 2
yy, xx = np.mgrid[:H, :W]
PRIOR = 2.0 * np.sin(2 * np.pi * xx / 24.0) * np.cos(2 * np.pi * yy / 32.0)  # strong where |PRIOR| large
MASK = np.zeros((H, W), bool)
MASK[16:48, 16:48] = True


def people(s, n_fam, rng, amp=1.0, prior_scale=1.0):
    acq, lab, zyg = {}, {}, {}
    for f in range(n_fam):
        mz = f % 2 == 0
        zyg[f] = "MZ" if mz else "DZ"
        shared, own = field(rng, s["g"], 3), [field(rng, s["g"], 3) for _ in range(2)]
        for t in range(2):
            g = shared if mz else np.sqrt(0.5) * shared + np.sqrt(0.5) * own[t]
            e = field(rng, s["e"], 3)
            for side in "LR":
                b = field(rng, s["b"], 3)
                z = prior_scale * PRIOR + amp * (g + e + b)
                lab[(f, t, side)] = z > 0
                for k in ("S0", "S1"):
                    off = 10 * rng.normal(0, s["eta"])
                    for r in range(2):
                        acq[(f, t, side, k, r)] = 100 + 30 * np.tanh(2 * z) + off + 10 * rng.normal(0, s["eps"], (H, W))
    return acq, lab, zyg


def segment(img):
    return gaussian_filter(img, 1.0) > 100


def topo(seg):
    m = seg & MASK
    comp, eul = int(label(m, connectivity=1).max()), int(euler_number(m, connectivity=1))
    return {"components": comp, "holes": comp - eul, "euler": eul, "perimeter": float(perimeter(m))}


TOPO = ("components", "holes", "euler", "perimeter")


def dice(a, b):
    a, b = a[MASK], b[MASK]
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def run(s, rng, n_fam=60, n_targets=120, n_str=16, amp=1.0, prior_scale=1.0):
    acq, lab, zyg = people(s, n_fam, rng, amp, prior_scale)
    keys = [(f, t, "L", "S0", 0) for f in range(n_fam) for t in range(2)]
    total = np.sum([acq[k] for k in keys], axis=0)
    rows = []
    for i in range(n_targets):
        f, t = i % n_fam, (i // n_fam) % 2
        truth = acq[(f, t, "L", "S0", 0)]
        oracle = segment(truth)
        pop = (total - acq[(f, 0, "L", "S0", 0)] - acq[(f, 1, "L", "S0", 0)]) / (len(keys) - 2)
        others = [j for j in range(n_fam) if j != f]
        strangers = [acq[(int(j), int(rng.integers(2)), "L", "S1", 0)] for j in rng.choice(others, n_str, replace=False)]
        fills = {"same_sitting": acq[(f, t, "L", "S0", 1)], "other_day": acq[(f, t, "L", "S1", 0)], "other_side": acq[(f, t, "R", "S0", 0)],
                 zyg[f]: acq[(f, 1 - t, "L", "S0", 0)], "population_mean": pop, "blurred_population_mean": gaussian_filter(pop, 2.0),
                 "blurred_stranger": gaussian_filter(strangers[0], 2.0)}
        fills.update({f"stranger_{j}": im for j, im in enumerate(strangers)})
        row, ot = {}, topo(oracle)
        for k, im in fills.items():
            filled = np.where(MASK, im, truth)
            seg = segment(filled)
            row[k] = {"dice": dice(seg, oracle), "accuracy": float((seg == oracle)[MASK].mean()), **topo(seg), **panel(im, truth, MASK)}
            row[k].update({f"topo_err_{q}": abs(row[k][q] - ot[q]) for q in TOPO})
        row["_truth_topo"] = ot
        rows.append(row)
    lower = {"rmse", "mse", "mae", "msle"} | {f"topo_err_{q}" for q in TOPO}
    metrics = ("dice", "accuracy", "rmse", "mse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim") + tuple(f"topo_err_{q}" for q in TOPO)
    out = {}
    for m in metrics:
        per = {}
        for row in rows:
            ref = np.array([v[m] for k, v in row.items() if k.startswith("stranger")])
            for k, v in row.items():
                if k.startswith(("stranger", "_")):
                    continue
                x = v[m]
                c = np.mean(ref > x) + 0.5 * np.mean(ref == x) if m in lower else np.mean(ref < x) + 0.5 * np.mean(ref == x)
                per.setdefault(k, {"c": [], "raw": []})
                per[k]["c"].append(c)
                per[k]["raw"].append(x)
            per.setdefault("stranger", {"c": [0.5], "raw": []})["raw"].extend(ref.tolist())
        out[m] = {k: {"mean_c": float(np.mean(v["c"])), "mean_raw": float(np.mean(v["raw"]))} for k, v in per.items()}
    tr = [r["_truth_topo"] for r in rows]
    out["topology_abs_error_vs_oracle"] = {k: {q: float(np.mean([abs(r[k][q] - t[q]) for r, t in zip(rows, tr) if k in r])) for q in TOPO}
                                           for k in rows[0] if not k.startswith("_") and k != "stranger_1"}
    out["oracle_topology_mean"] = {q: float(np.mean([t[q] for t in tr])) for q in TOPO}
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "doc": __doc__.split("\n\n")[0]}
    configs = (("genetic_share_0.54", BASE, 1.0), ("genetic_share_0.19", LOW_H, 1.0),
               ("folding_genetic_share_0.54", BASE, 0.25), ("folding_genetic_share_0.19", LOW_H, 0.25))
    for name, s, ps in configs:
        res[name] = run(s, rng, prior_scale=ps)
    with open(os.path.join(PROJECT, "results", "label_ladder.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for name, _, _ in configs:
        print("##", name)
        for m in ("dice", "accuracy", "rmse", "ssim") + tuple(f"topo_err_{q}" for q in TOPO):
            print(" ", m, {k: round(v["mean_c"], 2) for k, v in res[name][m].items()})
        print("  topology |err|", {k: tuple(round(v[q], 1) for q in TOPO) for k, v in res[name]["topology_abs_error_vs_oracle"].items()},
              "oracle", res[name]["oracle_topology_mean"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
