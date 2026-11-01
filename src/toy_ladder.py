"""Synthetic identity ladder with known variance components: closed forms for the propositions of the ISBI 2027 draft
and simulations that check them.

Generative model of one acquisition of side s of person p in sitting k (time t_k, device d_k), repeat r:
    x = g_p + e_p + b_{p,s} + a_p * t_k + eta_{p,k} + n_{p,k,r} * [d_k = B] + eps_{p,k,r}
with independent zero-mean Gaussian components per feature dimension: g genotype (shared fully by MZ co-twins, with
correlation 1/2 by DZ co-twins), e individual development, b side-specific development, a ageing slope, eta sitting
nuisance (shared by images of one sitting), n device-B nuisance (fresh per acquisition), eps measurement noise.
The difference of two acquisitions is Gaussian with per-dimension variance v_r set by the components they do not share,
so squared distances are v_r * chi2_D and the calibrated score against a reference with variance v_ref is
    E c = P(F_{D,D} > rho),   rho = v_r / v_ref   (Proposition 1).

Part A simulates feature vectors; part B builds 2D random-field 'tissue' and scores inpainting fills with the official
BraTS 2023 metric code. Only synthetic data. Output: results/toy_ladder.json.
"""
import json
import os
import sys

import numpy as np
import torch
from inpainting.challenge_metrics_2023 import generate_metrics
from scipy.ndimage import gaussian_filter
from scipy.special import betainc

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 0
# sittings: S0 baseline on device A, S1 another day on A, S2 another sitting on device B, S3 one time unit later on A
# (time is measured from each person's baseline, t = 0; a device change always starts a new sitting)
SITTINGS = {"S0": (0.0, "A"), "S1": (0.0, "A"), "S2": (0.0, "B"), "S3": (1.0, "A")}
BASE = dict(g=1.0, e=0.6, b=0.5, a=0.5, eta=0.3, n=0.6, eps=0.4)  # standard deviations, t in units of the follow-up
LOW_H = dict(BASE, g=0.6, e=1.0)  # genetic share below one half


def psi(rho, D):
    """P(F_{D,D} > rho) = I_{1/(1+rho)}(D/2, D/2): expected calibrated score of a rung with variance ratio rho."""
    return betainc(D / 2, D / 2, 1.0 / (1.0 + np.asarray(rho, dtype=float)))


def rung_variances(s: dict, dt: float = 1.0) -> dict:
    """Per-dimension variance of the difference of two acquisitions, by rung (Proposition 2 uses these)."""
    g, e, b, a, eta, n, eps = (s[k] ** 2 for k in ("g", "e", "b", "a", "eta", "n", "eps"))
    person = 2 * (g + e + b)
    return {
        "L0_same_sitting": 2 * eps,
        "L0_retest": 2 * (eps + eta),
        "L1_device": 2 * (eps + eta) + n,
        "L2_years": 2 * (eps + eta) + a * dt ** 2,
        "mirror_same_sitting": 2 * (b + eps),
        "MZ": 2 * (e + b + eps + eta),
        "DZ": g + 2 * (e + b + eps + eta),
        "L4": person + 2 * (eps + eta),
        "L4_device_AB": person + 2 * (eps + eta) + n,
        "L4_years": person + 2 * (eps + eta) + a * dt ** 2,
        "population_mean_fill": (person + 2 * (eps + eta)) / 2,
    }


def matched_reference(rung: str) -> str:
    return {"L1_device": "L4_device_AB", "L2_years": "L4_years"}.get(rung, "L4")


def predictions(s: dict, D: int) -> dict:
    v = rung_variances(s)
    out = {}
    for r in v:
        if r.startswith("L4"):
            continue
        rho = v[r] / v[matched_reference(r)]
        out[r] = {"rho": rho, "c": float(psi(rho, D))}
    out["L1_device_pooled_reference"] = {"rho": v["L1_device"] / v["L4"], "c": float(psi(v["L1_device"] / v["L4"], D))}
    vp = v["L4"] / 2
    out["identities"] = {"icc_retest": 1 - out["L0_retest"]["rho"], "falconer_h2": 2 * (out["DZ"]["rho"] - out["MZ"]["rho"]),
                         "true_genetic_share": s["g"] ** 2 / vp, "mirror_rho_le_mz_rho": out["mirror_same_sitting"]["rho"] <= out["MZ"]["rho"]}
    return out


# ---------------------------------------------------------------- part A: feature vectors
def simulate_vectors(s: dict, D: int, n_fam: int, rng: np.random.Generator) -> dict:
    """Acquisitions keyed (family, twin, side, sitting, repeat) -> D-vector; half the families MZ, half DZ."""
    X, zyg = {}, {}
    for f in range(n_fam):
        mz = f % 2 == 0
        zyg[f] = "MZ" if mz else "DZ"
        shared, own = rng.normal(0, s["g"], D), [rng.normal(0, s["g"], D) for _ in range(2)]
        for t in range(2):
            g = shared if mz else np.sqrt(0.5) * shared + np.sqrt(0.5) * own[t]
            e, a = rng.normal(0, s["e"], D), rng.normal(0, s["a"], D)
            etas = {k: rng.normal(0, s["eta"], D) for k in SITTINGS}  # one sitting covers both sides
            for side in "LR":
                b = rng.normal(0, s["b"], D)
                for k, (time, dev) in SITTINGS.items():
                    eta = etas[k]
                    for r in range(2):
                        n = rng.normal(0, s["n"], D) if dev == "B" else 0.0
                        X[(f, t, side, k, r)] = g + e + b + a * time + eta + n + rng.normal(0, s["eps"], D)
    return {"X": X, "zygosity": zyg, "n_fam": n_fam}


def rung_pairs(sim: dict, rng: np.random.Generator, n_ref: int = 20000) -> dict:
    X, n_fam = sim["X"], sim["n_fam"]
    pairs = {k: [] for k in ("L0_same_sitting", "L0_retest", "L1_device", "L2_years", "mirror_same_sitting", "MZ", "DZ")}
    for f in range(n_fam):
        for t in range(2):
            for side in "LR":
                base = X[(f, t, side, "S0", 0)]
                pairs["L0_same_sitting"].append((base, X[(f, t, side, "S0", 1)]))
                pairs["L0_retest"].append((base, X[(f, t, side, "S1", 0)]))
                pairs["L1_device"].append((base, X[(f, t, side, "S2", 0)]))
                pairs["L2_years"].append((base, X[(f, t, side, "S3", 0)]))
            pairs["mirror_same_sitting"].append((X[(f, t, "L", "S0", 0)], X[(f, t, "R", "S0", 0)]))
        pairs[sim["zygosity"][f]].append((X[(f, 0, "L", "S0", 0)], X[(f, 1, "L", "S0", 0)]))
    refs = {"L4": [], "L4_device_AB": [], "L4_years": []}
    for _ in range(n_ref):
        f1, f2 = rng.choice(n_fam, 2, replace=False)
        t1, t2 = rng.integers(2, size=2)
        refs["L4"].append((X[(f1, t1, "L", "S0", 0)], X[(f2, t2, "L", "S0", 0)]))
        refs["L4_device_AB"].append((X[(f1, t1, "L", "S0", 0)], X[(f2, t2, "L", "S2", 0)]))
        refs["L4_years"].append((X[(f1, t1, "L", "S0", 0)], X[(f2, t2, "L", "S3", 0)]))
    d = lambda ps: np.array([np.linalg.norm(a - b) for a, b in ps])
    return {k: d(v) for k, v in {**pairs, **refs}.items()}


def mean_c(d: np.ndarray, ref: np.ndarray) -> float:
    ref = np.sort(ref)
    above = len(ref) - np.searchsorted(ref, d, side="right")
    ties = np.searchsorted(ref, d, side="right") - np.searchsorted(ref, d, side="left")
    return float(np.mean((above + 0.5 * ties) / len(ref)))


def check_vectors(s: dict, D: int, n_fam: int, rng) -> dict:
    dist = rung_pairs(simulate_vectors(s, D, n_fam, rng), rng)
    pred = predictions(s, D)
    out = {}
    for r in ("L0_same_sitting", "L0_retest", "L1_device", "L2_years", "mirror_same_sitting", "MZ", "DZ"):
        ref = dist[matched_reference(r)]
        out[r] = {"c_sim": mean_c(dist[r], ref), "c_theory": pred[r]["c"],
                  "rho_sim": float(np.mean(dist[r] ** 2) / np.mean(ref ** 2)), "rho_theory": pred[r]["rho"], "n_pairs": int(len(dist[r]))}
    out["L1_device_pooled_reference"] = {"c_sim": mean_c(dist["L1_device"], dist["L4"]), "c_theory": pred["L1_device_pooled_reference"]["c"]}
    ref2 = dist["L4"] ** 2
    out["effective_D_from_strangers"] = float(2 * ref2.mean() ** 2 / ref2.var())
    out["falconer_h2_sim"] = 2 * (out["DZ"]["rho_sim"] - out["MZ"]["rho_sim"])
    out["identities_theory"] = pred["identities"]
    return out


def negative_control(s: dict, D: int, n_people: int, rng) -> dict:
    """Nuisance-only feature z = eta + eps: c within one sitting and across sittings (Proposition 4)."""
    eta = rng.normal(0, s["eta"], (n_people, 2, D))
    z = eta[:, :, None, :] + rng.normal(0, s["eps"], (n_people, 2, 2, D))  # person, sitting, repeat
    same = np.linalg.norm(z[:, 0, 0] - z[:, 0, 1], axis=1)
    across = np.linalg.norm(z[:, 0, 0] - z[:, 1, 0], axis=1)
    i, j = rng.integers(n_people, size=(2, 20000))
    keep = i != j
    ref = np.linalg.norm(z[i[keep], 0, 0] - z[j[keep], 1, 0], axis=1)
    rho = s["eps"] ** 2 / (s["eps"] ** 2 + s["eta"] ** 2)
    return {"same_sitting": {"c_sim": mean_c(same, ref), "c_theory": float(psi(rho, D))},
            "across_sittings": {"c_sim": mean_c(across, ref), "c_theory": 0.5}}


# ---------------------------------------------------------------- part B: 2D tissue and the BraTS panel
H = W = 64


def field(rng, sd, scale):
    """Smooth Gaussian random field with unit variance before scaling."""
    f = gaussian_filter(rng.normal(size=(H, W)), scale)
    return sd * f / f.std()


def template_image() -> np.ndarray:
    """Flat tissue (100) with a bright ring (160) outside both masks: the ring sets the intensity range, as bright
    structures do in MRI, while the masked region is homogeneous, so the challenge's percentile clipping leaves the
    masked target unclipped."""
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - 31.5, xx - 31.5)
    return 100.0 + 60.0 * ((r > 25) & (r < 30))


def tissue_people(s: dict, n_fam: int, rng, contrast: float = 1.0):
    """2D acquisitions: template plus identity fields (spatial scale 2-3 px), a sitting offset and noise. 'contrast'
    scales every person-level component together, so variance ratios and the genetic share do not change with it."""
    template = template_image()
    acq, zyg = {}, {}
    for f in range(n_fam):
        mz = f % 2 == 0
        zyg[f] = "MZ" if mz else "DZ"
        shared, own = field(rng, s["g"], 3), [field(rng, s["g"], 3) for _ in range(2)]
        for t in range(2):
            g = shared if mz else np.sqrt(0.5) * shared + np.sqrt(0.5) * own[t]
            e = field(rng, s["e"], 2)
            etas = {k: rng.normal(0, s["eta"]) for k in ("S0", "S1")}  # sitting nuisance: a global intensity offset
            for side in "LR":
                b = field(rng, s["b"], 2)
                for k in ("S0", "S1"):
                    eta = etas[k]
                    for r in range(2):
                        person = g + e + b + eta + rng.normal(0, s["eps"], (H, W))
                        acq[(f, t, side, k, r)] = template + contrast * 10 * person
    return {"acq": acq, "zygosity": zyg, "template": template, "n_fam": n_fam}


def panel(fill: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    t = torch.tensor(truth, dtype=torch.float32)[None, None]
    m = torch.tensor(mask)[None, None]
    p = torch.tensor(np.where(mask, fill, truth), dtype=torch.float32)[None, None]
    return generate_metrics(prediction=p, target=t, normalization_tensor=t * ~m, mask=m)


def donor_swap(people: dict, mask: np.ndarray, rng, n_targets: int, n_strangers: int = 16) -> dict:
    acq, zyg, n_fam = people["acq"], people["zygosity"], people["n_fam"]
    keys = [(f, t, "L", "S0", 0) for f in range(n_fam) for t in range(2)]
    total = np.sum([acq[k] for k in keys], axis=0)
    rows, clipped = [], []
    for i in range(n_targets):
        f, t = i % n_fam, (i // n_fam) % 2
        truth = acq[(f, t, "L", "S0", 0)]
        # population mean of the other families: neither the target nor its co-twin contributes
        pop_mean = (total - acq[(f, 0, "L", "S0", 0)] - acq[(f, 1, "L", "S0", 0)]) / (len(keys) - 2)
        fills = {"L0_same_sitting": acq[(f, t, "L", "S0", 1)], "L0_retest": acq[(f, t, "L", "S1", 0)],
                 "mirror_same_sitting": acq[(f, t, "R", "S0", 0)], zyg[f]: acq[(f, 1 - t, "L", "S0", 0)],
                 "population_mean": pop_mean}
        pm = pop_mean[mask]  # same template, its own mean kept; only its spread restored to the target's in-mask SD
        fills["population_mean_sd_restored"] = np.where(mask, pm.mean() + (pop_mean - pm.mean()) / (pm.std() + 1e-9) * truth[mask].std(), pop_mean)
        voided = truth * ~mask  # as in the challenge: percentiles of the voided image set the [0, 1] scaling
        v_max = np.percentile(voided, 99.5)
        clipped.append(float(np.mean(truth[mask] > v_max)))
        others = [g for g in range(n_fam) if g != f]
        for j in rng.choice(others, n_strangers, replace=False):
            fills[f"stranger_{len([k for k in fills if k.startswith('stranger')])}"] = acq[(int(j), int(rng.integers(2)), "L", "S1", 0)]
        scores = {k: panel(v, truth, mask) for k, v in fills.items()}
        rows.append(scores)
    metrics = ("rmse", "mse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim")
    higher = {"psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim"}
    out = {"n_targets": n_targets, "mask_pixels": int(mask.sum()), "share_masked_target_clipped": float(np.mean(clipped)),
           "n_population_mean": len(keys) - 2}
    # rho-hat (Proposition 3): per case, MSE of a fill over the mean MSE of that case's stranger fills, averaged over cases
    rho = {}
    for row in rows:
        ref = np.mean([row[k]["mse"] for k in row if k.startswith("stranger")])
        for k, v in row.items():
            if not k.startswith("stranger"):
                rho.setdefault(k, []).append(v["mse"] / ref)
    out["rho_hat_mse"] = {k: float(np.mean(v)) for k, v in rho.items()}
    for m in metrics:
        per = {}
        for row in rows:
            ref = np.array([row[k][m] for k in row if k.startswith("stranger")])
            for k, v in row.items():
                if k.startswith("stranger"):
                    continue
                x = v[m]
                c = np.mean(ref < x) + 0.5 * np.mean(ref == x) if m in higher else np.mean(ref > x) + 0.5 * np.mean(ref == x)
                per.setdefault(k, {"c": [], "raw": []})
                per[k]["c"].append(c)
                per[k]["raw"].append(x)
            per.setdefault("stranger", {"c": [], "raw": []})["raw"].extend(ref.tolist())
        out[m] = {k: {"mean_c": float(np.mean(v["c"])) if v["c"] else 0.5, "median_raw": float(np.median(v["raw"]))} for k, v in per.items()}
    return out


def tissue_rho_predictions(s: dict, n_pop: int) -> dict:
    """Expected per-pixel squared error of each fill over that of a stranger's tissue (2V), Proposition 5."""
    g, e, b, eta, eps = (s[k] ** 2 for k in ("g", "e", "b", "eta", "eps"))
    V = g + e + b + eta + eps
    return {"L0_same_sitting": eps / V, "L0_retest": (eps + eta) / V, "mirror_same_sitting": (b + eps) / V,
            "MZ": (e + b + eps + eta) / V, "DZ": (g / 2 + e + b + eps + eta) / V,
            "population_mean": 0.5 * (1 + 1 / n_pop), "population_mean_sd_restored": 1.0}


def ssim_constant_fill_identity(rng) -> dict:
    """Proposition 6: for a constant fill y = c, the SSIM map equals l(x, c) * C2 / (sigma_x^2 + C2), window-wise."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    x = 0.5 + field(rng, 0.05, 3)
    y = np.full_like(x, x.mean())
    _, smap = ssim(torch.tensor(y)[None, None].float(), torch.tensor(x)[None, None].float(), data_range=1.0, return_full_image=True)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = lambda a: gaussian_filter(a, 1.5, truncate=3.5, mode="reflect")
    mu, var = k(x), k(x * x) - k(x) ** 2
    pred = (2 * mu * x.mean() + C1) / (mu ** 2 + x.mean() ** 2 + C1) * C2 / (var + C2)
    inner = (slice(8, -8), slice(8, -8))
    return {"max_abs_diff_interior": float(np.abs(smap.numpy()[0, 0][inner] - pred[inner]).max()),
            "ssim_constant_fill_mean": float(smap.numpy()[0, 0][inner].mean())}


def falconer_check(s: dict, D: int, n_fam: int, rng, n_boot: int = 400) -> dict:
    """Proposition 2 on many families: 2 (rho_DZ - rho_MZ) against the true genetic share; the interval resamples MZ
    pairs, DZ pairs and reference pairs independently (reference pairs share families, so its term is understated)."""
    sim = simulate_vectors(s, D, n_fam, rng)
    X, zyg = sim["X"], sim["zygosity"]
    d2 = {"MZ": [], "DZ": []}
    for f in range(n_fam):
        d2[zyg[f]].append(np.sum((X[(f, 0, "L", "S0", 0)] - X[(f, 1, "L", "S0", 0)]) ** 2))
    i, j = rng.integers(n_fam, size=(2, 40000))
    keep = i != j
    ref = np.array([np.sum((X[(a, 0, "L", "S0", 0)] - X[(b, 1, "L", "S0", 0)]) ** 2) for a, b in zip(i[keep], j[keep])])
    mz, dz = np.array(d2["MZ"]), np.array(d2["DZ"])
    est = lambda m, z, r: 2 * (z.mean() - m.mean()) / r.mean()
    boot = [est(rng.choice(mz, len(mz)), rng.choice(dz, len(dz)), rng.choice(ref, len(ref))) for _ in range(n_boot)]
    return {"n_families": n_fam, "D": D, "h2_estimate": float(est(mz, dz, ref)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "true_genetic_share": s["g"] ** 2 / (rung_variances(s)["L4"] / 2)}


def main():
    rng = np.random.default_rng(SEED)
    from importlib.metadata import version
    out = {"model": __doc__.split("\n\n")[1].strip(), "params_sd": {"base": BASE, "low_genetic_share": LOW_H}, "seed": SEED,
           "versions": {p: version(p) for p in ("numpy", "scipy", "torch", "torchmetrics", "inpainting")}}
    out["vectors"] = {f"{name}_D{D}": check_vectors(s, D, 300, rng) for name, s in (("base", BASE), ("low_h", LOW_H)) for D in (8, 32)}
    out["negative_control_D8"] = negative_control(BASE, 8, 3000, rng)
    out["ssim_constant_fill"] = ssim_constant_fill_identity(rng)
    big = np.zeros((H, W), bool); big[16:48, 16:48] = True
    small = np.zeros((H, W), bool); small[26:38, 26:38] = True
    out["donor_swap"] = {}
    for name, s in (("base", BASE), ("low_h", LOW_H)):
        for contrast in (1.0, 0.1):
            people = tissue_people(s, 60, rng, contrast)
            for mname, mk in (("mask32", big), ("mask12", small)):
                res = donor_swap(people, mk, rng, 120)
                res["rho_predicted"] = tissue_rho_predictions(s, res["n_population_mean"])
                out["donor_swap"][f"{name}_contrast{contrast:g}_{mname}"] = res
    out["theory"] = {name: {"rung_variances": rung_variances(s), "pred_D8": predictions(s, 8), "pred_D32": predictions(s, 32)} for name, s in (("base", BASE), ("low_h", LOW_H))}
    rng2 = np.random.default_rng(SEED + 1)  # separate stream, so the results above do not change
    out["falconer_large"] = {f"{name}_D{D}": falconer_check(s, D, 2000, rng2) for name, s in (("base", BASE), ("low_h", LOW_H)) for D in (8, 32)}
    with open(os.path.join(PROJECT, "results", "toy_ladder.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k, v in out["vectors"].items():
        print(k, {r: (round(x["c_sim"], 3), round(x["c_theory"], 3)) for r, x in v.items() if isinstance(x, dict) and "c_sim" in x})
    print("negative control", out["negative_control_D8"], "| ssim identity", out["ssim_constant_fill"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
