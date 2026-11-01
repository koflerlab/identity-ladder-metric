"""Synthetic checks of the BraTS 2023 inpainting metric code (inpainting package, challenge_metrics_2023).

Settles three implementation questions behind hypotheses H11, H14 and H15 of the research plan (Section 9.1):
1. SSIM dimensionality. A [1, X, Y, Z] volume enters torchmetrics SSIM, which documents B x C x H x W input. If SSIM is
   computed per 2D slice with the first spatial axis taken as channels, permuting that axis identically in every input
   leaves the score unchanged, while the same permutation along the last axis (inside the 2D window) changes it.
2. PSNR clamping. Plain PSNR receives the reference's in-mask (min, max) as a tuple data range. If torchmetrics clamps
   the input to a tuple range, an overshoot beyond the reference's range is forgiven by PSNR but not by RMSE.
3. Per-case order. Within one case PSNR_eps, PSNR_01 and PSNR_01_eps are monotone in MSE, so they rank fills exactly
   as RMSE does; plain PSNR may not, through its clamping.
Only synthetic arrays are used. Output: results/brats_panel_checks_tm<torchmetrics version>.json.
"""
import json
import os
import sys
from importlib.metadata import version

import numpy as np
import torch
import torchmetrics
from inpainting.challenge_metrics_2023 import _normalize_with_percentiles, generate_metrics
from scipy.ndimage import gaussian_filter
from scipy.stats import kendalltau

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SHAPE = (40, 48, 36)
SEED = 0


def synthetic_case(rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth positive 'tissue' with a darker cuboid region to be inpainted; returns target, mask, voided target."""
    tissue = 150.0 + 30.0 * gaussian_filter(rng.normal(size=SHAPE), 2.0) / 0.1
    mask = np.zeros(SHAPE, dtype=bool)
    mask[12:28, 14:34, 10:26] = True
    tissue[mask] -= 60.0  # in-mask range sits below the surrounding tissue, so an overshoot can stay inside [0, 1]
    target = torch.tensor(tissue, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(mask).unsqueeze(0)
    return target, m, target * ~m


def fills(target: torch.Tensor, mask: torch.Tensor, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Fills of known kind: noise at several levels, blur, a constant mean, a gain change and overshoots."""
    t = target.numpy()[0]
    inside = mask.numpy()[0]
    out = {}
    for s in (2.0, 5.0, 10.0, 20.0):
        out[f"noise_sd{s:g}"] = t + inside * rng.normal(0, s, t.shape)
    for s in (1.0, 2.0, 4.0):
        out[f"blur_sigma{s:g}"] = np.where(inside, gaussian_filter(t, s), t)
    out["constant_mask_mean"] = np.where(inside, t[inside].mean(), t)
    out["gain_1.1"] = np.where(inside, 1.1 * t, t)
    hi = t[inside].max()
    for k, frac in (("overshoot_5pct", 0.05), ("overshoot_20pct", 0.20)):
        f = t.copy()
        sel = inside & (rng.random(t.shape) < frac)
        f[sel] = hi + 25.0  # above the reference's in-mask maximum, still below the surrounding tissue
        out[k] = f
    return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in out.items()}


def ssim_dimensionality(target, mask, voided, fill, rng) -> dict:
    base = generate_metrics(prediction=fill, target=target, normalization_tensor=voided, mask=mask)["ssim"]
    res = {"ssim_original": base}
    for name, axis in (("first_spatial_axis", 1), ("last_spatial_axis", 3)):
        perm = torch.tensor(rng.permutation(target.shape[axis]))
        p = [x.index_select(axis, perm) for x in (fill, target, voided, mask)]
        res[f"ssim_after_permuting_{name}"] = generate_metrics(prediction=p[0], target=p[1], normalization_tensor=p[2], mask=p[3])["ssim"]
    res["per_slice_2d"] = bool(abs(res["ssim_after_permuting_first_spatial_axis"] - base) < 1e-6
                               and abs(res["ssim_after_permuting_last_spatial_axis"] - base) > 1e-4)
    return res


def psnr_clamping(target, mask, voided, fill) -> dict:
    out = generate_metrics(prediction=fill, target=target, normalization_tensor=voided, mask=mask)
    t = _normalize_with_percentiles((target * mask).float(), normalization_tensor=voided)[mask].double()
    p = _normalize_with_percentiles((fill * mask).float(), normalization_tensor=voided)[mask].double()
    lo, hi = t.min(), t.max()
    r2 = (hi - lo) ** 2
    mse = torch.mean((p - t) ** 2)
    mse_clamped = torch.mean((p.clamp(lo, hi) - t) ** 2)
    res = {"psnr_reported": out["psnr"], "psnr_eps_reported": out["psnr_eps"],
           "psnr_from_unclamped_mse": float(10 * torch.log10(r2 / mse)),
           "psnr_from_clamped_mse": float(10 * torch.log10(r2 / mse_clamped)),
           "share_of_fill_voxels_above_reference_max": float((p > hi).double().mean())}
    res["plain_psnr_clamps_fill"] = bool(abs(res["psnr_reported"] - res["psnr_from_clamped_mse"]) < 1e-3
                                         and abs(res["psnr_reported"] - res["psnr_from_unclamped_mse"]) > 1e-2)
    return res


def per_case_order(target, mask, voided, fs: dict) -> dict:
    scores = {k: generate_metrics(prediction=v, target=target, normalization_tensor=voided, mask=mask) for k, v in fs.items()}
    names = list(fs)
    rmse = np.array([scores[k]["rmse"] for k in names])
    agree = {}
    for m in ("mse", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "mae", "msle", "ssim"):
        v = np.array([scores[k][m] for k in names])
        better_high = m.startswith("psnr") or m == "ssim"
        tau = kendalltau(-rmse, v if better_high else -v).statistic  # tau = 1: same order of fills as RMSE
        agree[m] = float(tau)
    return {"fills": names, "kendall_tau_with_rmse_order": agree,
            "scores": {k: {m: float(x) for m, x in s.items()} for k, s in scores.items()}}


def main(out_path: str | None = None):
    rng = np.random.default_rng(SEED)
    target, mask, voided = synthetic_case(rng)
    fs = fills(target, mask, rng)
    res = {"torchmetrics": torchmetrics.__version__, "inpainting": version("inpainting"), "torch": torch.__version__,
           "shape": [1, *SHAPE], "seed": SEED,
           "ssim_dimensionality": ssim_dimensionality(target, mask, voided, fs["noise_sd5"], rng),
           "psnr_clamping": psnr_clamping(target, mask, voided, fs["overshoot_20pct"]),
           "per_case_order": per_case_order(target, mask, voided, fs)}
    out_path = out_path or os.path.join(PROJECT, "results", f"brats_panel_checks_tm{torchmetrics.__version__}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("torchmetrics", "ssim_dimensionality", "psnr_clamping")}, indent=1))
    print("kendall tau vs RMSE order:", res["per_case_order"]["kendall_tau_with_rmse_order"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else None)
