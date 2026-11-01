"""Compatibility test for the BraTS inpainting metric code: the behaviours that hypotheses H11, H14 and H15 rely on.

If an update of the inpainting package or of torchmetrics changes SSIM dimensionality or PSNR clamping, these fail.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("inpainting")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import brats_panel_checks as bp  # noqa: E402


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(bp.SEED)
    target, mask, voided = bp.synthetic_case(rng)
    return target, mask, voided, bp.fills(target, mask, rng), rng


def test_ssim_is_computed_per_2d_slice(case):
    target, mask, voided, fs, rng = case
    res = bp.ssim_dimensionality(target, mask, voided, fs["noise_sd5"], rng)
    assert res["per_slice_2d"], res


def test_plain_psnr_clamps_the_fill_to_the_reference_range(case):
    target, mask, voided, fs, _ = case
    res = bp.psnr_clamping(target, mask, voided, fs["overshoot_20pct"])
    assert res["plain_psnr_clamps_fill"], res
    assert res["psnr_reported"] > res["psnr_eps_reported"]


def test_mse_family_ranks_fills_as_rmse_does(case):
    target, mask, voided, fs, _ = case
    tau = bp.per_case_order(target, mask, voided, fs)["kendall_tau_with_rmse_order"]
    for m in ("mse", "psnr_eps", "psnr_01", "psnr_01_eps"):
        assert tau[m] == pytest.approx(1.0), (m, tau[m])
    assert tau["psnr"] < 1.0  # clamping changes the order once fills overshoot
