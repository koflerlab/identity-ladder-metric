"""Checks of direction, twin-subset pairing and export semantics."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import label_robustness as lr
import ladder_report as report


def test_calibration_uses_half_ties_and_correct_direction():
    row = {"population_mean": {"rmse": 2, "dice": 2},
           "stranger_0": {"rmse": 1, "dice": 1}, "stranger_1": {"rmse": 2, "dice": 2},
           "stranger_2": {"rmse": 4, "dice": 4}, "stranger_3": {"rmse": 5, "dice": 5}}
    assert lr.calibrated(row, "population_mean", "rmse") == .625
    assert lr.calibrated(row, "population_mean", "dice") == .375


def test_twin_comparison_does_not_mix_zygosity_subsets():
    rows = []
    for z, f, pop, twin in (("MZ", 0, 0, 2), ("DZ", 1, 3, 0)):
        rows.append({"_zygosity": z, "_family": f,
                     "population_mean": {m: pop for m in lr.METRICS}, z: {m: twin for m in lr.METRICS},
                     "stranger_0": {m: 1 for m in lr.METRICS}})
    result = lr.paired_twin_contrasts(rows)
    assert result["MZ"]["metrics"]["rmse"]["delta_mean_c"] == 1
    assert result["DZ"]["metrics"]["rmse"]["delta_mean_c"] == -1
    assert result["MZ"]["metrics"]["dice"]["delta_mean_c"] == -1
    assert result["MZ"]["n_families"] == result["MZ"]["n_targets"] == 1


def test_export_keeps_missing_rho_and_raw_mean_vs_median_distinct():
    toy = {"donor_swap": {"toy1": {"rmse": {"population_mean": {"mean_c": .8, "median_raw": 3}},
                                  "rho_hat_mse": {"population_mean": .51}, "rho_predicted": {"population_mean": .5}}}}
    label = {"seed": 2, "label1": {"rmse": {"population_mean": {"mean_c": .9, "mean_raw": 4}}}}
    a, b = list(report.table_rows(toy, label))
    assert a["rho_hat_mse"] == pytest.approx(.51)
    assert b["rho_hat_mse"] == b["rho_predicted"] == ""
    assert (a["raw_statistic"], b["raw_statistic"]) == ("median_raw", "mean_raw")
