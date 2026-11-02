"""Protect split fidelity, category aggregation and endpoint denominators."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import care_reproduction as cr


def test_official_venous_selection_and_full_coverage():
    split = pd.DataFrame({"bdmap_id": ["fixture1_V", "fixture1_A", "fixture2_V"]})
    ids = cr.selected_ids(split)
    assert ids == {"fixture1", "fixture2"}
    frame = pd.DataFrame({"case_name": ["fixture2", "unused", "fixture1"], "value": [2, 3, 1]})
    assert cr.select_frame(frame, ids)["value"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="cover"):
        cr.select_frame(frame.iloc[:2], ids)
    with pytest.raises(ValueError, match="Duplicate"):
        cr.select_frame(pd.concat([frame, frame.iloc[:1]]), ids)


def test_categories_average_within_case_before_cohort_summary():
    index = pd.Index(["fixture1", "fixture2"])
    pixel = pd.DataFrame({"ssim_3d": [80, 90], "psnr_3d": [20, 30]}, index=index)
    seg = pd.DataFrame({c: [10., 20.] for cols in cr.GROUPS.values() for c in cols}, index=index)
    seg.loc["fixture1", "large_nsd_12"] = np.nan
    seg.loc["fixture2", "large_nsd_12"] = 100
    values = cr.endpoint_values(pixel, seg)
    assert values["large_nsd"].tolist() == [10, 36]
    assert cr.summarize(values["large_nsd"])["median"] == 23
    assert values["ssim"].tolist() == [80, 90]  # no accidental second percent conversion


def test_missing_denominators_and_infinite_rejection():
    stats = cr.summarize([1, np.nan, 3])
    assert stats == {"median": 2, "q1": 1.5, "q3": 2.5, "n_finite": 2, "n_missing": 1}
    with pytest.raises(ValueError, match="no finite"):
        cr.summarize([np.nan])
    with pytest.raises(ValueError, match="Infinite"):
        cr.summarize([1, np.inf])


def test_posthoc_all_phase_selection_preserves_original_rule():
    split = pd.DataFrame({"bdmap_id": ["fixture1_V", "fixture2_A"]})
    assert cr.selected_ids(split) == {"fixture1"}
    assert cr.selected_ids(split, "paper_all_phases_posthoc") == {"fixture1", "fixture2"}
    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        cr.selected_ids(pd.DataFrame({"bdmap_id": ["fixture1_V", "fixture1_A"]}), "paper_all_phases_posthoc")


def test_coverage_failure_reports_counts_without_case_keys(monkeypatch):
    monkeypatch.setattr(cr, "METHODS", ("fixture_method",))
    data = {cr.csv_path("fixture_method", care, kind): b"case_name,value\nfixture_case,1\n"
            for care in (False, True) for kind in ("pixel", "seg")}
    rows = cr.coverage_diagnostics(data, {"fixture_case", "absent_case"})
    assert len(rows) == 4
    assert all(r["n_present"] == 1 and r["n_missing"] == 1 for r in rows)
    assert "fixture_case" not in str(rows) and "absent_case" not in str(rows)
