"""Actual reproduction failure modes: provenance, partial coverage and PDF reference."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import care_iteration2 as it
import care_reproduction as cr


def test_tampered_bytes_and_inventory_are_rejected_before_aggregation():
    frozen = {"inputs": {"source": {"sha256": cr.digest(b"original")}}}
    cr.validate_inputs({"source": b"original"}, frozen)
    with pytest.raises(ValueError, match="Source changed"):
        cr.validate_inputs({"source": b"modified"}, frozen)
    for data in ({}, {"source": b"original", "unexpected": b"extra"}):
        with pytest.raises(ValueError, match="inventory"):
            cr.validate_inputs(data, frozen)


def fixture_frames():
    pixel = pd.DataFrame({"case_name": ["fixture_b", "fixture_a"], "ssim_3d": [90, 80], "psnr_3d": [30, 20]})
    seg = pd.DataFrame({"case_name": ["fixture_a"], **{c: [50.] for cols in cr.GROUPS.values() for c in cols}})
    return pixel, seg


def test_missing_segmentation_never_blocks_complete_pixel_endpoints():
    pixel, seg = fixture_frames()
    values, issues = it.endpoint_arrays(pixel, seg, {"fixture_a", "fixture_b"})
    assert values["ssim"].tolist() == [80, 90]
    assert values["psnr"].tolist() == [20, 30]
    assert set(issues) == set(cr.GROUPS)
    assert all(s == "blocked_coverage" for s in issues.values())
    assert not set(values).intersection(cr.GROUPS)  # the one available case is not substituted


def test_missing_pixel_does_not_block_complete_segmentation():
    pixel, seg = fixture_frames()
    seg = pd.concat([seg, seg.assign(case_name="fixture_b")], ignore_index=True)
    values, issues = it.endpoint_arrays(pixel.iloc[:1], seg, {"fixture_a", "fixture_b"})
    assert set(issues) == {"ssim", "psnr"}
    assert set(values) == set(cr.GROUPS)
    assert all(len(v) == 2 for v in values.values())


def test_table_extraction_ignores_neighboring_numbers_and_requires_108_triplets():
    def pdf(n):
        with pymupdf.open() as doc:
            for _ in range(9):
                doc.new_page(width=1200, height=2000)
            triplet = "1.0 (0.0, 2.0) "
            text = "Other table 9.0 (8.0, 10.0)\nTable 2:\n" + "\n".join(triplet * min(6, n-i) for i in range(0, n, 6))
            doc[8].insert_text((30, 30), text + "\nThese results\n7.0 (6.0, 8.0)", fontsize=10)
            return doc.tobytes()
    table = cr.paper_table(pdf(108))
    assert table.shape == (9, 2, 6, 3)
    np.testing.assert_array_equal(table[8, 1, 5], [1, 0, 2])
    with pytest.raises(ValueError, match="108"):
        cr.paper_table(pdf(107))
