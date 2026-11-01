"""Tests for the image-space ladder helpers on synthetic volumes and toy manifests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import image_ladder as il  # noqa: E402


def test_block_mean_on_mni_grid_shape():
    v = np.arange(99 * 117 * 95, dtype=float).reshape(99, 117, 95)
    b = il.block_mean(v)
    assert b.shape == (33, 39, 31)
    assert b[0, 0, 0] == pytest.approx(v[:3, :3, :3].mean())


def test_mirror_halves_of_a_symmetric_volume_are_identical():
    # A volume symmetric about the central x index of the 2 mm grid stays symmetric after 3-voxel blocks,
    # so the left half equals the reflected right half exactly; an asymmetric bump breaks the equality.
    rng = np.random.default_rng(0)
    half = rng.normal(size=(49, 6, 5))
    centre = rng.normal(size=(1, 6, 5))
    sym = np.concatenate([half, centre, half[::-1]], axis=0)  # 99 columns, mirror about index 49
    left, right = il.mirror_halves(il.block_mean(sym))
    assert left.shape == right.shape == (16, 2, 1)
    np.testing.assert_allclose(left, right)
    sym[10] += 5.0
    left, right = il.mirror_halves(il.block_mean(sym))
    assert np.abs(left - right).max() > 1.0


def test_mirror_halves_rejects_even_grid():
    with pytest.raises(ValueError):
        il.mirror_halves(np.zeros((4, 2, 2)))


def _row(**kw):
    base = {"dataset": "QTIM", "person": "a", "family": "f1", "session": "ses-01", "scanner": "QTIM"}
    base.update(kw)
    return pd.Series(base)


@pytest.mark.parametrize("a,b,rung", [
    (dict(dataset="SIMON", person="s", session="ses-001", scanner="x"),
     dict(dataset="SIMON", person="s", session="ses-001", scanner="x"), "L0_same_session"),
    (dict(dataset="SIMON", person="s", session="ses-001", scanner="x"),
     dict(dataset="SIMON", person="s", session="ses-002", scanner="x"), "L1L2_same_scanner_same_protocol"),
    (dict(dataset="SIMON", person="s", session="ses-001", scanner="x"),
     dict(dataset="SIMON", person="s", session="ses-002", scanner="y"), "L1L2_other_scanner_same_protocol"),
    (dict(), dict(session="ses-02"), "L2_same_person_later_visit"),
    (dict(), dict(person="b"), "L3_cotwin_zygosity_unknown"),
    (dict(), dict(person="b", family="f2"), "L4_unrelated"),
    (dict(dataset="IXI", person="p", family="p", scanner="Guys"),
     dict(dataset="IXI", person="q", family="q", scanner="Guys"), "L4_same_centre"),
    (dict(dataset="IXI", person="p", family="p", scanner="Guys"),
     dict(dataset="IXI", person="q", family="q", scanner="HH"), "L4_other_centre"),
    (dict(), dict(dataset="IXI", person="q", family="q"), "L4_cross_dataset"),
])
def test_pair_rung(a, b, rung):
    assert il.pair_rung(_row(**a), _row(**b)) == rung


def test_pair_table_distances_match_brute_force():
    m = pd.DataFrame([_row(person=p, family=p).to_dict() | {"years": y} for p, y in [("a", 12), ("b", 13), ("c", 14)]])
    X = np.random.default_rng(1).normal(size=(3, 7))
    p = il.pair_table(m, X)
    for r in p.itertuples():
        assert r.distance == pytest.approx(np.linalg.norm(X[r.i] - X[r.j]))
    assert set(p.rung) == {"L4_unrelated"}


def test_auc_lower_and_ties():
    assert il.auc_lower([0, 1], [2, 3]) == 1.0
    assert il.auc_lower([1], [1]) == 0.5


def test_qtim_identification_finds_own_first_visit():
    rows = [dict(dataset="QTIM", person=p, session=s) for p in "abc" for s in ("ses-01", "ses-02")]
    m = pd.DataFrame(rows)
    base = {"a": 0.0, "b": 10.0, "c": 20.0}
    X = np.array([[base[r["person"]] + (0.5 if r["session"] == "ses-02" else 0)] for r in rows])
    out = il.qtim_identification(m, X)
    assert out["queries"] == 3 and out["rank1"] == 1.0 and out["gallery_size"] == 3


def test_protocol_key_separates_sequences_and_keeps_missing():
    a = il.protocol_key({"RepetitionTime": 2.3, "InversionTime": 0.9, "FlipAngle": 9, "SliceThickness": 1.0})
    b = il.protocol_key({"RepetitionTime": 7.0, "InversionTime": 2.921, "FlipAngle": 90, "SliceThickness": 1.3})
    assert a != b and a == "2.3|0.9|na|9.0|1.0"
    assert il.protocol_key({}) == "na|na|na|na|na"
    # the same sequence written in ms (one SIMON sidecar does this) maps to the same key
    ms = il.protocol_key({"RepetitionTime": 2300, "InversionTime": 900, "FlipAngle": 9, "SliceThickness": 1.0})
    assert ms == a


def test_pair_rung_splits_cross_session_pairs_by_device_and_protocol():
    a = _row(dataset="SIMON", person="s", session="ses-001", scanner="x", device="SN1", protocol="p")
    b = _row(dataset="SIMON", person="s", session="ses-002", scanner="x-renamed", device="SN1", protocol="q")
    assert il.pair_rung(a, b) == "L1L2_same_scanner_other_protocol"


def test_dprime_is_signed():
    assert il.dprime(np.array([0.0, 1.0]), np.array([2.0, 3.0])) > 0 > il.dprime(np.array([2.0, 3.0]), np.array([0.0, 1.0]))


def test_pair_rung_splits_within_session_sequences():
    a = _row(dataset="SIMON", person="s", session="ses-001", scanner="x", protocol="2.3|0.9|9.0|1.0")
    b = _row(dataset="SIMON", person="s", session="ses-001", scanner="x", protocol="7.0|2.921|90.0|1.3")
    assert il.pair_rung(a, b) == "L0x_same_session_other_sequence"
    assert il.pair_rung(a, a.copy()) == "L0_same_session"


def test_strip_skull_separates_brain_from_shell():
    # bright 'brain' ball inside a dark gap and a bright scalp shell: erosion must cut the shell away
    z, y, x = np.indices((60, 60, 60)) - 30
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    noise = np.random.default_rng(0).uniform(1, 5, r.shape)  # MRI background is low noise, not exact zeros
    a = noise + np.where(r < 18, 100.0, 0.0) + np.where((r > 22) & (r < 26), 100.0, 0.0)
    _, mask = il.strip_skull(a, erode=3)
    assert mask[r < 15].all()
    assert not mask[(r > 23) & (r < 25)].any()


def test_mirror_test_uses_other_scans_of_the_same_person():
    rng = np.random.default_rng(0)
    mask = np.ones((5, 2, 2), dtype=bool)
    people = {p: rng.normal(size=(5, 2, 2)) for p in "abc"}
    rows, vols = [], []
    for p in "abc":
        for s in range(2):
            rows.append({"dataset": "QTIM", "person": p})
            vols.append(people[p] + rng.normal(0, 0.05, size=(5, 2, 2)))
    out = il.mirror_test(pd.DataFrame(rows), vols, mask)["QTIM"]
    assert out["n_same_person_scan_pairs"] == 6
    assert out["auc_same_half_same_person_below_other_person"] > 0.9


def test_qc_outliers_applies_absolute_floor():
    qc = pd.DataFrame({"scan": list("abcde"), "qc_template_corr": [0.7, 0.71, 0.69, 0.45, 0.7]})
    assert il.qc_outliers(qc, floor=0.5) == ["d"]
