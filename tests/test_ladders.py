"""Tests for the synthetic ladders behind H19: closed form of toy_ladder and topology descriptors of label_ladder.

The topology tests pin the descriptor definitions (holes = components - Euler number in 2D) and that topology errors
are calibrated as lower-is-better metrics, so the folding-anatomy result cannot silently flip sign.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import toy_ladder as tl  # noqa: E402


def test_closed_form_is_one_half_at_rho_one():
    for D in (8, 32):
        assert tl.psi(np.array([1.0]), D)[0] == pytest.approx(0.5, abs=1e-9)


def test_closed_form_decreases_in_rho():
    rho = np.linspace(0.05, 2.0, 50)
    assert np.all(np.diff(tl.psi(rho, 16)) < 0)


def test_population_mean_fill_has_rho_one_half():
    v = tl.rung_variances(tl.BASE)
    assert v["population_mean_fill"] / v["L4"] == pytest.approx(0.5)


@pytest.fixture(scope="module")
def ll():
    pytest.importorskip("inpainting")
    import label_ladder
    return label_ladder


def _mask_square(ll, inner):
    seg = np.zeros(ll.MASK.shape, bool)
    seg[20:44, 20:44] = True
    if inner:
        seg[28:36, 28:36] = False
    return seg


def test_topo_counts_one_component_and_one_hole(ll):
    t = ll.topo(_mask_square(ll, inner=True))
    assert (t["components"], t["holes"], t["euler"]) == (1, 1, 0)
    assert t["perimeter"] > ll.topo(_mask_square(ll, inner=False))["perimeter"]


def test_topo_of_filled_square_has_no_hole(ll):
    t = ll.topo(_mask_square(ll, inner=False))
    assert (t["components"], t["holes"], t["euler"]) == (1, 0, 1)


def test_prior_scale_one_keeps_original_anatomy(ll):
    a = ll.people(tl.BASE, 2, np.random.default_rng(0))[1]
    b = ll.people(tl.BASE, 2, np.random.default_rng(0), prior_scale=1.0)[1]
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_topology_errors_are_calibrated_lower_is_better(ll):
    out = ll.run(tl.BASE, np.random.default_rng(1), n_fam=6, n_targets=4, n_str=4)
    for q in ll.TOPO:
        m = out[f"topo_err_{q}"]
        assert m["stranger"]["mean_c"] == 0.5
        # a repeat in the same sitting reproduces the oracle topology better than strangers do
        assert m["same_sitting"]["mean_c"] >= 0.5, (q, m["same_sitting"])
