import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ladder_faces as lf  # noqa: E402


def test_sibling_shares_half_the_genotype():
    rng = np.random.default_rng(0)
    pairs = [(p["g"], lf.sibling_genome(p, rng)["g"]) for p in (lf.draw_genome(rng) for _ in range(4000))]
    a = np.concatenate([x for x, _ in pairs])
    b = np.concatenate([y for _, y in pairs])
    assert abs(np.corrcoef(a, b)[0, 1] - 0.5) < 0.02
    assert abs(b.std() - 1) < 0.02


def test_median_draw_keeps_the_median_candidate():
    top = np.zeros((100, 100, 3))
    levels = iter([0.5, 0.1, 0.3, 0.9, 0.7])
    chosen = lf.median_draw(lambda r: np.full((100, 100, 3), next(levels)), top, np.random.default_rng(0), k=5)
    assert np.allclose(chosen, 0.5)


def test_rmse_ignores_pixels_outside_the_box():
    a = np.zeros((100, 100, 3))
    b = a.copy()
    b[:20] = 1.0  # above the box
    assert lf.rmse(a, b) == 0.0
    b[50, 50] = 1.0  # inside the box
    assert lf.rmse(a, b) > 0.0
