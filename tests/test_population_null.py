from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from population_null import diagnostics, wrong_residual, serializable, shrinkage


def test_controlled_shrinkage_distances_and_projection():
    rng = np.random.default_rng(8)
    mu = np.full((20, 20), 100.)
    x = mu+rng.normal(size=mu.shape)
    mask = np.ones(mu.shape, bool)
    for lam in (0, .25, .5, .75, 1):
        d = diagnostics(x, mu+lam*(x-mu), mu, mask)
        assert d['residual_ratio'] == pytest.approx(lam)
        assert d['target_projection'] == pytest.approx(lam)
        assert d['target_distance']/d['target_mean_distance'] == pytest.approx(1-lam)
        assert d['residual_cosine'] == pytest.approx(1) if lam else d['residual_cosine'] is None


def test_wrong_texture_can_have_same_mean_distance_but_no_target_projection():
    rng = np.random.default_rng(9)
    mu = np.full((32, 32), 100.)
    x = mu+rng.normal(size=mu.shape)
    mask = np.ones(mu.shape, bool)
    y = wrong_residual(x, mu, mask, rng)
    d = diagnostics(x, y, mu, mask)
    assert d['residual_ratio'] == pytest.approx(1)
    assert d['target_projection'] == pytest.approx(0, abs=1e-12)
    assert d['orthogonal_ratio'] == pytest.approx(1)
    assert d['target_distance']/d['target_mean_distance'] == pytest.approx(np.sqrt(2))


def test_population_typical_target_is_low_information_not_failure():
    x = np.ones((8, 8)); m = np.ones_like(x, bool)
    d = diagnostics(x, x.copy(), x.copy(), m)
    assert d['low_information'] and d['target_projection'] is None
    assert d['target_distance'] == 0


def test_hidden_only_distances_and_explicit_infinite_psnr():
    x = np.ones((10, 10)); mask = np.zeros_like(x, bool); mask[2:6, 2:6] = True
    mu = x*.8; y = x.copy(); y[~mask] = 999
    assert diagnostics(x, y, mu, mask)['target_distance'] == 0
    assert serializable({'psnr': float('inf')}) == {'psnr': 'Infinity'}
    with pytest.raises(ValueError): serializable(float('nan'))


def test_oracle_and_mean_endpoints_are_exact_in_float32():
    rng = np.random.default_rng(12)
    x = rng.random((32, 32), dtype=np.float32)
    mu = rng.random(x.shape, dtype=np.float32)
    np.testing.assert_array_equal(shrinkage(x, mu, 1), x)
    np.testing.assert_array_equal(shrinkage(x, mu, 0), mu)
