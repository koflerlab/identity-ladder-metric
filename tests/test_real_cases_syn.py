import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import real_cases_syn as syn


def test_physical_reflection_corrects_half_open_parietal_bounds():
    affine = np.eye(4)
    affine[0, 3] = -50
    v = np.broadcast_to(np.arange(-50, 51)[:, None, None], (101, 2, 2))
    left = syn.rc.box_mask(v.shape, affine, ((-45, -13), (0, 2), (0, 2)))
    assert np.array_equal(syn.reflect(v, affine)[left], -v[left])
    assert set(syn.reflect(v, affine)[left]) == set(range(14, 46))
    affine[0, 3] = -49
    with pytest.raises(ValueError, match="centred"):
        syn.reflect(v, affine)


def test_scales_exclude_box_and_preserve_background():
    v = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
    brain = np.ones(v.shape, bool)
    brain[0] = False
    box = np.zeros(v.shape, bool)
    box[2, 2] = True
    out, meta = syn.scaled(v, brain, box, "p90")
    assert meta["p90"] == np.percentile(v[brain & ~box], 90)
    assert not out[~brain].any()
    out, _ = syn.scaled(v, brain, box, "z")
    assert np.isclose(out[brain & ~box].mean(), 10)
    assert np.isclose(out[brain & ~box].std(), 1)
    with pytest.raises(ValueError, match="invalid intensity"):
        syn.scaled(np.ones(v.shape), brain, box, "z")
