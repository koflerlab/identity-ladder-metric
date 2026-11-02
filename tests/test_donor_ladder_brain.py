"""Geometry checks of the brain donor ladder (no scans needed)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_brain as B


def test_mirror_index_reflects_mm_and_is_an_involution():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[0, 3] = -98.0  # MNI 2009a 1 mm grid: x from -98 to +98 over 197 voxels
    mi = B.mirror_index(affine, 197)
    x = affine[0, 0] * np.arange(197) + affine[0, 3]
    assert np.allclose(x[mi], -x)
    assert np.array_equal(mi[mi], np.arange(197))


def test_cut_window_and_shell_sizes():
    vol = np.arange(40 ** 3, dtype=np.float32).reshape(40, 40, 40)
    win = B.cut(vol, (10, 10, 10), (16, 16, 16))
    assert win.shape == (16 + 2 * B.SHELL,) * 3
    assert np.array_equal(B.cut(vol, (10, 10, 10), (16, 16, 16), shell=False), vol[10:26, 10:26, 10:26])
