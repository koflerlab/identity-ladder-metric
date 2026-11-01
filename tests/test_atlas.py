"""Score aggregation and no-target-leakage controls for atlas helpers."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import atlas_common as ac
import baseline_v0 as bv


def test_atlas_half_ties_and_direction():
    refs=[{'rmse':1.,'ssim':1.},{'rmse':2.,'ssim':2.},{'rmse':4.,'ssim':4.}]
    c=ac.calibrate({'rmse':1.,'ssim':1.},refs)
    assert np.isclose(c['rmse'],5/6)
    assert np.isclose(c['ssim'],1/6)


def test_aggregate_uses_available_targets_per_fill():
    rows=[{'scores':{'repeat':{'c':{'rmse':1.}},'mean':{'c':{'rmse':.6}}}},
          {'scores':{'repeat':{'c':{'rmse':.8}}}}]
    out=ac.aggregate(rows)
    assert out['repeat']['rmse']=={'mean_c':.9,'n_targets':2}
    assert out['mean']['rmse']=={'mean_c':.6,'n_targets':1}


def test_iris_placement_does_not_fit_hidden_target_pixels():
    rng=np.random.default_rng(9)
    target=rng.normal(100,10,(16,128));donor=rng.normal(90,12,(16,128))
    mask=np.zeros_like(target,dtype=bool);mask[4:12,50:82]=True
    changed=target.copy();changed[mask]+=1000
    for mode in ('sd','raw','affine'):
        a=bv.place(target,donor,mask,mode=mode)
        b=bv.place(changed,donor,mask,mode=mode)
        np.testing.assert_array_equal(a[mask],b[mask])
        np.testing.assert_array_equal(a[~mask],target[~mask])
