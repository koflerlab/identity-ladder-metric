"""Population-mean controls in synthetic and real visual examples.

Distances and residual projections are diagnostic coordinates, not a validated
identity metric. Shrinkage uses the known target deliberately, not a reconstruction
algorithm. Real case-level results must be written only to the private data root.
"""
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
import torch

from atlas_common import ROOT, METRICS, HIGHER as INTENSITY_HIGHER, score
import label_ladder as ll
import pupil_ladder as pl

LAMBDAS = (1., .75, .5, .25, 0.)
HIGHER = INTENSITY_HIGHER | {'dice', 'accuracy', 'pupil_dice'}


def serializable(obj):
    if isinstance(obj, dict): return {k: serializable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)): return [serializable(v) for v in obj]
    if isinstance(obj, np.ndarray): return serializable(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj): raise ValueError('Unexpected NaN in experiment')
        if np.isinf(obj): return 'Infinity' if obj > 0 else '-Infinity'
        return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    return obj


def save_json(path, obj):
    Path(path).write_text(json.dumps(serializable(obj), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def diagnostics(target, candidate, mean, mask):
    """Use one unchanging representation/ROI for all three distances."""
    if not np.any(mask): raise ValueError('Empty evaluation mask')
    a, b = (np.asarray(v, dtype=float)[mask] - mean[mask] for v in (target, candidate))
    aa, bb, ab = np.dot(a, a), np.dot(b, b), np.dot(a, b)
    out = {'target_distance': float(np.sqrt(np.mean((a-b)**2))),
           'mean_distance': float(np.sqrt(np.mean(b*b))),
           'target_mean_distance': float(np.sqrt(np.mean(a*a)))}
    if aa <= np.finfo(float).eps:
        out.update(residual_ratio=None, target_projection=None, residual_cosine=None,
                   orthogonal_ratio=None, low_information=True)
    else:
        out.update(residual_ratio=float(np.sqrt(bb/aa)), target_projection=float(ab/aa),
                   residual_cosine=float(ab/np.sqrt(aa*bb)) if bb > np.finfo(float).eps else None,
                   orthogonal_ratio=float(np.sqrt(max(bb-ab*ab/aa, 0)/aa)), low_information=False)
    return out


def wrong_residual(target, mean, mask, rng):
    """Matched residual norm and zero projection before nonnegativity clipping.

Clipping may alter both; diagnostics always use the actual resulting array.
"""
    a = (target-mean)[mask]
    noise = gaussian_filter(rng.normal(size=target.shape), 1.)[mask]
    aa = float(np.dot(a, a))
    if aa > 0: noise -= np.dot(noise, a)/aa*a
    noise *= np.sqrt(aa)/max(np.linalg.norm(noise), 1e-12)
    out = mean.copy(); out[mask] = np.maximum(0, mean[mask]+noise)
    return out


def tissue_extra(image, truth, mask):
    pred = ll.segment(np.where(mask, image, truth)); target = ll.segment(truth)
    pt, tt = ll.topo(pred), ll.topo(target)
    return {'dice': ll.dice(pred, target), 'accuracy': float(np.mean(pred[mask] == target[mask])),
            **{f'topo_err_{m}': abs(pt[m]-tt[m]) for m in ll.TOPO}}


def shrinkage(target, mean, lam):
    """Keep oracle/null endpoints bit-identical, avoiding cancellation roundoff."""
    if lam == 1: return target.copy()
    if lam == 0: return mean.copy()
    return mean+lam*(target-mean)


def evaluate(case):
    x, mu, mask = (case[k] for k in ('target', 'mean', 'mask'))
    extra = case.get('extra')
    def measures(y):
        return {**score(y, x, mask), **(extra(y, x, mask) if extra else {})}
    refs = [measures(y) for y in case['references']]
    mirror_refs = [measures(y) for y in case.get('mirror_references', case['references'])]
    images = {f'shrink_{lam:g}': shrinkage(x, mu, lam) for lam in LAMBDAS}
    images.update(case['comparators'])
    images['wrong_residual'] = wrong_residual(x, mu, mask, np.random.default_rng(812))
    values = {}
    for name, y in images.items():
        raw = measures(y)
        reference = mirror_refs if name == 'other_side' else refs
        c = {}
        for m, v in raw.items():
            r = np.asarray([z[m] for z in reference])
            c[m] = float(np.mean(r < v if m in HIGHER else r > v) + .5*np.mean(r == v))
        values[name] = {'raw': raw, 'c': c, 'mean_control': diagnostics(x, y, mu, mask), 'rank': {}}
    for m in next(iter(values.values()))['raw']:
        a = np.array([r['raw'][m] for r in values.values()])
        ranks = rankdata(-a if m in HIGHER else a, method='average')
        for row, rank in zip(values.values(), ranks): row['rank'][m] = float(rank)
    return {'scores': values, 'n_references': len(refs), 'mask_pixels': int(mask.sum())}, images


def synthetic_cases(kind, seed):
    rng = np.random.default_rng(seed)
    if kind == 'tissue':
        acq, _, zyg = ll.people(ll.BASE, 72, rng, prior_scale=.25)
        mu = np.mean([acq[(f, 0, 'L', 'S0', 0)] for f in range(8, 48)], axis=0)
        refs = [acq[(f, 0, 'L', 'S1', 0)] for f in range(48, 72)]
        for i in range(8):
            yield {'target': acq[(i, 0, 'L', 'S0', 0)], 'mean': mu, 'mask': ll.MASK,
                   'references': refs, 'extra': tissue_extra,
                   'comparators': {'repeat': acq[(i, 0, 'L', 'S0', 1)],
                                   'later': acq[(i, 0, 'L', 'S1', 0)],
                                   'other_side': acq[(i, 0, 'R', 'S0', 0)],
                                   'co_twin': acq[(i, 1, 'L', 'S0', 0)], 'stranger': refs[0],
                                   'blurred_stranger': gaussian_filter(refs[0], 2)},
                   'zygosity': zyg[i]}
    elif kind == 'eyes':
        targets, template, reference = pl.people(8, rng), pl.people(40, rng), pl.people(24, rng)
        mu = np.mean([pl.render(p[0], rng) for p in template], axis=0)
        refs = [pl.render(p[0], rng) for p in reference]
        for p in targets:
            yield {'target': pl.render(p[0], rng), 'mean': mu, 'mask': pl.MASK,
                   'references': refs, 'extra': lambda y, x, m: pl.shape_scores(np.where(m, y, x), x),
                   'comparators': {'repeat': pl.render(p[0], rng), 'dilated': pl.render(p[0], rng, 1.3),
                                   'other_side': pl.render(p[1], rng), 'stranger': refs[0],
                                   'blurred_stranger': gaussian_filter(refs[0], 2)}}
    else: raise ValueError(kind)


def summarize(rows):
    out = {}
    for fill in rows[0]['scores']:
        out[fill] = {}
        for group in ('raw', 'c', 'mean_control'):
            out[fill][group] = {}
            for metric in rows[0]['scores'][fill][group]:
                if metric == 'low_information': continue
                vals = [r['scores'][fill][group][metric] for r in rows]
                valid = [v for v in vals if v is not None]
                seeds = []
                for seed in (401, 402, 403):
                    vs = [r['scores'][fill][group][metric] for r in rows if r['seed'] == seed]
                    vs = [v for v in vs if v is not None]
                    if vs: seeds.append(float(np.mean(vs)))
                out[fill][group][metric] = {'mean': float(np.mean(valid)) if valid else None,
                                          'seed_min': min(seeds) if seeds else None,
                                          'seed_max': max(seeds) if seeds else None, 'n': len(valid)}
    return out


def synthetic():
    torch.set_num_threads(1)
    out = {'design': {'seeds': [401, 402, 403], 'targets_per_seed_per_modality': 8,
                      'template_people_or_families': 40, 'reference_people_or_families': 24,
                      'lambda': LAMBDAS, 'tissue_prior': .25,
                      'versions': {p: version(p) for p in ('numpy', 'scipy', 'torch', 'torchmetrics', 'inpainting')},
                      'source_sha256': {p: hashlib.sha256((ROOT/'src'/p).read_bytes()).hexdigest()
                                        for p in ('population_null.py', 'toy_ladder.py', 'label_ladder.py', 'pupil_ladder.py')}},
           'limits': ['Exploratory, 2D, synthetic parameters not biologically calibrated.',
                      'Seed ranges are min/max of three cohort means, not confidence intervals.',
                      'Shrinkage uses known truth; it is not a reconstruction method.',
                      'Residual diagnostics are not a validated combined identity score.',
                      'Infinity denotes exact-match PSNR; no clipping to a finite display value.'], 'modalities': {}}
    for kind in ('tissue', 'eyes'):
        rows = []
        for seed in (401, 402, 403):
            for i, case in enumerate(synthetic_cases(kind, seed)):
                result, images = evaluate(case)
                rows.append({'seed': seed, 'target_index': i, **result})
                if seed == 401 and i == 0:
                    np.savez_compressed(ROOT/f'results/null_example_{kind}.npz', target=case['target'],
                                        mean=case['mean'], mask=case['mask'], **images)
            print(f'{kind}: seed {seed}, 8 targets complete', flush=True)
        out['modalities'][kind] = {'n_independent_targets': 24, 'rows': rows, 'aggregate': summarize(rows)}
    save_json(ROOT/'results/population_null_synthetic.json', out)
    print('Population-null simulations complete: 24 tissue + 24 eye targets', flush=True)


if __name__ == '__main__': synthetic()
