"""One real cortical example at 1 mm; public-data derivatives stay local.

Affine-to-template preparation preserves individual folding rather than matching
donors to the target. All file/participant mappings and registration logs are
private. This is a descriptive image comparison, not an inpainting algorithm.
"""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '2')
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import image_ladder as il
from atlas_common import ROOT, PRIVATE as ATLAS

DATA = Path('C:/Projects/data')
PRIVATE = DATA / 'projects/identity-ladder-metric/run-2026-11-01-detail'
VERSION = 'affine-native-intensity-v1'
ROI_CONFIG = {'case_index': 0, 'slice_mni_z_mm': 55, 'roi_mni_x_mm': [-45, -13],
              'roi_mni_y_mm': [-65, -33], 'context_padding_mm': 16,
              'selection': 'Target anatomy inspected before alternative scores; posterior cortical folding, fixed first atlas case.'}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def ensure_store():
    """Serialize directory/registry changes, leaving existing raw data unchanged."""
    lock = DATA / '_registry/storage-maintenance.lock'
    with lock.open('x', encoding='utf-8') as fh:
        fh.write('identity-ladder detail experiment; directory and registry checkpoint\n')
    try:
        PRIVATE.mkdir(exist_ok=True)
        reg = DATA / '_registry/datasets.json'
        obj = json.loads(reg.read_text(encoding='utf-8-sig'))
        key = 'identity-ladder-visual-detail-20261101'
        if not any(e.get('id') == key for e in obj['datasets']):
            obj['datasets'].append({
                'id': key, 'canonical_paths': [str(PRIVATE.relative_to(DATA)).replace('\\', '/')],
                'content_status': 'derived_only', 'access': 'Local research derivatives; source restrictions retained.',
                'origin': 'User-requested detailed experimental comparisons, 2026-11-01.',
                'source': ['QTIM / OpenNeuro ds004169 v1.0.8 (local dataset_description.json: CC0)',
                           'UTIRIS V.1 (public download; redistribution not granted)'],
                'notes': 'Image/participant mappings, case-level scores and figures remain local. '
                         'Code and aggregate experimental evidence: identity-ladder-metric repository. '
                         'No new cohort or acquisition; reuses the 2026-11-01 visual-atlas pools.'})
            reg.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        assert PRIVATE.is_dir()
        assert any(e.get('id') == key for e in json.loads(reg.read_text(encoding='utf-8'))['datasets'])
    finally:
        lock.unlink()


def inputs():
    ledger = json.loads((ATLAS / 'real_mri/case_ledger.json').read_text())
    case = ledger['cases'][0]
    m = pd.read_csv(Path(il.PRIVATE) / 'manifest.csv').fillna('').set_index('scan')
    roles = [('target', case['target_scan']), ('later', case['later_scan'])]
    roles += [(f'reference_{i:02d}', s) for i, s in enumerate(ledger['reference_scans'])]
    roles += [(f'template_{i:02d}', s) for i, s in enumerate(ledger['template_scans'])]
    out = []
    for role, scan in roles:
        row = m.loc[scan]
        out.append(dict(role=role, scan=scan, path=row.path, person=row.person, family=row.family))
    target_family = out[0]['family']
    assert all(r['family'] != target_family for r in out[2:])
    assert not ({r['person'] for r in out if r['role'].startswith('reference')} &
                {r['person'] for r in out if r['role'].startswith('template')})
    return out


def exclusion(ni):
    """Broad bilateral cortical area excluded before choosing a final patch."""
    x, y, z = np.meshgrid(*(np.arange(s) for s in ni.shape), indexing='ij', sparse=True)
    a = ni.affine
    assert np.allclose(a[:3, :3], np.diag(np.diag(a[:3, :3])))
    xx, yy, zz = a[0, 0]*x+a[0, 3], a[1, 1]*y+a[1, 3], a[2, 2]*z+a[2, 3]
    return (np.abs(xx) >= 15) & (np.abs(xx) <= 70) & (yy >= -85) & (yy <= 0) & (zz >= 15) & (zz <= 75)


def prepare(limit=None):
    import ants
    import nibabel as nib
    from nilearn import datasets
    ensure_store()
    roi_path = PRIVATE / 'roi_config.json'
    if roi_path.exists():
        if json.loads(roi_path.read_text()) != ROI_CONFIG:
            raise ValueError('Existing ROI differs; preserve the old experiment before changing it')
    else:
        write(roi_path, ROI_CONFIG)
    rows = inputs()
    write(PRIVATE / 'input_ledger.json', {'version': VERSION, 'inputs': rows})
    for res in (1, 2):
        path = PRIVATE / f'template_{res}mm.nii.gz'
        if not path.exists(): nib.save(datasets.load_mni152_template(resolution=res), path)
        path = PRIVATE / f'brain_{res}mm.nii.gz'
        if not path.exists(): nib.save(datasets.load_mni152_brain_mask(resolution=res), path)
    tpl2 = ants.image_read(str(PRIVATE / 'template_2mm.nii.gz'))
    tpl1 = ants.image_read(str(PRIVATE / 'template_1mm.nii.gz'))
    ni2 = nib.load(PRIVATE / 'template_2mm.nii.gz')
    brain2 = nib.load(PRIVATE / 'brain_2mm.nii.gz').get_fdata() > 0
    reg_mask = tpl2.new_image_like((brain2 & ~exclusion(ni2)).astype(np.float32))
    for row in rows[:limit]:
        role = row['role']
        dest = PRIVATE / f'{role}_1mm.nii.gz'
        meta = PRIVATE / f'{role}_preparation.json'
        digest = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        if dest.exists() and meta.exists():
            old = json.loads(meta.read_text())
            if old['version'] != VERSION or old['source_sha256'] != digest:
                raise RuntimeError('Existing preparation has different provenance')
            print(f'{role}: cached', flush=True)
            continue
        start = time.monotonic()
        native = ants.image_read(row['path'])
        small = ants.resample_image(native, (2, 2, 2), use_voxels=False, interp_type=0)
        bias = ants.n4_bias_field_correction(small, shrink_factor=4, return_bias_field=True)
        corrected2 = small / bias
        with np.load(Path(il.PRIVATE) / 'features' / (row['scan'] + '.npz'), allow_pickle=False) as f:
            setting = int(f['strip_setting'])
        stripped, _ = il.strip_skull(corrected2.numpy(), **il.STRIP_SETTINGS[setting])
        reg = ants.registration(fixed=tpl2, moving=corrected2.new_image_like(stripped),
                                type_of_transform='Affine', mask=reg_mask, mask_all_stages=True,
                                aff_iterations=(1000, 500, 250, 50), random_seed=123,
                                outprefix=str(PRIVATE / (role + '_')), verbose=False)
        native_bias = ants.resample_image_to_target(bias, native, interp_type='linear')
        corrected_native = native.new_image_like(native.numpy() / np.maximum(native_bias.numpy(), 1e-6))
        warped = ants.apply_transforms(tpl1, corrected_native, reg['fwdtransforms'], interpolator='linear')
        ants.image_write(warped, str(dest))
        warped2 = reg['warpedmovout'].numpy()
        corr = float(np.corrcoef(warped2[brain2], tpl2.numpy()[brain2])[0, 1])
        source_ni = nib.load(row['path'])
        write(meta, {'version': VERSION, 'source_sha256': digest,
                     'source_spacing_mm': [float(x) for x in source_ni.header.get_zooms()[:3]],
                     'grid_spacing_mm': 1, 'transform': 'Affine', 'strip_setting': setting,
                     'template_correlation': corr, 'seconds': time.monotonic()-start,
                     'ants_version': ants.__version__, 'registration_seed': 123,
                     'registration_exclusion': '|x| in [15,70], y in [-85,0], z in [15,75] mm'})
        print(f'{role}: prepared; template correlation={corr:.3f}; {time.monotonic()-start:.1f}s', flush=True)
    return rows


def load_normalized(role):
    """Fit scale on visible template tissue outside the whole candidate ROI."""
    import nibabel as nib
    ni = nib.load(PRIVATE / 'template_1mm.nii.gz')
    brain = nib.load(PRIVATE / 'brain_1mm.nii.gz').get_fdata() > 0
    a = nib.load(PRIVATE / f'{role}_1mm.nii.gz').get_fdata(dtype=np.float32)
    fit = brain & ~exclusion(ni) & np.isfinite(a) & (a > 0)
    lo, hi = np.percentile(a[fit], [1, 99])
    return np.clip((a-lo)/(hi-lo), 0, 1), ni, brain


def anatomy():
    """Target only: choose anatomy before calculating any alternative's score."""
    a, ni, _ = load_normalized('target')
    zs = (25, 35, 45, 55, 65)
    fig, axes = plt.subplots(1, len(zs), figsize=(17, 4.5))
    for ax, mm in zip(axes, zs):
        z = round((mm-ni.affine[2, 3])/ni.affine[2, 2])
        ax.imshow(a[:, :, z].T, origin='lower', cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'MNI z = {mm} mm'); ax.set(xticks=[], yticks=[])
    fig.suptitle('Real target only: continuous T1 intensity, 1-mm sampling', fontsize=16)
    fig.text(.5, .02, 'Affine to independent template; no thresholding. Slice selection uses anatomy, not donor scores.', ha='center')
    fig.tight_layout(rect=(0, .07, 1, .91)); fig.savefig(PRIVATE / 'target_anatomy.png', dpi=160); plt.close(fig)
    print('Target anatomy contact sheet saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'anatomy'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        if args.action == 'prepare': prepare(args.limit)
        else: anatomy()
    except Exception as exc:
        if PRIVATE.exists(): (PRIVATE / 'error_local.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise RuntimeError(f'Detail preparation failed ({type(exc).__name__}); details retained locally') from None
