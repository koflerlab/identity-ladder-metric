"""Readable experimental figures: fixed crops, error maps, full metric tables.

All figures containing real data and their case-level tables stay in PRIVATE.
Synthetic cohort summaries and figures are safe to version in the repository.
"""
import argparse
import json
from pathlib import Path
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

import atlas_common as ac
import baseline_v0 as bv
import real_case_atlas as ra
import real_mri_detail as md
import population_null as pn

LABELS = {'shrink_1': 'Original', 'shrink_0.75': '75% residual', 'shrink_0.5': '50% residual',
          'shrink_0.25': '25% residual', 'shrink_0': 'Population mean', 'repeat': 'Repeat capture',
          'later': 'Later visit', 'other_side': 'Other hemisphere', 'stranger': 'Stranger tissue',
          'blurred_stranger': 'Blurred stranger', 'wrong_residual': 'Wrong detail',
          'co_twin': 'Co-twin', 'dilated': 'Dilated pupil', 'raw_mean': 'Raw mean'}
METRIC_LABELS = {'rmse': 'RMSE ↓', 'mse': 'MSE ↓', 'mae': 'MAE ↓', 'msle': 'MSLE ↓',
                 'psnr': 'PSNR ↑', 'psnr_eps': 'PSNR eps ↑', 'psnr_01': 'PSNR 01 ↑',
                 'psnr_01_eps': 'PSNR 01 eps ↑', 'ssim': 'SSIM ↑'}


def savefig(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def mri_case():
    cfg = json.loads((md.PRIVATE/'roi_config.json').read_text())
    target, ni, brain = md.load_normalized('target')
    z = round((cfg['slice_mni_z_mm']-ni.affine[2, 3])/ni.affine[2, 2])
    x0, x1 = [round((v-ni.affine[0, 3])/ni.affine[0, 0]) for v in cfg['roi_mni_x_mm']]
    y0, y1 = [round((v-ni.affine[1, 3])/ni.affine[1, 1]) for v in cfg['roi_mni_y_mm']]
    rows = json.loads((md.PRIVATE/'input_ledger.json').read_text())['inputs']
    slices = {}
    for row in rows:
        volume, _, _ = md.load_normalized(row['role'])
        slices[row['role']] = volume[:, :, z].T
    x = slices['target']; mask = np.zeros_like(x, bool); mask[y0:y1, x0:x1] = True
    assert brain[x0:x1, y0:y1, z].all()
    refs = [slices[r['role']] for r in rows if r['role'].startswith('reference')]
    families = {}
    for r in rows:
        if r['role'].startswith('template'):
            families.setdefault(r['family'], []).append(slices[r['role']])
    mu = np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
    return {'target': x, 'mean': mu, 'mask': mask, 'references': refs,
            'mirror_references': [r[:, ::-1] for r in refs],
            'comparators': {'later': slices['later'], 'other_side': x[:, ::-1], 'stranger': refs[0],
                            'blurred_stranger': gaussian_filter(refs[0], 2)},
            'roi': (y0, y1, x0, x1), 'context': (y0-16, y1+16, x0-16, x1+16), 'slices': slices}


def eye_case():
    ledger = json.loads((ac.PRIVATE/'real_eyes/case_ledger.json').read_text())
    case = ledger['cases'][0]
    target_item = ra.eye_item(case['sources']['L'][0]); x = target_item['strip']
    repeat_item = ra.eye_item(case['sources']['L'][1])
    other_item = ra.eye_item(case['sources']['R'][0], True)
    raw_refs = [ra.eye_item(p[0])['strip'] for p in ledger['reference_paths']]
    right_refs = [ra.eye_item(p[1], True)['strip'] for p in ledger['reference_paths']]
    mu = np.mean([ra.eye_item(p)['strip'] for p in ledger['template_paths']], axis=0)
    mask = np.zeros_like(x, bool); mask[4:12, 50:82] = True
    refs = [bv.place(x, r, mask, mode='sd') for r in raw_refs]
    fitted_mu = bv.place(x, mu, mask, mode='affine')
    return {'target': x, 'mean': fitted_mu, 'mask': mask, 'references': refs,
            'mirror_references': [bv.place(x, r, mask, mode='sd') for r in right_refs],
            'comparators': {'repeat': bv.place(x, repeat_item['strip'], mask, mode='sd'),
                            'other_side': bv.place(x, other_item['strip'], mask, mode='sd'),
                            'stranger': refs[0], 'raw_mean': bv.place(x, mu, mask, mode='raw'),
                            'blurred_stranger': bv.place(x, gaussian_filter(raw_refs[0], 2), mask, mode='sd')},
            'roi': (4, 12, 50, 82), 'context': (0, 16, 34, 98),
            'native_items': [target_item, repeat_item, other_item]}


def crop(a, bounds):
    y0, y1, x0, x1 = bounds
    return a[y0:y1, x0:x1]


def detailed_card(case, result, images, kind):
    brain = kind == 'mri'
    second = 'later' if brain else 'repeat'
    order = ['shrink_1', second, 'other_side', 'stranger', 'shrink_0', 'blurred_stranger']
    x, mask, roi = case['target'], case['mask'], case['roi']
    context = case['context']; vmin, vmax = (0, 1) if brain else np.percentile(x[mask], [1, 99])
    error_range = .4 if brain else 80
    fig, axes = plt.subplots(3, 6, figsize=(16, 11 if brain else 9.3), gridspec_kw={'height_ratios': [1, 1, .85]})
    for j, name in enumerate(order):
        im = np.where(mask, images[name], x)
        for r, bounds in enumerate((context, roi)):
            axes[r, j].imshow(crop(im, bounds), origin='lower' if brain else 'upper',
                              cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal' if brain else 'auto')
        y0, y1, x0, x1 = roi
        axes[0, j].add_patch(Rectangle((x0-context[2]-.5, y0-context[0]-.5), x1-x0, y1-y0,
                                       fill=False, ec='#e58c21', lw=1.5))
        diff = crop(images[name]-x, roi)
        last = axes[2, j].imshow(diff, origin='lower' if brain else 'upper', cmap='RdBu_r',
                                 vmin=-error_range, vmax=error_range, interpolation='nearest', aspect='equal' if brain else 'auto')
        title = LABELS[name]
        if not brain and name == 'other_side': title = 'Other eye (flipped)'
        if not brain and name == 'shrink_0': title = 'Mean: visible-tissue fit'
        axes[0, j].set_title(title, fontsize=12)
        s = result['scores'][name]
        axes[2, j].set_xlabel(f"RMSE {s['raw']['rmse']:.3f} | SSIM {s['raw']['ssim']:.3f}\n"
                             f"Distance to mean / original: {s['mean_control']['residual_ratio']:.2f}", fontsize=9)
        for ax in axes[:, j]: ax.set(xticks=[], yticks=[])
    for ax, text in zip(axes[:, 0], ['Tissue + context', 'Evaluated patch, enlarged', 'Candidate minus original']):
        ax.set_ylabel(text, fontsize=11)
    fig.suptitle('Real brain: individual folds, same location, continuous T1 intensities' if brain else
                 'Real eye: one iris, the same patch and alternative donor fills', fontsize=18, y=.975)
    fig.subplots_adjust(left=.045, right=.96, top=.77 if brain else .91, bottom=.14 if brain else .16,
                        hspace=.10, wspace=.07)
    if brain:
        location = fig.add_axes([.045, .795, .14, .15])
        location.imshow(x, cmap='gray', vmin=0, vmax=1, origin='lower')
        location.contour(mask, [.5], colors=['#e58c21'], linewidths=1.8)
        location.contour(mask[:, ::-1], [.5], colors=['#1ba5c7'], linewidths=1.8)
        location.set(xticks=[], yticks=[])
        fig.text(.22, .895, 'Orange: evaluated 32 x 32 mm cortical patch\n'
                 'Blue: corresponding position in the opposite hemisphere', fontsize=13, va='center')
        fig.text(.22, .835, 'Below: the same patch is replaced by each candidate.\n'
                 'Outside the orange box, the original image is unchanged.', fontsize=12, va='center')
    cax = fig.add_axes([.97, .25, .009, .17]); fig.colorbar(last, cax=cax)
    footer = ('QTIM; first fixed case; 1-mm affine template grid; 32 x 32 mm ROI. No threshold segmentation. '
              'One display window and one error scale for every column.\n'
              'Later visit is years apart. Mean: 8 people / 7 equally weighted families; references: 8 other families; target family excluded. '
              'Mirror is the actual opposite-side tissue reflected into this ROI.') if brain else (
              'UTIRIS NIR; first fixed case; estimated pupil/limbus coordinates, not ground truth. '
              'One target-patch display window for all columns (contrast enlarged; metrics unchanged).\n'
              'Iris-strip alignment and intensity fitting use visible pixels only. Mean and reference pools: 8 other people each. '
              'Local research illustration; no redistribution.')
    fig.text(.5, .04, footer, ha='center', fontsize=10)
    savefig(fig, md.PRIVATE/f'real_{kind}_detail.png')


def metric_table(result, kind):
    names = list(result['scores'])
    columns = [LABELS.get(n, n).replace('Population mean', 'Mean').replace(' ', '\n', 1) for n in names]
    metrics = list(ac.METRICS)
    rows, labels, colors = [], [], []
    for m in metrics:
        values = [result['scores'][n] for n in names]
        rows.append([f"{v['raw'][m]:.4g}\nc={v['c'][m]:.2f}; r={v['rank'][m]:g}" for v in values])
        labels.append(METRIC_LABELS[m])
        colors.append([plt.cm.YlGnBu(.08+.60*(1-(v['rank'][m]-1)/max(len(names)-1, 1))) for v in values])
    for m, label in [('target_distance', 'd(target, candidate) ↓'), ('mean_distance', 'd(candidate, mean)'),
                     ('target_mean_distance', 'd(target, mean)'), ('residual_ratio', 'Residual magnitude ratio'),
                     ('target_projection', 'Target residual projection'), ('residual_cosine', 'Residual cosine')]:
        vals = [result['scores'][n]['mean_control'][m] for n in names]
        rows.append(['undefined' if v is None else f'{v:.4g}' for v in vals]); labels.append(label)
        colors.append(['#f0f2f4']*len(names))
    fig = plt.figure(figsize=(max(17, len(names)*1.5+3), 10.5))
    ax = fig.add_axes([.18, .13, .81, .79]); ax.axis('off')
    tab = ax.table(cellText=rows, rowLabels=labels, colLabels=columns, cellColours=colors,
                   loc='center', cellLoc='center', bbox=[0, .02, 1, .93])
    tab.auto_set_font_size(False); tab.set_fontsize(9)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0: cell.set_facecolor('#dce3eb'); cell.set_text_props(weight='bold', fontsize=9)
    fig.suptitle(f'{kind}: all 9 package metrics + explicit population-mean controls', fontsize=17)
    fig.text(.55, .012, 'Cells: raw value; c = stranger percentile with half credit for ties; r = within-case rank (1 best, ties averaged).\n'
             'Ranks are over displayed candidates, including oracle paths. Distances below use one fixed image representation/ROI.\n'
             'Residual magnitude alone is not identity. Projection/cosine are diagnostics, not a validated final score. inf = exact-match PSNR.',
             ha='center', fontsize=10)
    savefig(fig, md.PRIVATE/f'{kind}_all_metrics.png')
    with (md.PRIVATE/f'{kind}_all_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        import csv
        writer = csv.writer(fh); writer.writerow(['candidate', 'group', 'metric', 'value'])
        for n in names:
            for group, vals in result['scores'][n].items():
                for m, v in vals.items(): writer.writerow([n, group, m, v])


def locations(case):
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.imshow(case['target'], cmap='gray', vmin=0, vmax=1, origin='lower')
    ax.contour(case['mask'], [.5], colors=['#e58c21'], linewidths=2)
    ax.contour(case['mask'][:, ::-1], [.5], colors=['#1ba5c7'], linewidths=2)
    ax.set_title('Real brain: evaluated patch and opposite hemisphere', fontsize=14)
    ax.set(xticks=[], yticks=[])
    fig.text(.5, .02, 'Orange: target region. Blue: actual opposite-side donor location.\n'
             'QTIM, first fixed case; MNI z = 55 mm; affine alignment, 1-mm grid.', ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1)); savefig(fig, md.PRIVATE/'real_mri_locations.png')


def registration_gallery(case):
    fig, axes = plt.subplots(3, 6, figsize=(15, 9))
    for ax, (name, a) in zip(axes.flat, case['slices'].items()):
        ax.imshow(a, origin='lower', cmap='gray', vmin=0, vmax=1)
        ax.contour(case['mask'], [.5], colors=['#e58c21'], linewidths=.7)
        ax.set_title(name.replace('_', ' '), fontsize=10); ax.set(xticks=[], yticks=[])
    fig.suptitle('Preparation QC: fixed slice in all 18 source images', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95)); savefig(fig, md.PRIVATE/'registration_qc.png')


def real():
    torch.set_num_threads(1)
    for kind, case in [('mri', mri_case()), ('eye', eye_case())]:
        result, images = pn.evaluate(case)
        pn.save_json(md.PRIVATE/f'{kind}_scores.json', result)
        np.savez_compressed(md.PRIVATE/f'{kind}_example.npz', target=case['target'], mean=case['mean'], mask=case['mask'], **images)
        detailed_card(case, result, images, kind); metric_table(result, kind)
        if kind == 'mri': locations(case); registration_gallery(case)
        print(f'Real {kind}: local detail card, all-metric table and shrinkage scores saved', flush=True)


def shrinkage_gallery():
    paths = [('Synthetic tissue', ac.ROOT/'results/null_example_tissue.npz', 'tissue'),
             ('Synthetic eye', ac.ROOT/'results/null_example_eyes.npz', 'eyes'),
             ('Real MRI tissue', md.PRIVATE/'mri_example.npz', 'mri'),
             ('Real iris texture', md.PRIVATE/'eye_example.npz', 'eye')]
    cfg = json.loads((md.PRIVATE/'roi_config.json').read_text())
    fig, axes = plt.subplots(4, 6, figsize=(15, 11))
    order = ['shrink_1', 'shrink_0.75', 'shrink_0.5', 'shrink_0.25', 'shrink_0', 'wrong_residual']
    for row, (label, path, kind) in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            mask = data['mask']; ys, xs = np.where(mask)
            bounds = (ys.min(), ys.max()+1, xs.min(), xs.max()+1)
            for j, name in enumerate(order):
                a = crop(data[name], bounds)
                lo, hi = (60, 140) if kind == 'tissue' else (np.percentile(data['target'][mask], [1, 99]) if kind == 'eye' else (0, 1))
                axes[row, j].imshow(a, cmap='gray', vmin=lo, vmax=hi, origin='lower' if kind == 'mri' else 'upper',
                                    aspect='auto' if kind == 'eye' else 'equal', interpolation='nearest')
                d = pn.diagnostics(data['target'], data[name], data['mean'], mask)
                axes[row, j].set_xlabel(f"Mean distance ratio {d['residual_ratio']:.2f}\nTarget projection {d['target_projection']:.2f}", fontsize=9)
                axes[row, j].set(xticks=[], yticks=[])
                if row == 0: axes[row, j].set_title(LABELS[name], fontsize=12)
                if j == 0: axes[row, j].set_ylabel(label, fontsize=12)
    fig.suptitle('What disappears when an individual image becomes the population mean?', fontsize=18, y=.99)
    fig.text(.5, .014, 'Same target and ROI within every row; one fixed grayscale per modality (real iris: shared target-derived contrast window). '
             'Shrinkage deliberately uses known truth; it is not an inpainting model.\n'
             'Last column: noise residual, matched in norm before nonnegative clipping, with target projection removed. '
             'Being far from the mean does not imply identity.\n'
             'Synthetic examples: seed 401, first target. Real examples: first preselected QTIM and UTIRIS cases; local research use.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.07, right=.99, top=.93, bottom=.11, hspace=.38, wspace=.10)
    savefig(fig, md.PRIVATE/'population_mean_four_modalities.png')


def cohort_plots():
    obj = json.loads((ac.ROOT/'results/population_null_synthetic.json').read_text())
    order = ['shrink_1', 'shrink_0.75', 'shrink_0.5', 'shrink_0.25', 'shrink_0', 'repeat', 'other_side', 'stranger', 'wrong_residual']
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    for ax, kind in zip(axes, ('tissue', 'eyes')):
        agg = obj['modalities'][kind]['aggregate']
        metrics = list(ac.METRICS)
        arr = np.array([[agg[n]['c'][m]['mean'] for n in order] for m in metrics])
        ax.imshow(arr, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for i in range(len(metrics)):
            for j in range(len(order)):
                ax.text(j, i, f'{arr[i,j]:.2f}', ha='center', va='center', fontsize=10,
                        color='black' if arr[i,j] > .65 else 'white')
        ax.set_xticks(range(len(order)), [LABELS[n].replace('Other hemisphere', 'Other side') for n in order], fontsize=10)
        ax.set_yticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics], fontsize=11)
        ax.set_title(f'Synthetic {kind}: 24 independent targets; all 9 image metrics', fontsize=15)
    fig.suptitle('High fidelity rank can persist as target-specific deviations disappear', fontsize=19)
    fig.text(.5, .015, 'Cells = mean stranger-calibrated c; 0.5 is the stranger reference anchor. '
             'One displayed stranger need not have c = 0.5.\n'
             '3 seeds x 8 targets; 40 independent template and 24 reference people/families per seed. '
             'Full raw values, ranks, extra shape metrics and seed ranges are saved.\n'
             'Mean-control diagnostics are shown in the companion experiment; these image scores alone are not an identity metric.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .95)); savefig(fig, ac.ROOT/'results/figures/population_null_all_metrics.png')


def extra_plots_and_report():
    obj = json.loads((ac.ROOT/'results/population_null_synthetic.json').read_text())
    order = ['shrink_1', 'shrink_0.75', 'shrink_0.5', 'shrink_0.25', 'shrink_0', 'repeat', 'other_side', 'stranger', 'wrong_residual']
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5))
    for ax, kind in zip(axes, ('tissue', 'eyes')):
        agg = obj['modalities'][kind]['aggregate']
        metrics = [m for m in agg['shrink_0']['c'] if m not in ac.METRICS]
        a = np.array([[agg[n]['c'][m]['mean'] for n in order] for m in metrics])
        ax.imshow(a, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for i in range(len(metrics)):
            for j in range(len(order)):
                ax.text(j, i, f'{a[i,j]:.2f}', ha='center', va='center', fontsize=10,
                        color='black' if a[i,j] > .65 else 'white')
        ax.set_xticks(range(len(order)), [LABELS[n].replace('Other hemisphere', 'Other side') for n in order], fontsize=10)
        ax.set_yticks(range(len(metrics)), [m.replace('topo_err_', '').replace('pupil_', '').replace('_', ' ') for m in metrics], fontsize=11)
        ax.set_title(f'Synthetic {kind}: additional segmentation / shape checks', fontsize=15)
    fig.suptitle('Additional shape metrics do not all measure individual detail', fontsize=18)
    fig.text(.5, .025, 'Mean c across 24 targets per modality; 3 seeds. All components and holes include ROI boundary effects.\n'
             'Pupil components tie along shrinkage at c = 0.5; the noise control can add components. Not official BraTS inpainting ranking metrics.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .95)); savefig(fig, ac.ROOT/'results/figures/population_null_shape_metrics.png')
    import csv
    with (ac.ROOT/'results/population_null_summary.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh); writer.writerow(['modality', 'candidate', 'group', 'metric', 'mean', 'seed_min', 'seed_max', 'n'])
        for kind, body in obj['modalities'].items():
            for fill, groups in body['aggregate'].items():
                for group, metrics in groups.items():
                    for metric, vals in metrics.items():
                        writer.writerow([kind, fill, group, metric, vals['mean'], vals['seed_min'], vals['seed_max'], vals['n']])
    lines = ['# Population-mean negative-control experiment — 1 November 2026', '',
             'The mean can receive a high image-fidelity rank after the target-specific residual has disappeared. '
             'Distance from the mean detects this collapse, but a matched-noise control demonstrates why distance alone cannot establish identity.', '',
             'Three seeds (401–403), eight independent targets per seed per modality: 24 tissue families and 24 eye people. '
             'Forty separate template families/people and 24 reference families/people per seed. '
             'The five shrinkage steps share each target and are not independent people. Tissue prior scale = 0.25.', '',
             '| Modality | Candidate | RMSE c [seed range] | SSIM c [seed range] | Residual magnitude | Target projection |',
             '|---|---|---:|---:|---:|---:|']
    for kind, body in obj['modalities'].items():
        for fill in ['repeat', 'shrink_0.5', 'shrink_0', 'wrong_residual']:
            a = body['aggregate'][fill]
            def fmt(m):
                v = a['c'][m]; return f"{v['mean']:.3f} [{v['seed_min']:.3f}, {v['seed_max']:.3f}]"
            lines.append(f"| {kind} | {LABELS[fill]} | {fmt('rmse')} | {fmt('ssim')} | "
                         f"{a['mean_control']['residual_ratio']['mean']:.3f} | {a['mean_control']['target_projection']['mean']:.3f} |")
    lines += ['', 'Ranges are minima/maxima of three seed-level means, not confidence intervals. '
              'The tissue repeat projection is below 1 because independent acquisition noise is not preserved.', '',
              '## What was measured', '',
              'For x (target), y (candidate), and mu (independent mean), all distances are RMS distances in the same ROI and representation. '
              'Report d(x,y), d(y,mu), d(x,mu); residual magnitude = ||y-mu|| / ||x-mu||; '
              'target projection = <y-mu,x-mu> / ||x-mu||²; also record residual cosine and orthogonal residual magnitude. '
              'These are diagnostic coordinates, not a validated combined identity score. '
              'Targets indistinguishable from the mean need a repeat-measurement noise floor; exact-zero cases are marked low-information.', '',
              'The oracle path y = mu + lambda (x-mu) deliberately uses the known target. '
              'Its residual magnitude and projection equal lambda by construction; that identity is an implementation check, not independent validation. '
              'Wrong-detail noise is orthogonal to the target residual and matched in norm before nonnegativity clipping; reported diagnostics use the actual clipped image. '
              'Residual cosine stays 1 during nonzero shrinkage, so cosine alone also misses loss of residual amplitude.', '',
              'All nine package outputs are saved: RMSE, MSE, MAE, MSLE, PSNR, PSNR_eps, PSNR_01, PSNR_01_eps and SSIM. '
              'Raw values, within-case ranks, stranger-calibrated c and shape diagnostics are in `population_null_synthetic.json`; '
              'all aggregate values and seed ranges are in `population_null_summary.csv`. Exact-oracle PSNR is encoded as the string Infinity where appropriate.', '',
              '## Real demonstrations', '',
              'One fixed QTIM MRI case and one fixed UTIRIS NIR case from the prior atlas. '
              'Real case-level scores, tables and figures stay in `C:/Projects/data/projects/identity-ladder-metric/run-2026-11-01-detail/`.', '',
              'MRI: native intensities sampled at 1 mm after affine-to-independent-template registration, no nonlinear folding match and no threshold segmentation. '
              'Fixed posterior cortical patch: x=[-45,-13), y=[-65,-33), MNI z=55 mm, 32×32 pixels. '
              'Target-only anatomy selected the patch before alternative scores. '
              'Broad bilateral candidate regions are excluded from registration matching and intensity fitting. '
              'N4 uses each full source image; this is not a masked-input inpainting algorithm. '
              'The template pool has eight people from seven families, weighted equally by family; reference pool has eight other families; '
              'the target family is excluded and template/reference families do not overlap. '
              'All 18 prepared slices were visually checked; no score-based replacement. Affine correspondence does not imply voxelwise biological correspondence.', '',
              'Eyes: the first preselected case, estimated normalized iris strip, same reference/template pools of eight people each. '
              'Registration and intensity fit use visible strip pixels only; the primary mean is visible-tissue affine fitted, and the raw mean is retained separately. '
              'The patch display uses one target-derived contrast window across candidates; this changes visualization only.', '',
              'These two cases illustrate behaviour, not population validation. The MRI later visit is years apart, not a same-session noise floor. '
              'Protocol matching and real segmentation ground truth are not established. Synthetic topology and pupil metrics are not transferred to the real cases as if ground truth existed.', '',
              '## Metric provenance and reproduction', '',
              'Official package outputs: https://github.com/BraTS-inpainting/inpainting#computed-metrics . '
              'Official 2026 ranking lists MSE, PSNR and SSIM: https://github.com/BraTS-inpainting/2026_challenge/blob/main/evaluation/README.md . '
              'These are 2D adaptations for illustration, not official 3D BraTS submissions. '
              'inpainting 0.0.12; torchmetrics 1.9.0; torch 2.10.0. Source hashes and other versions are saved with the synthetic run.', '',
              'Run `python src/real_mri_detail.py prepare` (reuses existing cache); `python src/population_null.py`; '
              '`python src/detail_report.py real`; `python src/detail_report.py summary`. '
              'Real preparation depends on the previously generated private atlas ledger and existing local public datasets. '
              'Keep the paused paper unchanged.']
    (ac.ROOT/'results/population_null_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['real', 'summary', 'cohort'])
    args = parser.parse_args()
    try:
        if args.action == 'real': real()
        elif args.action == 'cohort': cohort_plots(); extra_plots_and_report()
        else: shrinkage_gallery(); cohort_plots(); extra_plots_and_report()
    except Exception as exc:
        (md.PRIVATE/'report_error_local.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise RuntimeError(f'Detail report failed ({type(exc).__name__}); details retained locally') from None
