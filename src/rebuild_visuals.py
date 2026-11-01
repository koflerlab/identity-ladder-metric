"""Rebuild visual cards and five-step mixing dashboards from saved experiments.

No registration, donor selection or random simulation is repeated. Real examples,
their numeric tables and mixed real/synthetic figures stay in the registered local
data run. Only the explicitly requested paper figure is refreshed, not paper text.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
import numpy as np

import atlas_common as ac
import label_ladder as ll
import pupil_ladder as pl
import detail_report as dr
import real_mri_detail as md

BLUE, ORANGE, TEAL = '#2468a0', '#ca541e', '#008377'
STEPS = ['shrink_1', 'shrink_0.75', 'shrink_0.5', 'shrink_0.25', 'shrink_0']
STEP_LABELS = ['100% original', '75% original', '50% original', '25% original', 'Population mean']
METRICS = list(ac.METRICS)
TITLES = {'tissue': 'Synthetic tissue', 'eyes': 'Synthetic eye', 'mri': 'Real brain MRI', 'eye': 'Real iris'}


def save(fig, path):
    for ext in ('png', 'pdf'):
        fig.savefig(Path(path).with_suffix('.'+ext), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load(kind):
    real = kind in ('mri', 'eye')
    path = md.PRIVATE/f'{kind}_example.npz' if real else ac.ROOT/f'results/null_example_{kind}.npz'
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if real:
        values = json.loads((md.PRIVATE/f'{kind}_scores.json').read_text())['scores']
        body = None
    else:
        body = json.loads((ac.ROOT/'results/population_null_synthetic.json').read_text())['modalities'][kind]
        values = body['rows'][0]['scores']
    ys, xs = np.where(arrays['mask'])
    roi = (int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1)
    display = (60, 140) if kind == 'tissue' else ((0, 1) if kind != 'eye' else np.percentile(arrays['target'][arrays['mask']], [1, 99]))
    return arrays, values, body, roi, display


def series(values, body, fill, group, metric):
    """Return saved point/range; never invent uncertainty for a single real case."""
    if body is None:
        x = values[fill][group][metric]
        x = np.nan if x is None else float(x)
        return x, x, x
    a = body['aggregate'][fill][group][metric]
    return tuple(np.nan if a[k] is None else float(a[k]) for k in ('mean', 'seed_min', 'seed_max'))


def visual_brain(kind):
    data, values, _, roi, display = load(kind)
    synthetic = kind == 'tissue'
    names = ['shrink_1', 'repeat', 'later', 'other_side', 'co_twin', 'stranger', 'shrink_0'] if synthetic else \
            ['shrink_1', 'later', 'other_side', 'stranger', 'shrink_0', 'blurred_stranger']
    n = len(names); target, mask = data['target'], data['mask']
    y0, y1, x0, x1 = roi
    context = (0, 64, 0, 64) if synthetic else (y0-16, y1+16, x0-16, x1+16)
    origin = 'upper' if synthetic else 'lower'
    fig, axes = plt.subplots(3, n, figsize=(18, 10.5))
    for j, name in enumerate(names):
        fill = np.where(mask, data[name], target)
        axes[0, j].imshow(dr.crop(fill, context), cmap='gray', vmin=display[0], vmax=display[1], origin=origin, interpolation='nearest')
        axes[0, j].add_patch(Rectangle((x0-context[2]-.5, y0-context[0]-.5), x1-x0, y1-y0, fill=False, ec=ORANGE, lw=1.4))
        axes[1, j].imshow(dr.crop(fill, roi), cmap='gray', vmin=display[0], vmax=display[1], origin=origin, interpolation='nearest')
        if synthetic:
            axes[2, j].imshow(dr.crop(ll.segment(fill), roi), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        else:
            diff = axes[2, j].imshow(dr.crop(data[name]-target, roi), cmap='RdBu_r', vmin=-.4, vmax=.4, origin=origin, interpolation='nearest')
        title = dr.LABELS[name]
        if name == 'co_twin': title = 'MZ co-twin'
        axes[0, j].set_title(title, fontsize=12)
        for ax in axes[:, j]: ax.set(xticks=[], yticks=[])
        d = values[name]['mean_control']
        axes[2, j].set_xlabel(f"Mean-distance ratio {d['residual_ratio']:.2f}\nTarget projection {d['target_projection']:.2f}", fontsize=10)
    for ax, text in zip(axes[:, 0], ['Context + fill region', 'Evaluated tissue enlarged', 'Threshold segmentation' if synthetic else 'Candidate minus original']):
        ax.set_ylabel(text, fontsize=12)
    location = fig.add_axes([.04, .80, .16, .14])
    if synthetic:
        brain = np.concatenate([target, np.full((64, 8), np.nan), data['other_side'][:, ::-1]], axis=1)
        yy, xx = np.mgrid[:64, :136]
        brain[((xx-67.5)/63)**2+((yy-31.5)/31)**2 > 1] = np.nan
        location.imshow(brain, cmap='gray', vmin=60, vmax=140)
        for pos, col in [(15.5, ORANGE), (87.5, TEAL)]:
            location.add_patch(Rectangle((pos, 15.5), 32, 32, fill=False, ec=col, lw=2))
    else:
        location.imshow(target, cmap='gray', vmin=0, vmax=1, origin='lower')
        location.contour(mask, [.5], colors=[ORANGE], linewidths=1.6)
        location.contour(mask[:, ::-1], [.5], colors=[TEAL], linewidths=1.6)
        cax = fig.add_axes([.967, .23, .008, .13]); fig.colorbar(diff, cax=cax)
    location.set(xticks=[], yticks=[])
    fig.suptitle(f'{TITLES[kind]}: corresponding tissue, alternative donors and population mean', fontsize=19, y=.985)
    fig.text(.23, .89, 'Orange: target region. Teal: the corresponding location in the other hemisphere.', fontsize=12)
    fig.text(.23, .845, 'Bilateral arrangement is schematic; independent side component, already in target coordinates.' if synthetic else
             'QTIM; fixed posterior cortical patch, 32 x 32 mm. Native intensities on a 1-mm affine template grid.', fontsize=11)
    fig.text(.5, .025, ('Seed 401, first target of the saved 24-target tissue experiment; prior 0.25. Synthetic threshold masks, not MRI segmentation.\n'
                       'Mean: 40 independent template families; 24 reference families. Image and all controls use the same original and ROI.') if synthetic else
             ('First fixed QTIM case; later visit is years apart. Mean: 8 people / 7 equally weighted families; references: 8 other families.\n'
              'One grayscale and error scale for all panels. Registration does not equate individual folds. Local descriptive case, not a population result.'),
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.04, right=.96, top=.76, bottom=.13, hspace=.13, wspace=.07)
    path = ac.ROOT/'results/figures/visual_tissue_rebuilt' if synthetic else md.PRIVATE/'visual_mri_rebuilt'
    save(fig, path)


def visual_eyes():
    data, values, _, roi, display = load('eyes')
    order = ['shrink_1', 'repeat', 'dilated', 'other_side', 'stranger', 'shrink_0']
    fig, axes = plt.subplots(3, 6, figsize=(17, 9))
    oracle = pl.segment(data['target'])
    for j, name in enumerate(order):
        im = np.where(data['mask'], data[name], data['target'])
        seg = pl.segment(im)
        axes[0, j].imshow(im, cmap='gray', vmin=0, vmax=1)
        axes[1, j].imshow(seg, cmap='gray', vmin=0, vmax=1)
        axes[2, j].imshow(np.ones(im.shape), cmap='gray', vmin=0, vmax=1)
        axes[2, j].contour(oracle, [.5], colors=[BLUE], linewidths=2)
        axes[2, j].contour(seg, [.5], colors=[ORANGE], linewidths=1.2, linestyles='--')
        title = 'Other eye' if name == 'other_side' else dr.LABELS[name]
        axes[0, j].set_title(title, fontsize=13)
        for ax in axes[:, j]: ax.set(xticks=[], yticks=[])
        d = values[name]['mean_control']
        axes[2, j].set_xlabel(f"Mean-distance ratio {d['residual_ratio']:.2f}\nTarget projection {d['target_projection']:.2f}", fontsize=10)
    for ax, label in zip(axes[:, 0], ['Intensity', 'Estimated pupil', 'Pupil contours']): ax.set_ylabel(label, fontsize=12)
    fig.suptitle('Synthetic eye: repeat, dilation, fellow eye and population mean', fontsize=19)
    fig.text(.5, .025, 'Seed 401, first saved target. Blue contour: original; orange dashed: candidate. Fixed grayscale and 64 x 64 evaluation mask.\n'
             'Dilation preserves latent eye parameters; pixel residual projection is a diagnostic, not an invariant identity score.\n'
             'All 9 image metrics and 4 pupil metrics are saved; the companion mixing figure follows this same target.', ha='center', fontsize=10)
    fig.subplots_adjust(left=.04, right=.99, top=.91, bottom=.16, hspace=.08, wspace=.08)
    save(fig, ac.ROOT/'results/figures/visual_eyes_rebuilt')


def visual_real_eye():
    data, values, _, roi, display = load('eye')
    case = dr.eye_case()
    ledger = json.loads((ac.PRIVATE/'real_eyes/case_ledger.json').read_text())
    native = case['native_items']+[dr.ra.eye_item(ledger['reference_paths'][0][0])]
    fig = plt.figure(figsize=(18, 9.8)); grid = fig.add_gridspec(3, 12, height_ratios=[2.8, 1.1, 1])
    for j, (item, label) in enumerate(zip(native, ['Original eye', 'Repeat capture', 'Other eye (flipped)', 'Stranger eye'])):
        ax = fig.add_subplot(grid[0, 3*j:3*j+3]); ax.imshow(item['image'], cmap='gray', vmin=0, vmax=255)
        for boundary, col in [('pupil', ORANGE), ('iris', TEAL)]:
            x, y, radius = item['seg'][boundary]; ax.add_patch(Circle((x, y), radius, fill=False, ec=col, lw=1.3))
        ax.set_title(label, fontsize=13); ax.set(xticks=[], yticks=[])
    names = ['shrink_1', 'repeat', 'other_side', 'stranger', 'shrink_0', 'blurred_stranger']
    for j, name in enumerate(names):
        ax = fig.add_subplot(grid[1, 2*j:2*j+2]); err = fig.add_subplot(grid[2, 2*j:2*j+2])
        ax.imshow(dr.crop(data[name], roi), cmap='gray', vmin=display[0], vmax=display[1], aspect='auto', interpolation='nearest')
        err.imshow(dr.crop(data[name]-data['target'], roi), cmap='RdBu_r', vmin=-80, vmax=80, aspect='auto', interpolation='nearest')
        ax.set_title('Other eye' if name == 'other_side' else dr.LABELS[name], fontsize=11)
        for a in (ax, err): a.set(xticks=[], yticks=[])
        d = values[name]['mean_control']
        err.set_xlabel(f"Mean-distance ratio {d['residual_ratio']:.2f}\nTarget projection {d['target_projection']:.2f}", fontsize=10)
        if j == 0: ax.set_ylabel('Iris patch enlarged', fontsize=11); err.set_ylabel('Difference', fontsize=11)
    fig.suptitle('Real eye: original photographs and the same evaluated iris patch', fontsize=19, y=.98)
    fig.text(.5, .025, 'UTIRIS NIR; first fixed case. Circles are algorithm-estimated boundaries, not ground truth. Lower rows are normalized iris patches.\n'
             'One target-derived contrast window for all patches; image values and metrics are unchanged. Mean: 8 other people, visible-tissue affine fit.\n'
             'No whole-eye mean photograph is implied by the mean iris patch. Local research copy; no redistribution.', ha='center', fontsize=10)
    fig.subplots_adjust(left=.04, right=.99, top=.91, bottom=.15, hspace=.34, wspace=.18)
    save(fig, md.PRIVATE/'visual_eye_rebuilt')


def mixing(kind):
    data, values, body, roi, display = load(kind)
    fig = plt.figure(figsize=(15.5, 15))
    grid = fig.add_gridspec(5, 6, height_ratios=[1.5, 1.1, 1.1, 1.1, 1.0], hspace=.60, wspace=.88)
    # The five actual arrays are used; no interpolation of saved plot images.
    strip = grid[0, :].subgridspec(1, 5, wspace=.06)
    for j, name in enumerate(STEPS):
        ax = fig.add_subplot(strip[0, j])
        image = data[name] if kind == 'eyes' else dr.crop(data[name], roi)
        ax.imshow(image, cmap='gray', vmin=display[0], vmax=display[1], origin='lower' if kind == 'mri' else 'upper',
                  aspect='auto' if kind == 'eye' else 'equal', interpolation='nearest')
        ax.set_title(STEP_LABELS[j], fontsize=12)
        ax.set(xticks=[], yticks=[])
    xs = np.arange(5)
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[1+i//3, 2*(i%3):2*(i%3)+2]); rhs = ax.twinx()
        raw = np.array([series(values, body, name, 'raw', metric) for name in STEPS])
        cal = np.array([series(values, body, name, 'c', metric) for name in STEPS])
        shown = np.arange(1, 5) if metric.startswith('psnr') else xs
        ax.plot(shown, raw[shown, 0], '-o', color=BLUE, lw=1.8, ms=4)
        if body is not None: ax.fill_between(shown, raw[shown, 1], raw[shown, 2], color=BLUE, alpha=.13)
        rhs.plot(xs, cal[:, 0], '--s', color=ORANGE, lw=1.5, ms=3.5)
        if body is not None: rhs.fill_between(xs, cal[:, 1], cal[:, 2], color=ORANGE, alpha=.10)
        if metric.startswith('psnr'):
            value = raw[0, 0]
            ax.text(.02, .93, '100%: '+('∞' if np.isinf(value) else f'{value:.1f} dB'), transform=ax.transAxes,
                    color=BLUE, fontsize=9, va='top', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
        ax.set_title(dr.METRIC_LABELS[metric], fontsize=12, pad=7)
        ax.set_xticks(xs, ['100', '75', '50', '25', '0'], fontsize=9)
        ax.set_xlim(-.12, 4.12); rhs.set_ylim(-.04, 1.04); rhs.set_yticks([0, .5, 1])
        rhs.tick_params(axis='y', colors=ORANGE, labelsize=9); ax.tick_params(axis='y', colors=BLUE, labelsize=9)
        if i%3 == 0: ax.set_ylabel('Raw value', color=BLUE, fontsize=10)
        if i%3 == 2: rhs.set_ylabel('Calibrated c', color=ORANGE, fontsize=10)
        ax.set_xlabel('Original contribution, %', fontsize=9)
        ax.grid(color='#e8e8e8', lw=.7); ax.spines['top'].set_visible(False); rhs.spines['top'].set_visible(False)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
    distance = fig.add_subplot(grid[4, :3]); projection = fig.add_subplot(grid[4, 3:])
    # Common normalizer is the target-to-mean RMS distance within each case.
    for m, col, label in [('target_distance', BLUE, 'Distance to original'), ('mean_distance', ORANGE, 'Distance to mean')]:
        points = [series(values, body, n, 'mean_control', m) for n in STEPS]
        distance.plot(xs, [v[0] for v in points], '-o', color=col, label=label, ms=4)
        if body is not None: distance.fill_between(xs, [v[1] for v in points], [v[2] for v in points], color=col, alpha=.12)
    baseline = series(values, body, 'shrink_0', 'mean_control', 'target_mean_distance')[0]
    distance.axhline(baseline, color='#666', ls=':', label='Original-to-mean distance')
    distance.set_title('Three distances in the same representation', fontsize=12)
    distance.set_xticks(xs, ['100', '75', '50', '25', '0']); distance.set_xlabel('Original contribution, %', fontsize=10)
    distance.set_ylabel('RMS distance', fontsize=10); distance.legend(fontsize=8, frameon=False, loc='center left')
    positions = np.r_[xs, 6]
    for m, col, label, marker in [('residual_ratio', ORANGE, 'Distance-to-mean ratio', 'o'),
                                  ('target_projection', TEAL, 'Target residual projection', 'x')]:
        vals = [series(values, body, n, 'mean_control', m)[0] for n in STEPS+['wrong_residual']]
        projection.plot(xs, vals[:5], color=col, marker=marker, ms=5, label=label)
        projection.scatter([6], [vals[-1]], color=col, marker=marker, s=45)
    projection.set_xticks(positions, ['100', '75', '50', '25', '0', 'Noise\ncontrol'])
    projection.set_ylim(-.08, 1.15); projection.set_title('Noise can be far from the mean without target detail', fontsize=12)
    projection.legend(fontsize=8, frameon=False, loc='upper center'); projection.axvline(5, color='#aaa', ls=':', lw=.8)
    for ax in (distance, projection): ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#ededed')
    fig.suptitle(f'{TITLES[kind]} | original → population mean', fontsize=21, y=.991)
    legend = [Line2D([0], [0], color=BLUE, marker='o', label='Raw metric (left axis)'),
              Line2D([0], [0], color=ORANGE, ls='--', marker='s', label='Stranger-calibrated c (right axis)')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .970), ncol=2, frameon=False, fontsize=11)
    population = 'Images: seed 401, first target. Curves: 24 targets / 3 seeds; bands: min–max of seed means, NOT confidence intervals.' if body else \
                 'One fixed real case; curves belong to exactly the pictured patch. No uncertainty band or population claim.'
    fig.text(.5, .012, population+'\n'
             'y = lambda × original + (1-lambda) × mean. Same ROI and grayscale across all five steps. '
             'PSNR oracle values are printed; blue PSNR axes show 75%–0%.\n'
             'Mean-control curves equal lambda by construction; this is a diagnostic, not validation of a final identity score. '
             'Noise is orthogonal and norm-matched before nonnegative clipping.', ha='center', fontsize=10)
    fig.subplots_adjust(left=.07, right=.93, top=.93, bottom=.085)
    dest = ac.ROOT/f'results/figures/mixing_{kind}' if body else md.PRIVATE/f'mixing_{kind}'
    save(fig, dest)


def run():
    visual_brain('tissue'); visual_eyes(); visual_brain('mri'); visual_real_eye()
    for kind in TITLES:
        mixing(kind)
        print(f'{kind}: visual card and all-nine-metric mixing dashboard rebuilt', flush=True)
    paths = [ac.ROOT/'results/toy_ladder.json', ac.ROOT/'results/population_null_synthetic.json',
             ac.ROOT/'results/null_example_tissue.npz', ac.ROOT/'results/null_example_eyes.npz']
    manifest = {'date': '2026-11-01', 'mode': 'render cached experiments; no new simulation or registration',
                'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'mixing_weights_original': [1, .75, .5, .25, 0], 'metric_count': 9,
                'real_case_outputs': str(md.PRIVATE), 'synthetic_outputs': 'results/figures/',
                'real_case_counts': {'mri': 1, 'eye': 1}, 'synthetic_cohort_sizes': {'tissue': 24, 'eyes': 24},
                'notes': ['Raw curves and calibrated c share x but use explicitly separate y axes.',
                          'PSNR exact-oracle values are text labels, never finite replacements for infinity.',
                          'Synthetic images are one example; curves average the saved cohort. Real curves use the pictured case.']}
    (ac.ROOT/'results/visual_rebuild_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    try: run()
    except Exception as exc:
        (md.PRIVATE/'rebuild_error_local.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise RuntimeError(f'Visual rebuild failed ({type(exc).__name__}); details remain local') from None
