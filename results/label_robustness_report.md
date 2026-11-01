# H19 exploratory sensitivity experiment — 1 November 2026

Five fresh seeds (101-105), each with 60 simulated families (30 MZ, 30 DZ), two targets per family and 16 stranger donors per target. Two input variance settings and three prior scales: 30 runs, 3,600 target-condition evaluations. The same random draws are reused across settings: 300 simulated families per condition, not 1,800 independent families.

Every table cell below is mean [minimum, maximum] of five cohort means. These ranges are NOT confidence intervals. Higher c always means better relative to stranger fills; c=1 is not zero reconstruction error. Stranger c=0.5 is a fixed anchor. Both members of a target family are excluded from the population mean.

## Population-mean fill: calibrated scores

| Setting | RMSE | MAE | SSIM | Dice | Component-count error | Perimeter error |
|---|---|---|---|---|---|---|
| base_prior0 | 0.996 [0.993, 0.998] | 0.904 [0.896, 0.911] | 0.953 [0.947, 0.957] | 0.679 [0.554, 0.848] | 0.549 [0.467, 0.635] | 0.480 [0.217, 0.663] |
| base_prior0.25 | 0.996 [0.993, 0.998] | 0.908 [0.896, 0.917] | 0.956 [0.955, 0.957] | 0.759 [0.707, 0.810] | 0.541 [0.480, 0.582] | 0.576 [0.472, 0.666] |
| base_prior1 | 0.995 [0.993, 0.997] | 0.929 [0.918, 0.936] | 0.991 [0.985, 0.995] | 0.903 [0.896, 0.910] | 0.555 [0.491, 0.601] | 0.582 [0.494, 0.665] |
| low_h_prior0 | 0.997 [0.995, 0.998] | 0.907 [0.899, 0.912] | 0.953 [0.948, 0.959] | 0.671 [0.480, 0.871] | 0.505 [0.415, 0.599] | 0.512 [0.406, 0.649] |
| low_h_prior0.25 | 0.997 [0.995, 0.998] | 0.907 [0.901, 0.915] | 0.960 [0.953, 0.967] | 0.751 [0.673, 0.820] | 0.560 [0.510, 0.598] | 0.573 [0.518, 0.617] |
| low_h_prior1 | 0.995 [0.992, 0.999] | 0.934 [0.926, 0.943] | 0.990 [0.985, 0.993] | 0.903 [0.894, 0.913] | 0.563 [0.528, 0.604] | 0.598 [0.524, 0.655] |

## Paired population-mean minus MZ co-twin differences in c

Population mean and twin are compared on exactly the same 60 MZ targets (30 families) in each seed, with the same per-target stranger reference. Positive favours the population mean.

| Setting | RMSE delta-c | Dice delta-c | Component error delta-c | Perimeter error delta-c |
|---|---|---|---|---|
| base_prior0 | 0.017 [0.005, 0.030] | -0.267 [-0.384, -0.116] | -0.035 [-0.087, 0.077] | -0.121 [-0.402, 0.074] |
| base_prior0.25 | 0.015 [0.002, 0.030] | -0.167 [-0.216, -0.109] | -0.023 [-0.077, 0.051] | -0.022 [-0.145, 0.027] |
| base_prior1 | 0.020 [0.002, 0.050] | -0.041 [-0.062, -0.016] | -0.036 [-0.068, -0.013] | -0.037 [-0.099, 0.032] |
| low_h_prior0 | 0.255 [0.211, 0.333] | -0.000 [-0.208, 0.186] | -0.009 [-0.068, 0.034] | -0.021 [-0.224, 0.077] |
| low_h_prior0.25 | 0.257 [0.214, 0.349] | 0.090 [-0.020, 0.160] | 0.047 [-0.014, 0.111] | 0.049 [0.007, 0.099] |
| low_h_prior1 | 0.264 [0.195, 0.348] | 0.215 [0.150, 0.284] | 0.019 [-0.033, 0.117] | 0.090 [0.014, 0.198] |

## Interpretation

- Supported in this simulator: the population mean scores very highly on squared-error metrics across every tested prior and seed (RMSE mean c about 0.995-0.997). MAE is less favourable (about 0.90-0.93), SSIM is about 0.95-0.99.
- H19 needs a conditional statement. In LOW_H, the paired Dice advantage over MZ is positive for every seed at prior 1 (mean +0.215, range +0.150 to +0.284). At prior 0.25 the range crosses zero (mean +0.090, range -0.020 to +0.160). At prior 0 there is no consistent direction (mean -0.0002, range -0.208 to +0.186). BASE favours the MZ twin on Dice for all tested seeds and priors.
- Component-count scores of the mean are roughly stranger-like (c about 0.50-0.56 across condition means), but that does not validate an identity metric. Perimeter differences versus MZ can favour either fill. Rare holes and many ties limit hole-count resolution, including for repeat acquisitions.
- This does not establish the genetic-share threshold from the linear Gaussian model for nonlinear images or masks. The genetic parameters describe latent fields; they are not measured heritability of the rendered images.

## Limits and next discriminating check

Only synthetic 2D arrays, a threshold segmenter and the segmentation of the noisy true image as oracle were used. No real MRI, BraTS winner model or FastSurfer validation. Families share donors and population means within a seed. The common prior is hand-designed. Topology is measured after cutting by a square mask; perimeter is geometry, not a topological invariant.

Next: test mask-boundary dependence (full mask versus an inner region, and segmentation with versus without original target context). This is motivated by the image assembly and is not yet executed. Then test on real MRI when access is available.

## Reproduce and inspect

- `python src/label_robustness.py`: bounded experiment; resumes matching checkpoints, rejects design/source/version drift.
- `python src/ladder_report.py`: combined table, nine-metric toy figures, 15-metric robustness figure and paired MZ/DZ figures.
- `python src/label_examples.py`: first target by index, seed 101 / BASE, same donor selection; no score-based example selection.
- Full results and provenance: `label_robustness.json`; summary tables: `label_robustness_summary.csv`, `label_paired_twin_summary.csv`.
- `all_ladder_metrics.csv`: 1,116 rows from the original toy and label ladders; missing label-model rho remains blank, raw means and medians remain distinct.
- Validation: 91 tests passed; original BASE / seed-2 output reproduced exactly (max absolute difference 0); output completeness, hashes, score bounds, twin subsets and CSV uniqueness checked. Figures visually inspected. Old saved results and paused paper remain unchanged.
