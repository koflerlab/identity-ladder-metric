# Population-mean negative-control experiment — 1 November 2026

The mean can receive a high image-fidelity rank after the target-specific residual has disappeared. Distance from the mean detects this collapse, but a matched-noise control demonstrates why distance alone cannot establish identity.

Three seeds (401–403), eight independent targets per seed per modality: 24 tissue families and 24 eye people. Forty separate template families/people and 24 reference families/people per seed. The five shrinkage steps share each target and are not independent people. Tissue prior scale = 0.25.

| Modality | Candidate | RMSE c [seed range] | SSIM c [seed range] | Residual magnitude | Target projection |
|---|---|---:|---:|---:|---:|
| tissue | Repeat capture | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 | 0.974 |
| tissue | 50% residual | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.500 | 0.500 |
| tissue | Population mean | 0.995 [0.990, 1.000] | 0.950 [0.943, 0.958] | 0.000 | 0.000 |
| tissue | Wrong detail | 0.566 [0.479, 0.635] | 0.184 [0.094, 0.245] | 1.000 | -0.000 |
| eyes | Repeat capture | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 0.988 |
| eyes | 50% residual | 0.998 [0.995, 1.000] | 1.000 [1.000, 1.000] | 0.500 | 0.500 |
| eyes | Population mean | 0.797 [0.786, 0.812] | 0.835 [0.828, 0.844] | 0.000 | 0.000 |
| eyes | Wrong detail | 0.389 [0.234, 0.484] | 0.085 [0.042, 0.141] | 0.972 | 0.000 |

Ranges are minima/maxima of three seed-level means, not confidence intervals. The tissue repeat projection is below 1 because independent acquisition noise is not preserved.

## What was measured

For x (target), y (candidate), and mu (independent mean), all distances are RMS distances in the same ROI and representation. Report d(x,y), d(y,mu), d(x,mu); residual magnitude = ||y-mu|| / ||x-mu||; target projection = <y-mu,x-mu> / ||x-mu||²; also record residual cosine and orthogonal residual magnitude. These are diagnostic coordinates, not a validated combined identity score. Targets indistinguishable from the mean need a repeat-measurement noise floor; exact-zero cases are marked low-information.

The oracle path y = mu + lambda (x-mu) deliberately uses the known target. Its residual magnitude and projection equal lambda by construction; that identity is an implementation check, not independent validation. Wrong-detail noise is orthogonal to the target residual and matched in norm before nonnegativity clipping; reported diagnostics use the actual clipped image. Residual cosine stays 1 during nonzero shrinkage, so cosine alone also misses loss of residual amplitude.

All nine package outputs are saved: RMSE, MSE, MAE, MSLE, PSNR, PSNR_eps, PSNR_01, PSNR_01_eps and SSIM. Raw values, within-case ranks, stranger-calibrated c and shape diagnostics are in `population_null_synthetic.json`; all aggregate values and seed ranges are in `population_null_summary.csv`. Exact-oracle PSNR is encoded as the string Infinity where appropriate.

## Real demonstrations

One fixed QTIM MRI case and one fixed UTIRIS NIR case from the prior atlas. Real case-level scores, tables and figures stay in `C:/Projects/data/projects/identity-ladder-metric/run-2026-11-01-detail/`.

MRI: native intensities sampled at 1 mm after affine-to-independent-template registration, no nonlinear folding match and no threshold segmentation. Fixed posterior cortical patch: x=[-45,-13), y=[-65,-33), MNI z=55 mm, 32×32 pixels. Target-only anatomy selected the patch before alternative scores. Broad bilateral candidate regions are excluded from registration matching and intensity fitting. N4 uses each full source image; this is not a masked-input inpainting algorithm. The template pool has eight people from seven families, weighted equally by family; reference pool has eight other families; the target family is excluded and template/reference families do not overlap. All 18 prepared slices were visually checked; no score-based replacement. Affine correspondence does not imply voxelwise biological correspondence.

Eyes: the first preselected case, estimated normalized iris strip, same reference/template pools of eight people each. Registration and intensity fit use visible strip pixels only; the primary mean is visible-tissue affine fitted, and the raw mean is retained separately. The patch display uses one target-derived contrast window across candidates; this changes visualization only.

These two cases illustrate behaviour, not population validation. The MRI later visit is years apart, not a same-session noise floor. Protocol matching and real segmentation ground truth are not established. Synthetic topology and pupil metrics are not transferred to the real cases as if ground truth existed.

## Metric provenance and reproduction

Official package outputs: https://github.com/BraTS-inpainting/inpainting#computed-metrics . Official 2026 ranking lists MSE, PSNR and SSIM: https://github.com/BraTS-inpainting/2026_challenge/blob/main/evaluation/README.md . These are 2D adaptations for illustration, not official 3D BraTS submissions. inpainting 0.0.12; torchmetrics 1.9.0; torch 2.10.0. Source hashes and other versions are saved with the synthetic run.

Run `python src/real_mri_detail.py prepare` (reuses existing cache); `python src/population_null.py`; `python src/detail_report.py real`; `python src/detail_report.py summary`. Real preparation depends on the previously generated private atlas ledger and existing local public datasets. Keep the paused paper unchanged.
