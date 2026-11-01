# Synthetic pupil and iris experiment — 1 November 2026

Question: can intensity or pupil-mask scores confuse dilation of the same eye with a change of identity, and favour the population mean? Exploratory model, with protocol recorded in docs/HANDOFF.md before running.

Three seeds (201-203), each with 32 distinct target people (one target eye each), 40 separate template people and 32 separate reference people; 16 reference donors per target. Total 96 target people. The other eye belongs to the same simulated target person. No twins or real eye images in this test.

The same-eye dilation condition scales pupil radius by 1.3 and deforms the iris texture in normalized radial coordinates while keeping its latent parameters fixed. Images are 96 x 96, evaluated in the central 64 x 64. Pupil segmentation uses Gaussian smoothing sigma 0.5 and threshold 0.23. Area errors are pixels; symmetric average boundary distances are pixels; Dice is dimensionless. Scores are calibrated with half-credit for ties. No parameters were tuned after observing the results.

## Calibrated scores

Each cell is mean [minimum, maximum] over three cohort means, NOT a confidence interval. Higher c is a better relative score, not a claim of preserved identity.

| Candidate | RMSE c | SSIM c | Pupil Dice c | Boundary-distance c | Component-count c |
|---|---|---|---|---|---|
| Same eye: repeat | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] |
| Same eye: dilated | 0.004 [0.000, 0.008] | 0.006 [0.000, 0.012] | 0.007 [0.002, 0.012] | 0.003 [0.000, 0.008] | 0.500 [0.500, 0.500] |
| Other eye | 0.770 [0.719, 0.844] | 0.766 [0.703, 0.844] | 0.753 [0.699, 0.820] | 0.758 [0.701, 0.822] | 0.500 [0.500, 0.500] |
| Population mean | 0.794 [0.725, 0.840] | 0.842 [0.781, 0.885] | 0.649 [0.607, 0.721] | 0.648 [0.607, 0.723] | 0.500 [0.500, 0.500] |
| Blurred stranger | 0.645 [0.576, 0.707] | 0.638 [0.582, 0.703] | 0.500 [0.397, 0.561] | 0.499 [0.404, 0.559] | 0.500 [0.500, 0.500] |
| Stranger eye | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| Same eye: dilated; matched stranger reference | 0.449 [0.291, 0.529] | 0.432 [0.295, 0.506] | 0.457 [0.300, 0.542] | 0.454 [0.291, 0.541] | 0.500 [0.500, 0.500] |

## Findings and limits

- Same-eye repeats score c=1 on the intensity and pupil-shape metrics in this low-noise simulator. Dilation of the same eye scores near zero against normal-pupil strangers (RMSE 0.004, SSIM 0.006), despite unchanged latent identity parameters.
- Matching stranger dilation raises same-eye c to about 0.45 (RMSE) / 0.43 (SSIM), not back to the repeat level. Matching alone does not make these representations invariant to dilation in this simulator.
- The independent-population mean scores above the stranger anchor: RMSE c=0.794, SSIM c=0.842, pupil Dice c=0.649. It is not near-perfect on all metrics; this limits transfer of the previous tissue result.
- Component-count error is zero for every compared pupil; c=0.5 including same-eye repeats. Thus this topological descriptor is uninformative by construction here. No topology-based identity claim follows.
- Fellow-eye scores reflect the chosen shared size/shape parameters. They are not evidence about human iris bilaterality. Pupil shape is an acquisition/physiology-sensitive feature, not an established biometric identity descriptor.
- This is neither an inpainting-model evaluation nor a study of real pupil physiology. No occlusion, unknown pose, estimated segmentation or clinical data. The next discriminating control would compare iris texture in dilation-normalized coordinates, keeping preprocessing independent of identity labels.

## Artifacts and checks

- `python src/pupil_ladder.py`: all raw cohort aggregates, c scores, fixed-example values, versions and source hash in pupil_ladder.json; summary CSV also saved.
- `python src/pupil_report.py`: all 13 metrics as a heatmap and this report.
- `figures/pupil_fixed_example.png`: first target at seed 201, selected by index rather than score; one-example numbers differ from cohort means. The displayed stranger is the first reference donor, not the stranger anchor average.
- `figures/label_fixed_example_folding.png` and `figures/label_hemisphere_locations.png`: corrected synthetic-tissue illustration with other-day and opposite-side donors. The right-side patch is a separate simulated side, not a reflection of the target. Homologous coordinates and the schematic outline are defined by construction, not estimated from MRI.
- 95 tests passed; new checks cover pupil geometry, tie handling and use of the actual opposite-side array. Existing experiment JSONs and paper sources were preserved.
