# Protocol: donor-averaging ladder on brain MRI (QTIM, 2 Nov 2026)

Recorded before any result of this experiment was computed. It is the brain counterpart of
`PROTOCOL_donor_ladder_fundus.md`, with the same fills, measures and summary.

## Decision

On 2 Nov 2026 Kate approved this run ("Yes, run it") after being told that MRI registration QC is not met. All
results are labelled **pre-QC**. The scans use the existing affine registration only. SyN is not used here, because
it can warp individual folding towards the template.

## Data

QTIM (`siblings-mri`, OpenNeuro ds004169): every T1w scan already prepared by `real_cases.prepare`.

- Preparation: SynthStrip, N4, affine ANTs registration to the nilearn MNI 2009a symmetric template, 1 mm grid.
- The run uses `prep/scan_*_brain.nii.gz` from `real-cases-run` and `real-cases-run-expanded`, deduplicated by
  participant and session.
- Counts: 244 scans, 221 people, 23 people with a second session, 168 families (53 sibling pairs and 4 co-twin
  pairs among the prepared scans).
- No new download and no new registration.

## Preprocessing (fixed)

- Skull-stripped volume, own brain mask = voxels above zero.
- Each scan is z-scored inside its own brain mask.
- A patch counts as valid for a scan if at least 95% of its voxels lie inside that scan's mask.

## Patches (fixed before results)

- 32 cubes of 16 mm, with a 4 mm context shell (24 mm window).
- Centres are drawn once with seed 20261102 from the positions where at least 90% of the window lies inside the
  template brain mask (nilearn MNI152 1 mm brain mask). The template mask, not the data, defines where patches
  can go.
- The two frozen boxes of the real-case protocol (`deep` and `parietal`) are reported separately.
- The same positions are used for every target and donor.
- Mirroring: reflection across the template midline, x to -x in millimetres.

## Cohorts

Every pool excludes the target's whole family.

| Cohort | Donors |
|---|---|
| `population` | session-1 scans of people from other families |
| `population_both` | the same plus their mirrored hemispheres |
| `cohort` | same sex, age within 2 years |
| `knn` | the k nearest donors by Euclidean distance on the context shell of that patch |
| `other_hemisphere` | the target's own scan, mirrored |
| `sibling` | the first different-age member of the family |
| `cotwin` | the first same-age member of the family |
| `repeat` | the same person's session 2 |

## Measures

- Identical to the fundus protocol: k in {1, 2, 4, 8, 16, 32, 64} plus all donors; error and spread per target
  and patch, normalized by V, the mean error of the full population mean.
- Fit of (1 - R2)(1 + 1/k) per cohort; r = 1 - (error/V)/2 for single same-person or relative donors.
- 95% intervals by bootstrap over families (2000 resamples).
- Targets are session-1 scans.
- The co-twin rung has only 4 pairs and is reported as descriptive only.

## Outputs

- Private, `C:\Projects\data\projects\identity-ladder-metric\donor-ladder-brain-20261102-v1\`: patch arrays and
  per-target rows.
- Public: `results/donor_ladder_brain.json` (aggregates only).

## Amendment 1 (2 Nov 2026, after a degenerate first run)

The first run read `scan_*_brain.nii.gz` as the image. That file is the binary brain mask, so every z-scored cube
was zero and every error was 0. No interpretable value was produced. Intensities now come from `scan_*.nii.gz`
(N4-corrected, registered), and the brain mask from `scan_*_brain.nii.gz`. A scan with no intensity variation
inside its mask now stops the run. The degenerate cache and rows were deleted. Nothing else changes.
