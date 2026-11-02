# Protocol: robustness of the donor-averaging ladders (2 Nov 2026)

Recorded before any variant below was computed. It re-checks the results of `PROTOCOL_donor_ladder_fundus.md`
and `PROTOCOL_donor_ladder_brain.md` (both with Amendment 1) under three kinds of perturbation. Every result stays
**pre-QC**: registration visual QC (`docs/REGISTRATION_QC_PROTOCOL.md`) is not met, and the metric pause is lifted
only for the donor-ladder experiments. No identity score is computed.

## What is fixed

Everything not listed as a perturbation is as in the two parent protocols: the cohorts, the k values, the error and
spread measures, the normalization by V (the mean error of the full population mean of the same region), the fit
of (1 - R2)(1 + 1/k), r = 1 - (error/V)/2 for single donors, and the person (fundus) or family (brain) bootstrap
with 2000 resamples. The prepared images are reused: fundus images are re-loaded from the open datasets, brain
scans from the existing affine preparation in `real-cases-run` and `real-cases-run-expanded`.

## Variants

`B` marks the baseline; it must reproduce `results/donor_ladder_fundus.json` and `results/donor_ladder_brain.json`
before any other variant is read. A variant changes one thing at a time.

### Fundus (PAPILA and GRAPE)

| Variant | Change |
|---|---|
| `baseline` | seed 20261102, 48 px patches, 16 px ring, support 0.90, disc-fovea frame |
| `seed1` ... `seed5` | seed s (1-5) for patch positions, donor draws and bootstrap |
| `patch32`, `patch64` | patch side 32 or 64 px on the 256 grid (ring and support unchanged) |
| `support85`, `support95` | support threshold 0.85 or 0.95 |
| `reg_disc` | disc-only frame: the disc moved to the canonical disc position, isotropic scale that brings the field-of-view radius to the dataset median, no rotation and no fovea |
| `reg_jitter` | disc-fovea frame with each landmark moved by isotropic Gaussian noise, sd 10 px on the 512 image (seed 20261102) |

If a variant has fewer than 32 candidate windows, all candidates are used; if it has none, the variant is reported
as "no window" for that dataset. Landmark detection, the excluded images and the one-image-per-eye units are the
same in every variant, so the registration variants differ only in the transform. The 2D-ladder metric box is
reported separately in every variant, as before.

### Brain (QTIM)

| Variant | Change |
|---|---|
| `baseline` | seed 20261102, 16 mm cubes, 4 mm shell, template support 0.90, own-mask validity 0.95, affine |
| `seed1` ... `seed5` | seed s for cube positions, donor draws and bootstrap |
| `cube12`, `cube24`, `cube32` | cube side 12, 24 or 32 mm (shell unchanged) |
| `tsupport80`, `tsupport95` | template-support threshold 0.80 or 0.95 for cube positions |
| `valid90`, `valid99` | own-mask validity threshold 0.90 or 0.99 |
| `reg_affine_subset`, `reg_syn_subset` | the same subset with the affine scans or with SyN added on top of them |

The frozen `deep` and `parietal` boxes are reported separately in every variant; the cube-size and support
variants do not change them.

**SyN subset.** No SyN job was running when this protocol was written (no ANTs or registration process), and the
existing SyN outputs cover one development scan each (`real-cases-run/syn_amendment4`,
`registration-qc-mri-refined-20261102-v1`), so they cannot form a subset. One full-resolution SyN (Amendment-4
settings) took more than 10 minutes, so SyN is estimated on the 2 mm template grid from the affine scan (CC, radius
2, iterations 60/30/10, flow sigma 3, seed 123) and applied to the 1 mm intensities and brain mask. The subset is
drawn once with seed 20261102 before any SyN output exists: every person with a second session (both sessions),
plus 15 families with at least two session-1 members (two members each). Donor pools are restricted to the subset
in both arms, so the comparison is paired. SyN can pull individual folding towards the template; that is the
reason the parent protocol excluded it, and the reason it is a perturbation here, not a replacement.

## Headline numbers, per variant and region

1. Stranger k = 1 ratio: `population` error/V at k = 1 (one random draw), and its exact expectation over the whole
   donor pool (seed-free).
2. Fit of the `population` rung to (1 - R2)(1 + 1/k): R2 and the rms residual.
3. k_eff of the population mean against the finite-pool prediction k (N - 1)/(N - k), N the mean pool size.
4. Cohort R2.
5. kNN: best k, and whether the minimum is interior (U-shape: the best k beats both k = 1 and the largest k).
6. Fellow eye / other hemisphere r.
7. Other visit / repeat r.
8. Sibling r (brain).

## Verdict rules (fixed now)

Each conclusion is judged over every variant and dataset where it is defined:

| Conclusion | Holds if every variant has |
|---|---|
| One stranger costs about twice the mean | population k = 1 error/V in [1.8, 2.2] |
| Population rung follows 1 + 1/k with no free parameter | population abs(R2) < 0.05 and rms residual < 0.05 |
| Spread of a k-average is V/k with the finite-pool correction | k_eff / prediction in [0.85, 1.15] for k <= 32 |
| A demographic cohort explains little | cohort R2 < 0.15 |
| kNN is U-shaped with best k between 4 and 32 | interior minimum and best k in [4, 32] |
| The fellow eye loses to the population mean | fellow-eye r < 0.5 |
| The other visit (fundus) and the repeat scan (brain) beat the mean | r > 0.5 |
| A sibling loses to the population mean | sibling r < 0.5 |

**Sensitive**: the direction of the conclusion is kept in every variant but some value leaves the band.
**Breaks**: the direction flips in at least one variant (for example r crosses 0.5, or the kNN minimum moves to an
edge). The other-hemisphere rung is judged per region, because the parent results already differ by region.

## Outputs

- Private, `C:\Projects\data\projects\identity-ladder-metric\donor-ladder-robustness-20261102-v1\`: per-variant
  rows, the SyN outputs and the subset list.
- Public: `results/donor_ladder_robustness.json` and `results/donor_ladder_robustness.csv` (aggregates only),
  `results/figures/fig_donor_ladder_robustness.*`, and the verdicts in
  `results/donor_ladder_robustness_verdict.md`. No IDs, file names or images.
