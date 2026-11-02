# Protocol: donor-averaging ladder on fundus images (2 Nov 2026)

Recorded before any result of this experiment was computed.

## Decision

On 2 Nov 2026 Kate lifted the registration-QC pause for this experiment ("Yes, lift it, run"), after being told
that the visual QC gate (`docs/REGISTRATION_QC_PROTOCOL.md`, `docs/REGISTRATION_QC_AMENDMENT_fundus_frame.md`) is
not met. Every result of this experiment is labelled **pre-QC**: the disc-fovea landmarks are automatic and have
not been visually accepted. The pause stays in force for every other metric experiment.

## Question

How does the error of a fill built by averaging k donor images depend on the donor cohort and on k? The theory
predicts error/V = (1 - R2)(1 + 1/k) for a cohort that explains a share R2 of the individual variance, and
2(1 - r) for one same-person donor. No identity score is computed.

## Data

| Dataset | Images | What it adds |
|---|---|---|
| PAPILA | 488 images, 244 patients, both eyes, one device | fellow eye |
| GRAPE | 631 images, 144 subjects, follow-up visits, 3 devices | fellow eye at the same visit, the same eye at another visit, device |

Both datasets are open and already registered in `C:\Projects\data`. Clinical spreadsheets are used for age,
sex and device only. Patient-level outputs stay in the private project folder.

## Preprocessing (fixed)

1. `ladder2d.load_base(path, 512, rgb=True)`, then `ladder2d_register.fundus_canonical` (Amendment 2b
   disc-fovea frame).
2. Images whose landmarks are not found are excluded and counted.
3. Green channel, downsampled to 256.
4. Each image is z-scored inside its own field of view. This removes brightness and gain differences between
   cameras, so cohorts are compared on structure, not exposure.
5. A pixel is valid where it lies inside the field of view.

## Patches (fixed before results)

- 32 patch positions, 48 x 48 pixels on the 256 grid, drawn once with seed 20261102 uniformly from positions
  where at least 95% of the images of the dataset have full field-of-view support.
- The metric mask of the 2D ladder (the 183-329 box at 512, i.e. 91-165 at 256) is reported separately.
- The same positions are used for every target and every donor: no per-image jitter.
- The context of a patch is a 16-pixel ring around it.

## Cohorts

Every cohort excludes the target's own person. "Same side" means the same eye laterality. Fellow-eye and
other-side donors are reflected in the canonical frame.

| Cohort | Donors |
|---|---|
| `population` | all other people, same side |
| `population_both` | all other people, both sides, the other side reflected |
| `cohort` | same side, same sex, age within 5 years (GRAPE: also the same device) |
| `knn` | same side, the k nearest donors by Euclidean distance on the context ring of that patch |
| `fellow_eye` | the same person's other eye, reflected (GRAPE: the same visit) |
| `other_visit` | GRAPE only: the same person and eye at the nearest other visit |

## Measures

- k in {1, 2, 4, 8, 16, 32, 64}, plus all donors for `population` and `cohort`.
- For `population`, `population_both` and `cohort`, the k donors are a random draw (seed fixed) per target.
- For each target, patch and fill:
  - error = mean squared difference over valid pixels;
  - spread = mean squared distance of the fill from the population mean (Proposition 2).
- The target's error is its mean over the 32 patches.
- Normalization: V is the mean error of the full `population` mean, so the population mean has error/V = 1 by
  construction.
- Per cohort: fit (1 - R2) by least squares to error/V = (1 - R2)(1 + 1/k), and report the residual.
- For same-person donors: r = 1 - (error/V)/2.
- k_eff of the population mean at each k, from Proposition 2.
- 95% intervals by bootstrap over persons (2000 resamples).

## Outputs

- Private, `C:\Projects\data\projects\identity-ladder-metric\donor-ladder-fundus-20261102-v1\`: per-target
  table, code snapshot, and the example strip with real images.
- Public: `results/donor_ladder_fundus.json` and the curve figures in `results/figures/`. These hold aggregates
  only: no IDs, file names or images.

## Not done here

No visual QC decision. No identity or re-identification score. No change to the 2D-ladder protocol or to its
frozen selections.

## Amendment 1 (2 Nov 2026, after the first run, before any error value was looked at)

The first run found no 80 x 80 window (48-pixel patch plus 16-pixel ring) with full field-of-view support in at
least 95% of PAPILA images. The best window reached 92%: PAPILA photographs are centred on the optic disc, so the
fovea side lies near the edge of the field. Only coverage counts were inspected, never an error value.

Candidate windows at each support threshold:

| Dataset | 0.95 | 0.90 | 0.85 |
|---|---|---|---|
| PAPILA | 0 | 81 | 216 |
| GRAPE | 26 | 61 | 101 |

The threshold is lowered to 90% for both datasets. Patch size, ring width and the number of patches are
unchanged. Images without full support on a patch are still excluded from that patch, as before. The same run
also fixed a crash in the GRAPE other-visit lookup (missing index column). The first run's PAPILA rows are
overwritten.
