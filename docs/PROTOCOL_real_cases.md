# Protocol: real cases under the regression-to-the-mean results

Status: DRAFT for Kate's approval, written 2026-11-01. Not frozen. No real-data result of this analysis has been
computed: only metadata (`participants.tsv`) and the template were read. The protocol is frozen by Kate's approval and
the commit that records it; git history gives provenance, not external preregistration. Choices and their rejected
branches: `docs/DECISION_TREES.md`. Datasets: `docs/DATASETS.md`.

## Question

On real T1w MRI, does the population-mean fill outrank real tissue from the person's own sibling under the official
BraTS inpainting metrics, although the sibling shares part of the individual anatomy and the mean shares none? And do
the other synthetic results of the ISBI draft (repeat scan, other hemisphere, stranger, mean, mean with restored
spread, controlled shrinkage, label space, all nine metrics) hold on real cases? The unit of evidence is one target
person, one per family.

## Inputs

| Input | SHA-256 | Role |
|---|---|---|
| QTIM `participants.tsv` (OpenNeuro ds004169 v1.0.8) | `a013daf4672804d4f5a837ce75f2b0a52b67270c7ce4d09d1d01dc2c5aeff582` | metadata for the selection |
| QTIM `dataset_description.json` | `0f1dc04f0d8e777ae3ce9e264fb621c4227c9b6c842d5a4898e85a8a9f5ecc61` | version and licence (CC0) |
| `selection.json` (private: holds participant IDs) | `cb264436a1346f6a83b17e7d3abd8f3bc27ac10174ad89c8d7339eec421f7f31` | targets and pools, made by `src/real_cases_select.py`, seed 20261101 |
| MNI ICBM152 2009a symmetric T1, nilearn 0.13.1 | `421a10e872fd6cadae7f61d358dffbcc1795a497d61ee76c5dda2503e1a1e9e6` | registration target |
| same, grey matter probability | `97a5ca69bd24db37a9cb7b32525e1733a209af904129bf1cd36da06d24243bed` | brain mask source (`load_mni152_brain_mask`) |
| same, white matter probability | `382d92812de4744f9c86c7a0e4f680dc317a0a50e4da1f0153618a6798c7b7db` | brain mask source |
| scan manifest (private): SHA-256 of each of the 244 T1w files | recorded in Amendment 1, after download and before any fill is scored | images |

Software: inpainting 0.0.12 (official package), torchmetrics 1.9.0, antspyx 0.6.3, nilearn 0.13.1, nibabel 5.4.2,
numpy 1.26.4, Python 3.11. Segmenters: `deepmi/fastsurfer@sha256:8aa18bae0a287a8925cb26c8903e77043880f21911f2e08821253e41a54f8e7f`,
`freesurfer/freesurfer@sha256:f3e22b472afd9ca2b269d1b40c061ac891efaec070d09e387f4b024ea681e266` (8.0.0, `mri_synthseg`).

The run script `src/real_cases.py` (written after approval) refuses to run if any hash above, or the scan manifest
hash of Amendment 1, differs. It writes `results/real_cases.json` (aggregates) and `results/real_cases_manifest.json`
with the protocol, code and input hashes, git commit, seed, package versions and run time. Per-target scores, fills
and QC images go to the private project folder of the data root only.

## Selection (already run on metadata only)

`src/real_cases_select.py`, seed 20261101. Adults (18 or older) scanned coronal at session 1.

- Set S (siblings): one target per family that has a different-age relative, so a certain non-twin sibling; the
  sibling is the relative closest in age. 49 targets, age gaps 1 to 7 years.
- Set R (repeat): test-retest participants coronal in both sessions, one per family, not selected in set S. 23
  targets, second scan about 3.5 months later.
- Stranger pool: 40 people, one per family, from families with no selected person.
- Mean pool: 60 people, one per family, from further such families (equal family weights by construction).
- 244 T1w scans to download, about 3 GB.

## Definitions

- Space: every scan N4-corrected, registered affinely (12 degrees of freedom, ANTs) to the template, resampled to
  its 1 mm grid with linear interpolation, then divided by its white-matter peak (upper mode of the intensity
  histogram inside the template brain mask, box excluded).
- Box: x [-45, -13), y [-65, -33), z [43, 67) mm in template space (left parietal cortex), 32 x 32 x 24 mm, 97.0% of
  its voxels inside the template brain mask. Mirrored box: x [13, 45), same y and z, 97.6% inside.
- Fill f of target x: x outside the box, the donor's intensities inside it. Donors:
  - repeat (set R): the target's second scan;
  - other hemisphere: the target's own mirrored box, flipped left to right;
  - sibling (set S): the sibling's scan;
  - stranger k = 1..16: 16 people of the stranger pool, drawn with the seed;
  - population mean: voxel mean of the 60 mean-pool scans;
  - mean with restored spread: the mean inside the box rescaled to the target's standard deviation there;
  - shrinkage y_lambda = mu + lambda (x - mu) for lambda in {0.75, 0.5, 0.25} (lambda 1 is the target, 0 the mean);
  - wrong detail: stranger 1's residual from the mean, made orthogonal to x - mu and scaled to the norm of x - mu.
- Metrics: all nine outputs of `inpainting.challenge_metrics_2023.generate_metrics` on the 3D volume, with
  `normalization_tensor` = target outside the box, as in the synthetic runs.
- Calibrated score per target and metric: c = P(d_str > d) + P(d_str = d) / 2 against the 16 stranger fills, oriented
  so that higher c means closer to the target.
- Error ratio: rho = MSE(fill) / mean MSE of the 16 stranger fills, per target. For the sibling, r = 1 - rho is the
  share of the target's individual part the sibling carries under the model.
- Distance-to-mean ratio ||f - mu|| / ||x - mu|| and target projection <f - mu, x - mu> / ||x - mu||^2 inside the box.
- Label space: segmenter chosen by the rule in `DECISION_TREES.md` D8, applied to every fill and to the real target
  (the oracle). Inside the box: Dice per label with at least 50 oracle voxels, their mean, voxel accuracy, Euler number
  and component count of cortical grey matter. c as above against the stranger fills.
- Uncertainty: bootstrap over targets, 2000 resamples, percentile 95% intervals, seed 20261101.

## Claims under test

Primary (set S, intensity, RMSE):

- **C1 Sibling against the mean.** Per target, delta = c_mean - c_sibling under RMSE. Report the mean delta with its
  interval and the share of targets where the mean's RMSE is below the sibling's.
- **C2 Sibling against strangers.** c_sibling under RMSE with its interval, rho_sibling and r = 1 - rho_sibling.

Secondary (descriptive, both sets unless stated):

- **C3** rho of the population mean, against the model value 1/2 (1 + 1/60) = 0.508.
- **C4** Repeat scan (set R): c_repeat and c_mean under RMSE, paired difference.
- **C5** Other hemisphere: c and rho.
- **C6** Mean with restored spread: c against the plain mean.
- **C7** Controlled shrinkage: c for each lambda and for the wrong-detail control; distance-to-mean ratio and target
  projection (these equal lambda by construction and are reported as a check, not as evidence).
- **C8** All nine metrics: mean c of every fill under every metric; Kendall tau of the within-target ranking of each
  metric against RMSE.
- **C9** Label space (the first n_L targets of set S in seeded order, D9): c under Dice, accuracy and topology for the
  mean, the sibling and the other hemisphere.

## Prespecified interpretation

- C1 is called "the population mean outranks the sibling under RMSE" if the 95% interval of the mean delta lies above
  0, and "the sibling outranks the mean" if it lies below 0. Otherwise: "no difference resolved", with the numbers.
- C2 is called "the sibling is closer than a stranger" if the 95% interval of c_sibling lies above 0.5.
- Kate's question ("is the sibling phenotypically closer than the mean?") is answered "the sibling carries individual
  anatomy, yet RMSE ranks it behind the mean" only if both labels hold.
- Two primary claims; both are reported whatever the outcome; no correction beyond stating that there are two.
- Every other claim is descriptive and gets no label. Thresholds are this protocol's own conventions, fixed now.

## Sensitivity analyses (reported apart, never replacing the primary)

1. Strangers not matched on orientation (sagittal strangers allowed).
2. Sex-matched strangers.
3. Mean pool of 20 (the first 20 of the mean pool).
4. Nonlinear registration (SyN), set S only.
5. z-scored intensities inside the brain mask instead of the white-matter peak.
6. Other boxes: frontal, temporal and occipital, each mirrored, and 16 mm and 40 mm cubes at the primary centre;
   coordinates set on the template by the same 95% rule and recorded in an amendment before they are run.
7. The other segmenter on a subset, if time allows.

## Evidence boundary

- One dataset: QTIM young adults, T1w only, one acquisition orientation (coronal). No BraTS data, no challenge
  fills, no tumour-shaped masks.
- Donors are real tissue of real people, not outputs of inpainting methods. The question is what the metrics reward,
  not how any method behaves.
- Affine template space: misregistration adds to every donor's error, the repeat scan's included.
- Siblings are certain non-twin siblings, so the sibling rung includes an age difference of 1 to 7 years. Co-twins
  are not used: their zygosity is not public.
- The segmenter is FastSurfer or SynthSeg, not the challenge's own downstream U-Net.
- n is at most 49 (set S) and 23 (set R) before QC. One box location in the primary.
- Only aggregate results enter the repository. No participant ID, image or per-case table is published.

## Open decisions for Kate before freezing

1. Figures of real QTIM cases: the data are CC0, but the project rule keeps real images out of the public repository.
   Should the paper and the README show one real case (a cropped box, no face), or only aggregate plots?
2. Is "certain non-twin sibling" the right relative for the paper, with co-twins left out?
3. Label space on a subset sized by the CPU budget (D9), at least 5 targets: acceptable for the deadline?

## Amendments

Amendment 1 (planned, before any fill is scored): scan manifest hash after download; measured registration and
segmentation times; the segmenter chosen by the D8 rule and n_L by the D9 rule; QC exclusion counts with reasons.

Any other change is a dated amendment made before the affected results are seen; a change made after is labelled
post hoc and does not replace the prespecified analysis.

Approved by Kate: (date, commit)
