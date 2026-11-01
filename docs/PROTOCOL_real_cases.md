# Protocol: real cases under the regression-to-the-mean results

Status: FROZEN 2026-11-01 on Kate's approval ("Approve as is"), before any scan was downloaded or any real-data
result computed: only metadata (`participants.tsv`) and the template were read. The freezing commit is the first
commit that contains this paragraph; its hash goes into every results manifest. Git history gives provenance, not
external preregistration. Changes from here on are dated amendments at the end of this file. Choices and their rejected
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
  mean, the sibling and the other hemisphere. Preceded by the blur sweep of D8 (target blurred inside the box at
  sigma 1, 2, 3, 4 mm, Dice against the oracle).
- **C10** BraTS challenge models (D11): the fills of BraTS23_1, BraTS24_1 and BraTS23_3 (`force_cpu=True`, at most
  20 minutes per fill, n_B by the D11 budget) on the first n_B targets of set S, in the target's BraTS space, scored like every other fill: all nine metrics, c against the 16 strangers,
  distance to the mean and target projection, placed against the sibling and the population mean of the same
  targets.

## Prespecified interpretation

- C1 is called "the population mean outranks the sibling under RMSE" if the 95% interval of the mean delta lies above
  0, and "the sibling outranks the mean" if it lies below 0. Otherwise: "no difference resolved", with the numbers.
- C2 is called "the sibling is closer than a stranger" if the 95% interval of c_sibling lies above 0.5.
- Kate's question ("is the sibling phenotypically closer than the mean?") is answered "the sibling carries individual
  anatomy, yet RMSE ranks it behind the mean" only if both labels hold.
- Two primary claims; both are reported whatever the outcome; no correction beyond stating that there are two.
- C9 is interpreted only if the blur sweep shows that the segmenter sees blur: the 95% interval of the mean Dice drop
  from the unblurred target to the 3 mm blur lies above 0. Otherwise label space is reported as uninformative for
  that segmenter, not as evidence for or against the bias.
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
- n is at most 49 (set S) and 23 (set R) before QC. One box location in the primary, fixed before any data because
  the ordering of relatives is reported to vary by region.
- The challenge models (C10) see healthy young adults from one scanner, not the glioma patients they were trained on:
  a domain shift for them. Only the three CPU models run (Kate's decision, 2026-11-01): BraTS23_1, BraTS24_1 and
  BraTS23_3. Not tested, so the result does not cover the full leaderboard: BraTS23_2 (Durrer, diffusion),
  BraTS24_2 (Ferreira), BraTS24_3 (Durrer), BraTS25_1A (Zhang), BraTS25_1B (Ferreira), BraTS25_2 (Ha), BraTS26_1
  (Komurcu), BraTS26_2A (Hung), BraTS26_2B (Cap), BraTS26_3 (Persson), BraTS26_4A (Danese) and BraTS26_4B
  (Karimaghaloo). Model list as checked by the parent session from `brats` 0.1.14; not re-verified here.
- No realism measure (FID or Fréchet distances with medical features): too few fills for them (D12).
- Only aggregate results enter the repository. No participant ID, image or per-case table is published.

## Decisions at freezing (2026-11-01)

1. Figures of real QTIM cases: not decided by Kate, so the project rule holds: no real image enters the public
   repository; the paper and README get aggregate numbers and plots only. Showing one real case later needs Kate's
   decision and an amendment.
2. Relative: certain non-twin siblings only, co-twins left out, as written.
3. Label space on a subset sized by the CPU budget (D9), at least 5 targets, as written.
4. BraTS model arm (C10): CPU models only (Kate). Pulling the three model images and the preprocessing pipeline is a
   separate download that Kate approves with the sizes stated, before it starts.

## Amendments

### Amendment 1: the whole dataset, label space by structure, blur asymmetry (2026-11-01, 20:40 +04, before any result)

Approved by Kate ("окей, давай", relayed by the parent session, after she asked for results on the dataset rather
than on one brain). Made before any real-data result was seen. State of the run at this point: the run session had
acquired the data-root lock (16:36 UTC) and was downloading the 244 scans of `selection.json` (11 new scans complete,
6 in progress); its download script was not yet committed; no scan had been registered, filled, segmented or scored.

The frozen design stays as written and is reported as written: set S (49 targets) and set R (23 targets) with their
40-person stranger pool and 60-person mean pool, claims C1 to C10 and their labels. This amendment adds an expanded
analysis next to it; it does not replace the primary.

**A1.1 Expanded intensity analysis (no segmentation).**

- Eligible: adults scanned coronal at session 1 (779 people in 466 families, counted from `participants.tsv`).
- Targets: one per family with at least two eligible members (287 families), chosen with seed 20261101 among
  members who have a relative, preferring a member who has both a same-age and a different-age relative.
- Relative rows: "co-twin, zygosity unknown" (a same-age relative; 260 families) and "non-twin sibling" (the
  different-age relative closest in age; 49 families). Both rows on the same target where a family has both (22).
- Repeat row: the second session, for targets in the test-retest subsample scanned coronal in both sessions.
- Population mean: the 179 single-member families, one person each, fixed, so the mean pool is disjoint from every
  target family. Deviation from the proposed 200 to 300: 287 target families plus disjoint pools of 150 to 200 and
  200 to 300 families would need about 640 families, and QTIM has 466 eligible ones; 179 is the largest fixed mean
  pool that removes no target family.
- Strangers: 16 per target, drawn with the seed from the other 286 multi-member families, never from the target's
  own family and never from the mean pool. Strangers are not globally disjoint from targets (a target of one family
  serves as a stranger to another), so target scores share some strangers; the bootstrap over targets ignores this
  mild dependence, stated as a limitation.
- Fills, metrics, c, rho, distance to the mean, target projection and the bootstrap as in Definitions. Box: the
  primary parietal box.
- Claims, with the same prespecified labels as C1 and C2: **C1-all** and **C2-all** for the sibling on this design;
  **C11** co-twin against the mean (delta c under RMSE) and **C12** co-twin against strangers (c under RMSE, interval
  against 0.5). No sign is predicted for C11: the co-twin row mixes MZ and DZ pairs.
- Downloads: every eligible session-1 scan plus the retest session-2 scans, 826 T1w, about 10 GB at the measured 12.4
  MB per scan, under the data-root lock, registry updated.
- CPU budget: families are processed in a seeded order fixed before any image is opened; if registration and
  scoring do not finish within 12 hours of wall time, the analysis uses the families finished by then, reported with
  their number.

**A1.2 Label space by structure.**

- Prespecified hypothesis: Dice against the oracle favours the population mean in deep grey matter and the
  ventricles, where a strong population prior fixes the boundary, but not in the cortex, where individual folding
  sets it.
- Label groups (names of the FreeSurfer colour table, which both segmenters use): cortex (the whole-cortex label or
  all cortical parcels, whichever the segmenter outputs); cerebral white matter; deep grey matter (thalamus,
  caudate, putamen, pallidum); ventricles (lateral, inferior lateral, third, fourth); hippocampus and amygdala.
- Boxes: the primary parietal box (cortex) plus two 24 mm cubes centred on the left-hemisphere centroid of the deep
  grey matter group and of the lateral ventricle, found by segmenting the template T1 (input table) with the chosen
  segmenter; mirrored for the other hemisphere. Their coordinates go into Amendment 2 before any subject is scored.
- Report: Dice c per label group, for the mean, the sibling, the co-twin and the other hemisphere, paired delta c
  (mean minus relative).
- Labels: "mode trap in deep structures" if the 95% interval of delta c (mean minus sibling, and separately mean
  minus co-twin) lies above 0 in deep grey matter and in the ventricles while the interval for the cortex includes
  0 or lies below it. Otherwise the numbers are reported without the label.
- Budget: within D9 (8 hours of segmentation), split equally over the three boxes; n_L per box by the D9 formula.

**A1.3 Blur asymmetry (extends the D8 blur sweep).**

- In each label-space box, blur the target's true tissue and stranger 1's tissue at sigma 0, 1, 2, 3 and 4 mm,
  inside the box; score both against the target: Dice against the oracle, and RMSE.
- Bias predicts that blur hurts the truth but helps the stranger; insensitivity predicts flat curves for both.
- Labels, on Dice: "bias pattern" if the 95% interval of the truth's drop from sigma 0 to 3 mm lies above 0 and the
  interval of the stranger's gain lies above 0; "insensitive" if both intervals include 0; otherwise "mixed". The
  RMSE curves are reported descriptively (Proposition 1 already predicts that smoothing a stranger lowers its RMSE).
- The D8 rule stays: label-space claims (C9, A1.2) are interpreted only if the truth's Dice falls with blur.

**A1.4 Consequence for the paper (not an analysis).** Proposition 2 is a one-line remark; the contribution is the
measured answer to when and where Dice falls into the mode trap (A1.2, A1.3).

### Amendment 2 (planned, before any fill is scored)

Scan manifest hashes after download (the original 244 and the expanded 826); the expanded selection file and its
SHA-256; measured registration and segmentation times; the segmenter chosen by the D8 rule and n_L by the D9 rule;
coordinates of the deep grey matter and ventricle boxes (A1.2); QC exclusion counts with reasons; `brats` package
version, model image digests, time per inference and n_B (D11).

Any other change is a dated amendment made before the affected results are seen; a change made after is labelled
post hoc and does not replace the prespecified analysis.

Approved by Kate: 2026-11-01, "Approve as is" (freezing commit: see Status).
