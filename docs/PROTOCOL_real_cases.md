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

**A1.5 Clarifications (2026-11-01, 21:20 +04, before any label-space or expanded result).**

- Budget: the frozen C9 runs as written, on the parietal box with its own D9 budget of 8 hours. A1.2 has a separate
  budget of 8 hours, split equally over its three boxes (parietal, deep grey matter, ventricles), all on the A1.1
  design. A box whose n_L by the D9 formula is below 5 is reported as not run, and the A1.2 label is then not
  assigned. The time per volume in the D9 formula may be measured as throughput with several segmentations in
  parallel.
- Expanded selection: `selection_expanded.json` (private, holds participant IDs), SHA-256
  `fcfca9875f9b895b1d96ac11b357248ab950a972eed7b7b125e878e83b8d3dc6`, made by `src/real_cases_select.py --expanded`
  from the same `participants.tsv`, seed 20261101: 287 targets, 260 with a co-twin, 49 with a sibling, 22 with both,
  24 with a repeat scan; mean pool 179; 826 scans.

**A1.6 Data-QC report (2026-11-01, 21:35 +04, before any fill is scored).** Kate asked for a descriptive report on
every downloaded scan ("Yes, add QC report", relayed). It changes nothing in the analysis and excludes nothing
beyond D7; an exclusion it suggests goes to Kate as a proposed amendment, decided before any fill is scored.

- Per scan: BIDS sidecar fields (TR, TE, TI, flip angle, field strength, scanner model, software version);
  acquisition orientation from `participants.tsv`; header facts (voxel size, matrix, orientation code, data type,
  count of non-finite voxels); SynthStrip brain-mask volume; N4-corrected intensity percentiles 1, 5, 50, 95 and 99
  inside the mask, raw and relative to the white-matter peak; the native white-matter peak; an SNR proxy (mean of
  voxels within 10% of the white-matter peak over the standard deviation of non-zero background below the Otsu
  threshold, more than 10 mm outside the brain); template correlation; box coverage in the scan's own mask.
- Flags: more than 3 MAD from the median on any variable; any departure from the most common acquisition value.
- Outputs: per-scan table, histogram overlay and thumbnails of flagged scans in the private run folder
  (`real-cases-run/data_qc/`); an aggregate summary without IDs in `results/real_cases_data_qc_summary.json`.
- The T1w JSON sidecars of the 244 scans are fetched under the same data-root lock, with their own manifest
  (`sidecar_manifest.json`, private).

**A1.7 Order under the CPU budget and C10 scope (2026-11-01, 21:55 +04, Kate's decisions, before any result).**
About 27 hours of CPU remain before the deadline for about 38 hours of planned work, so the run order is: frozen
C1 to C8 with sensitivities 2, 3 and 5; then A1.1 (C1-all, C2-all, C11, C12); then C10; then frozen C9; then A1.2.
Whatever does not finish is reported as not run. C10 runs only the two rank-1 models, BraTS23_1 and BraTS24_1
(Zhang, Ying-Weng; images `brainles/brats23_inpainting_ying_weng` and `brainles/brats24_inpainting_ying_weng`,
about 9 GB compressed), approved for download by Kate; BraTS23_3 is not run and is added to the evidence boundary's
list of untested models. The preprocessing (`brats` 0.1.14, `brainles-preprocessing` 0.6.14 with HD-BET) goes in a
separate virtual environment, so the frozen environment (numpy 1.26.4, antspyx 0.6.3) is unchanged.

**A1.4 Consequence for the paper (not an analysis).** Proposition 2 is a one-line remark; the contribution is the
measured answer to when and where Dice falls into the mode trap (A1.2, A1.3).

### Amendment 2, part 0: Kate's decisions (2026-11-02, before any fill is scored)

Written by the protocol session from Kate's messages to the run session. State of the run at this point: all 244
frozen scans skull-stripped (SynthStrip) and registered; registration QC computed (6 scans flagged for visual review);
box statistics being recomputed for the new box; no fill made or scored on any box.

**A2.1 One evaluation box, chosen for registration quality.**

- Why the box changed: after registration, the frozen parietal box aligned poorly across people. Mean within-box
  correlation with the template over the mean pool: 0.59 (median over all candidate boxes 0.57).
- Search, run before any fill existed: every 32 x 32 x 24 mm box in the left hemisphere whose box and mirror lie inside
  the template brain mask (221,857 candidates), ranked by the mean within-box correlation with the template over the
  frozen mean pool (59 scans). No target, relative, repeat or stranger scan and no fill entered the search.
- Result: x [-32, 0), y [-19, 13), z [-2, 22) mm, mean r 0.85, 99.996% inside the template brain mask; mirror x [1,
  33), same y and z. The search ran while one mean-pool scan was still being registered; rerun on all 60 as a check
  (box fixed whatever it showed), it returned the same box, mean r 0.853 (parietal 0.592; candidate percentiles 5, 50,
  95: 0.381, 0.568, 0.771). By its coordinates it covers left deep grey matter (thalamus, caudate, putamen), adjacent white
  matter and the edge of the lateral ventricle; it was not segmented.
- Kate's decision: offered the new box as primary with the parietal box kept as a sensitivity, and then five boxes
  fixed before scoring, she answered "ну не делай все пять делай один и смотри по результатам" ("don't do all five,
  do one and look at the results"). The best-aligned box is the primary box for C1 and C2 and for every intensity
  arm: C3 to C8, sensitivities 2, 3 and 5, A1.1 (C1-all, C2-all, C11, C12) and C10.
- What the criterion costs, stated before any result: it picks the place where people are most alike after
  registration, so the place where a person's own deviation from the population is smallest. That favours the
  population mean under RMSE, the direction of C1's first label. The run session told Kate this before she decided.
  A C1 label "the population mean outranks the sibling" on this box is weaker evidence than on a box fixed before
  any data, and it is reported together with this paragraph.
- A box added after any result has been seen is labelled post hoc and does not replace this one.
- Relation to the frozen design (Amendment 1: "reported as written"). Kate was asked, with the cost stated (about
  40 minutes of CPU, 32 of 244 scans lost to D7 on that box), whether the frozen parietal box is also scored, and
  answered "Да, оба бокса" (2026-11-02, about 08:50 +04), and gave the run session the same answer ("Yes, both
  boxes"). A first scoring job on the A2.1 box had been started and was stopped after about 10 minutes, before any
  target finished; no score of any box existed or had been read. So:
  - the frozen parietal box is scored as written: C1 to C8, D7 as written (A2.2), white-matter peak with the parietal
    box excluded, its own mirror x [13, 45). It runs right after the frozen scoring of the A2.1 box, before A1.1;
  - C1 and C2 carry their prespecified labels on both boxes, reported side by side; neither replaces the other. That
    makes four labelled primary results (two claims, two boxes), stated as such, with no correction beyond that;
  - sensitivities 2, 3 and 5, A1.1 and C10 use the A2.1 box only; C9 stays on the parietal box (A1.5);
  - the sentence of part A that running the parietal box later "would be post hoc" is superseded by this decision.
    The parietal box is the frozen primary (19e47a7), and the decision to score it was taken before any result was
    seen by Kate.

**A2.4 Sensitivity 8: without the Siemens Trio scans (Kate, 2026-11-02, about 08:50 +04, "Да, добавить", before any
result was reported).** The data-QC report (part A) found 5 of the 244 scans from a Siemens Trio, 4 of them with
another sequence; all others share one scanner and protocol. On the A2.1 box, the frozen C1 to C8 are repeated
without them: a target whose own, sibling or repeat scan is a Trio scan is dropped (no replacement, as in D7); a Trio
stranger is removed from each target's 16 strangers, and c uses the remaining ones; the mean pool holds no Trio scan.
Reported apart, never replacing the primary.

**A2.2 Box coverage rule of D7.** D7 holds as written: a box less than 95% inside the scan's own brain mask excludes
the scan. On the A2.1 box every scan has 100% coverage (part A), so nothing is excluded and no decision is needed. On
the frozen parietal box 32 of 244 scans fall below 95% (median 97.0%, 5th percentile 94.4%, minimum 91.4%); the rule
applies to it as written. Losses by role: set S targets 6, set S siblings 7, set R targets 5, set R repeats 3,
strangers 6, mean pool 5. Parietal n after D7: set S 37 of 49 (C1, C2), set R 17 of 23, stranger pool 34, mean pool
55. A2.1 box: no losses (49, 23, 40, 60). D7 names the box; the mirror is only the donor region of the other-hemisphere row. So a scan
is excluded by its evaluation box, and a target whose mirror falls below 95% (parietal: 6 of 244, median 97.4%; A2.1
mirror: none) keeps its other rows and loses only the other-hemisphere row (C5), counted.

**A2.3 Label-space budget, a check, no change.** D9 gives n_L = floor(budget / (24 x time per volume)). At the
measured 215 s per SynthSeg volume, the frozen C9 (28,800 s) gets n_L = 5, the minimum. Each A1.2 box (9,600 s) gets
n_L = 1, so by A1.5 A1.2 runs only if the measured throughput with segmentations in parallel reaches 80 s per volume
or less (120 volumes in 9,600 s); otherwise it is reported as not run. Both come last in the A1.7 order.

### Amendment 2, part A: frozen-design inputs, QC and implementation (2026-11-02, before any fill is scored)

Written by the run session. State: 244 scans skull-stripped and registered, box statistics recomputed for the A2.1
box, QC and the data-QC report done; no fill made or scored on any box.

- Scan manifest (244 T1w, OpenNeuro snapshot 1.0.8, size and S3 ETag MD5 verified, 2.69 GB): SHA-256
  `4016dcf4a30ab95c76170ba714598c435792a260d70a39780dd6701928b0dcb8`. Sidecar manifest (244 JSON, data-QC report):
  `96e0cad07beba51f724bc71eeb60435673c50fcb34c5849b68575420bd01f947`. Both private, next to `selection.json`.
- Registration: median 25 s per scan, measured with two registrations and SynthStrip running at once (17 s on one
  scan with the machine otherwise idle). Template correlation 0.41 to 0.58, median 0.51.
- QC (D7): 6 scans below median - 3 MAD (0.448) were reviewed on three orthogonal slices: aligned, darker or
  lower in contrast, no visible failure; none excluded. Box coverage in each scan's own SynthStrip mask: 100% for
  every scan on the A2.1 box, so D7 excludes nothing and A2.2 needs no decision for it (on the parietal box, 32 of 244
  scans would have fallen below 95%, minimum 91.4%; that box is not scored). Exclusions: 0.
- Relation to the frozen design (A2.1, open point): following Kate's "do one", only the A2.1 box is scored; the frozen
  parietal box is not run, and running it later would be post hoc.
- Data-QC report (A1.6), descriptive, nothing excluded: 5 of 244 scans come from a Siemens Trio instead of the 4 T
  MedSpec (2 set S targets, 1 set S sibling, 2 strangers), 4 of them with another sequence (TR 2.5 s, TI 1.1 s, flip
  15 degrees); all others share one protocol. Proposed to Kate as a possible sensitivity (without these scans), not
  applied.
- Implementation details the protocol left open, fixed now:
  - brain masks: SynthStrip (`mri_synthstrip`, pinned FreeSurfer 8.0.0 image) on each native scan; the mask serves the
    affine registration (skull-stripped N4 scan to the 2 mm template, ANTs, template brain mask, seed 123) and is "the
    scan's own brain mask" of D7. The crude Otsu strip of the image ladder covered only 76% of the parietal box on the
    first scan (mask check only, no score);
  - N4 on a 2 mm copy (shrink factor 4), bias field resampled to the native grid; the N4 native scan is resampled once
    to the 1 mm template grid (linear);
  - every volume is multiplied by the template brain mask (skull-stripped input, as in the challenge);
  - white-matter peak: highest-intensity peak of the smoothed (Gaussian, 2 bins) 256-bin histogram between the 0.5 and
    99.5 percentiles with height and prominence at least 10% and 5% of the tallest peak, inside the template brain
    mask with the box excluded (prominence added after a synthetic test where tail noise won);
  - metrics on the whole template-grid volume, tensor shape [1, 197, 233, 189] as in the package's file interface;
  - mirrored box: the exact reflection on the grid (A2.1); restored spread over box voxels inside the template mask;
  - strangers: 16 without replacement from the QC-passing pool, numpy default_rng(20261101), set S targets first in
    selection order, then set R; sex-matched (sensitivity 2) from default_rng([20261101, 2]); mean pool of 20
    (sensitivity 3) = the first 20 in selection order; z-scores (sensitivity 5) inside the template mask, box excluded,
    plus 10 so the package's clip at zero never acts; Kendall tau-b over all fills of a target with its 16 strangers;
  - sensitivity 1 cannot run (no sagittal scan in the frozen download); 4, 6 and 7 are not run in this pass.
- Segmenter timing, the D8 choice and n_L are measured and recorded before any label-space fill is segmented (D8),
  after the intensity arms in the A1.7 order.

### Amendment 3: implementation error in the white-matter peak (2026-11-02, about 10:05 +04, AFTER the first results)

Found after the first results of both boxes were computed (commit 93de63d) and relayed to Kate, while checking an
implausible sensitivity result (the 20-scan mean closer to targets than the 60-scan mean). The run session traced it
to the white-matter-peak rule of part A (highest histogram peak with prominence at least 5% of the tallest): in 77 of
244 scans white matter forms a broad shoulder without prominence, so the rule took the grey-matter peak, and those
scans were scaled by their grey-matter level (box mean about 1.5 to 2.1 instead of about 0.89). Native and template
estimates agree, so the rule failed, not the registration. Affected by role: set S targets 17, siblings 16, set R
targets 11, repeats 11, strangers 10, mean pool 12.

- Consequence: every c, rho and label in 93de63d that uses the white-matter scale is invalid, for both boxes. That
  run stays in the repository, marked withdrawn, and is not cited as a result.
- The protocol's definition (Definitions: "upper mode of the intensity histogram inside the template brain mask, box
  excluded") is unchanged. The implementation did not meet it.
- Fix, approved by Kate before any rescoring ("Одобряю", 2026-11-02, about 10:15 +04): the white-matter peak is the
  mode of the voxels above the two-class Otsu threshold of the brain histogram, inside the template brain mask, box
  excluded; same histogram binning and smoothing as part A. Fixed now, before any rescoring.
- Acceptance, on QC quantities only, never on c, rho or labels: (a) native and template estimates agree; (b) the
  distribution of box means over the 244 scans is no longer bimodal; (c) on every scan, the ratio of the new peak to
  the median intensity inside the template white-matter probability above 0.9 (box excluded) is reported, and every
  scan off by more than 10% is inspected and listed. If a check fails, nothing is scored and the failure goes to Kate
  before any other rule is chosen.
- Then both boxes are rescored, with every arm that uses the white-matter scale (C1 to C8; sensitivities 2, 3 and
  8; the A1.6 values relative to the white-matter peak). The corrected run is labelled a post hoc correction of an
  implementation error, and the withdrawn run is reported as such, with this paragraph. The A2.1 box search and D7
  coverage do not depend on intensity scale and stand.

### Amendment 2 (planned, before any fill is scored)

Scan manifest hashes after download (the original 244 and the expanded 826); the expanded selection file and its
SHA-256; measured registration and segmentation times; the segmenter chosen by the D8 rule and n_L by the D9 rule;
coordinates of the deep grey matter and ventricle boxes (A1.2); QC exclusion counts with reasons; `brats` package
version, model image digests, time per inference and n_B (D11).

Any other change is a dated amendment made before the affected results are seen; a change made after is labelled
post hoc and does not replace the prespecified analysis.

Approved by Kate: 2026-11-01, "Approve as is" (freezing commit: see Status).
