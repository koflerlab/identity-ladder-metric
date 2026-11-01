# Decision trees for the real-case analysis

Every analysis choice of the real-data cases, with its branches, the proposed branch and what would make us switch.
Written 2026-11-01, before any real-data result of this analysis was computed. The chosen branches are frozen in
`docs/PROTOCOL_real_cases.md` once Kate approves it; a later change is a dated amendment there, never a silent edit.

Facts the trees rely on (QTIM, OpenNeuro ds004169 v1.0.8; aggregates from `participants.tsv` and the dataset README,
see `docs/DATASETS.md`):

- one T1w scan per session, so no same-session repeat (no L0 rung);
- the T1w acquisition orientation changed during the study (coronal for most, sagittal for the rest), and the README
  warns that a vascular pulsation artefact varies with orientation;
- no zygosity in the public release; same-age relatives are probable co-twins, different-age relatives are certain
  non-twin siblings;
- test-retest subsample: 78 adults aged 20 to 28, interval about 3.5 months; 47 of them coronal in both sessions;
- 125 participants in 50 families have a different-age relative, both scanned coronal at session 1.

## D1. Which rungs, from which targets

- Target set S (siblings): participants with a different-age relative, both coronal at session 1. Rungs: other
  hemisphere, sibling, strangers, population mean, mean with restored spread, controlled shrinkage.
- Target set R (repeat): test-retest participants coronal in both sessions. Rungs: repeat scan (about 3.5 months),
  other hemisphere, strangers, population mean, mean with restored spread, controlled shrinkage.
- Not used: same-session repeat (not acquired); later visit after 3.5 years (only adolescents, ages 12 to 17, a
  different population from the adult pools); co-twins of unknown zygosity (see D6).
- Proposed: both sets, analysed and reported separately. One target per family in each set; a person is a target in
  at most one set.

## D2. Stranger reference

- Pool: people from families with no target in either set, so never a relative of any target.
- Orientation:
  - match the target's orientation (proposed). Mixing orientations would add the pulsation artefact to every stranger
    distance and inflate every c.
  - ignore orientation: sensitivity only.
- Session: the stranger's first session.
- Age: adults only (18 or older), the same population as both target sets.
- Sex:
  - not matched (proposed). Affine registration removes overall head size; the pool stays large.
  - sex-matched strangers: sensitivity.
- Number: K = 16 strangers per target, as in the synthetic runs, drawn with a fixed seed from a stranger pool that is
  disjoint from the population-mean pool (D3) and excludes the target's family.

## D3. Population mean

- Pool: M people from other families, adults, coronal at session 1, disjoint from the stranger pool and from all
  targets and relatives.
- Weighting: equal weight per family (families of two would otherwise count twice).
- Leave-target-family-out: automatic, since the pool holds no member of any target family.
- Size:
  - M = 60 (proposed): enough for the mean to lose individual detail, cheap to register.
  - M = 20: sensitivity (a less converged mean keeps more residual individuality).
- Restored spread: the mean inside the mask rescaled to the target's standard deviation there, as in the synthetic
  runs. This uses the target, so it is a diagnostic fill, not a method.

## D4. Mask placement and size

- Shape:
  - 3D box (proposed): the official package scores volumes, and BraTS fills are 3D.
  - a 2D patch on one slice, as in the earlier visual case: kept only for figures.
- Placement:
  - one fixed box in the left parietal cortex, placed on the template before any subject data is opened, at least 95%
    inside the template brain mask (proposed). The box is mirrored across the midline for the other-hemisphere
    donor.
  - several locations (frontal, temporal, occipital, each mirrored): sensitivity.
  - random placements: rejected for the primary, because a score could then depend on a lucky location.
  - The ordering of relatives against strangers is reported to depend on the region (`docs/REPRODUCTION_BACKLOG.md`,
    theme 5, T5-05 and T5-09), so the primary region is fixed here, before any data. Sensitivity boxes are
    reported region by region, never pooled into the primary and never used to choose it.
- Size:
  - 32 x 32 x 24 mm (proposed): about the size of the earlier 32 x 32 mm patch, extended in depth.
  - 16 mm and 40 mm cubes: sensitivity (the share of individual detail inside the mask changes with size).

## D5. Registration and intensity

- Space: every scan resampled to the symmetric MNI ICBM152 2009a template at 1 mm (bundled with nilearn); all fills
  and metrics in template space, linear interpolation.
- Registration:
  - affine, 12 degrees of freedom, ANTs (proposed). It keeps each person's folding, so a donor's tissue is real tissue
    of another brain in the same place, and the population mean is the blurred average that inpainting metrics favour.
  - nonlinear (SyN): sensitivity on set S only. It moves donor folds towards the target and sharpens the mean, so it
    removes part of the identity signal that the ladder measures.
- Bias field: N4 on every scan before registration.
- Intensity scale:
  - each scan divided by its white-matter peak (the upper mode of the intensity histogram inside the template brain
    mask, the evaluation box excluded) (proposed). Donors and the mean then share the target's scale.
  - z-score inside the brain mask: sensitivity.
  - raw intensities: rejected; scanner gain would dominate stranger distances.

## D6. "Family member" without zygosity

- Same-age relative: a probable co-twin, MZ or DZ unknown. A mixture of two rungs with different expected r.
- Different-age relative: a certain non-twin sibling, expected genetic correlation 1/2, as in the model.
- Options:
  - certain non-twin siblings only (proposed). Matches the "sibling" of the ISBI draft; no mixture.
  - all relatives pooled: rejected; the result would depend on the unknown MZ share.
  - infer zygosity from the images: rejected; circular, it would use the quantity under study.
  - request zygosity under a data transfer agreement: for the identity-ladder paper, not for ISBI.
- Which sibling: the one with the smallest age gap; ties broken by a fixed seed. Age gap reported (1 to 7 years).

## D7. QC exclusions

- All QC decisions are made before any fill is scored, and never replaced by score.
- Registration: normalised cross-correlation of each registered scan with the template inside the brain mask.
  Scans below median minus 3 MAD are inspected visually on three orthogonal slices; excluded only on a visible
  failure. Exclusions listed with the reason, without participant IDs in the repository.
- Missing or unreadable file, or a box less than 95% inside the scan's own brain mask: excluded, counted.
- MRIQC image quality metrics: not used to exclude (no prespecified threshold); reported descriptively if downloaded.
- If a target is excluded, the family has no target in that set (no replacement by another member).

## D8. Segmenter (label space) and its failure branch

- Candidates: FastSurfer (Docker `deepmi/fastsurfer`, image digest recorded) and SynthSeg (`mri_synthseg` in
  `freesurfer/freesurfer:8.0.0`). Both run on CPU here. The challenge's own downstream check is described as a U-Net
  segmentation; neither tool is that exact model.
- Choice:
  - FastSurfer segmentation only, if it takes 5 minutes or less per volume on this machine; otherwise SynthSeg
    (proposed). Timing is measured on one scan before any fill is segmented; timing is not a result.
  - the other tool: sensitivity on a subset, if time allows.
- Oracle: the same segmenter on the real target scan, as in the challenge.
- Scores inside the box: Dice per label for labels with at least 50 voxels in the oracle box, their mean, voxel
  accuracy, and Euler number and component count of cortical grey matter.
- Uninformative or biased? Deep-learning segmenters are reported to be robust to degraded input (backlog T3-02,
  T3-07), so a downstream check can fail to penalise blur because it does not see blur at all, not because it
  favours the mean. Branch test, a blur sweep on every label-space target: the real target blurred inside the box
  with a Gaussian of sigma 1, 2, 3 and 4 mm, segmented, Dice against the oracle.
  - Dice falls with sigma (the 95% interval of the drop from no blur to 3 mm lies above 0): the segmenter sees blur;
    label-space results are interpreted (a mean that still scores well is a bias of the metric).
  - Dice does not fall: the segmenter is insensitive here; label space is reported as uninformative for it, not as
    evidence for or against the bias.
- Failure: a crash or empty output is retried once. A fill that still fails is counted as failed for its rung, never
  replaced. If more than 10% of fills fail, the whole label analysis switches to the other segmenter, and both
  failure counts are reported.

## D9. Number of targets and CPU budget

- Machine: 8 CPU cores, 16 GB RAM, no GPU.
- Intensity analysis (cheap): all eligible targets of both sets, after QC.
- Downloads: targets, their siblings or second sessions, the stranger pool (40 people) and the mean pool (60 people):
  about 300 T1w scans, under 4 GB.
- Registration: 4 scans in parallel; the budget is the measured time per scan times about 300.
- Label analysis (expensive): about 24 volumes per target (original, other hemisphere, relative or repeat, 16
  strangers, mean, restored mean, three shrinkage steps, wrong-detail control). The number of label-space targets is
  n_L = floor(8 hours / (24 x measured time per volume)), at least 5, taken in the seeded target order of set S; the
  rest of the targets are scored on intensity only.
- If the budget does not allow n_L of 5: label space is reported as not run, not cut down silently.

## D10. Statistics and tracking

- Unit: the target. One target per family, so targets are independent families.
- Uncertainty: bootstrap over targets, 2000 resamples, percentile 95% intervals, fixed seed.
- Tracking: as in `icml-repro/verdict_audit`, no experiment tracker. A frozen protocol, a run script that refuses to
  run when an input hash differs, and a `manifest.json` with protocol, code and input hashes, git commit, seed and
  package versions. icml-repro uses trackio only to publish per-claim logbook pages for the ICML challenge; nothing
  here is published that way.

## D11. BraTS challenge models as an extra arm

Facts as checked by the parent session from the `brats` 0.1.14 package (Apache-2.0, Docker backend) on 2026-11-01:
15 inpainting algorithms from 2023 to 2026; `Inpainter(...).infer_single(t1n=<voided T1n>, mask=<mask>,
output_file=...)`; the orchestrator refuses any algorithm not marked CPU-compatible, so without a GPU only
BraTS23_1, BraTS24_1 and BraTS23_3 can run here. Not re-verified in this session.

- Why: on healthy QTIM brains with our own box, the ground truth is real tissue, so no Synapse data is needed, and
  the challenge models' fills become rows next to repeat, other hemisphere, sibling, stranger and population mean.
- Models: decided by Kate on 2026-11-01 ("CPU models only"): BraTS23_1 (Zhang, 2023 rank 1), BraTS24_1 (Zhang,
  2024 rank 1) and BraTS23_3 (Huo, 2023 rank 3), run with `force_cpu=True` on this machine. The 12 GPU-only
  entries (the diffusion models and every 2025 and 2026 entry) are out of scope and named as not tested.
- Preprocessing: the BraTS pipeline of the same package (`brats[preprocessing]`: registration and skull stripping
  as the models expect) on each target scan.
- Evaluation space:
  - the target's BraTS space (proposed). Every other fill (donors, mean, shrinkage) is brought there with exactly one
    resampling (composed transforms), so all fills pass through the same number of interpolations; the box is
    mapped there with nearest neighbour.
  - our template space with the model output resampled back: rejected; it adds an interpolation, a smoothing, to
    the model fills only.
- Input: the target T1n with the box voided, as in the challenge.
- Scoring: all nine metrics, c against the same 16 strangers brought into that space, distance to the mean and
  target projection; label space only if the D9 budget allows.
- CPU budget per fill: one inference (one target, one model) may take at most 20 minutes of wall time; one at a
  time, with all 8 cores. The time per fill is measured on the first target of set S for each model before the
  others are run (timing is not a result).
  - first fill within 20 minutes: n_B = min(49, floor(4 hours / measured time per fill)) targets for that model, in
    the seeded order of set S, at least 5.
  - first fill over 20 minutes, or n_B below 5: the model is reported as not feasible on this machine, with its
    measured time, and not run further.
  - a later fill over 20 minutes is stopped and counted as timed out for its model, never replaced.
- Failure: a failed inference is retried once, then counted for its model, never replaced.
- Limits: QTIM scanner and healthy young adults against BraTS training data (glioma patients, many centres): a
  domain shift for the models, reported as such. Pulling the model images is a download that needs Kate's yes, with
  sizes stated.
- Next, not in this protocol: FastSurfer-LIT (backlog T3-01) and fastWDM3D (T2-09), the most direct published
  inpainting comparators.

## D12. Realism measures

- Fréchet-type distances with medical feature extractors, or FID: rejected as a fix here. Their value depends on
  the feature extractor and they are biased at small sample sizes (backlog T6-02, T6-03); n here is at most 49.
- Instead: distance to the mean and target projection (D10, `PROTOCOL_real_cases.md`), which need no sample of
  fills.
