# Protocol: the identity ladder on real 2D images (bilateral, repeat and ageing rungs)

Status: FROZEN 2 Nov 2026 on Kate's approval of draft 2 ("Approve, freeze it"), before any pixel was read
or any distance or score was computed on real data under this protocol. Any later change is a dated amendment, and
results computed before an amendment are never relabelled. Git commit and file hashes give provenance, not external
preregistration. Scope: the next paper (the identity ladder). Nothing here goes into the ISBI 2027 paper.

Change from draft 1 (Kate, 2 Nov): ChestX-ray14 is dropped. A chest film is one midline image with no homologous
left and right halves, its follow-ups are mostly driven by illness, and positioning varies strongly. Datasets are
chosen for a bilateral rung together with a repeat or ageing rung.

## Question

When two real 2D images are compared, how far up the identity ladder does their distance place them? The ladder is
calibrated against nuisance-matched strangers: 0.5 means "as close as a typical matched stranger", 1 means "closer
than every matched stranger". Sub-questions:

1. Does the same eye stay above the stranger level across visits months and years apart, and across a camera change?
   Where does the fellow eye (the bilateral rung) sit relative to them?
2. Where does a population-mean image sit on the same ladder, and how does the score move along the shrinkage path
   from the person's own image to the mean? A representation that places the mean above the person rewards
   regression to the mean.
3. How much of each rung is explained by capture nuisance alone (grey-level mean and spread)?
4. Does a label space (vessel segmentation and its topology) penalise the mean where pixel spaces do not?

Open 2D data with twins do not exist (every twin set is by request), so this protocol has no family rungs.

## Inputs

| Input | SHA-256 | Role |
|---|---|---|
| GRAPE `grape_img_info_w_file.csv` (re-identification repository table, as in the pilot) | `bc79566f4687f7512628ee26959eca0fdb11dd859d217ebf82be99035fd2e02e` | person, eye, visit time, camera |
| GRAPE CFP images (figshare collection 6406319 v1, registry `grape-fundus`, 631 images, 144 people) | per-file SHA-256 written to the manifest at the start of the run | fundus images |
| GRAPE RETFound features released with the re-identification repository | as in the pilot | representation R, GRAPE only |
| UPOL (registry `upol-iris`, 384 images, 64 people, 3 per eye) | per-file, as above | iris, bilateral and same-sitting repeat |
| UTIRIS near-infrared images (registry `utiris`, 778 images, 78 people; 15 files with conflicting eye labels excluded by the existing loader) | per-file, as above | iris, bilateral and same-sitting repeat |
| `selection_grape.json` (private, data root) | `d3c236d60d125f886ea93ed2804a815b0007e12ccc1082c01b36c2d1a7a4708a` | GRAPE pairs, references, mean pools |
| `selection_iris.json` (private, data root) | `c3a41f09ac7d622c8215788b99c2d67ce1ecfc729d05ba33aab1b319240e107c` | iris pairs, references, mean pools |
| DINOv2 ViT-S/14, Hugging Face `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` (Apache-2.0), `model.safetensors` (88,249,960 bytes; matches the Hugging Face LFS record) | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`; `config.json` `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` | general-purpose representation D |

The selection files hold image names and person IDs and stay in the data root. They were produced by
`src/ladder2d_select.py` from metadata and file names only (seed 20261101). The run script refuses to start if the
metadata, selection or model hash differs, and it records the SHA-256 of every image it reads.

## Definitions

**Units.** The unit of identity is the eye for same-eye rungs and the person for the fellow-eye rung. Clusters for
uncertainty are people.

**Rungs.** Each pair has an anchor x and a candidate y. One pair per unit per rung, drawn with the seed among all
eligible pairs.
- GRAPE (gap from Interval Years): `same_camera_<1y` (95 pairs, 53 people), `same_camera_1y+` (71, 37),
  `other_camera` (same eye, any gap, 152, 83), `fellow_eye_same_visit` (same date and camera, 119, 119). Anchor:
  the earlier visit; for the fellow eye, the right eye.
- UPOL and UTIRIS NIR (one sitting each, no dates): `same_eye_same_sitting` (L0: UPOL 128 pairs, UTIRIS 156) and
  `fellow_eye_same_sitting` (UPOL 64, UTIRIS 78). Anchor: first image in file order; for the fellow eye, the right
  eye.
- Population mean: for every pair, the candidate is replaced by the pixel-wise mean of the base images of M = 16
  other people matched to y. The mean is an image and goes through every representation like any image.
- Shrinkage path (GRAPE and iris): y_lambda = mu + lambda (y - mu) for lambda in {1, 0.75, 0.5, 0.25, 0}, computed on
  base images, where mu is the pair's mean image. lambda = 1 is the own rung, lambda = 0 the mean rung.

**Nuisance-matched strangers.** For each pair, one image from each of up to M + K other people (K = 200, in practice
all eligible people) matched to y is drawn; the first M make the mean, the rest are the reference, so mean members
and reference strangers are disjoint people and the pair's own person is never among them. Match keys: GRAPE, same
camera and same eye side as y (no age match; too few people); iris, same dataset, sensor and eye side as y. A pair
needs at least M + 30 matched people, otherwise it is dropped and counted (no pair was dropped in the selection).
The reference of a pair is the set of distances d(x, s) over its K strangers; the same reference scores the own
rung, every point of the shrinkage path and the mean.

**Base image.** Read the file, centre-crop to a square on the shorter side, resize to 224 x 224 with the PIL BOX
filter, scale to [0, 1]; keep RGB (fundus) or grey (iris). No registration: positioning and eye rotation between
captures are part of the within-person nuisance.

**Representations (all prespecified).**
- P (pixels, primary): grey base image (ITU-R 601 luma) resized to 64 x 64 (BOX), z-scored per image. Euclidean.
- D (DINOv2 [1]): base image (grey replicated to three channels), ImageNet normalisation, `pooler_output` of
  `facebook/dinov2-small`; 1 minus cosine similarity. Four CPU threads.
- N (nuisance only, negative control): mean and SD of the grey base image, each standardised over the dataset.
  Euclidean. It carries no anatomy.
- R (GRAPE only, secondary): released RETFound features, as in the pilot. Own rungs only (a mean or path image
  cannot be embedded with released features).
- V (GRAPE only, label space): vessel segmentation, computed by a fixed classical pipeline so that nothing is trained
  on these images: green channel of the 512 x 512 base image (same crop, BOX resize), field-of-view mask = red
  channel above 0.1, CLAHE (`skimage.exposure.equalize_adapthist`, defaults), Frangi vesselness
  (`skimage.filters.frangi`, sigmas 1 to 4, `black_ridges=True`) inside the mask, vessels = the top 12% of
  vesselness inside the mask, objects under 30 pixels removed. Distances between two segmentations:
  1 minus Dice; and, after skeletonisation (`skimage.morphology.skeletonize`), the absolute difference in the number
  of connected components and in the number of branch points (skeleton pixels with at least three 8-neighbours).
  Each distance is calibrated separately. Segmentations are not checked against manual labels (GRAPE has none);
  they are an instrument, not ground truth.

**Calibrated score.** c(d) = P(d_stranger > d) + 0.5 P(d_stranger = d) over the pair's reference
(`grape_calibration.calibrated_score`). 1 - c is the empirical stranger P-value of the pair, in the sense of the
random-pair significance of TM-score [2]; resolution 1/K. Mean c over a rung is the AUC of that rung against its
matched strangers.

**Path diagnostics (P space, descriptive).** For every pair and lambda: d(x, y_lambda), d(y_lambda, mu), d(x, mu),
the residual magnitude ||y_lambda - mu|| / ||y - mu||, the target projection
<x - mu, y_lambda - mu> / ||x - mu||^2 and the residual cosine. Linear identities hold by construction before
z-scoring; they are recorded, not tested.

**Uncertainty.** Person-cluster bootstrap, 2,000 resamples, percentile 95% intervals, seed 20261101. Paired contrasts
use the pair-level difference within the bootstrap.

## Claims under test

Primary: GRAPE, representations P and D. Iris sets: secondary (see evidence boundary).

**C1 Persistence.** Mean c with interval for every rung.
**C2 Time and camera.** Differences `same_camera_<1y` minus `same_camera_1y+` and `same_camera_<1y` minus
`other_camera` (unpaired, person bootstrap).
**C3 Bilateral rung.** Difference `fellow_eye_same_visit` minus `same_camera_<1y`. Iris: `fellow_eye_same_sitting`
minus `same_eye_same_sitting`.
**C4 Nuisance control.** For each rung, mean c under N and the paired difference c_R minus c_N for R in {P, D}.
**C5 Population mean and path.** Mean c of the mean rung, mean c at each lambda, and the paired difference c_mean
minus c_own for each rung. Primary contrast: `same_camera_1y+`.
**C6 Label space.** C1 and C5 repeated in V (Dice and the two topology distances).

## Prespecified interpretation

- "Identity persists on rung r in representation R" only if the lower 95% bound of mean c is above 0.5 AND the lower
  bound of c_R minus c_N is above 0.
- "Time (or camera) lowers the score" only if the lower bound of the C2 difference is above 0; otherwise "no
  detectable effect".
- "R rewards the population mean" if the lower bound of c_mean minus c_own on `same_camera_1y+` is above 0; "R places
  the mean below the person" if the upper bound is below 0; otherwise "not separated".
- "The label space penalises the mean" if the V result is "below the person" while P is "rewards" or "not separated".
- A rung whose c_N has a lower bound above 0.5 is flagged "nuisance-driven" whatever R gives.
- No threshold, filter parameter or representation is tuned after looking at scores. Number of claims: six, over up to
  five representations; no multiplicity correction; the number of intervals reported is stated.

## Sensitivity analyses (reported separately, never substituted for the primary)

- S1 Fellow eye flipped horizontally before comparison (P and V).
- S2 Pixel resolution 32 x 32 and 16 x 16 (P), the second matching the pilot.
- S3 Sample invariance (in the spirit of KING's invariance to the sample [3]): the reference split into two random
  halves of its strangers; report rung means under each half and their difference.
- S4 Reference pooled over the eye side (camera matched only), to show what side matching changes.
- S5 Vessel threshold at the top 8% and 16% instead of 12% (V).

## Outputs

`results/ladder2d.json`: aggregates only (per dataset, representation and rung: n pairs, n people, mean and median c
with intervals, tolerance curve at c >= 0.5, 0.9, 0.95, 0.99, paired contrasts, path means per lambda).
`results/ladder2d_manifest.json`: protocol, code, input and image-set SHA-256, git commit, seed, package versions,
thread count, run time. Per-pair scores, image names, segmentations and any image stay in
`C:\Projects\data\projects\identity-ladder-metric\run-2026-11-02-2d-ladder\`.

## Evidence boundary

- GRAPE has no same-visit repeat of an eye, so there is no L0 floor on fundus: the closest same-eye rung is months
  apart. The iris sets give an L0 floor, but on a different organ and capture.
- GRAPE was analysed in the 1 Nov pilot (`results/grape_calibration.json`: pixels 16 x 16, RETFound, flipped fellow
  eye, camera-pair references) and the iris sets in `results/iris_calibration.json`. The rungs and their rough levels
  are therefore known. New under this protocol: substitution references matched to the candidate, the nuisance
  control on GRAPE, the mean rung and shrinkage path, the label space, and DINOv2. Own-rung results are a partly
  unblinded replication; the mean, path, label-space and DINOv2 results are first looks.
- GRAPE is a glaucoma follow-up cohort from one hospital; visits and camera changes follow clinical care, and the
  camera change is confounded with time. Iris sets: one sitting each, small (64 and 79 people), UTIRIS has known eye
  label problems.
- No registration. Neither P, D nor V is an identity metric; they are the instruments being tested. The vessel
  pipeline is classical and unvalidated on GRAPE.
- Daugman's degrees-of-freedom estimate [4] applies to fractional Hamming distances of binary codes; it is not
  computed for these continuous distances.

## Ethics and data

- GRAPE: CC0 1.0 (figshare API, checked 2026-10-23); data paper [5]. UPOL and UTIRIS: terms as recorded in the
  registry. All are public and de-identified by their providers.
- Secondary analysis of public de-identified images. Whether Kate's institution requires an exemption determination
  for this use: `[to confirm with Kate]`.
- Real images, image names, person IDs, segmentations and per-pair results never enter this public repository.

## Deferred datasets (need Kate's action; added by amendment)

- UQ longitudinal 3D-TBP tile and dermoscopy dataset (doi:10.48610/a13deaf): both body sides and the same lesions at
  6-monthly visits. Listed as open, licence "Other" with a UQ data reuse agreement; the eSpace page shows a
  human-verification check that only Kate can pass.
- SpeakingFaces (CC BY 4.0): face halves and two sessions on separate days; access is requested through Hugging Face.
- Not used: RIDB (eye split and session timing not stated, so neither a clean bilateral nor a clean repeat rung);
  PAPILA (bilateral only, no repeat); ChestX-ray14 (see the change note above).

## Amendment 1 (2 Nov 2026): BraTS inpainting metric baseline (descriptive)

Requested by Kate after the primary results were committed (98d16f8) and before any of the scores below were
computed: how close test-retest and mirrored patches are under the BraTS inpainting metrics, per dataset. Descriptive;
it replaces nothing above and adds no claim.

- Truth: the anchor x as a grey base image (224 x 224, as above). Mask: one fixed centred square of 64 x 64 pixels
  (rows and columns 80 to 143) for every dataset. On iris this square covers mostly the pupil and inner iris; on
  fundus the centre of the field between disc and macula. Fixed now, not tuned.
- Fill: the candidate-slot image of the same pair, inserted inside the mask only. Rows: same-sitting repeat (iris
  L0), other visit (GRAPE same camera under 1 year and 1 year or more, and other camera), mirrored other side (the
  fellow eye flipped left-right, as in S1, with the strangers flipped the same way), population mean (lambda = 0),
  and strangers (the pair's own reference set).
- Scores: all nine outputs of the official `inpainting` package through `toy_ladder.panel` (RMSE, MSE, MAE, MSLE,
  PSNR, PSNR_eps, PSNR_01, PSNR_01_eps, SSIM), with the package's normalisation from the voided truth. Report raw
  medians with interquartile range per dataset and row, and c against the pair's matched strangers with the
  direction of each metric (higher is better for SSIM and the PSNR family) and half credit for ties; mean c with the
  person-cluster bootstrap interval as above.
- Output: `results/ladder2d_brats_baseline.json` and `.csv` (aggregates only) and a short figure-free summary
  `results/ladder2d_brats_baseline.md`. A column for the QTIM MRI run (repeat 3.5 months, other hemisphere) is left
  empty, to be filled from that run's own aggregate output, not recomputed here.
- SpeakingFaces is added to this table by a later amendment once its frame subset is downloaded and selected.

## Amendment 2 (2 Nov 2026): registration before patch comparison

Kate, after looking at the real figures: the unregistered numbers are distorted by misalignment. Measured before this
amendment, on 60 same-sitting UTIRIS pairs: the pupil centre moves by a median of 32 pixels of 224 between two captures
of the same eye, half the width of the 64-pixel patch. Results computed without registration (the primary run and
Amendment 1) are kept and labelled "unregistered"; they are not relabelled. From here on, every patch comparison is
made after registration; this applies to the Amendment 1 baseline and the real-data figures first, and to a rerun of
the primary ladder second.

- Iris (UPOL, UTIRIS): each image is mapped to one canonical frame with the existing segmentation
  (`iris_codes.segment`, integro-differential pupil and limbus circles): limbus centre to the image centre, limbus
  radius to 80 pixels, 224 x 224 output. Images failing the existing checks (pupil/iris ratio 0.15 to 0.75, centre
  offset at most 0.25 of the iris radius, or a segmentation error) are excluded and counted; a pair whose anchor or
  candidate fails is dropped; strangers and mean members that fail are skipped (a pair keeps at least 30 strangers and
  8 mean members, or is dropped). The fellow eye is flipped in the canonical frame. Patch: rows 80 to 143, columns 136
  to 199 of the canonical image, which covers the pupil margin, the iris band and the limbus on one side, instead of
  the centred square that lay on the pupil.
- Fundus (GRAPE): every candidate-slot image (own later image, fellow eye after flipping, each stranger, each mean
  member) is registered to the anchor at 512 pixels: ORB keypoints (800) on the CLAHE green channel inside the field
  of view eroded by 15 pixels, cross-checked matches with ratio 0.8, RANSAC similarity transform (residual 3 pixels,
  500 trials, fixed seed). Accepted if at least 12 inliers, scale 0.8 to 1.25 and rotation under 20 degrees;
  otherwise the image is used unregistered and counted. The registered 512 image is resized to the 224 base (BOX).
  Patch: the centred square, as before.
- The population mean is the mean of registered members. Metrics, c, strangers and bootstrap as in Amendment 1.

## References

1. Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, et al. DINOv2: learning robust visual features without supervision [Preprint]. arXiv:2304.07193. 2023.
2. Xu J, Zhang Y. How significant is a protein structure similarity with TM-score = 0.5? Bioinformatics. 2010;26(7):889-95. doi:10.1093/bioinformatics/btq066
3. Manichaikul A, Mychaleckyj JC, Rich SS, Daly K, Sale M, Chen WM. Robust relationship inference in genome-wide association studies. Bioinformatics. 2010;26(22):2867-73. doi:10.1093/bioinformatics/btq559
4. Daugman J. How iris recognition works. IEEE Trans Circuits Syst Video Technol. 2004;14(1):21-30. doi:10.1109/TCSVT.2003.818350
5. Huang X, Kong X, Shen Z, Ouyang J, Li Y, Jin K, et al. GRAPE: a multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma management. Sci Data. 2023;10(1):520. doi:10.1038/s41597-023-02424-4
