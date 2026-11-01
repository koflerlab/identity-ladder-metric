# Protocol: the identity ladder on real 2D images (test-retest and ageing rungs)

Status: DRAFT, 2 Nov 2026. Not frozen. No image has been read and no distance or score has been computed on real
data under this protocol. It is frozen by a commit after Kate's approval; any later change is a dated amendment, and
results computed before an amendment are never relabelled.
Git commit and file hashes give provenance, not external preregistration.
Scope: the next paper (the identity ladder). Nothing here goes into the ISBI 2027 paper.

## Question

When two real 2D medical images are compared, how far up the identity ladder does their distance place them? The
ladder is calibrated against nuisance-matched strangers, so a score of 0.5 means "as close as a typical matched
stranger" and 1 means "closer than every matched stranger". Three sub-questions:

1. Does a person's own image stay above the stranger level months and years later, and does the score fall with the
   time gap (test-retest and ageing rungs)?
2. Where does a population-mean image sit on the same ladder? If a representation places the mean above the person's
   own later image, it rewards regression to the mean and cannot serve as an identity metric without a penalty.
3. How much of each rung is explained by capture nuisance alone (grey-level mean and spread)?

Open 2D data with twins do not exist (every twin set is by request), so this protocol has no family rungs.

## Inputs

| Input | SHA-256 | Role |
|---|---|---|
| ChestX-ray14 `Data_Entry_2017_v2020.csv` (NIH Clinical Center Box, file `f_219760887468`, 9,003,496 bytes) | `c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1` | metadata: patient, follow-up order, age, sex, view, pixel spacing, finding labels |
| ChestX-ray14 `images_001.tar.gz` (2,008,470,987 bytes, MD5 `fe8ed0a6961412fddcbb3603c11b3698` matched) | `[filled at freezing]` | images, first 4,999 in sorted index order |
| ChestX-ray14 `images_002.tar.gz` (3,952,623,504 bytes, MD5 `ab07a2d7cbe6f65ddd97b4ed7bde10bf`) | `[filled at freezing]` | images 5,000 to 14,999 |
| GRAPE `grape_img_info_w_file.csv` (the table of the re-identification repository already used in the pilot) | `bc79566f4687f7512628ee26959eca0fdb11dd859d217ebf82be99035fd2e02e` | GRAPE metadata: patient, eye, visit time, camera |
| GRAPE CFP images (figshare collection 6406319 v1, registry `grape-fundus`) | per-file manifest in the data root | fundus images |
| GRAPE RETFound features (released with the re-identification repository) | as in the pilot | secondary representation, GRAPE only |
| `selection_cxr.json` (private, data root) | `528fcbc4681dab8bea3fb659bff38b0eaded75bfda2d1db01c5e1eb971e4459e` | pairs, references and mean pools, all variants |
| `selection_grape.json` (private, data root) | `fef83ffaa6183779d0b97cf32ebc26901e9e9d564e10c06fa696bda593300526` | same for GRAPE |
| DINOv2 ViT-S/14, Hugging Face `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` (Apache-2.0), `model.safetensors` | `[filled at freezing]` | general-purpose image representation |

The selection files hold image names and patient IDs and stay in the data root. They were produced by
`src/ladder2d_select.py` from metadata only (seed 20261101). The run script refuses to start if any hash differs.

ChestX-ray14 subset: only the first two of the twelve archives were downloaded (6.0 of 45 GB; the disk is shared
with another running job). Patient IDs rise roughly with enrolment time, so these archives hold the patients with
the longest follow-up. The subset is not a random sample of the dataset.

## Definitions

**Units and images.** ChestX-ray14: the unit of identity is the patient. Universe: PA views, age at least 18, in the
two archives (9,431 images, 3,532 patients). AP views are excluded (mostly bedside films with a different geometry).
GRAPE: the unit is the eye for same-eye rungs and the patient for the fellow-eye rung; all 631 images, 144 patients.

**Image preprocessing (fixed, no registration).** Read the file, convert to 8-bit grey (ITU-R 601 luma for colour or
RGBA files), centre-crop to a square on the shorter side, resize to 224 x 224 with the PIL BOX filter, scale to
[0, 1]. This is the base image b. CXR positioning varies between films; that variation is part of the within-person
nuisance and is not removed.

**Representations.**
- P (pixels, primary): b resized to 64 x 64 (BOX), z-scored per image, flattened. Distance: Euclidean.
- D (DINOv2): b replicated to three channels (GRAPE: the RGB image, same crop and resize), ImageNet normalisation,
  `pooler_output` of `facebook/dinov2-small` [1]. Distance: 1 minus cosine similarity. One thread pool of 4 CPU threads.
- N (nuisance only, negative control): the mean and standard deviation of b, each standardised by its mean and SD over
  the dataset universe. Distance: Euclidean. It carries no anatomy.
- R (GRAPE only, secondary): the released RETFound features, as in the pilot. No mean rung (a mean image cannot be
  embedded with released features).

**Pairs and rungs.** Each pair has an anchor x (the earlier image; for the fellow eye, the right eye) and a candidate
y. One pair per unit per rung, drawn with the seed among all eligible pairs.
- ChestX-ray14 (gap = difference of the recorded ages in whole years; there are no acquisition dates):
  `short_0y` gap 0 (less than one year apart, possibly days); `mid_2-4y` gap 2 to 4 (about 1 to 5 years);
  `long_5y+` gap 5 or more (more than 4 years). A gap of one year (0 to 2 years) is not used. Pairs whose recorded
  age decreases with follow-up order are excluded and counted.
- GRAPE (gap from Interval Years): `same_camera_<1y`, `same_camera_1y+`, `other_camera` (same eye, any gap),
  `fellow_eye_same_visit` (same date, same camera). GRAPE has no same-session repeats.
- Population mean (both datasets): the candidate is the pixel-wise mean of the base images of M other patients
  matched to the anchor (M = 32 for ChestX-ray14, 16 for GRAPE). The mean is an image and goes through P, D and N like
  any image (P z-scores it, which restores its global spread). The mean rung is computed for every pair that has
  enough matched strangers, so it can be compared with that pair's own rung on the same anchor.

**Nuisance-matched strangers (the reference).** For a pair (x, y), the reference is the set of distances d(x, s) for
s drawn from other patients matched to y: one image per patient, up to K = 200 patients.
- ChestX-ray14 match keys: same sex, same original pixel spacing (device proxy, 12 values), age within 2 years of y,
  widened once to 5 years if fewer than 50 patients qualify; pairs with fewer than 50 are dropped and counted.
- GRAPE match keys: same camera and same eye side as y; no age match (too few patients); minimum 30.
- Mean rung reference: d(x, s) for strangers matched to x, disjoint from the M mean members, same minimum.
- The pair's own patient never appears among strangers or mean members.

**Calibrated score.** c(d) = P(d_stranger > d) + 0.5 P(d_stranger = d) over the pair's reference
(`grape_calibration.calibrated_score`). 1 - c is the empirical stranger P-value of the pair, in the sense of Xu and
Zhang's random-pair significance of TM-score [2]; its resolution is 1/K. The mean of c over a rung is the AUC of that
rung against its matched strangers.

**Uncertainty.** Patient-cluster bootstrap (resample patients with all their pairs), 2,000 resamples, percentile 95%
intervals, seed 20261101. Paired contrasts (mean rung minus own rung) are bootstrapped on the pair-level difference.

## Claims under test (ChestX-ray14 primary; representations P and D)

**C1 Persistence.** For each of `short_0y`, `mid_2-4y`, `long_5y+`: mean c with its interval.
**C2 Decline with time.** Differences short minus mid and mid minus long (unpaired, patient bootstrap); Spearman
correlation of c with the gap over all selected same-person pairs.
**C3 Nuisance control.** For each rung, mean c under N, and the paired difference c_R minus c_N (R = P or D).
**C4 Population mean.** Mean c of the mean rung, and the paired difference c_mean minus c_own for the pairs of each
rung (same anchor). Primary contrast: `long_5y+`.

GRAPE: the same quantities for its rungs and representations, reported as secondary (see evidence boundary).

## Prespecified interpretation

- "Identity persists at gap g in representation R" only if the lower 95% bound of mean c is above 0.5 AND the lower
  bound of c_R minus c_N is above 0.
- "The ladder declines with time in R" only if both adjacent differences have lower bounds above 0. Otherwise report
  the numbers as "no detectable decline" or "partly ordered".
- "R rewards the population mean" if the lower bound of c_mean minus c_own on `long_5y+` is above 0 (the mean is
  placed above the same person years later). "R places the mean below the person" if the upper bound is below 0.
  Otherwise "not separated".
- A rung whose c_N is itself above 0.5 with a lower bound above 0.5 is flagged "nuisance-driven" in the report,
  whatever R gives.
- No threshold is tuned after looking at scores. Four claims, two representations: no multiplicity correction; the
  number of intervals reported is stated.

## Sensitivity analyses (reported separately, never substituted for the primary)

- S1 Disease confound: pairs with "No Finding" on both images, strangers and mean members also "No Finding" only.
  Follow-up films are often taken because of illness, so the ageing rung is confounded with disease.
- S2 Age tolerance fixed at 5 years for all references.
- S3 Sample invariance (in the spirit of KING's invariance to the sample [3]): references drawn only from a random
  half A of the patients, then only from the complementary half B; report rung means under each and their difference.
- S4 Pixel resolution 32 x 32 and 16 x 16 instead of 64 x 64 (P only).
- S5 Reference not matched on age (sex and device only), to show what age matching changes.
- S6 GRAPE fellow eye with the candidate flipped horizontally (P only).

## Selection counts (metadata only, before any image is read)

ChestX-ray14 primary: `short_0y` 777 pairs, `mid_2-4y` 841, `long_5y+` 503 (one per patient each); median K 93, 100
and 112; the mean rung is available for 671, 730 and 439 of them. Dropped for fewer than 50 matched strangers: 98, 103,
150. GRAPE: `same_camera_<1y` 95, `same_camera_1y+` 71, `other_camera` 152, `fellow_eye_same_visit` 119; K 47 to 90.
Counts for S1 to S5 are in the selection report (private) and will be copied to the results JSON.

## Outputs

`results/ladder2d.json`: aggregates only (per dataset, variant, representation and rung: n pairs, n patients,
mean and median c with intervals, tolerance curve at c >= 0.5, 0.9, 0.95, 0.99, paired contrasts).
`results/ladder2d_manifest.json`: protocol, code and input SHA-256, git commit, seed, package versions, thread count,
run time. Per-pair scores, image names and any image stay in `C:\Projects\data\projects\identity-ladder-metric\
run-2026-11-02-2d-ladder\`.

## Evidence boundary

- ChestX-ray14: one hospital (NIH Clinical Center), a non-random subset enriched for long follow-up, no acquisition
  dates (gaps from whole-year ages), pixel spacing as the only device proxy, and follow-ups driven by illness. Results
  describe this subset, not chest radiographs in general.
- GRAPE was already analysed in the pilot (`results/grape_calibration.json`, pixels 16 x 16 and RETFound, pooled
  camera-pair references). Under this protocol only the matched-substitution reference, the mean rung, the nuisance
  control and DINOv2 are new. GRAPE results are a partly unblinded replication, not a confirmatory test.
- DINOv2 is a general-purpose model, not trained on medical images or for identity. Neither P nor D is an identity
  metric; they are the instruments being tested.
- This is not a re-identification study. Deep models trained for it already recover patient identity in
  ChestX-ray14 with AUC 0.994, also ten and more years after the first film [4]; we train nothing and link nothing to
  outside identities.
- Daugman's degrees-of-freedom estimate [5] applies to fractional Hamming distances of binary codes; it is not
  computed here for continuous distances.

## Ethics and data

- ChestX-ray14: public, de-identified by the provider. The NIH FAQ (downloaded with the data) states that use is
  unrestricted, with a link to the original download site, an acknowledgement of the NIH Clinical Center and a
  citation of the CVPR 2017 paper [6]. GRAPE: CC0 1.0 (figshare API, checked 2026-10-23); data paper [7].
- Secondary analysis of public de-identified images. Whether Kate's institution requires an exemption determination
  for this use: `[to confirm with Kate]`.
- Real images, image names, patient IDs and per-pair results never enter this public repository. Data sync only to
  Kate's own second laptop (data-root policy).

## Deferred datasets (need Kate's action; not part of this freeze)

- UQ longitudinal 3D-TBP tile and dermoscopy dataset (doi:10.48610/a13deaf): listed as open, licence "Other" with a
  UQ data reuse agreement link; the eSpace page shows a human-verification check that only Kate can pass. Same
  lesion over 6-monthly visits would add a lesion-level ageing rung. Added by amendment once downloaded.
- SpeakingFaces (CC BY 4.0): access is requested through Hugging Face. Two sessions on separate days.

## References

1. Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, et al. DINOv2: learning robust visual features without supervision [Preprint]. arXiv:2304.07193. 2023.
2. Xu J, Zhang Y. How significant is a protein structure similarity with TM-score = 0.5? Bioinformatics. 2010;26(7):889-95. doi:10.1093/bioinformatics/btq066
3. Manichaikul A, Mychaleckyj JC, Rich SS, Daly K, Sale M, Chen WM. Robust relationship inference in genome-wide association studies. Bioinformatics. 2010;26(22):2867-73. doi:10.1093/bioinformatics/btq559
4. Packhäuser K, Gündel S, Münster N, Syben C, Christlein V, Maier A. Deep learning-based patient re-identification is able to exploit the biometric nature of medical chest X-ray data. Sci Rep. 2022;12(1):14851. doi:10.1038/s41598-022-19045-3
5. Daugman J. How iris recognition works. IEEE Trans Circuits Syst Video Technol. 2004;14(1):21-30. doi:10.1109/TCSVT.2003.818350
6. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-Ray8: hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2017. p. 3462-71. doi:10.1109/CVPR.2017.369
7. Huang X, Kong X, Shen Z, Ouyang J, Li Y, Jin K, et al. GRAPE: a multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma management. Sci Data. 2023;10(1):520. doi:10.1038/s41597-023-02424-4
