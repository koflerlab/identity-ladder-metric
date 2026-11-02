# Protocol: registration before identity distances (a separate study)

Status: DRAFT 1, 2 Nov 2026, for Kate's approval. Nothing under this protocol has been computed: no registration
quality metric, no c, no figure. On approval it is frozen; any later change is a dated amendment, and results computed
before an amendment are never relabelled. Git commit and file hashes give provenance, not external preregistration.
Scope: the next paper (the identity ladder). Nothing here goes into the ISBI 2027 paper.

Why a separate study (Kate, 2 Nov): the 2D ladder numbers were distorted by poor registration, and registration,
harmonisation and generative alignment are different methods, sometimes different for each data type. This protocol
fixes which methods are compared, which quality metrics are measured and how a method is chosen, before any of them
is run on the frozen selections. It does not replace the visual QC gate (`docs/REGISTRATION_QC_PROTOCOL.md`, owned by
another session): a method chosen here must still pass that gate before any identity result is computed with it.

## Question

Before an identity distance is computed between two images, how should images of each data type be brought into
correspondence so that every image that enters a comparison (the person's own other image, the fellow eye or other
side, each matched stranger, each member of the population mean) is treated the same way, and no method favours the
person?

Sub-questions, per data type:

1. Accuracy. How close does each method bring homologous anatomy, measured on structures the method did not use to
   fit, and on images moved by a known transform?
2. Failure. On how many images or pairs does each method fail, and what happens to a failure?
3. Fairness. Is accuracy, and is failure, the same for the person's own images as for the fellow eye, the strangers
   and the mean members? A method that aligns the same person better than a stranger raises the calibrated score c of
   the person for reasons that have nothing to do with identity.
4. Downstream effect. How do calibrated c of the own rung, the fellow eye and the population mean move under each
   method, compared with no registration?

The method for each data type is chosen on questions 1 to 3 only. Question 4 is reported but never used to choose a
method, so the choice cannot be tuned to the identity result.

## Data types and inputs

| Data type | Input | SHA-256 | Status |
|---|---|---|---|
| Fundus | GRAPE CFP (registry `grape-fundus`) and `selection_grape.json` (private, data root) | selection `d3c236d60d125f886ea93ed2804a815b0007e12ccc1082c01b36c2d1a7a4708a`; metadata `bc79566f4687f7512628ee26959eca0fdb11dd859d217ebf82be99035fd2e02e`; per-image SHA-256 to the run manifest | available |
| Iris, NIR | UTIRIS (registry `utiris`) and UPOL (registry `upol-iris`), `selection_iris.json` (private) | `c3a41f09ac7d622c8215788b99c2d67ce1ecfc729d05ba33aab1b319240e107c`; per-image to the manifest | available |
| Face | SpeakingFaces (CC BY 4.0) | to be recorded by amendment when downloaded and selected | download pending (storage lock held by the 2D session) |
| Brain MRI | QTIM, through the existing `src/real_cases_syn.py` pipeline (another session) | from that pipeline's own manifest | later; read-only from that pipeline |
| Fundus, external accuracy | FIRE (134 fundus pairs with ground-truth correspondences [1]) | to be recorded if Kate approves the download | optional, not downloaded |

Code entering the study at its start: `src/ladder2d_register.py` `c3b7e396e5954a9fea9b68bd961c6bc61f7ad7c6ffed4567b8c2720a28634dd3`,
`src/iris_codes.py` `1f294f963b5f327e05173649b6af18cc9a1738bdc34d1d8af8567e005000bc92`,
`src/ladder2d.py` `81a9a42973c5351e2988592b7c581193a49a5626cbf07527328dcdf5de20aeee`,
`docs/PROTOCOL_2d_ladder.md` `b19e809950ecfef030e8e74a927024cb86c0cb203118031da410c93429372d72` (at 00f9534).
Every run refuses to start if an input hash differs, and records the SHA-256 of every image it reads.

The pairs, matched strangers and mean pools are those of the frozen 2D selections, without reselection, replacement
or ranking by quality. Roles of an image within a pair: anchor, own candidate (same eye, other capture), fellow eye,
stranger (the pair's reference), mean member.

## Method families

Two kinds of method are distinguished, because they differ in how they can favour the person.

- **Per-image frame.** Each image is mapped on its own to one fixed frame from structures detected in that image
  alone (pupil and limbus circles, disc and fovea, facial landmarks, a brain template). The mapping of an image does
  not depend on the image it is compared with, so the method is symmetric by construction; it can still be biased if
  it fails or is less accurate on some people.
- **Pairwise.** The candidate is moved onto the anchor by optimising agreement between the two (keypoints,
  intensity, deformation). The more alike two images are, the easier the fit, so the person's own image is fitted
  more often and more closely than a stranger (in the GRAPE check of Amendment 2b, ORB registered every same-eye pair,
  0 of 20 fellow-eye pairs and no strangers). A pairwise method is admitted only if it is applied in exactly the same way to every
  candidate slot (own, fellow, every stranger, every mean member), and its fairness is measured below.

Rules common to all methods:

- A failure is never replaced by the unregistered image. The failed image is excluded and counted, by role; a pair
  whose anchor or candidate fails is dropped and counted; strangers and mean members that fail are skipped, and a pair
  keeps at least 30 strangers and 8 mean members or is dropped and counted.
- The fellow eye or other side is reflected left-right before registration, and the reflection is recorded as part of
  the transform, so a reflection is never estimated as a rotation.
- Harmonisation references are fixed population references built from people outside the pair, never the anchor.
- Every parameter below is fixed now. A parameter change after any quality metric is seen is an amendment, and the
  earlier result is kept.

### Fundus (GRAPE)

| Code | Family | Method |
|---|---|---|
| G0 | none | centre square crop, 512 base (as the 2D protocol); reference row for every metric |
| G1 | per-image landmark frame | disc-fovea similarity frame of Amendment 2b (`ladder2d_register.fundus_canonical`) |
| G2 | pairwise keypoint | ORB 800 keypoints on the CLAHE green channel inside the field of view eroded by 15 px, ratio 0.8, RANSAC similarity (OpenCV `estimateAffinePartial2D`, 3 px, 500 iterations, `setRNGSeed(20261101)`) [2, 3]; accepted if at least 12 inliers, scale 0.8 to 1.25, rotation under 20 degrees (Amendment 2a), else failure |
| G3 | pairwise intensity | G1 frame, then ECC maximisation [4] (OpenCV `findTransformECC`, Euclidean motion, grey 512 image inside the joint field of view, 100 iterations, epsilon 1e-5); failure if ECC does not converge or the update exceeds 10 px or 5 degrees |
| G4 | deformable | G3, then dense optical flow (OpenCV DIS, preset MEDIUM) on the green channel, smoothed by a Gaussian of sigma 8 px; failure if any Jacobian determinant of the smoothed field is at most 0 |
| H1 | intensity harmonisation | G1 frame, then per-channel histogram matching inside the field of view to a fixed reference: the per-channel pooled histogram of 50 images from people outside every selected pair, drawn with the seed |
| H2 | device harmonisation | G1 frame, then per-camera standardisation: each channel z-scored inside the field of view with the camera's mean and SD estimated on the same 50-image pool |
| E1 | generative alignment | cross-camera image translation (CycleGAN-type [5]); not run under this draft: it needs training compute and its own amendment. Evaluated, if run, with every metric below plus the hallucination concern of distribution-matching losses [6] |

### Iris, NIR (UTIRIS, UPOL)

| Code | Family | Method |
|---|---|---|
| I0 | none | centre square crop, 224 base |
| I1 | per-image landmark frame | limbus centre to the image centre, limbus radius to 80 px (Amendment 2; UTIRIS); UPOL is run too and its failures counted |
| I2 | per-image landmark frame | pupil centre to the image centre, working width kept (Amendment 2a) |
| I3 | per-image normalisation | rubber-sheet unwrapping between the pupil and limbus circles [7], 64 radial x 256 angular samples, which removes pupil dilation; no rotation search |
| I4 | pairwise rotation | I3, then circular shift of the unwrapped image over plus or minus 16 angular samples to the best correlation with the anchor, as in iris matching [7]; applied to every candidate slot |
| I5 | pairwise intensity | I1 (UTIRIS) or I2 (UPOL), then ECC, Euclidean motion, as G3 |
| H1 | intensity harmonisation | I1 or I2, then histogram matching inside the iris annulus to a fixed population reference of 50 images per dataset |

Pupil dilation is a real within-person change. I3 removes it by construction, so its downstream effect is reported
separately and it is never the only iris method carried forward.

### Face (SpeakingFaces, visible images; by amendment once downloaded)

| Code | Family | Method |
|---|---|---|
| F0 | none | centre crop |
| F1 | per-image landmark frame | five-point similarity alignment (eye centres, nose tip, mouth corners) to a fixed template, the alignment used before face embeddings such as ArcFace [8]; landmark detector fixed by amendment before the data are read |
| F2 | pairwise intensity | F1, then ECC, Euclidean motion |
| F3 | deformable | F1, then thin-plate spline from dense landmarks to the anchor; expected to erase identity, included as a negative example |

The thermal channel and cross-spectral alignment are out of scope.

### Brain MRI (QTIM; later, read-only)

The QTIM pipeline (rigid and affine to a template, then symmetric diffeomorphic SyN [9]) belongs to another session
and is not run or duplicated here. When its outputs exist, this study reads them and computes the same quality
metrics. Families to compare by a later amendment: affine only, SyN, intensity standardisation [10], and, for
repeat scans, a halfway space in which neither time point is privileged, which avoids the bias that asymmetric
processing introduces into longitudinal estimates [11, 12]. Harmonisation of derived features across scanners
(ComBat [13, 14]) is a feature-level method and is evaluated only through the downstream metrics.

## Quality metrics

All distances in pixels of the 224 analysis grid unless stated. Every metric is reported per data type, method and
role, with person-cluster bootstrap 95% intervals (2,000 resamples, seed 20261101) and its denominator.

**Q1 Accuracy.**
- Q1a Known-transform recovery (all methods). Each image is moved by a random similarity transform drawn with the
  seed (rotation up to 10 degrees, scale 0.9 to 1.1, shift up to 10% of the width) and, for G4 and F3, also by a
  smooth random displacement field (Gaussian sigma 16 px, maximum 4 px). The method is run again; target registration
  error = mean distance, over a 16 x 16 grid of points inside the valid region, between where the recovered and the
  true mapping send each point. For a per-image frame this measures the repeatability of the frame; for a pairwise
  method, recovery of the moved image onto its own original. Primary accuracy metric where no manual landmarks exist.
- Q1b Held-out structure error (all methods). Distance between independently detected structures that the method
  did not use to fit, after mapping. Fundus: Dice of the vessel maps (the fixed classical pipeline of the 2D protocol)
  in a ring of 0.3 to 0.6 field-of-view radii around the disc, and, for G2 to G4, disc and fovea position error.
  Iris: pupil-centre distance for I1, limbus-centre distance for I2 (UTIRIS, where the limbus is found). Face:
  landmarks of the dense set not used by F1. Local structure overlap is the surrogate that best separates accurate
  from inaccurate registration [15].
- Q1c Same-eye patch correlation (Pearson, grey, the patch of the 2D protocol). Reported because it was used in the
  checks so far, but labelled a surrogate: image similarity cannot by itself distinguish accurate from inaccurate
  registration [15]. Never used alone to choose a method.
- Q1d Inverse consistency (pairwise methods). Register a to b and b to a; mean distance of the composed mapping from
  the identity over the grid [16].
- Q1e Plausibility (deformable methods). Fraction of the field with Jacobian determinant at most 0, and the 99th
  percentile of the absolute log Jacobian.
- Q1f External accuracy (optional, fundus). If Kate approves the FIRE download: mean landmark error on its 134
  annotated pairs for G1 to G4, under FIRE's own protocol [1].

**Q2 Failure.** Computational failure rate per data type, method and role; pairs dropped per rung. Computational
failures are kept separate from visual QC failures, which are counted under the QC gate.

**Q3 Fairness.**
- Q3a Failure parity: failure rate of strangers and mean members minus that of own candidates; fellow eye minus own.
- Q3b Accuracy parity: median Q1a and Q1b by role; stranger minus own, fellow minus own, paired within person where
  possible.
- Q3c Candidate-slot null: for every pair, one stranger not in its reference is passed through the candidate slot
  with exactly the code path of the own candidate, and scored against the pair's reference. Its mean c must be 0.5;
  a method that treats the candidate slot differently from the reference shows here.
- Q3d Swap symmetry: anchor and candidate exchanged; mean of the paired difference in c, the asymmetry bias that
  longitudinal processing is known to introduce [11, 12].
- Q3e Reflection control: every same-eye candidate also passed reflected and explicitly un-reflected; mean paired
  difference in c from the unreflected case must be 0. A coordinate test, not a fellow-eye comparison.

**Q4 Downstream ladder.** Calibrated c (`grape_calibration.calibrated_score`, as in the 2D protocol) in the pixel
representation P (primary) and the nuisance control N, for: the closest own rung (GRAPE `same_camera_<1y`, iris
`same_eye_same_sitting`), the fellow eye, and the population mean (lambda = 0). Contrast: c_mean minus c_own and c_own
minus c_fellow, per method, and each method minus G0 or I0. Q4 diagnoses what a method does to the ladder; it is not
an identity result while the QC gate is not met.

## Claims under test

**R1 Accuracy.** Per data type, the ranking of methods by Q1b (primary) and Q1a, with intervals.
**R2 Failure.** Failure rate per method and role.
**R3 Fairness.** Q3a to Q3e per method.
**R4 Choice.** One recommended method per data type (one per dataset for iris), chosen by the rule below.
**R5 Downstream.** Q4 per method, descriptive.

## Prespecified interpretation

- A method is **eligible** for a data type only if all of these hold: computational failure is at most 10% in every
  role (the 90% bar of the QC gate); no Q3a difference has a 95% interval entirely above 5 percentage points against
  the stranger or fellow role; no Q3b difference has an interval entirely above 1 pixel in Q1a or entirely above 0.05
  in Q1b Dice in the person's favour; the Q3c interval contains 0.5; the Q3d and Q3e intervals contain 0. For
  deformable methods, also: Jacobian determinant at most 0 on 0% of the field after smoothing.
- A method that fails only the fairness conditions is reported as **biased in the person's favour** and is never
  used for identity results, however accurate.
- Among eligible methods, the **recommended** one has the lowest median Q1b; methods whose Q1b intervals overlap are
  tied, and the tie goes to the method with fewer degrees of freedom (per-image frame, then similarity, then affine,
  then deformable), then to the lower median Q1a. Q4 does not enter the choice.
- If no method is eligible for a data type, the data type is reported as **not registrable under this protocol** and
  no registered ladder is computed for it.
- The recommended method then goes to the visual QC gate; only after that gate passes is the 2D ladder rerun with it,
  under an amendment to `docs/PROTOCOL_2d_ladder.md`.
- No threshold or parameter is tuned after looking at any metric. Number of claims: five, over up to eight methods
  per data type; no multiplicity correction; the number of intervals reported is stated.

## Experiments (a separate network)

Each method is its own experiment with its own script, tests and private run directory, and writes one aggregate
file; a shared evaluator computes Q1 to Q4 identically for every method.

- `src/reg_methods.py`: one function per method code above, each returning (registered image, transform, valid mask,
  failure reason); unit tests on synthetic images with known transforms (recovery, reflection order, failure
  handling, no fallback to the unregistered image).
- `src/reg_eval.py`: Q1 to Q4 from a method's outputs; tests on synthetic data where the answer is known (an
  intentionally biased toy method must fail Q3c, a symmetric one must pass).
- `src/reg_run.py --dataset grape|utiris|upol --method G1`: one experiment; private outputs in
  `C:\Projects\data\projects\identity-ladder-metric\run-<date>-registration\<dataset>_<method>\`.
- Order: GRAPE and UTIRIS first (G0 to G4, H1, H2; I0 to I5, H1), then UPOL, then faces after download, then MRI
  from the QTIM outputs. CPU: at most four threads, one experiment at a time (shared machine).

## Outputs

`results/registration/<dataset>_<method>.json`: aggregates only (per role: n, failure rate, Q1 medians and
intervals; Q3 contrasts; Q4 means and contrasts). `results/registration/comparison.md` and `.csv`: one table per data
type, methods in rows, Q1b, Q1a, failure, fairness verdict, eligibility and Q4 in columns. `results/registration/
manifest.json`: protocol, code, input and image-set SHA-256, git commit, seed, package versions, threads, run time.
Images, image names, person IDs, transforms and per-pair values stay in the private run directory.

## Evidence boundary

- No manual landmarks exist for GRAPE, UPOL or UTIRIS. Accuracy rests on held-out automatic structures (Q1b) and on
  known-transform recovery (Q1a). Q1a measures recovery of a synthetic motion, not of real eye or head motion, and
  automatic structures carry their own detector error. FIRE (Q1f) would give real ground truth for fundus only.
- The fairness tests detect a method that treats roles differently; they cannot prove that real anatomy is aligned
  equally well in all roles, because strangers have different anatomy and no ground truth exists for them.
- Deformable and generative methods can remove individual anatomy; Q1e and Q4 show this only indirectly.
- The selections were already analysed unregistered (2D protocol primary run and Amendment 1), and same-eye patch
  correlations under G1, G2, I1 and I2 were seen in the registration checks of Amendments 2a and 2b. Those methods are
  therefore partly unblinded; the others are first looks.
- Iris sets are small and single-sitting; GRAPE is one hospital's glaucoma cohort. Face and MRI are not covered until
  their amendments.

## Ethics and data

As in the 2D protocol: public de-identified data under their recorded terms; whether Kate's institution requires an
exemption determination for this secondary use: `[to confirm with Kate]`. Real images, image names, person IDs,
transforms, overlays and per-pair values never enter this public repository. Downloads (FIRE, SpeakingFaces) only
under the storage-maintenance lock and with Kate's approval.

## References

1. Hernandez-Matas C, Zabulis X, Triantafyllou A, Anyfanti P, Douma S, Argyros AA. FIRE: fundus image registration dataset. Model Artif Intell Ophthalmol. 2017;1(4):16-28. doi:10.35119/maio.v1i4.42
2. Rublee E, Rabaud V, Konolige K, Bradski G. ORB: an efficient alternative to SIFT or SURF. In: 2011 International Conference on Computer Vision. IEEE; 2011. p. 2564-71. doi:10.1109/ICCV.2011.6126544
3. Fischler MA, Bolles RC. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun ACM. 1981;24(6):381-95. doi:10.1145/358669.358692
4. Evangelidis GD, Psarakis EZ. Parametric image alignment using enhanced correlation coefficient maximization. IEEE Trans Pattern Anal Mach Intell. 2008;30(10):1858-65. doi:10.1109/TPAMI.2008.113
5. Zhu JY, Park T, Isola P, Efros AA. Unpaired image-to-image translation using cycle-consistent adversarial networks. In: 2017 IEEE International Conference on Computer Vision (ICCV). IEEE; 2017. p. 2242-51. doi:10.1109/ICCV.2017.244
6. Cohen JP, Luck M, Honari S. Distribution matching losses can hallucinate features in medical image translation. In: Medical Image Computing and Computer Assisted Intervention (MICCAI 2018). Lecture Notes in Computer Science. Cham: Springer; 2018. p. 529-36. doi:10.1007/978-3-030-00928-1_60
7. Daugman J. How iris recognition works. IEEE Trans Circuits Syst Video Technol. 2004;14(1):21-30. doi:10.1109/TCSVT.2003.818350
8. Deng J, Guo J, Xue N, Zafeiriou S. ArcFace: additive angular margin loss for deep face recognition. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2019. p. 4685-94. doi:10.1109/CVPR.2019.00482
9. Avants BB, Epstein CL, Grossman M, Gee JC. Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Med Image Anal. 2008;12(1):26-41. doi:10.1016/j.media.2007.06.004
10. Nyul LG, Udupa JK, Zhang X. New variants of a method of MRI scale standardization. IEEE Trans Med Imaging. 2000;19(2):143-50. doi:10.1109/42.836373
11. Yushkevich PA, Avants BB, Das SR, Pluta J, Altinay M, Craige C. Bias in estimation of hippocampal atrophy using deformation-based morphometry arises from asymmetric global normalization: an illustration in ADNI 3 T MRI data. Neuroimage. 2010;50(2):434-45. doi:10.1016/j.neuroimage.2009.12.007
12. Reuter M, Fischl B. Avoiding asymmetry-induced bias in longitudinal image processing. Neuroimage. 2011;57(1):19-21. doi:10.1016/j.neuroimage.2011.02.076
13. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics. 2007;8(1):118-27. doi:10.1093/biostatistics/kxj037
14. Fortin JP, Cullen N, Sheline YI, Taylor WD, Aselcioglu I, Cook PA, et al. Harmonization of cortical thickness measurements across scanners and sites. Neuroimage. 2018;167:104-20. doi:10.1016/j.neuroimage.2017.11.024
15. Rohlfing T. Image similarity and tissue overlaps as surrogates for image registration accuracy: widely used but unreliable. IEEE Trans Med Imaging. 2012;31(2):153-63. doi:10.1109/TMI.2011.2163944
16. Christensen GE, Johnson HJ. Consistent image registration. IEEE Trans Med Imaging. 2001;20(7):568-82. doi:10.1109/42.932742
