# What face recognition solved, and what transfers to the identity ladder

Date: 2026-11-02. Literature and design note, no experiments. It complements theme 7 of
[REPRODUCTION_BACKLOG.md](REPRODUCTION_BACKLOG.md) and does not repeat its entries; those are referred to by ID
(T7-11 Daugman, T7-12 Daugman and Downing, T7-13 Sun et al., T7-14 Sturim and Reynolds, T7-16 Yager and Dunstone,
T7-17 Brümmer and du Preez, T7-19 Ali et al., T7-21 NIST demographics).

Verification: every reference below was resolved against Crossref (title, authors, venue, pages, year, DOI) and its
abstract was retrieved from OpenAlex. Statements about content are taken only from those abstracts. Where only the
metadata could be retrieved this is said in the row, and no content claim is made. Pages for IEEE proceedings were not
checked on ieeexplore.

## (a) What face recognition did, per problem

### 1. One recipe under all three: a normalised embedding scored by cosine

The modern systems that cope with ageing and masks share one design. A network maps an image to a vector, the vector
is L2-normalised, and two images are compared by the cosine of the angle between their vectors.

- SphereFace [R1] states the open-set criterion directly: the largest within-person distance should be smaller than the
  smallest between-person distance, and the A-Softmax loss imposes this as an angular margin on a hypersphere.
- CosFace [R2] reformulates softmax as a cosine loss by L2-normalising both features and weights "to remove radial
  variations", then adds a cosine margin.
- ArcFace [R3] uses an additive angular margin, and its sub-centre variant gives each class K centres so that noisy or
  hard faces do not pull the main centre.

What this means for calibration: the margin losses do not output a probability. They make genuine and impostor cosine
distributions separate. The operating point is still set on an impostor distribution (a threshold at a chosen false
accept rate), which is why NIST reports false match rate at fixed thresholds and why the threshold varies across
demographic groups (T7-21; Robinson et al. [R4] report that subgroup-specific thresholds reduce gaps and raise overall
performance). The embedding gives order; calibration comes from the strangers. This is exactly the split our
c(d) = P(d_stranger > d) makes.

### 2. Ageing

What was done:

- Benchmarks with real age gaps. MORPH [R5] is a longitudinal database with spans from a few months to over twenty
  years, and its baseline shows that age progression lowers recognition rates. AgeDB [R6] is manually collected with
  year-accurate age labels, built so that age-invariant verification can be tested without label noise. CACD [R7] is
  the cross-age celebrity set (metadata verified, abstract not retrieved).
- Separating identity from age inside the feature. Wen et al. [R8] learn age-invariant deep features with a latent
  factor guided CNN and test on MORPH Album 2, FG-NET and CACD-VS. Wang et al. (DAL) [R9] factor a mixed feature into an
  identity-dependent and an age-dependent component that are made uncorrelated through adversarial learning with a
  canonical mapping module. OE-CNN [R10] decomposes features orthogonally (metadata only).
- Measuring the decline instead of hiding it. Best-Rowden and Jain [R11] fit mixed-effects regression to genuine
  similarity scores of COTS matchers on two mugshot databases: genuine scores fall with elapsed time, but 99% of subjects
  are still recognised at 0.01% FAR up to about 6 years, and age, sex and race only marginally change the trend. Deb et
  al. [R12] apply multilevel models to genuine scores of children aged 2 to 18 over up to six years.

Transfer: the "years later" rung should not be one number. Following [R11], model genuine c(d) as a function of
elapsed time with subject random effects, and report the time at which the rung's c falls to a fixed level. The DAL idea
(an age component made uncorrelated with the identity component) is the template for removing the ageing direction from
a brain embedding, but only if age is a nuisance for the question; for brain-age work it is the signal.

### 3. Masks and occlusion

What was done:

- NIST measured it. FRVT part 6A [R13] ran pre-pandemic 1:1 algorithms, which were not designed for masks, on masked
  images. Part 6B [R14] repeated this with newer algorithms and reports evidence that a number of developers adapted
  their algorithms to masked faces.
- Benchmarks were built: the InsightFace track of the Masked Face Recognition challenge [R15] has a 7K-identity masked
  test set plus children and multi-racial sets, kept private to avoid privacy problems; Wang et al. [R16] release real
  and synthetic masked face recognition sets.
- Quality enters the score. MagFace [R17] learns an embedding whose magnitude increases monotonically with how likely
  the face is to be recognised, so magnitude is a quality measure. Probabilistic face embeddings [R18] represent each
  face as a Gaussian; the variance expresses uncertainty and predicts matching accuracy.
- Partial regions alone can identify. Park et al. [R19] use the periocular region as a biometric, including in
  partially occluded faces and with the iris and eye masked.

Transfer: an inpainted or lesion-masked region is our "mask". Two lessons. First, score the visible part and the
filled part separately, as mask reports do, and never let the fill raise identity above what the visible part supports.
Second, a quality term (MagFace magnitude, PFE variance) belongs next to c(d), so that a low-information region reports
low confidence rather than a confident stranger-like score.

### 4. The other side: half face, profile, left versus right

What the literature shows:

- Profile versus frontal is a match task that is harder for machines than for humans. On CFP [R20] human accuracy drops
  only from 96.24% (frontal-frontal) to 94.57% (frontal-profile), while all tested algorithms degrade by more than 10%.
- Periocular systems treat side as information: Park et al. [R19] list "use of side information (left or right) in
  matching" among the factors studied (the abstract does not give the result).
- Ears are partly symmetric. Abaza and Ross [R21] find some degree of symmetry between left and right ears that could be
  exploited for matching, and also enough asymmetry to be useful in fusion with face.
- Irises are not. The left and right eyes of one person follow the same distribution as unrelated eyes (T7-12).
- Face systems routinely mirror one half onto the other to fill occlusions (e.g. [R22] uses mirrored parts of the face
  to fix occluded areas), which treats the two halves as approximately the same identity signal.

So "same person, other side" is a match for face halves and partly for ears, and a non-match for iris texture. The
rung's position is a property of the trait, not of the ladder.

### 5. Twins

Phillips et al. [R23] measured three top MBE 2010 algorithms on 126 identical twin pairs (same day) and 24 pairs (one
year apart), with covariates and bootstrap confidence intervals (the abstract does not state the error rates). With
T7-13 (twin impostors harder for finger and face, not significantly for iris) this supports keeping MZ and DZ as
separate impostor cohorts, not folding them into strangers.

## (b) Proposal for our metric

1. Embedding. Use a fixed, pretrained identity embedding per modality, L2-normalised, and keep the pixel metrics as
   comparators. For brain MRI the candidates are a learned embedding trained with a margin loss on subject identity
   across sessions (ArcFace style, [R3]) and a non-learned shape descriptor such as BrainPrint [R24], which was built for
   subject identification and asymmetry analysis. The non-learned one is the control that cannot have memorised the
   test subjects.
2. Distance. d = 1 - cos(u, v) on normalised embeddings, the same quantity margin losses shape [R1-R3]. For pixel or
   region metrics keep RMSE inside the mask, as now.
3. Normalisation. c(d) = P(d_stranger > d) against strangers matched on scanner pair, protocol, sex and age band, i.e.
   a cohort normalisation in the T-norm sense (T7-14) done non-parametrically. Report c per demographic cell, because a
   pooled null is miscalibrated across groups (T7-21, [R4]). For reporting as evidence, add a log-likelihood ratio
   from the genuine and stranger score densities at the same nuisance condition, and evaluate it with Cllr (T7-17);
   the LR depends on the calibration data (T7-19), so the cohort is part of the result.
4. Quality. Report next to c a quality value for each image or region (embedding magnitude [R17] or embedding
   variance [R18]); flag pairs where quality is below a fixed level instead of giving them a c.
5. Time. For "another day" and "years later" fit c against elapsed time with subject random effects, as [R11] does for
   genuine scores, and report the slope and the time to a fixed c.
6. The unilateral rung. Define it as: left hemisphere (or left eye, ear) of person A against the mirrored right
   counterpart of person A, after the same mirroring is applied to every stranger pair. Score it against a stranger
   null that is also cross-side (left of A against mirrored right of B). Comparing cross-side genuine pairs to
   same-side strangers would mix the side effect into the identity score. Report the rung's c and say per trait
   whether it is a match (c well above 0.5, as expected for face halves), partial (ears, [R21]) or at chance (iris
   texture, T7-12). For brain MRI this is an open, testable question.

## (c) Candidates for the reproduction backlog

| Ref | What to reproduce | Why |
|---|---|---|
| R11 Best-Rowden and Jain 2018 | Mixed-effects model of genuine scores against elapsed time | Template for the "years later" rung |
| R17 MagFace | Magnitude as quality on a brain embedding | Quality term next to c |
| R18 PFE | Embedding variance as uncertainty | Alternative quality term |
| R9 DAL | Decorrelating an age component from identity | Ageing as nuisance |
| R20 CFP | Frontal-profile versus frontal-frontal drop | Model for cross-side versus same-side rung |
| R21 Abaza and Ross 2010 | Left-right ear matching | A trait with partial bilateral identity |
| R23 Phillips et al. 2011 | Twin impostor cohort with bootstrap CIs | MZ rung design |
| R24 BrainPrint | Subject identification and asymmetry from shape spectra | Non-learned brain embedding baseline |
| R4 Robinson et al. 2020 | Subgroup thresholds | Per-cell calibration of c |

## References

- R1. Liu W, Wen Y, Yu Z, Li M, Raj B, Song L. SphereFace: deep hypersphere embedding for face recognition. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017. p. 6738-46. doi:10.1109/CVPR.2017.713
- R2. Wang H, Wang Y, Zhou Z, Ji X, Gong D, Zhou J, et al. CosFace: large margin cosine loss for deep face recognition. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2018. p. 5265-74. doi:10.1109/CVPR.2018.00552
- R3. Deng J, Guo J, Yang J, Xue N, Kotsia I, Zafeiriou S. ArcFace: additive angular margin loss for deep face recognition. IEEE Trans Pattern Anal Mach Intell. 2022;44(10):5962-79. doi:10.1109/TPAMI.2021.3087709 (conference version: CVPR 2019, p. 4685-94, doi:10.1109/CVPR.2019.00482; the sub-centre claim is from the journal abstract)
- R4. Robinson JP, Livitz G, Henon Y, Qin C, Fu Y, Timoner S. Face recognition: too bias, or not too bias? In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2020. p. 1-10. doi:10.1109/CVPRW50498.2020.00008
- R5. Ricanek K, Tesafaye T. MORPH: a longitudinal image database of normal adult age-progression. In: 7th International Conference on Automatic Face and Gesture Recognition (FGR06). 2006. p. 341-5. doi:10.1109/FGR.2006.78
- R6. Moschoglou S, Papaioannou A, Sagonas C, Deng J, Kotsia I, Zafeiriou S. AgeDB: the first manually collected, in-the-wild age database. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2017. p. 1997-2005. doi:10.1109/CVPRW.2017.250
- R7. Chen BC, Chen CS, Hsu WH. Cross-age reference coding for age-invariant face recognition and retrieval. In: Computer Vision, ECCV 2014. Lecture Notes in Computer Science. 2014. p. 768-83. doi:10.1007/978-3-319-10599-4_49 (metadata only, abstract not retrieved)
- R8. Wen Y, Li Z, Qiao Y. Latent factor guided convolutional neural networks for age-invariant face recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016. p. 4893-901. doi:10.1109/CVPR.2016.529
- R9. Wang H, Gong D, Li Z, Liu W. Decorrelated adversarial learning for age-invariant face recognition. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2019. p. 3522-31. doi:10.1109/CVPR.2019.00364
- R10. Wang Y, Gong D, Zhou Z, Ji X, Wang H, Li Z, et al. Orthogonal deep features decomposition for age-invariant face recognition. In: Computer Vision, ECCV 2018. Lecture Notes in Computer Science. 2018. p. 764-79. doi:10.1007/978-3-030-01267-0_45 (metadata only, abstract not retrieved)
- R11. Best-Rowden L, Jain AK. Longitudinal study of automatic face recognition. IEEE Trans Pattern Anal Mach Intell. 2018;40(1):148-62. doi:10.1109/TPAMI.2017.2652466
- R12. Deb D, Nain N, Jain AK. Longitudinal study of child face recognition. In: 2018 International Conference on Biometrics (ICB). 2018. p. 225-32. doi:10.1109/ICB2018.2018.00042
- R13. Ngan M, Grother P, Hanaoka K. Ongoing Face Recognition Vendor Test (FRVT) part 6A: face recognition accuracy with masks using pre-COVID-19 algorithms. Gaithersburg (MD): National Institute of Standards and Technology; 2020. Report No.: NISTIR 8311. doi:10.6028/NIST.IR.8311
- R14. Ngan M, Grother P, Hanaoka K. Ongoing Face Recognition Vendor Test (FRVT) part 6B. (Crossref gives the title only up to the part number.) Gaithersburg (MD): National Institute of Standards and Technology; 2020. Report No.: NISTIR 8331. doi:10.6028/NIST.IR.8331
- R15. Deng J, Guo J, An X, Zhu Z, Zafeiriou S. Masked face recognition challenge: the InsightFace track report. In: 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). 2021. p. 1437-44. doi:10.1109/ICCVW54120.2021.00165
- R16. Wang Z, Huang B, Wang G, Yi P, Jiang K. Masked face recognition dataset and application. IEEE Trans Biom Behav Identity Sci. 2023;5(2):298-304. doi:10.1109/TBIOM.2023.3242085
- R17. Meng Q, Zhao S, Huang Z, Zhou F. MagFace: a universal representation for face recognition and quality assessment. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021. p. 14220-9. doi:10.1109/CVPR46437.2021.01400
- R18. Shi Y, Jain AK. Probabilistic face embeddings. In: 2019 IEEE/CVF International Conference on Computer Vision (ICCV). 2019. p. 6901-10. doi:10.1109/ICCV.2019.00700
- R19. Park U, Jillela RR, Ross A, Jain AK. Periocular biometrics in the visible spectrum. IEEE Trans Inf Forensics Secur. 2011;6(1):96-106. doi:10.1109/TIFS.2010.2096810
- R20. Sengupta S, Chen JC, Castillo C, Patel VM, Chellappa R, Jacobs DW. Frontal to profile face verification in the wild. In: 2016 IEEE Winter Conference on Applications of Computer Vision (WACV). 2016. p. 1-9. doi:10.1109/WACV.2016.7477558
- R21. Abaza A, Ross A. Towards understanding the symmetry of human ears: a biometric perspective. In: 2010 IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS). 2010. p. 1-7. doi:10.1109/BTAS.2010.5634535
- R22. Cao L, Li H, Guo H, Wang B. Robust PCA for face recognition with occlusion using symmetry information. In: 2019 IEEE International Conference on Networking, Sensing and Control (ICNSC). 2019. p. 323-8. doi:10.1109/ICNSC.2019.8743225
- R23. Phillips PJ, Flynn PJ, Bowyer KW, Vorder Bruegge RW, Grother PJ, Quinn GW, et al. Distinguishing identical twins by face recognition. In: Face and Gesture 2011. 2011. p. 185-92. doi:10.1109/FG.2011.5771395
- R24. Wachinger C, Golland P, Kremen W, Fischl B, Reuter M. BrainPrint: a discriminative characterization of brain morphology. Neuroimage. 2015;109:232-48. doi:10.1016/j.neuroimage.2015.01.032

Notes on verification. R12 publication year is 2018 per Crossref and OpenAlex. R19: OpenAlex gives 2010 (online), the
issue is 2011;6(1), 2011 used. R14 subtitle not given by Crossref and left out. Not
verified and left out: Finn et al. 2015 connectome fingerprinting (doi:10.1038/nn.4135, metadata found, no abstract
retrieved); a primary source for Z-norm (already noted as missing in the backlog).
