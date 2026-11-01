# Research plan: a calibrated anatomical-identity distance for brain MRI inpainting

**Owner:** Ekaterina Kondrateva · **Host:** Hertie Institute for AI in Brain Health, University of Tübingen · **Supervisor:** Dr. Florian Kofler
**Status:** working document, 18 October 2026. Internal companion to `C:\Projects\09_legal\hertlie_ai\package\04_Concept_Note_Kondrateva.docx`; the concept note stays the short external version. Experiments runnable on data already in hand: `Experiment_Plan_Available_Data.md`.

---

## 1. Problem

The BraTS *Local Synthesis of Healthy Brain Tissue via Inpainting* challenge scores submissions with SSIM, PSNR and RMSE, restricted to healthy-tissue masks where ground truth exists. Its organisers state in their own discussion that human expert perception is not fully aligned with established ML metrics, and call for synthesis metrics that close this gap, possibly complemented by human surrogate models [1].

Our MIDL 2026 work measured the floor below which a segmentation pipeline cannot report change: up to 7–8% volume variation in small subcortical structures under controlled test–retest conditions [2]. A parallel MIDL 2026 validation paper approached the same variability from the harmonization side, with HACA3⁺ trained on 132 scanners at 64 sites [3].

**This plan connects the two:** use measured, biologically meaningful variability as the *unit* in which the accuracy of synthesized brain tissue is expressed.

## 2. Core idea: the identity ladder

A good distance between two brains should be ordered by how much of the brain's *identity* differs, not by how much the pixels differ. We want a distance *d* whose values sit on an empirical ladder:

| Rung | Pair compared | What differs | Expected *d* |
|---|---|---|---|
| L0 | same person, same scanner, back-to-back repeat | acquisition noise only | smallest |
| L1 | same person, different scanners | + scanner and protocol | small |
| L2 | same person, years apart | + ageing; individual folding persists | slightly above L1 |
| L3a | monozygotic (MZ) twin | same genome; folding details differ | above L2 |
| L3b | dizygotic (DZ) twin | half the genome | above L3a |
| L4 | unrelated person | everything | largest |

The analogy that motivates it: a fingerprint stays the same while the hand ages; a face stays recognisable while it ages; identical twins share a face but not a fingerprint.

**Applied to inpainting:** a synthesized region is judged by where it lands on this ladder relative to the real tissue it replaces — "indistinguishable from a re-scan" (≤ L1), "as different as the same brain years later" (L2), "as different as a twin's brain" (L3), or "a different brain" (L4). This targets the gap between metrics and expert perception that the challenge organisers named [1].

**The ordering is a hypothesis, not an assumption.** In particular, L2 vs L3 is not guaranteed: 15 years of ageing may move some features further than a twin difference does. Section 9 lists this as a falsifiable hypothesis.

## 3. What the literature says about each rung

All statements below are checked against the abstract or full text of the cited work.

### L0 and L1 — scanner noise is a distinct, pipeline-dependent rung

- In one healthy subject scanned 477 times on 123 scanners, the standard deviation of thalamus volume was 0.08 mL for repeats on the same scanner but 0.28 mL across scanners with a dedicated 3D CNN, and 0.09 mL vs 0.62 mL with FastSurfer [4]. For the thalamus, cross-scanner noise is therefore several times same-scanner noise, and the ratio depends on the pipeline.
- Our own MIDL 2026 benchmark reports up to 7–8% volume variation in small subcortical structures even under controlled test–retest conditions [2].
- **Consequence:** L0 and L1 must be estimated separately, per pipeline and per structure.

### L2 — identity survives ageing in shape space, but folding drifts

- BrainPrint, a Laplace–Beltrami spectral description of cortical and subcortical shapes, identified subjects with 99.9% accuracy across more than 3000 ADNI scans from almost 700 subjects with three to six longitudinal scans each, using the cross-sectional FreeSurfer pipeline so that subject identity was not used in processing [5]. Individual shape is stable enough across years to act as a fingerprint.
- In 322 healthy adults aged 20–85, the local gyrification index decreased with age independently of regional surface area [6]. Folding geometry does change with age, so L2 should sit measurably above L1.
- In the same BrainPrint study, a descriptor based on the local gyrification index alone reached 86.4% (product classifier) and 81.4% (voting classifier), well below full spectral shape [5]. Folding complexity alone is a weaker fingerprint than full shape.

### L3 — twins share global morphometry far more than folding detail

- Total cortical surface area and mean cortical thickness were both highly heritable (0.89 and 0.81) in 110 MZ and 92 DZ male twin pairs, with near-zero genetic correlation between them [7]. **Global volume and thickness are therefore expected to put MZ twins close to L1** — a feature space built on them will not separate the twin rung.
- Sulcal pit depth heritability in the Human Connectome Project pedigree cohort ranged between 0.2 and 0.5, and the authors conclude that cortical shape is determined primarily by non-genetic factors [8]. This supports the expectation that MZ twins differ in folding detail.
- Across four test–retest datasets, sulcal metrics had intraclass correlation with interquartile range 0.65–0.85; sulcal width was the most heritable metric, and earlier-forming sulci were more heritable [9].
- **Counter-evidence:** a genome-wide study of vertex-wise sulcal depth in 33,748 UK Biobank participants found higher locus yield and heritability for sulcal depth than for cortical thickness and surface area [10]. "Folding" is not one feature: depth can be heritable while pit positions or the arrangement of later-forming sulci may not be. **The descriptor choice decides whether L3 separates.**
- BrainPrint shape distances and ranks between twins were significantly lower for MZ than DZ pairs (VETSA, 67 MZ and 51 DZ pairs), with cortical structures, particularly white matter, showing the most significant differences; volumetric (tetrahedral) meshes were usually more significant than surface meshes [5]. This supports L3a < L3b in shape space.

### L4 — trivially separated in shape space

The 99.9% identification rate [5] implies that unrelated subjects are well separated in spectral shape space.

### The central design tension

The features that best separate MZ twins are those with *low* heritability, and these are folding metrics, whose test–retest reliability (intraclass correlation interquartile range 0.65–0.85 [9]) has to be weighed against that separation rather than assumed sufficient. A feature is useful for the ladder only if **its twin separation exceeds its scanner noise**. Feature selection therefore has to be driven by the ratio (L3a distance) / (L1 noise), not by heritability or reliability alone.

## 4. Metric design

### 4.1 Lessons from how existing metrics are built

- **HD95** (standard definition): first a field of local errors (surface point to nearest surface point), then a robust summary (95th percentile instead of maximum). Lesson: build an error field first, summarise second. Its weakness: it is in millimetres, so not normalised across structures or resolutions.
- **Surface Dice:** surface overlap within an organ-specific tolerance, where the tolerances were computed from interobserver variation between three consultant oncologists, and a stringency of 0.95 was chosen [11]. Lesson: **normalise by the variability one cannot remove.** Our construction is the same with scanner and biological variability in place of interobserver variability.
- **Metrics Reloaded** covers image-level classification, object detection, semantic and instance segmentation [12]. Synthesis is outside its stated scope, so there is no off-the-shelf recommendation for inpainting.
- **Human alignment:** standard segmentation metrics and losses correlate only moderately with expert quality assessment, especially for the enhancing tumour compartment [13]. Deep Quality Estimation trains surrogate models on ratings from 15 expert neuroradiologists (1–6 stars) and approximates segmentation quality within a margin comparable to human intra-rater reliability [14]. Expert ratings themselves are noisy: in 60 brain tumour segmentation cases, inter-rater agreement on quality was Krippendorff α = 0.34 [15].

### 4.2 Candidate feature spaces (the operator *f*)

Ranked by expected value for the ladder:

1. **Spectral shape** of cortical and subcortical surfaces and volumes (Laplace–Beltrami eigenvalues, BrainPrint-style) — proven identity signal and MZ < DZ ordering [5].
2. **Folding descriptors** — sulcal pits, depth, width, local gyrification [8,9]. Select per the ratio in Section 3.
3. **Persistent homology of a filtration over a geometric function** on the cortical surface (thickness, sulcal depth). Persistence diagrams of sublevel sets of cortical thickness were introduced for cortical surface data [16], and persistence-based kernels have been used to build similarity matrices between subjects' cortical thickness maps [17]. This is the defensible version of a "topological distance".
   *Reasoning, not a cited result:* pure topological invariants (Betti numbers) of FreeSurfer-style cortical surfaces are fixed by topology correction and are identical for every healthy brain, so they cannot order the ladder. The ordering must come from geometry that persistence encodes.
   *Unverified gap:* I did not find prior work that uses persistent homology for individual or twin identification from cortical folding. That may be a contribution, but absence of evidence was not established exhaustively.
4. **Learned embeddings**, e.g. NeuroFM, an encoder for T1-weighted MRI that outputs brain age, grey- plus white-matter volume, lateral ventricle volume, sex and an optional latent embedding, trained (per its repository) on 100,000 synthetic healthy volumes [18]. Risk: an embedding tuned to population-level morphometry may be invariant to individual identity and fail at L3. Admitted only if it passes the ladder test (WP2).
5. **Global volumes and mean thickness** — included as a **negative control**, expected to fail at L3 [7].

### 4.3 Normalised distance

For a feature space *f*, structure or region *r*, and a pair of images *(a, b)*:

$$z_r(a,b) = \frac{d\big(f_r(a), f_r(b)\big)}{\max\big(\sigma_{L1,r},\ \tau_{\min}\big)}$$

where σ_L1,r is the across-scanner spread of the same feature in the same region (from L1 data) and τ_min is a floor that prevents division by very small spreads in large, stable structures. By construction, L1 pairs land near z ≈ 1. **Exact zero is neither attainable nor the target**; the target for "same brain" is "inside the scanner-noise band".

### 4.4 Summaries for inpainting (the Dice + HD95 pattern)

Computed inside the inpainting mask and a collar around its boundary:

| Summary | Definition | Captures | Reads as |
|---|---|---|---|
| **TA(k)** | fraction of mask locations with z ≤ k, for k ∈ {1, 2, 3} | bulk agreement | "x% of the synthesized region is within k re-scan noise units" |
| **AUC-TA** | area under TA(k) over k | bulk, non-saturating | single bulk score |
| **z₉₅** | 95th percentile of z over the mask | localised artefacts | unitless worst case — HD95 fixed for its lack of normalisation |
| **Ladder position** | where the pair falls among the empirical L1–L4 distributions | identity | verbal grade |

### 4.5 Statistics of the ladder

- **Discriminability** quantifies how similar an individual's repeated samples are relative to others without assuming univariate, Gaussian or Euclidean data; its authors show that the intraclass correlation coefficient and fingerprinting can fail to separate accidental from systematic deviations even in simple settings [19]. **Primary statistic for each rung contrast.** ICC is reported only for univariate features.
- **Identifiability** from repeated measurements has been maximised for functional connectomes by group-level eigenmode reconstruction [20], and extended to test–retest pairs within and across sites [21]. Analogous analysis applies here to structural feature spaces.
- **Separating L1 from L2:** a crossed random-effects model with subject, scanner and time terms, fitted on cohorts where the same scanner recurs across years (Section 5).

### 4.6 Regional formulation for BraTS masks

Whole-brain fingerprints do not transfer to a few cubic centimetres. The ladder is therefore calibrated **with the same masks** on healthy data: place a BraTS-like mask at the same anatomical location in (same subject, scanner 1 vs 2), (same subject, years apart), (MZ twin), (DZ twin), (unrelated), compute the same regional *z*, and obtain rung distributions per anatomical location. Ground truth exists only for healthy masks [1]; for real tumour regions only the distribution-level ladder applies.

## 5. Data per rung

| Rung / use | Dataset | Origin | Content (verified) |
|---|---|---|---|
| L0, L1 (one person) | **FTHP** [22]; describing paper [4] | Germany | one healthy male, 557 images, 157 sessions with 3–5 repeated scans each, 116 MRI machines |
| L0, L1 (many people) | **ON-Harmony** [23] | UK | 20 healthy participants, each scanned on 6 3T scanners at 5 sites from 3 vendors; 9 of them with 6 within-scanner repeats |
| L1 (many people) | **SRPBS traveling subjects** [24] | Japan | nine healthy traveling subjects, 12 scanners |
| L1 (many people) | **MASiVar** [25] | USA | 14 healthy adults and 83 children across three sites and four scanners; Hays et al. used 5 adults scanned on 3–4 scanners at three institutions [3]. Its diffusion variability rose consistently from intrasession to intersession to interscanner to intersubject, with interscanner variability sometimes approaching intersubject variability |
| L1 (two people, many scanners) | **Chinese travelling subjects, CHIMGEN consortium** [26] | China | two traveling subjects, 28 MR scanners, 995 brain imaging measures; all measures showed inter-scanner heterogeneity, lowest for cortical volume and surface area and highest for functional connectivity. Data availability to check |
| L0 | **Maclaren test–retest** [27] | USA | 3 participants, each scanned 40 times |
| L0 | **Midnight Scan Club** [28] | USA | 10 participants, four T1w and four T2w images each |
| L0 (many people, many sites) | **CoRR** [29] | international (18 sites), coordinated from China and the USA | 1,629 individuals, 5,093 resting-state fMRI scans, 18 international sites, with repeated scans for test–retest reliability |
| L0, L2 | **SLIM** [30] | China | healthy young adults in southwest China rescanned within three and a half years; structural, diffusion and resting-state MRI |
| L1, L2 | **SIMON** [31] | Canada | one healthy volunteer, 73 sessions, 36 scanners (13 models, three vendors), 2001–2018, age 28–46; some scanners recur across 2014–2018 |
| L1, L2 | **Hawco et al.** [32] | Canada, USA | travelling human phantoms scanned annually at three sites: 30 scans, 4 people, 6 MRIs, 4 years |
| robustness, metric validation | **MR-ART** [33] | Hungary | 148 healthy adults, T1w acquired still and with slight and excessive head motion; neuroradiologist artefact scores per scan |
| **L3 (primary); also L0 and L2** | **QTIM** [34]; T1w reference paper [35] | Australia | young adult twins and siblings aged 18–30 plus adolescent twins: 1202 participants in 682 families (445 families of two, 36 of three, one of four); **test–retest subsample of 78** (about 3.5 months); longitudinal subsample of 62 (about 3.5 years); T1w acquisition orientation changed mid-study after a scanner software upgrade (955 coronal, 247 sagittal at session 1). Family membership gives a related-vs-unrelated rung; zygosity gives L3a vs L3b |
| L3a, L3b (fallback) | **QTAB** [36] | Australia | adolescent twins; session 1: 109 MZ and 97 DZ pairs, ages 9–14; session 2: 81 MZ and 71 DZ pairs, ages 10–16; T1w MP2RAGE |
| L3a, L3b (fallback) | **HCP Young Adult** [37] | USA | genotype-verified zygosity (ZygosityGT); 36 pairs self-reported as DZ were genetically MZ |
| L3 (candidate, Latin America) | **TwinsMX** [38] | Mexico | Mexican national twin registry with over 2800 twins; brain MRI has been added to the registry. Imaging availability to check |
| L4 | **IXI** [39] | UK | nearly 600 healthy subjects from three hospitals (Hammersmith, Guy's, Institute of Psychiatry); T1, T2, PD, MRA and diffusion |
| lesions (concept-note Track A) | **LUMIERE** [40]; tool-consistency study [41] | Switzerland | 91 glioblastoma patients, 638 study dates, 2487 images, with automated segmentations; on these data two automated tools matched expert tumour-volume trends in about 80% of cases and failed to count lesions accurately across time |
| lesions (concept-note Track A) | **MSSEG-2** [42] | France | 100 multiple sclerosis patients, each with two time points, 3D FLAIR from different centres and scanners (OFSEP cohort) |
| inpainting | **BraTS inpainting** [1] | international | healthy-tissue masks with ground truth |

Access conditions for these datasets are tracked separately and settled before any data are requested.

**Further candidates by region** (twin and travelling-subject designs; availability to check for all):

| Rung / use | Dataset | Origin | Content (verified) |
|---|---|---|---|
| L3 (adolescent twins) | **Chongqing Adolescent Twin Study (CATS)** [43] | China | 136 typically developing twins aged 12–19; structural, resting-state and diffusion MRI, with image quality benchmarked against HCP Development |
| L1 (diffusion, many people) | **Japanese travelling subjects** [44] | Japan | 300 scans of 69 travelling subjects on 13 MRI models and 2 protocols |
| L3 (older adult twins) | **Osaka twin study** [45] | Japan | 20 MZ and 17 DZ pairs aged 41–85 on 3T MRI; total brain, grey and white matter volume highly heritable |
| L1 (structural and diffusion) | **Travelling subjects of Richter et al.** [46] | UK, Finland | 73 healthy volunteers, 161 scans, two sites, four machines, one T1 and five diffusion protocols |
| L3 (adult twins) | **Hungarian Twin Registry** [47] | Hungary | 118 healthy adult twins with T1w MRI |
| L2 (ageing, many people) | **BLSA** [48] | USA | longitudinal MRI of older adults (92 non-demented adults aged 59–85 at baseline, rescanned at 2 and 4 years in the first longitudinal report); BLSA is part of HACA3⁺'s training data [3], so it is not used to test HACA3⁺ |
| L3 (older male twins) | **VETSA** [5] | USA | 67 MZ and 51 DZ male twin pairs, analysed with BrainPrint |

Targeted searches in English, Chinese, Spanish, Portuguese and Indonesian found no twin or travelling-subject MRI cohort from Taiwan, Korea, Indonesia, India, the rest of Southeast Asia, Brazil or Africa; that absence is not established exhaustively.

## 6. Work packages and milestones

**WP1 — Reproduce the travelling-subject validation of Hays et al. [3] on FTHP and MASiVar [25], and extend it to SIMON, ON-Harmony and SRPBS.**
Per-session segmentation (SLANT labels as in [3], plus FastSurfer and SynthSeg as in [2]); coefficient of variation of regional volume and DSC across sessions, unharmonized vs HACA3 vs HACA3⁺ (public weights). The TREAT-MS inter-site result of [3] — HACA3 and HACA3⁺ statistically indistinguishable — cannot be reproduced because that dataset is private; state this explicitly.
*Output:* L0 and L1 noise per structure, per pipeline, with and without harmonization.

**WP1b — Lesion stability (concept-note Track A).**
Change-detection stability of lesion volume and count across equally valid pipelines and preprocessing configurations on the longitudinal LUMIERE glioblastoma cohort and the two-time-point MSSEG-2 cohort.
*Output:* the detectability floor for lesion-based endpoints, reported as a lower bound.

**WP2 — Build and test the ladder.**
L0/L1 from FTHP, ON-Harmony, SRPBS, MASiVar and the Chinese travelling subjects; dense L0 from Maclaren and the Midnight Scan Club, and multi-site L0 from CoRR and SLIM; multi-subject repeat noise from the 78 QTIM test–retest participants; L2 from SIMON (and Hawco et al.) via the crossed model; L3 from QTIM — a related-vs-unrelated rung from family membership, then MZ vs DZ from zygosity, with QTAB or HCP as fallback and TwinsMX as a Latin American candidate; L4 from IXI and across subjects; motion robustness from MR-ART. Evaluate feature spaces 1–5 of Section 4.2. Compute discriminability per rung contrast, with ON-Harmony and QTIM test–retest as the main multi-subject sets.
*Output:* a table of rung separations per feature space; the selected feature set.
*Go/no-go:* if the MZ distribution overlaps L1 in every feature space, fall back to a three-rung ladder (same brain / related / unrelated) and report the overlap as a result.

**WP3 — Metric for BraTS inpainting.**
Regional operator and calibration with BraTS-like masks on healthy data (Section 4.6); TA(k), AUC-TA, z₉₅, ladder position. Re-rank BraTS inpainting submissions and compare with the SSIM/PSNR/RMSE ranking. Validate against a DQE-style surrogate [14], against the neuroradiologist artefact scores of MR-ART [33], and with a small forced-choice expert study designed around the low agreement of free quality ratings [15].

**WP4 — Optional.** (a) NeuroFM latent ladder test [18], cheap on CPU; (b) a no-reference, density-based branch for real tumour regions where no ground truth exists, kept separate from the reference-based metric because a generative scorer inherits its own biases.

**Targets.** WP1 + WP2: MIDL 2027 (target), with MICCAI 2027 or a BraTS workshop as fallback. WP3: MICCAI / BraTS workshop 2027, or a journal.

## 7. Experimental design

Every analysis works on **pairs of scans labelled by rung**, so one table of pair distances serves the ladder, the harmonization test and the inpainting metric alike. Precedents show the ladder is plausible: individuals can be identified from FreeSurfer anatomy across three scans over two years [49], and multi-modal fingerprints place MZ twins closer than DZ twins and non-twin siblings [50]. What is unknown is how far apart the rungs sit in each feature space, and whether they stay apart once scanner noise is accounted for.

### 7.1 Pair construction

| Pair label | Definition | Main sources |
|---|---|---|
| L0 | same person, same scanner, same session | SIMON repeats, ON-Harmony within-scanner repeats, Maclaren, FTHP |
| L1 | same person, different scanner, under one year apart | FTHP, ON-Harmony, SRPBS, MASiVar, SIMON |
| L2 | same person, at least one year apart, same scanner where available | SIMON, Hawco et al., QTIM and SLIM longitudinal, BLSA |
| L3r / L3a / L3b | related / MZ / DZ pairs from one family | QTIM (CATS, QTAB, HCP as further or fallback cohorts) |
| L4 | unrelated people, matched to L3 pairs on age (±3 years), sex, scanner and acquisition orientation | the same cohorts |

Matching L4 to L3 on demographics and acquisition is what lets a distance difference be read as relatedness rather than as age, sex or protocol.

### 7.2 Statistics

- **Primary statistic:** discriminability [19] for each adjacent rung contrast (L0–L1, L1–L2, L1–L3, L3a–L3b, L3–L4).
- **Inference:** permutation tests that shuffle labels only within exchangeability blocks (subjects for L0–L2, families for L3), and cluster bootstrap over subjects or families for intervals. Twins act as biological ground truth for a distance, as in topological inference on twin networks [51].
- **Per-feature separation index:** difference of median distances between adjacent rungs divided by the pooled median absolute deviation; reported with bootstrap intervals and FDR control across structures and feature spaces.
- **L1 vs L2:** crossed random-effects model with subject, scanner and time on cohorts where scanners recur (SIMON, Hawco et al.).
- **Leakage control:** features and τ_min are chosen on one set of cohorts and families (ON-Harmony, a training split of QTIM families) and evaluated on others (SIMON, SRPBS, held-out QTIM families).

### 7.3 Harmonization arm (WP1)

- **Arms:** native; registration-only (mandatory); ComBat on derived measures [52]; a travelling-subject model that separates measurement bias from sampling bias [53], in the spirit of the travelling-subject GLM that outperformed ComBat on diffusion data [44]; HACA3 [54]; HACA3⁺ [3].
- **Success criterion:** harmonization must shrink L1 towards L0 **without shrinking the L3 and L4 separations** — biological signal retained.
- **Do-no-harm check:** in a travelling-subject cohort without a structural scanner effect, harmonization obscured true biological effects [46]; every arm is therefore compared with native data on the biological contrasts, not only on scanner variance.
- **Contamination:** cohorts in a model's training data (for HACA3⁺: OASIS-3, BLSA, HCP, IXI) are excluded from that model's evaluation.

### 7.4 Inpainting-metric validation (WP3)

- **Donor-swap calibration (proposed here):** on healthy scans, fill a mask with tissue taken from the same person on another scanner, from a twin, and from an unrelated person, after registration. A valid metric must order these fills as re-scan < twin < unrelated; SSIM, PSNR and RMSE are tested for the same ordering.
- **Downstream-equivalence precedent:** lesion filling has been evaluated by the bias it leaves in downstream morphometry [55,56]; the proposed metric generalizes that idea with a calibrated unit.
- **Controlled degradations:** blur, intensity shift, misalignment and stripe artefacts at graded strengths. SSIM and PSNR are known to be insensitive or oversensitive to specific distortions [57], so each metric's response curve is reported.
- **Normalization alternatives:** z in units of σ_L1 (primary) and centile scores against normative lifespan charts [58] (secondary), reported side by side.
- **Human reference:** a DQE-style surrogate [14], MR-ART neuroradiologist artefact scores [33] and a small forced-choice reader study.

### 7.5 Controls

- **Positive controls:** an image compared with itself (d = 0); synthetic noise at the measured L0 level; the donor swaps above.
- **Negative control:** global volume and mean thickness (H3).
- **Confounds modelled:** acquisition orientation (QTIM), age and development (QTAB, CATS), head motion (MR-ART), registration quality, and training-data overlap.
- **Power:** simulated from the σ estimates of E0 and E1 before the twin analyses are run, not assumed.

## 8. Lessons from other fields and metric design rules

Several fields have had to judge a distance between objects that differ between instances and change over time against the noise of measuring them. Their solutions share a pattern: a floor from repeat measurements of the same object, a ceiling or null from random pairs, a calibrated scale between the two, a curve over tolerances instead of a single threshold, and validation against an external criterion. The rules in Section 8.3 extend the metric design of Section 4.

### 8.1 Where the segmentation precedents come from

- The Jaccard overlap coefficient comes from plant ecology: Jaccard compared the floras of alpine districts as the number of shared species over all species of the two places [59].
- The Hausdorff distance was used to compare a model with a binary image at all relative positions, without point correspondences and tolerating small position errors such as those of edge detectors [60]. Taha and Hanbury attribute the quantile version (the q-th quantile of distances instead of the maximum) to the same group and advise against using the plain Hausdorff distance directly because of its outlier sensitivity [61].
- The 95th-percentile Hausdorff distance entered segmentation challenges with explicit outlier rationales. In PROMISE12 (100 prostate MRI cases from four centres) a single pixel could decide the plain Hausdorff distance [62]. In BRATS the maximum was too susceptible to small outlying subregions and to false-positive regions that postprocessing would remove [63].
- Surface Dice was introduced for radiotherapy contouring, with an organ-specific tolerance taken from the interobserver variation of three consultant oncologists, to reflect the task of correcting automated contours [11]. Against the time needed to adjust thoracic auto-contours, surface Dice correlated more strongly (R = −0.69) than volumetric Dice (R = −0.32), and a new added-path-length measure most strongly (R = 0.87) [64].
- Challenge scoring already anchors scores to human variability. PROMISE12 maps each metric so that a perfect value scores 100 and the second observer's average scores 85, floored at 0, ranking in a similar way to the MICCAI 2007 liver challenge [62,65]. The WMH challenge ranks robustness across scanners separately, from the standard deviation of each method's per-scanner median [66].

### 8.2 Analogues in other fields

| Field | Device | Transfer to the identity ladder |
|---|---|---|
| Protein structure comparison | RMSD normalised to 100 residues with shuffled decoys [67]; GDT, the share of residues within 1, 2, 4 and 8 Å, averaged [68]; TM-score with a size-dependent distance scale [69] and an extreme-value null for random pairs that gives p = 5.5 × 10⁻⁷ at TM-score 0.5 [70]; lDDT, superposition-free, with tolerances widened to the range of several reference structures [71] | size-independent L4 baseline; mean over tolerances (AUC-TA); p-value under the L4 null; tolerance from several repeat scans of one subject |
| Noise and relatedness in protein structures | crystal-to-NMR differences exceeded the spread within the NMR ensemble in all 148 pairs [72]; in regions that keep the same fold, structural divergence of homologues grows exponentially with sequence divergence, and these regions shrink for distant homologues [73] | expect and measure L1 > L0; treat relatedness as a curve as well as discrete rungs |
| Cheminformatics | the conventional Tanimoto ≥ 0.85 cut-off gave only about a 30% chance that a similar compound is also active [74]; a statistical null gives Z-scores and p-values for similarity scores [75] | publish the empirical meaning of every z threshold |
| Metrology and quantitative imaging | repeatability, intermediate-precision and reproducibility conditions, where intermediate precision allows new measuring systems at the same location [76]; repeatability and reproducibility coefficients as least significant differences [77]; agreement judged on differences rather than correlation, and limited by each method's repeatability [78] | split L1 into same-site and cross-site pairs; take tolerances from the empirical L1 distribution |
| Analytical chemistry (proficiency testing) | z-scores scale a laboratory's deviation from the assigned value [79], with z within ±2 acceptable and z at or beyond ±3 unacceptable in a typical scheme [80]; the standard deviation for proficiency assessment can be set from method precision, previous rounds, regulatory or expert criteria, a general model or same-round data [81]; the HorRat ratio of observed to predicted inter-laboratory scatter, with predicted scatter a function of concentration, flags both high and suspiciously low values [82], although recent data no longer follow the predicting equation [83] | z = d / σ_L1 is a proficiency z-score: state how σ is set, make it depend on region size, flag z far below the L0 floor, recalibrate on current scanners |
| Biometrics | same-eye versus different-eye distance distributions with a threshold-free decidability index; the same-eye distribution moved strongly with acquisition while the different-eye distribution hardly did; left and right eyes of one person and eyes of monozygotic twins scored like unrelated eyes [84], although human observers saw similarity in such pairs [85]; genuine fingerprint scores fall with the time between captures while impostor scores change negligibly [86] | report full rung distributions and d′; σ per scanner pair; test that L3 does not collapse onto L4; model L2 as a function of interval |
| Genetics | kinship classes from monozygotic twins to unrelated pairs with boundaries derived on a powers-of-2 scale, robust to population structure [87]; benchmark accuracy fell from 92–99% for first- and second-degree relatives to below 43% for seventh-degree relatives [88] | explicit rung boundaries; lower resolution expected for distant rungs; robustness to site or vendor groups |
| Geometric morphometrics | Procrustes ANOVA splits variation among individuals, asymmetry and digitising error within one shape space [89]; ratios against morphologically dissimilar subjects can make measurement error look unimportant [90] | one nested variance model in the units of d; do not judge noise against L4 alone |
| Weather forecasting | skill relative to a reference forecast, with conditional and unconditional bias terms [91]; equitability, under which all constant and random forecasts receive the same expected score, scaled to 0, while a perfect forecast scores 1 [92], a property popular scores meet only asymptotically [93]; skill as a function of neighbourhood size [94]; Lorenz's hypothesis that small initial differences grow until two states differ like randomly chosen states [95], with a predictability limit at 95% of saturation per scale [96]; idealised test fields, on which every analysed distance measure performed poorly in pathological cases [97] | two-anchor score with an equitability check at the real sample size; tolerance curve; synthetic test suite with degenerate masks |
| Remote sensing | radiometric normalisation fitted on no-change pixels chosen by a chi-square no-change probability and tested on withheld no-change pixels [98] | harmonise on healthy tissue outside the mask and test on held-out tissue |
| Particle physics | an optimal-transport metric between collider events [99]; detector effects measured in the same units [100] | σ_L1 as the size of a known perturbation in the metric's own units |

### 8.3 Design rules adopted

1. **Two anchors.** Report S = (d_L4 − d) / (d_L4 − d_L0) next to z: a repeat scan scores 1, an unrelated brain 0, and a fill worse than an unrelated brain scores below 0. Check that a population template or a blurred average does not earn S > 0, and compute the expected S of random L4 pairs at the actual number of regions [92,93].
2. **Tolerance curve** in units of σ_L1 (for example 0.5, 1, 2 and 4 × σ_L1), with its mean as AUC-TA [11,68,71,94].
3. **σ per structure, per region size and per scanner pair** [11,69,82,84].
4. **A fitted L4 null and explicit rung boundaries**, with p-values and, for each threshold, the observed pass rate of L0/L1 pairs and fail rate of L3/L4 pairs [70,74,87].
5. **Multi-reference tolerance** when a subject has several repeat scans [71].
6. **One nested random-effects model** (subject × scanner × session × family), with L1 split into same-site and cross-site pairs [76,89].
7. **Signed bias reported separately from scatter** [78,91].
8. **The twin rung is tested, not assumed** [84,85].
9. **Harmonisation fitted on tissue outside the mask** and checked on held-out tissue [98].
10. **A synthetic test suite** with degenerate and tiny masks [97], and a flag for z far below the L0 floor [82], with the floor re-estimated on current scanners, because modern measurements can routinely beat an outdated predicted precision [83].

## 9. Falsifiable hypotheses

- **H1:** σ_L0 < σ_L1 for every structure and pipeline (cf. [4]).
- **H2:** in spectral shape space, L1 < L2 < L4 on SIMON-scale time spans.
- **H3 (negative control):** in global volume and mean-thickness space, MZ twins are not separable from L1.
- **H4:** in a selected folding, spectral or persistence space, L1 < L3a < L3b < L4.
- **H5:** the L2 vs L3a order is feature-dependent; it is reported, not assumed.
- **H6:** the ranking of BraTS inpainting methods by AUC-TA and z₉₅ differs from the SSIM/PSNR/RMSE ranking, and fills with visible artefacts land at or above the twin rung.

### 9.1 Baseline metrics on the ladder (added 1 November 2026; every item is a hypothesis to test)

Every baseline is one cell of a grid, feature space × comparison metric, so that a failure can be traced either to the features or to the comparison.

- **Feature spaces:** F0, a nuisance-only control (global mean and SD of intensity); F1, raw pixels or voxels; F2, handcrafted domain features (iris code, minutiae, vessel graph, regional volumes); F3, generic pretrained encoders; F4, domain foundation models; F5, identity-trained matchers.
- **Comparison metrics by family:** pixel fidelity (MSE, RMSE, MAE, MSLE, PSNR, normalised cross-correlation, mutual information); structural and perceptual (SSIM, MS-SSIM, LPIPS, DISTS); distribution-level (FID, KID, MMD), which score populations rather than pairs; identity scores (embedding cosine, iris-code Hamming distance, matcher scores); biometric operating metrics (EER, FMR at a fixed FNMR, d′, ROC AUC, rank-1); reliability (ICC, within-subject coefficient of variation, repeatability coefficient, Dice between re-scans); identifiability (fingerprinting rank-1, differential identifiability, discriminability, I2C2); relatedness (MZ and DZ correlations, heritability, kinship-verification accuracy); normalised scores (z against a reference, cohort score normalisation, the two-anchor S of Section 8.3, and c(d) = P(d_stranger > d) with a stranger reference matched on the pair's nuisance condition). Which family is the standard in which field is being confirmed against sources.

**The ladder exam.** Each cell is scored on seven tests.

- **T1, order:** L0 < L1 < L4 and L2 < L4 must hold, because the same person is closer than a stranger. The positions of L3 and of the mirror rung are measured, not graded. Scored by the AUC between adjacent rungs and the number of inversions.
- **T2, anchors:** the values at L0 and at L4 across devices, datasets, mask sizes and modalities.
- **T3, nuisance:** (a) the same metric on F0 must not reproduce the ladder; (b) the ratio (L1 − L0)/(L4 − L0), a device change against a change of person; (c) the shift between a nuisance-matched and a pooled stranger reference.
- **T4, resolution:** d′ and AUC of adjacent rungs with intervals, and the number of pairs needed.
- **T5, known answer:** synthetic degradations (blur, noise, compression, intensity gain, sub-pixel shift) move the score monotonically within the L0–L1 band; an identity swap lands at L4; donor-swap fills order re-scan < twin < stranger (Section 7.4).
- **T6, comparability:** one rung gives one value across modalities, for example the fellow eye in iris and in fundus.
- **T7, consequences:** whether the ranking of methods (inpainting, face avatars derived from MRI) changes between a baseline and the ladder (H6).

**Hypotheses on baselines.**

- **H7:** raw-scale metrics (SSIM, PSNR, MSE, MAE) fail T2: their values at L0 and at L4 differ between datasets, devices and mask sizes, so a raw value cannot be read as a distance in identity.
- **H8 (negative control):** on 2D images, pixel distances fail T3a. Pilot evidence: in UPOL, a feature made only of the mean and SD of grey level gave c = 0.919 for the same eye and 0.884 for the fellow eye, against 0.923 and 0.805 for 16 × 16 pixels; in UTIRIS NIR the control gave 0.793 and 0.764, and the classical iris code 0.913 for the same eye (`results/iris_calibration.json`, `results/iris_codes.json`).
- **H9:** a pooled stranger reference biases c by about one rung. Pilot evidence (GRAPE, RETFound features): with a reference pooled over camera pairs, a camera change cost 0.260 [0.183, 0.329] in c; with a reference matched on the device pair and the eye relation, 0.096 [0.048, 0.146] (`results/audit_2026-11-01.json`).
- **H10:** in MRI geometry features, a protocol change moves a pair about as far as a change of person (T3b ratio near 1). Pilot evidence: median distance 18.4 for SIMON pairs across scanner and protocol, 16.4 for IXI strangers scanned at the same centre, and 7.4 for repeats within one session (`results/image_ladder.json`; raw medians, not calibrated).

**The BraTS inpainting panel.** The challenge code [101] normalises the reference and the fill to [0, 1] with the 0.5th and 99.5th percentiles of the voided input, clipping values outside that range. It then reports SSIM (the SSIM map of the masked images, averaged inside the mask) and, on the voxels inside the mask, MSE, RMSE, MAE, MSLE, PSNR, PSNR_eps, PSNR_01 and PSNR_01_eps. PSNR and PSNR_eps take the data range from the reference inside the mask, the _01 variants fix it to [0, 1], and the _eps variants add machine epsilon to the MSE.

- **H11 (redundancy):** within a case, the data range of PSNR and PSNR_eps comes from the reference alone and is the same for every fill, so MSE, RMSE, PSNR_eps, PSNR_01 and PSNR_01_eps are monotone functions of one another and rank fills identically; they can differ only after averaging over cases. Plain PSNR follows the same order except where a fill leaves the reference's range inside the mask, because torchmetrics clamps the input to a tuple data range (Section 11). The challenge ranks the teams within each case on SSIM, PSNR and RMSE and sums the three metric ranks with equal weights [1], so the final ranking is expected to weight squared error about twice and SSIM once. Hypothesis: the panel carries at most five per-case signals (the MSE family, plain PSNR through its clamping, MAE, MSLE and SSIM), and MSLE tracks MSE closely on [0, 1] data. Test: per-case rank agreement of PSNR and RMSE, and rank correlations of the whole panel across donor-swap fills.
- **H12 (blur beats a stranger):** the MSE family and PSNR score a blurred or population-average fill better than real tissue from a stranger, and possibly better than tissue from a co-twin, because squared error rewards the conditional mean; such fills would then sit between L1 and L3 on the donor-swap ladder although they carry no individual anatomy (compare design rule 1 in Section 8.3). Test: T5 with the reference blurred at increasing σ, a template fill and donor fills.
- **H13 (PSNR range):** because PSNR and PSNR_eps take their data range from the reference inside the mask, the same error gives a lower PSNR in a homogeneous region than in a region of high contrast, so PSNR fails T2 across mask locations while PSNR_01 does not. Test: one fixed degradation applied in masks at different locations.
- **H14 (SSIM at the mask border):** SSIM is computed on images set to zero outside the mask, so windows that cross the mask border compare zeros with zeros; SSIM is expected to rise as masks become smaller or thinner at a constant fill quality (T2 across mask sizes). Because the code computes SSIM per 2D slice (Section 11), the effect should depend on the mask's extent within the slice plane, not across slices.
- **H15 (clipping):** both images are clipped at the 99.5th percentile of the voided input, so hyperintense artefacts above it are capped and under-penalised by every metric of the panel. Plain PSNR in addition clamps the fill to the reference's range inside the mask, so an overshoot beyond that range is forgiven by PSNR but not by RMSE. Test: T5 with bright artefacts of increasing intensity, inside and outside the reference's range.

### 9.2 Cross-modal hypotheses (added 1 November 2026)

- **H16 (five axes, 2D first):** every modality is analysed on five axes: bilaterality, test–retest, twins, ageing and unrelated people. Analyses start with 2D modalities that have one fixed acquisition geometry (iris, fundus, fingerprints, palms, ears, frontal faces), where L1 is expected to be small against L4; 3D MRI, where protocol change dominates (H10), follows.
- **H17 (the contralateral organ as a twin inside one person):** a fellow eye, the contralateral finger or the other ear shares genome, age and environment with its partner but develops separately, so the expected order is same side < contralateral side ≤ MZ co-twin < DZ co-twin < stranger. Where the contralateral rung falls relative to the MZ rung is measured per modality.
- **H18 (L3 is modality-specific):** in iris and fingerprint texture the MZ rung is expected near L4, in faces near L0–L1. A metric can be right in both modalities while giving opposite L3 positions, which is why T1 grades only L0 < L1 < L4 and L2 < L4.

- **H19 (pointwise downstream metrics inherit the bias):** under squared error an identity-free population-mean fill has half the expected error of a stranger's tissue and beats an MZ co-twin's tissue when the genetic share is below one half (Proposition 5 of the ISBI draft; synthetic check: rho-hat 0.50 vs 0.504 predicted). Hypothesis: the same holds in label space, because Dice or voxel accuracy against the oracle segmentation rewards the most probable (population-typical) label, so a blurred or average fill segments closer to the oracle than realistic donor tissue from a stranger and possibly from a co-twin. Kate's observation from the BraTS inpainting challenge: blurry fills won on the intensity metrics, and the downstream FastSurfer check also favoured them. Test: donor-swap in label space (re-scan, mirror, co-twin, stranger, population mean, blur) on synthetic tissue with a threshold segmenter, then MRI with FastSurfer; topological descriptors of the segmentation (component and hole counts, Euler characteristic, boundary curvature) and population-level realism are expected to separate the average fill.
  - First synthetic test (1 Nov 2026, `src/label_ladder.py`, `results/label_ladder.json`; two tissue classes from a population prior plus individual fields, threshold segmenter, c per case against 16 stranger fills): with genetic share 0.19 the population-mean fill reached Dice c 0.90 against 0.61 for the MZ and 0.68 for the DZ co-twin (RMSE 0.99 vs 0.66; SSIM 0.99 vs 0.70); with genetic share 0.54 the MZ co-twin stayed above it (Dice 0.95 vs 0.91). So H19 holds in label space in this model. Topology did not separate the average fill here (component-count error 0.8, as for the other side; strangers 1.1-1.4), probably because the prior dominates the label maps; a folding-like anatomy is needed for the topology test.

### 9.3 First test: baseline v0 on the 2D pilots

- **Data:** UTIRIS NIR and UPOL (iris), GRAPE (fundus).
- **Cells:** F0, F1, F2 (iris code) and F4 (the released RETFound features for GRAPE), each with the BraTS panel, normalised cross-correlation, mutual information, cosine or Hamming distance, EER, d′, AUC, rank-1, ICC, differential identifiability, discriminability, z, S and c.
- **2D donor-swap:** on unrolled iris strips, which are aligned by construction, a patch is masked and filled from the same eye in another capture, from the fellow eye (flipped), from a stranger, and with blur, a population template and an intensity gain; the panel is computed inside the mask with the challenge code [101]. This gives T5 and H11–H15 a 2D test before MRI.
- **Tests:** T1–T6. T7 needs a generator and comes with the avatar benchmark.

## 10. Risks

1. **Reliability vs individuality** (Section 3): the features that separate twins may be the noisiest.
2. **Thin L1 and L2 estimates:** FTHP and SIMON are one subject each, SRPBS nine and ON-Harmony twenty; the 78 QTIM test–retest participants add breadth for repeat noise within one study, not for scanner variation. σ must be pooled per structure, not per voxel, and reported with intervals.
3. **Acquisition confounds in the twin cohort:** QTIM changed its T1w acquisition orientation mid-study, and the dataset documentation notes a vascular pulsation artefact that varies with orientation [34]; twin pairs must be compared within one orientation or the orientation modelled. QTIM's young adults (18–30) reduce the age mismatch with the adult L1 and L2 cohorts. If the fallback QTAB is used, its twins are adolescents and its MP2RAGE sequence was reissued during session 2 [36].
4. **Registration confound** for between-subject comparisons: either registration-free features or registration variability folded into the noise band.
5. **τ_min choice** changes z in stable structures; report sensitivity.
6. **Ground truth only for healthy masks** [1].
7. **Expert reference is itself noisy** [15]; the reader study validates ordering, not absolute agreement.

## 11. Open items (not citation issues)

- The PMLR version of Hays et al. lists 14 authors; the arXiv version lists 16. The first six, used in the reference below, are identical.
- Hays et al. give 14 TREAT-MS participants in the text and 15 subjects in their validation table [3].
- The FTHP numbers in the describing paper (477 volumes, 123 scanners [4]) differ from the released dataset (557 images, 116 machines [22]); the released set is what WP1 uses.
- NeuroFM is described here only from its repository page; its preprint was not read.
- Settled on 1 November 2026 with synthetic volumes (`src/brats_panel_checks.py`; `results/brats_panel_checks_tm1.1.2.json` and `results/brats_panel_checks_tm1.9.0.json`, identical under both torchmetrics versions):
  - The challenge code [101] computes SSIM per 2D slice. Volumes of shape [1, 240, 240, 155] enter torchmetrics SSIM, whose documentation describes B × C × H × W inputs; permuting the first spatial axis in every input left SSIM unchanged, while the same permutation along the last axis changed it.
  - Plain PSNR clamps the fill to the reference's in-mask range, because the code passes that range to torchmetrics as a tuple data range. With 20% of the fill voxels above the reference's in-mask maximum, plain PSNR was 12.3 dB, equal to PSNR from the clamped error, against 9.1 dB for PSNR_eps from the unclamped error. Across 11 synthetic fills of one case, MSE, PSNR_eps, PSNR_01 and PSNR_01_eps ranked the fills exactly as RMSE did (Kendall τ = 1), plain PSNR and MSLE with τ = 0.89, MAE and SSIM with τ = 0.67 (H11, H15).

---

## References

1. Kofler F, Meissen F, Steinbauer F, Graf R, Ehrlich SK, Reinke A, et al. The Brain Tumor Segmentation (BraTS) challenge: local synthesis of healthy brain tissue via inpainting [Preprint]. arXiv:2305.08992. 2023.
2. Kondrateva E, Mohamed AZ, Barg S, Kofler F. Benchmarking the reproducibility of brain tissue segmentation across MRI scanners. In: Proceedings of the 9th International Conference on Medical Imaging with Deep Learning. Proceedings of Machine Learning Research. Vol 315. PMLR; 2026. p. 4240–71.
3. Hays SP, Zuo L, Chaudhary MFA, Bartz KM, Remedios SW, Zhang J, et al. Harmonizing MR images across 100+ scanners: multi-site validation with traveling subjects and real-world protocols. In: Proceedings of the 9th International Conference on Medical Imaging with Deep Learning. Proceedings of Machine Learning Research. Vol 315. PMLR; 2026. p. 4703–21.
4. Opfer R, Krüger J, Spies L, Ostwaldt AC, Kitzler HH, Schippling S, et al. Automatic segmentation of the thalamus using a massively trained 3D convolutional neural network: higher sensitivity for the detection of reduced thalamus volume by improved inter-scanner stability. Eur Radiol. 2023;33(3):1852–61. doi:10.1007/s00330-022-09170-y
5. Wachinger C, Golland P, Kremen W, Fischl B, Reuter M. BrainPrint: a discriminative characterization of brain morphology. Neuroimage. 2015;109:232–48. doi:10.1016/j.neuroimage.2015.01.032
6. Hogstrom LJ, Westlye LT, Walhovd KB, Fjell AM. The structure of the cerebral cortex across adult life: age-related patterns of surface area, thickness, and gyrification. Cereb Cortex. 2013;23(11):2521–30. doi:10.1093/cercor/bhs231
7. Panizzon MS, Fennema-Notestine C, Eyler LT, Jernigan TL, Prom-Wormley E, Neale M, et al. Distinct genetic influences on cortical surface area and cortical thickness. Cereb Cortex. 2009;19(11):2728–35. doi:10.1093/cercor/bhp026
8. Le Guen Y, Auzias G, Leroy F, Noulhiane M, Dehaene-Lambertz G, Duchesnay E, et al. Genetic influence on the sulcal pits: on the origin of the first cortical folds. Cereb Cortex. 2018;28(6):1922–33. doi:10.1093/cercor/bhx098
9. Pizzagalli F, Auzias G, Yang Q, Mathias SR, Faskowitz J, Boyd JD, et al. The reliability and heritability of cortical folds and their genetic correlations across hemispheres. Commun Biol. 2020;3(1):510. doi:10.1038/s42003-020-01163-1
10. van der Meer D, Kaufmann T, Shadrin AA, Makowski C, Frei O, Roelfs D, et al. The genetic architecture of human cortical folding. Sci Adv. 2021;7(51):eabj9446. doi:10.1126/sciadv.abj9446
11. Nikolov S, Blackwell S, Zverovitch A, Mendes R, Livne M, De Fauw J, et al. Clinically applicable segmentation of head and neck anatomy for radiotherapy: deep learning algorithm development and validation study. J Med Internet Res. 2021;23(7):e26151. doi:10.2196/26151
12. Maier-Hein L, Reinke A, Godau P, Tizabi MD, Buettner F, Christodoulou E, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods. 2024;21(2):195–212. doi:10.1038/s41592-023-02151-z
13. Kofler F, Ezhov I, Isensee F, Balsiger F, Berger C, Koerner M, et al. Are we using appropriate segmentation metrics? Identifying correlates of human expert perception for CNN training beyond rolling the DICE coefficient. Mach Learn Biomed Imaging. 2023;2023:27–71. doi:10.59275/j.melba.2023-dg1f
14. Kofler F, Ezhov I, Fidon L, Horvath I, de la Rosa E, LaMaster J, et al. Deep quality estimation: creating surrogate models for human quality ratings. In: Brainlesion: glioma, multiple sclerosis, stroke and traumatic brain injuries. Lecture Notes in Computer Science. Cham: Springer; 2023. p. 3–13. doi:10.1007/978-3-031-33842-7_1
15. Hoebel KV, Bridge CP, Ahmed S, Akintola O, Chung C, Huang RY, et al. Expert-centered evaluation of deep learning algorithms for brain tumor segmentation. Radiol Artif Intell. 2024;6(1):e220231. doi:10.1148/ryai.220231
16. Chung MK, Bubenik P, Kim PT. Persistence diagrams of cortical surface data. In: Information processing in medical imaging. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer; 2009. p. 386–97. doi:10.1007/978-3-642-02498-6_32
17. Pachauri D, Hinrichs C, Chung MK, Johnson SC, Singh V. Topology-based kernels with application to inference problems in Alzheimer's disease. IEEE Trans Med Imaging. 2011;30(10):1760–70. doi:10.1109/TMI.2011.2147327
18. NeuroFM [Internet]. GitHub; [cited 2026 Oct 18]. Available from: https://github.com/rockNroll87q/NeuroFM
19. Bridgeford EW, Wang S, Wang Z, Xu T, Craddock C, Dey J, et al. Eliminating accidental deviations to minimize generalization error and maximize replicability: applications in connectomics and genomics. PLoS Comput Biol. 2021;17(9):e1009279. doi:10.1371/journal.pcbi.1009279
20. Amico E, Goñi J. The quest for identifiability in human functional connectomes. Sci Rep. 2018;8(1):8254. doi:10.1038/s41598-018-25089-1
21. Bari S, Amico E, Vike N, Talavage TM, Goñi J. Uncovering multi-site identifiability based on resting-state functional connectomes. Neuroimage. 2019;202:115967. doi:10.1016/j.neuroimage.2019.06.045
22. Frequently traveling human phantom (FTHP) MRI dataset [Internet]. NITRC; [cited 2026 Oct 18]. Available from: https://www.nitrc.org/projects/fthp/
23. Warrington S, Torchi A, Mougin O, Campbell J, Ntata A, Craig M, et al. A multi-site, multi-modal travelling-heads resource for brain MRI harmonisation. Sci Data. 2025;12(1):609. doi:10.1038/s41597-025-04822-2
24. Tanaka SC, Yamashita A, Yahata N, Itahashi T, Lisi G, Yamada T, et al. A multi-site, multi-disorder resting-state magnetic resonance image database. Sci Data. 2021;8(1):227. doi:10.1038/s41597-021-01004-8
25. Cai LY, Yang Q, Kanakaraj P, Nath V, Newton AT, Edmonson HA, et al. MASiVar: multisite, multiscanner, and multisubject acquisitions for studying variability in diffusion weighted MRI. Magn Reson Med. 2021;86(6):3304–20. doi:10.1002/mrm.28926
26. Xiao X, Liu J, Guo L, Xue K, Wang S, Liu F, et al. Mitigating inter-scanner heterogeneity in brain MRI data: assessing its impact on association analyses and the effectiveness of ComBat harmonization in multi-site neuroimaging studies. Neuroimage. 2026;325:121642. doi:10.1016/j.neuroimage.2025.121642
27. Maclaren J, Han Z, Vos SB, Fischbein N, Bammer R. Reliability of brain volume measurements: a test-retest dataset. Sci Data. 2014;1(1):140037. doi:10.1038/sdata.2014.37
28. Gordon EM, Laumann TO, Gilmore AW, Newbold DJ, Greene DJ, Berg JJ, et al. The Midnight Scan Club (MSC) dataset [dataset]. OpenNeuro; 2023. doi:10.18112/openneuro.ds000224.v1.0.4
29. Zuo XN, Anderson JS, Bellec P, Birn RM, Biswal BB, Blautzik J, et al. An open science resource for establishing reliability and reproducibility in functional connectomics. Sci Data. 2014;1:140049. doi:10.1038/sdata.2014.49
30. Liu W, Wei D, Chen Q, Yang W, Meng J, Wu G, et al. Longitudinal test-retest neuroimaging data from healthy young adults in southwest China. Sci Data. 2017;4:170017. doi:10.1038/sdata.2017.17
31. Duchesne S, Dieumegarde L, Chouinard I, Farokhian F, Badhwar A, Bellec P, et al. Structural and functional multi-platform MRI series of a single human volunteer over more than fifteen years. Sci Data. 2019;6(1):245. doi:10.1038/s41597-019-0262-8
32. Hawco C, Dickie EW, Herman G, Turner JA, Argyelan M, Malhotra AK, et al. A longitudinal multi-scanner multimodal human neuroimaging dataset. Sci Data. 2022;9(1):332. doi:10.1038/s41597-022-01386-3
33. Nárai Á, Hermann P, Auer T, Kemenczky P, Szalma J, Homolya I, et al. Movement-related artefacts (MR-ART) dataset [dataset]. OpenNeuro; 2022. doi:10.18112/openneuro.ds004173.v1.0.2
34. Strike LT, Blokland GAM, Hansell NK, Martin NG, Toga AW, Thompson PM, et al. Queensland Twin IMaging (QTIM) [dataset]. OpenNeuro; 2026. doi:10.18112/openneuro.ds004169.v1.0.8
35. Strike LT, Hansell NK, Couvy-Duchesne B, Thompson PM, de Zubicaray GI, McMahon KL, et al. Genetic complexity of cortical structure: differences in genetic and environmental factors influencing cortical surface area and thickness. Cereb Cortex. 2019;29(3):952–62. doi:10.1093/cercor/bhy002
36. Strike LT, Hansell NK, Chuang KH, Miller JL, de Zubicaray GI, Thompson PM, et al. The Queensland Twin Adolescent Brain Project, a longitudinal study of adolescent brain development. Sci Data. 2023;10(1):195. doi:10.1038/s41597-023-02038-w
37. Human Connectome Project. 1200 subjects data release [Internet]. WU-Minn HCP Consortium; [cited 2026 Oct 18]. Available from: https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release
38. García-Vilchis B, Román-López TV, Ramírez-González D, López-Camaño XJ, Murillo-Lechuga V, Díaz-Téllez X, et al. TwinsMX: exploring the genetic and environmental influences on health traits in the Mexican population. Twin Res Hum Genet. 2024;27(2):85–96. doi:10.1017/thg.2024.18
39. IXI dataset [Internet]. Biomedical Image Analysis Group, Imperial College London; [cited 2026 Oct 18]. Available from: https://brain-development.org/ixi-dataset/
40. Suter Y, Knecht U, Valenzuela W, Notter M, Hewer E, Schucht P, et al. The LUMIERE dataset: longitudinal glioblastoma MRI with expert RANO evaluation. Sci Data. 2022;9(1):768. doi:10.1038/s41597-022-01881-7
41. Suter Y, Notter M, Meier R, Loosli T, Schucht P, Wiest R, et al. Evaluating automated longitudinal tumor measurements for glioblastoma response assessment. Front Radiol. 2023;3:1211859. doi:10.3389/fradi.2023.1211859
42. Commowick O, Cervenansky F, Cotton F, Dojat M. MSSEG-2 challenge proceedings: multiple sclerosis new lesions segmentation challenge using a data management and processing infrastructure. In: MICCAI 2021 – 24th International Conference on Medical Image Computing and Computer Assisted Intervention; 2021 Sep; Strasbourg, France. 126 p. Available from: https://inria.hal.science/hal-03358968
43. Zhu Y, Fu Y, Han J, Wang R, Ma X, Hu X, et al. The Chongqing Adolescent Twin Study: an integrative multimodal brain imaging and non-imaging dataset. Sci Data. 2025;12(1):1211. doi:10.1038/s41597-025-05449-z
44. Saito Y, Kamagata K, Andica C, Maikusa N, Uchida W, Takabayashi K, et al. Traveling subject-informed harmonization increases reliability of brain diffusion tensor and neurite mapping. Aging Dis. 2023;15(6):2770–85. doi:10.14336/AD.2023.1020
45. Lukies MW, Watanabe Y, Tanaka H, Takahashi H, Ogata S, Omura K, et al. Heritability of brain volume on MRI in middle to advanced age: a twin study of Japanese adults. PLoS One. 2017;12(4):e0175800. doi:10.1371/journal.pone.0175800
46. Richter S, Winzeck S, Correia M, Kornaropoulos E, Manktelow A, Outtrim J, et al. Validation of cross-sectional and longitudinal ComBat harmonization methods for magnetic resonance imaging data on a travelling subject cohort. Neuroimage Rep. 2022;2(4):100136. doi:10.1016/j.ynirp.2022.100136
47. Alijanpourotaghsara A, Vessal A, Alijanpour A, Strelnikov D, Piroska M, Persely A, et al. Morphometric analysis of neocortical and infratentorial structures: genetic and environmental insights from a twin neuroanatomical study. Medicina (Kaunas). 2025;61(2):261. doi:10.3390/medicina61020261
48. Resnick SM, Pham DL, Kraut MA, Zonderman AB, Davatzikos C. Longitudinal magnetic resonance imaging studies of older adults: a shrinking brain. J Neurosci. 2003;23(8):3295–301. doi:10.1523/JNEUROSCI.23-08-03295.2003
49. Valizadeh SA, Liem F, Mérillat S, Hänggi J, Jäncke L. Identification of individual subjects on the basis of their brain anatomical features. Sci Rep. 2018;8(1):5611. doi:10.1038/s41598-018-23696-6
50. Kumar K, Toews M, Chauvin L, Colliot O, Desrosiers C. Multi-modal brain fingerprinting: a manifold approximation based framework. Neuroimage. 2018;183:212–26. doi:10.1016/j.neuroimage.2018.08.006
51. Chung MK, Lee H, DiChristofano A, Ombao H, Solo V. Exact topological inference of the resting-state brain networks in twins. Netw Neurosci. 2019;3(3):674–94. doi:10.1162/netn_a_00091
52. Fortin JP, Cullen N, Sheline YI, Taylor WD, Aselcioglu I, Cook PA, et al. Harmonization of cortical thickness measurements across scanners and sites. Neuroimage. 2018;167:104–20. doi:10.1016/j.neuroimage.2017.11.024
53. Yamashita A, Yahata N, Itahashi T, Lisi G, Yamada T, Ichikawa N, et al. Harmonization of resting-state functional MRI data across multiple imaging sites via the separation of site differences into sampling bias and measurement bias. PLoS Biol. 2019;17(4):e3000042. doi:10.1371/journal.pbio.3000042
54. Zuo L, Liu Y, Xue Y, Dewey BE, Remedios SW, Hays SP, et al. HACA3: a unified approach for multi-site MR image harmonization. Comput Med Imaging Graph. 2023;109:102285. doi:10.1016/j.compmedimag.2023.102285
55. Prados F, Cardoso MJ, Kanber B, Ciccarelli O, Kapoor R, Gandini Wheeler-Kingshott CAM, et al. A multi-time-point modality-agnostic patch-based method for lesion filling in multiple sclerosis. Neuroimage. 2016;139:376–84. doi:10.1016/j.neuroimage.2016.06.053
56. Battaglini M, Jenkinson M, De Stefano N. Evaluating and reducing the impact of white matter lesions on brain volume measurements. Hum Brain Mapp. 2012;33(9):2062–71. doi:10.1002/hbm.21344
57. Dohmen M, Klemens MA, Baltruschat IM, Truong T, Lenga M. Similarity and quality metrics for MR image-to-image translation. Sci Rep. 2025;15(1):3853. doi:10.1038/s41598-025-87358-0
58. Bethlehem RAI, Seidlitz J, White SR, Vogel JW, Anderson KM, Adamson C, et al. Brain charts for the human lifespan. Nature. 2022;604(7906):525–33. doi:10.1038/s41586-022-04554-y
59. Jaccard P. The distribution of the flora in the alpine zone. New Phytol. 1912;11(2):37–50. doi:10.1111/j.1469-8137.1912.tb05611.x
60. Huttenlocher DP, Klanderman GA, Rucklidge WJ. Comparing images using the Hausdorff distance. IEEE Trans Pattern Anal Mach Intell. 1993;15(9):850–63. doi:10.1109/34.232073
61. Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Med Imaging. 2015;15:29. doi:10.1186/s12880-015-0068-x
62. Litjens G, Toth R, van de Ven W, Hoeks C, Kerkstra S, van Ginneken B, et al. Evaluation of prostate segmentation algorithms for MRI: the PROMISE12 challenge. Med Image Anal. 2014;18(2):359–73. doi:10.1016/j.media.2013.12.002
63. Menze BH, Jakab A, Bauer S, Kalpathy-Cramer J, Farahani K, Kirby J, et al. The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Trans Med Imaging. 2015;34(10):1993–2024. doi:10.1109/TMI.2014.2377694
64. Vaassen F, Hazelaar C, Vaniqui A, Gooding M, van der Heyden B, Canters R, et al. Evaluation of measures for assessing time-saving of automatic organ-at-risk segmentation in radiotherapy. Phys Imaging Radiat Oncol. 2020;13:1–6. doi:10.1016/j.phro.2019.12.001
65. Heimann T, van Ginneken B, Styner MA, Arzhaeva Y, Aurich V, Bauer C, et al. Comparison and evaluation of methods for liver segmentation from CT datasets. IEEE Trans Med Imaging. 2009;28(8):1251–65. doi:10.1109/TMI.2009.2013851
66. Kuijf HJ, Biesbroek JM, De Bresser J, Heinen R, Andermatt S, Bento M, et al. Standardized assessment of automatic segmentation of white matter hyperintensities and results of the WMH segmentation challenge. IEEE Trans Med Imaging. 2019;38(11):2556–68. doi:10.1109/TMI.2019.2905770
67. Carugo O, Pongor S. A normalized root-mean-square distance for comparing protein three-dimensional structures. Protein Sci. 2001;10(7):1470–3. doi:10.1110/ps.690101
68. Zemla A. LGA: a method for finding 3D similarities in protein structures. Nucleic Acids Res. 2003;31(13):3370–4. doi:10.1093/nar/gkg571
69. Zhang Y, Skolnick J. Scoring function for automated assessment of protein structure template quality. Proteins. 2004;57(4):702–10. doi:10.1002/prot.20264
70. Xu J, Zhang Y. How significant is a protein structure similarity with TM-score = 0.5? Bioinformatics. 2010;26(7):889–95. doi:10.1093/bioinformatics/btq066
71. Mariani V, Biasini M, Barbato A, Schwede T. lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. Bioinformatics. 2013;29(21):2722–8. doi:10.1093/bioinformatics/btt473
72. Andrec M, Snyder DA, Zhou Z, Young J, Montelione GT, Levy RM. A large data set comparison of protein structures determined by crystallography and NMR: statistical test for structural differences and the effect of crystal packing. Proteins. 2007;69(3):449–65. doi:10.1002/prot.21507
73. Chothia C, Lesk AM. The relation between the divergence of sequence and structure in proteins. EMBO J. 1986;5(4):823–6. doi:10.1002/j.1460-2075.1986.tb04288.x
74. Martin YC, Kofron JL, Traphagen LM. Do structurally similar molecules have similar biological activity? J Med Chem. 2002;45(19):4350–8. doi:10.1021/jm020155c
75. Baldi P, Nasr R. When is chemical similarity significant? The statistical distribution of chemical similarity scores and its extreme values. J Chem Inf Model. 2010;50(7):1205–22. doi:10.1021/ci100010v
76. Joint Committee for Guides in Metrology. International vocabulary of metrology – basic and general concepts and associated terms (VIM). 3rd ed. JCGM 200:2012. Sèvres: Bureau International des Poids et Mesures; 2012. doi:10.59161/JCGM200-2012
77. Raunig DL, McShane LM, Pennello G, Gatsonis C, Carson PL, Voyvodic JT, et al. Quantitative imaging biomarkers: a review of statistical methods for technical performance assessment. Stat Methods Med Res. 2015;24(1):27–67. doi:10.1177/0962280214537344
78. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986;1(8476):307–10. doi:10.1016/S0140-6736(86)90837-8
79. Analytical Methods Committee, AMCTB No. 74. z-Scores and other scores in chemical proficiency testing—their meanings, and some common misconceptions. Anal Methods. 2016;8(28):5553–5. doi:10.1039/c6ay90078j
80. Lee H, Jung G, Min J, Kim H, Jeong W, Kim T. Korean proficiency tests for pesticide residues in rice: comparison of various proficiency testing evaluation methods and identification of critical factors for multiresidue analysis. Foods. 2023;12(10):2085. doi:10.3390/foods12102085
81. Black K, Carson P, Jenkins E. A comparison of proficiency testing evaluation criteria approaches for fitness-for-purpose. Accred Qual Assur. 2024;29(5–6):375–9. doi:10.1007/s00769-024-01590-1
82. Horwitz W, Albert R. The Horwitz ratio (HorRat): a useful index of method performance with respect to precision. J AOAC Int. 2006;89(4):1095–109. doi:10.1093/jaoac/89.4.1095
83. Ehling S, Thompson JJ, Schimpf KJ, Pacquette LH, Haselberger PA. A contemporary look at the precision of modern analytical methods in food analysis and the relevance of the Horwitz equation. J AOAC Int. 2025;108(4):566–71. doi:10.1093/jaoacint/qsaf026
84. Daugman J. How iris recognition works. IEEE Trans Circuits Syst Video Technol. 2004;14(1):21–30. doi:10.1109/TCSVT.2003.818350
85. Hollingsworth K, Bowyer KW, Lagree S, Fenker SP, Flynn PJ. Genetically identical irises have texture similarity that is not detected by iris biometrics. Comput Vis Image Underst. 2011;115(11):1493–502. doi:10.1016/j.cviu.2011.06.010
86. Yoon S, Jain AK. Longitudinal study of fingerprint recognition. Proc Natl Acad Sci U S A. 2015;112(28):8555–60. doi:10.1073/pnas.1410272112
87. Manichaikul A, Mychaleckyj JC, Rich SS, Daly K, Sale M, Chen WM. Robust relationship inference in genome-wide association studies. Bioinformatics. 2010;26(22):2867–73. doi:10.1093/bioinformatics/btq559
88. Ramstetter MD, Dyer TD, Lehman DM, Curran JE, Duggirala R, Blangero J, et al. Benchmarking relatedness inference methods with genome-wide data from thousands of relatives. Genetics. 2017;207(1):75–82. doi:10.1534/genetics.117.1122
89. Klingenberg CP, Barluenga M, Meyer A. Shape analysis of symmetric structures: quantifying variation among individuals and asymmetry. Evolution. 2002;56(10):1909–20. doi:10.1111/j.0014-3820.2002.tb00117.x
90. Collyer ML, Adams DC. Interrogating random and systematic measurement error in morphometric data. Evol Biol. 2024;51(1):179–207. doi:10.1007/s11692-024-09627-6
91. Murphy AH. Skill scores based on the mean square error and their relationships to the correlation coefficient. Mon Weather Rev. 1988;116(12):2417–24. doi:10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2
92. Gandin LS, Murphy AH. Equitable skill scores for categorical forecasts. Mon Weather Rev. 1992;120(2):361–70. doi:10.1175/1520-0493(1992)120<0361:ESSFCF>2.0.CO;2
93. Hogan RJ, Ferro CAT, Jolliffe IT, Stephenson DB. Equitability revisited: why the "equitable threat score" is not equitable. Weather Forecast. 2010;25(2):710–26. doi:10.1175/2009WAF2222350.1
94. Roberts NM, Lean HW. Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events. Mon Weather Rev. 2008;136(1):78–97. doi:10.1175/2007MWR2123.1
95. Lorenz EN. The predictability of a flow which possesses many scales of motion. Tellus. 1969;21(3):289–307. doi:10.1111/j.2153-3490.1969.tb00444.x
96. Dalcher A, Kalnay E. Error growth and predictability in operational ECMWF forecasts. Tellus A. 1987;39(5):474–91. doi:10.3402/tellusa.v39i5.11774
97. Gilleland E, Skok G, Brown BG, Casati B, Dorninger M, Mittermaier MP, et al. A novel set of geometric verification test fields with application to distance measures. Mon Weather Rev. 2020;148(4):1653–73. doi:10.1175/MWR-D-19-0256.1
98. Canty MJ, Nielsen AA. Automatic radiometric normalization of multitemporal satellite imagery with the iteratively re-weighted MAD transformation. Remote Sens Environ. 2008;112(3):1025–36. doi:10.1016/j.rse.2007.07.013
99. Komiske PT, Metodiev EM, Thaler J. Metric space of collider events. Phys Rev Lett. 2019;123(4):041801. doi:10.1103/PhysRevLett.123.041801
100. Komiske PT, Mastandrea R, Metodiev EM, Naik P, Thaler J. Exploring the space of jets with CMS open data. Phys Rev D. 2020;101(3):034009. doi:10.1103/PhysRevD.101.034009
101. Kofler F. inpainting: official package to compute metrics for the BraTS inpainting challenge [Internet]. Version 0.0.12. GitHub; 2023 [updated 2026 Jun 10; cited 2026 Nov 1]. Available from: https://github.com/BraTS-inpainting/inpainting

### Reference notes

- [28], [33], [34]: dataset citations; author lists from DataCite ([28], [33]) and, for [34], from the dataset's own `dataset_description.json`, because the DataCite record for that DOI lists only G. de Zubicaray as creator.
- [35]: cited because the QTIM dataset asks users of its T1w data to cite it.
- [42]: proceedings volume on HAL (hal-03358968v3); no DOI.
- [44]: year, volume, issue and pages as in the PubMed record (published online in 2023, assigned to volume 15, issue 6).
- [59]: an English translation of Jaccard's 1907 French article in Revue générale des Sciences; content checked on the scanned journal pages.
- [66]: author order as in PubMed, the PMC author manuscript and arXiv; the Crossref deposit lists the authors in a different order.
- [67]: title spelling as in PubMed; the published title has the typo "spuare".
- [69]: an erratum corrects a typo in one formula (Proteins 2007;68(4):1020, doi:10.1002/prot.21643).
- [76]: a standards document; content checked on the BIPM publication.
- [96]: primary DOI as registered in Crossref; the earlier Wiley DOI 10.1111/j.1600-0870.1987.tb00322.x resolves to the same article.
