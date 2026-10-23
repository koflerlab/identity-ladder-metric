# Experiment plan: what can run on data already in hand

**Companion to:** `Research_Plan_Identity_Ladder_Metric.md` (rungs L0–L4, metric design, references).
**Status:** 18 October 2026. Inventory checked against `C:\Projects\data` (registry, catalog, file counts) and the MIDL 2026 repository `C:\Projects\02_academia\studies\brain-mri-segmentation`. Access conditions of each dataset are tracked separately and are not part of this plan.

---

## 1. Inventory

### 1.1 On disk now

| Dataset | Origin | Location | What is there | Ladder rungs |
|---|---|---|---|---|
| **SIMON** | Canada | `data/brain_images/SIMON_data` | 599 NIfTI, 73 sessions; T1w 99, T2w 76, FLAIR 78, PD 71 (+ DWI, BOLD). **15 sessions have ≥ 2 T1w** (8 × 2, 5 × 3, 2 × 5) | **L0** (within-session repeats), **L1** (35 institution–model combinations), **L2** (age 29.7–46.4 in the phenotype file) |
| **SIMON phenotype** | Canada | repo `data/SIMON_pheno.csv` | 73 sessions: acquisition date, institution (33), manufacturer (3), model (15), age | crossed scanner × time design: **18 institution–model combinations recur, covering 56 sessions** |
| **SIMON metric tables** | Canada | repo `data/simon_freesurfer.csv`, `simon_fastserfer.csv`, `consecutive_fs8_SIMON.csv` | consecutive-session pairs × 34 labels (FreeSurfer, 72 sessions; FastSurfer, 69) and × 18 labels (FreeSurfer 8): volumes, Dice, surface Dice, HD95 | L1 + L2 mixed, separable with the phenotype file |
| **IXI** | UK | `data/datasets/neuroimaging/ixi/version-unverified/raw` | T1 581, T2 578, PD 578; three centres (Guys 322, HH 187, IOP 75); age in `IXI.xls` | **L4** (unrelated adults), cross-sectional ageing |
| **QTIM, partial** (registry id `siblings-mri`) | Australia | `data/brain_images/siblings` | Queensland Twin IMaging, OpenNeuro ds004169 v1.0.8. Local copy: 23 adolescents (12–16), 68 T1w files, two complete families. Full dataset: 1202 participants in 682 families, test–retest 78 (~3.5 months), longitudinal 62 (~3.5 years) | after full download: **L0 multi-subject**, **L2**, **L3 related**, L3a/L3b with zygosity |
| **SRPBS traveling subjects** | Japan | `data/brain_images/SRPBS_TS_traveling_subjects.tar.gz` (20.9 GB, not extracted) + derived repo tables (FreeSurfer 8: 405 pairs × 34 labels; FastSurfer: 81 pairs × 18 labels) | nine healthy traveling subjects, 12 scanners | **L1 multi-subject**. I have not opened it; you run its steps until its terms are reviewed |

### 1.2 Internal notes (not for the shared version)

| Item | Note |
|---|---|
| Earlier label maps and registrations (July 2026 audit: FreeSurfer 8 SIMON 69, FastSurfer SIMON 87, SRPBS FreeSurfer 8 90, SRPBS FastSurfer 143) | on `D:\data`, which is not connected today; outside the canonical data root and not in the registry |
| BraTS inpainting 2026 review PDFs (`02_academia/reviews/brats_impainting_2026`) | confidential peer-review material; must not inform this plan; not opened |
| Personal pilot (`experiments/kate_n1_2026`) | personal data; not used |

### 1.3 To download

| Dataset | Origin | Rung / use | Published size |
|---|---|---|---|
| **ON-Harmony** (OpenNeuro ds004712) | UK | L0 + L1 multi-subject: 20 participants × 6 scanners, 9 with 6 within-scanner repeats | 62.6 GB |
| **FTHP** | Germany | L0 + L1, one person, 116 scanners; exact replication of Hays et al. | to check |
| **MASiVar** | USA | L1 multi-subject; Hays et al.'s third validation set (5 adults on 3–4 scanners at three institutions) | to check |
| **Chinese travelling subjects** (Xiao et al. 2026, CHIMGEN consortium) | China | L1: two people on 28 scanners | availability to check |
| **CoRR** | international, China and USA | L0 across 18 sites: 1,629 individuals, 5,093 resting-state scans | to check |
| **SLIM** (southwest China) | China | L0 + L2: young adults rescanned within 3.5 years | to check |
| **TwinsMX** (Mexico) | Mexico | L3 candidate: national twin registry with brain MRI | availability to check |
| **Maclaren test–retest** (ds000239) | USA | L0: 3 participants × 40 scans | 0.9 GB |
| **Midnight Scan Club** (ds000224) | USA | L0: 10 participants × 4 T1w + 4 T2w | 27 GB (full) |
| **MR-ART** (ds004173) | Hungary | motion robustness; neuroradiologist artefact scores | 4.5 GB |
| **QTIM, full** (ds004169) | Australia | L0 multi-subject, L2, L3 | 147.7 GB in full; fetch T1w only (about 101 GB free on C:) |
| **QTAB** (ds004146); HCP Young Adult | Australia; USA (HCP) | L3 fallback | QTAB 683 GB in full |
| **LUMIERE** (figshare) | Switzerland | lesions: 91 glioblastoma patients, 638 study dates | 32.6 GB (MRI) |
| **MSSEG-2** | France | lesions: 100 MS patients × 2 time points | to check |
| **BraTS inpainting** | international | WP3 inpainting metric | to check |
| Hawco et al. | Canada, USA | L1 + L2 | to check |
| **Chongqing Adolescent Twin Study** | China | L3: 136 adolescent twins, structural, resting-state and diffusion MRI | to check |
| **Japanese travelling subjects** (Saito et al.) | Japan | L1: 69 travelling subjects on 13 MRI models (diffusion) | to check |
| **Osaka twin study** | Japan | L3: 20 MZ and 17 DZ older adult pairs | availability to check |
| **Richter et al. travelling subjects** | UK, Finland | L1: 73 volunteers, 161 scans, 4 machines | availability to check |
| **Hungarian Twin Registry** | Hungary | L3: 118 adult twins with T1w | availability to check |
| **BLSA** | USA | L2: longitudinal MRI of older adults; in HACA3⁺ training data, so not for E2 | request |
| **VETSA** | USA | L3: 67 MZ and 51 DZ older male pairs | availability to check |

### 1.4 Tools (July 2026 audit, not re-run today)

WSL Python 3.12; BrainChop 0.2.5 and TIGERBx 0.2.3 environments (TIGERBx timed out on the SIMON smoke test); **Docker daemon inactive**.

**"Almost everything" is right for L0, L1, L2 and L4 and for the harmonization reproduction. The downloads above add multi-subject scanner noise (ON-Harmony), dense repeats (Maclaren, Midnight Scan Club), motion robustness (MR-ART), the twin rung (QTIM) and lesion stability (LUMIERE, MSSEG-2).**

---

## 2. Experiments, in order

Pair construction, statistics, harmonization arms and controls follow research plan Section 7. Every experiment that loops over scans starts with a one-scan smoke test that asserts the output exists, prints seconds per scan and the projected total, and warns if no GPU is present. Every external tool gets a compatibility test before it is wrapped.

### E0 — Separate scanner from time in the tables you already have *(no compute, this week)*

- **Input:** the three SIMON metric tables + `SIMON_pheno.csv`.
- **Do:** join each consecutive pair to both sessions' institution, model and date. Label every pair *same scanner* (same institution and model) or *different scanner*, with its time gap in days. Fit, per structure, |Δvolume %| (and 1 − Dice) ~ scanner change + log(time gap), with the pair as the unit.
- **QC first:** some rows have Dice = 0, which indicates failed registration or segmentation. Apply the surface-based quality filter from the MIDL paper before fitting, and report how many pairs were removed.
- **Output:** per-structure scanner effect vs time effect — the first quantitative L1 vs L2 split.
- **Limits:** consecutive pairs only, so long time gaps are under-represented; one subject.

### E1 — L0: within-session and dense repeats

- **E1a, SIMON:** segment every T1w in the 15 sessions with repeats (26 extra acquisitions) with FastSurfer and SynthSeg; compare within-session pairs (L0) with between-session pairs on the same and on different scanners.
- **E1b, Maclaren and Midnight Scan Club:** the same pipelines on 3 participants × 40 scans and 10 participants × 4 T1w, for a dense L0 estimate across more than one person.
- **Output:** σ_L0 per structure and pipeline; the ratio σ_L1 / σ_L0. Tests H1 of the research plan.

### E2 — Reproduce the harmonization claim of Hays et al. on SIMON

- **Why SIMON:** it is a single-subject, multi-scanner set like FTHP, it has the four contrasts HACA3⁺ uses (T1w, T2w, FLAIR, PD), and **it is not in HACA3⁺'s training data** (training used OASIS-3, BLSA, HCP, IXI, TREAT-MS and other private sites). It is an out-of-training test of their claim.
- **Arms** (as in the repo's `docs/HARMONIZATION_EXPERIMENT.md`): native, registration-only, HACA3, HACA3⁺ (public weights, github.com/shays15/haca3-plus); the registration-only arm is mandatory.
- **Segmenters:** FastSurfer, SynthSeg; SLANT if it runs, to match Hays et al.
- **Metric:** coefficient of variation of regional volume and Dice across sessions, per arm, with sessions harmonized to one reference session.
- **Output:** does harmonization reduce L1 without moving L0 or distorting small structures? Then E2b: the same on FTHP and MASiVar (exact replication of two of Hays et al.'s three validation sets) and on ON-Harmony and SRPBS (many people, not among HACA3⁺'s named training datasets).
- **Do not use IXI here:** it is in HACA3⁺'s training set.

### E3 — Shape space on SIMON: does identity hold across scanners and 17 years?

- **Input:** FreeSurfer or FastSurfer surface outputs for all 73 sessions: reuse those from D if they include surfaces, otherwise recompute.
- **Do:** BrainPrint descriptors with the `brainprint` package (Deep-MI, 0.4.0 on PyPI). **Caveat:** per its PyPI page, tetrahedral mesh support is no longer maintained, and in the BrainPrint paper the tetrahedral meshes separated MZ from DZ twins best. Plan for surface-only descriptors, or restore the volumetric part separately.
- **Also:** sulcal depth / local gyrification and a persistence-diagram feature on cortical thickness — candidate feature spaces 2–3 of the research plan.
- **Output:** within-SIMON shape distances for L0, L1 and L2 (using E0's labels), and the crossed scanner × time model in shape space. Tests H2 in part.

### E4 — L4 reference from IXI, and motion robustness from MR-ART

- **E4a, IXI:** a subsample stratified by centre and age (start with 100); the same segmentation and feature spaces as E3; between-subject and SIMON-to-IXI distances. **Output:** the L4 distribution and a first separation ratio per feature, (L4 distance) / (L1 noise); features that fail here are dropped early.
- **E4b, MR-ART:** the same features on still vs slight vs excessive motion scans of the same 148 people. **Output:** how far motion moves each feature relative to the ladder, and a check of the metric against the neuroradiologist artefact scores.

### E5 — Ladder statistics with many subjects

- **Why:** discriminability needs repeated measurements for many subjects; SIMON is one subject and IXI has one scan each.
- **ON-Harmony (20 participants, 6 scanners each):** discriminability and within- vs between-subject distances with scanner variation; the 9 participants with within-scanner repeats give L0 vs L1 inside the same people.
- **QTIM test–retest (78 participants, ~3.5 months):** the same, for repeat noise within one study. Restrict to one T1w acquisition orientation, or model it: the orientation changed mid-study and the dataset notes an orientation-dependent pulsation artefact.
- **SRPBS (nine subjects, 12 scanners):** the same analysis as ON-Harmony, as a second multi-scanner cohort; yours to run for now. MASiVar and the Chinese travelling subjects (two people, 28 scanners) are further multi-scanner cohorts if their data can be obtained.
- **CoRR and SLIM:** multi-site and long-interval repeat noise in Chinese and international cohorts, if their structural scans suit the pipelines.
- **Output:** within-person noise versus between-person distance on many people, with and without scanner change; the main multi-subject result for the MIDL paper.

### E6 — Regional calibration, before any inpainting *(prepares WP3)*

- **Do:** place masks at fixed anatomical locations in SIMON sessions, ON-Harmony and IXI subjects and compute the regional z for L0, L1, L2 and L4 at the same location. Use synthetic masks until the BraTS healthy-mask definition is in hand; swap them in once the data are downloaded.
- **Output:** per-location rung distributions and the τ_min sensitivity analysis — the calibration the inpainting metric will use.

### E6b — Donor-swap calibration of the inpainting metric *(research plan Section 7.4)*

- **Do:** on healthy scans, fill a mask with tissue from the same person on another scanner, from a twin and from an unrelated person (after registration), and also with blur, intensity shift, misalignment and stripe artefacts at graded strengths.
- **Test:** the proposed metric must order the fills re-scan < twin < unrelated and rise monotonically with artefact strength; SSIM, PSNR and RMSE are tested for the same orderings.
- **Inputs:** SIMON and ON-Harmony for re-scan donors, QTIM families for twin donors, IXI for unrelated donors.

### E7 — Twin rung

- **E7a, related vs unrelated:** pairs within a QTIM family versus pairs across families, in every feature space. This mixes MZ twins, DZ twins and non-twin siblings into one "related" rung. Tests whether genetic relatedness is visible at all (L1 < related < unrelated), and tests H3 in weakened form: if global volumes already fail to separate related from L1, they will fail for MZ twins.
- **E7b, MZ vs DZ:** with QTIM zygosity (QTAB or HCP as fallback, TwinsMX as a Latin American candidate); tests H4 (L1 < MZ < DZ < L4). Go/no-go of research plan WP2.
- **In both:** compare pairs within one acquisition orientation.

### E8 — Lesion change-detection stability *(research plan WP1b, concept-note Track A)*

- **LUMIERE (91 glioblastoma patients, 638 study dates):** lesion volume and count trajectories from two or more automated segmenters and a preprocessing grid; agreement of the trend across pipelines, against the published finding that two tools matched expert volume trends in about 80% of cases and failed at lesion counts across time (research plan [41]).
- **MSSEG-2 (100 MS patients × 2 time points):** stability of new-lesion detection across pipelines and preprocessing.
- **Output:** the detectability floor for lesion-based endpoints, reported as a lower bound.

---

## 3. What this gives for MIDL 2027

| Paper part | Experiments | Ready? |
|---|---|---|
| Stability floor: L0 vs L1, per pipeline | E0, E1a | **yes**, data on disk |
| Dense L0 across people | E1b | after small downloads (Maclaren, Midnight Scan Club) |
| Harmonization: out-of-training reproduction of Hays et al. | E2 (SIMON), E2b (FTHP, ON-Harmony) | SIMON **yes**; others after download |
| Identity ladder L0–L2–L4 in shape space | E3, E4a | **yes**, if surfaces exist or compute allows |
| Motion robustness | E4b | after MR-ART download (4.5 GB) |
| Multi-subject ladder | E5 | after ON-Harmony and QTIM downloads; SRPBS on disk |
| Related rung | E7a | after QTIM download |
| MZ vs DZ rung | E7b | after zygosity access |
| Lesion stability | E8 | after LUMIERE and MSSEG-2 |

## 4. Actions, in priority order

1. **Reconnect drive D** and check whether the FreeSurfer outputs include surfaces (`surf/`). This decides whether E3 is a re-analysis or a recomputation.
2. **Small downloads first:** MR-ART (4.5 GB), Maclaren (0.9 GB), then ON-Harmony (62.6 GB) and QTIM T1w only; register each in the data registry.
3. **Then:** FTHP, LUMIERE, Midnight Scan Club, MSSEG-2 and the BraTS inpainting data.
4. **Access conditions** (zygosity, SRPBS, MASiVar, MSSEG-2, BraTS, Hawco et al., the Chinese travelling subjects, CoRR, SLIM, TwinsMX): settled later and tracked separately; zygosity has the longest lead time.
5. **Start the Docker daemon** or set up native FastSurfer/HACA3⁺ environments, or wait for cluster access.
6. **Registry housekeeping** (not urgent): relabel `siblings-mri` as QTIM; register the derived outputs on D. Registry changes go through the storage-maintenance lock.
