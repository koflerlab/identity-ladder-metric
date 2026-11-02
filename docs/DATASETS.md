# Datasets

Two sources, cross-checked on 2026-11-01:

- the **Identity Ladder Dataset Tree**, the project's catalogue of what kinds of identity data exist (291 rows, each
  found by one research agent and checked against the official page or data paper by a second; private claude.ai
  artifact, version 1793522584-f862). Its summary, decision points, first batch and full catalogue are reproduced
  from the "Dataset Tree" section on;
- the shared data registry (`_registry/datasets.json` of the data root), which records what is actually local, with
  licence and access as recorded there.

The first sections cover the datasets this project uses or plans to use, with facts checked against each dataset's
own metadata where stated. No local paths and no participant IDs. "Licence as recorded" quotes the registry; it is
not legal advice. GRAPE, UPOL, UTIRIS, IXI and SIMON are local but not in the tree; QTIM is in the tree only as an
unverified lead (see "Corrections to the tree").

Project rule for all real data: real images, per-case tables and real-case figures stay in the local data root and
never enter this public repository, whatever the licence allows. The repository holds code, aggregate results and
synthetic figures.

## Brain MRI

| Dataset | Source | Licence as recorded | Local status | Rungs it serves | Redistribution |
|---|---|---|---|---|---|
| QTIM (Queensland Twin IMaging), registry id `siblings-mri` | OpenNeuro ds004169 v1.0.8, doi:10.18112/openneuro.ds004169.v1.0.8 | CC0 in the dataset's `dataset_description.json`; registry entry updated 2026-11-01 (licence, version, DOI, acquisitions with SHA-256 manifests) | partial: 244 of 1,202 participants local (278 T1w scans) after the 2026-11-01 download of the 244 frozen real-case scans and their JSON sidecars; metadata complete | repeat scan (test-retest subsample, N = 78, interval about 3.5 months); later visit (longitudinal subsample, N = 62, interval about 3.5 years, adolescents); other hemisphere; family member; stranger; population mean | CC0 permits it; project rule above applies |
| SIMON | not recorded in the registry ("original URLs/licence must be retained") | "terms to verify" | raw present: one person, 73 sessions, 99 T1 acquisitions | same session, another day, other scanner or protocol, years later (one person only) | terms to verify: do not re-host |
| IXI | https://brain-development.org/ixi-dataset/ | CC BY-SA 3.0, verified at the original site 2026-10-18 | raw present | strangers, other centres | allowed under CC BY-SA 3.0; project rule above applies |
| SRPBS traveling subjects | not recorded | "agreement governed per user; no agent analysis before review" | archive present, not opened | other scanner, many people (planned) | agreement governs; no agent access until Kate reviews the terms |
| BraTS inpainting 2023 and 2025 | Synapse (challenge data and winner models) | not checked: needs Kate's own registration and acceptance of the terms | not local, not in the registry | the challenge fills to be scored on the ladder (planned) | challenge terms govern |
| MNI ICBM152 template | bundled with nilearn (symmetric ICBM152 2009a; used as registration target) | not recorded in the registry | installed with nilearn | registration target, not a rung | not redistributed by this project |

QTIM facts checked against `participants.tsv`, `participants.json` and the dataset README (aggregates only):

- 1,202 participants in 682 families: 445 families of two, 36 of three, 1 of four, 200 single participants.
- There is no zygosity column. The README states that zygosity exists but requires a data transfer agreement and
  institutional ethics approval through QIMR Berghofer. With the public release, a relative is a "family member"
  (co-twin of unknown zygosity, or non-twin sibling), not MZ or DZ.
- Each session holds one T1w scan. `ses01_T1w_acq` and `ses02_T1w_acq` record the acquisition orientation (coronal or
  sagittal), changed by a scanner software upgrade; the README warns that a vascular pulsation artefact varies with
  orientation. There is no same-session repeat, so the L0 rung is not available in QTIM.
- MRIQC image quality metrics and FreeSurfer 5.3 measures are released as derivatives (not yet downloaded).
- Local T1w before 2026-11-01: 22 first-session scans and 12 second-session scans; 12.4 MB per scan on average (34 local scans); the release has 1,342 T1w scans (1,202 first sessions, 140 second), about 17 GB at that average.
- Downloaded 2026-11-01 (`src/real_cases_fetch.py`): the 244 T1w scans of the frozen real-case selection, 2.69 GB (11.0 MB per scan), from version-pinned snapshot 1.0.8 URLs, each verified by size and S3 ETag MD5, plus their 244 JSON sidecars for the descriptive data-QC report. SHA-256 manifests are private, next to the selection; the scan manifest hash is in `docs/PROTOCOL_real_cases.md`, Amendment 2. The expanded A1.1 set (826 scans) follows.

## Eye and iris

| Dataset | Source | Licence as recorded | Local status | Rungs it serves | Redistribution |
|---|---|---|---|---|---|
| GRAPE fundus | figshare collection 6406319 v1, doi:10.6084/m9.figshare.c.6406319.v1 | CC0 1.0 (figshare API, checked 2026-10-23) | partial | same eye another visit, same or other camera; fellow eye same visit | allowed |
| UPOL iris | https://phoenix.inf.upol.cz/iris/ (version unverified) | no licence stated; citation requested | raw present | same eye same sitting; fellow eye | not granted: do not re-host |
| UTIRIS | UTIRIS V.1, maintainer's page at utiris.wordpress.com | no licence; copyright CIPCE, University of Tehran; cite Hosseini et al. 2010 | raw present | same eye same sitting; visible versus near-infrared in one sitting; fellow eye | not granted: do not re-host |

## Other 2D biometric sets (registered for this project, not used in any result)

Downloaded on 2026-10-31 and 2026-11-01 as candidates for 2D ladders. Which rungs each can serve has not been audited.
The registry rule for all of them: never print file names or show images in prompts or outputs.

| Dataset | Source | Licence as recorded |
|---|---|---|
| AMI ear | ctim.ulpgc.es AMI ear database (version unverified) | CC BY-NC-ND 3.0 |
| RIDB | Mendeley Data tjw3zwntv6 v1, doi:10.17632/tjw3zwntv6.1 | CC BY 4.0 |
| FEI face | fei.edu.br face database (version unverified) | research only; no distribution or reproduction |
| MUCT | GitHub StephenMilborrow/muct, commit 31ff0ac | no licence file; images may not be reproduced publicly except eight listed subjects |
| PAPILA | figshare article 14798004 v2, doi:10.6084/m9.figshare.14798004 | GPL 3.0+ (figshare API) |
| Dorsal hand veins | GitHub wilchesf/dorsalhandveins, commit a4b32e7 | no licence file |
| 11k Hands | sites.google.com/view/11khands (version unverified) | "free for reasonable academic fair use"; per-image metadata: aggregates only |
| Tongji contactless palmprint | cslinzhang.github.io/ContactlessPalm (version unverified) | no licence stated |

## Project derivatives

| Registry id | Content | Access |
|---|---|---|
| `identity-ladder-visual-detail-20261101` | prepared QTIM and UTIRIS cases for the detailed visual comparisons: image and participant mappings, case-level scores, figures | local only; source restrictions retained |

## Not available

- Zygosity for QTIM: needs a data transfer agreement and ethics approval (see above).
- Any repeat within one QTIM session (L0): not acquired by the study.

## Corrections to the tree

- QTIM (OpenNeuro ds004169) is filed under "head: ear" as an unverified lead. It is brain MRI, verified here against
  its own metadata (see "Brain MRI" above), and partly local. Moved to "tomography: brain".
- CASIA-Twins is filed under "tomography: brain"; its own evidence describes face, iris and fingerprint. Moved to
  "head: face 2D / video".
- CFP (Celebrities in Frontal-Profile) is filed under "eye: fundus"; it is a face set. Moved to "head: face 2D / video".
- The QTIM longitudinal interval: the tree says four years, `participants.json` says about 3.5 years.
- Other rows are reproduced as filed; they were not re-audited row by row here.

## The Dataset Tree: what kinds of data exist

Counts per kind of data and status, from the tree (291 rows). Status: "download now" is open without an account; "your action" needs an account or terms that Kate accepts herself; "decision" must be requested (data transfer agreement, institutional signature); "lead" is an unverified pointer; "cut" has no access route, cannot be verified or is commercial.

| Kind of data | download now | your action | decision | lead | cut | total |
|---|---|---|---|---|---|---|
| 3D surface: dental / intraoral | 4 | 1 | 4 | 0 | 1 | 10 |
| 3D surface: face / head / body | 0 | 0 | 4 | 0 | 2 | 6 |
| body: gait (vision) | 0 | 0 | 0 | 1 | 1 | 2 |
| eye: OCT / OCTA | 0 | 0 | 7 | 1 | 4 | 12 |
| eye: fundus | 4 | 0 | 5 | 3 | 6 | 18 |
| eye: iris | 0 | 0 | 10 | 4 | 5 | 19 |
| eye: periocular / sclera | 0 | 0 | 2 | 0 | 1 | 3 |
| hand: fingerprint | 0 | 1 | 2 | 2 | 5 | 10 |
| hand: palm / vein / knuckle / geometry | 5 | 2 | 6 | 6 | 10 | 29 |
| head: ear | 2 | 0 | 1 | 2 | 1 | 6 |
| head: face 2D / video | 7 | 2 | 11 | 5 | 15 | 40 |
| other / unclassified | 0 | 0 | 0 | 0 | 1 | 1 |
| radiology 2D: chest | 1 | 2 | 4 | 1 | 0 | 8 |
| radiology 2D: dental | 2 | 0 | 3 | 1 | 1 | 7 |
| radiology 2D: mammography | 0 | 0 | 4 | 1 | 0 | 5 |
| radiology 2D: skeletal / DXA | 0 | 0 | 1 | 3 | 0 | 4 |
| radiology 2D: skull / cephalogram / sinus | 1 | 0 | 4 | 0 | 0 | 5 |
| skin: moles / dermoscopy / photos | 5 | 0 | 1 | 0 | 0 | 6 |
| small traits: lip / nail / foot / capillary / tongue | 0 | 2 | 2 | 0 | 1 | 5 |
| tomography: brain | 1 | 1 | 12 | 8 | 3 | 25 |
| tomography: other CT / MRI | 4 | 0 | 2 | 0 | 0 | 6 |
| nonvisual | 17 | 3 | 14 | 14 | 16 | 64 |

## Decision points: twins only by request

No open 2D dataset in the tree has twins. These 30 requestable sets add the twin axis (2D first, then by number of axes). Axes: B bilateral, R test-retest, T twins, A ageing, S many unrelated people.

| Dataset | Kind | Axes | Twins |
|---|---|---|---|
| [ND-TWINS-2009-2010](https://cvrl.nd.edu/projects/data/) | head: face 2D / video | B R T A S | MZ only, self-identified: Paone 2014 says all participants self-identified as identical ... |
| [WVU / FBI Twins Day multimodal biometric collection (Twinsburg, Ohio, 2010-2019)](https://biic.wvu.edu/data-sets/twins-day-dataset-2010-1015) | eye: iris | B R T S | self-report: 'identical', 'identical-mirror' and 'identical-ambiguous' when both twins ... |
| [Twins Eye Study in Tasmania (TEST) + Brisbane Adolescent Twin Study (BATS) retinal photographs](https://doi.org/10.1161/hypertensionaha.109.132902) | eye: fundus | B T S | 374 MZ + 536 DZ pairs plus 322 siblings; zygosity method not in abstract |
| [Brisbane Twin Naevus Study (full-body imaging + dermoscopy)](https://doi.org/10.1111/bjd.14291) | skin: moles / dermoscopy / photos | T S | MZ/DZ labelled (method not stated in the abstract): 76 MZ and 144 DZ individuals |
| [CAS twin fingerprint dataset (Tao et al. 2012)](https://doi.org/10.1371/journal.pone.0035704) | hand: fingerprint | R T | 83 identical pairs. For the overlapping Sun et al. 2010 subset, twin status was assigned ... |
| [Copenhagen MZ twin frontal sinus profile radiographs (Kjaer et al.)](https://doi.org/10.3109/00016489.2012.677064) | radiology 2D: skull / cephalogram / sinus | T S | MZ only; zygosity method not stated in abstract; 42 pairs |
| [Danish Twin Registry: LSADT facial photographs, perceived-age study (Christensen et al. 2009)](https://www.sdu.dk/en/forskning/dtr/researcher/guidelines) | head: face 2D / video | T S | 1,826 twins aged 70+ (MZ and DZ); counts and zygosity method not in the abstract |
| [Danish young-adult twin fundus cohort (retinal vascular fractal study)](https://doi.org/10.1167/iovs.17-22072) | eye: fundus | T S | MZ/DZ labelled (method not stated in abstract): 50 MZ + 49 DZ same-sex pairs, aged 20-46 |
| [Guangzhou Twin Eye Study (GTES)](https://www.cambridge.org/core/product/identifier/S183242741900118X/type/journal_article) | eye: OCT / OCTA | T S | MZ/DZ genotyped (16 STR markers, PowerPlex 16) for same-sex pairs; opposite-sex = DZ; 657 ... |
| [UK Biobank: genetically inferred relatives (fundus/OCT, MRI)](https://biobank.ctsu.ox.ac.uk/ukb/ukb/docs/ukb_genetic_data_description.txt) | eye: OCT / OCTA | T S | MZ twins not stated in the official document; relatedness pairs up to third degree via ... |
| [Wonder Ears identical-twin ear images (Nejati et al. 2012)](https://dblp.org/rec/conf/icpr/NejatiZSM012.html) | head: ear | B T | 39 identical pairs; zygosity method not stated |
| [Fine-Grained Face Verification dataset of identical twins (Hu, Lu, Tan 2015)](https://doi.org/10.1109/icb.2015.7139079) | head: face 2D / video | T | 455 MZ pairs scraped from the web; zygosity unverified |
| [Iranian web-scraped twins face database (Mozaffari & Behravan 2011)](unverified) | head: face 2D / video | T | 1,902 twin pairs; zygosity unknown (scraped) |
| [ND Twins Face Recognition Test Set (Notre Dame CVRL)](https://cvrl.nd.edu/projects/data/) | head: face 2D / video | T | identical twins form the different-person pairs; labelling not stated (the parent ... |
| [Hungarian Twin Registry palatal intraoral scan cohort (Semmelweis University; Simon, Vag et al.)](https://doi.org/10.1186/s12903-020-01261-w) | 3D surface: dental / intraoral | B R T A S | MZ/DZ by standardized questionnaire (stated nearly 99% accurate); 64 MZ, 33 DZ same-sex ... |
| [3D-TEC (3D Twins Expression Challenge)](https://cvrl.nd.edu/projects/data/) | 3D surface: face / head / body | B R T S | MZ, self-reported (survey Table 1): 107 pairs, 214 individuals |
| [ABCD Study twin hub](https://abcdstudy.org) | tomography: brain | B T A S | MZ/DZ genotyped: 772 pairs analysed (Maes 2022); about 800 same-sex pairs planned at four ... |
| [Adelaide twin dental collections (Craniofacial Biology Research Group, University of Adelaide)](https://doi.org/10.1017/thg.2012.78) | 3D surface: dental / intraoral | B T A S | MZ/DZ (abstract); DNA zygosity and 125 MZ / 89 DZSS / 49 DZOS pair counts come from a ... |
| [EMIF-AD PreclinAD (Netherlands Twin Register MZ twins aged 60+)](https://doi.org/10.1186/s13195-018-0406-7) | nonvisual | B T A S | MZ pairs only, genotyped: Amsterdam buccal cells taken for zygosity, Affymetrix Axiom/6 ... |
| [HCP Young Adult: MEG subset and 3T retest (correction/addition)](https://humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) | tomography: brain | B R T S | MZ/DZ genotyped for part of the sample; 36 self-reported DZ pairs were genetically MZ |
| [Older Australian Twins Study (OATS)](https://atlaslongitudinaldatasets.ac.uk/datasets/oats) | tomography: brain | B T A S | present: 161 MZ and 124 DZ pairs plus 1 MZ triplet set; zygosity method not stated on the ... |
| [TwinsUK Twins MR Imaging Study (NCT06014567)](https://clinicaltrials.gov/study/NCT06014567) | tomography: brain | B T A S | MZ/DZ (TwinsUK); counts not yet stated |
| [TwinsUK adult twin registry](https://twinsuk.ac.uk/) | tomography: brain | T A S | present, MZ and DZ; the public pages do not state how zygosity is assigned (one ... |
| [TwinsUK eye imaging (SD-OCT, colour fundus)](https://twinsuk.ac.uk/researchers/access-data-and-samples/request-access/) | eye: OCT / OCTA | B T S | MZ/DZ labelled (method not stated in abstracts read): 1,150 pairs with OCT (Jarrar 2024) ... |
| [TwinsUK lumbar spine MRI (baseline and 8-12-year follow-up)](https://twinsuk.ac.uk/researchers/access-data-and-samples/request-access/) | tomography: brain | T A S | MZ/DZ present in TwinsUK; zygosity method and pair counts for this subset not checked |
| [ABCD Study twin hub subsample](https://www.nbdc-datahub.org/data-access-process) | tomography: brain | T S | MZ/DZ genotyped: 772 pairs aged 9-10 analysed with genotype zygosity at 4 twin hub sites ... |
| [Beijing Twin Study (BeTwiSt) MRI subsample / Beijing Twins Brain-Behavior Association Project](https://atlaslongitudinaldatasets.ac.uk/datasets/betwist) | tomography: brain | T A | MZ and DZ (Atlas); 54 MZ + 54 DZ MRI pairs and genotype-based zygosity are unverified ... |
| [BrainSCALE (Netherlands Twin Register longitudinal twin MRI)](https://ntr-data-request.psy.vu.nl) | tomography: brain | T A | MZ and DZ twins plus older siblings; zygosity method not verified |
| [TwinsUK 3D facial images (3dMDface)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5008732/) | 3D surface: face / head / body | T S | 1,380 female twins analysed: 263 MZ pairs, 341 DZ pairs, 172 unpaired (75 MZ, 97 DZ) ... |
| [Lithuanian young-adult twins 3D facial scans (Vuollo et al. 2015)](https://doi.org/10.1017/thg.2015.16) | 3D surface: face / head / body | T | 106 pairs, MZ/DZ by DNA testing |

## Proposed first download batch and what happened

The tree's batch: 2D, low domain shift, open without an account, sizes checked 2026-11-01. Local status from the registry.

| Dataset | Kind | What it gives the ladder | Size | Local |
|---|---|---|---|---|
| AMI Ear Database | ear | left vs right ear of 100 people, one camera; same-sitting poses | 36 MB | registered (`ami-ear`) |
| Dorsal Hand Veins DB1/DB2 | hand veins (NIR) | both hands, 138 people, re-test after two months | about 1.0 GB | registered (`dorsal-hand-veins`) |
| Tongji Contactless Palmprint (ROI images) | palmprint | both palms of 300 people, two sessions about 61 days apart | 94 MB (+4.6 GB originals, RAR) | registered (`tongji-contactless-palmprint`) |
| 11K Hands | hands, dorsal and palmar | both hands of 190 people aged 18 to 75 | 632 MB | registered (`11k-hands`) |
| PAPILA | fundus | both eyes of 244 patients, one Topcon camera | 591 MB | registered (`papila`) |
| RIDB | fundus | both eyes of 20 people, five images each (session timing unknown) | 100 images | registered (`ridb`) |
| FEI Face Database | face 2D | 200 people, 14 poses each, one studio; left and right rotations | 354 MB | registered (`fei-face`) |
| MUCT | face 2D | about 276 people, five fixed cameras in one sitting | 276 MB | registered (`muct`) |
| Notre Dame FACE Features Set | face features | feature patterns including twin photographs (features only) | 0.2 MB | download failed; not registered |

## Masked and occluded faces (not yet in the tree)

Added 2026-11-02 for the occlusion lessons in [FACE_ID_LESSONS.md](FACE_ID_LESSONS.md) and theme 8 of the backlog. These rows are not in the Dataset Tree or the registry (changing those needs the maintenance lock in `data/AGENTS.md`), so the counts below do not include them. Nothing was downloaded.

| Dataset | Dim | Axes | Status | Notes |
|---|---|---|---|---|
| [RMFRD (Real-world Masked Face Recognition Dataset), Wang et al. 2023](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) | 2D image | S | download now (Google Drive or Baidu link in the repo README) | README: 525 people, 5 thousand masked and 90 thousand unmasked faces, crawled from the web and labelled. GitHub API reports no licence (checked 2026-11-02): do not re-host. Paper: doi:10.1109/TBIOM.2023.3242085 (R16 in FACE_ID_LESSONS.md). |
| [SMFRD (Simulated Masked Face Recognition Dataset), same repo](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) | 2D image | S | download now (same README) | README: masks drawn onto public face sets, 10 thousand people and 500 thousand faces. Same licence situation. Synthetic masks are the closest 2D analogue of an inpainting mask with known ground truth. |
| [MFR challenge InsightFace track test sets](https://doi.org/10.1109/ICCVW54120.2021.00165) | 2D image | S | cut: test images not released | Abstract: 7K-identity masked set plus children and multi-racial sets, evaluated only through an online system (T8-12). |

## Catalogue

Every row of the tree, grouped by kind of data. Twins and notes are shortened; the full verification notes stay in the tree. "Local" names the registry entry when the data are in the data root.

### 3D surface: dental / intraoral (10)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [Granada/Porto 3D intraoral scan and 2D photograph superimposition dataset](https://doi.org/10.5281/zenodo.21769403) | other | high | S | none | download now |  |
| [MMDental (Hangzhou Dental Hospital)](https://doi.org/10.6084/m9.figshare.28505276) | 3D volume | medium | B S | none | download now |  |
| [Teeth3DS+ (3DTeethSeg'22 / 3DTeethLand)](https://crns-smartvision.github.io/teeth3ds) | other | medium | B S | none | download now |  |
| [Teeth3DS+ / 3DTeethSeg'22 / 3DTeethLand](https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges) | other | medium | B S | none | download now |  |
| [ToothFairy2 (AImageLab, University of Modena and Reggio Emilia)](https://ditto.ing.unimore.it/toothfairy2/) | 3D volume | medium | B S | none | your action: account / terms |  |
| [Adelaide twin dental collections (Craniofacial Biology Research Group, University of Adelaide)](https://doi.org/10.1017/thg.2012.78) | other | medium | B T A S | MZ/DZ (abstract); DNA zygosity and 125 MZ / 89 DZSS / 49 ... | decision: request |  |
| [Beijing intraoral scan identification set (Zhou et al. 2024)](https://doi.org/10.3390/bioengineering11090873) | other | low | B R S | none | decision: request |  |
| [Hungarian Twin Registry palatal intraoral scan cohort (Semmelweis University; Simon, Vag et al.)](https://doi.org/10.1186/s12903-020-01261-w) | other | low | B R T A S | MZ/DZ by standardized questionnaire (stated nearly 99% ... | decision: request |  |
| [Pre/post-orthodontic 3D dental model dataset (Zenodo 11392406; hospital not confirmed)](https://zenodo.org/records/11392406) | other | low | B S | none | decision: request |  |
| [Yuendumu longitudinal growth study (Central Australia, 1951-1971)](https://doi.org/10.1017/upo9780987073006) | other | medium | B A S | Not confirmed | cut: no access route found |  |

### 3D surface: face / head / body (6)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [3D-TEC (3D Twins Expression Challenge)](https://cvrl.nd.edu/projects/data/) | 3D volume | low | B R T S | MZ, self-reported (survey Table 1): 107 pairs, 214 ... | decision: request |  |
| [3DBodyTex.v1 (with Skin3D lesion annotations)](https://cvi2.uni.lu/3dbodytexv1/) | other | medium | B S | none | decision: request |  |
| [Lithuanian young-adult twins 3D facial scans (Vuollo et al. 2015)](https://doi.org/10.1017/thg.2015.16) | other | low | T | 106 pairs, MZ/DZ by DNA testing | decision: request |  |
| [TwinsUK 3D facial images (3dMDface)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5008732/) | other | low | T S | 1,380 female twins analysed: 263 MZ pairs, 341 DZ pairs ... | decision: institution only |  |
| [CAESAR (Civilian American and European Surface Anthropometry Resource)](https://saemobilus.sae.org/content/RR-017CD) | 3D volume | medium | B S | none | cut: unverifiable |  |
| [CASIA-3D FaceV1](http://www.cbsr.ia.ac.cn) | 3D volume | low | B R S | none | cut: unverifiable |  |

### body: gait (vision) (2)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [BVC-UNN Body Dataset (CVRL)](https://cvrl.nd.edu/projects/data/) | - | - | - | - | lead: unverified |  |
| [OU-ISIR Gait Database, Multi-View Large Population (OU-MVLP)](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html) | video | low | S | none | cut: no access route found |  |

### eye: OCT / OCTA (12)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [Guangzhou Twin Eye Study (GTES)](https://www.cambridge.org/core/product/identifier/S183242741900118X/type/journal_article) | 2D image | low | T S | MZ/DZ genotyped (16 STR markers, PowerPlex 16) for same-sex ... | decision: request |  |
| [Isfahan MISP: OCT data and colour fundus images of left and right eyes of 50 healthy persons](https://misp.mui.ac.ir/en/database) | 2D image | low | B S | none | decision: request |  |
| [OCTA-500](https://ieee-dataport.org/open-access/octa-500) | 3D volume | low | S | none | decision: request |  |
| [TwinsUK eye imaging (SD-OCT, colour fundus)](https://twinsuk.ac.uk/researchers/access-data-and-samples/request-access/) | 3D volume | low | B T S | MZ/DZ labelled (method not stated in abstracts read): 1,150 ... | decision: request |  |
| [UK Biobank eye measures (fundus photographs + retinal OCT)](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21016) | 2D image | low | B A S | not labelled in eye fields; genetic kinship (KING) ... | decision: request |  |
| [UK Biobank: genetically inferred relatives (fundus/OCT, MRI)](https://biobank.ctsu.ox.ac.uk/ukb/ukb/docs/ukb_genetic_data_description.txt) | 2D image | medium | T S | MZ twins not stated in the official document; relatedness ... | decision: request |  |
| [ZJUT-EIFD (external and internal fingerprint database)](https://doi.org/10.1109/TPAMI.2023.3334760) | 3D volume | low | - | none | decision: request |  |
| [EMIF-AD PreclinAD Amsterdam elderly MZ twins (SD-OCT, fundus, MRI, amyloid PET)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6900176) | - | - | - | - | lead: unverified |  |
| [AI-READI](https://docs.aireadi.org/docs/3/dataset/retinal-photography/) | 2D image | high | S | none | cut: no access route found |  |
| [Copenhagen Twin Cohort Eye Study](https://doi.org/10.1111/aos.15136) | 2D image | medium | T A S | MZ/DZ genotyped (forensic genetic testing) ... | cut: no access route found |  |
| [Healthy Twin Study, Korea (retinal OCT)](https://doi.org/10.1136/bjophthalmol-2020-316512) | 3D volume | low | T S | MZ pairs labelled (count and method not in abstract) ... | cut: no access route found |  |
| [Multi-device SD-OCT reproducibility set (Iowa Reference Algorithm study)](https://arxiv.org/abs/1612.06442) | 3D volume | high | R | none | cut: no access route found |  |

### eye: fundus (18)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [Chákṣu](https://figshare.com/articles/dataset/Ch_k_u_A_glaucoma_specific_fundus_image_database/20123135) | 2D image | high | S | none | download now |  |
| [DeepDRiD](https://github.com/deepdrdoc/DeepDRiD) | 2D image | medium | S | none | download now |  |
| [PAPILA](https://doi.org/10.6084/m9.figshare.14798004.v1) | 2D image | low | B S | none | download now | papila |
| [RIDB (Retina Identification Database)](https://data.mendeley.com/datasets/tjw3zwntv6/1) | 2D image | low | B R S | none | download now | ridb |
| [AREDS (Age-Related Eye Disease Study) photographs, dbGaP phs000001](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000001.v3.p1) | 2D image | medium | A S | none | decision: request |  |
| [BRSET (Brazilian Multilabel Ophthalmological Dataset)](https://physionet.org/content/brazilian-ophthalmological/) | 2D image | medium | B S | none | decision: request |  |
| [Danish young-adult twin fundus cohort (retinal vascular fractal study)](https://doi.org/10.1167/iovs.17-22072) | 2D image | low | T S | MZ/DZ labelled (method not stated in abstract): 50 MZ + 49 ... | decision: request |  |
| [FCP paired-camera fundus set (Topcon vs Mediwork portable)](https://arxiv.org/abs/2206.09202) | 2D image | high | B S | none | decision: request |  |
| [Twins Eye Study in Tasmania (TEST) + Brisbane Adolescent Twin Study (BATS) retinal photographs](https://doi.org/10.1161/hypertensionaha.109.132902) | 2D image | medium | B T S | 374 MZ + 536 DZ pairs plus 322 siblings; zygosity method ... | decision: request |  |
| [AREDS fundus photographs (dbGaP phs000001)](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000001.v3.p1) | - | - | - | - | lead: unverified |  |
| [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) | - | - | - | - | lead: unverified |  |
| [UK Biobank genetic relatedness (Bycroft 2018) as an L3 source for UKB fundus](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6786975/fullTextXML) | - | - | - | - | lead: unverified |  |
| [FIRE (Fundus Image Registration dataset)](https://projects.ics.forth.gr/cvrl/fire/) | 2D image | low | R S | none | cut: unverifiable |  |
| [MoDRIA (Multimodal Database of Retinal Images in Africa)](https://www.must.ac.ug/mudsreh/mbarara-university-data-science-research-hub-multimodal-database-of-retinal-images-in-africa-modria/) | 2D image | low | S | none | cut: no access route found |  |
| [ODIR-5K (Ocular Disease Intelligent Recognition)](https://odir2019.grand-challenge.org/dataset/) | 2D image | medium | B S | none | cut: no access route found |  |
| [Rotterdam Study ophthalmic fundus archive](https://arxiv.org/abs/2608.31094) | 2D image | high | A S | none | cut: no access route found |  |
| [TOP (Tsukazaki Optos Public) UWF dataset](unverified) | 2D image | medium | B R S | none | cut: unverifiable |  |
| [VARIA retinal authentication database](http://www.varpa.es/research/biometrics.html) | 2D image | low | R S | none | cut: unverifiable |  |

### eye: iris (19)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [IIT Delhi Iris Database v1.0](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm) | 2D image | low | R S | none | decision: request |  |
| [ND MZ/NMZ Iris Dataset](https://cvrl.nd.edu/projects/data/) | 2D image | low | B S | none in the release: 'MZ' pairs are left-right eyes of one ... | decision: request |  |
| [ND Monozygotic / Non-monozygotic (MZ/NMZ) Iris Dataset (Notre Dame CVRL)](https://cvrl.nd.edu/projects/data/) | 2D image | low | B R S | none in the released set. Its 'MZ' pairs are the left vs ... | decision: request |  |
| [ND-CrossSensor-Iris-2013](https://cvrl.nd.edu/projects/data/) | 2D image | low | R S | none | decision: request |  |
| [ND-IRIS-0405](https://cvrl.nd.edu/projects/data/) | 2D image | low | B R S | none | decision: request |  |
| [ND-Iris-Template-Aging-2008-2010](https://cvrl.nd.edu/projects/data/) | 2D image | low | R A S | none | decision: request |  |
| [ND-TimeLapseIris-2012](https://cvrl.nd.edu/projects/data/) | 2D image | low | B R A | none | decision: request |  |
| [SBVPI (Sclera Blood Vessels, Periocular and Iris)](https://sclera.fri.uni-lj.si/datasets.html) | 2D image | low | B | none | decision: request |  |
| [UBIRIS.v2](https://iris.di.ubi.pt/ubiris2.html) | 2D image | high | B R S | none | decision: request |  |
| [WVU / FBI Twins Day multimodal biometric collection (Twinsburg, Ohio, 2010-2019)](https://biic.wvu.edu/data-sets/twins-day-dataset-2010-1015) | 2D image | medium | B R T S | self-report: 'identical', 'identical-mirror' and ... | decision: request |  |
| [BXGrid identical-twin iris images (Notre Dame; Hollingsworth 2010/2011, Yuan 2025)](https://github.com/yongleyuan/mz-iris-pairing) | - | - | - | - | lead: unverified |  |
| [CASIA multimodal twins and families (Sun et al. 2010; Han et al.)](https://academicweb.nd.edu/~kwb/Bowyer_Flynn_BTAS_2016.pdf) | - | - | - | - | lead: unverified |  |
| [ND Twins Days 2009 iris twins (Hollingsworth et al.)](https://academicweb.nd.edu/~kwb/Hollingsworth_EtAl_CVIU_2011.pdf) | - | - | - | - | lead: unverified |  |
| [ND-GFI (Gender From Iris), Notre Dame](https://cvrl.nd.edu/projects/data/) | - | - | - | - | lead: unverified |  |
| [CASIA multimodal twins collection (Beijing twins festival; Sun et al. 2010, Tao et al. 2012)](https://doi.org/10.1117/12.851369) | 2D image | low | B R T S | MZ/DZ assigned by looking at face images (not genotyped) ... | cut: no access route found |  |
| [CASIA-Iris-Africa](https://arxiv.org/abs/2302.13049) | 2D image | medium | B R S | none | cut: no access route found |  |
| [CSIP (Cross-Sensor Iris and Periocular)](unverified (csip.di.ubi.pt did not respond)) | 2D image | high | R S | none | cut: unverifiable |  |
| [PolyU Cross-Spectral Iris Database](https://www4.comp.polyu.edu.hk/~csajaykr/database.php) | 2D image | low | B R S | none | cut: no access route found |  |
| [SDUMLA-HMT](http://mla.sdu.edu.cn/sdumla-hmt.html) | 2D image | low | B R S | none | cut: unverifiable |  |

### eye: periocular / sclera (3)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [MASD (Multi-Angle Sclera Dataset)](https://sites.google.com/hyderabad.bits-pilani.ac.in/mig/datasets) | 2D image | low | B S | none | decision: request |  |
| [UFPR-Periocular](https://web.inf.ufpr.br/vri/databases/ufpr-periocular/) | 2D image | high | B R S | none | decision: request |  |
| [UBIPr (UBI Periocular)](https://iris.di.ubi.pt/ubipr.html) | 2D image | medium | B R S | none | cut: unverifiable |  |

### hand: fingerprint (10)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [SOCOFing (Sokoto Coventry Fingerprint Dataset)](https://arxiv.org/abs/1807.10609) | 2D image | medium | B S | none | your action: account / terms |  |
| [CAS twin fingerprint dataset (Tao et al. 2012)](https://doi.org/10.1371/journal.pone.0035704) | 2D image | low | R T | 83 identical pairs. For the overlapping Sun et al. 2010 ... | decision: request |  |
| [NIST Special Database 300 (SD 300)](https://www.nist.gov/itl/iad/image-group/nist-special-database-300) | 2D image | low | B S | none | decision: request |  |
| [PolyU Contactless 2D to Contact-based 2D Fingerprint Database](https://www4.comp.polyu.edu.hk/~csajaykr/database.php) | - | - | - | - | lead: unverified |  |
| [Srihari et al. 2008 twin fingerprints (J Forensic Identification 58(1))](https://academicweb.nd.edu/~kwb/Bowyer_Flynn_BTAS_2016.pdf) | - | - | - | - | lead: unverified |  |
| [Brazil on-production longitudinal neonatal fingerprint database](https://arxiv.org/abs/2504.20104) | 2D image | medium | R A | none | cut: no access route found |  |
| [FVC2004 (DB1-DB4)](http://bias.csr.unibo.it/fvc2004/databases.asp) | 2D image | medium | R | none | cut: no access route found |  |
| [IIITD MOLF (Multi-sensor Optical and Latent Fingerprint)](https://research.iiitd.edu.in/groups/iab/molf.html) | 2D image | medium | R S | none | cut: no access route found |  |
| [Infant-ID longitudinal infant fingerprints (MSU / Agra)](https://arxiv.org/abs/2010.03624) | 2D image | medium | R A | none | cut: no access route found |  |
| [Twin ten-print live-scan set (Liu and Srihari 2009)](https://doi.org/10.1007/978-3-642-03521-0_5) | 2D image | medium | B T S | 188 MZ + 39 DZ sets (Bowyer-Flynn 2016); zygosity method ... | cut: no access route found |  |

### hand: palm / vein / knuckle / geometry (29)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [11K Hands](https://sites.google.com/view/11khands) | 2D image | low | B R S | none | download now | 11k-hands |
| [Dorsal Hand Veins DB1/DB2 (Pontificia Universidad Javeriana)](https://github.com/wilchesf/dorsalhandveins) | 2D image | low | B R S | none | download now | dorsal-hand-veins |
| [MPD (Tongji Mobile Palmprint Database)](https://cslinzhang.github.io/MobilePalmPrint/) | 2D image | high | B R S | none | download now |  |
| [Mutah MOHI / WEHI hand image databases](https://www.mutah.edu.jo/biometrix/hand-images-databases.aspx) | 2D image | high | R S | none | download now |  |
| [Tongji Contactless Palmprint Dataset](https://cslinzhang.github.io/ContactlessPalm/) | 2D image | low | B R S | none | download now | tongji-contactless-palmprint |
| [FV-USM finger vein database](http://drfendi.com/fv_usm_database/) | 2D image | low | B R S | none | your action: account / terms |  |
| [Osteoarthritis Initiative (OAI)](https://nda.nih.gov/oai/) | 2D image | low | B A S | none | your action: account / terms |  |
| [IIT Delhi Finger Knuckle Database v1.0](http://www4.comp.polyu.edu.hk/~csajaykr/IITD/iitd_knuckle.htm) | 2D image | medium | S | none | decision: request |  |
| [PolyU 3D Finger Knuckle Images Database](https://www4.comp.polyu.edu.hk/~csajaykr/3DKnuckle.htm) | other | medium | R S | none | decision: request |  |
| [PolyU Contact-free 3D/2D Hand Images Database v1.0](http://www4.comp.polyu.edu.hk/~csajaykr/myhome/database_request/3dhand/Hand3D.htm) | other | low | R S | none | decision: request |  |
| [PolyU-IITD Contactless Palmprint Images Database v3.0](https://www4.comp.polyu.edu.hk/~csajaykr/palmprint3.htm) | 2D image | medium | B R A S | none | decision: request |  |
| [Tecnocampus Hand Image Database](https://doi.org/10.1007/s12559-013-9230-3) | 2D image | medium | R S | none | decision: request |  |
| [UTFVP (University of Twente Finger Vascular Pattern)](https://www.utwente.nl/en/eemcs/ds/downloads/utfvp/) | 2D image | low | B R | none | decision: request |  |
| [FYO multimodal vein database (Eastern Mediterranean University)](https://fyo.emu.edu.tr/en) | - | - | - | - | lead: unverified |  |
| [Genetically identical forearm and thigh vein patterns (NTU, Zhang, Tang, Li, Kong)](https://personal.ntu.edu.sg/adamskong/publication/Hengyi%20SSCI%20preprint.pdf) | - | - | - | - | lead: unverified |  |
| [IIT Delhi Touchless Palmprint Database v1.0](https://www4.comp.polyu.edu.hk/~csajaykr/database.php) | - | - | - | - | lead: unverified |  |
| [PUT Vein database (Poznan University of Technology)](https://pythonhosted.org/bob.db.putvein/) | - | - | - | - | lead: unverified |  |
| [PolyU-IITD Contactless Palmprint v3.0](https://web.comp.polyu.edu.hk/csajaykr/database.php) | - | - | - | - | lead: unverified |  |
| [UTFVP finger vein (University of Twente)](https://www.utwente.nl/en/eemcs/dmb/downloads/utfvp/) | - | - | - | - | lead: unverified |  |
| [AAOF Craniofacial Growth Legacy Collection (incl. Forsyth Twin Sample, Bolton-Brush, Michigan, Oregon, Burlin…](https://www.aaoflegacycollection.org) | 2D image | medium | B T A S | Present (Forsyth Twin Sample, Moorrees study); zygosity by ... | cut: unverifiable |  |
| [Bosphorus Hand Vein Database](http://bosphorus.ee.boun.edu.tr/) | 2D image | low | S | none | cut: unverifiable |  |
| [CASIA Multi-Spectral Palmprint Database (CASIA-MS-PalmprintV1)](https://tsapps.nist.gov/BDBC/Search/Details/225) | 2D image | low | B R S | none | cut: no access route found |  |
| [Dorsal Hand Dataset for Recognition System (IEEE DataPort)](https://ieee-dataport.org/documents/dorsal-hand-dataset-recognition-system) | 2D image | low | B S | none | cut: commercial |  |
| [Identical-twin palmprint set (Kong, Zhang, Lu 2006)](https://doi.org/10.1016/j.patcog.2006.04.035) | 2D image | low | B T S | MZ only, 53 pairs (Bowyer-Flynn 2016); zygosity method not ... | cut: no access route found |  |
| [PolyU Contactless Hand Dorsal Images Database](https://www4.comp.polyu.edu.hk/~csajaykr/database.php) | 2D image | medium | R A S | none | cut: no access route found |  |
| [PolyU Finger Image Database v1.0](https://www4.comp.polyu.edu.hk/~csajaykr/database.php) | 2D image | low | R S | none | cut: no access route found |  |
| [PolyU Finger-Knuckle-Print (FKP) Database](http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm) | 2D image | low | B R S | none | cut: unverifiable |  |
| [PolyU Multispectral Palmprint Database](https://tsapps.nist.gov/BDBC/Search/Details/476) | 2D image | low | B R S | none | cut: no access route found |  |
| [UHDVD (Unconstrained Hand Dorsal Veins Database)](https://www.i-csrs.org/database/Veins_Database.rar) | 2D image | medium | B S | none | cut: unavailable |  |

### head: ear (6)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [AMI Ear Database](https://ctim.ulpgc.es/research_works/ami_ear_database/) | 2D image | low | B R S | none | download now | ami-ear |
| [EarVN1.0](https://data.mendeley.com/datasets/yws3v3mwx3) | 2D image | high | B S | none | download now |  |
| [Wonder Ears identical-twin ear images (Nejati et al. 2012)](https://dblp.org/rec/conf/icpr/NejatiZSM012.html) | 2D image | medium | B T | 39 identical pairs; zygosity method not stated | decision: request |  |
| [ND-Collection F and G (CVRL)](https://cvrl.nd.edu/projects/data/) | - | - | - | - | lead: unverified |  |
| [Queensland Twin Adolescent Brain (QTAB), OpenNeuro ds004146](https://openneuro.org/datasets/ds004146) | - | - | - | - | lead: unverified |  |
| [AWE / UERC (Annotated Web Ears)](https://service.tib.eu/ldmservice/dataset/annotated-web-ears--awe--dataset) | 2D image | high | B S | none | cut: unverifiable |  |

### head: face 2D / video (40)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [CACD (Cross-Age Celebrity Dataset)](https://bcsiriuschen.github.io/CARC/) | 2D image | high | A S | none | download now |  |
| [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html) | 2D image | low | B R S | none | download now | fei-face |
| [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) | 2D image | high | A S | none | download now |  |
| [MUCT Landmarked Face Database](http://www.milbo.org/muct/) | 2D image | low | B R S | none | download now | muct |
| [Notre Dame FACE Features Set (includes twins photographs, features only)](https://cvrl.nd.edu/projects/data/) | other | medium | T | twins photographs included; labelling not stated | download now |  |
| [SpeakingFaces](https://issai.nu.edu.kz/download-speaking-faces/) | video | low | B R S | none | download now |  |
| [TSKinFace (Tri-Subject Kinship Face)](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/TSKinFace.html) | 2D image | high | S | none | download now |  |
| [Chicago Face Database (CFD, CFD-MR, CFD-INDIA)](https://www.chicagofaces.org) | 2D image | low | B S | none | your action: account / terms |  |
| [FIW (Families in the Wild)](https://fulab.sites.northeastern.edu/fiw-download) | 2D image | high | S | none labelled (not re-verified) | your action: account / terms |  |
| [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/) | 2D image | high | A S | none | decision: request |  |
| [Color FERET](https://www.nist.gov/programs-projects/face-recognition-technology-feret) | 2D image | medium | B R S | none | decision: request |  |
| [Danish Twin Registry: LSADT facial photographs, perceived-age study (Christensen et al. 2009)](https://www.sdu.dk/en/forskning/dtr/researcher/guidelines) | 2D image | medium | T S | 1,826 twins aged 70+ (MZ and DZ); counts and zygosity ... | decision: request |  |
| [Fine-Grained Face Verification dataset of identical twins (Hu, Lu, Tan 2015)](https://doi.org/10.1109/icb.2015.7139079) | 2D image | high | T | 455 MZ pairs scraped from the web; zygosity unverified | decision: request |  |
| [Iranian web-scraped twins face database (Mozaffari & Behravan 2011)](unverified) | 2D image | high | T | 1,902 twin pairs; zygosity unknown (scraped) | decision: request |  |
| [Lock3DFace](https://irip.buaa.edu.cn/lock3dface/index.html) | video | low | R S | none | decision: request |  |
| [ND Twins Face Recognition Test Set (Notre Dame CVRL)](https://cvrl.nd.edu/projects/data/) | 2D image | medium | T | identical twins form the different-person pairs; labelling ... | decision: request |  |
| [ND-Collection B](https://cvrl.nd.edu/projects/data/) | 2D image | low | B R A S | none | decision: request |  |
| [ND-NIVL](https://cvrl.nd.edu/projects/data/) | 2D image | medium | R S | none | decision: request |  |
| [ND-TWINS-2009-2010](https://cvrl.nd.edu/projects/data/) | 2D image | medium | B R T A S | MZ only, self-identified: Paone 2014 says all participants ... | decision: request |  |
| [SCface (Surveillance Cameras Face Database)](https://scface.org/) | 2D image | high | R S | none | decision: request |  |
| [Face Research Lab London Set](https://api.figshare.com/v2/articles/5047666) | - | - | - | - | lead: unverified |  |
| [HDA Doppelganger face database (Rathgeb et al. 2021)](https://api.openalex.org/works/doi:10.1109/biosig52210.2021.9548306) | - | - | - | - | lead: unverified |  |
| [NIST Special Database 18 - Mugshot Identification Database](https://www.nist.gov/srd/nist-special-database-18) | - | - | - | - | lead: unverified |  |
| [NIST Special Database 32 - Multiple Encounter Dataset (MEDS-I/II)](https://www.nist.gov/itl/iad/btg/special-database-32-multiple-encounter-dataset-meds) | - | - | - | - | lead: unverified |  |
| [WVU non-twin biometric collection 2008-2019](https://arxiv.org/pdf/2208.11822) | - | - | - | - | lead: unverified |  |
| [ARL-VTF (ARL Visible-Thermal Face)](https://openaccess.thecvf.com/content/WACV2021/html/Poster_A_Large-Scale_Time-Synchronized_Visible_and_Thermal_Face_Dataset_WACV_2021_paper.html) | 2D image | medium | R S | none | cut: no access route found |  |
| [CALFW (Cross-Age LFW)](http://whdeng.cn/CALFW/index.html) | 2D image | high | A S | none | cut: no access route found |  |
| [CASIA NIR-VIS 2.0](http://www.cbsr.ia.ac.cn) | 2D image | medium | B R A S | none | cut: no access route found |  |
| [CASIA-Twins (Beijing Twins Festival multibiometric)](http://biometrics.idealtest.org) | 2D image | medium | B T S | MZ/DZ assigned by observers from face similarity: 51 MZ + ... | cut: no access route found |  |
| [CFP (Celebrities in Frontal-Profile), CFP-FP protocol](http://www.cfpw.io/) | 2D image | high | B S | none | cut: no access route found |  |
| [CMU Multi-PIE](https://www.ri.cmu.edu/projects/project_418.html) | 2D image | low | B R S | none | cut: commercial |  |
| [Celeb Twins Test Set (CTTS)](https://arxiv.org/abs/2609.01141) | 2D image | high | B T S | MZ (celebrity; zygosity source not stated); 80 sets of twins | cut: no access route found |  |
| [DoppelVer](https://par.nsf.gov/biblio/10491771) | 2D image | high | S | none (unrelated look-alikes) | cut: unverifiable |  |
| [FG-NET Aging Database](http://www-prima.inrialpes.fr/FGnet/html/benchmarks.html) | 2D image | high | A S | none | cut: unverifiable |  |
| [Family101](https://chenlab.ece.cornell.edu/projects/KinshipClassification/) | 2D image | high | S | none | cut: unverifiable |  |
| [IIIT-Delhi Look-alike Face Database](https://research.iiitd.edu.in/groups/iab/facedatabases.html) | 2D image | high | S | none (look-alikes) | cut: unverifiable |  |
| [KinFaceW-I / KinFaceW-II](https://www.kinfacew.com) | 2D image | high | S | none | cut: no access route found |  |
| [LAG (Large Age-Gap)](https://arxiv.org/abs/1602.06149) | 2D image | high | A S | none | cut: no access route found |  |
| [MORPH (Academic / Longitudinal)](https://uncw.edu/research/innovation/technology-portfolio/morph) | 2D image | medium | B A S | none | cut: unverifiable |  |
| [Tufts Face Database](https://tdface.ece.tufts.edu) | 2D image | medium | R S | none | cut: no access route found |  |

### other / unclassified (1)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [TwinsUK gut microbiome (Goodrich et al.)](https://doi.org/10.1016/j.cell.2014.09.053) | tabular or omics | medium | T S | present; the abstract says 416 twin pairs without giving ... | cut: no access route found |  |

### radiology 2D: chest (8)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [ChestX-ray14 (NIH CXR8/CXR14)](https://nihcc.app.box.com/v/ChestXray-NIHCC) | 2D image | low | R A S | none | download now |  |
| [NIAID TB Portals](https://tbportals.niaid.nih.gov/) | 2D image | high | S | none | your action: account / terms |  |
| [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | 2D image | low | R A S | none | your action: account / terms |  |
| [BRAX (Brazilian chest X-ray dataset)](https://physionet.org/content/brax/) | 2D image | low | R A S | none | decision: request |  |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | 2D image | low | R A S | none | decision: request |  |
| [MIMIC-CXR / MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr/) | 2D image | low | R A S | none | decision: request |  |
| [VinDr-CXR](https://physionet.org/content/vindr-cxr/) | 2D image | low | S | none | decision: request |  |
| [Chest ImaGenome (PhysioNet)](https://physionet.org/content/chest-imagenome/) | - | - | - | - | lead: unverified |  |

### radiology 2D: dental (7)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [Panoramic Dental X-rays With Segmented Mandibles (Noor Medical Imaging Center)](https://data.mendeley.com/datasets/hxt48yk462) | 2D image | low | S | none | download now |  |
| [STS-Tooth (STS-3D-Tooth CBCT part)](https://doi.org/10.5281/zenodo.10597292) | 3D volume | medium | B S | none | download now |  |
| [Korean paired dental panoramic radiographs (Choi et al. 2024)](https://doi.org/10.1371/journal.pone.0312537) | 2D image | medium | R A S | none | decision: request |  |
| [Multimodal dental dataset (Guangxi; Huang, Liu, Tang et al.) on PhysioNet](https://physionet.org/content/multimodal-dental-dataset/1.1.0/) | 3D volume | medium | B R S | none | decision: request |  |
| [SNUDH antemortem/postmortem panoramic pairs (Choi et al.)](https://doi.org/10.1371/journal.pone.0312537) | 2D image | low | A S | none | decision: institution only |  |
| [Seoul National University Dental Hospital panoramic AM/PM pairs (Choi et al. 2024)](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11500890/fullTextXML) | - | - | - | - | lead: unverified |  |
| [Tufts Dental Database](https://tdd.ece.tufts.edu/) | 2D image | low | S | none | cut: no access route found |  |

### radiology 2D: mammography (5)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [CSAW-CC (Cohort of Screen-Aged Women, case-control subset)](https://researchdata.se/en/catalogue/dataset/2021-204-1) | 2D image | low | B A S | none | decision: request |  |
| [EMBED (EMory BrEast imaging Dataset)](https://doi.org/10.1148/ryai.220047) | 2D image | low | B R A S | none | decision: request |  |
| [OPTIMAM Mammography Image Database (OMI-DB)](https://medphys.royalsurrey.nhs.uk/omidb/) | 2D image | low | B A S | none | decision: request |  |
| [VinDr-Mammo](https://www.physionet.org/content/vindr-mammo/) | 2D image | low | B S | none | decision: request |  |
| [CMMD (Chinese Mammography Database, TCIA)](https://www.cancerimagingarchive.net/collection/cmmd/) | - | - | - | - | lead: unverified |  |

### radiology 2D: skeletal / DXA (4)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [UK Biobank imaging enhancement](https://doi.org/10.1038/s41467-020-15948-9) | 3D volume | low | S | none stated (population cohort, not a twin registry) | decision: request |  |
| [Osteoporotic Fractures in Men (MrOS) Online](https://mrosonline.ucsf.edu/) | - | - | - | - | lead: unverified |  |
| [TwinsUK / Healthy Ageing Twin Study longitudinal DXA (hip, lumbar spine, forearm)](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&rettype=abstract&retmode=text&id=22589082,18810303) | - | - | - | - | lead: unverified |  |
| [UK Biobank DXA images (field 20158)](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20158) | - | - | - | - | lead: unverified |  |

### radiology 2D: skull / cephalogram / sinus (5)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [Aariz cephalometric benchmark](https://doi.org/10.6084/m9.figshare.27986417) | 2D image | high | S | none | download now |  |
| [Brazilian frontal sinus CT recognition set (de Souza, Marana, Weber)](https://doi.org/10.1016/j.forsciint.2018.03.029) | 3D volume | medium | S | none | decision: request |  |
| [CQ500 head CT (CARING, New Delhi; Qure.ai)](http://headctstudy.qure.ai) | 3D volume | medium | B S | none | decision: request |  |
| [Copenhagen MZ twin frontal sinus profile radiographs (Kjaer et al.)](https://doi.org/10.3109/00016489.2012.677064) | 2D image | medium | T S | MZ only; zygosity method not stated in abstract; 42 pairs | decision: request |  |
| [Frontal sinus antemortem/postmortem CT pairs (Torimitsu et al., Tokyo)](https://doi.org/10.1007/s00414-025-03564-5) | 2D image | high | R A S | none | decision: request |  |

### skin: moles / dermoscopy / photos (6)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [ISIC 2024 SLICE-3D](https://challenge.isic-archive.com/data/2024/) | 2D image | low | S | none | download now |  |
| [SIIM-ISIC 2020 Melanoma Classification dataset](https://doi.org/10.34970/2020-ds01) | 2D image | medium | S | none | download now |  |
| [SLICE-3D (ISIC 2024)](https://doi.org/10.1038/s41597-024-03743-w) | 2D image | medium | S | none | download now |  |
| [UQ longitudinal 3D-TBP tile + dermoscopy dataset (Mind Your Moles + HOPS cohorts)](https://doi.org/10.48610/a13deaf) | 2D image | low | B R A S | none (relatedness not recorded; only a 'close blood ... | download now |  |
| [iToBoS dataset (3D-TBP skin regions)](https://doi.org/10.6084/m9.figshare.28452545) | 2D image | low | - | none | download now |  |
| [Brisbane Twin Naevus Study (full-body imaging + dermoscopy)](https://doi.org/10.1111/bjd.14291) | 2D image | medium | T S | MZ/DZ labelled (method not stated in the abstract): 76 MZ ... | decision: request |  |

### small traits: lip / nail / foot / capillary / tongue (5)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [Biometric 220x6 Human Footprint (IEEE DataPort)](https://ieee-dataport.org/open-access/biometric-220x6-human-footprint) | 2D image | low | R S | none | your action: account / terms |  |
| [SUT-Lips-DB (database of lip traces)](https://ieee-dataport.org/open-access/sut-lips-db-database-lips-traces) | 2D image | medium | - | none | your action: account / terms |  |
| [ANFC nailfold capillaroscopy dataset (Tsinghua)](https://github.com/THU-CS-PI-LAB/ANFC-Automated-Nailfold-Capillary) | video | medium | - | none | decision: request |  |
| [MNIT Footprint Biometric Database](https://sites.google.com/view/ritikushwaha/footprint-biometric) | 2D image | medium | B | none | decision: request |  |
| [Fingernail plate transient-biometric dataset (Barbosa et al. 2016)](https://www.open-access.bcu.ac.uk/326) | 2D image | medium | R | none | cut: unverifiable |  |

### tomography: brain (25)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [Enhanced NKI-Rockland Sample test-retest (eNKI_RS_TRT)](https://fcon_1000.projects.nitrc.org/indi/pro/eNKI_RS_TRT/FrontPage.html) | 3D volume | low | B R | none | download now |  |
| [Kirby21: Multi-Modal MRI Reproducibility Resource (Kennedy Krieger Institute)](https://www.nitrc.org/projects/multimodal) | 3D volume | low | B R | none | your action: account / terms |  |
| [ABCD Study twin hub](https://abcdstudy.org) | 3D volume | high | B T A S | MZ/DZ genotyped: 772 pairs analysed (Maes 2022); about 800 ... | decision: request |  |
| [ABCD Study twin hub subsample](https://www.nbdc-datahub.org/data-access-process) | 3D volume | high | T S | MZ/DZ genotyped: 772 pairs aged 9-10 analysed with genotype ... | decision: request |  |
| [Beijing Twin Study (BeTwiSt) MRI subsample / Beijing Twins Brain-Behavior Association Project](https://atlaslongitudinaldatasets.ac.uk/datasets/betwist) | 3D volume | medium | T A | MZ and DZ (Atlas); 54 MZ + 54 DZ MRI pairs and ... | decision: request |  |
| [BrainSCALE (Netherlands Twin Register longitudinal twin MRI)](https://ntr-data-request.psy.vu.nl) | 3D volume | medium | T A | MZ and DZ twins plus older siblings; zygosity method not ... | decision: request |  |
| [HCP Young Adult: MEG subset and 3T retest (correction/addition)](https://humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) | other | low | B R T S | MZ/DZ genotyped for part of the sample; 36 self-reported DZ ... | decision: request |  |
| [Lifespan HCP in Aging (HCP-A) / Adult Aging Brain Connectome (AABC)](https://www.humanconnectome.org/study/hcp-lifespan-aging) | 3D volume | medium | B A S | none stated | decision: request |  |
| [Lifespan HCP in Development (HCP-D)](https://www.humanconnectome.org/study/hcp-lifespan-development) | 3D volume | medium | B S | none stated | decision: request |  |
| [OASIS-3](https://sites.wustl.edu/oasisbrains/home/oasis-3/) | 3D volume | high | B A S | none | decision: request |  |
| [Older Australian Twins Study (OATS)](https://atlaslongitudinaldatasets.ac.uk/datasets/oats) | 3D volume | medium | B T A S | present: 161 MZ and 124 DZ pairs plus 1 MZ triplet set ... | decision: request |  |
| [TwinsUK Twins MR Imaging Study (NCT06014567)](https://clinicaltrials.gov/study/NCT06014567) | 3D volume | low | B T A S | MZ/DZ (TwinsUK); counts not yet stated | decision: request |  |
| [TwinsUK adult twin registry](https://twinsuk.ac.uk/) | other | low | T A S | present, MZ and DZ; the public pages do not state how ... | decision: request |  |
| [TwinsUK lumbar spine MRI (baseline and 8-12-year follow-up)](https://twinsuk.ac.uk/researchers/access-data-and-samples/request-access/) | 3D volume | high | T A S | MZ/DZ present in TwinsUK; zygosity method and pair counts ... | decision: request |  |
| [Consortium for Reliability and Reproducibility (CoRR)](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/) | - | - | - | - | lead: unverified |  |
| [HCP Young Adult S1200 (if not already on the known list)](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) | - | - | - | - | lead: unverified |  |
| [Midnight Scan Club, OpenNeuro ds000224](https://openneuro.org/datasets/ds000224) | - | - | - | - | lead: unverified |  |
| [MyConnectome, OpenNeuro ds000031](https://openneuro.org/datasets/ds000031) | - | - | - | - | lead: unverified |  |
| [ON-Harmony travelling heads, OpenNeuro ds004712](https://openneuro.org/datasets/ds004712) | - | - | - | - | lead: unverified |  |
| [Queensland Twin IMaging (QTIM), OpenNeuro ds004169](https://openneuro.org/datasets/ds004169) | - | - | - | - | lead: unverified | siblings-mri (partial) |
| [SRPBS Traveling Subject MRI dataset](https://bicr-resource.atr.jp/srpbsts/) | - | - | - | - | lead: unverified | srpbs-traveling-subjects (archive, not opened) |
| [VETSA (Vietnam Era Twin Study of Aging)](https://atlaslongitudinaldatasets.ac.uk/datasets/vetsa) | - | - | - | - | lead: unverified |  |
| [Netherlands Twin Register BrainSCALE (Brain Structure and Cognition: Adolescent Longitudinal twin study)](https://doi.org/10.1017/thg.2012.4) | 3D volume | medium | B R T A S | present: 112 families with 9-year-old twins plus an older ... | cut: no access route found |  |
| [TwinsUK MRI study (King's College London, NCT06014567)](https://clinicaltrials.gov/study/NCT06014567) | 3D volume | low | B T A | present (identical and non-identical TwinsUK volunteers) ... | cut: no access route found |  |
| [UK Biobank repeat imaging](https://www.ukbiobank.ac.uk/get-involved/repeat-imaging-project) | 3D volume | medium | B A S | none labelled (not verified) | cut: unverifiable |  |

### tomography: other CT / MRI (6)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [National Lung Screening Trial (NLST)](https://www.cancerimagingarchive.net/collection/nlst/) | 3D volume | high | A S | none | download now |  |
| [QUADRA_HC: total-body [18F]FDG PET/CT test-retest in healthy controls](https://zenodo.org/records/16686025) | 3D volume | low | B R | none | download now |  |
| [RIDER Lung CT ("coffee-break" test-retest)](https://www.cancerimagingarchive.net/collection/rider-lung-ct/) | 3D volume | medium | R | none | download now |  |
| [VerSe'19/'20 vertebrae CT benchmark](https://github.com/anjany/verse) | 3D volume | high | S | none | download now |  |
| [HNSCC-3DCT-RT (TCIA)](https://www.cancerimagingarchive.net/collection/hnscc-3dct-rt/) | 3D volume | low | B R S | none | decision: request |  |
| [New Mexico Decedent Image Database (NMDID)](https://nmdid.unm.edu/) | 3D volume | low | B S | none | decision: request |  |

### nonvisual (64)

| Dataset | Dim | Shift | Axes | Twins | Status | Local |
|---|---|---|---|---|---|---|
| [1000 Genomes Project 30x high-coverage cohort (expanded, 602 trios)](https://doi.org/10.1016/j.cell.2022.08.004) | tabular or omics | low | S | none stated | download now |  |
| [CMU Keystroke Dynamics Benchmark Data Set](https://roymaxion.github.io/projects/keystroke-benchmark/benchmark-data-set.html) | other | low | R S | none | download now |  |
| [CN-Celeb (multi-genre speaker recognition)](https://openslr.org/82) | 1D signal | high | R S | none | download now |  |
| [CODE-15% (12-lead ECG, Telehealth Network of Minas Gerais)](https://zenodo.org/records/4916206) | 1D signal | medium | R A S | none | download now |  |
| [Danish twin methylation series (GEO GSE61496)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE61496) | tabular or omics | medium | T S | present and monozygotic: the GEO record's primary ... | download now |  |
| [E-Risk twin methylation (GEO GSE105018)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE105018) | tabular or omics | medium | T S | present, MZ/DZ; the data paper reports site-specific ... | download now |  |
| [ECG-ID Database (PhysioNet)](https://physionet.org/content/ecgiddb/1.0.0/) | 1D signal | low | R S | none | download now |  |
| [GEUVADIS RNA-sequencing of 1000 Genomes lymphoblastoid lines](https://doi.org/10.1038/nature12531) | tabular or omics | medium | S | none | download now |  |
| [GazeBase](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) | 1D signal | low | R A S | none | download now |  |
| [Heartprint (multisession finger ECG)](https://figshare.com/articles/dataset/Heartprint_A_Multisession_ECG_Dataset_for_Biometric_Recognition/20105354) | 1D signal | low | R A S | none | download now |  |
| [HuGaDB (Human Gait Database)](https://github.com/romanchereshnev/HuGaDB) | 1D signal | low | B | none | download now |  |
| [Human Microbiome Project phase 1 (HMP1)](https://www.hmpdacc.org/hmp/) | tabular or omics | medium | B R S | none | download now |  |
| [JuDo1000](https://osf.io/5zpvk/) | 1D signal | low | B R S | none | download now |  |
| [Lehne 450K technical replication set (GEO GSE55763)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE55763) | tabular or omics | medium | R S | none | download now |  |
| [Longitudinal EEG Test-Retest Reliability in Healthy Individuals (Universidad de Antioquia)](https://openneuro.org/datasets/ds007176) | 1D signal | low | B R A S | none | download now |  |
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 1D signal | medium | R A S | none | download now |  |
| [Test-retest resting and cognitive-state EEG dataset (Wang et al., Southwest University)](https://openneuro.org/datasets/ds004148) | 1D signal | low | B R S | none | download now |  |
| [HMOG (Hand Movement, Orientation and Grasp) dataset](https://hmog-dataset.github.io/hmog/) | 1D signal | low | R S | none | your action: account / terms |  |
| [SU-AIS BB-MAS (behavioural biometrics multi-device multi-activity)](https://ieee-dataport.org/open-access/su-ais-bb-mas-syracuse-university-and-assured-information-security-behavioral-biometrics) | 1D signal | medium | S | none | your action: account / terms |  |
| [ZJU-GaitAcc](https://ytzhang.pages.dev/datasets/zju-gaitacc) | 1D signal | low | R S | none | your action: account / terms |  |
| [BioSec.Lab PPG Dataset (Biosec1)](https://www.comm.utoronto.ca/~biometrics/PPG_Dataset/index.html) | 1D signal | low | R S | none | decision: request |  |
| [Cleveland Family Study (CFS)](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000284.v2.p1) | 1D signal | high | A S | none stated | decision: request |  |
| [DeepSignDB (on-line signature)](https://github.com/BiDAlab/DeepSignDB) | 1D signal | medium | R S | none | decision: request |  |
| [EMIF-AD PreclinAD (Netherlands Twin Register MZ twins aged 60+)](https://doi.org/10.1186/s13195-018-0406-7) | other | medium | B T A S | MZ pairs only, genotyped: Amsterdam buccal cells taken for ... | decision: request |  |
| [GLOBEM (multi-year smartphone and Fitbit sensing)](https://www.physionet.org/content/globem/1.1/) | tabular or omics | high | R A S | none | decision: request |  |
| [MOBIO (bimodal mobile biometric database)](https://idiap.ch/dataset/mobio) | video | medium | R S | none | decision: request |  |
| [MOBIUS (Mobile Ocular Biometrics In Unconstrained Settings)](https://sclera.fri.uni-lj.si/datasets.html) | 2D image | medium | B S | none | decision: request |  |
| [OMEGA: Open MEG Archive (McConnell Brain Imaging Centre, McGill)](https://www.mcgill.ca/bic/neuroinformatics/omega/) | 1D signal | low | B R S | none | decision: request |  |
| [PolyU Contactless Finger Knuckle Images Database v3.0](https://www4.comp.polyu.edu.hk/~csajaykr/fn2.htm) | 2D image | high | R A S | none | decision: request |  |
| [RSR2015 (text-dependent speaker verification corpus)](https://www.a-star.edu.sg/enterprise/innovation-offerings/ready-to-sign-licenses/rsr2015-overview-specifications) | 1D signal | medium | R S | none | decision: request |  |
| [Sleep Heart Health Study (SHHS)](https://biolincc.nhlbi.nih.gov/studies/shhs/) | 1D signal | medium | A S | none | decision: request |  |
| [Temple University Hospital EEG Corpus (TUEG)](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) | 1D signal | high | B R A S | none | decision: request |  |
| [Trinity College Dublin Speaker Ageing database (TCDSA)](https://theses.eurasip.org/theses/555/automatic-recognition-of-ageing-speakers/) | 1D signal | high | A | none | decision: request |  |
| [VoxCeleb1/VoxCeleb2 with cross-age trial lists (Vox-CA)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | 1D signal | high | R A S | none | decision: request |  |
| [ATVS long-term on-line signature database (BiosecurID + Biosecure common users)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0069897) | - | - | - | - | lead: unverified |  |
| [Autonomic Aging cardiovascular dataset (PhysioNet)](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/) | - | - | - | - | lead: unverified |  |
| [CEDAR twins handwriting collection (Srihari 2008)](https://pubmed.ncbi.nlm.nih.gov/18366576/) | - | - | - | - | lead: unverified |  |
| [CVL-Database (offline handwriting, TU Wien)](https://zenodo.org/records/1492267) | - | - | - | - | lead: unverified |  |
| [CYBHi off-the-person ECG (Check Your Biosignals Here)](https://zenodo.org/records/2381823) | - | - | - | - | lead: unverified |  |
| [Cam-CAN (Cambridge Centre for Ageing and Neuroscience)](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/) | - | - | - | - | lead: unverified |  |
| [EEG Motor Movement/Imagery Dataset (PhysioNet eegmmidb)](https://physionet.org/content/eegmmidb/1.0.0/) | - | - | - | - | lead: unverified |  |
| [GEO GSE154566 (longitudinal MZ twin methylation, E-Risk victimization)](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id=200154566) | - | - | - | - | lead: unverified |  |
| [GEO twin-identity methylation series: GSE255870 and GSE225544](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id=200255870,200225544) | - | - | - | - | lead: unverified |  |
| [GaitRec ground reaction force dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC7217853/) | - | - | - | - | lead: unverified |  |
| [MCYT bimodal baseline corpus (fingerprint and signature)](https://api.openalex.org/works/doi:10.1049/ip-vis:20031078) | - | - | - | - | lead: unverified |  |
| [MIMIC-IV-ECG (PhysioNet)](https://physionet.org/content/mimic-iv-ecg/) | - | - | - | - | lead: unverified |  |
| [Sleep-EDF Database Expanded (PhysioNet)](https://physionet.org/content/sleep-edfx/1.0.0/) | - | - | - | - | lead: unverified |  |
| [UNB StepUP-P150 plantar pressure footstep dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12350911/) | - | - | - | - | lead: unverified |  |
| [Audio-Visual Twins Database (AVTD)](unknown) | video | medium | T | present, zygosity not stated; 39 pairs | cut: no access route found |  |
| [BHSig260 (Bengali and Hindi offline signatures)](unknown (SigNet cites a goo.gl short link; goo.gl is discontinued)) | 2D image | low | R S | none | cut: no access route found |  |
| [BMPD (Birjand University Mobile Palmprint Database)](https://cv.birjand.ac.ir/zahiri/en/articlesInPublications/41606) | 2D image | high | S | none | cut: unverifiable |  |
| [BiosecurID](https://arxiv.org/abs/2111.03472) | 2D image | low | R S | none | cut: no access route found |  |
| [BiosecurID multimodal biometric database](https://arxiv.org/abs/2111.03472) | other | low | R S | none | cut: no access route found |  |
| [CEDAR offline signature dataset](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar) | 2D image | low | R S | none | cut: no access route found |  |
| [Identical-twin clean-speech database (Ariyaeeinia et al.)](https://discovery.dundee.ac.uk/en/publications/a-test-of-the-effectiveness-of-speaker-verification-for-different/) | 1D signal | medium | T S | MZ, 49 pairs; zygosity method not stated | cut: no access route found |  |
| [King Saud University Arabic Speech Database (LDC2014S02)](https://catalog.ldc.upenn.edu/LDC2014S02) | 1D signal | medium | R S | none | cut: commercial |  |
| [Longitudinal Corpus of Finnish Spoken in Helsinki (LCFSH)](https://datakatalogi.helsinki.fi/handle/123456789/334) | 1D signal | high | A | none | cut: unverifiable |  |
| [Multi-week MZ twin EEG study (Yang, Hwang, Wu, Huang)](https://doi.org/10.1038/s41598-022-21384-0) | 1D signal | low | R T | present, MZ only: 10 pairs; zygosity method not found in ... | cut: no access route found |  |
| [Netherlands Twin Register adult resting EEG cohort (Smit et al.)](https://doi.org/10.1111/j.1469-8986.2005.00352.x) | 1D signal | low | T S | present: 142 MZ and 167 DZ pairs plus siblings; zygosity ... | cut: no access route found |  |
| [OU-ISIR Inertial Sensor Gait Database](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html) | 1D signal | low | B R S | none | cut: no access route found |  |
| [Spanish siblings speech corpus (San Segundo and Kunzel)](https://loquens.revistas.csic.es/index.php/loquens/article/download/26/75?inline=1) | 1D signal | medium | R T | MZ/DZ labelled (12 MZ pairs; 10 DZ individuals); zygosity ... | cut: no access route found |  |
| [UTSig (University of Tehran Persian offline signature dataset)](https://arxiv.org/abs/1603.03235) | 2D image | low | B R S | none | cut: no access route found |  |
| [Warsaw BioBase-Ageing (iris subset BioBase-Ageing-Iris; companion fingerprint, hand-geometry, face, signature…](https://www.scitepress.org/Papers/2013/42458/42458.pdf) | 2D image | low | B R A | none | cut: no access route found |  |
| [Washington State Twin Registry: Sleep Duration and Metabolism in Twins (actigraphy)](https://wstwinregistry.org/2016/02/16/sleep/) | 1D signal | low | R T | present, MZ only; zygosity method not stated | cut: no access route found |  |
