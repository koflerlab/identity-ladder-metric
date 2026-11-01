# Datasets

Every dataset this project uses or plans to use, as recorded in the shared data registry (`_registry/datasets.json`
of the data root) on 2026-11-01, with facts checked against the dataset's own metadata where stated. No local paths
and no participant IDs. "Licence as recorded" quotes the registry; it is not legal advice.

Project rule for all real data: real images, per-case tables and real-case figures stay in the local data root and
never enter this public repository, whatever the licence allows. The repository holds code, aggregate results and
synthetic figures.

## Brain MRI

| Dataset | Source | Licence as recorded | Local status | Rungs it serves | Redistribution |
|---|---|---|---|---|---|
| QTIM (Queensland Twin IMaging), registry id `siblings-mri` | OpenNeuro ds004169 v1.0.8, doi:10.18112/openneuro.ds004169.v1.0.8 | CC0 in the dataset's `dataset_description.json`; the registry entry still says "terms/consent to verify" and needs updating | partial: 23 of 1,202 participants local, 12 of them with two sessions; metadata complete | repeat scan (test-retest subsample, N = 78, interval about 3.5 months); later visit (longitudinal subsample, N = 62, interval about 3.5 years, adolescents); other hemisphere; family member; stranger; population mean | CC0 permits it; project rule above applies |
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
- Local T1w: 22 first-session scans and 12 second-session scans; 12.4 MB per scan on average (34 local scans), so about 18 GB for all T1w scans of the release.

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
