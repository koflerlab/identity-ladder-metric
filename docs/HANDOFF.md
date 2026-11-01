# Handoff: identity-ladder-metric (updated 2026-11-01, evening)

## Start here (next session)
Load skills first: token-hygiene, data:statistical-analysis, manuscript-compliance-check (human MRI data), bib-verify (any citation), pre-commit-security (before commits), humanizer (any text under Kate's name; no em dashes).
Follow the research discipline of `C:\Projects\02_academia\studies\icml-repro\verdict_audit`: a frozen `PROTOCOL.md` (question, inputs with SHA-256, definitions, claims under test, prespecified interpretation, sensitivity analyses kept separate from primary, evidence boundary, dated amendments), a script that refuses to run if an input hash differs, and a results `manifest.json` (protocol/code/input hashes, git commit, seed, versions). Also look there for how trackio and claims are used before choosing experiment tracking here.

Scope decided by Kate (1 Nov):
- The ISBI 2027 paper (`paper/isbi2027/`, deadline 2 Nov 23:59 ET) is ONLY about regression to the mean in inpainting evaluation, plus how other fields fight it (image restoration, weather, protein structure). The identity ladder is the NEXT paper; keep it out of ISBI.
- Kate wants a REAL case under every synthetic result, with real metrics: (1) intensity table (repeat, other hemisphere, family member, stranger, population mean, mean with restored spread), (2) controlled shrinkage path, (3) label space with a real segmenter, (4) all nine package metrics.
- README = project only; everything about a paper lives in its paper folder (`paper/isbi2027/README.md`). README shows case images only; plots live in `docs/figures_draft.md`.

Do in this order:
1. `docs/DATASETS.md`: every dataset used or planned, from `C:\Projects\data\_registry\datasets.json` (source, licence as recorded, local status, which rung it serves, redistribution limits; no local paths, no participant IDs).
2. Decision trees (`docs/DECISION_TREES.md`): the analysis choices and their branches: stranger reference (matched how), mask placement and size, registration (affine vs nonlinear, template), intensity normalisation, QC exclusions, what counts as "family member" when zygosity is unknown, segmenter (FastSurfer vs SynthSeg) and its failure branch, how many targets.
3. `docs/PROTOCOL_real_cases.md`, frozen and committed BEFORE any real-data result is computed; Kate approves it.
4. Then run it.

Facts already checked (1 Nov, evening):
- QTIM = OpenNeuro ds004169 v1.0.8, CC0, registered as `siblings-mri` (status partial) at `brain_images/siblings`. Only 23 of 1,202 participants are local (12 with two sessions). The public release has `family_id` but NO zygosity: the real rung is "family member (co-twin or sibling)". `group` marks the test-retest (about 3.5 months, N=78) and longitudinal (4 years) subsamples. T1w is about 13 MB per scan, about 18 GB for all T1w; full release 148 GB; C: has 105 GB free. Download T1w only, with the data-root maintenance lock from `C:\Projects\data\AGENTS.md`, and update the registry.
- Docker images present: `deepmi/fastsurfer:latest`, `freesurfer/freesurfer:8.0.0` (has mri_synthseg). No GPU; 8 CPU, 16 GB RAM: budget segmentation runs.
- Codex's real cases (1 QTIM MRI, 1 UTIRIS iris) stay private in `C:/Projects/data/projects/identity-ladder-metric/run-2026-11-01-detail/`.
- A background literature agent was asked to write `docs/REPRODUCTION_BACKLOG.md` (independent search, bib-verify, reproduction priorities). Check whether it exists and review it before relying on it.

Done today (all pushed to GitHub, public repo): Fig. 1 fixed and kept at IEEE size; all-metrics table; H19 topology test; README rebuilt (ladder figure with bilateral and population-mean rungs, domain-shift marker, regression-to-the-mean section citing the 2023 overview and 2025 winner); ISBI draft rebuilt (4 pages, title "Fighting regression to the mean in the evaluation of brain MRI inpainting"); licences MIT + CC BY 4.0. FastSurfer observation from the challenge is unpublished: it is a `\todo` for Florian in the paper.

## Earlier status (verified 1 Nov 2026, detailed real cases and population-null experiment complete)
- **Working convention from Kate (1 Nov 2026):** use the `token-hygiene` skill; record meaningful checkpoints here and commit completed, checked changes in small logical units rather than accumulating a large uncommitted batch. Reuse saved experiment results and avoid repeating checks without a new change or unresolved concern.
- **Detailed MRI preparation complete:** `src/real_mri_detail.py` preserves native T1 intensities (native spacing about 0.9 x 0.9375 x 0.9375 mm) on a 1-mm affine template grid. Eighteen prepared images: first atlas target + later visit, 8 reference people/families, 8 template people from 7 families. Mean uses equal family weights after the pool audit, not 8 independent-family weights. First target and all 18 aligned slices visually reviewed; no gross alignment failure, no score-based case replacement. Fixed posterior cortical ROI: MNI z=55 mm, x=[-45,-13), y=[-65,-33), all 1,024 pixels within template brain. Private output: `C:/Projects/data/projects/identity-ladder-metric/run-2026-11-01-detail/`, registered in the shared dataset registry. Code preparation checkpoint: `3fca61e`.
- **Population-null experiment complete:** `src/population_null.py`, `results/population_null_synthetic.json`, `results/population_null_summary.csv`, `results/population_null_report.md`. Three seeds x eight independent targets per modality: 24 tissue families and 24 eye people. Five controlled shrinkage steps plus repeat, side, stranger and noise controls. Tissue at 50% residual still has mean RMSE/SSIM c=1.000; the pure mean has c=0.995/0.950. Eye mean has c=0.797/0.835. Wrong-detail noise has residual magnitude about one but target projection about zero. Source hashes, exact oracle/null endpoints, metric counts and score/rank bounds checked. Full suite: **103 passed**, 5 existing torchmetrics warnings.
- **Kate's mean-distance requirement implemented as diagnostics:** T8 now records d(target,candidate), d(candidate,mean), d(target,mean), residual magnitude, target projection, residual cosine and orthogonal residual. A combined identity score, thresholds and validation remain open. The controlled shrinkage identities hold by construction and are not independent validation. Noise controls show that atypicality alone is insufficient; residual cosine alone also misses amplitude loss. Exact-zero target residual is flagged low-information; a real repeat-based resolution threshold remains to be established.
- **Requested real demonstrations complete:** `src/detail_report.py` generates local `real_mri_detail.png` (whole-slice bilateral locations + 64-mm context + enlarged 32-mm grey patch + errors), `real_eye_detail.png`, `mri_all_metrics.png` / `eye_all_metrics.png` with CSVs, and `population_mean_four_modalities.png`. Final main cards, combined shrinkage montage and MRI metric table visually reviewed. Both cases are fixed first cases from the earlier atlas. Eye patch contrast is enlarged using one shared target-derived display window; measurements are unchanged. All nine official package outputs are included; the 2026 leaderboard ranks MSE/PSNR/SSIM. Additional synthetic Dice/topology/pupil metrics are exported separately. This is a 2D diagnostic on one real case per modality, not official 3D BraTS evaluation or population validation.
- **Current atlas checkpoint:** 12 additional synthetic eyes, 6 tissue conditions, 6 UTIRIS cases and 6 QTIM longitudinal cases generated. Last code validation: 98 tests passed (5 torchmetrics warnings). Real images and per-case mappings are local in `C:/Projects/data/projects/identity-ladder-metric/run-2026-11-01-visual-atlas/`; repo has aggregate JSONs and code. These atlases are intermediate, not the final paired demonstration requested by Kate. Native MRI gallery exists; its visual review and higher-resolution matching remain unfinished.

Goal: the "average beats real tissue" direction (H19 / ISBI Prop. 5). Latest evidence: `results/population_null_report.md`; earlier sensitivity evidence: `results/label_robustness_report.md`.
- Latest requested visual test: synthetic pupil/iris, `src/pupil_ladder.py`, `results/pupil_ladder.json`, `results/pupil_report.md`. Three seeds x 32 target people (96 targets), independent template/reference pools. Dilation +30% retains eye identity parameters but scores RMSE c=0.004 / SSIM c=0.006 against normal-pupil strangers. Dilation-matched references raise these to 0.449 / 0.432, not the repeat level. The independent mean scores RMSE 0.794 / SSIM 0.842 / pupil Dice 0.649. Component counts tie for all images (c=0.5); no biometric claim.
- Pupil visuals: `results/figures/pupil_fixed_example.{png,pdf}` (seed 201, first target; intensity, segmentation, contour overlay), `pupil_all_metrics.{png,pdf}` (all 9 intensity + 4 pupil metrics). Source/versions and 3-seed min/max are saved; ranges are not confidence intervals.
- Corrected omission: `src/label_examples.py` now plots other-day AND other-side donors. `label_fixed_example_folding.{png,pdf}` is a compact 7-column view; `label_hemisphere_locations.{png,pdf}` shows corresponding positions in a bilateral schematic and the aligned crops. The other-side sample was already scored before; it is NOT a mirrored copy of the target. The outline is illustrative, no real MRI or registration.
- Latest checks: 95 tests passed, including opposite-side array identity, pupil geometry and half-credit for ties; representative figures visually inspected. Existing experiment JSONs and paused paper sources unchanged. See `git log -1` for this visual-test checkpoint.
- Combined export complete: `results/all_ladder_metrics.csv`, 1,116 rows from both original ladders, all 9 intensity metrics plus label metrics. Missing label rho is explicitly unavailable. Original 56-row CSV is preserved.
- New figures in `results/figures/`: all 9 toy metrics for BASE and LOW_H; all 15 robustness metrics; paired MZ and DZ comparisons; fixed synthetic image/segmentation example. PNG and PDF, generated by `src/ladder_report.py` and `src/label_examples.py`. Paused paper sources and paper figures remain unchanged; the synthetic example illustration is now corrected as described above.
- H19 sensitivity complete: `src/label_robustness.py`, `results/label_robustness.json`. Five fresh seeds x two variance settings x prior scales 0, 0.25, 1; 60 families / 120 targets per seed, 16 strangers per target. 30 runs / 3,600 target-condition evaluations; shared draws across conditions, 300 independent simulated families per condition. Results include versions/source hashes, seed-level variation and paired twin contrasts.
- Result: population mean RMSE c remains about 0.995-0.997. Dice advantage over MZ is NOT universal: LOW_H at prior 1 has delta-c +0.215 [seed range +0.150,+0.284]; at prior 0.25 +0.090 [-0.020,+0.160]; at prior 0 -0.0002 [-0.208,+0.186]. BASE favours MZ on Dice in every tested seed/condition. Seed ranges are NOT confidence intervals.
- Component-count scores of the mean are roughly stranger-like; perimeter is inconsistent against MZ. These descriptors are not validated identity measures. Holes are rare; topology includes mask-border cuts.
- Prior sensitivity checkpoint checks: 91 tests passed (5 torchmetrics tensor-copy warnings); historical BASE / seed-2 result reproduced exactly; source hashes, complete run grid, score bounds, twin subsets and combined CSV validated. Manual pre-commit review: no secrets, participant data or unsafe calls found in the new changes (synthetic data only).
- Prior checkpoints: `5eebee0` (initial experiments), `f815166` (sensitivity experiment). See `git log -1` for the visual-test checkpoint. No push requested.

Next actions:
0. Latest requested experimental delivery is complete as recorded above. Further work is proposed, not running: test residual diagnostics against geometry-preserving nuisance changes (pupil dilation/registration shifts), establish a repeat-based low-information threshold, and extend the high-resolution MRI display to the other five fixed atlas cases if needed. Do not treat the known shrinkage path as evidence that the final identity metric is already solved.
1. Pupil follow-up: compare iris texture in dilation-normalized coordinates; no identity-based tuning. Proposed, not run.
2. Boundary-control experiment: compare whole-mask versus inner-region evaluation, and segmentation with/without original target context. This is proposed, NOT run. It tests whether the current assembly/smoothing helps the mean at the boundary.
3. Real BraTS MRI: Kate registers on Synapse and accepts terms herself; then data/model access and donor-swap/FastSurfer protocol. No new MRI data were acquired in this continuation.
4. Keep the ISBI draft paused. Text/bibliography/main.pdf are not validated against corrected or new results. Decisions on submission, Tongji cleanup and ND FACE remain open.
Use project-continuity for resumption, research-experiment for analyses, pre-commit-security before commits.

## Historical status before the continuation checkpoint (1 Nov 2026)
- Plan: hypotheses H7-H18 and baseline exam T1-T7 in Research_Plan section 9; BraTS code facts settled (SSIM per 2D slice, plain PSNR clamps; `src/brats_panel_checks.py`, `tests/test_brats_panel_checks.py`).
- Baseline v0 fixed after review (`src/baseline_v0.py`, 81 tests pass): H12 supported only for an affine template; GRAPE camera change on the stated anchor costs more than a change of person (c-scale T3b 1.55).
- Theory + synthetic ladder: `src/toy_ladder.py`, `results/toy_ladder.json` (closed form P(F>rho) matches; tissue redesigned: no clipping, rho-hat vs predictions).
- ISBI 2027 draft (deadline 2 Nov 23:59 ET) in `paper/isbi2027/` — PAUSED by Kate. Verification findings to apply: `scratchpad isbi_verification.json` (circularity wording, ethics/COI/funding, refs, layout). Bib update agent was stopped mid-way: check `refs.bib` before use.
- Data: batch 1 (8 open 2D sets) downloaded and registered; Tongji 26 GB incl. 4.7 GB duplicate awaiting Kate's cleanup decision; ND FACE features failed (retry). Dataset tree artifact: https://claude.ai/artifact/5MkpXbuvPzYoBUWNxv81LA. Ladder audit from files was stopped (rerun if needed).
- New ideas (Kate, not started): compute the ladder not on raw intensities but on masks and topological objects (segmentations, persistence), or on 2D/3D meshes (shape distances); choose the math for these; or start directly from DINO latent embeddings.
- Kate picks 1-2 directions for 2 Nov. Candidate she flagged as most important: the "average beats real tissue" result (H19 in plan section 9.2). To do: (1) clearer simulations, regenerate Fig. 1 from the corrected run (the v0 figure with c<=0.02 for the SD-matched mean and the low-contrast SSIM collapse was a clipping/mean-shift artefact); (2) H19 simulation in label space with topology descriptors; (3) all BraTS metrics, not only RMSE/SSIM; (4) example masks/fills of challenge winners via the BraTS orchestrator (needs BraTS data via Synapse registration = Kate's action, plus model weights).

The repository holds the paper on a calibrated identity metric (IdLadder scale). It was renamed from
`identity-ladder-repro` on 2026-10-31 (GitHub `kondratevakate/identity-ladder-metric`, private; private run data in
`C:\Projects\data\projects\identity-ladder-metric\`).

## State of the repository
- Committed: GRAPE reproduction and audit, plans in `docs/`, SIMON E0 (`a6105d8`), rename (`aafea62`).
- **Committed in `b8fb197` (verified from git history):** `src/grape_calibration.py`, `src/iris_calibration.py`,
  `src/iris_codes.py`, `src/image_ladder.py`, their tests, `results/{grape_calibration,iris_calibration,iris_codes,
  image_ladder,audit_2026-11-01}.json`, this file.
- Historical tests at that checkpoint: 51 passed; current verification is recorded above.
- Every result below was re-derived by an independent audit (`results/audit_2026-11-01.json`) and the code was fixed
  where the audit found bugs; numbers are from the fixed code.

## The calibrated score
For a pair distance d: c(d) = P(d_stranger > d), strangers drawn from unrelated-person pairs that share the pair's
nuisance condition (camera or sensor pair, eye relation). 0.5 = a typical stranger, 1 = closer than every stranger.
Mean c over a rung = AUC of that rung against strangers. Same idea as TM-score's random-pair calibration.
Design rules learned tonight:
1. The stranger reference must be matched on the exact nuisance condition (camera pair, sensor pair, protocol);
   a pooled "different camera" reference distorted GRAPE scores by up to 0.3.
2. Always report a nuisance-only negative control (here: mean and SD of grey level). In UPOL it reproduces the
   pixel scores, so pixel rungs there measure capture conditions, not the eye.
3. Within-session (same sitting) pairs share capture nuisance that no stranger reference can remove; report them as
   their own rung.

## Results (pilots, small n; see caveats in the JSON files)
**GRAPE fundus** (`results/grape_calibration.json`; camera-pair reference, visit = acquisition date)
| Rung | RETFound c | pixels c |
|---|---|---|
| same eye, other visit, same camera | 0.873 [0.832, 0.906] | 0.714 |
| same eye, other visit, other camera | 0.777 [0.735, 0.812] | 0.669 |
| fellow eye, same visit | 0.907 [0.883, 0.930] | 0.692; flipped 0.810 |
- Camera change costs RETFound about 0.10 (not 0.26 as with the pooled reference); camera is confounded with time
  (cross-camera pairs are about twice as far apart). No detectable camera effect in pixels.
- The fellow eye at the same visit is as close as the same eye at another visit: visit-level shared nuisance.

**Iris** (`results/iris_calibration.json`, `results/iris_codes.json`; UPOL and UTIRIS downloaded 2026-10-31)
| Rung | exposure-only control | pixels (z-scored) | iris code (UTIRIS NIR) |
|---|---|---|---|
| same eye, same sitting, NIR | 0.79 | 0.79 | 0.91 [0.89, 0.94] |
| fellow eye, NIR (flipped) | 0.76 | 0.69 | 0.68 [0.64, 0.72] |
| same eye, VIS vs NIR (same sitting) | — | 0.56 (raw pixels) | 0.57 (secondary, 70% QC) |
- The classical iris code (integro-differential segmentation, log-Gabor, Hamming) separates the same eye well
  (median HD 0.18 vs strangers 0.41, d' 2.2) but has about 35 degrees of freedom (Daugman: about 249), so it cannot
  test Daugman's fellow-eye result; flip effects in inner and outer radial bands cancel.
- UTIRIS VIS and NIR were taken in one sitting (constant 11.2 h clock offset): cross-spectral L0, not L1 over days.
- Visible-light segmentation passes QC for a minority of images (UPOL 22%, UTIRIS VIS 47%): secondary only.

**MRI image-space ladder** (`results/image_ladder.json`, run-2026-11-01-image-ladder-v2)
Skull-stripped brain-to-brain registration to MNI2009a sym (the first run registered whole heads to the brain-only
template and is flagged INVALID). 218 of 232 scans pass QC (all SIMON CUBE/DIR/Core-3D T1 scans fail).
| Rung (SIMON unless stated) | intensity median | geometry median |
|---|---|---|
| L0 same session, same protocol | 15.6 | 7.4 |
| same scanner, same protocol, other session | 18.4 | 8.9 |
| other scanner, same protocol | 20.3 | 10.2 |
| same scanner, other protocol | 30.4 | 16.0 |
| other scanner, other protocol | 31.6 | 18.4 |
| IXI strangers, same / other centre | 34.4 / 43.2 | 16.4 / 20.7 |
| QTIM same person 3.5 y later / co-twin / unrelated | 23.3 / 30.9 / 32.5 | 13.9 / 16.7 / 19.8 |
- Protocol (sequence) change is the largest nuisance, larger than scanner change; in geometry it is nearly as large
  as a change of person. Protocol is its own rung and must be matched or modelled.
- QTIM: same person vs unrelated AUC 0.997 (intensity), 0.93 (geometry); rank-1 re-identification 10/11.
- Cross-scan mirror half carries identity (QTIM AUC 0.76) but less than the same half (0.93-0.99).
- QTIM family pairs are 2 families of same-sex co-twins with unknown zygosity: descriptive only.

## Datasets
Downloaded and registered 2026-10-31 (terms in the agent memory file, not here): `upol-iris`, `utiris`.
Verified leads (2026-11-01, nothing downloaded):
- **Open and official:** UBIPr and AFHIRIS (iris/periocular); MUCT, KinFaceW, Family101, CALFW, LAG, CFP-FP, IMDB-WIKI
  (faces; most are web-scraped celebrity images — consent caveat). CACD: one audit says official and open, another
  says the official images are gone — unresolved. OLIVES (fundus, L2) remains the best open L2 source.
- **CASIA:** the claim that hycasia.github.io / nlpr-sir.github.io host CASIA-IrisV4 (incl. Twins) is false; their
  download buttons point to the dead host biometrics.idealtest.org. Only route: the BIT application / e-mail to CASIA.
- **Kaggle:** no twin face/iris/periocular set found; CASIA re-uploads and the WebFace 112x112 copy are unofficial
  (do not use); FIW (relatives) is official via a Kaggle click-through; FLUID (UWF fundus, longitudinal, two devices)
  by request. The Kaggle verification agent stalled, so Kaggle verdicts are single-pass.
- **Twins with MZ/DZ in 2D:** none obtainable without an institution (ND-TWINS, 3D-TEC, TwinsUK) or an application
  (CASIA-Iris-Twins, zygosity not stated).

## Next steps (decisions for Kate)
1. Overnight work is already committed in `b8fb197`; see current checkpoint above.
2. Approve downloads: UBIPr, AFHIRIS, MUCT, KinFaceW/Family101 (kinship faces), OLIVES (~34 GB).
3. MRI registration quality is modest (template correlation 0.64-0.68). Options: a whole-head template (official
   ICBM152 2009 package, about 60 MB) or a skull-stripping model (SynthStrip); both are downloads.
4. Send the CASIA application / e-mail (Iris Subject Ageing, Iris-Twins); your account and captcha.
5. Metric design: protocol as an explicit rung; nuisance-only control in every report; matched references.
6. Open from before: metric name (nDDT / pnDDT?), MIDL scope, three-layer scheme (discuss first).

## H19 continuation protocol (1 Nov 2026; recorded before new runs)
Exploratory sensitivity analysis, not an independent confirmatory validation of the earlier selected model.
Question: does the population-mean fill retain its intensity / Dice advantage when the common anatomical prior weakens, and are topology/geometry scores less favourable?
Design: five fresh seeds (101-105), 60 simulated families per seed (30 MZ, 30 DZ), both twins as targets (120), 16 stranger donors per target. Two variance settings (BASE, LOW_H), prior scales 0, 0.25, 1. Reset the same seed for each setting for paired random fields and donor choices. Total 30 runs, 3,600 target-condition evaluations; only 300 distinct simulated families per condition, reused across conditions.
Outcomes: all nine BraTS intensity metrics; Dice, accuracy; oracle-relative errors in components, holes, Euler number and perimeter. Report mean calibrated c and min/max of seed-level means, not confidence intervals. Compare population mean with each twin type on that same target subset (paired delta-c); the old all-target population mean versus MZ-only average was not a paired comparison.
Controls: same-sitting repeat, other day, other side, MZ/DZ donor, stranger anchor (c=0.5 by construction), blurred mean and blurred stranger. Population mean excludes the whole target family. No parameter selection based on the new scores.
A negative population-minus-MZ delta across seeds would limit the claim that the mean beats a co-twin. If topology remains as favourable as intensity, it does not supply the proposed discriminator. Results describe this simulation only; no hypothesis-test threshold or real-MRI claim is planned.
Feasibility: 8 targets / 8 donors took 0.78 s with one torch thread; seed 900 is excluded from analysis.
Implementation/output: `src/label_robustness.py`, `results/label_robustness.json` (resumable, versions and source hashes). Existing JSON results and paused paper text remain preserved. Also export both existing ladders to a reproducible combined table and plot all nine metrics outside the paused paper.

## Visual pupil test and omitted hemisphere donor (protocol, 1 Nov 2026)
Kate requests a pupil-like visual test and the missing opposite-hemisphere position in the tissue illustration.
Plan before results: a synthetic eye with a dark pupil, radial iris texture and a fixed outer iris. Compare same-eye repeat, same eye with 30% pupil dilation (identity parameters unchanged), fellow eye, stranger, independent-population mean, and blurred stranger. Parameters are illustrative, not calibrated to human anatomy. Main question: do appearance scores reward the mean and/or penalize same-eye dilation? Do pupil shape metrics add discrimination, and do component counts collapse to ties?
Three fresh seeds (201-203), 32 target people with one target eye each per seed, 40 independent template people, 32 independent reference people; 16 reference donors per target. Template and reference pools are disjoint from targets and each other. Also use dilated stranger references for the dilated-eye comparison, keeping the same donor identities. All nine BraTS metrics plus pupil Dice, area error, symmetric boundary distance and component-count error. Descriptive cohort means and their three-seed ranges only, no confidence intervals or biological claims. Fixed visual example: seed 201, target index 0, before looking at scores.
The old brain-like picture contains synthetic random fields, not MRI. Add the already-scored other-side donor and the other-day donor; show a schematic full bilateral arrangement and homologous patch locations. Coordinates are defined by the simulator, not anatomical registration. Keep previous experiment numeric results unchanged.

## Expanded visual atlas protocol (1 Nov 2026, before case selection/results)
Requested: more synthetic cases and real cases. Build 12 synthetic eye examples (seeds 301-303, first four targets each) with controlled dilation, intensity change, translation, blur and occlusion, plus 6 synthetic tissue examples (first two targets at prior 0, 0.25, 1). Use fixed indices rather than outcome-based selection; save all 9 intensity metrics and pupil metrics where meaningful.
Real eyes: first six distinct eligible UTIRIS NIR people in stable source order; require two same-eye images and one opposite-eye image passing existing numerical pupil/limbus QC. Exclude folder/filename side conflicts using existing loader. Report selection counts and all exclusions, visually review contours without silently replacing unfavourable cases. Use a disjoint fixed template/reference subset of other eligible people. Photographs, per-case scores and source mappings stay only in a private local atlas folder; repository receives aggregate evidence and code. Pupil circles are estimates, not ground truth. Same-eye images are repeat captures, not longitudinal visits.
Real MRI: first six QTIM people with two QC-passing visits in the existing registered 6-mm cache. Reuse fixed template-space slice/ROI and opposite-side mirror. Compare later visit, contralateral side, unrelated same-cohort donor, population mean and blurred donor; target family excluded from donors/templates. Registered scans are smoothed/downsampled derivatives, not native-resolution images or a segmentation benchmark. QTIM dataset_description.json identifies OpenNeuro ds004169 v1.0.8, CC0. Do not call the second visit a same-session repeat or assume MZ status.
Interpretation: descriptive illustrations and small-sample paired scores only; not representative prevalence estimates, no new confidence intervals. BraTS-specific real inpainting validation remains separate.

## Detail and population-null experiment (1 Nov 2026, before new scores)
- Real-MRI case: first case of the existing atlas, unchanged person selection. Reuse its later visit and the existing disjoint reference/template pools; exclude target families. Affine registration to an independent symmetric template preserves individual folding differences. Broad bilateral candidate ROI excluded from registration similarity and intensity fitting. N4 correction uses each source image; this is an image-comparison diagnostic, not a claim of a masked-input reconstruction algorithm. Inspect target anatomy to fix the final slice/patch; record coordinates before scoring alternatives. Real-eye case: first existing UTIRIS case and its recorded disjoint pools.
- Controlled path: y = mu + lambda * (x-mu), lambda in {1, .75, .5, .25, 0}, using the same hidden region and known x as an oracle diagnostic. Synthetic tissue: seeds 401-403, eight targets from distinct families per seed, separate template/reference families, weak prior .25. Synthetic eyes: seeds 401-403, eight target people each, 40 template and 24 reference people. Shared draws within each path; 24 independent synthetic targets per modality, not 120 people. Real paths are one illustration per modality, not population evidence.
- Compare all nine image metrics from the official package, their within-case ranks and stranger calibration. Add three Euclidean/RMSE distances in one fixed representation, residual magnitude ratio, target-residual projection and residual cosine; these are diagnostics, not an established combined score. Mean and unrelated/noisy texture controls are required. No claims of identity from an atypical/noisy residual. Exact-oracle PSNR infinity is recorded explicitly rather than silently truncated.
- Metric scope verified 1 Nov 2026: the official package README lists nine outputs (https://github.com/BraTS-inpainting/inpainting#computed-metrics). The 2026 challenge evaluation and infrastructure name SSIM, PSNR and MSE for ranking (https://github.com/BraTS-inpainting/2026_challenge/blob/main/evaluation/README.md ; https://github.com/Sage-Bionetworks-Challenges/brats-infra). Dice/topology are additional research diagnostics, not additional official inpainting leaderboard metrics. Slice illustrations use a 2D adaptation, not a submitted 3D BraTS evaluation. No real segmentation ground truth is available for these cases.
