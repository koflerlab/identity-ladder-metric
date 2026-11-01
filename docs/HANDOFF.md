# Handoff — identity-ladder-metric (updated 2026-11-01)

The repository holds the paper on a calibrated identity metric (IdLadder scale). It was renamed from
`identity-ladder-repro` on 2026-10-31 (GitHub `kondratevakate/identity-ladder-metric`, private; private run data in
`C:\Projects\data\projects\identity-ladder-metric\`).

## State of the repository
- Committed: GRAPE reproduction and audit, plans in `docs/`, SIMON E0 (`a6105d8`), rename (`aafea62`).
- **Not committed (2026-11-01 overnight work):** `src/grape_calibration.py`, `src/iris_calibration.py`,
  `src/iris_codes.py`, `src/image_ladder.py`, their tests, `results/{grape_calibration,iris_calibration,iris_codes,
  image_ladder,audit_2026-11-01}.json`, this file.
- Tests: `python -m pytest tests/ -q` → 51 passed.
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
1. Commit the overnight work (code, tests, results, this file)?
2. Approve downloads: UBIPr, AFHIRIS, MUCT, KinFaceW/Family101 (kinship faces), OLIVES (~34 GB).
3. MRI registration quality is modest (template correlation 0.64-0.68). Options: a whole-head template (official
   ICBM152 2009 package, about 60 MB) or a skull-stripping model (SynthStrip); both are downloads.
4. Send the CASIA application / e-mail (Iris Subject Ageing, Iris-Twins); your account and captcha.
5. Metric design: protocol as an explicit rung; nuisance-only control in every report; matched references.
6. Open from before: metric name (nDDT / pnDDT?), MIDL scope, three-layer scheme (discuss first).
