# Handoff — identity-ladder-metric (2026-10-24)

## State of the repository
- Commits: GRAPE fundus reproduction and audit; plans in `docs/`; `a6105d8` SIMON E0 (corrected).
- Tests: `python -m pytest tests/ -q` → 13 passed.
- Not committed: this file.

## SIMON E0 (`src/simon_e0.py`, `results/simon_e0.json`, `results/simon_e0_per_structure.csv`)
One healthy person, FastSurfer volumes, 69 sessions on 33 scanners (ses-029 excluded: whole-session failure,
median Dice ~0.1 against both neighbours; not a left/right swap, rho = +0.40).
Model: log(volume) ~ years + (1|scanner), exact profile REML; scanner-cluster bootstrap, 1000 draws.

| Quantity (median over 17 structures) | Value [95% CI] |
|---|---|
| Scanner SD, one hemisphere | 2.7% [1.7, 3.3] |
| Residual SD (repeat + state), one hemisphere | 2.7% [1.4, 3.4] |
| Total SD ratio, asymmetry index / hemisphere | 0.75 [0.63, 0.83]; lower in 14/17 structures |
| Shared (cancelling) nuisance fraction | 0.72 [0.66, 0.81] |
| Annual change | -0.04%/y [-0.18, +0.15] — aliased with scanner era, not an ageing rate |

- Mirror axis cancels most global nuisance. This **reverses** the first (lbfgs) run, which was wrong.
- Robust to: keeping ses-029 (0.77, 13/17), merging possible scanner aliases, manufacturer fixed effect.
- statsmodels default optimiser agrees to 8e-6; lbfgs stopped early in 24/51 fits (root cause of the old error).
- Asymmetry sign fully consistent: Amygdala, Caudal ACC, Hippocampus, Inferior Parietal, Superior Frontal, Thalamus.
- Caveat: same-scanner pairs <1 year (1.4% vs 2.9% different scanner) come 73% from one scanner.

## Datasets found (read-only searches, nothing downloaded)
Access terms and sync rules are kept in the agent memory file, not here.
- Fundus/OCT: AI-READI (up to 6 devices per eye at one visit → L1), OLIVES (visits every 4–16 weeks → L2, open),
  DeepDRiD (fellow-eye pairs, open), mBRSET, FIRE. No public fundus data with twins.
- Iris, Asia: CASIA portal now at www.idealtest.org (old host is dead). Iris Subject Ageing (3 sensors, 2009 vs 2013),
  CASIA-Iris-Twins (100 pairs, MZ/DZ not stated), Thousand (L4). Individual registration; captcha by the user.
- Iris, Latin America: only UFPR (Brazil) live; needs institutional signatory.
- CASIA portal also lists: CASIA 3D Face, CASIA Face Image, CASIA-Face-Africa, Fingerprint Subject Ageing.
  Descriptions **not read**: the site's TLS certificate has expired, so fetches fail.

## Next steps
1. Read the dataset papers for CASIA 3D Face / Face Image / Fingerprint Subject Ageing (Crossref/PubMed), or
   read the portal pages yourself after accepting the certificate warning.
2. Decide downloads: OLIVES (~34 GB) and DeepDRiD — register in `C:\Projects\data` and add `.stignore` first
   (biometric use).
3. Register at CASIA (user action) — can go together with the email to Florian.
4. Open design decisions: metric name (IdLadder scale; nDDT/pnDDT?), geometry vs intensity features,
   mirror half as an axis vs a rung (E0 supports a separate axis), MIDL scope incl. predicted version.
5. Not yet in the plan: IdLadder three-layer scheme and state layer (agreed to discuss first).
6. Optional: ResNet10t GRAPE row (needs timm + weights download).
