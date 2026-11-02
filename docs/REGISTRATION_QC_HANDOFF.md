# Registration and visual QC first

Kate's latest decision, 2026-11-02, supersedes the metric-experiment queue: registration is very poor and mirrored versions of the same person often do not overlay the crop. Do not proceed to other metrics until registration is solved well and visual QC passes at least 90% of the predefined comparisons, including mirrored comparisons. No such passing visual-QC result has been established in this chat.

## Authorized work now

Registration, crop/anatomical correspondence and visual QC, across GRAPE fundus, UPOL/UTIRIS iris and the separate QTIM MRI pipeline. Inspect existing implementations and jobs before starting anything. Do not duplicate the QTIM SyN pilot or another session's 2D registration work. Respect shared-data instructions and storage locks. Keep individual images, IDs, overlays and QC rows private; public repository receives code, protocol, aggregate QC counts and hashes only.

Freeze a QC review set and denominator before assessing pass rate. Display full-field context, the exact crop in both images, landmark/contour overlays and checkerboard/blend views. Inspect ordinary and mirrored comparisons separately. Check physical reflection/canonical coordinates, orientation, transforms, interpolation, field coverage and whether the crop contains homologous anatomy. A good global correlation does not demonstrate correct local crop correspondence. Do not deform away individual anatomy merely to improve apparent alignment.

Specify pass/fail criteria and report reviewed/pass/fail/unreviewable counts overall and by dataset and comparison type. Failed/unreviewable intended comparisons cannot silently disappear or be replaced by easier ones. Report mirrored-case performance explicitly so an overall number cannot conceal a failed mirror branch. Automated checks alone are not visual acceptance. Mark the gate not met until there is an inspectable visual-QC record supporting the threshold; keep Kate's visual acceptance distinct from automatic flags.

Current `docs/PROTOCOL_2d_ladder.md` Amendment 2 was added by another session (commit `a849d98`) and includes iris canonicalization and fundus ORB/RANSAC. Its fallback clause says a failed fundus registration may be used unregistered. This conflicts with the newer QC gate: do not use such failures for metric scoring. Preserve the frozen history and record a new dated amendment rather than silently rewriting it. Latest user instruction takes precedence.

## Read in order

1. `AGENTS.md`, this file, `docs/HANDOFF.md` (shared ownership/current jobs).
2. `docs/PROTOCOL_2d_ladder.md`, especially Amendments 1 and 2; inspect `src/ladder2d_register.py` if present. At the handoff this was another session's untracked work: do not overwrite it or treat it as validated.
3. `docs/PROTOCOL_real_cases.md` Amendment 4 and `src/real_cases_syn.py`; check the existing job/output state before any new MRI preparation. Historical affine/P90 issues are documented in HANDOFF.
4. `src/ladder2d_figures.py` and `src/ladder2d_brats.py` for crop and mirror definitions; do not rerun their metric loops.
5. `results/METRICS_STATUS.md`: keep it visible, but label the old numbers provisional/unregistered or unvalidated. Regenerate with `python src/metrics_board.py` after meaningful QC milestones; it reads aggregates only.

## Preserved state from this chat

- `bccf8ca`: project metric-visibility instructions, ICML-repro workflow, board generator, frozen CARE amendment 2 and endpoint-specific runner. Full suite at that checkpoint: 162 passed, 10 existing torchmetrics warnings.
- CARE iteration 2 completed BEFORE the pause: `results/care_iteration2.json`, report and CSV. SSIM 18/18 and PSNR 18/18 published median/IQR triplets match. Anatomy: 63 numerical mismatches, 8 blocked coverage endpoints, 1 unavailable endpoint. This reuses author scores, not images; image-level CARE/H19 was not run. Preserve it as historical work and start no more CARE/other metric experiments now.
- Latest CARE outputs and board display updates were not yet committed when Kate requested the pause. Preserve and checkpoint them; do not rerun to regenerate identical evidence. Original R1 and amendment-1 outputs are intact.
- The screenshot table is incomplete: it shows c for MSE/SSIM only and QTIM placeholders. Full nine-output/raw/interval/count summaries are in `results/METRICS_STATUS.*`. The unregistered values must not be treated as accepted evidence. The population-mean row pools same-eye rungs, so its GRAPE comparison with one narrower rung is not paired.

First deliverable in the new chat: verified inventory of registration implementations/running jobs, explicit fixed visual-QC criteria/review set, and a small inspectable registration/crop QC pilot including a mirror failure. Continue registration and QC; resume metric experiments only after the required gate, not merely after a script runs successfully.
