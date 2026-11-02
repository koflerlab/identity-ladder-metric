# Review: CARE reproduction (R1), state of 2026-11-02 11:10

Reviewer: Claude (separate session from the one running the reproduction). Scope: `docs/PROTOCOL_care_reproduction.md`,
`docs/PROTOCOL_care_reproduction_amendment1.md`, `src/care_reproduction.py`, `tests/test_care_reproduction.py` and
the `results/care_reproduction*` files as they stood at 11:10. Nothing in those files was changed by this review.

## Verdict

Worth doing, and worth finishing. R1 is cheap, the protocol discipline is good, and it has already produced a real
reproducibility finding. Its value for H19 is indirect: R1 can show whether CARE's published anatomy scores can be
trusted as a comparator, not whether segmentation-based scores reject a population mean. That needs R2 plus the H19
extension.

## Disclosure: evidence this review has seen

To locate the 0/108 mismatch, the reviewer recomputed Table 2 from the pinned public CSVs (same commit
`2443d490`, same split file, file hashes equal to `results/care_reproduction_inputs.json`) with the all-phase
selection of Amendment 1. These are aggregates on the 36-case selection, so the 36-case results are now seen
evidence. Any further amendment is post hoc and must say so.

| Endpoint family | 36-case triplets within 0.05 of Table 2 |
|---|---|
| Pixel (SSIM, PSNR), 9 methods x baseline/CARE | 36 of 36 |
| Anatomy (NSD large, NSD small, clDice intestine, clDice vessels) | 0 of 72 |

Example, NAF baseline: SSIM 92.5 (91.1, 94.0) and PSNR 29.4 (28.9, 30.8), identical to the paper. NSD large 85.5
against 83.7 printed; clDice intestine 29.6 against 71.6 printed. Positional slicing, as in the official
`step6_read_result_csv_for_table1st2nd.py`, gives the same numbers as the column-name grouping used here, so the
grouping is not the cause.

## Findings

1. **The paper's selection is all 36 test cases, not the V-only 13.** The pixel columns match exactly on 36 cases.
   The released `step6` rule (`x[:-2] if x[-1]=="V"`, reading a missing `dataset_splits/` path) does not produce
   Table 2. The 13-case run is a faithful test of the released script and should stay reported as such.
2. **The released segmentation scores do not reproduce the anatomy columns.** With the right selection and the
   official grouping, none of the 72 anatomy triplets match, some by a factor of two (intestine clDice). These are
   exactly the columns relevant to H19. Most likely the CSVs in `resultsCSVcldice/` at this commit are not the ones
   behind Table 2 (other run, other label map or other tolerance). This is a hypothesis, not checked.
3. **Amendment 1's coverage rule blocked the most informative test.** One missing Lineformer segmentation row
   stopped aggregation for the whole 36-case selection, including pixel endpoints whose files are complete for all
   9 methods. Coverage should be judged per method and per file type: aggregate what is complete, mark the rest
   blocked. This is a design change and needs a dated amendment.
4. **The paper's significance test ignores pairing.** Caption and `step6` use a two-sided Mann-Whitney U test on
   baseline vs CARE scores of the same scans. A paired test (Wilcoxon signed-rank) is the matching one. The
   reproduction already reports paired improved/tied/worsened counts without p-values; keep it that way, and mention
   the point only as a note on the paper, not as a result.
5. **Code.** Hash guards, refusal to overwrite manifests, no case keys in outputs and no zero imputation are all as
   the protocol requires. Small points:
   - `execute()` looks up each fetched name in the frozen manifest before checking the inventory, so an unexpected
     source raises `KeyError` instead of the intended message. Check `set(data) == set(frozen["inputs"])` first.
   - `paper_table()` depends on page index 8 and a regex; the count check of 108 guards it, but there is no test.
   - No test covers the hash guard in `execute()`. A test with a tampered byte would catch a silent regression.
   - The official `step6` has its own bug (`ssim_scores[~np.isnan(psnr_scores)]` after `psnr_scores` was already
     filtered). It is harmless here because no pixel score is NaN; worth one line in the report.
6. **Licence.** The CARE repository says CC BY-NC-ND. Only aggregates and hashes are stored and no upstream code is
   copied, which fits. Do not commit the downloaded CSVs or a modified copy of their scripts.

## Recommended next steps

1. Amendment 2 (post hoc, dated, states that the reviewer saw the 36-case aggregates): per-method coverage; report
   pixel endpoints on 36 cases as reproduced; report anatomy endpoints as not reproducible from the released scores.
   Do not search other score folders to force a match.
2. Ask the authors which score files and label map produced Table 2 (a GitHub issue on MrGiovanni/CARE). This is a
   public post, so it needs Kate's yes and her wording.
3. Then decide on R2: check that the segmentator weights and the 36 test CTs are actually downloadable and what the
   compute cost is on 8 CPU without GPU. R2 is what makes the H19 test possible: feed a population-mean fill, a
   shrinkage path and a wrong-detail fill through their segmentator and score NSD and clDice next to Dice.
4. For the ISBI paper due today: CARE is the clearest published example of "evaluate through downstream
   segmentation". It belongs in related work as the fix the paper questions. The finding above is not ready to cite
   as a result.
