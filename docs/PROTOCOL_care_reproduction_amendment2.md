# CARE amendment 2: independent endpoint coverage and visible iteration 2

2026-11-02. Kate asked for another attempt and for each metric and result to remain visible throughout all iterations. Original R1 and amendment 1 outputs are preserved, not overwritten.

Exposure: `docs/REVIEW_care_reproduction.md`, written by another active session, has now been read. That reviewer reports that, on all 36 test cases, 36/36 pixel median/IQR triplets match Table 2 and 0/72 anatomy triplets match when available exported rows are used. These outcomes are already seen evidence, not a fresh blind forecast. The reviewer did not change this run's files. The new execution checks the same pinned official release independently; it is an exploratory correction of coverage logic, not a new confirmatory trial.

Claims/endpoints are independent: SSIM, PSNR, large-organ NSD, small-organ NSD, intestinal clDice, vessel clDice. All six are higher-is-better. Keep the fixed 9 methods and 2 treatment states; each endpoint has 18 published median/IQR triplets. A match means all three printed values agree within 0.050001, as originally frozen. Do not change source folder, label groups, units or tolerance.

Eligibility remains all 36 unique test-case keys in the fixed official split, all phases. Require complete coverage separately for each method, treatment state and input file type. Pixel endpoints need the pixel file, anatomy endpoints need the segmentation file. A missing segmentation case blocks only that method/state's four anatomy endpoints. It must not block SSIM/PSNR or another method. Do not aggregate the 35 available cases and compare them as if they were the full cohort. Do not impute zero for missing or undefined metric values.

Among completely covered endpoints, retain the authors' within-case category nanmean and across-case nan-aware median/IQR, report finite/missing denominators, and mark no-finite endpoints unavailable. Report both total triplets (108) and eligible/compared counts; blocked is distinct from numerical mismatch. Descriptive paired changes need matching keys and report finite-pair counts. No new significance tests.

Before run, freeze this document, iteration-2 code and the original input manifest into `results/care_iteration2_inputs.json`. On run reject modified inventories, bytes, source manifest, protocol or code. Hash checking precedes aggregation. Save an immutable `results/care_iteration2.json` plus report/CSV. Original results remain linked. A current-state record and `results/METRICS_STATUS.md` are updated before retrieval and after completion/failure, without patient identifiers.

Implementation checks: tampered input and extra/missing input inventory; 108-triplet PDF extraction; a missing anatomy case cannot block a complete pixel endpoint; incomplete cases cannot silently enter the 36-case analysis. These checks protect the actual failure modes, not a mirror of implementation syntax.

This remains score-aggregation reproduction only. It cannot establish anatomical preservation, identity or H19. Image-level CARE and population-mean/shrinkage/wrong-detail tests remain separate unexecuted claims. No public issue, author message or external publication is authorized by this amendment.
