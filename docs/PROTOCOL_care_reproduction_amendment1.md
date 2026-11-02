# CARE R1 amendment 1: diagnostics after first failed execution

2026-11-02, written before inspecting the 36-case score aggregates.

The first frozen R1 run failed, as specified, at baseline InTomo small-organ NSD: none of the 13 selected cases had a finite category score. No result file was produced. A diagnostic count pass then inspected all endpoint denominators, without reporting metric values: the official split contains 36 unique stripped case keys, 23 arterial (A) and 13 venous (V). The released table script's V-only rule therefore selects 13, whereas the paper's Table 2 caption says 36 CT scans. Endpoint finite counts vary and some clDice endpoints have fewer than 13 cases. These observations are already seen evidence, not prespecified findings.

Retain the initial protocol and source manifest unchanged. Extend error handling to save failed endpoints and denominator counts and continue diagnostics for all fixed methods. A failed endpoint remains a failure; do not replace missing values with zero or label an incomplete table reproduced.

Add a separate, explicitly post hoc sensitivity: include all 36 A/V cases from that same official split, stripping the same two-character phase suffix, with no other filtering, imputation, tuning or input-folder change. This sensitivity follows the paper's stated case count; it does not overwrite the original 13-case reproduction of the published script rule. Report table agreement and missing counts separately for each selection. Freeze this amendment's hash and the 36-case rule in a separate amendment manifest before execution. If either analysis still differs, report that rather than searching alternative score folders for a match.

No raw CT, model or new metric experiment is authorized by this amendment beyond the existing user scope. The independent H19 mean-fill extension remains unexecuted. A denominator or publication/code mismatch is a reproducibility issue, not evidence that CARE fails to preserve identity.
