# Reproduction workflow and visible metric ledger

Adopted 2026-11-02 at Kate's request, using the ICML-repro experience. No dedicated reproduction SKILL.md was found in that repository's `.claude` directory. The available `research-experiment`, `project-continuity`, `token-hygiene` and personal `pre-commit-security` skills cover the work. Notebook-discipline was inspected but is not applied to non-notebook scripts.

Concrete reference artifacts read, not copied as scientific evidence:
- `C:/Projects/02_academia/studies/icml-repro/reproductions/caml/CLAIM_DECOMPOSITION.md`: split a paper into independently testable claims.
- `.../caml/FROZEN_FORECAST.md`: record expectations and feasibility before new outcomes; do not invent a retrospective forecast.
- `.../caml/INDEPENDENT_CHECKS.md`: separate deterministic implementation checks, smoke tests and full benchmark reproduction; compare printed and released configurations.
- `.../caml/PUBLICATION_PREFLIGHT.md`: retain exact versions, isolated caches and explicit evidence boundaries. Publication is a separate user-authorized action.
- `C:/Projects/02_academia/studies/icml-repro/verdict_audit/PROTOCOL.md`: input hashes, explicit units and definitions, frozen analysis and dated amendments. A git/hash freeze is not external preregistration.

For each claim, the visible ledger must name the metric, direction, data/split, denominator, printed reference, current result, comparison tolerance, and status. A missing endpoint is blocked or unavailable, never zero and never a reason to stop an independent complete endpoint. Do not change the eligible cohort to manufacture completeness.

`python src/metrics_board.py` rebuilds `results/METRICS_STATUS.md` and its aggregate JSON/CSV from saved sources. Keep that same file open beside the chat. Rebuild before a new experiment, at completed milestones, and before the final answer; do not create a new tab/table per iteration. CARE iteration 2 calls it at start and finish. Other run scripts are not retroactively instrumented: explicitly run this builder around them.

All iteration artifacts remain immutable. The board identifies the current iteration and links historical results. Failed and superseded results stay inspectable. No participant-level data, credentials or raw CSVs enter the public repository. QTIM affine/incorrect-scale results must not be promoted into the current table while the SyN/P90 rerun is unvalidated.

The BraTS table and the 2D representation ladder have different preprocessing/regions and stay in separate panels. Within the BraTS summary, the population-mean row pools same-eye rungs; do not call its contrast against a narrower rung a paired comparison. `c` is a calibration of each metric, not an additional independent distance. A high score is not proof of identity, especially for the mean.
