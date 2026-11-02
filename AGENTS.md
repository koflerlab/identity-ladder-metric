# Identity Ladder: metric visibility and reproduction

- Follow `docs/HANDOFF.md` and `docs/REPRODUCTION_WORKFLOW.md`; preserve concurrent ownership and frozen protocols.
- Kate wants current metrics and results visible throughout every iteration. Run `python src/metrics_board.py` before and after an experiment and before the final reply. Keep `results/METRICS_STATUS.md` open in the existing side panel; update it in place.
- Every reported metric needs a named definition/direction, data/split, denominator, raw-versus-calibrated scale, result and evidence status. Mark unavailable, invalid, blocked and not run distinctly. Do not fill gaps with zeros.
- Reproduce independently testable endpoints even if another endpoint is blocked. Preserve eligibility per endpoint; never quietly reduce the cohort. Keep original reproduction separate from exploratory changes and H19 extensions.
- Preserve each iteration's artifacts and protocol/input/code hashes; the board is a generated latest view, not a replacement for history. No new global skills or public logbooks are implied.
- Do not modify another session's files or duplicate its running QTIM/SyN jobs. Commit explicit owned paths only; inspect the staged diff for secrets and personal/health data.
