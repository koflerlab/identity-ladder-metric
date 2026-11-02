"""CARE claim-level reproduction, with coverage independent for each endpoint."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import io
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pandas as pd

import care_reproduction as cr
import metrics_board

PROTOCOL = cr.ROOT / "docs/PROTOCOL_care_reproduction_amendment2.md"
FREEZE = cr.ROOT / "results/care_iteration2_inputs.json"
OUTPUT = cr.ROOT / "results/care_iteration2.json"
STATE = cr.ROOT / "results/care_iteration2_state.json"


def update_state(stage):
    STATE.write_text(json.dumps({"iteration": 2, "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": list(cr.METRICS)}, indent=2) + "\n", encoding="utf-8")
    metrics_board.build()
    print(stage, flush=True)


def endpoint_arrays(pixel, seg, eligible):
    """Do not shrink the cohort or let an incomplete modality block another."""
    values, issues = {}, {}
    for kind, frame in (("pixel", pixel), ("seg", seg)):
        names = ("ssim", "psnr") if kind == "pixel" else tuple(cr.GROUPS)
        try:
            selected = cr.select_frame(frame, eligible)
        except ValueError:
            for name in names:
                issues[name] = "blocked_coverage"
            continue
        if kind == "pixel":
            values.update({"ssim": selected["ssim_3d"].to_numpy(float), "psnr": selected["psnr_3d"].to_numpy(float)})
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for name, cols in cr.GROUPS.items():
                    values[name] = np.nanmean(selected[cols].to_numpy(float), axis=1)
    return values, issues


def check_freeze(data):
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    for name, path in (("protocol_sha256", PROTOCOL), ("code_sha256", Path(__file__)),
                       ("helper_sha256", Path(cr.__file__)), ("source_manifest_sha256", cr.FREEZE)):
        if frozen[name] != cr.digest(path.read_bytes()):
            raise ValueError(f"Iteration-2 frozen artifact changed: {name}")
    original = json.loads(cr.FREEZE.read_text(encoding="utf-8"))
    cr.validate_inputs(data, original)
    if original["protocol_sha256"] != cr.digest(cr.PROTOCOL.read_bytes()):
        raise ValueError("Original protocol changed")


def aggregate(data):
    check_freeze(data)
    eligible = cr.selected_ids(pd.read_csv(io.BytesIO(data[cr.SPLIT])), "paper_all_phases_posthoc")
    expected = cr.paper_table(data["paper.pdf"])
    rows, paired = [], []
    for mi, method in enumerate(cr.METHODS):
        by_state = []
        for ci, care in enumerate((False, True)):
            pixel, seg = [pd.read_csv(io.BytesIO(data[cr.csv_path(method, care, k)])) for k in ("pixel", "seg")]
            values, issues = endpoint_arrays(pixel, seg, eligible)
            by_state.append(values)
            for ki, metric in enumerate(cr.METRICS):
                reference = dict(zip(("median", "q1", "q3"), expected[mi, ci, ki].tolist()))
                row = {"method": method, "care": care, "metric": metric, "higher_is_better": True,
                       "n_eligible": len(eligible), "paper": reference, "status": issues.get(metric),
                       "median": None, "q1": None, "q3": None, "n_finite": None, "n_missing": None,
                       "max_abs_error": None}
                if metric in values:
                    try:
                        stats = cr.summarize(values[metric])
                        error = max(abs(stats[k] - reference[k]) for k in reference)
                        row.update(stats, max_abs_error=error, status="matched" if error <= .050001 else "mismatch")
                    except ValueError:
                        row.update(status="unavailable", n_finite=int(np.isfinite(values[metric]).sum()),
                                   n_missing=int(np.isnan(values[metric]).sum()))
                rows.append(row)
        for metric in cr.METRICS:
            if metric not in by_state[0] or metric not in by_state[1]:
                continue
            a, b = by_state[0][metric], by_state[1][metric]
            good = np.isfinite(a) & np.isfinite(b)
            delta = b[good] - a[good]
            paired.append({"method": method, "metric": metric, "n_pairs": int(good.sum()),
                           "median_change": float(np.median(delta)) if len(delta) else None,
                           "improved": int((delta > 0).sum()), "tied": int((delta == 0).sum()), "worsened": int((delta < 0).sum())})
    summaries = {}
    for metric in cr.METRICS:
        selected = [r for r in rows if r["metric"] == metric]
        summaries[metric] = {status: sum(r["status"] == status for r in selected)
                            for status in ("matched", "mismatch", "blocked_coverage", "unavailable")}
    return {"iteration": 2, "scope": "Released-score aggregation only; image-level CARE and H19 not run.",
            "selection": "all 36 official test cases, all phases; post hoc amendment 2",
            "n_eligible_cases": len(eligible), "n_triplets": len(rows), "rows": rows,
            "metric_summary": summaries, "paired_changes": paired,
            "coverage": cr.coverage_diagnostics(data, eligible),
            "exposure": "Reviewer 36-case aggregates were seen before this execution; see amendment 2.",
            "manifest": {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_commit": cr.SHA, "freeze_sha256": cr.digest(FREEZE.read_bytes()),
                "protocol_sha256": cr.digest(PROTOCOL.read_bytes()), "code_sha256": cr.digest(Path(__file__).read_bytes()),
                "source_manifest_sha256": cr.digest(cr.FREEZE.read_bytes()),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cr.ROOT, text=True).strip(),
                "versions": {p: version(p) for p in ("numpy", "pandas", "requests", "PyMuPDF")}}}


def save(out):
    OUTPUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat = [{**{k: v for k, v in row.items() if k != "paper"}, **{f"paper_{k}": v for k, v in row["paper"].items()}}
            for row in out["rows"]]
    pd.DataFrame(flat).to_csv(cr.ROOT / "results/care_iteration2_table.csv", index=False)
    lines = ["# CARE: iteration 2, current endpoint results", "", out["scope"], "", out["selection"], "",
             "| Metric | Match / 18 | Mismatch | Blocked coverage | Undefined |", "|---|---|---|---|---|"]
    for metric, counts in out["metric_summary"].items():
        lines.append(f"| {metric} | {counts['matched']} | {counts['mismatch']} | {counts['blocked_coverage']} | {counts['unavailable']} |")
    lines += ["", "## Every original and current value", "", "Values are median [Q1, Q3]. Pixel SSIM and anatomy scores are percentages; PSNR is dB. All are higher-is-better.", "",
              "| Method | CARE | Metric | Paper | Current | Finite / eligible | Status |", "|---|---|---|---|---|---|---|"]
    def fmt(v):
        return "not available" if v["median"] is None else f"{v['median']:.1f} [{v['q1']:.1f}, {v['q3']:.1f}]"
    for row in out["rows"]:
        lines.append(f"| {row['method']} | {row['care']} | {row['metric']} | {fmt(row['paper'])} | {fmt(row)} | {row['n_finite']} / {row['n_eligible']} | {row['status']} |")
    lines += ["", "- Coverage is assessed per file type and method. Lineformer anatomy needs 36 cases; its 35-case exports are not silently substituted.",
              "- Finite denominators may vary under the authors' nan-aware category averaging. No missing or undefined score is replaced by zero.",
              "- The V-only 13-case result remains a reproduction attempt of the released script, not of the paper's full test selection.",
              "- Anatomical mismatches do not establish the cause, nor refute CARE or establish H19. Released CSV provenance, label maps, tolerance and missing-value semantics need investigation.",
              "- The official table script's SSIM/PSNR NaN-filter issue is inactive here when all pixel scores are finite. Baseline/CARE comparisons are paired descriptively; no unpaired p-values are added.",
              "- Source: Lin et al., NeurIPS 2025, https://github.com/MrGiovanni/CARE at pinned commit in the manifest. No upstream CSV or code is redistributed.",
              "- Original R1 and amendment-1 reports are preserved. Reviewer aggregates were seen before this run; this is a disclosed post hoc correction."]
    (cr.ROOT / "results/care_iteration2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        if FREEZE.exists():
            raise SystemExit("Iteration-2 freeze already exists; do not overwrite")
        FREEZE.write_text(json.dumps({"protocol_sha256": cr.digest(PROTOCOL.read_bytes()),
            "code_sha256": cr.digest(Path(__file__).read_bytes()), "helper_sha256": cr.digest(Path(cr.__file__).read_bytes()),
            "source_manifest_sha256": cr.digest(cr.FREEZE.read_bytes())}, indent=2) + "\n", encoding="utf-8")
        update_state("протокол итерации 2 зафиксирован; шесть метрик ожидают пересчёта")
        return
    if OUTPUT.exists():
        raise SystemExit("Iteration-2 results already exist; preserve and create a new dated iteration")
    update_state("итерация 2: получаем закреплённые входные файлы")
    try:
        data = cr.fetch_all()
        update_state("итерация 2: проверяем хеши, покрытие и шесть метрик отдельно")
        out = aggregate(data)
        save(out)
        update_state("итерация 2 завершена: см. статус каждой метрики ниже")
        print(json.dumps(out["metric_summary"]), flush=True)
    except Exception as error:
        update_state("итерация 2 остановлена: " + type(error).__name__)
        raise


if __name__ == "__main__":
    main()
