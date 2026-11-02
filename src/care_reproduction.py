"""Reaggregate CARE Table 2 from pinned public scores; never emit case-level data.

This does not run CT reconstruction or nnU-Net. Public CSVs are read transiently
in memory; only hashes and aggregate results are saved. Freeze before scoring.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SHA = "2443d490c59eebaac9e8420d56b45bffed113feb"
BASE = f"https://raw.githubusercontent.com/MrGiovanni/CARE/{SHA}/"
PAPER = "https://papers.nips.cc/paper_files/paper/2025/file/307f375e35616bbc2861033966b44976-Paper-Conference.pdf"
SPLIT = "STEP3-CAREModel/splits/BDMAP_O_AV_meta_test.csv"
METHODS = ("intratomo", "nerf", "tensorf", "r2_gaussian", "naf", "FDK", "SART", "ASD_POCS", "Lineformer")
METRICS = ("ssim", "psnr", "large_nsd", "small_nsd", "intestine_cldice", "vessel_cldice")
GROUPS = {
    "large_nsd": [f"large_nsd_{i}" for i in (12, 10, 11, 13, 17)],
    "small_nsd": [f"small_nsd_{i}" for i in (8, 2, 3, 5, 7)],
    "intestine_cldice": ["tubular_cldice_6", "tubular_cldice_16"],
    "vessel_cldice": [f"vessel_cldice_{i}" for i in (1, 9, 15, 20, 21)],
}
PROTOCOL = ROOT / "docs/PROTOCOL_care_reproduction.md"
FREEZE = ROOT / "results/care_reproduction_inputs.json"
OUTPUT = ROOT / "results/care_reproduction.json"
AMENDMENT = ROOT / "docs/PROTOCOL_care_reproduction_amendment1.md"
AMENDMENT_FREEZE = ROOT / "results/care_reproduction_amendment1_inputs.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_inputs(data, frozen):
    if set(data) != set(frozen["inputs"]):
        raise ValueError("Source inventory changed after freeze")
    for name, content in data.items():
        if frozen["inputs"][name]["sha256"] != digest(content):
            raise ValueError(f"Source changed after freeze: {name}")


def csv_path(method, care, kind):
    return f"ReconstructionPipeline/resultsCSVcldice/BDMAP_O_{method}_50_{kind}{'_care' if care else ''}.csv"


def sources():
    paths = [SPLIT, "LICENSE", "ReconstructionPipeline/metric_utils.py",
             "ReconstructionPipeline/step5_calculateMetrics.py",
             "ReconstructionPipeline/step6_read_result_csv_for_table1st2nd.py"]
    paths += [csv_path(m, c, k) for m in METHODS for c in (False, True) for k in ("pixel", "seg")]
    return {**{p: BASE + p for p in paths}, "paper.pdf": PAPER}


def fetch_all():
    def get(item):
        name, url = item
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                return name, response.content
            except requests.RequestException:
                if attempt == 2:
                    raise RuntimeError(f"Cannot retrieve pinned source: {name}") from None
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(get, sources().items()))


def selected_ids(split, selection="author_venous"):
    values = split["bdmap_id"].astype(str)
    if selection == "author_venous":
        values = values[values.str.endswith("V")]
    elif selection == "paper_all_phases_posthoc":
        if not values.str.endswith(("_A", "_V")).all():
            raise ValueError("Unexpected phase suffix in official split")
    else:
        raise ValueError("Unknown case selection")
    ids = values.str[:-2]
    if ids.empty or ids.duplicated().any():
        raise ValueError("Empty or duplicate selected split keys")
    return set(ids)


def select_frame(frame, eligible):
    if frame["case_name"].duplicated().any():
        raise ValueError("Duplicate score keys; case identifiers suppressed")
    selected = frame[frame["case_name"].isin(eligible)].set_index("case_name").sort_index()
    if set(selected.index) != eligible:
        raise ValueError("Score file does not cover the selected split; identifiers suppressed")
    return selected


def endpoint_values(pixel, seg):
    if not pixel.index.equals(seg.index):
        raise ValueError("Pixel/segmentation case sets differ")
    values = {"ssim": pixel["ssim_3d"].to_numpy(float), "psnr": pixel["psnr_3d"].to_numpy(float)}
    for metric, columns in GROUPS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values[metric] = np.nanmean(seg[columns].to_numpy(float), axis=1)
    return values


def coverage_diagnostics(data, eligible):
    """Return aggregate coverage only, never source case keys."""
    rows = []
    for method in METHODS:
        for care in (False, True):
            for kind in ("pixel", "seg"):
                frame = pd.read_csv(io.BytesIO(data[csv_path(method, care, kind)]))
                keys = set(frame["case_name"])
                rows.append({"method": method, "care": care, "kind": kind,
                             "n_expected": len(eligible), "n_present": len(eligible & keys),
                             "n_missing": len(eligible - keys),
                             "n_duplicate_keys": int(frame["case_name"].duplicated().sum())})
    return rows


def summarize(values):
    values = np.asarray(values, dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite score in released endpoint")
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Endpoint has no finite cases")
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3),
            "n_finite": int(finite.size), "n_missing": int(values.size - finite.size)}


def paper_table(data):
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        page = doc[8].get_text()
    section = page.split("Table 2:", 1)[1].split("These results", 1)[0]
    triplets = re.findall(r"(\d+\.\d+)\s*\((\d+\.\d+),\s*(\d+\.\d+)\)", section)
    if len(triplets) != 108:
        raise ValueError(f"Expected 108 Table 2 metric triplets, found {len(triplets)}")
    return np.asarray(triplets, dtype=float).reshape(9, 2, 6, 3)


def execute(data, selection="author_venous"):
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != digest(PROTOCOL.read_bytes()):
        raise ValueError("Protocol changed after input freeze")
    validate_inputs(data, frozen)
    amendment = json.loads(AMENDMENT_FREEZE.read_text(encoding="utf-8"))
    if amendment["amendment_sha256"] != digest(AMENDMENT.read_bytes()):
        raise ValueError("Amendment changed after freeze")
    if amendment["original_manifest_sha256"] != digest(FREEZE.read_bytes()):
        raise ValueError("Original input manifest changed after amendment")
    eligible = selected_ids(pd.read_csv(io.BytesIO(data[SPLIT])), selection)
    coverage = coverage_diagnostics(data, eligible)
    if any(r["n_missing"] or r["n_duplicate_keys"] for r in coverage):
        suffix = "" if selection == "author_venous" else "_all_phases_posthoc"
        blocked = {"status": "blocked_case_coverage", "selection": selection,
                   "n_selected_cases": len(eligible), "official_commit": SHA,
                   "scope": "No score aggregation performed for this selection; incomplete case coverage.",
                   "coverage": coverage, "manifest": {
                       "protocol_sha256": digest(PROTOCOL.read_bytes()),
                       "code_sha256": digest(Path(__file__).read_bytes()),
                       "input_manifest_sha256": digest(FREEZE.read_bytes()),
                       "amendment_sha256": digest(AMENDMENT.read_bytes()),
                       "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}}
        OUTPUT.with_stem(OUTPUT.stem + suffix).write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        lines = ["# CARE Table 2 case-coverage failure", "", blocked["scope"], "",
                 f"Selection: {selection}; required cases: {len(eligible)}. Original source manifest and amendment remain unchanged.",
                 "", "| Method | CARE | File type | Required | Present | Missing |",
                 "|---|---|---|---|---|---|"]
        lines += [f"| {r['method']} | {r['care']} | {r['kind']} | {r['n_expected']} | {r['n_present']} | {r['n_missing']} |" for r in coverage]
        lines += ["", "Do not substitute a smaller subset or impute missing scores to claim reproduction. No H19 result follows from this failure."]
        (ROOT / f"results/care_reproduction{suffix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({k: blocked[k] for k in ("selection", "status", "n_selected_cases")}))
        return
    expected = paper_table(data["paper.pdf"])
    rows, mismatches, changes, input_issues = [], [], [], []
    for mi, method in enumerate(METHODS):
        arrays = []
        for ci, care in enumerate((False, True)):
            frames = [select_frame(pd.read_csv(io.BytesIO(data[csv_path(method, care, k)])), eligible)
                      for k in ("pixel", "seg")]
            vals = endpoint_values(*frames)
            arrays.append(vals)
            for ki, metric in enumerate(METRICS):
                try:
                    stats = summarize(vals[metric])
                except ValueError as exc:
                    input_issues.append({"method": method, "care": care, "metric": metric, "issue": str(exc)})
                    stats = {"median": None, "q1": None, "q3": None,
                             "n_finite": int(np.isfinite(vals[metric]).sum()),
                             "n_missing": int(np.isnan(vals[metric]).sum())}
                printed = dict(zip(("median", "q1", "q3"), expected[mi, ci, ki].tolist()))
                error = max(abs(stats[k] - printed[k]) for k in printed) if stats["median"] is not None else None
                row = {"method": method, "care": care, "metric": metric, **stats,
                       "paper": printed, "max_abs_error": error, "matches_rounding": error is not None and error <= .050001}
                rows.append(row)
                if not row["matches_rounding"]:
                    mismatches.append({k: row[k] for k in ("method", "care", "metric", "median", "paper", "max_abs_error")})
        for metric in METRICS:
            before, after = arrays[0][metric], arrays[1][metric]
            valid = np.isfinite(before) & np.isfinite(after)
            delta = after[valid] - before[valid]
            changes.append({"method": method, "metric": metric, "n_pairs": int(valid.sum()),
                            "improved": int((delta > 0).sum()), "tied": int((delta == 0).sum()),
                            "worsened": int((delta < 0).sum()),
                            "median_paired_change": float(np.median(delta)) if delta.size else None})
    out = {
        "status": "incomplete_endpoints" if input_issues else "table_reproduced" if not mismatches else "table_mismatch",
        "selection": selection,
        "scope": "Reaggregation of released author scores only; no image-level reproduction and no H19 test.",
        "official_commit": SHA, "n_selected_cases": len(eligible),
        "independent_participants": "Not established from score files and split.",
        "n_metric_triplets": len(rows), "n_values_compared": len(rows) * 3,
        "n_matching_triplets": sum(r["matches_rounding"] for r in rows),
        "mismatches": mismatches, "rows": rows, "paired_changes": changes, "input_issues": input_issues,
        "manifest": {
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest(PROTOCOL.read_bytes()), "code_sha256": digest(Path(__file__).read_bytes()),
            "input_manifest_sha256": digest(FREEZE.read_bytes()),
            "amendment_sha256": digest(AMENDMENT.read_bytes()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "versions": {p: version(p) for p in ("numpy", "pandas", "requests", "PyMuPDF")},
        },
        "limits": ["Author-produced scores are reused, not independently recomputed from CT images.",
                   "The table and selected reference numbers were seen before protocol writing.",
                   "No mean-fill, domain-shift or identity experiment has been executed here.",
                   "Missing values follow authors' nan-aware rules; endpoint denominators are reported.",
                   "Lineformer CSV mapping to SAX-NeRF is provisional and judged against table agreement."]}
    suffix = "" if selection == "author_venous" else "_all_phases_posthoc"
    output = OUTPUT.with_stem(OUTPUT.stem + suffix)
    output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k != "paper"} for r in rows]
    pd.DataFrame(flat).to_csv(ROOT / f"results/care_reproduction{suffix}_table.csv", index=False)
    report = ["# CARE Table 2 reproduction", "", out["scope"], "",
              f"Selection: {selection}. Pinned official commit: `{SHA}`. Selected cases: {len(eligible)}; independent people not established.",
              f"Triplets matching all printed median/IQR values within rounding: {out['n_matching_triplets']}/{len(rows)}.",
              "", "| Method | CARE | SSIM | PSNR | NSD large | NSD small | clDice intestine | clDice vessels |",
              "|---|---|---|---|---|---|---|---|"]
    for method in METHODS:
        for care in (False, True):
            rr = [r for r in rows if r["method"] == method and r["care"] == care]
            report.append(f"| {method} | {care} | " + " | ".join(
                f"{r['median']:.1f} ({r['q1']:.1f}, {r['q3']:.1f}); n={r['n_finite']}" if r["median"] is not None
                else "unavailable; n=0" for r in rr) + " |")
    report += ["", "## Interpretation", "", *[f"- {v}" for v in out["limits"]],
               "- Read the JSON for mismatch details, finite/missing counts and paired descriptive changes.",
               "- Original pipeline, weights and image-level data remain unchecked. This stage cannot refute or confirm H19.",
               "- Source attribution: Lin et al., NeurIPS 2025, https://github.com/MrGiovanni/CARE ; official repository LICENSE says CC BY-NC-ND. No upstream code is copied here."]
    (ROOT / f"results/care_reproduction{suffix}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("selection", "status", "n_selected_cases", "n_metric_triplets", "n_matching_triplets")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-inputs", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--freeze-amendment", action="store_true")
    args = parser.parse_args()
    if args.run and OUTPUT.exists():
        raise SystemExit("Historical R1 outputs already exist; preserve them and run care_iteration2.py")
    if args.freeze_inputs and FREEZE.exists():
        raise SystemExit("Input manifest already exists; preserve it and use a dated amendment for changes")
    if args.freeze_amendment:
        if AMENDMENT_FREEZE.exists():
            raise SystemExit("Amendment manifest already exists")
        AMENDMENT_FREEZE.write_text(json.dumps({"amendment_sha256": digest(AMENDMENT.read_bytes()),
            "original_manifest_sha256": digest(FREEZE.read_bytes()),
            "additional_selection": "paper_all_phases_posthoc"}, indent=2) + "\n", encoding="utf-8")
        print("Amendment frozen without score aggregation")
        return
    data = fetch_all()
    if args.freeze_inputs:
        frozen = {"official_commit": SHA, "protocol_sha256": digest(PROTOCOL.read_bytes()),
                  "inputs": {k: {"url": sources()[k], "sha256": digest(v), "bytes": len(v)} for k, v in data.items()}}
        FREEZE.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "inputs_frozen_without_aggregation", "n_sources": len(data)}))
    else:
        execute(data)
        execute(data, "paper_all_phases_posthoc")


if __name__ == "__main__":
    main()
