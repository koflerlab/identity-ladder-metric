"""BraTS inpainting metric baseline on the 2D ladder pairs (docs/PROTOCOL_2d_ladder.md, Amendment 1; descriptive).

Truth = the anchor's grey base image; a fixed centred 64 x 64 mask is filled from the candidate-slot image of the same
pair (own image, fellow eye flipped, population mean, or each matched stranger). All nine outputs of the official
`inpainting` package are computed through toy_ladder.panel; c is each fill's position among the pair's strangers,
in the direction of the metric, with half credit for ties.

Writes results/ladder2d_brats_baseline.{json,csv,md} (aggregates only); per-pair scores stay in the private run folder.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import ladder2d as L  # noqa: E402
import ladder2d_select as sel  # noqa: E402
from atlas_common import HIGHER, METRICS, calibrate  # noqa: E402
from toy_ladder import panel  # noqa: E402

MASK = np.zeros((L.BASE, L.BASE), bool)
MASK[80:144, 80:144] = True
ROWS = {"same_eye_same_sitting": "same-sitting repeat",
        "same_camera_<1y": "other visit, same camera, under 1 year",
        "same_camera_1y+": "other visit, same camera, 1 year or more",
        "other_camera": "other visit, other camera",
        "fellow_eye_same_visit": "mirrored other side (fellow eye flipped)",
        "fellow_eye_same_sitting": "mirrored other side (fellow eye flipped)"}
ROW_ORDER = ["same-sitting repeat", "other visit, same camera, under 1 year", "other visit, same camera, 1 year or more",
             "other visit, other camera", "mirrored other side (fellow eye flipped)", "population mean",
             "stranger (matched)"]
DATASETS = ["GRAPE", "UPOL", "UTIRIS", "QTIM MRI (from its own run)"]


def score_pair(p: dict, imgs: "L.Images", mask: np.ndarray = MASK, flip: bool | None = None) -> dict:
    """Nine metrics for own fill, mean fill and every stranger fill; c of own and mean against the strangers.

    flip=None flips the candidate slot for fellow-eye pairs; a registered view that already flipped passes False.
    """
    flip = p["rung"].startswith("fellow") if flip is None else flip
    cand = (lambda a: a[:, ::-1]) if flip else (lambda a: a)
    truth = L.grey(imgs.get(p["anchor"])).astype(np.float64)
    fills = {"own": L.grey(imgs.get(p["candidate"])), "mean": L.grey(imgs.mean(p["mean_members"]))}
    raw = {k: panel(cand(v).astype(np.float64), truth, mask) for k, v in fills.items()}
    raw = {k: {m: float(v[m]) for m in METRICS} for k, v in raw.items()}
    refs = [{m: float(v) for m, v in panel(cand(L.grey(imgs.get(s))).astype(np.float64), truth, mask).items()
             if m in METRICS} for s in p["reference"]]
    return {"raw": raw, "strangers": refs, "c": {k: calibrate(v, refs) for k, v in raw.items()}}


def summarise(per_pair: list) -> dict:
    """Per dataset and row: raw median and IQR of each metric, and mean c with the person-cluster interval."""
    out = {}
    for ds in sorted({r["dataset"] for r in per_pair}):
        rs = [r for r in per_pair if r["dataset"] == ds]
        _, counts = L.person_weights(np.unique([r["person"] for r in rs]))
        pid = {p: i for i, p in enumerate(np.unique([r["person"] for r in rs]))}
        groups = {}
        for r in rs:
            groups.setdefault(ROWS[r["rung"]], []).append((r, "own"))
            if not r["rung"].startswith("fellow"):
                groups.setdefault("population mean", []).append((r, "mean"))
        d = {}
        for row, items in groups.items():
            ix = np.array([pid[r["person"]] for r, _ in items])
            d[row] = {"n_pairs": len(items), "n_people": int(len(set(ix)))}
            for m in METRICS:
                v = np.array([r["raw"][k][m] for r, k in items])
                c = np.array([r["c"][k][m] for r, k in items])
                d[row][m] = {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                             "c": L.boot_mean(c, ix, counts)}
        strangers = [s for r in rs if not r["rung"].startswith("fellow") for s in r["strangers"]]
        d["stranger (matched)"] = {"n_fills": len(strangers), "c": "0.5 by construction"}
        for m in METRICS:
            v = np.array([s[m] for s in strangers])
            d["stranger (matched)"][m] = {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)),
                                                                                 float(np.percentile(v, 75))]}
        out[ds] = d
    return out


def table(summary: dict) -> pd.DataFrame:
    rows = []
    for row in ROW_ORDER:
        for m in METRICS:
            rec = {"row": row, "metric": m, "higher_is_better": m in HIGHER}
            for ds in DATASETS:
                e = summary.get(ds, {}).get(row, {}).get(m)
                if e is None:
                    rec[f"{ds} median [IQR]"] = rec[f"{ds} c [95% CI]"] = ""
                    continue
                rec[f"{ds} median [IQR]"] = f"{e['median']:.4g} [{e['iqr'][0]:.4g}, {e['iqr'][1]:.4g}]"
                rec[f"{ds} c [95% CI]"] = ("0.5" if "c" not in e else
                                           f"{e['c']['mean']:.3f} [{e['c']['ci95'][0]:.3f}, {e['c']['ci95'][1]:.3f}]")
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    hashes = L.guard()
    selections = {"grape": json.load(open(L.EXPECTED["selection_grape"][0], encoding="utf-8")),
                  "iris": json.load(open(L.EXPECTED["selection_iris"][0], encoding="utf-8"))}
    u_grape = sel.grape_universe(pd.read_csv(sel.GRAPE_META))
    utiris, _ = sel.utiris_table()
    u_iris = sel.iris_universe(sel.upol_table(), utiris)
    jobs = [("GRAPE", selections["grape"], {n: os.path.join(L.GRAPE_IMG, n) for n in u_grape.image}, True)]
    for ds, root in (("UPOL", sel.UPOL), ("UTIRIS", sel.UTIRIS)):
        u = u_iris[u_iris.dataset == ds]
        jobs.append((ds, selections["iris"], {n: os.path.join(root, n) for n in u.image}, False))
    per_pair = []
    for ds, s, paths, rgb in jobs:
        imgs = L.Images(paths, rgb, v=False)
        prs = [p for p in s["pairs"] if p["dataset"] == ds]
        for p in prs:
            per_pair.append({"dataset": ds, "rung": p["rung"], "person": p["person"], **score_pair(p, imgs)})
        print(f"{ds}: {len(prs)} pairs, {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(sel.OUT_DIR, "brats_baseline_pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(per_pair, fh)
    summary = summarise(per_pair)
    import inpainting
    import torchmetrics
    meta = {"protocol": "docs/PROTOCOL_2d_ladder.md, Amendment 1 (descriptive)", "mask": "centred 64 x 64 of 224 x 224",
            "inputs_sha256": hashes, "code_sha256": L.sha256(__file__), "seed": L.SEED,
            "versions": {"inpainting": getattr(inpainting, "__version__", "not exposed"),
                         "torchmetrics": torchmetrics.__version__},
            "qtim_column": "empty: filled from the QTIM MRI run's own aggregate output, not recomputed here",
            "run_seconds": round(time.time() - t0)}
    res = os.path.join(L.PROJECT, "results")
    with open(os.path.join(res, "ladder2d_brats_baseline.json"), "w", encoding="utf-8") as fh:
        json.dump({**meta, "summary": summary}, fh, indent=1)
    table(summary).to_csv(os.path.join(res, "ladder2d_brats_baseline.csv"), index=False)
    print(table(summary)[table(summary).metric.isin(["mse", "psnr", "ssim"])].to_string(index=False))


def registered_jobs():
    """(dataset, pairs, view, mask) with every image in its registered frame (Amendment 2, 2a, 2b)."""
    import ladder2d_register as R
    selections = {"grape": json.load(open(L.EXPECTED["selection_grape"][0], encoding="utf-8")),
                  "iris": json.load(open(L.EXPECTED["selection_iris"][0], encoding="utf-8"))}
    u_grape = sel.grape_universe(pd.read_csv(sel.GRAPE_META))
    utiris, _ = sel.utiris_table()
    u_iris = sel.iris_universe(sel.upol_table(), utiris)
    iris_mask = np.zeros_like(MASK)
    iris_mask[R.IRIS_BOX] = True
    imgs = L.Images({n: os.path.join(L.GRAPE_IMG, n) for n in u_grape.image}, True, v=True)
    yield "GRAPE", [p for p in selections["grape"]["pairs"]], R.FundusView(imgs), MASK
    for ds, root, frame in (("UPOL", sel.UPOL, "pupil"), ("UTIRIS", sel.UTIRIS, "limbus")):
        u = u_iris[u_iris.dataset == ds]
        view = R.IrisView({n: os.path.join(root, n) for n in u.image}, frame=frame)
        yield ds, [p for p in selections["iris"]["pairs"] if p["dataset"] == ds], view, iris_mask


def usable(p: dict, view, k_min: int = 30, m_min: int = 8) -> dict | None:
    """Pair with failed images removed; None if the anchor or candidate failed or too few strangers or members remain."""
    if not (view.ok(p["anchor"]) and view.ok(p["candidate"])):
        return None
    ref = [s for s in p["reference"] if view.ok(s)]
    mem = [s for s in p["mean_members"] if view.ok(s)]
    if len(ref) < k_min or len(mem) < m_min:
        return None
    return {**p, "reference": ref, "mean_members": mem}


def main_registered():
    t0 = time.time()
    hashes = L.guard()
    per_pair, exclusions = [], {}
    for ds, prs, view, mask in registered_jobs():
        kept = [q for q in (usable(p, view) for p in prs) if q is not None]
        exclusions[ds] = {"images_failed_registration": int(sum(not q["ok"] for q in view.qc.values())),
                          "images": len(view.qc), "pairs_selected": len(prs), "pairs_kept": len(kept)}
        for p in kept:
            per_pair.append({"dataset": ds, "rung": p["rung"], "person": p["person"], **score_pair(p, view, mask)})
        print(f"{ds}: {exclusions[ds]}, {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(sel.OUT_DIR, "brats_baseline_registered_pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(per_pair, fh)
    summary = summarise(per_pair)
    meta = {"protocol": "docs/PROTOCOL_2d_ladder.md, Amendments 1, 2, 2a, 2b (registered)",
            "patch": {"GRAPE": "centred 64 x 64 in the disc-fovea frame", "UPOL": "rows 80-143, cols 136-199, pupil frame",
                      "UTIRIS": "rows 80-143, cols 136-199, limbus frame"},
            "exclusions": exclusions, "inputs_sha256": hashes, "code_sha256": {f: L.sha256(os.path.join(L.PROJECT, "src", f))
                                                                              for f in ("ladder2d_brats.py", "ladder2d_register.py")},
            "seed": L.SEED, "run_seconds": round(time.time() - t0)}
    res = os.path.join(L.PROJECT, "results")
    with open(os.path.join(res, "ladder2d_brats_baseline_registered.json"), "w", encoding="utf-8") as fh:
        json.dump({**meta, "summary": summary}, fh, indent=1)
    table(summary).to_csv(os.path.join(res, "ladder2d_brats_baseline_registered.csv"), index=False)
    print(table(summary)[table(summary).metric.isin(["mse", "ssim"])].to_string(index=False))


if __name__ == "__main__":
    main_registered() if "--registered" in sys.argv else main()
