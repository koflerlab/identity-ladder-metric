"""Real-data versions of the synthetic visual and mixing figures (src/rebuild_visuals.py layout), one per dataset.

Same setup as the BraTS baseline (Amendment 1): truth = anchor grey base image, fixed centred 64 x 64 mask filled
from a candidate. Mixing path: own repeat or later image (100%) to the population mean of 16 matched strangers (0%).
The pictured case is chosen by a fixed rule (first person in ID order who has every column), never by score.
Figures hold real images and stay in the private run folder; only this code is in the repository.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import ladder2d as L  # noqa: E402
import ladder2d_brats as B  # noqa: E402
import ladder2d_select as sel  # noqa: E402
from atlas_common import METRICS, calibrate  # noqa: E402
from rebuild_visuals import BLUE, ORANGE, STEP_LABELS, TEAL, save  # noqa: E402
from detail_report import METRIC_LABELS  # noqa: E402
from toy_ladder import panel  # noqa: E402

FIG_DIR = os.path.join(sel.OUT_DIR, "figures")
LAMS = (1.0, 0.75, 0.5, 0.25, 0.0)
PRIMARY = {"GRAPE": "same_camera_<1y", "UPOL": "same_eye_same_sitting", "UTIRIS": "same_eye_same_sitting"}
STEP_TEXT = {"GRAPE": ["100% other visit"] + STEP_LABELS[1:], "UPOL": ["100% repeat"] + STEP_LABELS[1:],
             "UTIRIS": ["100% repeat"] + STEP_LABELS[1:]}
STEP_TEXT = {k: [v[0]] + [s.replace("original", v[0][5:]) for s in v[1:]] for k, v in STEP_TEXT.items()}
M = B.MASK
BOX = (80, 144)
BIG = (round(80 * L.V_SIZE / L.BASE), round(144 * L.V_SIZE / L.BASE))


def mean_control(fill: np.ndarray, truth: np.ndarray, mu: np.ndarray) -> dict:
    """Residual ratio and target projection inside the mask (the synthetic figures' diagnostics)."""
    f, x, m = fill[M], truth[M], mu[M]
    t, r = x - m, f - m
    return {"residual_ratio": float(np.linalg.norm(r) / max(np.linalg.norm(t), 1e-12)),
            "target_projection": float(r @ t / max(t @ t, 1e-12)),
            "target_distance": float(np.sqrt(np.mean((f - x) ** 2))),
            "mean_distance": float(np.sqrt(np.mean((f - m) ** 2))),
            "target_mean_distance": float(np.sqrt(np.mean(t ** 2)))}


def pupil(img: np.ndarray, win: int = 56) -> np.ndarray:
    """Estimated pupil: inside the central window, the largest dark region (below a quarter of the way from the darkest
    non-background value to the median) that does not touch the window edge; holes (reflections) filled."""
    out = np.zeros_like(img, bool)
    c = img.shape[0] // 2
    w = img[c - win:c + win, c - win:c + win]
    tissue = w[w > 0.02]
    if tissue.size == 0:
        return out
    thr = tissue.min() + 0.25 * (np.median(tissue) - tissue.min())
    lab, n = ndimage.label((w < thr) & (w > 0.02))
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    sizes = [(int((lab == k).sum()), k) for k in range(1, n + 1) if k not in edge]
    if not sizes:
        return out
    out[c - win:c + win, c - win:c + win] = ndimage.binary_fill_holes(lab == max(sizes)[1])
    return out


def path_scores(pairs: list, imgs: "L.Images", strangers: list) -> dict:
    """Per lambda and metric: raw values and c over all pairs; plus mean-control diagnostics and a stranger control."""
    raw = {lam: {m: [] for m in METRICS} for lam in LAMS}
    cal = {lam: {m: [] for m in METRICS} for lam in LAMS}
    diag = {lam: [] for lam in LAMS}
    stranger_diag = []
    for p, refs in zip(pairs, strangers):
        x = L.grey(imgs.get(p["anchor"])).astype(np.float64)
        y = L.grey(imgs.get(p["candidate"])).astype(np.float64)
        mu = L.grey(imgs.mean(p["mean_members"])).astype(np.float64)
        for lam in LAMS:
            f = mu + lam * (y - mu)
            v = {m: float(s) for m, s in panel(f, x, M).items() if m in METRICS}
            c = calibrate(v, refs)
            for m in METRICS:
                raw[lam][m].append(v[m])
                cal[lam][m].append(c[m])
            diag[lam].append(mean_control(f, x, mu))
        s = L.grey(imgs.get(p["reference"][0])).astype(np.float64)
        stranger_diag.append(mean_control(s, x, mu))
    return {"raw": raw, "c": cal, "diag": diag, "stranger": stranger_diag, "people": [p["person"] for p in pairs]}


def mixing_figure(ds: str, case: dict, imgs: "L.Images", res: dict, n_people: int):
    fig = plt.figure(figsize=(15.5, 15))
    grid = fig.add_gridspec(5, 6, height_ratios=[1.5, 1.1, 1.1, 1.1, 1.0], hspace=.60, wspace=.88)
    strip = grid[0, :].subgridspec(1, 5, wspace=.06)
    x = L.grey(imgs.get(case["anchor"]))
    y, mu = L.grey(imgs.get(case["candidate"])), L.grey(imgs.mean(case["mean_members"]))
    lo, hi = np.percentile(x, [1, 99])
    for j, lam in enumerate(LAMS):
        ax = fig.add_subplot(strip[0, j])
        ax.imshow(np.where(M, mu + lam * (y - mu), x)[40:184, 40:184], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.add_patch(Rectangle((BOX[0] - 40 - .5, BOX[0] - 40 - .5), 64, 64, fill=False, ec=ORANGE, lw=1.2))
        ax.set_title(STEP_TEXT[ds][j], fontsize=12)
        ax.set(xticks=[], yticks=[])
    xs = np.arange(5)
    _, counts = L.person_weights(np.unique(res["people"]))
    pid = {p: i for i, p in enumerate(np.unique(res["people"]))}
    ix = np.array([pid[p] for p in res["people"]])
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(grid[1 + i // 3, 2 * (i % 3):2 * (i % 3) + 2])
        rhs = ax.twinx()
        r = np.array([res["raw"][lam][metric] for lam in LAMS])
        q = np.percentile(r, [25, 50, 75], axis=1)
        ax.plot(xs, q[1], "-o", color=BLUE, lw=1.8, ms=4)
        ax.fill_between(xs, q[0], q[2], color=BLUE, alpha=.13)
        cb = [L.boot_mean(np.array(res["c"][lam][metric]), ix, counts) for lam in LAMS]
        rhs.plot(xs, [c["mean"] for c in cb], "--s", color=ORANGE, lw=1.5, ms=3.5)
        rhs.fill_between(xs, [c["ci95"][0] for c in cb], [c["ci95"][1] for c in cb], color=ORANGE, alpha=.10)
        rhs.axhline(.5, color=ORANGE, lw=.6, ls=":")
        ax.set_title(METRIC_LABELS[metric], fontsize=12, pad=7)
        ax.set_xticks(xs, ["100", "75", "50", "25", "0"], fontsize=9)
        ax.set_xlim(-.12, 4.12)
        rhs.set_ylim(-.04, 1.04)
        rhs.set_yticks([0, .5, 1])
        rhs.tick_params(axis="y", colors=ORANGE, labelsize=9)
        ax.tick_params(axis="y", colors=BLUE, labelsize=9)
        if i % 3 == 0:
            ax.set_ylabel("Raw value", color=BLUE, fontsize=10)
        if i % 3 == 2:
            rhs.set_ylabel("Calibrated c", color=ORANGE, fontsize=10)
        ax.set_xlabel("Own-image contribution, %", fontsize=9)
        ax.grid(color="#e8e8e8", lw=.7)
        ax.spines["top"].set_visible(False)
        rhs.spines["top"].set_visible(False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    distance, projection = fig.add_subplot(grid[4, :3]), fig.add_subplot(grid[4, 3:])
    for key, col, label in [("target_distance", BLUE, "Distance to original"), ("mean_distance", ORANGE, "Distance to mean")]:
        v = np.array([[d[key] for d in res["diag"][lam]] for lam in LAMS])
        q = np.percentile(v, [25, 50, 75], axis=1)
        distance.plot(xs, q[1], "-o", color=col, label=label, ms=4)
        distance.fill_between(xs, q[0], q[2], color=col, alpha=.12)
    distance.axhline(np.median([d["target_mean_distance"] for d in res["diag"][0.0]]), color="#666", ls=":",
                     label="Original-to-mean distance")
    distance.set_title("Three distances in the same representation", fontsize=12)
    distance.set_xticks(xs, ["100", "75", "50", "25", "0"])
    distance.set_xlabel("Own-image contribution, %", fontsize=10)
    distance.set_ylabel("RMS distance in the patch", fontsize=10)
    distance.legend(fontsize=8, frameon=False, loc="center left")
    for key, col, label, marker in [("residual_ratio", ORANGE, "Distance-to-mean ratio", "o"),
                                    ("target_projection", TEAL, "Target residual projection", "x")]:
        vals = [np.median([d[key] for d in res["diag"][lam]]) for lam in LAMS]
        projection.plot(xs, vals, color=col, marker=marker, ms=5, label=label)
        projection.scatter([6], [np.median([d[key] for d in res["stranger"]])], color=col, marker=marker, s=45)
    projection.set_xticks(np.r_[xs, 6], ["100", "75", "50", "25", "0", "Matched\nstranger"])
    projection.set_title("A stranger can be far from the mean without the person's detail", fontsize=12)
    projection.legend(fontsize=8, frameon=False, loc="upper center")
    projection.axvline(5, color="#aaa", ls=":", lw=.8)
    for ax in (distance, projection):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#ededed")
    fig.suptitle(f"Real {NAMES[ds]} | own image → population mean", fontsize=21, y=.991)
    legend = [Line2D([0], [0], color=BLUE, marker="o", label="Raw metric, median (left axis)"),
              Line2D([0], [0], color=ORANGE, ls="--", marker="s", label="Stranger-calibrated c, mean (right axis)")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .970), ncol=2, frameon=False, fontsize=11)
    fig.text(.5, .012, f"Images: first person in ID order with every rung (fixed rule). Curves: {len(res['people'])} pairs, "
             f"{n_people} people ({PRIMARY[ds]}); blue band: interquartile range over pairs; orange band: 95% person-cluster "
             "bootstrap interval.\nf = lambda x own image + (1-lambda) x mean of 16 matched strangers, filled into a fixed centred "
             "64 x 64 square of the anchor (orange box). c against the pair's matched strangers; dotted line c = 0.5.\n"
             "Diagnostics in the grey patch. Local research copy of public data; no redistribution.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.07, right=.93, top=.93, bottom=.085)
    save(fig, os.path.join(FIG_DIR, f"mixing_{ds.lower()}_real"))


NAMES = {"GRAPE": "fundus (GRAPE)", "UPOL": "iris (UPOL)", "UTIRIS": "near-infrared iris (UTIRIS)"}


def visual_case(ds: str, pairs: list, u: pd.DataFrame, aligned=None) -> tuple:
    """Fixed rule: first person in ID order whose primary-rung pair has every column available (and, with aligned,
    passes the display alignment rule)."""
    rows = u.set_index("image")
    fallback = []
    for p in sorted([p for p in pairs if p["rung"] == PRIMARY[ds]], key=lambda q: (q["person"], q["anchor"])):
        a = rows.loc[p["anchor"]]
        mine = u[u.person == p["person"]]
        fellow = mine[(mine.side != a.side) & (abs(mine.time - a.time) < 1e-6) & (mine.device == a.device)]
        cols = {"Original": p["anchor"], ("Other visit" if ds == "GRAPE" else "Repeat capture"): p["candidate"]}
        if ds == "GRAPE":
            other_cam = mine[(mine.unit == a.unit) & (mine.device != a.device)]
            if other_cam.empty:
                continue
            cols["Other camera"] = sorted(other_cam.image)[0]
        if fellow.empty:
            continue
        cols["Other eye (flipped)"] = sorted(fellow.image)[0]
        cols["Matched stranger"] = p["reference"][0]
        cols["Population mean"] = None
        if aligned is None:
            return p, cols
        off = aligned(cols)
        if off <= DISPLAY_MAX_OFFSET:
            return p, cols
        fallback.append((off, p, cols))
    if fallback:  # no case within the threshold: the smallest worst-case landmark offset (still no metric)
        _, p, cols = min(fallback, key=lambda t: t[0])
        return p, cols
    raise ValueError(f"no complete case in {ds}")


DISPLAY_MAX_OFFSET = 8.0  # pixels of the 224 base image; display rule of 2 Nov 2026, fixed before redrawing


def landmark_xy(path: str, ds: str, flip: bool) -> np.ndarray | None:
    """Pupil centre (iris) or disc and fovea (fundus), in 224-base pixels; position only, no metric."""
    import iris_codes as ic
    import ladder2d_register as R
    if ds == "GRAPE":
        rgb = L.load_base(path, L.V_SIZE, True)
        lm = R.fundus_landmarks(rgb[:, ::-1] if flip else rgb)
        return None if lm is None else np.concatenate(lm) * L.BASE / L.V_SIZE
    g = L.load_base(path, L.BASE, False)
    try:
        x, y, _ = ic.pupil_blob((g[:, ::-1] if flip else g) * 255.0)
    except ValueError:
        return None
    return np.array([x, y])


def alignment_rule(ds: str, paths: dict):
    """Display rule: every non-stranger column within DISPLAY_MAX_OFFSET of the original's landmarks."""
    def ok(cols: dict) -> float:
        """Largest landmark offset of the non-stranger columns from the original (inf if a landmark is missing)."""
        ref = landmark_xy(paths[cols["Original"]], ds, False)
        if ref is None:
            return np.inf
        worst = 0.0
        for label, name in cols.items():
            if name is None or label in ("Original", "Matched stranger"):
                continue
            xy = landmark_xy(paths[name], ds, label.startswith("Other eye"))
            if xy is None:
                return np.inf
            worst = max(worst, float(np.max(np.hypot(*(xy - ref).reshape(-1, 2).T))))
        return worst
    return ok


def visual_figure(ds: str, case: dict, cols: dict, imgs: "L.Images"):
    x = L.grey(imgs.get(case["anchor"]))
    mu = L.grey(imgs.mean(case["mean_members"]))
    lo, hi = np.percentile(x, [1, 99])
    big = ds == "GRAPE"
    if big:
        x5 = imgs.get(case["anchor"], True)
        mu5 = imgs.mean(case["mean_members"], True)
        m5 = np.zeros(x5.shape[:2], bool)
        m5[BIG[0]:BIG[1], BIG[0]:BIG[1]] = True
        oracle = L.vessel_masks(x5, fractions=(L.V_FRACTIONS[0],))[0]
    else:
        oracle = pupil(x)
    n = len(cols)
    fig, axes = plt.subplots(3, n, figsize=(17 * n / 6, 9))
    for j, (label, name) in enumerate(cols.items()):
        flip = label.startswith("Other eye")
        src = mu if name is None else L.grey(imgs.get(name))
        src = src[:, ::-1] if flip else src
        filled = np.where(M, src, x)
        if big:
            s5 = mu5 if name is None else imgs.get(name, True)
            s5 = s5[:, ::-1] if flip else s5
            seg = L.vessel_masks(np.where(m5[..., None], s5, x5), fractions=(L.V_FRACTIONS[0],))[0]
            ref = oracle
            crop = slice(BIG[0] - 40, BIG[1] + 40)
        else:
            seg, ref, crop = pupil(filled), oracle, slice(None)
        axes[0, j].imshow(filled, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[0, j].add_patch(Rectangle((BOX[0] - .5, BOX[0] - .5), 64, 64, fill=False, ec=ORANGE, lw=1.2))
        axes[1, j].imshow(seg[crop, crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[2, j].imshow(np.ones(seg[crop, crop].shape), cmap="gray", vmin=0, vmax=1)
        axes[2, j].contour(ref[crop, crop], [.5], colors=[BLUE], linewidths=1.2 if big else 2)
        axes[2, j].contour(seg[crop, crop], [.5], colors=[ORANGE], linewidths=.8 if big else 1.2, linestyles="--")
        axes[0, j].set_title(label, fontsize=13)
        for ax in axes[:, j]:
            ax.set(xticks=[], yticks=[])
        d = mean_control(filled.astype(np.float64), x.astype(np.float64), mu.astype(np.float64))
        axes[2, j].set_xlabel(f"Mean-distance ratio {d['residual_ratio']:.2f}\nTarget projection {d['target_projection']:.2f}",
                              fontsize=10)
    rows = ["Intensity (patch filled)", "Vessels around the patch" if big else "Estimated pupil",
            "Vessel contours" if big else "Pupil contours"]
    for ax, label in zip(axes[:, 0], rows):
        ax.set_ylabel(label, fontsize=12)
    cols_text = ", ".join(k.lower() for k in list(cols)[1:])
    fig.suptitle(f"Real {NAMES[ds]}: {cols_text}", fontsize=17)
    seg_text = ("Vessels: fixed classical pipeline (CLAHE, Frangi, top 12% inside the field of view) on the 512-pixel image, "
                "shown around the patch." if big else "Pupil: largest dark region by a fixed threshold, an estimate, not ground truth.")
    fig.text(.5, .025, "First person in ID order with every column and landmarks within 8 px of the original, else the smallest offset (display rule, not chosen by score). Blue contour: original; "
             "orange dashed: candidate.\nFixed centred 64 x 64 patch (orange box) filled from each candidate; one grey window "
             "from the original. " + seg_text + "\nPopulation mean: 16 matched strangers. Local research copy of public data; "
             "no redistribution.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.04, right=.99, top=.90, bottom=.16, hspace=.08, wspace=.08)
    save(fig, os.path.join(FIG_DIR, f"visual_{ds.lower()}_real"))


BLUR_SIGMA = 3.0  # pixels of the 224 base image; fixed before drawing


def card_fills(cols: dict, imgs: "L.Images", case: dict) -> dict:
    """Grey candidate images per column (fellow eye flipped; mean; blurred matched stranger added)."""
    out = {}
    for label, name in cols.items():
        src = L.grey(imgs.mean(case["mean_members"])) if name is None else L.grey(imgs.get(name))
        out[label] = src[:, ::-1] if label.startswith("Other eye") else src
    out["Blurred stranger"] = ndimage.gaussian_filter(out["Matched stranger"], BLUR_SIGMA)
    return out


def detail_card(ds: str, case: dict, cols: dict, imgs: "L.Images"):
    """Layout of the real-brain card: location, context with box, enlarged patch, candidate minus original."""
    x = L.grey(imgs.get(case["anchor"])).astype(np.float64)
    mu = L.grey(imgs.mean(case["mean_members"])).astype(np.float64)
    fills = card_fills(cols, imgs, case)
    lo, hi = np.percentile(x, [1, 99])
    n = len(fills)
    fig, axes = plt.subplots(3, n, figsize=(2.45 * n + .8, 8.8))
    ctx, roi = slice(40, 184), slice(BOX[0], BOX[1])
    for j, (label, src) in enumerate(fills.items()):
        filled = np.where(M, src, x)
        axes[0, j].imshow(filled[ctx, ctx], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[0, j].add_patch(Rectangle((BOX[0] - 40 - .5, BOX[0] - 40 - .5), 64, 64, fill=False, ec=ORANGE, lw=1.2))
        axes[1, j].imshow(filled[roi, roi], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        diff = axes[2, j].imshow((filled - x)[roi, roi], cmap="RdBu_r", vmin=-.4, vmax=.4, interpolation="nearest")
        v = panel(filled.astype(np.float64), x, M)
        d = mean_control(filled.astype(np.float64), x, mu)
        axes[2, j].set_xlabel(f"RMSE {float(v['rmse']):.3f} | SSIM {float(v['ssim']):.3f}\n"
                              f"Dist. to mean / original {d['residual_ratio']:.2f}", fontsize=8.5)
        axes[0, j].set_title(label, fontsize=11)
        for ax in axes[:, j]:
            ax.set(xticks=[], yticks=[])
    for ax, text in zip(axes[:, 0], ["Tissue + context", "Evaluated patch, enlarged", "Candidate minus original"]):
        ax.set_ylabel(text, fontsize=11)
    loc = fig.add_axes([.03, .80, .11, .15])
    loc.imshow(x, cmap="gray", vmin=lo, vmax=hi)
    loc.add_patch(Rectangle((BOX[0], BOX[0]), 64, 64, fill=False, ec=ORANGE, lw=1.5))
    loc.set(xticks=[], yticks=[])
    cax = fig.add_axes([.965, .14, .008, .2])
    fig.colorbar(diff, cax=cax)
    fig.suptitle(f"Real {NAMES[ds]}: same location, real repeat, other side, stranger and mean", fontsize=16, y=.985)
    fig.text(.17, .9, "Orange: evaluated 64 x 64 patch (fixed centre of the 224-pixel image).\n"
             "Other eye: the fellow eye flipped left-right, so the same square covers the mirrored position.", fontsize=11)
    fig.text(.17, .83, "Below: the same patch is replaced by each candidate. Outside the orange box, the original image is "
             "unchanged.", fontsize=11)
    fig.text(.5, .02, f"First person in ID order with every column and landmarks within {DISPLAY_MAX_OFFSET:g} px of the original, else the smallest offset (display rule, no metric). One display window and one "
             "error scale for every column; official BraTS package metrics.\nMean: 16 matched strangers, the person excluded. "
             f"Blurred stranger: Gaussian sigma {BLUR_SIGMA:g} px. No registration. Local research copy of public data; "
             "no redistribution.", ha="center", fontsize=9.5)
    fig.subplots_adjust(left=.075, right=.95, top=.76, bottom=.13, hspace=.13, wspace=.12)
    save(fig, os.path.join(FIG_DIR, f"card_{ds.lower()}_real"))


def donor_strip(ds: str, case: dict, cols: dict, imgs: "L.Images"):
    """Layout of the synthetic-tissue strip: intensity row and segmentation row for every donor position."""
    x = L.grey(imgs.get(case["anchor"]))
    fills = {k: v for k, v in card_fills(cols, imgs, case).items() if k != "Blurred stranger"}
    lo, hi = np.percentile(x, [1, 99])
    big = ds == "GRAPE"
    n = len(fills)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.4))
    ctx = slice(40, 184)
    for j, (label, src) in enumerate(fills.items()):
        filled = np.where(M, src, x)
        if big:  # vessels on the 512 image, filled the same way, shown at the context crop
            x5 = imgs.get(case["anchor"], True)
            name = cols[label]
            s5 = imgs.mean(case["mean_members"], True) if name is None else imgs.get(name, True)
            s5 = s5[:, ::-1] if label.startswith("Other eye") else s5
            m5 = np.zeros(x5.shape[:2], bool)
            m5[BIG[0]:BIG[1], BIG[0]:BIG[1]] = True
            seg = L.vessel_masks(np.where(m5[..., None], s5, x5), fractions=(L.V_FRACTIONS[0],))[0]
            c5 = slice(round(40 * L.V_SIZE / L.BASE), round(184 * L.V_SIZE / L.BASE))
            seg_view, off, side = seg[c5, c5], BIG[0] - c5.start, BIG[1] - BIG[0]
        else:
            seg_view, off, side = pupil(filled)[ctx, ctx], BOX[0] - 40, 64
        axes[0, j].imshow(filled[ctx, ctx], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[0, j].add_patch(Rectangle((BOX[0] - 40 - .5, BOX[0] - 40 - .5), 64, 64, fill=False, ec=ORANGE, lw=1))
        axes[1, j].imshow(seg_view, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].add_patch(Rectangle((off - .5, off - .5), side, side, fill=False, ec=ORANGE, lw=1))
        axes[0, j].set_title(label, fontsize=11)
        for ax in axes[:, j]:
            ax.set(xticks=[], yticks=[])
    axes[0, 0].set_ylabel("Intensity", fontsize=11)
    axes[1, 0].set_ylabel("Vessels" if big else "Estimated pupil", fontsize=11)
    fig.suptitle(f"Real {NAMES[ds]}: all donor positions, including the other side", fontsize=15)
    fig.text(.5, .03, "First person in ID order with every column and landmarks within 8 px of the original, else the smallest offset (display rule). Orange box: fill / evaluation region. One grey "
             "window for all intensities.\nOther eye: the fellow eye flipped left-right. Outside the box: original image. "
             + ("Vessels: CLAHE, Frangi, top 12% in the field of view." if big else "Pupil: fixed dark threshold, an estimate.")
             + "\nNo twins in open 2D data, so no co-twin column. Local research copy of public data; no redistribution.",
             ha="center", fontsize=9.5)
    fig.subplots_adjust(left=.04, right=.99, top=.86, bottom=.2, hspace=.08, wspace=.08)
    save(fig, os.path.join(FIG_DIR, f"donors_{ds.lower()}_real"))


def main():
    L.guard()
    os.makedirs(FIG_DIR, exist_ok=True)
    stored = json.load(open(os.path.join(sel.OUT_DIR, "brats_baseline_pairs.json"), encoding="utf-8"))
    selections = {"GRAPE": json.load(open(L.EXPECTED["selection_grape"][0], encoding="utf-8")),
                  "iris": json.load(open(L.EXPECTED["selection_iris"][0], encoding="utf-8"))}
    u_grape = sel.grape_universe(pd.read_csv(sel.GRAPE_META))
    utiris, _ = sel.utiris_table()
    u_iris = sel.iris_universe(sel.upol_table(), utiris)
    jobs = [("GRAPE", selections["GRAPE"], u_grape, {n: os.path.join(L.GRAPE_IMG, n) for n in u_grape.image}, True)]
    for ds, root in (("UPOL", sel.UPOL), ("UTIRIS", sel.UTIRIS)):
        u = u_iris[u_iris.dataset == ds]
        jobs.append((ds, selections["iris"], u, {n: os.path.join(root, n) for n in u.image}, False))
    only = [a for a in sys.argv[1:] if a in ("GRAPE", "UPOL", "UTIRIS")]
    for ds, s, u, paths, rgb in jobs:
        if only and ds not in only:
            continue
        prs = [p for p in s["pairs"] if p["dataset"] == ds]
        mine = [r for r in stored if r["dataset"] == ds]
        assert len(mine) == len(prs) and all(r["rung"] == p["rung"] for r, p in zip(mine, prs))
        prim = [(p, r["strangers"]) for p, r in zip(prs, mine) if p["rung"] == PRIMARY[ds]]
        case, cols = visual_case(ds, prs, u, alignment_rule(ds, paths))
        need = {n for p, _ in prim for n in [p["anchor"], p["candidate"], p["reference"][0], *p["mean_members"]]}
        need |= {n for n in cols.values() if n} | set(case["mean_members"]) | {case["anchor"], case["candidate"]}
        imgs = L.Images({n: paths[n] for n in sorted(need)}, rgb, v=ds == "GRAPE")
        res = path_scores([p for p, _ in prim], imgs, [r for _, r in prim])
        mixing_figure(ds, case, imgs, res, len(set(res["people"])))
        visual_figure(ds, case, cols, imgs)
        detail_card(ds, case, cols, imgs)
        donor_strip(ds, case, cols, imgs)
        print(f"{ds}: figures written ({len(prim)} pairs on the path)", flush=True)


if __name__ == "__main__":
    main()
