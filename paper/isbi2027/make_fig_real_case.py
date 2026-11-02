"""Real-MRI case figure of the ISBI 2027 draft: one QTIM target, the frozen parietal box filled from each donor rung.

Data: QTIM (OpenNeuro ds004169, CC0), session-1 scans, N4 and affine ANTs registration to MNI 2009a symmetric 1 mm
(real_cases.prepare); pre-QC, registration visual QC not met. Intensities are z-scored inside each scan's own brain
mask, as in src/donor_ladder_brain.py, and the cubes come from that experiment's private cache. The target is chosen
mechanically: the first session-1 scan in participant order whose parietal box is valid for the target, a sibling and
a repeat scan. Numbers under each column: this person's error / error of the population mean, and the same ratio
averaged over all targets (results/donor_ladder_brain.json). Prints counts and ratios only, never IDs.
Style: make_figures.py (two-column IEEE width, Times 7 pt, orange box).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import donor_ladder_brain as B  # noqa: E402
from make_figures import GRID, INK, MUTED, ORANGE  # noqa: E402  (also applies the paper rcParams)

CROP = (slice(30, 170), slice(20, 210))   # axial crop of the 197 x 233 grid, whole brain at the box level
DISPLAY = (-2.5, 2.5)                     # z-score window
# column key, title, group-mean entry of results/donor_ladder_brain.json; closest donor first, averages last
COLUMNS = [("target", "Target", None),
           ("repeat", "Repeat scan", ("repeat", "1")),
           ("other_hemisphere", "Mirror side", ("other_hemisphere", "1")),
           ("sibling", "Sibling", ("sibling", "1")),
           ("stranger", "Stranger", ("population", "1")),
           ("knn", "16 look-alikes", ("knn", "16")),
           ("cohort", "Same sex and age", ("cohort", "all")),
           ("population", "Dataset mean", ("population", "all"))]


def case():
    """The target slice and every fill of its parietal box (arrays only; no IDs leave this function)."""
    t = B.scans()
    pos, _, _ = B.positions(np.random.default_rng(B.SEED))
    pi = [p[0] for p in pos].index("parietal")
    _, start, size = pos[pi]
    with np.load(os.path.join(B.PRIVATE, "patches.npz")) as z:
        X_all, M_all, C_all, V_all = (z[f"{k}{pi}"] for k in ("cube", "mcube", "shell", "valid"))
    tg = t[t.session == "ses-01"].reset_index()
    ix = tg["index"].to_numpy()
    X, M, C, valid = X_all[ix], M_all[ix], C_all[ix].astype(float), V_all[ix]
    fam, age, sex = tg.family.to_numpy(), tg.age.to_numpy(), tg.sex.to_numpy()
    rep = {r.person: r.Index for r in t[t.session == "ses-02"].itertuples()}
    cands = []
    for i in np.flatnonzero(valid):
        sib = [j for j in np.flatnonzero((fam == fam[i]) & valid) if j != i and age[j] != age[i]]
        r = rep.get(tg.person.iloc[i])
        if sib and r is not None and V_all[r]:
            cands.append((i, sib[0], r))
    i, js, jr = cands[0]
    others = np.flatnonzero((fam != fam[i]) & valid)
    coh = others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]
    nn = others[np.argsort(((C[others] - C[i]) ** 2).sum(1))]
    fills = {"target": X[i], "repeat": X_all[jr], "other_hemisphere": M[i], "sibling": X[js],
             "stranger": X[np.random.default_rng(B.SEED).choice(others)], "knn": X[nn[:16]].mean(0),
             "cohort": X[coh].mean(0), "population": X[others].mean(0)}
    # pool sizes over all valid targets, for the caption
    n_pop, n_coh = [], []
    for k in np.flatnonzero(valid):
        o = np.flatnonzero((fam != fam[k]) & valid)
        n_pop.append(len(o))
        n_coh.append(int(((sex[o] == sex[k]) & (np.abs(age[o] - age[k]) <= 2)).sum()))
    row = t.loc[ix[i]]
    vol = nib.load(row.path.replace("_brain.nii.gz", ".nii.gz")).get_fdata(dtype=np.float32)
    mask = nib.load(row.path).get_fdata(dtype=np.float32) > 0.5
    zv = np.where(mask, (vol - vol[mask].mean()) / (vol[mask].std() + 1e-6), DISPLAY[0])  # outside brain: black
    pools = {"n_candidates": len(cands), "population_median": float(np.median(n_pop)),
             "population_range": [int(min(n_pop)), int(max(n_pop))], "cohort_median": float(np.median(n_coh)),
             "cohort_iqr": [float(np.percentile(n_coh, 25)), float(np.percentile(n_coh, 75))],
             "this_population": int(len(others)), "this_cohort": int(len(coh))}
    return zv, start, size, fills, pools


def draw(zv, start, size, fills, group, pools, dest_pdf, dest_png):
    zc = size[2] // 2
    sl0 = zv[:, :, start[2] + zc]
    sx, sy = slice(start[0], start[0] + size[0]), slice(start[1], start[1] + size[1])
    fig, axes = plt.subplots(2, len(COLUMNS), figsize=(7.0, 2.45),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.06, "height_ratios": [1.36, 1]})
    fig.subplots_adjust(top=0.73, bottom=0.07, left=0.035, right=0.995)
    for j, (key, title, gkey) in enumerate(COLUMNS):
        cube = fills[key].reshape(size)[:, :, zc]
        sl = sl0.copy()
        sl[sx, sy] = cube
        # neurological orientation: subject's left on the left, anterior up
        axes[0, j].imshow(sl[CROP].T, origin="lower", cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1],
                          interpolation="nearest")
        axes[0, j].add_patch(Rectangle((start[0] - CROP[0].start - .5, start[1] - CROP[1].start - .5), size[0],
                                       size[1], fill=False, ec=ORANGE, lw=0.8))
        axes[1, j].imshow(cube.T, origin="lower", cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1],
                          interpolation="nearest")
        axes[0, j].set_title(title, fontsize=7, color=INK, pad=3)
        if gkey is None:
            axes[1, j].set_xlabel("MSE = 0 (itself)", fontsize=6.3, color=INK, labelpad=3)
        else:
            g = group[gkey[0]][gkey[1]]["err_over_V"]
            axes[1, j].set_xlabel(f"MSE = {g:.2f} × mean", fontsize=6.3, color=INK, labelpad=3)
        for ax in axes[:, j]:
            ax.set(xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set(color=INK, lw=0.6)
    axes[0, 0].set_ylabel("Slice", fontsize=7)
    axes[1, 0].set_ylabel("Box, enlarged", fontsize=7)
    # group brackets above the column titles: related anatomy vs averages of other people
    for a, b, text in ((1, 4, "Real anatomy from another scan or person"), (5, 7, "Averages of other people")):
        x0, x1 = axes[0, a].get_position().x0 + 0.004, axes[0, b].get_position().x1 - 0.004
        y = 0.83
        fig.add_artist(plt.Line2D([x0, x0, x1, x1], [y - 0.015, y, y, y - 0.015], color=MUTED, lw=0.6))
        fig.text((x0 + x1) / 2, y + 0.01, text, ha="center", va="bottom", fontsize=7, color=MUTED)
    fig.suptitle("Real brain: pixel error ranks the dataset mean above related anatomy", fontsize=9, y=0.985)
    fig.savefig(dest_png, dpi=250)  # repository copy keeps the title
    fig._suptitle.remove()  # the paper's caption carries the message
    fig.savefig(dest_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def theory(group, dest_pdf, dest_png):
    """The parietal-box numbers of the case figure next to the prediction: (a) averages of k donors against
    (1 - R2)(1 + 1/k); (b) every column of the case figure with its 95% interval (family bootstrap).
    Unit: the target's own individual variation V, i.e. the error of the dataset mean."""
    blue, green, amber = "#2a78d6", "#1baf7a", "#eda100"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.95), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.55})
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.07, right=0.985)
    note = dict(fontsize=5.8, color=MUTED, linespacing=1.15)
    ks = np.array([1, 2, 4, 8, 16, 32, 64])
    kk = np.geomspace(1, 64, 200)
    r2 = group["cohort"]["fit"]["R2"]
    a.plot(kk, 1 + 1 / kk, color=blue, lw=1.3, zorder=2, label="Predicted for k strangers: 1 + 1/k")
    a.plot(kk, (1 - r2) * (1 + 1 / kk), color=green, lw=0.9, ls=(0, (3, 2)), zorder=2,
           label=f"Predicted for same sex and age (R$^2$ = {r2:.2f})")
    for c, col, lab, dx in (("population", blue, "Strangers, measured", 0.97),
                            ("cohort", green, "Same sex and age, measured", 1.03),
                            ("knn", amber, "Look-alikes, measured", 1.0)):
        y = np.array([group[c][str(k)]["err_over_V"] for k in ks])
        lo = np.array([group[c][str(k)]["ci95"][0] for k in ks])
        hi = np.array([group[c][str(k)]["ci95"][1] for k in ks])
        a.errorbar(ks * dx, y, yerr=[y - lo, hi - y], fmt="o", ms=3.2, color=col, mec="white", mew=0.5,
                   elinewidth=0.7, capsize=0, zorder=3, label=lab)
    a.axhline(1, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    a.text(1.1, 0.975, "dataset mean = 1", va="top", **note)
    a.set_xscale("log", base=2)
    a.set_xticks(ks, [str(k) for k in ks])
    a.set_xlim(0.8, 80)
    a.set_ylim(0.85, 2.1)
    a.set_xlabel("Number of people averaged, k")
    a.set_ylabel("Error, relative to the dataset mean")
    a.set_title("(a) The more people averaged, the closer to the blur", fontsize=7, loc="left", pad=22)
    a.legend(frameon=False, fontsize=5.6, loc="upper right", handlelength=1.6)

    # rows of the case figure; predictions where they need no measured input: 2 for one stranger, 1 for the
    # dataset mean, (1 - R2)(1 + 1/n) for the same-sex-and-age mean; related donors give r = 1 - ratio / 2
    n_coh = 51
    rows = [("repeat", "1", "Repeat scan", "#008300", None), ("other_hemisphere", "1", "Mirror side", "#e87ba4", None),
            ("sibling", "1", "Sibling", "#6250d6", None), ("population", "1", "Stranger", blue, 2.0),
            ("knn", "16", "16 look-alikes", amber, None),
            ("cohort", "all", "Same sex and age", green, (1 - r2) * (1 + 1 / n_coh)),
            ("population", "all", "Dataset mean", blue, 1.0)]
    y = np.arange(len(rows))[::-1]
    for yi, (c, k, lab, col, th) in zip(y, rows):
        v = group[c][k]
        e = v["err_over_V"]
        if th is not None:
            b.scatter(th, yi, s=26, marker="D", facecolor="white", edgecolor=col, lw=0.8, zorder=2)
        b.errorbar(e, yi, xerr=[[e - v["ci95"][0]], [v["ci95"][1] - e]], fmt="o", ms=3.6, color=col, mec="white",
                   mew=0.4, elinewidth=0.8, capsize=0, zorder=3)
        shared = f"   shares {100 * (1 - e / 2):.0f}%" if c in ("repeat", "other_hemisphere", "sibling") else ""
        b.text(2.95, yi, f"{e:.2f}{shared}", va="center", ha="right", fontsize=6.2, color=INK)
    b.axvline(1, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    b.axvline(2, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    top = len(rows) - 0.45
    b.text(1.0, top, "1: the blur misses only\nthe target's own variation", ha="center", va="bottom", **note)
    b.text(2.0, top, "2: a stranger adds\ntheir own on top", ha="center", va="bottom", **note)
    b.axhline(2.5, color=GRID, lw=0.6)
    b.set_yticks(y, [r[2] for r in rows])
    b.set_ylim(-0.6, len(rows) - 0.45)
    b.set_xlim(0, 3.0)
    b.set_xticks([0, 0.5, 1, 1.5, 2])
    b.scatter([], [], s=26, marker="D", facecolor="white", edgecolor=MUTED, lw=0.8, label="predicted")
    b.scatter([], [], s=14, color=MUTED, label="measured")
    b.legend(frameon=False, fontsize=5.6, loc="lower left", bbox_to_anchor=(0.37, 0.27), handletextpad=0.2)
    b.set_xlabel("Error, relative to the dataset mean (lower = 'better')")
    b.set_title("(b) Only the same person's scan beats the blur", fontsize=7, loc="left", pad=22)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Why the blur wins: errors on a real brain match the prediction", fontsize=9, y=0.995)
    fig.savefig(dest_png, dpi=250)  # repository copy keeps the title
    fig._suptitle.remove()  # the paper's caption carries the message
    fig.savefig(dest_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    res = json.load(open(os.path.join(ROOT, "results", "donor_ladder_brain.json"), encoding="utf-8"))
    group = res["QTIM"]["regions"]["parietal"]["cohorts"]
    theory(group, os.path.join(HERE, "overleaf", "fig_real_theory.pdf"),
           os.path.join(ROOT, "results", "figures", "fig_real_case_theory.png"))
    if "--theory-only" in sys.argv:
        return
    zv, start, size, fills, pools = case()
    draw(zv, start, size, fills, group, pools, os.path.join(HERE, "overleaf", "fig_real_case.pdf"),
         os.path.join(ROOT, "results", "figures", "fig_real_case_parietal.png"))
    print(json.dumps(pools))


if __name__ == "__main__":
    main()
