"""Figures of the fundus donor ladder (docs/PROTOCOL_donor_ladder_fundus.md). Pre-QC.

Public: results/figures/fig_donor_ladder_fundus.{pdf,png}, aggregates from results/donor_ladder_fundus.json.
Private: the example strip with real images, written to the private run folder only.
Palette: categorical slots 1-6 of the dataviz reference palette in fixed order (adjacent pairs validated);
every series is direct-labelled and has its own marker because slots 3-5 are below 3:1 contrast.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_fundus as F  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"population": ("#2a78d6", "o", "population, same side"),
          "population_both": ("#eb6834", "s", "population, both sides"),
          "cohort": ("#1baf7a", "^", "cohort (age, sex; GRAPE: + device)"),
          "knn": ("#eda100", "v", "kNN by context"),
          "fellow_eye": ("#e87ba4", "D", "own other eye"),
          "other_visit": ("#008300", "P", "own eye, other visit")}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def curves(res, region="random_patches"):
    names = [d for d in ("PAPILA", "GRAPE") if d in res]
    fig, axes = plt.subplots(len(names), 2, figsize=(7.0, 2.3 * len(names)), squeeze=False,
                             gridspec_kw={"wspace": 0.3, "hspace": 0.55})
    handles = []
    for row, name in enumerate(names):
        reg = res[name]["regions"][region]
        for col, (key, ci_key, ylab) in enumerate((("err_over_V", "ci95", "error / V"),
                                                     ("spread_over_V", "spread_ci95", "spread from μ / V"))):
            ax = axes[row, col]
            for c, (colour, marker, text) in SERIES.items():
                d = reg["cohorts"].get(c)
                if not d:
                    continue
                ks = sorted((k for k in d if k.isdigit()), key=int)
                kk = np.array([int(k) for k in ks])
                y = np.array([d[k][key] for k in ks])
                lo = y - np.array([d[k][ci_key][0] for k in ks])
                hi = np.array([d[k][ci_key][1] for k in ks]) - y
                ax.errorbar(kk, y, yerr=[lo, hi], color=colour, marker=marker, ms=3.5, lw=1.2 if len(kk) > 1 else 0,
                            elinewidth=0.6, capsize=1.5, mec="white", mew=0.5)
                if col == 0 and "fit" in d:
                    kf = np.geomspace(1, 64, 60)
                    ax.plot(kf, d["fit"]["one_minus_R2"] * (1 + 1 / kf), color=colour, lw=0.6, ls=(0, (2, 2)))
                if "all" in d:
                    ax.axhline(d["all"][key], color=colour, lw=0.5, ls=(0, (1, 2)))
                if col == 0 and text not in [h.get_label() for h in handles]:
                    handles.append(plt.Line2D([], [], color=colour, marker=marker, ms=3.5, lw=1.2 if len(kk) > 1 else 0,
                                              mec="white", mew=0.5, label=text))
            ax.set_xscale("log", base=2)
            ax.set_xticks(F.KS, [str(k) for k in F.KS])
            ax.set_xlim(0.8, 90)
            ax.set_xlabel("donors averaged, k")
            ax.set_ylabel(ylab)
            ax.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            n = res[name]["counts"]["n_units"]
            ax.set_title(f"{name} ({n} eyes): {'error of the fill' if col == 0 else 'spread of the fill'}",
                         loc="left", fontsize=7, color=INK)
            if col == 0:
                ax.axhline(2, color=MUTED, lw=0.5)
                ax.annotate("theory: one stranger = 2", (0.85, 2.02), fontsize=5.6, color=MUTED, va="bottom")
    handles += [plt.Line2D([], [], color=MUTED, lw=0.6, ls=(0, (2, 2)), label="fit (1 − R²)(1 + 1/k)"),
                plt.Line2D([], [], color=MUTED, lw=0.5, ls=(0, (1, 2)), label="all donors")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.06))
    dest = os.path.join(ROOT, "results", "figures")
    os.makedirs(dest, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dest, f"fig_donor_ladder_fundus.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def examples(name="PAPILA", k=16):
    """Private: both eyes of the first person in ID order, metric box filled from each cohort."""
    t = F.papila_table() if name == "PAPILA" else F.grape_table()
    z = np.load(os.path.join(F.PRIVATE, f"{name}_canonical.npz"))
    img, fov, ok, dr = z["img"], z["fov"], z["ok"], z["disc_right"]
    u = F.units(t, ok, dr)
    both = u.groupby("person").filter(lambda g: len(g) == 2).sort_values(["person", "side"])
    pid = both.person.iloc[0]
    y, x, s = F.METRIC
    box = (slice(y, y + s), slice(x, x + s))
    rng = np.random.default_rng(F.SEED)
    cols = ["target", "population mean", "population, both sides", f"cohort, k={k}", f"kNN, k={k}",
            "one stranger", "own other eye"]
    fig, axes = plt.subplots(2, len(cols), figsize=(7.0, 2.4), gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    for row, (_, e) in enumerate(both[both.person == pid].iterrows()):
        i = e.idx
        same = u[(u.person != pid) & (u.side == e.side)].idx.to_numpy()
        anyside = u[u.person != pid].idx.to_numpy()
        coh = u[(u.person != pid) & (u.side == e.side) & (u.sex.astype(str) == str(e.sex))
                & ((u.age - e.age).abs() <= 5)].idx.to_numpy()
        ry, rx = slice(y - F.RING, y + s + F.RING), slice(x - F.RING, x + s + F.RING)
        ring = F.ring_mask(s)
        ctx = img[:, ry, rx][:, ring]
        knn = same[np.argsort(((ctx[same] - ctx[i]) ** 2).sum(1))][:k]
        fellow = u[(u.person == pid) & (u.side != e.side)].idx.to_numpy()
        fills = [img[i][box], img[same].mean(0)[box], img[anyside].mean(0)[box],
                 img[rng.permutation(coh)[:k]].mean(0)[box] if coh.size else np.zeros((s, s)),
                 img[knn].mean(0)[box], img[rng.choice(same)][box], img[fellow[0]][box]]
        for j, (f, title) in enumerate(zip(fills, cols)):
            full = img[i].copy()
            full[box] = f
            ax = axes[row, j]
            ax.imshow(np.where(fov[i], full, np.nan), cmap="gray", vmin=-2.5, vmax=2.5, interpolation="nearest")
            ax.add_patch(plt.Rectangle((x - .5, y - .5), s, s, fill=False, ec="#eb6834", lw=0.6))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=6, color=INK)
            if j == 0:
                ax.set_ylabel("right eye" if e.side == "R" else "left eye (reflected)", fontsize=6, color=INK)
    fig.savefig(os.path.join(F.PRIVATE, f"examples_{name}.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    res = json.load(open(os.path.join(ROOT, "results", "donor_ladder_fundus.json"), encoding="utf-8"))
    curves(res)
    for name in ("PAPILA", "GRAPE"):
        if name in res:
            examples(name)


if __name__ == "__main__":
    main()
