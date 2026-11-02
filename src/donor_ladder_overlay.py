"""Theory, calibrated synthetic and real fundus results of the donor ladder on one plot (pre-QC).

Theory: error/V = (1 - R2)(1 + 1/k) for a rung, 2(1 - r) for one same-person donor; the population rung (R2 = 0)
is parameter-free. Synthetic: Monte Carlo of the same additive model with each dataset's fitted R2 and r and its
number of eyes; kNN is an actual nearest-neighbour search in a finite pool, where context reveals a low-dimensional
latent part of the patch, so its bias at large k comes from the pool, not from a fitted curve. Real: the aggregate
of results/donor_ladder_fundus.json. Synthetic data are generated on the fly and never saved.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_fundus as F  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RUNGS = {"population": ("#2a78d6", "population mean of k strangers"),
         "cohort": ("#1baf7a", "cohort (age, sex)"),
         "knn": ("#eda100", "kNN by context"),
         "fellow_eye": ("#e87ba4", "own mirror side (other eye / hemisphere)"),
         "other_visit": ("#008300", "own, later visit or scan"),
         "sibling": ("#6250d6", "sibling (brain)")}
DATA_MARK = {"PAPILA": "s", "GRAPE": "^", "QTIM": "o"}
BRAIN_ROLE = {"other_hemisphere": "fellow_eye", "repeat": "other_visit"}  # same ladder role as the fundus rungs
LINESTYLE = {"PAPILA": "-", "GRAPE": (0, (4, 2)), "QTIM": (0, (1, 1.5))}
D, Q, SEED = 256, 4, 20261102   # pixels per synthetic patch, latent dimension seen by the context


def simulate(n, r2_cohort, r2_knn, r_pairs, rng, reps=20):
    """Synthetic error/V per rung and k, n eyes, V = 1 per pixel."""
    out = {c: {} for c in RUNGS}  # rungs without data stay empty
    for c in ("population", "cohort"):
        r2 = 0.0 if c == "population" else r2_cohort
        for k in F.KS:
            p = rng.standard_normal((reps, n, D))
            x = np.sqrt(r2) * p + np.sqrt(1 - r2) * rng.standard_normal((reps, n, D))
            f = np.sqrt(r2) * p + np.sqrt(1 - r2) * rng.standard_normal((reps, n, k, D)).mean(2)
            out[c][k] = float(((x - f) ** 2).mean())
    # kNN: patch = sqrt(r2) * M u + sqrt(1 - r2) * w with u in R^Q; neighbours chosen on u in a pool of n eyes
    M = np.linalg.qr(rng.standard_normal((D, Q)))[0] * np.sqrt(D / Q)
    errs = {k: [] for k in F.KS}
    for _ in range(reps):
        u = rng.standard_normal((n, Q))
        x = np.sqrt(r2_knn) * u @ M.T + np.sqrt(1 - r2_knn) * rng.standard_normal((n, D))
        dist = ((u[:, None] - u[None]) ** 2).sum(-1)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, 1)
        for k in F.KS:
            if k < n:
                errs[k].append(((x - x[order[:, :k]].mean(1)) ** 2).mean())
    out["knn"] = {k: float(np.mean(v)) for k, v in errs.items() if v}
    for c, r in r_pairs.items():
        d = rng.standard_normal((reps, n, D))
        y = r * d + np.sqrt(1 - r ** 2) * rng.standard_normal((reps, n, D))
        out[c][1] = float(((d - y) ** 2).mean())
    return out


def main():
    res = json.load(open(os.path.join(ROOT, "results", "donor_ladder_fundus.json"), encoding="utf-8"))
    brain = os.path.join(ROOT, "results", "donor_ladder_brain.json")
    if os.path.exists(brain):
        q = json.load(open(brain, encoding="utf-8"))["QTIM"]
        coh = q["regions"]["random_patches"]["cohorts"]
        q["regions"]["random_patches"]["cohorts"] = {BRAIN_ROLE.get(c, c): v for c, v in coh.items()}
        q["counts"]["n_units"] = q["counts"]["n_targets"]
        res["QTIM"] = q
    rng = np.random.default_rng(SEED)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    kk = np.geomspace(1, 64, 80)
    ax.plot(kk, 1 + 1 / kk, color=RUNGS["population"][0], lw=1.4, zorder=1)
    ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.axhline(2, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.annotate("population mean", (20, 1.0), xytext=(0, -7), textcoords="offset points", fontsize=5.6, color=MUTED)
    ax.annotate("one stranger", (1.05, 2.0), xytext=(0, 3), textcoords="offset points", fontsize=5.6, color=MUTED)
    offset = {"PAPILA": 0.92, "GRAPE": 1.0, "QTIM": 1.08}
    for name, mark in DATA_MARK.items():
        if name not in res:
            continue
        reg = res[name]["regions"]["random_patches"]["cohorts"]
        r2c, r2k = reg["cohort"]["fit"]["R2"], reg["knn"]["fit"]["R2"]
        knn1 = 1 - reg["knn"]["1"]["err_over_V"] / 2  # same-pool share seen at k = 1
        pairs = {c: reg[c]["r_implied"] for c in ("fellow_eye", "other_visit", "sibling") if c in reg and "r_implied" in reg[c]}
        for c in ("fellow_eye", "other_visit", "sibling"):
            if c in reg and "r_implied" not in reg[c]:
                pairs[c] = 1 - reg[c]["1"]["err_over_V"] / 2
                reg[c]["r_implied"] = pairs[c]
        syn = simulate(res[name]["counts"]["n_units"], r2c, knn1, pairs, rng)
        ls = LINESTYLE[name]
        ax.plot(kk, (1 - r2c) * (1 + 1 / kk), color=RUNGS["cohort"][0], lw=0.7, ls=ls, zorder=1)
        for c, (colour, _) in RUNGS.items():
            if c not in reg:
                continue
            ks = sorted((int(k) for k in reg[c] if k.isdigit()))
            ax.scatter(np.array(ks) * offset[name], [reg[c][str(k)]["err_over_V"] for k in ks], marker=mark, s=16,
                       color=colour, edgecolor="white", lw=0.5, zorder=3)
            sk = sorted(syn[c])
            ax.scatter(np.array(sk) * offset[name], [syn[c][k] for k in sk], marker=mark, s=16,
                       facecolor="none", edgecolor=colour, lw=0.7, zorder=2)
            if c in pairs:
                ax.hlines(2 * (1 - pairs[c]), 0.75, 1.35, color=colour, lw=0.8, ls=ls, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(F.KS, [str(k) for k in F.KS])
    ax.set_xlim(0.7, 90)
    ax.set_ylim(0.15, 2.3)
    ax.set_xlabel("donors averaged, k")
    ax.set_ylabel("error of the fill / V")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    colour_leg = [Line2D([], [], color=c, lw=0, marker="o", ms=4, label=t) for c, t in RUNGS.values()]
    source_leg = [Line2D([], [], color=MUTED, lw=1.2, label="theory, 1 + 1/k (no fit)"),
                  Line2D([], [], color=MUTED, lw=0.7, label="theory, fitted R² or r (PAPILA)"),
                  Line2D([], [], color=MUTED, lw=0.7, ls=(0, (4, 2)), label="theory, fitted R² or r (GRAPE)"),
                  Line2D([], [], color=MUTED, lw=0.7, ls=(0, (1, 1.5)), label="theory, fitted R² or r (QTIM)"),
                  Line2D([], [], color=MUTED, lw=0, marker="s", ms=4, label="PAPILA fundus, real"),
                  Line2D([], [], color=MUTED, lw=0, marker="^", ms=4, label="GRAPE fundus, real"),
                  Line2D([], [], color=MUTED, lw=0, marker="o", ms=4, label="QTIM brain MRI, real"),
                  Line2D([], [], color=MUTED, lw=0, marker="s", ms=4, mfc="none", label="synthetic: same shape, open")]
    l1 = ax.legend(handles=colour_leg, loc="upper right", frameon=False, fontsize=5.4, title="rung", title_fontsize=5.6)
    ax.add_artist(l1)
    ax.legend(handles=source_leg, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=5.4)
    dest = os.path.join(ROOT, "results", "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dest, f"fig_donor_ladder_overlay.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
