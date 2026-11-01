"""Figure 1 of the ISBI 2027 draft from results/toy_ladder.json (synthetic data only).

(a) closed form E c = P(F_{D,D} > rho) for D = 8 and 32 with the simulated rungs; (b) mean c of inpainting fills on
synthetic 2D tissue scored with the official BraTS code, RMSE vs SSIM, normal and low contrast.
Palette: categorical slots 1-2 of the dataviz reference palette, validated (CVD dE 24.7, contrast >= 3:1).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from toy_ladder import psi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "toy_ladder.json")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
RUNG_LABEL = {"L0_same_sitting": "same sitting", "L0_retest": "other day", "L1_device": "other device", "L2_years": "years later",
              "mirror_same_sitting": "other side", "MZ": "MZ co-twin", "DZ": "DZ co-twin"}


# paper: two-column IEEE width, caption in LaTeX; display: large standalone copy with its provenance caption
STYLES = {"paper": dict(size=(7.0, 2.25), wspace=0.42, font=7, small=6, title=7, legend=5.6, anchor=1.06, lw=1.2, dot=14, mark=16, dpi=200),
          "display": dict(size=(13.6, 5.6), wspace=0.47, font=11, small=10, title=13, legend=10, anchor=1.10, lw=2, dot=35, mark=44, dpi=180)}


def main():
    r = json.load(open(RES, encoding="utf-8"))
    draw(r, "paper", os.path.join(HERE, "overleaf"))
    draw(r, "display", os.path.join(HERE, "..", "..", "results", "figures"))


def draw(r, style, dest):
    st = STYLES[style]
    plt.rcParams["font.size"] = st["font"]
    fig, (a, b) = plt.subplots(1, 2, figsize=st["size"], gridspec_kw={"width_ratios": [1, 1.15], "wspace": st["wspace"]})
    if style == "display":
        fig.subplots_adjust(top=.77, bottom=.20, left=.07, right=.985)
    rho = np.linspace(0, 1.3, 300)
    for D, col in ((8, BLUE), (32, ORANGE)):
        a.plot(rho, psi(rho, D), color=col, lw=st["lw"], label=f"theory, D = {D}", zorder=2)
        for name in ("base", "low_h"):
            v = r["vectors"][f"{name}_D{D}"]
            pts = [(x["rho_theory"], x["c_sim"]) for k, x in v.items() if k in RUNG_LABEL]
            a.scatter(*zip(*pts), s=st["dot"], facecolor=col, edgecolor="white", linewidth=0.6, zorder=3,
                      label=f"simulation, D = {D}" if name == "base" else None)
    a.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    a.text(0.53, 0.43, "population-mean limit\nunder the Gaussian model", color=MUTED, fontsize=st["small"], va="top")
    a.axhline(0.5, color=GRID, lw=0.6, zorder=0)
    a.text(1.28, 0.52, "strangers", color=MUTED, fontsize=st["small"], ha="right", va="bottom")
    a.set_xlim(0, 1.3); a.set_ylim(0, 1.03)
    a.set_xlabel(r"variance ratio $\rho_r = v_r / v_{\mathrm{ref}}$"); a.set_ylabel("mean calibrated score $c$")
    a.set_title("(a) Closed form vs simulation", fontsize=st["title"], loc="left", color=INK)
    a.legend(frameon=False, fontsize=st["legend"], loc="lower left", handlelength=1.5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    cfg_hi, cfg_lo = r["donor_swap"]["low_h_contrast1_mask32"], r["donor_swap"]["low_h_contrast0.1_mask32"]
    rows = [("L0_same_sitting", "same sitting"), ("L0_retest", "other day"), ("mirror_same_sitting", "other side"),
            ("MZ", "MZ co-twin"), ("DZ", "DZ co-twin"), ("population_mean", "population mean"),
            ("population_mean_sd_restored", "mean, SD restored")]
    y = np.arange(len(rows))[::-1]
    for metric, col, dy in (("rmse", BLUE, 0.2), ("ssim", ORANGE, -0.07)):
        for cfg, filled, lab in ((cfg_hi, True, "contrast 1"), (cfg_lo, False, "contrast 0.1")):
            xs = [cfg[metric].get(k, {"mean_c": np.nan})["mean_c"] for k, _ in rows]
            b.scatter(xs, y + dy - (0 if filled else 0.13), s=st["mark"], marker="o" if filled else "D", facecolor=col if filled else "white", edgecolor=col,
                      linewidth=0.9, zorder=3, label=f"{metric.upper()}, {lab}")
    b.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    b.text(0.51, -0.62, "stranger", color=MUTED, fontsize=st["small"], va="center")
    b.set_ylim(-0.85, len(rows) - 0.5)
    b.set_yticks(y); b.set_yticklabels([lab for _, lab in rows])
    b.set_xlim(-0.02, 1.02); b.set_xlabel("mean $c$ of the fill against stranger fills")
    b.set_title("(b) Synthetic tissue, corrected intensity model", fontsize=st["title"], loc="left", color=INK)
    b.grid(axis="x", color=GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=st["legend"], loc="lower left", bbox_to_anchor=(0.0, st["anchor"]), ncol=2, handletextpad=0.3, columnspacing=0.8)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    if style == "display":
        fig.text(.5, .04, f"Rebuilt from the current saved results/toy_ladder.json; seed {r['seed']}. Panel (b): LOW_H; 120 targets; 32 x 32 mask.\n"
                 "RMSE and SSIM are highlighted here; all nine package outputs are in the companion figure. Single-seed estimates, no uncertainty bars.",
                 ha="center", fontsize=st["small"])
    fig.savefig(os.path.join(dest, "fig_ladder.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(dest, "fig_ladder.png"), bbox_inches="tight", dpi=st["dpi"])
    plt.close(fig)


if __name__ == "__main__":
    main()
