"""Identity ladder diagram: every rung placed at its model variance ratio rho = v_rung / v_matched_strangers.

Two parameter sets from toy_ladder (genetic share 0.54 and 0.19) show that the identity-free population mean sits at
rho = 1/2 in both, while the MZ co-twin moves across it. Rungs marked * change the acquisition domain (scanner or
protocol), so their strangers are matched on the same domain pair. Output: results/figures/identity_ladder.{png,pdf}.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from toy_ladder import BASE, LOW_H, matched_reference, rung_variances  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RUNGS = [("L0_same_sitting", "same person, same session"), ("L0_retest", "same person, another day"),
         ("L2_years", "same person, years later"), ("L1_device", "same person, other scanner *"),
         ("mirror_same_sitting", "same person, other hemisphere / other eye"), ("MZ", "MZ co-twin (same genome)"),
         ("population_mean_fill", "population mean (no identity)"), ("DZ", "DZ co-twin or sibling (half the genome)"),
         ("L4", "unrelated person")]
INK, MUTED, BLUE, ORANGE, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e4e3df"


def rhos(s):
    v = rung_variances(s)
    return {k: v[k] / v[matched_reference(k)] for k, _ in RUNGS}


def main():
    sets = [("genetic share 0.54", rhos(BASE)), ("genetic share 0.19", rhos(LOW_H))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (name, r) in zip(axes, sets):
        order = sorted(RUNGS, key=lambda kv: r[kv[0]])
        for i, (k, lab) in enumerate(order):
            mean = k == "population_mean_fill"
            col = ORANGE if mean else BLUE
            ax.plot([0, r[k]], [i, i], color=GRID, lw=1, zorder=1)
            ax.scatter(r[k], i, s=60, marker="D" if mean else "o", facecolor="white" if mean else col, edgecolor=col,
                       linewidth=1.5, zorder=3)
            ax.text(r[k] + 0.03, i, f"{lab}   {r[k]:.2f}", va="center", fontsize=9, color=ORANGE if mean else INK)
        ax.axvline(0.5, color=ORANGE, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(0, 2.1)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlabel(r"variance ratio $\rho$ against matched strangers (0 = identical, 1 = stranger)")
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Identity ladder: expected distance of each pair, Gaussian variance-component model", fontsize=12, x=0.02,
                 ha="left")
    fig.text(0.02, -0.04, "* domain shift: strangers are matched on the same scanner pair. Squared error gives the population "
             "mean rho = 1/2 whatever the genetic share, so it overtakes the MZ co-twin when the share is below 1/2.",
             fontsize=8.5, color=MUTED)
    out = os.path.join(PROJECT, "results", "figures", "identity_ladder")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=180)
    print({n: {k: round(v, 3) for k, v in r.items()} for n, r in sets})


if __name__ == "__main__":
    main()
