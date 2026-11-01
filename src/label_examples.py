"""Fixed, non-selected synthetic example for the H19 sensitivity experiment.

Seed 101, BASE, family 0 / twin 0 (first MZ target) at all three prior strengths.
Uses the same first stranger as label_ladder.run; no score-based example selection.
The figure shows in-mask substitution with the target unchanged outside the mask.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import label_ladder as ll

ROOT = Path(__file__).resolve().parents[1]


def example(prior):
    rng = np.random.default_rng(101)
    acq, _, _ = ll.people(ll.BASE, 60, rng, prior_scale=prior)
    keys = [(f, t, "L", "S0", 0) for f in range(60) for t in range(2)]
    pop = (np.sum([acq[k] for k in keys], axis=0) - acq[(0, 0, "L", "S0", 0)]
           - acq[(0, 1, "L", "S0", 0)]) / 118
    donor_families = rng.choice(list(range(1, 60)), 16, replace=False)
    stranger = acq[(int(donor_families[0]), int(rng.integers(2)), "L", "S1", 0)]
    truth = acq[(0, 0, "L", "S0", 0)]
    fills = [truth, acq[(0, 0, "L", "S0", 1)], acq[(0, 0, "L", "S1", 0)],
             acq[(0, 0, "R", "S0", 0)], acq[(0, 1, "L", "S0", 0)], stranger, pop]
    return [np.where(ll.MASK, v, truth) for v in fills]


def comparison(priors, filename):
    fig, axes = plt.subplots(2 * len(priors), 7, figsize=(16, 4 * len(priors) + 1.7))
    names = ["True image / oracle", "Same-sitting repeat", "Other day", "Other hemisphere*",
             "MZ co-twin", "Stranger tissue", "Population mean"]
    for i, prior in enumerate(priors):
        images = example(prior)
        for col, im in enumerate(images):
            for row, img in ((2 * i, im), (2 * i + 1, ll.segment(im))):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray", vmin=60 if row % 2 == 0 else 0, vmax=140 if row % 2 == 0 else 1)
                ax.add_patch(Rectangle((15.5, 15.5), 32, 32, fill=False, edgecolor="#e98c28", linewidth=.8))
                ax.set(xticks=[], yticks=[])
                if row == 0:
                    ax.set_title(names[col], fontsize=11)
        axes[2*i, 0].set_ylabel(f"Prior {prior:g}\nIntensity", fontsize=11)
        axes[2*i+1, 0].set_ylabel("Segmentation", fontsize=11)
    fig.suptitle("Synthetic tissue: all donor positions, including the other hemisphere", fontsize=16, y=.98)
    fig.text(.5, .018, "Seed 101; BASE; first target (family 0, twin 0), chosen by index rather than score.\n"
             "Orange box: fill / evaluation region. Same grayscale (60-140) for all intensities.\n"
             "*Other side at corresponding coordinates in this model, already aligned; not a flipped copy of the target.\n"
             "Outside the box: original target. Threshold segmentation; synthetic fields, not real MRI or FastSurfer.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.87 if len(priors) == 1 else .935,
                        bottom=.23 if len(priors) == 1 else .10, left=.055, right=.99, hspace=.08, wspace=.05)
    dest = ROOT / "results" / "figures"
    dest.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"{filename}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def hemisphere_locations():
    acq, _, _ = ll.people(ll.BASE, 60, np.random.default_rng(101), prior_scale=.25)
    left = acq[(0, 0, "L", "S0", 0)]
    right_aligned = acq[(0, 0, "R", "S0", 0)]
    brain = np.zeros((64, 136))
    brain[:, :64] = left
    brain[:, 72:] = right_aligned[:, ::-1]
    y, x = np.mgrid[:64, :136]
    silhouette = ((x - 67.5) / 63) ** 2 + ((y - 31.5) / 31) ** 2 <= 1
    silhouette &= np.abs(x - 67.5) > 4
    brain = np.where(silhouette, brain, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), gridspec_kw={"width_ratios": [2, 1, 1]})
    axes[0].imshow(brain, cmap="gray", vmin=60, vmax=140)
    for x0, color in ((15.5, "#d46c0a"), (87.5, "#1689b2")):
        axes[0].add_patch(Rectangle((x0, 15.5), 32, 32, fill=False, edgecolor=color, linewidth=2))
    axes[0].annotate("", xy=(48, 9), xytext=(88, 9), arrowprops={"arrowstyle": "<->", "color": "#555555"})
    axes[0].set_title("Same position in opposite hemispheres\n(schematic arrangement)")
    for ax, patch, title, color in zip(axes[1:], [left[16:48, 16:48], right_aligned[16:48, 16:48]],
                                      ["Target patch", "Opposite-side patch\nmirrored into target coordinates"],
                                      ["#d46c0a", "#1689b2"]):
        ax.imshow(patch, cmap="gray", vmin=60, vmax=140)
        ax.set_title(title, color=color)
        for spine in ax.spines.values():
            spine.set_color(color); spine.set_linewidth(2)
    for ax in axes:
        ax.set(xticks=[], yticks=[])
    fig.suptitle("Where the other-hemisphere donor comes from", fontsize=16)
    fig.text(.5, .025, "Synthetic fields with shared person components and separate side components; not an anatomical MRI.\n"
             "The outline is illustrative only. Homologous locations are defined by construction; no registration was estimated.\n"
             "Seed 101, BASE, first target, prior 0.25. Both patches are shown on the same intensity scale.", ha="center", fontsize=10)
    fig.subplots_adjust(top=.78, bottom=.22, wspace=.25)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "results" / "figures" / f"label_hemisphere_locations.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    comparison((0, .25, 1), "label_fixed_example")
    comparison((.25,), "label_fixed_example_folding")
    hemisphere_locations()


if __name__ == "__main__":
    main()
