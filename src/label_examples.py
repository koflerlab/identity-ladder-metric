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
    fills = [truth, acq[(0, 0, "L", "S0", 1)], acq[(0, 1, "L", "S0", 0)], stranger, pop]
    return [np.where(ll.MASK, v, truth) for v in fills]


def main():
    fig, axes = plt.subplots(6, 5, figsize=(12, 14))
    names = ["True image / oracle", "Same-sitting repeat", "MZ co-twin", "Stranger tissue", "Population mean"]
    for i, prior in enumerate((0, .25, 1)):
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
    fig.suptitle("What does the average fill look like? A fixed synthetic example", fontsize=16, y=.98)
    fig.text(.5, .018, "Seed 101; BASE; first target (family 0, twin 0), chosen by index rather than score.\n"
             "Orange box: fill / evaluation region. Same grayscale (60-140) for all intensities.\n"
             "Outside the box: original target. Segmentation: Gaussian smoothing + threshold, not FastSurfer or real MRI.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.935, bottom=.085, left=.075, right=.99, hspace=.08, wspace=.05)
    dest = ROOT / "results" / "figures"
    dest.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"label_fixed_example.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
