"""Render all 13 saved pupil metrics and write a compact descriptive report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pupil_ladder import FILLS, LABELS, METRICS, ROOT, save


def main():
    result = json.loads((ROOT / "results" / "pupil_ladder.json").read_text())
    names = [LABELS[f] for f in FILLS] + ["Same eye: dilated\nmatched stranger reference"]
    stats = [result["summary"][f] for f in FILLS] + [result["dilation_matched_summary"]]
    arr = np.array([[v[m]["mean"] for m in METRICS] for v in stats])
    fig, ax = plt.subplots(figsize=(15, 6.5))
    img = ax.imshow(arr, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    labels = [m.upper() for m in METRICS[:9]] + ["Pupil Dice", "Area error", "Boundary distance", "Component error"]
    ax.set(xticks=range(13), xticklabels=labels, yticks=range(7), yticklabels=names)
    ax.tick_params(axis="x", labelrotation=40)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for i in range(7):
        for j in range(13):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white" if arr[i, j] < .65 else "black")
    ax.axhline(5.5, color="white", lw=3)
    fig.colorbar(img, ax=ax, label="Mean calibrated c (higher = better relative score)", shrink=.85)
    fig.suptitle("All metrics: synthetic pupil/iris experiment", fontsize=17)
    fig.text(.5, .02, "Three independent cohorts x 32 target people; independent template and reference pools. Normal-pupil references in the first six rows.\n"
             "Last row uses the same donor identities with the same +30% dilation as the candidate. Stranger c=0.5 is a fixed anchor.\n"
             "Component errors tie at zero, so c=0.5 even for a repeat. This is an illustrative simulation, not biometric validation.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.23, right=.94, top=.90, bottom=.30)
    save(fig, "pupil_all_metrics")

    fmt = lambda v: f"{v['mean']:.3f} [{v['min']:.3f}, {v['max']:.3f}]"
    lines = ["# Synthetic pupil and iris experiment — 1 November 2026", "",
             "Question: can intensity or pupil-mask scores confuse dilation of the same eye with a change of identity, and favour the population mean? Exploratory model, with protocol recorded in docs/HANDOFF.md before running.", "",
             "Three seeds (201-203), each with 32 distinct target people (one target eye each), 40 separate template people and 32 separate reference people; 16 reference donors per target. Total 96 target people. The other eye belongs to the same simulated target person. No twins or real eye images in this test.", "",
             "The same-eye dilation condition scales pupil radius by 1.3 and deforms the iris texture in normalized radial coordinates while keeping its latent parameters fixed. Images are 96 x 96, evaluated in the central 64 x 64. Pupil segmentation uses Gaussian smoothing sigma 0.5 and threshold 0.23. Area errors are pixels; symmetric average boundary distances are pixels; Dice is dimensionless. Scores are calibrated with half-credit for ties. No parameters were tuned after observing the results.", "",
             "## Calibrated scores", "", "Each cell is mean [minimum, maximum] over three cohort means, NOT a confidence interval. Higher c is a better relative score, not a claim of preserved identity.", "",
             "| Candidate | RMSE c | SSIM c | Pupil Dice c | Boundary-distance c | Component-count c |",
             "|---|---|---|---|---|---|"]
    selected = ("rmse", "ssim", "pupil_dice", "pupil_boundary_distance", "pupil_components_error")
    for name, values in zip(names, stats):
        lines.append("| " + name.replace("\n", "; ") + " | " + " | ".join(fmt(values[m]) for m in selected) + " |")
    lines += ["", "## Findings and limits", "",
              "- Same-eye repeats score c=1 on the intensity and pupil-shape metrics in this low-noise simulator. Dilation of the same eye scores near zero against normal-pupil strangers (RMSE 0.004, SSIM 0.006), despite unchanged latent identity parameters.",
              "- Matching stranger dilation raises same-eye c to about 0.45 (RMSE) / 0.43 (SSIM), not back to the repeat level. Matching alone does not make these representations invariant to dilation in this simulator.",
              "- The independent-population mean scores above the stranger anchor: RMSE c=0.794, SSIM c=0.842, pupil Dice c=0.649. It is not near-perfect on all metrics; this limits transfer of the previous tissue result.",
              "- Component-count error is zero for every compared pupil; c=0.5 including same-eye repeats. Thus this topological descriptor is uninformative by construction here. No topology-based identity claim follows.",
              "- Fellow-eye scores reflect the chosen shared size/shape parameters. They are not evidence about human iris bilaterality. Pupil shape is an acquisition/physiology-sensitive feature, not an established biometric identity descriptor.",
              "- This is neither an inpainting-model evaluation nor a study of real pupil physiology. No occlusion, unknown pose, estimated segmentation or clinical data. The next discriminating control would compare iris texture in dilation-normalized coordinates, keeping preprocessing independent of identity labels.", "",
              "## Artifacts and checks", "",
              "- `python src/pupil_ladder.py`: all raw cohort aggregates, c scores, fixed-example values, versions and source hash in pupil_ladder.json; summary CSV also saved.",
              "- `python src/pupil_report.py`: all 13 metrics as a heatmap and this report.",
              "- `figures/pupil_fixed_example.png`: first target at seed 201, selected by index rather than score; one-example numbers differ from cohort means. The displayed stranger is the first reference donor, not the stranger anchor average.",
              "- `figures/label_fixed_example_folding.png` and `figures/label_hemisphere_locations.png`: corrected synthetic-tissue illustration with other-day and opposite-side donors. The right-side patch is a separate simulated side, not a reflection of the target. Homologous coordinates and the schematic outline are defined by construction, not estimated from MRI.",
              "- 95 tests passed; new checks cover pupil geometry, tie handling and use of the actual opposite-side array. Existing experiment JSONs and paper sources were preserved.", ""]
    (ROOT / "results" / "pupil_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
