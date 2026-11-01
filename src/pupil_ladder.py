"""Exploratory synthetic pupil/iris test, not an anatomical or biometric validation.

Independent target, template and reference pools. A dilation changes acquisition
geometry while retaining the eye's latent iris texture and pupil-shape parameters.
All nine BraTS intensity metrics and four pupil-mask metrics are evaluated.
"""
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label
from scipy.special import expit
import torch

from toy_ladder import panel

ROOT = Path(__file__).resolve().parents[1]
H = 96
Y, X = np.mgrid[:H, :H]
R = np.hypot(X - 47.5, Y - 47.5)
THETA = np.arctan2(Y - 47.5, X - 47.5)
MASK = np.zeros((H, H), dtype=bool)
MASK[16:80, 16:80] = True
INTENSITY = ("rmse", "mse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim")
SHAPE = ("pupil_dice", "pupil_area_error", "pupil_boundary_distance", "pupil_components_error")
METRICS = INTENSITY + SHAPE
HIGHER = {"psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim", "pupil_dice"}
FILLS = ("same_eye_repeat", "same_eye_dilated", "other_eye", "population_mean", "blurred_stranger", "stranger")
LABELS = {"truth": "Target eye", "same_eye_repeat": "Same eye: repeat", "same_eye_dilated": "Same eye: dilated",
          "other_eye": "Other eye", "population_mean": "Population mean", "blurred_stranger": "Blurred stranger", "stranger": "Stranger eye"}


def people(n, rng):
    out = []
    for _ in range(n):
        radius = rng.normal(18.5, 1.3)
        common_shape = rng.normal(0, .018, 4)
        eyes = []
        for _ in range(2):
            eyes.append({"radius": float(np.clip(radius + rng.normal(0, .6), 16, 21)),
                         "shape": common_shape + rng.normal(0, .018, 4),
                         "phase": rng.uniform(-np.pi, np.pi, 4),
                         "texture_phase": rng.uniform(-np.pi, np.pi, 8)})
        out.append(eyes)
    return out


def render(eye, rng, dilation=1.):
    radial = eye["radius"] * dilation * (1 + sum(a * np.cos((k + 2) * THETA + phase)
                                                 for k, (a, phase) in enumerate(zip(eye["shape"], eye["phase"]))))
    # Iris coordinates deform with the pupil; latent angular/radial texture remains fixed.
    u = (R - radial) / (40 - radial)
    texture = sum(.025 * np.sin(freq * THETA + eye["texture_phase"][j]) *
                  np.cos((j % 3 + 1) * 2 * np.pi * u + eye["texture_phase"][j + 4])
                  for j, freq in enumerate((13, 19, 27, 37)))
    iris = .49 + .10 * np.cos(2 * np.pi * u) + texture
    image = .9 + expit((40 - R) / .6) * (iris - .9)
    pupil = expit((radial - R) / .5)
    image = image * (1 - pupil) + .06 * pupil
    return np.clip(image + rng.normal(0, .006, image.shape), 0, 1)


def segment(image):
    return (gaussian_filter(image, .5) < .23) & MASK


def shape_scores(pred, target):
    a, b = segment(pred), segment(target)
    border_a = a & ~binary_erosion(a)
    border_b = b & ~binary_erosion(b)
    if not border_a.any() or not border_b.any():
        raise ValueError("Empty pupil boundary; inspect the simulated case")
    assd = (distance_transform_edt(~border_b)[border_a].mean() +
            distance_transform_edt(~border_a)[border_b].mean()) / 2
    return {"pupil_dice": float(2 * (a & b).sum() / (a.sum() + b.sum())),
            "pupil_area_error": float(abs(int(a.sum()) - int(b.sum()))),
            "pupil_boundary_distance": float(assd),
            "pupil_components_error": float(abs(label(a)[1] - label(b)[1]))}


def scores(image, truth):
    return {**{k: float(v) for k, v in panel(image, truth, MASK).items() if k in INTENSITY},
            **shape_scores(image, truth)}


def c_score(value, references, metric):
    ref = np.asarray(references)
    return float(np.mean(ref < value if metric in HIGHER else ref > value) + .5 * np.mean(ref == value))


def run(seed, n_targets=32, n_template=40, n_reference=32, n_strangers=16):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    targets = people(n_targets, streams[0])
    templates = people(n_template, streams[1])
    references = people(n_reference, streams[2])
    rng = streams[3]
    mean_image = np.mean([render(p[0], streams[1]) for p in templates], axis=0)
    normal_refs = [render(p[0], streams[2]) for p in references]
    dilated_refs = [render(p[0], streams[2], 1.3) for p in references]
    values = {f: {m: {"c": [], "raw": []} for m in METRICS} for f in FILLS}
    matched = {m: [] for m in METRICS}
    example = None
    for i, person in enumerate(targets):
        truth = render(person[0], rng)
        ids = rng.choice(n_reference, n_strangers, replace=False)
        ref_scores = [scores(normal_refs[j], truth) for j in ids]
        dilated_scores = [scores(dilated_refs[j], truth) for j in ids]
        images = {"same_eye_repeat": render(person[0], rng), "same_eye_dilated": render(person[0], rng, 1.3),
                  "other_eye": render(person[1], rng), "population_mean": mean_image,
                  "blurred_stranger": gaussian_filter(normal_refs[ids[0]], 2.), "stranger": normal_refs[ids[0]]}
        this_case = {}
        for fill, image in images.items():
            v = scores(image, truth)
            this_case[fill] = {"raw": v, "c": {m: c_score(v[m], [s[m] for s in ref_scores], m) for m in METRICS}}
            for m in METRICS:
                values[fill][m]["raw"].append(v[m] if fill != "stranger" else np.mean([s[m] for s in ref_scores]))
                values[fill][m]["c"].append(this_case[fill]["c"][m] if fill != "stranger" else .5)
        for m in METRICS:
            matched[m].append(c_score(this_case["same_eye_dilated"]["raw"][m], [s[m] for s in dilated_scores], m))
        if i == 0:
            example = {"images": {"truth": truth, **images}, "scores": this_case}
    out = {f: {m: {"mean_c": float(np.mean(v["c"])), "mean_raw": float(np.mean(v["raw"]))}
               for m, v in ms.items()} for f, ms in values.items()}
    return {"seed": seed, "scores": out, "same_eye_dilated_matched_c": {m: float(np.mean(v)) for m, v in matched.items()},
            "first_case_scores": example["scores"]}, example


def plot_example(example):
    order = ("truth", "same_eye_repeat", "same_eye_dilated", "other_eye", "stranger", "population_mean", "blurred_stranger")
    fig, axes = plt.subplots(3, 7, figsize=(17, 8))
    oracle = segment(example["images"]["truth"])
    for j, fill in enumerate(order):
        image = example["images"][fill]
        pred = segment(image)
        axes[0, j].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(LABELS[fill], fontsize=11)
        axes[1, j].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(np.ones(image.shape), cmap="gray", vmin=0, vmax=1)
        axes[2, j].contour(oracle, levels=[.5], colors=["#1588b2"], linewidths=1.8)
        axes[2, j].contour(pred, levels=[.5], colors=["#c15c12"], linewidths=1, linestyles="--")
        if fill != "truth":
            v = example["scores"][fill]
            axes[2, j].set_xlabel(f"RMSE c={v['c']['rmse']:.2f}; SSIM c={v['c']['ssim']:.2f}\n"
                                  f"Pupil Dice={v['raw']['pupil_dice']:.2f}; boundary={v['raw']['pupil_boundary_distance']:.1f}px", fontsize=8)
        for ax in axes[:, j]:
            ax.set(xticks=[], yticks=[])
    for ax, label_text in zip(axes[:, 0], ("Intensity", "Pupil mask", "Contour overlay")):
        ax.set_ylabel(label_text, fontsize=11)
    fig.suptitle("Synthetic eye: changing the pupil is not changing the eye", fontsize=17)
    fig.text(.5, .015, "Fixed example: seed 201, target 0. Blue: target pupil; dashed orange: candidate. Dilation +30%, same latent iris texture.\n"
             "Numbers describe this one example, not cohort means. c uses 16 normal-pupil stranger donors; masked region is central 64 x 64.\n"
             "Illustrative simulator, not human eye data. A pupil outline alone is not an identity measure.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.04, right=.99, top=.90, bottom=.19, hspace=.08, wspace=.06)
    save(fig, "pupil_fixed_example")


def save(fig, name):
    dest = ROOT / "results" / "figures"
    dest.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    out = {"design": {"seeds": [201, 202, 203], "n_targets_per_seed": 32, "n_template_per_seed": 40,
                       "n_reference_per_seed": 32, "n_strangers_per_target": 16, "dilation": 1.3,
                       "pool_separation": "Independent people in target, template and reference pools; one left-eye target per person",
                       "mask_pixels": int(MASK.sum()), "pupil_threshold": .23,
                       "versions": {p: version(p) for p in ("numpy", "scipy", "torch", "torchmetrics", "inpainting")},
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "runs": [],
           "limits": ["Synthetic 2D eyes with illustrative, non-biologically-calibrated parameters.",
                      "Pupil dilation is acquisition/physiology variation, not identity loss.",
                      "Pupil shapes are connected by construction; topology may be uninformative.",
                      "All ranges are min/max over three cohort means, not confidence intervals.",
                      "Same-eye repeat and mean are compared to normal-pupil stranger references; dilation-matched references are also reported.",
                      "Known pupil center, no estimated registration, illumination/occlusion/real iris segmentation not validated."]}
    for seed in out["design"]["seeds"]:
        start = time.perf_counter()
        result, example = run(seed)
        result["runtime_s"] = round(time.perf_counter() - start, 2)
        out["runs"].append(result)
        if seed == 201:
            plot_example(example)
        print(f"seed={seed} complete {result['runtime_s']}s", flush=True)
    def summary(vals):
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    out["summary"] = {f: {m: summary([r["scores"][f][m]["mean_c"] for r in out["runs"]]) for m in METRICS} for f in FILLS}
    out["dilation_matched_summary"] = {m: summary([r["same_eye_dilated_matched_c"][m] for r in out["runs"]]) for m in METRICS}
    (ROOT / "results" / "pupil_ladder.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    rows = [{"fill": f, "metric": m, **v} for f, ms in out["summary"].items() for m, v in ms.items()]
    rows += [{"fill": "same_eye_dilated_matched_reference", "metric": m, **v} for m, v in out["dilation_matched_summary"].items()]
    with (ROOT / "results" / "pupil_ladder_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
