"""Shared plotting/scoring for bounded local case atlases."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toy_ladder import panel

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path("C:/Projects/data/projects/identity-ladder-metric/run-2026-11-01-visual-atlas")
METRICS = ("rmse", "mse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim")
HIGHER = {"ssim", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps"}


def score(fill, truth, mask):
    return {m: float(v) for m, v in panel(fill, truth, mask).items() if m in METRICS}


def calibrate(values, references):
    out = {}
    for m, x in values.items():
        ref = np.array([r[m] for r in references])
        out[m] = float(np.mean(ref < x if m in HIGHER else ref > x) + .5 * np.mean(ref == x))
    return out


def aggregate(rows):
    out = {}
    for row in rows:
        for fill, values in row["scores"].items():
            for m, v in values["c"].items():
                out.setdefault(fill, {}).setdefault(m, []).append(v)
    return {f: {m: {"mean_c": float(np.mean(v)), "n_targets": len(v)} for m, v in ms.items()} for f, ms in out.items()}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def gallery(cases, path, title, footer, vmin=0, vmax=1, overlays=False):
    """One row per case; caller supplies identifier-free display labels only."""
    keys = list(cases[0]["images"])
    fig, axes = plt.subplots(len(cases), len(keys), figsize=(2.1*len(keys), 2.15*len(cases)+1.1), squeeze=False)
    for i, case in enumerate(cases):
        for j, key in enumerate(keys):
            ax = axes[i,j]
            im = case["images"][key]
            ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            if overlays and "mask" in case:
                ax.contour(case["mask"], [.5], colors=["#e58c21"], linewidths=.8)
            if i==0: ax.set_title(key, fontsize=10)
            if j==0: ax.set_ylabel(f"Case {i+1:02d}", fontsize=10)
            ax.set(xticks=[], yticks=[])
    fig.suptitle(title, fontsize=16, y=.99)
    fig.text(.5,.012,footer,ha="center",fontsize=9)
    fig.subplots_adjust(left=.045,right=.99,top=.93,bottom=.075,hspace=.08,wspace=.05)
    fig.savefig(path,dpi=160,bbox_inches="tight");plt.close(fig)
