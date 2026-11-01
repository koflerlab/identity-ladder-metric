"""Reproducible tables and exploratory figures; preserves the paused paper figures.

Usage: python src/ladder_report.py
No simulation is rerun: this reads saved toy, label and (if complete) robustness results.
Raw median (toy) and raw mean (label) remain explicitly distinguished.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
INTENSITY = ("rmse", "mse", "mae", "msle", "psnr", "psnr_eps", "psnr_01", "psnr_01_eps", "ssim")
DOWNSTREAM = ("dice", "accuracy", "topo_err_components", "topo_err_holes", "topo_err_euler", "topo_err_perimeter")
NAMES = {m: m.upper() for m in INTENSITY}
NAMES.update(dice="Dice", accuracy="Accuracy", topo_err_components="Component-count error",
             topo_err_holes="Hole-count error", topo_err_euler="Euler-number error", topo_err_perimeter="Perimeter error")
COLORS = ("#2166ac", "#c65418")


def table_rows(toy, label):
    for source, configs in (("toy", toy["donor_swap"]), ("label", label)):
        for config, values in configs.items():
            if not isinstance(values, dict) or "rmse" not in values:
                continue
            for metric in INTENSITY + DOWNSTREAM:
                for fill, v in values.get(metric, {}).items():
                    raw_stat = "median_raw" if "median_raw" in v else "mean_raw"
                    yield {"source": source, "config": config, "fill": fill, "metric": metric,
                           "mean_c": v["mean_c"], "raw_statistic": raw_stat, "raw_value": v[raw_stat],
                           "rho_hat_mse": values.get("rho_hat_mse", {}).get(fill, ""),
                           "rho_predicted": values.get("rho_predicted", {}).get(fill, ""),
                           "rho_status": "not_estimated_for_nonlinear_label_model" if source == "label"
                                         else "stranger_anchor" if fill == "stranger" else "intensity_mse_ratio",
                           "stranger_is_fixed_anchor": fill == "stranger"}


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def toy_panels(toy, setting):
    fills = [("L0_same_sitting", "Same sitting"), ("L0_retest", "Other day"),
             ("mirror_same_sitting", "Other side"), ("MZ", "MZ co-twin"), ("DZ", "DZ co-twin"),
             ("population_mean", "Population mean"), ("population_mean_sd_restored", "Mean, SD restored")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), sharex=True, sharey=True)
    y = np.arange(len(fills))[::-1]
    for ax, metric in zip(axes.flat, INTENSITY):
        for j, contrast in enumerate((1, .1)):
            values = toy["donor_swap"][f"{setting}_contrast{contrast:g}_mask32"][metric]
            ax.scatter([values[k]["mean_c"] for k, _ in fills], y + (.13 if j == 0 else -.13),
                       marker="o" if j == 0 else "D", s=42, color=COLORS[j],
                       label=f"Contrast {contrast:g}", zorder=3)
        ax.axvline(.5, color="#777777", ls="--", lw=.9)
        ax.set(title=NAMES[metric], xlim=(-.025, 1.025), ylim=(-.6, 6.6),
               yticks=y, yticklabels=[v for _, v in fills], xticks=[0, .25, .5, .75, 1])
        ax.grid(axis="x", color="#e5e5e5")
        ax.tick_params(labelleft=True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Mean calibrated score c")
    fig.suptitle(f"All nine BraTS metrics on synthetic tissue | {setting} variance setting", fontsize=17, y=.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    fig.text(.5, .016, "120 targets per condition; 32 x 32 mask. Higher c = a better score against stranger fills.\n"
             "Dashed line: stranger anchor (0.5). MZ/DZ use their own family subsets. Existing single-seed results; no uncertainty bars.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.13, right=.98, bottom=.105, top=.90, hspace=.25, wspace=.55)
    save(fig, f"all_brats_metrics_{setting}")


def robustness_panels(report):
    fig, axes = plt.subplots(5, 3, figsize=(13, 15), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, INTENSITY + DOWNSTREAM):
        for i, name in enumerate(("base", "low_h")):
            x = np.array([0, .25, 1.])
            vals = [report["summary"][f"{name}_prior{p:g}"]["metrics"][m]["population_mean"] for p in x]
            mean = np.array([v["mean"] for v in vals])
            err = np.array([[v["mean"] - v["min"] for v in vals], [v["max"] - v["mean"] for v in vals]])
            ax.errorbar(x + (i - .5) * .015, mean, yerr=err, marker="o", capsize=3,
                        lw=1.6, color=COLORS[i], label=f"Population mean: {name}")
        ax.axhline(.5, ls="--", lw=.9, color="#777777")
        ax.set(title=NAMES[m], xticks=[0, .25, 1], ylim=(-.025, 1.025), xlim=(-.06, 1.06))
        ax.grid(color="#ececec")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean calibrated c")
    for ax in axes[-1]:
        ax.set_xlabel("Population prior scale")
    fig.suptitle("Does the population mean still win when the common anatomy weakens?", fontsize=16, y=.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .966), ncol=2, frameon=False)
    fig.text(.5, .013, "Five independent simulated cohorts, each with 60 families / 120 targets; same draws across conditions.\n"
             "Bars: min-max of five cohort means, NOT confidence intervals. Dashed line: stranger anchor.\n"
             "Higher c = smaller errors / higher similarity. Perimeter is geometry; topology includes mask-border cuts. 2D only.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, top=.925, bottom=.085, hspace=.35, wspace=.22)
    save(fig, "label_robustness_all_metrics")


def paired_panels(report, zygosity):
    fig, axes = plt.subplots(5, 3, figsize=(13, 15), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, INTENSITY + DOWNSTREAM):
        for i, name in enumerate(("base", "low_h")):
            x = np.array([0, .25, 1.])
            vals = [report["summary"][f"{name}_prior{p:g}"]["paired_twin_delta"][zygosity][m] for p in x]
            mean = np.array([v["mean"] for v in vals])
            err = np.array([[v["mean"] - v["min"] for v in vals], [v["max"] - v["mean"] for v in vals]])
            ax.errorbar(x + (i - .5) * .015, mean, yerr=err, marker="o", capsize=3,
                        lw=1.6, color=COLORS[i], label=name)
        ax.axhline(0, ls="--", lw=.9, color="#777777")
        ax.set(title=NAMES[m], xticks=[0, .25, 1], ylim=(-1.025, 1.025), xlim=(-.06, 1.06))
        ax.grid(color="#ececec")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Paired delta-c: mean minus twin")
    for ax in axes[-1]:
        ax.set_xlabel("Population prior scale")
    fig.suptitle(f"Population mean versus {zygosity} co-twin: paired on the same targets", fontsize=16, y=.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .966), ncol=2, frameon=False)
    fig.text(.5, .013, f"Positive: population mean scores better. Negative: {zygosity} co-twin scores better.\n"
             "Five independent simulated cohorts; 30 families of this twin type / 60 targets per cohort.\n"
             "Bars: min-max of five cohort means, NOT confidence intervals. Paired contrasts use the same stranger reference.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, top=.925, bottom=.085, hspace=.35, wspace=.22)
    save(fig, f"label_paired_{zygosity.lower()}_all_metrics")


def main():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    toy = json.loads((RESULTS / "toy_ladder.json").read_text())
    label = json.loads((RESULTS / "label_ladder.json").read_text())
    rows = list(table_rows(toy, label))
    with (RESULTS / "all_ladder_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for setting in ("base", "low_h"):
        toy_panels(toy, setting)
    path = RESULTS / "label_robustness.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report["status"] == "complete":
            robustness_panels(report)
            for zygosity in ("MZ", "DZ"):
                paired_panels(report, zygosity)
            summary = []
            for config, data in report["summary"].items():
                for metric, fills in data["metrics"].items():
                    for fill, v in fills.items():
                        summary.append({"config": config, "metric": metric, "fill": fill, **v})
            with (RESULTS / "label_robustness_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0]))
                writer.writeheader(); writer.writerows(summary)
            paired = [{"config": config, "zygosity": z, "metric": m, **v}
                      for config, data in report["summary"].items()
                      for z, metrics in data["paired_twin_delta"].items() for m, v in metrics.items()]
            with (RESULTS / "label_paired_twin_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(paired[0]))
                writer.writeheader(); writer.writerows(paired)
    print(f"Exported {len(rows)} metric rows from both ladders; figures: {FIGS}")


if __name__ == "__main__":
    main()
