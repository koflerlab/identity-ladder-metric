# Draft figures (kept out of the README for now)

Plots set aside while the README shows only case images. The script that regenerates each one from saved results is
named in the heading.

## Fig. 1 of the ISBI draft (`paper/isbi2027/make_figures.py`)

![Fig. 1](../results/figures/fig_ladder.png)

(a) Closed form $E\,c = P(F_{D,D} > \rho)$ against simulation, $D = 8$ and $32$. (b) Synthetic 2D tissue, genetic share 0.19,
ho)$ against simulation, D = 8 and 32. (b) Synthetic 2D tissue, genetic share 0.19,
32 x 32 mask, RMSE and SSIM from the official BraTS code; filled marks contrast 1, open marks contrast 0.1.

## Step-by-step shrinkage to the mean (`src/rebuild_visuals.py`)

![Mixing, synthetic tissue](../results/figures/mixing_tissue.png)

![Mixing, synthetic eyes](../results/figures/mixing_eyes.png)

## Population null, all metrics (`src/detail_report.py summary`)

![All nine BraTS outputs under the population null](../results/figures/population_null_all_metrics.png)

![Shape metrics under the population null](../results/figures/population_null_shape_metrics.png)

## All BraTS metrics on the synthetic ladder (`src/ladder_report.py`)

![All BraTS metrics, genetic share 0.54](../results/figures/all_brats_metrics_base.png)

![All BraTS metrics, genetic share 0.19](../results/figures/all_brats_metrics_low_h.png)

## H19 robustness (`src/ladder_report.py`)

![Robustness, all metrics](../results/figures/label_robustness_all_metrics.png)

![Mean minus MZ twin, paired](../results/figures/label_paired_mz_all_metrics.png)

![Mean minus DZ twin, paired](../results/figures/label_paired_dz_all_metrics.png)

## Pupil experiment, all metrics (`src/pupil_report.py`)

![Pupil experiment, all metrics](../results/figures/pupil_all_metrics.png)
