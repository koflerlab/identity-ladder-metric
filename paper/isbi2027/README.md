# ISBI 2027 paper: regression to the mean in brain MRI inpainting evaluation

Draft paper on why pointwise metrics (MSE/RMSE/PSNR, SSIM, Dice against the oracle) favour the population-mean fill,
tested on real brain MRI (QTIM). Four pages, references included; the ISBI fifth page is not used.

## Layout

```
make_fig_real_case.py   draws both paper figures from the brain donor ladder
make_figures.py         older synthetic figures (not in the current draft)
overleaf/               everything Overleaf needs: main.tex, refs.bib, IEEEbib.bst, spconf.sty, figures
```

To upload: zip `overleaf/` and use New Project, Upload Project in Overleaf.

## Building locally

```bash
python paper/isbi2027/make_fig_real_case.py
```

```bash
latexmk -pdf -cd paper/isbi2027/overleaf/main.tex
```

`overleaf/fig_real_case.pdf` (Fig. 1: one QTIM brain, parietal box filled from each donor) and
`overleaf/fig_real_theory.pdf` (Fig. 2: the same box against the theory) are drawn by `make_fig_real_case.py`. Fig. 1
needs the private cube cache of the brain donor ladder and the prepared QTIM scans in `C:\Projects\data`; Fig. 2 and
Table 1 come from `results/donor_ladder_brain.json` (`python paper/isbi2027/make_fig_real_case.py --theory-only`
redraws Fig. 2 alone). The repository README shows the same figures with a title,
`results/figures/fig_real_case_parietal.png` and `results/figures/fig_real_case_theory.png`. Affine registration only;
registration visual QC is not complete.

## Open before submission

- Downstream U-Net sentence in the introduction: now argued from Proposition 2; cite a published downstream-Dice
  result here once one is public (the one known example is a confidential challenge submission).
- Robustness paragraph: 32 mm cubes, mask thresholds and the SyN subset
  (`docs/PROTOCOL_donor_ladder_robustness.md`).
