# ISBI 2027 paper: regression to the mean in brain MRI inpainting evaluation

Draft paper on why pointwise metrics (MSE/RMSE/PSNR, SSIM, Dice against the oracle) favour the population-mean fill,
built on the experiments of this repository.

## Layout

```
make_figures.py   regenerates the paper figure from results/toy_ladder.json
overleaf/         everything Overleaf needs: main.tex, refs.bib, IEEEbib.bst, spconf.sty, figures
```

To upload: zip `overleaf/` and use New Project, Upload Project in Overleaf.

## Building locally

```bash
python paper/isbi2027/make_figures.py
```

```bash
latexmk -pdf -cd paper/isbi2027/overleaf/main.tex
```

`overleaf/fig_cases.pdf` is drawn by `make_figures.py` from the saved example `results/null_example_tissue.npz` (no new run).

## Open before submission (deadline 2 Nov 2026, 23:59 ET)

- Downstream U-Net sentence in the introduction: Florian to confirm whether it can be stated.
- Funding and conflicts of interest.
- Real-data cases next to each synthetic result (protocol pending).
