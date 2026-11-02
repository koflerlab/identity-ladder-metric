# BraTS inpainting metrics on real 2D repeat, mirror and mean fills

Protocol: `docs/PROTOCOL_2d_ladder.md`, Amendment 1 (descriptive, written before scoring). Full table with all nine
package outputs: `ladder2d_brats_baseline.csv`; numbers with intervals: `ladder2d_brats_baseline.json`.

Setup. The truth is a real image; a fixed centred 64 x 64 square (of 224 x 224) is filled from another real image of
the same pair and scored with the official BraTS inpainting package. c is the share of matched strangers' fills that
score worse (0.5 = a typical stranger, 1 = better than every stranger). Datasets: GRAPE fundus (437 pairs, 144
people), UPOL iris (192 pairs, 64 people), UTIRIS near-infrared iris (234 pairs, 78 people).

| Fill (c for MSE / SSIM) | GRAPE | UPOL | UTIRIS | QTIM MRI |
|---|---|---|---|---|
| same-sitting repeat | no repeat in GRAPE | 0.81 / 0.81 | 0.69 / 0.68 | from its own run |
| other visit, same camera, under 1 year | 0.65 / 0.79 | | | |
| other visit, same camera, 1 year or more | 0.67 / 0.76 | | | |
| other visit, other camera | 0.57 / 0.67 | | | |
| mirrored other side (fellow eye flipped) | 0.76 / 0.70 | 0.73 / 0.74 | 0.60 / 0.61 | from its own run |
| population mean of 16 matched strangers | 0.65 / 0.94 | 0.92 / 0.91 | 0.88 / 0.81 | from its own run |

What it shows:
- A person's own repeat or later image fills the patch better than a typical stranger (c 0.57 to 0.81), but far from
  every stranger: these metrics separate people weakly.
- The population mean scores as well as or better than the person's own image on almost every metric and dataset
  (UPOL MSE c 0.92 against 0.81 for the same-sitting repeat; GRAPE SSIM 0.94 against 0.79). The only exception is MAE
  on UPOL (0.83 against 0.89). The leaderboard metrics reward a blurred average over the real person.
- The mirrored fellow eye sits close to the repeat and later-visit rows, so these metrics barely distinguish "the
  same eye" from "the other eye of the same person".
- On GRAPE the other-camera row scores lower than the same-camera rows (MSE c 0.57 against 0.65 and 0.67), but the
  intervals overlap and no difference was tested here.

Limits: one fixed central patch per image (on iris mostly pupil and inner iris), no registration, small public
cohorts; the population-mean row pools all same-eye pairs of a dataset. 2D adaptation of the metrics, not a BraTS
submission. Per-pair scores stay in the private run folder.
