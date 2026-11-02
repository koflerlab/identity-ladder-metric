# CARE: iteration 2, current endpoint results

Released-score aggregation only; image-level CARE and H19 not run.

all 36 official test cases, all phases; post hoc amendment 2

| Metric | Match / 18 | Mismatch | Blocked coverage | Undefined |
|---|---|---|---|---|
| ssim | 18 | 0 | 0 | 0 |
| psnr | 18 | 0 | 0 | 0 |
| large_nsd | 0 | 16 | 2 | 0 |
| small_nsd | 0 | 15 | 2 | 1 |
| intestine_cldice | 0 | 16 | 2 | 0 |
| vessel_cldice | 0 | 16 | 2 | 0 |

## Every original and current value

Values are median [Q1, Q3]. Pixel SSIM and anatomy scores are percentages; PSNR is dB. All are higher-is-better.

| Method | CARE | Metric | Paper | Current | Finite / eligible | Status |
|---|---|---|---|---|---|---|
| intratomo | False | ssim | 82.7 [80.9, 84.7] | 82.7 [80.9, 84.7] | 36 / 36 | matched |
| intratomo | False | psnr | 24.5 [23.3, 25.1] | 24.5 [23.3, 25.1] | 36 / 36 | matched |
| intratomo | False | large_nsd | 2.4 [1.6, 3.6] | 23.9 [15.4, 38.3] | 35 / 36 | mismatch |
| intratomo | False | small_nsd | 0.0 [0.0, 0.0] | not available | 0 / 36 | unavailable |
| intratomo | False | intestine_cldice | 6.3 [1.2, 17.4] | 4.0 [1.2, 11.4] | 27 / 36 | mismatch |
| intratomo | False | vessel_cldice | 4.7 [0.0, 21.6] | 11.7 [0.0, 25.7] | 7 / 36 | mismatch |
| intratomo | True | ssim | 76.7 [74.0, 79.7] | 76.7 [74.0, 79.7] | 36 / 36 | matched |
| intratomo | True | psnr | 22.0 [20.6, 22.8] | 22.0 [20.6, 22.8] | 36 / 36 | matched |
| intratomo | True | large_nsd | 30.9 [26.4, 40.6] | 70.6 [58.7, 77.0] | 36 / 36 | mismatch |
| intratomo | True | small_nsd | 7.1 [3.3, 13.3] | 9.4 [1.2, 22.1] | 32 / 36 | mismatch |
| intratomo | True | intestine_cldice | 33.5 [24.4, 43.6] | 22.5 [14.8, 27.0] | 34 / 36 | mismatch |
| intratomo | True | vessel_cldice | 47.9 [35.4, 62.2] | 46.2 [38.4, 58.3] | 35 / 36 | mismatch |
| nerf | False | ssim | 87.7 [84.5, 89.4] | 87.7 [84.5, 89.4] | 36 / 36 | matched |
| nerf | False | psnr | 26.6 [24.7, 27.4] | 26.6 [24.7, 27.4] | 36 / 36 | matched |
| nerf | False | large_nsd | 22.1 [8.2, 35.8] | 47.2 [36.4, 56.8] | 36 / 36 | mismatch |
| nerf | False | small_nsd | 0.2 [0.0, 4.3] | 10.6 [2.3, 35.2] | 17 / 36 | mismatch |
| nerf | False | intestine_cldice | 17.2 [9.0, 28.0] | 10.8 [4.8, 15.8] | 32 / 36 | mismatch |
| nerf | False | vessel_cldice | 29.0 [19.8, 42.1] | 35.5 [19.2, 53.2] | 31 / 36 | mismatch |
| nerf | True | ssim | 80.5 [78.5, 83.0] | 80.5 [78.5, 83.0] | 36 / 36 | matched |
| nerf | True | psnr | 23.4 [21.9, 24.4] | 23.4 [21.9, 24.4] | 36 / 36 | matched |
| nerf | True | large_nsd | 58.0 [49.1, 63.0] | 77.0 [71.3, 83.0] | 36 / 36 | mismatch |
| nerf | True | small_nsd | 25.3 [14.2, 30.1] | 27.1 [15.6, 34.9] | 35 / 36 | mismatch |
| nerf | True | intestine_cldice | 53.5 [47.4, 59.6] | 26.1 [18.2, 37.4] | 34 / 36 | mismatch |
| nerf | True | vessel_cldice | 55.0 [45.3, 65.6] | 56.0 [47.0, 64.7] | 36 / 36 | mismatch |
| tensorf | False | ssim | 89.3 [87.3, 91.1] | 89.3 [87.3, 91.1] | 36 / 36 | matched |
| tensorf | False | psnr | 28.0 [27.3, 29.7] | 28.0 [27.3, 29.7] | 36 / 36 | matched |
| tensorf | False | large_nsd | 44.6 [21.8, 53.5] | 61.3 [43.5, 74.8] | 36 / 36 | mismatch |
| tensorf | False | small_nsd | 12.1 [1.1, 28.4] | 23.9 [15.5, 34.3] | 28 / 36 | mismatch |
| tensorf | False | intestine_cldice | 29.5 [19.2, 36.4] | 12.4 [4.8, 20.9] | 34 / 36 | mismatch |
| tensorf | False | vessel_cldice | 46.1 [36.8, 58.5] | 48.0 [40.8, 62.4] | 36 / 36 | mismatch |
| tensorf | True | ssim | 88.8 [87.2, 90.1] | 88.8 [87.2, 90.1] | 36 / 36 | matched |
| tensorf | True | psnr | 27.5 [27.2, 28.2] | 27.5 [27.2, 28.2] | 36 / 36 | matched |
| tensorf | True | large_nsd | 75.3 [65.7, 79.3] | 83.6 [74.5, 87.2] | 36 / 36 | mismatch |
| tensorf | True | small_nsd | 38.2 [30.2, 46.4] | 38.2 [23.3, 53.7] | 36 / 36 | mismatch |
| tensorf | True | intestine_cldice | 67.9 [62.8, 75.8] | 34.4 [23.4, 42.4] | 34 / 36 | mismatch |
| tensorf | True | vessel_cldice | 58.8 [51.0, 68.4] | 59.0 [44.9, 66.8] | 36 / 36 | mismatch |
| r2_gaussian | False | ssim | 93.3 [92.1, 94.4] | 93.3 [92.1, 94.4] | 36 / 36 | matched |
| r2_gaussian | False | psnr | 28.8 [27.4, 29.7] | 28.8 [27.4, 29.7] | 36 / 36 | matched |
| r2_gaussian | False | large_nsd | 76.3 [65.9, 84.6] | 81.6 [70.7, 86.6] | 36 / 36 | mismatch |
| r2_gaussian | False | small_nsd | 59.1 [41.1, 68.3] | 62.6 [45.1, 72.8] | 36 / 36 | mismatch |
| r2_gaussian | False | intestine_cldice | 53.9 [45.6, 68.1] | 24.7 [16.4, 33.3] | 34 / 36 | mismatch |
| r2_gaussian | False | vessel_cldice | 64.5 [59.4, 78.6] | 66.2 [59.4, 78.8] | 36 / 36 | mismatch |
| r2_gaussian | True | ssim | 89.6 [87.9, 90.7] | 89.6 [87.9, 90.7] | 36 / 36 | matched |
| r2_gaussian | True | psnr | 27.3 [26.2, 28.9] | 27.3 [26.2, 28.9] | 36 / 36 | matched |
| r2_gaussian | True | large_nsd | 82.5 [74.0, 88.3] | 86.0 [75.7, 90.2] | 36 / 36 | mismatch |
| r2_gaussian | True | small_nsd | 55.7 [42.5, 67.3] | 52.1 [43.7, 69.5] | 36 / 36 | mismatch |
| r2_gaussian | True | intestine_cldice | 75.4 [68.0, 82.8] | 35.0 [26.4, 47.8] | 34 / 36 | mismatch |
| r2_gaussian | True | vessel_cldice | 67.6 [61.8, 76.9] | 65.7 [58.5, 76.0] | 36 / 36 | mismatch |
| naf | False | ssim | 92.5 [91.1, 94.0] | 92.5 [91.1, 94.0] | 36 / 36 | matched |
| naf | False | psnr | 29.4 [28.9, 30.8] | 29.4 [28.9, 30.8] | 36 / 36 | matched |
| naf | False | large_nsd | 83.7 [76.2, 86.8] | 85.5 [72.9, 90.1] | 36 / 36 | mismatch |
| naf | False | small_nsd | 61.0 [51.1, 70.2] | 62.0 [49.7, 74.9] | 36 / 36 | mismatch |
| naf | False | intestine_cldice | 71.6 [62.2, 77.8] | 29.6 [20.6, 40.5] | 34 / 36 | mismatch |
| naf | False | vessel_cldice | 75.4 [70.1, 81.8] | 70.2 [65.5, 79.5] | 36 / 36 | mismatch |
| naf | True | ssim | 92.8 [91.5, 93.5] | 92.8 [91.5, 93.5] | 36 / 36 | matched |
| naf | True | psnr | 29.6 [28.6, 30.3] | 29.6 [28.6, 30.3] | 36 / 36 | matched |
| naf | True | large_nsd | 87.9 [83.8, 91.3] | 88.5 [80.3, 91.4] | 36 / 36 | mismatch |
| naf | True | small_nsd | 68.1 [52.4, 76.7] | 66.2 [51.8, 74.9] | 36 / 36 | mismatch |
| naf | True | intestine_cldice | 81.8 [74.1, 85.9] | 38.7 [26.3, 47.3] | 34 / 36 | mismatch |
| naf | True | vessel_cldice | 71.8 [63.4, 80.8] | 70.2 [61.8, 79.6] | 36 / 36 | mismatch |
| FDK | False | ssim | 92.5 [91.1, 94.2] | 92.5 [91.1, 94.2] | 36 / 36 | matched |
| FDK | False | psnr | 29.5 [28.9, 30.9] | 29.5 [28.9, 30.9] | 36 / 36 | matched |
| FDK | False | large_nsd | 83.8 [77.3, 86.8] | 86.8 [79.5, 90.0] | 36 / 36 | mismatch |
| FDK | False | small_nsd | 62.4 [52.4, 70.6] | 63.0 [55.2, 74.0] | 36 / 36 | mismatch |
| FDK | False | intestine_cldice | 68.2 [59.8, 76.9] | 29.5 [21.4, 40.7] | 34 / 36 | mismatch |
| FDK | False | vessel_cldice | 76.5 [66.2, 80.3] | 73.6 [65.5, 80.5] | 36 / 36 | mismatch |
| FDK | True | ssim | 92.4 [91.5, 93.4] | 92.4 [91.5, 93.4] | 36 / 36 | matched |
| FDK | True | psnr | 29.7 [29.0, 30.6] | 29.7 [29.0, 30.6] | 36 / 36 | matched |
| FDK | True | large_nsd | 87.1 [82.3, 90.9] | 86.4 [78.0, 89.6] | 36 / 36 | mismatch |
| FDK | True | small_nsd | 67.0 [55.6, 77.1] | 65.7 [52.9, 73.8] | 36 / 36 | mismatch |
| FDK | True | intestine_cldice | 81.5 [75.5, 85.7] | 35.5 [26.1, 46.4] | 34 / 36 | mismatch |
| FDK | True | vessel_cldice | 71.7 [66.4, 83.2] | 69.9 [61.0, 76.2] | 36 / 36 | mismatch |
| SART | False | ssim | 92.5 [91.1, 94.2] | 92.5 [91.1, 94.2] | 36 / 36 | matched |
| SART | False | psnr | 29.5 [28.9, 31.0] | 29.5 [28.9, 31.0] | 36 / 36 | matched |
| SART | False | large_nsd | 84.6 [76.9, 88.1] | 84.2 [78.9, 90.4] | 36 / 36 | mismatch |
| SART | False | small_nsd | 61.3 [53.7, 75.4] | 64.4 [54.0, 75.1] | 36 / 36 | mismatch |
| SART | False | intestine_cldice | 70.0 [61.0, 77.6] | 29.4 [19.6, 42.1] | 34 / 36 | mismatch |
| SART | False | vessel_cldice | 78.7 [65.6, 83.6] | 74.2 [64.7, 82.6] | 36 / 36 | mismatch |
| SART | True | ssim | 93.3 [92.7, 94.3] | 93.3 [92.7, 94.3] | 36 / 36 | matched |
| SART | True | psnr | 30.3 [29.5, 30.9] | 30.3 [29.5, 30.9] | 36 / 36 | matched |
| SART | True | large_nsd | 88.2 [84.2, 90.8] | 86.6 [78.8, 91.1] | 36 / 36 | mismatch |
| SART | True | small_nsd | 68.5 [56.0, 75.5] | 65.1 [48.3, 77.7] | 36 / 36 | mismatch |
| SART | True | intestine_cldice | 83.5 [77.8, 87.5] | 38.0 [26.4, 46.5] | 34 / 36 | mismatch |
| SART | True | vessel_cldice | 75.2 [62.2, 83.3] | 69.0 [63.1, 77.8] | 36 / 36 | mismatch |
| ASD_POCS | False | ssim | 92.3 [91.6, 94.2] | 92.3 [91.6, 94.2] | 36 / 36 | matched |
| ASD_POCS | False | psnr | 29.5 [28.9, 30.7] | 29.5 [28.9, 30.7] | 36 / 36 | matched |
| ASD_POCS | False | large_nsd | 83.8 [76.5, 87.2] | 85.6 [78.5, 89.5] | 36 / 36 | mismatch |
| ASD_POCS | False | small_nsd | 61.0 [51.4, 73.6] | 59.9 [49.6, 75.5] | 36 / 36 | mismatch |
| ASD_POCS | False | intestine_cldice | 71.3 [61.8, 77.4] | 28.9 [20.2, 40.2] | 34 / 36 | mismatch |
| ASD_POCS | False | vessel_cldice | 74.7 [67.4, 81.1] | 70.6 [63.9, 81.7] | 36 / 36 | mismatch |
| ASD_POCS | True | ssim | 92.4 [91.8, 93.5] | 92.4 [91.8, 93.5] | 36 / 36 | matched |
| ASD_POCS | True | psnr | 29.4 [28.4, 30.1] | 29.4 [28.4, 30.1] | 36 / 36 | matched |
| ASD_POCS | True | large_nsd | 88.1 [82.6, 90.9] | 86.4 [79.0, 89.9] | 36 / 36 | mismatch |
| ASD_POCS | True | small_nsd | 67.8 [56.2, 75.4] | 62.9 [53.7, 74.6] | 36 / 36 | mismatch |
| ASD_POCS | True | intestine_cldice | 80.7 [74.9, 84.9] | 36.4 [26.3, 46.6] | 34 / 36 | mismatch |
| ASD_POCS | True | vessel_cldice | 76.4 [66.6, 79.9] | 71.1 [59.2, 80.0] | 36 / 36 | mismatch |
| Lineformer | False | ssim | 93.7 [92.8, 95.0] | 93.7 [92.8, 95.0] | 36 / 36 | matched |
| Lineformer | False | psnr | 29.6 [29.0, 31.0] | 29.6 [29.0, 31.0] | 36 / 36 | matched |
| Lineformer | False | large_nsd | 82.4 [73.9, 88.9] | not available | None / 36 | blocked_coverage |
| Lineformer | False | small_nsd | 66.5 [55.0, 78.0] | not available | None / 36 | blocked_coverage |
| Lineformer | False | intestine_cldice | 68.0 [56.3, 77.9] | not available | None / 36 | blocked_coverage |
| Lineformer | False | vessel_cldice | 79.3 [62.4, 85.1] | not available | None / 36 | blocked_coverage |
| Lineformer | True | ssim | 92.5 [91.2, 93.3] | 92.5 [91.2, 93.3] | 36 / 36 | matched |
| Lineformer | True | psnr | 29.4 [28.8, 30.1] | 29.4 [28.8, 30.1] | 36 / 36 | matched |
| Lineformer | True | large_nsd | 88.7 [81.9, 90.9] | not available | None / 36 | blocked_coverage |
| Lineformer | True | small_nsd | 67.1 [54.5, 75.1] | not available | None / 36 | blocked_coverage |
| Lineformer | True | intestine_cldice | 81.5 [76.0, 86.8] | not available | None / 36 | blocked_coverage |
| Lineformer | True | vessel_cldice | 72.0 [62.9, 84.1] | not available | None / 36 | blocked_coverage |

- Coverage is assessed per file type and method. Lineformer anatomy needs 36 cases; its 35-case exports are not silently substituted.
- Finite denominators may vary under the authors' nan-aware category averaging. No missing or undefined score is replaced by zero.
- The V-only 13-case result remains a reproduction attempt of the released script, not of the paper's full test selection.
- Anatomical mismatches do not establish the cause, nor refute CARE or establish H19. Released CSV provenance, label maps, tolerance and missing-value semantics need investigation.
- The official table script's SSIM/PSNR NaN-filter issue is inactive here when all pixel scores are finite. Baseline/CARE comparisons are paired descriptively; no unpaired p-values are added.
- Source: Lin et al., NeurIPS 2025, https://github.com/MrGiovanni/CARE at pinned commit in the manifest. No upstream CSV or code is redistributed.
- Original R1 and amendment-1 reports are preserved. Reviewer aggregates were seen before this run; this is a disclosed post hoc correction.
