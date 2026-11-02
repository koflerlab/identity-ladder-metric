# CARE Table 2 reproduction

Reaggregation of released author scores only; no image-level reproduction and no H19 test.

Selection: author_venous. Pinned official commit: `2443d490c59eebaac9e8420d56b45bffed113feb`. Selected cases: 13; independent people not established.
Triplets matching all printed median/IQR values within rounding: 0/108.

| Method | CARE | SSIM | PSNR | NSD large | NSD small | clDice intestine | clDice vessels |
|---|---|---|---|---|---|---|---|
| intratomo | False | 81.7 (79.9, 83.0); n=13 | 23.9 (22.8, 24.6); n=13 | 27.2 (16.3, 41.9); n=13 | unavailable; n=0 | 4.2 (2.7, 7.5); n=11 | 16.3 (8.1, 24.4); n=2 |
| intratomo | True | 77.3 (73.3, 78.1); n=13 | 21.5 (20.2, 22.3); n=13 | 73.8 (67.5, 76.7); n=13 | 14.0 (1.2, 26.6); n=13 | 22.5 (17.7, 31.6); n=12 | 46.5 (43.4, 57.9); n=13 |
| nerf | False | 85.6 (84.2, 88.3); n=13 | 25.3 (24.7, 26.7); n=13 | 50.0 (38.5, 66.5); n=13 | 14.4 (9.2, 23.9); n=8 | 12.8 (5.1, 14.9); n=12 | 33.5 (16.2, 47.2); n=10 |
| nerf | True | 79.0 (78.4, 81.0); n=13 | 22.2 (21.1, 22.6); n=13 | 77.4 (75.7, 82.2); n=13 | 31.7 (20.3, 34.9); n=13 | 25.4 (18.0, 29.1); n=12 | 54.4 (47.1, 58.6); n=13 |
| tensorf | False | 87.3 (86.1, 90.3); n=13 | 27.6 (26.8, 28.0); n=13 | 60.6 (52.6, 67.2); n=13 | 23.3 (15.3, 34.3); n=9 | 12.1 (6.7, 17.4); n=12 | 57.6 (47.6, 68.6); n=13 |
| tensorf | True | 88.7 (86.5, 89.7); n=13 | 27.3 (26.9, 27.5); n=13 | 84.4 (78.5, 86.3); n=13 | 46.0 (33.5, 56.4); n=13 | 36.1 (25.0, 42.1); n=12 | 62.7 (51.6, 70.0); n=13 |
| r2_gaussian | False | 92.4 (89.7, 93.5); n=13 | 27.7 (26.3, 28.7); n=13 | 80.9 (73.7, 85.5); n=13 | 63.7 (47.2, 69.3); n=13 | 24.4 (16.9, 34.3); n=12 | 74.7 (55.9, 80.0); n=13 |
| r2_gaussian | True | 88.4 (84.8, 89.9); n=13 | 27.3 (24.9, 27.8); n=13 | 84.5 (75.7, 90.1); n=13 | 51.3 (41.5, 67.4); n=13 | 35.2 (26.7, 42.3); n=12 | 75.9 (58.6, 84.6); n=13 |
| naf | False | 92.0 (90.4, 92.8); n=13 | 29.3 (28.0, 29.6); n=13 | 87.1 (75.6, 90.1); n=13 | 61.4 (56.5, 69.3); n=13 | 30.4 (21.5, 41.5); n=12 | 70.0 (68.1, 83.2); n=13 |
| naf | True | 92.6 (92.0, 92.8); n=13 | 29.5 (29.0, 30.0); n=13 | 88.3 (83.9, 91.3); n=13 | 65.0 (56.8, 70.8); n=13 | 38.8 (28.9, 45.7); n=12 | 79.5 (70.5, 84.9); n=13 |
| FDK | False | 92.0 (90.3, 93.1); n=13 | 29.1 (28.3, 29.5); n=13 | 86.6 (81.9, 88.3); n=13 | 63.6 (57.5, 68.0); n=13 | 28.6 (22.4, 41.5); n=12 | 73.1 (66.9, 82.4); n=13 |
| FDK | True | 92.2 (91.3, 92.8); n=13 | 29.3 (29.0, 29.7); n=13 | 87.4 (80.8, 89.8); n=13 | 66.4 (59.4, 73.7); n=13 | 34.1 (28.4, 43.7); n=12 | 78.6 (70.2, 82.4); n=13 |
| SART | False | 91.9 (90.6, 92.6); n=13 | 29.2 (28.2, 29.6); n=13 | 84.3 (79.3, 90.4); n=13 | 63.7 (54.5, 76.1); n=13 | 27.6 (20.4, 41.9); n=12 | 76.1 (65.4, 83.8); n=13 |
| SART | True | 93.3 (92.7, 93.9); n=13 | 30.0 (29.5, 30.3); n=13 | 88.1 (81.3, 91.0); n=13 | 64.2 (50.6, 77.5); n=13 | 36.8 (27.2, 45.5); n=12 | 77.7 (69.8, 85.3); n=13 |
| ASD_POCS | False | 92.1 (90.7, 92.9); n=13 | 29.2 (28.2, 29.7); n=13 | 86.7 (82.9, 88.5); n=13 | 58.7 (50.2, 77.8); n=13 | 31.3 (20.5, 39.2); n=12 | 72.3 (64.9, 83.5); n=13 |
| ASD_POCS | True | 92.2 (91.6, 92.8); n=13 | 28.8 (28.3, 29.3); n=13 | 85.4 (81.3, 89.7); n=13 | 63.8 (55.2, 74.5); n=13 | 35.0 (28.9, 46.2); n=12 | 80.2 (71.6, 83.2); n=13 |
| Lineformer | False | 93.2 (92.3, 94.1); n=13 | 29.3 (28.8, 29.6); n=13 | 83.6 (73.2, 86.0); n=13 | 63.1 (48.7, 78.0); n=13 | 26.7 (21.0, 42.3); n=12 | 76.1 (63.5, 85.8); n=13 |
| Lineformer | True | 92.3 (91.5, 93.3); n=13 | 29.7 (28.9, 29.9); n=13 | 89.1 (81.5, 93.1); n=13 | 65.2 (63.2, 69.7); n=13 | 36.2 (26.9, 43.8); n=12 | 76.5 (70.1, 85.0); n=13 |

## Interpretation

- Author-produced scores are reused, not independently recomputed from CT images.
- The table and selected reference numbers were seen before protocol writing.
- No mean-fill, domain-shift or identity experiment has been executed here.
- Missing values follow authors' nan-aware rules; endpoint denominators are reported.
- Lineformer CSV mapping to SAX-NeRF is provisional and judged against table agreement.
- Read the JSON for mismatch details, finite/missing counts and paired descriptive changes.
- Original pipeline, weights and image-level data remain unchecked. This stage cannot refute or confirm H19.
- Source attribution: Lin et al., NeurIPS 2025, https://github.com/MrGiovanni/CARE ; official repository LICENSE says CC BY-NC-ND. No upstream code is copied here.
