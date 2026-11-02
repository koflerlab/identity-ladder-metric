# CARE reproduction and H19 extension

Date: 2026-11-02. Authorized by Kate in the literature-comparison chat: add all selected papers to the project backlog and start reproducing them.

## Question and evidence boundary

Stage R1: can an independent aggregation of the authors' released per-scan scores recover Table 2 of Lin et al., NeurIPS 2025, *Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?*

This is a reproduction of published score aggregation, not an independent image-level reconstruction or segmentation experiment. The paper and selected Table 2 numbers have already been seen. It is not a blinded or novel confirmatory study. No new metric outcome from the CSVs has been inspected before writing this protocol.

Stage R2: execute the released segmentator and metrics on the authors' eligible CT images/reconstructions, after verifying weights, data, split and resource access. Stage H19: separately add independent population-mean, mean-with-restored-spread, shrinkage and wrong-detail controls. R1 cannot answer H19.

## Frozen sources and inputs

- Official code: https://github.com/MrGiovanni/CARE at `2443d490c59eebaac9e8420d56b45bffed113feb`.
- Paper: https://papers.nips.cc/paper_files/paper/2025/file/307f375e35616bbc2861033966b44976-Paper-Conference.pdf, Table 2, page 9.
- Scores: `ReconstructionPipeline/resultsCSVcldice/BDMAP_O_{method}_50_{pixel|seg}{_care}.csv` at that commit.
- Split: `STEP3-CAREModel/splits/BDMAP_O_AV_meta_test.csv`. The released table script refers instead to a missing `ReconstructionPipeline/dataset_splits/` location. Use the existing official split explicitly and record the difference.
- Methods, fixed order: intratomo, nerf, tensorf, r2_gaussian, naf, FDK, SART, ASD_POCS, Lineformer. The final entry corresponds to the paper's SAX-NeRF row; verify its table agreement rather than assuming it.
- Before aggregation, `--freeze-inputs` records SHA-256 and byte counts of all sources without storing or printing case-level rows. `--run` rejects any changed source. Preserve this manifest; do not silently refresh it after results.
- Public score CSVs and split are processed in memory. No participant identifiers, raw CSVs, images or weights enter this repository. No change to the shared data tree/registry is required for this transient read. Any subsequent persistent acquisition must use the data-root lock and registry.

## Definitions and checks

Use the official split rule: retain venous-phase split entries ending in V and remove their final two characters to match score keys. Require unique score keys, matching selected pixel/segmentation and baseline/CARE case sets, and complete coverage of the selected split. Count scans/cases, not proven independent people.

Metrics in table order: SSIM, PSNR, NSD large organs, NSD small organs, clDice intestine, clDice vessels. Scores already have their published units: SSIM and anatomy scores in percent, PSNR in dB. Do not rescale again.

For each scan, average available structure scores within the fixed anatomical category, using the category names/labels in official `step5_calculateMetrics.py`. Then compute median, 25th and 75th percentiles across selected scans, matching the authors' nan-aware aggregation. Report missing counts per endpoint. Fail if an endpoint has no finite cases.

Extract the 18 x 6 printed median/IQR triplets from Table 2, excluding other tables. Each reconstructed value must differ from its printed counterpart by at most 0.050001, the tolerance for one-decimal publication rounding. Report all mismatches, rather than tuning source folder, split or metric definition to improve agreement. Compare paired case-level changes descriptively, with the number of improved/tied/worsened cases, without claiming independent subjects or adding significance tests.

## H19 extension, not yet executed

Question: can segmentation-based metrics reward an identity-free mean more than real donor anatomy under controlled acquisition differences? Outcomes must include ordinary Dice as a comparator, CARE's area-weighted NSD at the original 2 mm tolerance, and clDice. Compact brain tissues and tubular structures require separate interpretation; clDice is not automatically an appropriate cortical-identity measure.

Before new image scoring, fix eligible dataset and participants, acquisition/domain-shift model, preprocessing, voxel spacing, registration sensitivity, segmentator version, disjoint family-aware mean/reference pools, ROI and label categories. Preserve true-image segmentation as one reference and annotated/simulation labels, where available, as a separate reference. Include all three distances d(x,y), d(y,mu), d(x,mu). Distance from mu alone is not identity evidence. A successful mean rejection by NSD/clDice would limit H19, and must be retained.

The concurrent QTIM SyN and frozen real-case jobs are separate. Do not duplicate, change their protocol, or reuse invalid affine results as validated evidence.

## Outputs and completion

`src/care_reproduction.py`; `results/care_reproduction_inputs.json`; aggregate JSON/CSV/report under `results/care_reproduction*`. Manifest includes protocol/code hashes, Git commit, package versions and retrieval time. R1 is complete only after input checks, all table comparisons and report generation. Record R2/H19 access and compute limits separately; never label R1 as full CARE reproduction.
