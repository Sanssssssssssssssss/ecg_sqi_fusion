# CleanBUT Bad-Core Targeted Synthetic Grid

This is a generator-target scan. CleanBUT-Core is used only as a target/diagnostic subset; original BUT 10s P1 remains the benchmark.

## Current Best CPU Fit

- Best variant: `sc_overlap_compact_pca_core_013_2f3509c08cf4`
- Weighted score: `0.3137`
- Bad 64D distance: `0.5936` vs prior baseline `~0.748`
- Medium 64D distance: `0.2434` vs prior baseline `~0.311`

## Figures

- `figures/top_rules_64d_overlay.png`: CleanBUT-Core background with top targeted PTB rules.
- `figures/best_rule_64d_overlay.png`: best current candidate in the same PCA space.
- `figures/bad_core_centroid_shift.png`: class centroid gaps in CleanBUT 64D PCA.
- `figures/classwise_distance_bars.png`: class-wise distance leaderboard.

## Top No-Training Candidates

| rank | variant_id | family | score | class_worst_64d_KS | good_64d_KS | medium_64d_KS | bad_64d_KS | bad_distance_improvement_vs_baseline | medium_regression_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sc_overlap_compact_pca_core_013_2f3509c08cf4 | bad_compact_pca_core | 0.3137 | 0.5936 | 0.3840 | 0.2434 | 0.5936 | 0.2065 | -0.2168 |
| 2 | sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | bad_1530_locked_lowpc2_core | 0.3192 | 0.6550 | 0.3310 | 0.2449 | 0.6550 | 0.1243 | -0.2122 |
| 3 | sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | bad_1530_locked_lowpc2_core | 0.3261 | 0.6687 | 0.3629 | 0.2378 | 0.6687 | 0.1060 | -0.2349 |
| 4 | sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | bad_1530_locked_lowpc2_core | 0.3286 | 0.6717 | 0.3500 | 0.2434 | 0.6717 | 0.1020 | -0.2168 |
| 5 | sc_overlap_1530_locked_lowpc2_core_006_e5e5057f4973 | bad_1530_locked_lowpc2_core | 0.3326 | 0.6756 | 0.3681 | 0.2449 | 0.6756 | 0.0968 | -0.2122 |
| 6 | sc_overlap_compact_pca_core_014_5e87749494a8 | bad_compact_pca_core | 0.3352 | 0.6333 | 0.3500 | 0.2434 | 0.6333 | 0.1533 | -0.2168 |
| 7 | sc_overlap_1530_locked_lowpc2_core_007_d70e8787c022 | bad_1530_locked_lowpc2_core | 0.3364 | 0.6756 | 0.3951 | 0.2449 | 0.6756 | 0.0968 | -0.2122 |
| 8 | sc_overlap_compact_pca_core_011_263ba3f411cb | bad_compact_pca_core | 0.3371 | 0.6361 | 0.3568 | 0.2378 | 0.6361 | 0.1496 | -0.2349 |
| 9 | sc_overlap_compact_pca_core_010_c36b12f5d8db | bad_compact_pca_core | 0.3386 | 0.6439 | 0.3996 | 0.2378 | 0.6439 | 0.1391 | -0.2349 |
| 10 | sc_overlap_compact_pca_core_012_6e8fb3bc5ca8 | bad_compact_pca_core | 0.3388 | 0.6357 | 0.3613 | 0.2434 | 0.6357 | 0.1502 | -0.2168 |
| 11 | sc_overlap_1530_hfedge_spike_core_017_154377ba421c | bad_1530_hfedge_spike_core | 0.3466 | 0.6111 | 0.3528 | 0.2449 | 0.6111 | 0.1830 | -0.2122 |
| 12 | sc_overlap_1530_hfedge_spike_core_016_82087c83915a | bad_1530_hfedge_spike_core | 0.3538 | 0.6111 | 0.3999 | 0.2449 | 0.6111 | 0.1830 | -0.2122 |
| 13 | sc_overlap_qrs_visible_compact_core_003_eda02d8ffad5 | bad_qrs_visible_compact_core | 0.4456 | 0.5541 | 0.3393 | 0.2434 | 0.5541 | 0.2593 | -0.2168 |
| 14 | sc_overlap_qrs_visible_compact_core_002_8b19fad8c8bf | bad_qrs_visible_compact_core | 0.4553 | 0.6006 | 0.3528 | 0.2378 | 0.6006 | 0.1971 | -0.2349 |
| 15 | sc_overlap_qrs_visible_compact_core_001_4bbcde62404f | bad_qrs_visible_compact_core | 0.4625 | 0.6006 | 0.3999 | 0.2378 | 0.6006 | 0.1971 | -0.2349 |
| 16 | sc_overlap_qrs_visible_compact_core_000_64c2418873c8 | bad_qrs_visible_compact_core | 0.4688 | 0.7596 | 0.3629 | 0.2378 | 0.7596 | -0.0155 | -0.2349 |
| 17 | sc_overlap_qrs_visible_compact_core_004_e3fe0a939f9d | bad_qrs_visible_compact_core | 0.4804 | 0.7536 | 0.3996 | 0.2434 | 0.7536 | -0.0075 | -0.2168 |
| 18 | sc_overlap_1530_hfedge_spike_core_015_db4c41318cdb | bad_1530_hfedge_spike_core | 0.4884 | 0.7258 | 0.3629 | 0.2449 | 0.7258 | 0.0296 | -0.2122 |

## Training Results

| mode | variant_id | acc | macro_f1 | good_recall | medium_recall | bad_recall | balanced_macro | clean_diag_macro | score | bad_64d_KS | medium_64d_KS | good_64d_KS | domain_separability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quick | sc_overlap_compact_pca_core_013_2f3509c08cf4 | 0.6505 | 0.4442 | 0.4978 | 0.8353 | 0.0122 | 0.3842 | 0.9772 | 0.3137 | 0.5936 | 0.2434 | 0.3840 | 1.0000 |
| quick | sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | 0.6749 | 0.4545 | 0.5459 | 0.8434 | 0.0024 | 0.3890 | 0.9629 | 0.3192 | 0.6550 | 0.2449 | 0.3310 | 1.0000 |
| quick | sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | 0.5797 | 0.4877 | 0.3857 | 0.7605 | 0.3504 | 0.4925 | 0.8075 | 0.3261 | 0.6687 | 0.2378 | 0.3629 | 1.0000 |
| quick | sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | 0.6649 | 0.4456 | 0.5330 | 0.8351 | 0.0000 | 0.3802 | 0.5144 | 0.3286 | 0.6717 | 0.2434 | 0.3500 | 1.0000 |
| quick | sc_overlap_1530_locked_lowpc2_core_006_e5e5057f4973 | 0.6703 | 0.4443 | 0.4887 | 0.8818 | 0.0000 | 0.3574 | 0.9174 | 0.3326 | 0.6756 | 0.2449 | 0.3681 | 1.0000 |
| quick | sc_overlap_compact_pca_core_014_5e87749494a8 | 0.4870 | 0.3442 | 0.0618 | 0.8396 | 0.4550 | 0.3956 | 0.7555 | 0.3352 | 0.6333 | 0.2434 | 0.3500 | 1.0000 |
| quick | sc_overlap_qrs_visible_compact_core_003_eda02d8ffad5 | 0.6518 | 0.5239 | 0.5069 | 0.8152 | 0.1752 | 0.4757 | 0.9598 | 0.4456 | 0.5541 | 0.2434 | 0.3393 | 1.0000 |
| quick | sc_overlap_qrs_visible_compact_core_002_8b19fad8c8bf | 0.6729 | 0.4545 | 0.4745 | 0.8972 | 0.0146 | 0.3860 | 0.8862 | 0.4553 | 0.6006 | 0.2378 | 0.3528 | 1.0000 |

## Notes

- Selection uses CleanBUT train-target core features and does not inspect BUT test predictions.
- Synthetic `sqi_iSQI` remains a single-lead detector-agreement proxy.
- `all` defaults to CPU distribution fitting only; pass `--run_training` for quick/full training after visual review.
