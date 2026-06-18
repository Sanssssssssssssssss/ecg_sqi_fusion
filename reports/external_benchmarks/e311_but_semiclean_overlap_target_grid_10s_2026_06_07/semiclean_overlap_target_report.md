# SemiCleanBUT-Overlap Target Fit

SemiCleanBUT is a filtered diagnostic/generator target. It keeps good/medium overlap while forcing bad onto the separated right island; formal original BUT remains only a reference metric.

## Target Definition

- Selected counts: `{'good': 4800, 'medium': 4800, 'bad': 4800}`
- Ambiguous fraction: `0.553`
- Bad selected with pc1 < 9.0: `0.0000`
- Bad rule: `bad & pc1 >= 9.0 & knn_label_purity >= 0.95 & pca_margin >= 8.0` by default.
- Good/medium rule: relaxed label-purity/margin with isolated outliers removed, then top confidence per class.

## Best CPU 64D Fit

- Best variant: `sc_overlap_compact_pca_core_013_2f3509c08cf4`
- SemiClean score: `0.3137`
- 64D KS good/medium/bad: `0.384/0.243/0.594`

## Top CPU Candidates

| rank | variant_id | family | semiclean_score | good_64d_KS | medium_64d_KS | bad_64d_KS | good_PCA_range_gap | medium_PCA_range_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sc_overlap_compact_pca_core_013_2f3509c08cf4 | bad_compact_pca_core | 0.3137 | 0.3840 | 0.2434 | 0.5936 | 0.0219 | 0.0213 |
| 2 | sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | bad_1530_locked_lowpc2_core | 0.3192 | 0.3310 | 0.2449 | 0.6550 | 0.0165 | 0.0259 |
| 3 | sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | bad_1530_locked_lowpc2_core | 0.3261 | 0.3629 | 0.2378 | 0.6687 | 0.0157 | 0.0295 |
| 4 | sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | bad_1530_locked_lowpc2_core | 0.3286 | 0.3500 | 0.2434 | 0.6717 | 0.0204 | 0.0213 |
| 5 | sc_overlap_1530_locked_lowpc2_core_006_e5e5057f4973 | bad_1530_locked_lowpc2_core | 0.3326 | 0.3681 | 0.2449 | 0.6756 | 0.0179 | 0.0259 |
| 6 | sc_overlap_compact_pca_core_014_5e87749494a8 | bad_compact_pca_core | 0.3352 | 0.3500 | 0.2434 | 0.6333 | 0.0204 | 0.0213 |
| 7 | sc_overlap_1530_locked_lowpc2_core_007_d70e8787c022 | bad_1530_locked_lowpc2_core | 0.3364 | 0.3951 | 0.2449 | 0.6756 | 0.0126 | 0.0259 |
| 8 | sc_overlap_compact_pca_core_011_263ba3f411cb | bad_compact_pca_core | 0.3371 | 0.3568 | 0.2378 | 0.6361 | 0.0151 | 0.0295 |
| 9 | sc_overlap_compact_pca_core_010_c36b12f5d8db | bad_compact_pca_core | 0.3386 | 0.3996 | 0.2378 | 0.6439 | 0.0141 | 0.0295 |
| 10 | sc_overlap_compact_pca_core_012_6e8fb3bc5ca8 | bad_compact_pca_core | 0.3388 | 0.3613 | 0.2434 | 0.6357 | 0.0180 | 0.0213 |
| 11 | sc_overlap_1530_hfedge_spike_core_017_154377ba421c | bad_1530_hfedge_spike_core | 0.3466 | 0.3528 | 0.2449 | 0.6111 | 0.0179 | 0.0259 |
| 12 | sc_overlap_1530_hfedge_spike_core_016_82087c83915a | bad_1530_hfedge_spike_core | 0.3538 | 0.3999 | 0.2449 | 0.6111 | 0.0131 | 0.0259 |
| 13 | sc_overlap_qrs_visible_compact_core_003_eda02d8ffad5 | bad_qrs_visible_compact_core | 0.4456 | 0.3393 | 0.2434 | 0.5541 | 0.0146 | 0.0213 |
| 14 | sc_overlap_qrs_visible_compact_core_002_8b19fad8c8bf | bad_qrs_visible_compact_core | 0.4553 | 0.3528 | 0.2378 | 0.6006 | 0.0179 | 0.0295 |
| 15 | sc_overlap_qrs_visible_compact_core_001_4bbcde62404f | bad_qrs_visible_compact_core | 0.4625 | 0.3999 | 0.2378 | 0.6006 | 0.0131 | 0.0295 |
| 16 | sc_overlap_qrs_visible_compact_core_000_64c2418873c8 | bad_qrs_visible_compact_core | 0.4688 | 0.3629 | 0.2378 | 0.7596 | 0.0157 | 0.0295 |

## SemiClean Training Diagnostics

| status | variant_id | prediction_mode | n_per_class | acc | macro_f1 | good_recall | medium_recall | bad_recall | ambiguous_fraction | original_but_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| above_target_or_closest | sc_overlap_qrs_visible_compact_core_002_8b19fad8c8bf | calibrated | 2400 | 0.8906 | 0.8882 | 0.6896 | 0.9821 | 1.0000 | 0.2267 | 0.4545 |
| above_target_or_closest | sc_overlap_qrs_visible_compact_core_002_8b19fad8c8bf | raw | 2400 | 0.8856 | 0.8827 | 0.6721 | 0.9846 | 1.0000 | 0.2267 | 0.4545 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | calibrated | 2400 | 0.8257 | 0.8190 | 0.5642 | 0.9129 | 1.0000 | 0.2267 | 0.4877 |
| above_target_or_closest | sc_overlap_compact_pca_core_014_5e87749494a8 | calibrated | 2400 | 0.7837 | 0.7632 | 0.3892 | 0.9621 | 1.0000 | 0.2267 | 0.3442 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | raw | 2400 | 0.7674 | 0.7501 | 0.4133 | 0.8888 | 1.0000 | 0.2267 | 0.4877 |
| above_target_or_closest | sc_overlap_compact_pca_core_014_5e87749494a8 | raw | 2400 | 0.7661 | 0.7418 | 0.3550 | 0.9433 | 1.0000 | 0.2267 | 0.3442 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_006_e5e5057f4973 | calibrated | 1200 | 0.9333 | 0.9336 | 0.8358 | 0.9933 | 0.9708 | 0.0008 | 0.4443 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_006_e5e5057f4973 | raw | 1200 | 0.8900 | 0.8906 | 0.7517 | 0.9983 | 0.9200 | 0.0008 | 0.4443 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | raw | 1200 | 0.6292 | 0.5252 | 0.9283 | 0.9583 | 0.0008 | 0.0008 | 0.4456 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | calibrated | 600 | 0.6322 | 0.5296 | 0.9283 | 0.9600 | 0.0083 | 0.0000 | 0.4456 |
| in_0p95_band_max_coverage | sc_overlap_compact_pca_core_013_2f3509c08cf4 | raw | 4802 | 0.9404 | 0.9403 | 0.9456 | 0.8784 | 0.9971 | 0.5535 | 0.4442 |
| in_0p95_band_max_coverage | sc_overlap_compact_pca_core_013_2f3509c08cf4 | calibrated | 4200 | 0.9471 | 0.9470 | 0.9567 | 0.8845 | 1.0000 | 0.4895 | 0.4442 |
| in_0p95_band_max_coverage | sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | calibrated | 4200 | 0.9442 | 0.9441 | 0.8717 | 0.9610 | 1.0000 | 0.4895 | 0.4545 |
| in_0p95_band_max_coverage | sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | raw | 3600 | 0.9472 | 0.9470 | 0.8603 | 0.9814 | 1.0000 | 0.4107 | 0.4545 |
| in_0p95_band_max_coverage | sc_overlap_qrs_visible_compact_core_003_eda02d8ffad5 | raw | 2400 | 0.9582 | 0.9581 | 0.9871 | 0.8875 | 1.0000 | 0.2267 | 0.5239 |
| in_0p95_band_max_coverage | sc_overlap_qrs_visible_compact_core_003_eda02d8ffad5 | calibrated | 2400 | 0.9568 | 0.9566 | 0.9896 | 0.8808 | 1.0000 | 0.2267 | 0.5239 |

## Figures

- `figures/semiclean_target_pca.png`: selected SemiClean target and lowest-confidence shell.
- `figures/top_rules_semiclean_overlay.png`: top synthetic rules against SemiClean target.
- `figures/best_semiclean_overlay.png`: best current synthetic fit.
- `figures/semiclean_relaxation_curve.png`: filtered diagnostic accuracy as boundary is relaxed.
- `semiclean_boundary_waveform_galleries/`: lowest-confidence included examples.

## Caveat

This filtered diagnostic deliberately uses all splits for a class-balanced target because the formal CleanBUT clean-core test split is not class-complete for bad. It is not a replacement benchmark.
