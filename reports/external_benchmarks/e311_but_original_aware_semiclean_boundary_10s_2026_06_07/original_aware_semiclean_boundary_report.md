# Original-Aware SemiCleanBUT Boundary

This package widens the SemiClean diagnostic boundary with prediction-free original-region quotas. It does not replace formal original BUT 10s P1; original test remains report-only.

## Region Atlas

| class_name | original_region | n | fraction_within_class | median_confidence | median_pc1 | median_pc2 |
| --- | --- | --- | --- | --- | --- | --- |
| good | clean_core | 3264 | 0.1915 | 1.0341 | -5.3177 | -1.9745 |
| good | good_medium_overlap | 9243 | 0.5423 | 0.6689 | -3.4703 | -1.1210 |
| good | outlier_low_confidence | 4536 | 0.2662 | 0.4470 | -4.5811 | 6.1408 |
| medium | clean_core | 1970 | 0.1854 | 1.0416 | 0.5391 | 1.3089 |
| medium | good_medium_overlap | 5127 | 0.4824 | 0.6697 | -0.4171 | -0.7394 |
| medium | medium_bad_overlap | 9 | 0.0008 | 0.6740 | 4.2098 | -0.2722 |
| medium | outlier_low_confidence | 3522 | 0.3314 | 0.2661 | -2.3738 | 10.0268 |
| bad | near_bad_boundary | 120 | 0.0227 | 0.3906 | 9.0042 | -0.9591 |
| bad | outlier_low_confidence | 1201 | 0.2272 | 0.4405 | 10.3169 | 0.0395 |
| bad | right_bad_island | 3964 | 0.7500 | 0.7572 | 10.4236 | -0.0348 |

## Region-Quota Boundary Levels

| level_n_per_class | class_name | selected_n | ambiguous_fraction | outlier_fraction | region_clean_core | region_good_medium_overlap | region_medium_bad_overlap | region_right_bad_island | region_near_bad_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3600 | good | 3600 | 0.3400 | 0.0000 | 2376 | 1224 | 0 | 0 | 0 |
| 3600 | medium | 3600 | 0.4528 | 0.0000 | 1970 | 1621 | 9 | 0 | 0 |
| 3600 | bad | 3600 | 0.6672 | 0.0000 | 0 | 0 | 0 | 3600 | 0 |
| 4200 | good | 4200 | 0.4400 | 0.0000 | 2352 | 1848 | 0 | 0 | 0 |
| 4200 | medium | 4200 | 0.5310 | 0.0000 | 1970 | 2221 | 9 | 0 | 0 |
| 4200 | bad | 4200 | 0.7148 | 0.0276 | 0 | 0 | 0 | 3964 | 120 |
| 4800 | good | 4800 | 0.5500 | 0.0000 | 2160 | 2640 | 0 | 0 | 0 |
| 4800 | medium | 4800 | 0.5896 | 0.0000 | 1970 | 2821 | 9 | 0 | 0 |
| 4800 | bad | 4800 | 0.7504 | 0.1492 | 0 | 0 | 0 | 3964 | 120 |
| 5200 | good | 5200 | 0.6200 | 0.0200 | 1976 | 3120 | 0 | 0 | 0 |
| 5200 | medium | 5200 | 0.6212 | 0.0000 | 1970 | 3221 | 9 | 0 | 0 |
| 5200 | bad | 5200 | 0.7696 | 0.2146 | 0 | 0 | 0 | 3964 | 120 |

## Best Filtered Diagnostics

| status | variant_id | prediction_mode | level_n_per_class | acc | macro_f1 | good_recall | medium_recall | bad_recall | ambiguous_fraction | original_but_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| above_target_or_closest | sc_overlap_qrs_visible_compact_core_003_eda02d8ffad5 | raw | 3600 | 0.9369 | 0.9366 | 0.9658 | 0.8447 | 1.0000 | 0.4867 | 0.5239 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | raw | 3600 | 0.9357 | 0.9353 | 0.8250 | 0.9822 | 1.0000 | 0.4867 | 0.4545 |
| above_target_or_closest | sc_overlap_qrs_visible_compact_core_002_8b19fad8c8bf | raw | 3600 | 0.8496 | 0.8428 | 0.5658 | 0.9831 | 1.0000 | 0.4867 | 0.4545 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_006_e5e5057f4973 | raw | 3600 | 0.8404 | 0.8390 | 0.6253 | 0.9956 | 0.9003 | 0.4867 | 0.4443 |
| above_target_or_closest | sc_overlap_compact_pca_core_014_5e87749494a8 | raw | 3600 | 0.7343 | 0.6942 | 0.2508 | 0.9519 | 1.0000 | 0.4867 | 0.3442 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | raw | 3600 | 0.7294 | 0.6958 | 0.2850 | 0.9033 | 1.0000 | 0.4867 | 0.4877 |
| above_target_or_closest | sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | raw | 3600 | 0.5916 | 0.4942 | 0.8533 | 0.9208 | 0.0006 | 0.4867 | 0.4456 |
| in_0p95_band_max_coverage | sc_overlap_compact_pca_core_013_2f3509c08cf4 | raw | 3600 | 0.9544 | 0.9543 | 0.9483 | 0.9147 | 1.0000 | 0.4867 | 0.4442 |

## Original Region Accuracy

| variant_id | split | original_region | n | acc | macro_f1 | class_counts |
| --- | --- | --- | --- | --- | --- | --- |
| sc_overlap_compact_pca_core_013_2f3509c08cf4 | val | clean_core | 141 | 1.0000 | 0.6667 | {"good": 117, "medium": 24, "bad": 0} |
| sc_overlap_compact_pca_core_013_2f3509c08cf4 | val | good_medium_overlap | 639 | 0.8717 | 0.4714 | {"good": 597, "medium": 42, "bad": 0} |
| sc_overlap_compact_pca_core_013_2f3509c08cf4 | val | outlier_low_confidence | 376 | 0.4840 | 0.5232 | {"good": 255, "medium": 39, "bad": 82} |
| sc_overlap_compact_pca_core_013_2f3509c08cf4 | test | clean_core | 569 | 0.9789 | 0.6174 | {"good": 38, "medium": 531, "bad": 0} |
| sc_overlap_compact_pca_core_013_2f3509c08cf4 | test | good_medium_overlap | 2902 | 0.8680 | 0.5753 | {"good": 1191, "medium": 1711, "bad": 0} |
| sc_overlap_compact_pca_core_013_2f3509c08cf4 | test | near_bad_boundary | 119 | 0.0000 | 0.0000 | {"good": 0, "medium": 0, "bad": 119} |
| sc_overlap_compact_pca_core_013_2f3509c08cf4 | test | outlier_low_confidence | 4882 | 0.5324 | 0.3509 | {"good": 2411, "medium": 2179, "bad": 292} |
| sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | val | clean_core | 141 | 1.0000 | 0.6667 | {"good": 117, "medium": 24, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | val | good_medium_overlap | 639 | 0.6338 | 0.3388 | {"good": 597, "medium": 42, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | val | outlier_low_confidence | 376 | 0.3298 | 0.4265 | {"good": 255, "medium": 39, "bad": 82} |
| sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | test | clean_core | 569 | 0.9701 | 0.5896 | {"good": 38, "medium": 531, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | test | good_medium_overlap | 2902 | 0.8815 | 0.5832 | {"good": 1191, "medium": 1711, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | test | near_bad_boundary | 119 | 0.0000 | 0.0000 | {"good": 0, "medium": 0, "bad": 119} |
| sc_overlap_1530_locked_lowpc2_core_008_339f9da4a1fa | test | outlier_low_confidence | 4882 | 0.5422 | 0.3645 | {"good": 2411, "medium": 2179, "bad": 292} |
| sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | val | clean_core | 141 | 0.3404 | 0.2287 | {"good": 117, "medium": 24, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | val | good_medium_overlap | 639 | 0.1111 | 0.0741 | {"good": 597, "medium": 42, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | val | outlier_low_confidence | 376 | 0.2606 | 0.3634 | {"good": 255, "medium": 39, "bad": 82} |
| sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | test | clean_core | 569 | 0.7944 | 0.4914 | {"good": 38, "medium": 531, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | test | good_medium_overlap | 2902 | 0.7629 | 0.5007 | {"good": 1191, "medium": 1711, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | test | near_bad_boundary | 119 | 1.0000 | 0.3333 | {"good": 0, "medium": 0, "bad": 119} |
| sc_overlap_1530_locked_lowpc2_core_009_d4aa8154e2b5 | test | outlier_low_confidence | 4882 | 0.4431 | 0.3273 | {"good": 2411, "medium": 2179, "bad": 292} |
| sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | val | clean_core | 141 | 1.0000 | 0.6667 | {"good": 117, "medium": 24, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | val | good_medium_overlap | 639 | 0.7778 | 0.4086 | {"good": 597, "medium": 42, "bad": 0} |
| sc_overlap_1530_locked_lowpc2_core_005_8d4343ee0314 | val | outlier_low_confidence | 376 | 0.3457 | 0.4305 | {"good": 255, "medium": 39, "bad": 82} |

## Files

- `original_region_atlas.csv`: original-region labels for every BUT window.
- `region_quota_boundary_manifest.csv`: level selections plus sample weights and soft labels for boundary-aware training.
- `region_quota_boundary_metrics.csv`: region-quota filtered diagnostic metrics.
- `original_region_model_metrics.csv`: original BUT model behavior by region and split.
- `figures/original_region_atlas_pca.png`: original labels and region atlas.
- `figures/region_quota_boundary_levels_pca.png`: boundary-level selections.
- `figures/region_quota_relaxation_curve.png`: acc as boundary widens.
- `figures/original_region_accuracy_heatmap.png`: val/test region accuracy for the best original macro candidate.
- `original_region_waveform_galleries/`: overlap, bad island, near-boundary, and boundary-shell examples.

## Interpretation

If the 4800/class level remains below 0.95, inspect medium recall and the good-medium overlap rows first. Bad is intentionally mostly locked to the right island, with only a controlled near-boundary quota as the level widens.
