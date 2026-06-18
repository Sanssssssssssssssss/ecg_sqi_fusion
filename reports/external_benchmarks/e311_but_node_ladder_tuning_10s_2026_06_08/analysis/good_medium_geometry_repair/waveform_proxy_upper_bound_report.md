# Waveform Proxy Upper-Bound Diagnostic

This is a diagnostic upper bound, not a final model.  It trains simple classifiers on deterministic statistics computed from waveform channels only.

Rules held fixed:

- Synthetic train is used for fitting.
- Synthetic val is used for candidate selection.
- Synthetic test and BUT buckets are report-only.
- No 47-column SQI/geometry table is used as model input.

Generated: `2026-06-15T11:23:24`
Stat set: `extended`; channel mode: `robust3`; feature dim: `279`
Selected by synthetic val: `proxy_extra_trees_balanced` with selection score `1.489652`

## Metrics

| Candidate | Split/Bucket | Selected | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proxy_logreg_l2 | synthetic_val | False | 0.986620 | 0.981950 | 1.000000 | 0.979167 | 0.996094 | 0 | 0 | 1 |
| proxy_logreg_l2 | synthetic_test | False | 0.997950 | 0.997721 | 0.997908 | 0.998377 | 0.995851 | 1 | 2 | 1 |
| proxy_hgb_l2_leaf | synthetic_val | False | 0.986038 | 0.981620 | 0.997543 | 0.978220 | 1.000000 | 1 | 1 | 0 |
| proxy_hgb_l2_leaf | synthetic_test | False | 0.993849 | 0.993173 | 0.995816 | 0.993506 | 0.991701 | 2 | 7 | 2 |
| proxy_extra_trees_balanced | synthetic_val | True | 0.994183 | 0.993596 | 0.992629 | 0.996212 | 0.988281 | 3 | 4 | 3 |
| proxy_extra_trees_balanced | synthetic_test | True | 0.994874 | 0.994126 | 0.995816 | 0.995942 | 0.987552 | 2 | 5 | 3 |
| proxy_random_forest_balanced | synthetic_val | False | 0.981966 | 0.977401 | 0.992629 | 0.974432 | 0.996094 | 3 | 5 | 1 |
| proxy_random_forest_balanced | synthetic_test | False | 0.994362 | 0.993637 | 0.991632 | 0.996753 | 0.987552 | 4 | 4 | 3 |
| proxy_logreg_l2 | original_test_all_10s+ | False | 0.827415 | 0.702411 | 0.920879 | 0.799593 | 0.299270 | 288 | 841 | 30 |
| proxy_logreg_l2_badcal_synthval | original_test_all_10s+ | False | 0.826943 | 0.701259 | 0.920879 | 0.798690 | 0.299270 | 288 | 841 | 30 |
| proxy_logreg_l2 | original_all_10s+ | False | 0.839483 | 0.858484 | 0.801971 | 0.860369 | 0.918448 | 3320 | 1396 | 98 |
| proxy_logreg_l2_badcal_synthval | original_all_10s+ | False | 0.839149 | 0.857921 | 0.801913 | 0.859334 | 0.918638 | 3318 | 1396 | 97 |
| proxy_logreg_l2 | bad_core_nearboundary | False | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| proxy_logreg_l2_badcal_synthval | bad_core_nearboundary | False | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| proxy_logreg_l2 | bad_outlier_stress | False | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 30 |
| proxy_logreg_l2_badcal_synthval | bad_outlier_stress | False | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 30 |
| proxy_hgb_l2_leaf | original_test_all_10s+ | False | 0.767017 | 0.664653 | 0.929670 | 0.677361 | 0.291971 | 255 | 1412 | 29 |
| proxy_hgb_l2_leaf_badcal_synthval | original_test_all_10s+ | False | 0.710747 | 0.568466 | 0.922802 | 0.573430 | 0.311436 | 186 | 1339 | 23 |
| proxy_hgb_l2_leaf | original_all_10s+ | False | 0.825373 | 0.847665 | 0.809717 | 0.806361 | 0.914096 | 3241 | 2034 | 180 |
| proxy_hgb_l2_leaf_badcal_synthval | original_all_10s+ | False | 0.807501 | 0.815742 | 0.807311 | 0.744825 | 0.934153 | 3120 | 1950 | 77 |
| proxy_hgb_l2_leaf | bad_core_nearboundary | False | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| proxy_hgb_l2_leaf_badcal_synthval | bad_core_nearboundary | False | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| proxy_hgb_l2_leaf | bad_outlier_stress | False | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 29 |
| proxy_hgb_l2_leaf_badcal_synthval | bad_outlier_stress | False | 0.030822 | 0.019934 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 23 |
| proxy_extra_trees_balanced | original_test_all_10s+ | True | 0.782116 | 0.678393 | 0.898901 | 0.731812 | 0.289538 | 368 | 1187 | 27 |
| proxy_extra_trees_balanced_badcal_synthval | original_test_all_10s+ | True | 0.782116 | 0.678393 | 0.898901 | 0.731812 | 0.289538 | 368 | 1187 | 27 |
| proxy_extra_trees_balanced | original_all_10s+ | True | 0.828499 | 0.854363 | 0.781259 | 0.852747 | 0.932072 | 3728 | 1565 | 90 |
| proxy_extra_trees_balanced_badcal_synthval | original_all_10s+ | True | 0.828529 | 0.854408 | 0.781259 | 0.852747 | 0.932261 | 3728 | 1565 | 89 |
| proxy_extra_trees_balanced | bad_core_nearboundary | True | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| proxy_extra_trees_balanced_badcal_synthval | bad_core_nearboundary | True | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| proxy_extra_trees_balanced | bad_outlier_stress | True | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 27 |
| proxy_extra_trees_balanced_badcal_synthval | bad_outlier_stress | True | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 27 |
| proxy_random_forest_balanced | original_test_all_10s+ | False | 0.796272 | 0.687123 | 0.901648 | 0.756891 | 0.287105 | 358 | 1076 | 32 |
| proxy_random_forest_balanced_badcal_synthval | original_test_all_10s+ | False | 0.778813 | 0.647772 | 0.901648 | 0.723000 | 0.291971 | 352 | 1076 | 30 |
| proxy_random_forest_balanced | original_all_10s+ | False | 0.816604 | 0.845425 | 0.755442 | 0.857452 | 0.931693 | 4168 | 1515 | 92 |
| proxy_random_forest_balanced_badcal_synthval | original_all_10s+ | False | 0.809413 | 0.833840 | 0.755442 | 0.833835 | 0.934342 | 4162 | 1515 | 78 |
| proxy_random_forest_balanced | bad_core_nearboundary | False | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| proxy_random_forest_balanced_badcal_synthval | bad_core_nearboundary | False | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| proxy_random_forest_balanced | bad_outlier_stress | False | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 31 |
| proxy_random_forest_balanced_badcal_synthval | bad_outlier_stress | False | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 30 |

## Interpretation

- If this upper bound is still far below the 47-feature tabular model on BUT, the missing piece is not just Transformer capacity; the waveform-derived synthetic proxy distribution is not aligned enough with BUT.
- If this upper bound is high on BUT, then the next Transformer should explicitly learn these proxy statistics at patch/encoder level.
- The `_badcal_synthval` rows use a bad threshold calibrated on synthetic val only. BUT labels are never used for threshold selection.
