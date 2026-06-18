# Waveform SQI-Plus Search

Strict input contract: deterministic SQI/statistics are computed from the ECG waveform at inference. No 47-column sidecar and no original BUT labels are used for training or selection.

## Synthetic-Val Selected View

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sqi_plus_histgb_l2_balanced | synthetic_val | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 | 0 | 0 | 1 |
| sqi_plus_histgb_l2_balanced | synthetic_test | 0.997437 | 0.996888 | 0.995816 | 0.999188 | 0.991701 | 2 | 1 | 2 |
| sqi_plus_histgb_l2_balanced | original_test_all_10s+ | 0.848413 | 0.721752 | 0.947802 | 0.818798 | 0.287105 | 190 | 797 | 180 |
| sqi_plus_histgb_l2_balanced | original_all_10s+ | 0.871708 | 0.887332 | 0.855190 | 0.868178 | 0.932072 | 2468 | 1396 | 245 |
| sqi_plus_histgb_l2_balanced | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| sqi_plus_histgb_l2_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 179 |
| sqi_plus_histgb_lite_balanced | synthetic_val | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 | 0 | 0 | 1 |
| sqi_plus_histgb_lite_balanced | synthetic_test | 0.996925 | 0.996405 | 0.995816 | 0.998377 | 0.991701 | 2 | 2 | 2 |
| sqi_plus_histgb_lite_balanced | original_test_all_10s+ | 0.841925 | 0.718198 | 0.951648 | 0.802982 | 0.289538 | 176 | 867 | 162 |
| sqi_plus_histgb_lite_balanced | original_all_10s+ | 0.867187 | 0.883795 | 0.850965 | 0.860369 | 0.933207 | 2540 | 1479 | 222 |
| sqi_plus_histgb_lite_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqi_plus_histgb_lite_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 162 |
| sqi_plus_histgb_badguard | synthetic_val | 0.999418 | 0.999190 | 1.000000 | 1.000000 | 0.996094 | 0 | 0 | 1 |
| sqi_plus_histgb_badguard | synthetic_test | 0.997437 | 0.996888 | 0.995816 | 0.999188 | 0.991701 | 2 | 1 | 2 |
| sqi_plus_histgb_badguard | original_test_all_10s+ | 0.842987 | 0.716855 | 0.947802 | 0.808631 | 0.284672 | 190 | 841 | 180 |
| sqi_plus_histgb_badguard | original_all_10s+ | 0.869553 | 0.885399 | 0.856011 | 0.860181 | 0.932072 | 2454 | 1480 | 244 |
| sqi_plus_histgb_badguard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| sqi_plus_histgb_badguard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 178 |
| sqi_plus_rf_depth10 | synthetic_val | 0.995928 | 0.995548 | 0.997543 | 0.996212 | 0.992188 | 1 | 4 | 2 |
| sqi_plus_rf_depth10 | synthetic_test | 0.993337 | 0.991970 | 0.995816 | 0.995130 | 0.979253 | 2 | 6 | 5 |
| sqi_plus_rf_depth10 | original_test_all_10s+ | 0.840274 | 0.718112 | 0.914835 | 0.830095 | 0.289538 | 310 | 752 | 146 |
| sqi_plus_rf_depth10 | original_all_10s+ | 0.852409 | 0.873634 | 0.798803 | 0.898758 | 0.932072 | 3429 | 1076 | 212 |
| sqi_plus_rf_depth10 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqi_plus_rf_depth10 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 146 |
| sqi_plus_logreg | synthetic_val | 0.995346 | 0.994988 | 0.997543 | 0.994318 | 0.996094 | 1 | 5 | 1 |
| sqi_plus_logreg | synthetic_test | 0.994874 | 0.994819 | 0.991632 | 0.995942 | 0.995851 | 4 | 5 | 1 |
| sqi_plus_logreg | original_test_all_10s+ | 0.766309 | 0.531447 | 0.884066 | 0.739494 | 0.012165 | 422 | 1144 | 145 |
| sqi_plus_logreg | original_all_10s+ | 0.802737 | 0.829581 | 0.748636 | 0.843056 | 0.896121 | 4284 | 1633 | 286 |
| sqi_plus_logreg | bad_core_nearboundary | 0.033613 | 0.021680 | 0.000000 | 0.000000 | 0.033613 | 0 | 0 | 115 |
| sqi_plus_logreg | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 30 |
| sqi_plus_extra_depth12 | synthetic_val | 0.995346 | 0.994725 | 0.992629 | 0.998106 | 0.988281 | 3 | 2 | 3 |
| sqi_plus_extra_depth12 | synthetic_test | 0.995387 | 0.993896 | 0.995816 | 0.998377 | 0.979253 | 2 | 2 | 5 |
| sqi_plus_extra_depth12 | original_test_all_10s+ | 0.843577 | 0.720220 | 0.905769 | 0.843877 | 0.289538 | 343 | 691 | 124 |
| sqi_plus_extra_depth12 | original_all_10s+ | 0.814874 | 0.845193 | 0.720413 | 0.908355 | 0.931504 | 4765 | 974 | 193 |
| sqi_plus_extra_depth12 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqi_plus_extra_depth12 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 124 |
| sqi_plus_mlp_small | synthetic_val | 0.994183 | 0.993378 | 0.997543 | 0.992424 | 0.996094 | 1 | 5 | 1 |
| sqi_plus_mlp_small | synthetic_test | 0.992824 | 0.992162 | 0.981172 | 0.998377 | 0.987552 | 9 | 2 | 3 |
| sqi_plus_mlp_small | original_test_all_10s+ | 0.771381 | 0.526855 | 0.860714 | 0.769544 | 0.000000 | 507 | 1019 | 158 |
| sqi_plus_mlp_small | original_all_10s+ | 0.659849 | 0.476354 | 0.736842 | 0.864509 | 0.000000 | 4485 | 1439 | 5030 |
| sqi_plus_mlp_small | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 |
| sqi_plus_mlp_small | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 39 |
| sqi_plus_extra_depth8 | synthetic_val | 0.993019 | 0.991689 | 0.985258 | 1.000000 | 0.976562 | 6 | 0 | 6 |
| sqi_plus_extra_depth8 | synthetic_test | 0.994874 | 0.993409 | 0.991632 | 0.999188 | 0.979253 | 4 | 1 | 5 |
| sqi_plus_extra_depth8 | original_test_all_10s+ | 0.836263 | 0.715063 | 0.884890 | 0.847040 | 0.289538 | 419 | 677 | 134 |
| sqi_plus_extra_depth8 | original_all_10s+ | 0.792936 | 0.828514 | 0.674001 | 0.914753 | 0.931504 | 5556 | 906 | 203 |
| sqi_plus_extra_depth8 | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| sqi_plus_extra_depth8 | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 134 |

## Best Original Test Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---:|---:|---:|---:|---:|
| sqi_plus_histgb_l2_balanced | 0.848413 | 0.721752 | 0.947802 | 0.818798 | 0.287105 |
| sqi_plus_extra_depth12 | 0.843577 | 0.720220 | 0.905769 | 0.843877 | 0.289538 |
| sqi_plus_histgb_badguard | 0.842987 | 0.716855 | 0.947802 | 0.808631 | 0.284672 |
| sqi_plus_histgb_lite_balanced | 0.841925 | 0.718198 | 0.951648 | 0.802982 | 0.289538 |
| sqi_plus_rf_depth10 | 0.840274 | 0.718112 | 0.914835 | 0.830095 | 0.289538 |
| sqi_plus_extra_depth8 | 0.836263 | 0.715063 | 0.884890 | 0.847040 | 0.289538 |
| sqi_plus_mlp_small | 0.771381 | 0.526855 | 0.860714 | 0.769544 | 0.000000 |
| sqi_plus_logreg | 0.766309 | 0.531447 | 0.884066 | 0.739494 | 0.012165 |

## Top Feature Importances

### sqi_plus_extra_depth12
- `sqi_band_15_30`: 0.0476045
- `band_15_30`: 0.0408634
- `diff_zero_crossing_rate`: 0.0392926
- `hf_ratio_mean`: 0.0386839
- `nonqrs_hf_ratio2`: 0.0378358
- `diff_p50`: 0.0370894
- `zero_crossing_rate`: 0.0311624
- `nonqrs_diff_p95_2`: 0.0304714
- `diff_p75`: 0.0280444
- `sqi_band_5_15`: 0.0245981
- `qrs_background_energy_ratio`: 0.0233142
- `flatline_ratio_0.025`: 0.0229173

### sqi_plus_extra_depth8
- `diff_zero_crossing_rate`: 0.0523537
- `band_15_30`: 0.0444624
- `nonqrs_hf_ratio2`: 0.0369525
- `sqi_band_15_30`: 0.0362519
- `diff_p75`: 0.0338953
- `diff_p50`: 0.0333643
- `zero_crossing_rate`: 0.0326483
- `hf_ratio_mean`: 0.0305693
- `nonqrs_diff_p95_2`: 0.0266895
- `flatline_ratio_0.025`: 0.0239579
- `qrs_background_energy_ratio`: 0.0232179
- `sqi_band_5_15`: 0.0226394

### sqi_plus_rf_depth10
- `nonqrs_hf_ratio2`: 0.120689
- `diff_zero_crossing_rate`: 0.05178
- `nonqrs_diff_p95_2`: 0.0420396
- `sqi_band_5_15`: 0.0413854
- `hjorth_complexity_proxy`: 0.0400179
- `sqi_band_1_5`: 0.0323817
- `zero_crossing_rate`: 0.031663
- `diff_p50`: 0.0311772
- `flatline_ratio_0.025`: 0.0303018
- `flatline_ratio_0.015`: 0.0290792
- `hf_ratio_mean`: 0.0286598
- `band_0.5_5`: 0.0281635

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_plus_search_metrics.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_plus_search_summary.json`
- Feature schema: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_plus_feature_schema.json`
- Feature cache: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_sqi_plus_features.npz`
