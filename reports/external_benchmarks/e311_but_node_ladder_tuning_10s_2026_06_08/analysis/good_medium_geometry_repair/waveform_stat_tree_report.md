# Waveform Stat Tree Search

Thresholds/models are fit on synthetic train and selected by synthetic validation only. Original BUT is report-only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stat_tree_depth3_balanced | synthetic_val | 0.942990 | 0.936411 | 0.953317 | 0.928977 | 0.984375 | 16 | 45 | 4 |
| stat_tree_depth3_balanced | synthetic_test | 0.940543 | 0.943793 | 0.972803 | 0.919643 | 0.983402 | 12 | 97 | 4 |
| stat_tree_depth3_balanced | original_test_all_10s+ | 0.648932 | 0.500474 | 0.650824 | 0.691821 | 0.170316 | 1265 | 873 | 180 |
| stat_tree_depth3_balanced | original_all_10s+ | 0.762198 | 0.786015 | 0.730564 | 0.740215 | 0.908420 | 4582 | 2245 | 277 |
| stat_tree_depth3_balanced | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 |
| stat_tree_depth3_balanced | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 61 |
| stat_tree_depth4_balanced | synthetic_val | 0.963933 | 0.956117 | 0.943489 | 0.966856 | 0.984375 | 20 | 5 | 4 |
| stat_tree_depth4_balanced | synthetic_test | 0.964634 | 0.964344 | 0.953975 | 0.965097 | 0.983402 | 21 | 41 | 4 |
| stat_tree_depth4_balanced | original_test_all_10s+ | 0.656600 | 0.505325 | 0.637912 | 0.717126 | 0.170316 | 1312 | 761 | 181 |
| stat_tree_depth4_balanced | original_all_10s+ | 0.657270 | 0.708062 | 0.475914 | 0.823203 | 0.908420 | 8922 | 1363 | 284 |
| stat_tree_depth4_balanced | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 |
| stat_tree_depth4_balanced | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 62 |
| stat_tree_depth5_balanced | synthetic_val | 0.949389 | 0.939598 | 0.943489 | 0.939394 | 1.000000 | 20 | 12 | 0 |
| stat_tree_depth5_balanced | synthetic_test | 0.955407 | 0.949281 | 0.962343 | 0.945617 | 0.991701 | 17 | 42 | 2 |
| stat_tree_depth5_balanced | original_test_all_10s+ | 0.620856 | 0.522398 | 0.641209 | 0.619069 | 0.459854 | 1278 | 801 | 62 |
| stat_tree_depth5_balanced | original_all_10s+ | 0.656724 | 0.700841 | 0.497272 | 0.767971 | 0.947209 | 8536 | 1450 | 79 |
| stat_tree_depth5_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stat_tree_depth5_balanced | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 62 |
| stat_forest_depth4_balanced | synthetic_val | 0.984293 | 0.983324 | 0.992629 | 0.982955 | 0.976562 | 3 | 18 | 6 |
| stat_forest_depth4_balanced | synthetic_test | 0.949769 | 0.953115 | 0.995816 | 0.926136 | 0.979253 | 2 | 91 | 5 |
| stat_forest_depth4_balanced | original_test_all_10s+ | 0.754040 | 0.658779 | 0.927473 | 0.654541 | 0.289538 | 264 | 1529 | 34 |
| stat_forest_depth4_balanced | original_all_10s+ | 0.806651 | 0.835809 | 0.773983 | 0.796857 | 0.931693 | 3852 | 2159 | 101 |
| stat_forest_depth4_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stat_forest_depth4_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 34 |
| stat_forest_depth6_balanced | synthetic_val | 0.990692 | 0.990227 | 0.992629 | 0.990530 | 0.988281 | 3 | 10 | 3 |
| stat_forest_depth6_balanced | synthetic_test | 0.981035 | 0.980986 | 0.995816 | 0.974838 | 0.983402 | 2 | 31 | 4 |
| stat_forest_depth6_balanced | original_test_all_10s+ | 0.775392 | 0.673744 | 0.918407 | 0.702892 | 0.289538 | 297 | 1315 | 41 |
| stat_forest_depth6_balanced | original_all_10s+ | 0.823704 | 0.850021 | 0.784662 | 0.832424 | 0.932072 | 3670 | 1781 | 106 |
| stat_forest_depth6_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stat_forest_depth6_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 41 |
| stat_extra_depth5_balanced | synthetic_val | 0.983130 | 0.982221 | 0.992629 | 0.981061 | 0.976562 | 3 | 20 | 6 |
| stat_extra_depth5_balanced | synthetic_test | 0.989749 | 0.988615 | 0.993724 | 0.990260 | 0.979253 | 3 | 12 | 5 |
| stat_extra_depth5_balanced | original_test_all_10s+ | 0.764657 | 0.666248 | 0.928297 | 0.674198 | 0.289538 | 261 | 1442 | 31 |
| stat_extra_depth5_balanced | original_all_10s+ | 0.818121 | 0.845488 | 0.779088 | 0.824238 | 0.931693 | 3765 | 1868 | 99 |
| stat_extra_depth5_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stat_extra_depth5_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 31 |
| stat_extra_depth7_balanced | synthetic_val | 0.982548 | 0.981670 | 0.992629 | 0.980114 | 0.976562 | 3 | 21 | 6 |
| stat_extra_depth7_balanced | synthetic_test | 0.985648 | 0.984820 | 0.993724 | 0.983766 | 0.979253 | 3 | 20 | 5 |
| stat_extra_depth7_balanced | original_test_all_10s+ | 0.771971 | 0.671373 | 0.926923 | 0.689336 | 0.289538 | 266 | 1375 | 29 |
| stat_extra_depth7_balanced | original_all_10s+ | 0.822430 | 0.848968 | 0.783606 | 0.830354 | 0.931693 | 3688 | 1803 | 97 |
| stat_extra_depth7_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| stat_extra_depth7_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 29 |
| stat_logreg_balanced | synthetic_val | 0.979639 | 0.974325 | 0.997543 | 0.969697 | 0.992188 | 1 | 7 | 2 |
| stat_logreg_balanced | synthetic_test | 0.992312 | 0.992439 | 0.995816 | 0.990260 | 0.995851 | 2 | 12 | 1 |
| stat_logreg_balanced | original_test_all_10s+ | 0.768550 | 0.656973 | 0.902198 | 0.704700 | 0.272506 | 356 | 1283 | 32 |
| stat_logreg_balanced | original_all_10s+ | 0.812447 | 0.839768 | 0.768057 | 0.824520 | 0.931315 | 3953 | 1803 | 92 |
| stat_logreg_balanced | bad_core_nearboundary | 0.932773 | 0.321739 | 0.000000 | 0.000000 | 0.932773 | 0 | 0 | 8 |
| stat_logreg_balanced | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 24 |

## Top Features

### stat_tree_depth3_balanced

- `band_5_15`: 0.519975
- `diff_zero_crossing_rate`: 0.45359
- `qrs_peak_energy_ratio`: 0.0261224
- `band_30_45`: 0.000313058
- `abs_p90`: 0
- `abs_p50`: 0
- `abs_p75`: 0
- `diff_p50`: 0
- `diff_p95`: 0
- `abs_p99`: 0

### stat_tree_depth4_balanced

- `band_5_15`: 0.507556
- `diff_zero_crossing_rate`: 0.450057
- `qrs_peak_energy_ratio`: 0.0256693
- `peak_count`: 0.0117778
- `flatline_ratio_0.025`: 0.00452097
- `band_30_45`: 0.000418687
- `abs_p75`: 0
- `abs_p50`: 0
- `diff_p95`: 0
- `abs_p99`: 0

### stat_tree_depth5_balanced

- `band_5_15`: 0.499383
- `diff_zero_crossing_rate`: 0.442832
- `qrs_peak_energy_ratio`: 0.025088
- `peak_count`: 0.0115881
- `diff_p50`: 0.00914996
- `detail_ratio_p90`: 0.00890856
- `stress_lowfreq_detail_gap`: 0.00158205
- `qrs_peak_slope_median`: 0.000680491
- `band_15_30`: 0.000564193
- `flatline_ratio_0.01`: 0.000223614

### stat_forest_depth4_balanced

- `diff_p50`: 0.0721152
- `hf_ratio_mean`: 0.0614624
- `diff_p75`: 0.0610493
- `diff_zero_crossing_rate`: 0.0594216
- `band_0.5_5`: 0.0572844
- `flatline_ratio_0.025`: 0.0519037
- `zero_crossing_rate`: 0.0501888
- `band_5_15`: 0.0453004
- `band_15_30`: 0.0443792
- `baseline251_swing`: 0.0428345

### stat_forest_depth6_balanced

- `diff_p50`: 0.0682022
- `diff_zero_crossing_rate`: 0.0672576
- `band_5_15`: 0.0650694
- `flatline_ratio_0.025`: 0.0613807
- `diff_p75`: 0.0551523
- `zero_crossing_rate`: 0.0544349
- `hjorth_complexity_proxy`: 0.0485459
- `flatline_ratio_0.01`: 0.0436402
- `baseline251_swing`: 0.043489
- `hf_ratio_mean`: 0.0431075

### stat_extra_depth5_balanced

- `band_15_30`: 0.0711147
- `diff_p75`: 0.0631128
- `hf_ratio_mean`: 0.0588803
- `zero_crossing_rate`: 0.0546031
- `diff_p50`: 0.0500003
- `qrs_background_energy_ratio`: 0.0469165
- `diff_zero_crossing_rate`: 0.045578
- `band_5_15`: 0.0453827
- `baseline_reversal_rate`: 0.0397223
- `detail_ratio_mean`: 0.0294894

### stat_extra_depth7_balanced

- `band_15_30`: 0.0610664
- `hf_ratio_mean`: 0.0585797
- `diff_p50`: 0.0573998
- `diff_p75`: 0.0532264
- `diff_zero_crossing_rate`: 0.0469944
- `zero_crossing_rate`: 0.0463487
- `qrs_background_energy_ratio`: 0.0382949
- `flatline_ratio_0.025`: 0.034395
- `baseline_reversal_rate`: 0.0342362
- `band_5_15`: 0.0312458

### stat_logreg_balanced

- `band_15_30`: 0.0671342
- `qrs_peak_energy_ratio`: 0.0351961
- `stress_lowfreq_detail_gap`: 0.0158528
- `qrs_background_energy_ratio`: 0.0141
- `detail_ratio_mean`: 0.00950126
- `peak_amp_median`: 0.0078125
- `hf_ratio_mean`: 0.00519298
- `qrs_template_corr_median`: 0.00493789
- `flatline_ratio_0.025`: 0.00379641
- `abs_p75`: 0.00364626

