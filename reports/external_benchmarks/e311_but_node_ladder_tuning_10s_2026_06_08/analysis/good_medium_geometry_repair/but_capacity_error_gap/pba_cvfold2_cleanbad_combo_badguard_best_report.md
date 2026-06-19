# BUT Capacity Error Gap: pba_cvfold2_cleanbad_combo_badguard_best

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 4316 | 0.910102 | 2244 | 0.954545 | 1255 | 0.772112 | 817 | 1 | nan | nan | nan |
| but_test | 2154 | 0.924791 | 910 | 0.989011 | 451 | 0.662971 | 793 | 1 | 105001 | nan | nan |
| but_test | 1225 | 0.903673 | 891 | 0.923681 | 334 | 0.850299 | 0 | nan | 100001 | nan | nan |
| but_test | 942 | 0.876858 | 683 | 0.900439 | 259 | 0.814672 | 0 | nan | 100001 | good_medium_overlap | nan |
| but_test | 830 | 0.826506 | 528 | 0.981061 | 302 | 0.556291 | 0 | nan | 105001 | good_medium_overlap | nan |
| but_test | 793 | 1 | 0 | nan | 0 | nan | 793 | 1 | 105001 | right_bad_island | nan |
| but_test | 793 | 1 | 0 | nan | 0 | nan | 793 | 1 | 105001 | right_bad_island | bad |
| but_test | 683 | 0.900439 | 683 | 0.900439 | 0 | nan | 0 | nan | 100001 | good_medium_overlap | good |
| but_test | 596 | 0.857383 | 185 | 0.956757 | 411 | 0.812652 | 0 | nan | 111001 | nan | nan |
| but_test | 530 | 0.966038 | 382 | 1 | 148 | 0.878378 | 0 | nan | 105001 | clean_core | nan |
| but_test | 528 | 0.981061 | 528 | 0.981061 | 0 | nan | 0 | nan | 105001 | good_medium_overlap | good |
| but_test | 498 | 0.85743 | 183 | 0.956284 | 315 | 0.8 | 0 | nan | 111001 | good_medium_overlap | nan |
| but_test | 382 | 1 | 382 | 1 | 0 | nan | 0 | nan | 105001 | clean_core | good |
| but_test | 315 | 0.8 | 0 | nan | 315 | 0.8 | 0 | nan | 111001 | good_medium_overlap | medium |
| but_test | 302 | 0.556291 | 0 | nan | 302 | 0.556291 | 0 | nan | 105001 | good_medium_overlap | medium |
| but_test | 283 | 0.992933 | 208 | 1 | 75 | 0.973333 | 0 | nan | 100001 | clean_core | nan |
| but_test | 259 | 0.814672 | 0 | nan | 259 | 0.814672 | 0 | nan | 100001 | good_medium_overlap | medium |
| but_test | 208 | 1 | 208 | 1 | 0 | nan | 0 | nan | 100001 | clean_core | good |
| but_test | 183 | 0.956284 | 183 | 0.956284 | 0 | nan | 0 | nan | 111001 | good_medium_overlap | good |
| but_test | 148 | 0.878378 | 0 | nan | 148 | 0.878378 | 0 | nan | 105001 | clean_core | medium |
| but_test | 97 | 0.85567 | 2 | 1 | 95 | 0.852632 | 0 | nan | 111001 | clean_core | nan |
| but_test | 95 | 0.852632 | 0 | nan | 95 | 0.852632 | 0 | nan | 111001 | clean_core | medium |
| but_test | 75 | 0.973333 | 0 | nan | 75 | 0.973333 | 0 | nan | 100001 | clean_core | medium |
| but_test | 48 | 0.916667 | 42 | 0.928571 | 6 | 0.833333 | 0 | nan | 114001 | nan | nan |
| but_test | 46 | 0.934783 | 42 | 0.928571 | 4 | 1 | 0 | nan | 114001 | good_medium_overlap | nan |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 105001 | right_bad_island | bad | bad | 793 |
| but_test | 100001 | good_medium_overlap | good | good | 615 |
| but_test | 105001 | good_medium_overlap | good | good | 518 |
| but_test | 105001 | clean_core | good | good | 382 |
| but_test | 111001 | good_medium_overlap | medium | medium | 252 |
| but_test | 100001 | good_medium_overlap | medium | medium | 211 |
| but_test | 100001 | clean_core | good | good | 208 |
| but_test | 111001 | good_medium_overlap | good | good | 175 |
| but_test | 105001 | good_medium_overlap | medium | medium | 168 |
| but_test | 105001 | clean_core | medium | medium | 130 |
| but_test | 105001 | good_medium_overlap | medium | good | 126 |
| but_test | 111001 | clean_core | medium | medium | 81 |
| but_test | 100001 | clean_core | medium | medium | 73 |
| but_test | 100001 | good_medium_overlap | good | medium | 68 |
| but_test | 111001 | good_medium_overlap | medium | good | 55 |
| but_test | 100001 | good_medium_overlap | medium | good | 48 |
| but_test | 114001 | good_medium_overlap | good | good | 39 |
| but_test | 122001 | near_bad_boundary | bad | bad | 24 |
| but_test | 118001 | good_medium_overlap | good | good | 23 |
| but_test | 115001 | good_medium_overlap | good | good | 20 |
| but_test | 103002 | clean_core | good | good | 19 |
| but_test | 126001 | good_medium_overlap | good | good | 19 |
| but_test | 105001 | clean_core | medium | good | 18 |
| but_test | 103001 | good_medium_overlap | good | good | 17 |
| but_test | 113001 | good_medium_overlap | good | good | 17 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| good | amplitude_entropy | 0.607376 | 0.610091 | 0.69261 | 0.0825194 | 0.541665 | 0.687473 | 0.639864 | 0.764974 | 2142 | 102 |
| good | low_amp_ratio | 0.584034 | 0.352 | 0.2272 | -0.1248 | 0.2024 | 0.4776 | 0.15056 | 0.30368 | 2142 | 102 |
| good | sqi_sSQI | 0.551354 | 4.18122 | 3.40632 | -0.774906 | 3.47029 | 4.98977 | 1.51903 | 4.09392 | 2142 | 102 |
| good | sqi_kSQI | 0.544818 | 23.4524 | 17.8793 | -5.57306 | 18.1587 | 32.4098 | 10.7271 | 22.8699 | 2142 | 102 |
| good | baseline_step | 0.542484 | 0.26444 | 0.561273 | 0.296832 | 0.133712 | 0.647486 | 0.341852 | 1.01623 | 2142 | 102 |
| good | pca_margin | 0.538749 | 2.48436 | 0.355415 | -2.12895 | 0.168799 | 3.8462 | -0.926428 | 2.18192 | 2142 | 102 |
| good | class_margin_percentile | 0.538749 | 0.652262 | 0.234436 | -0.417826 | 0.203444 | 0.937206 | 0.0829842 | 0.572153 | 2142 | 102 |
| good | pca_margin_rank | 0.538749 | 0.652262 | 0.234436 | -0.417826 | 0.203444 | 0.937206 | 0.0829842 | 0.572153 | 2142 | 102 |
| good | pc1 | 0.498133 | -4.26379 | -2.44838 | 1.81541 | -5.96002 | -2.19974 | -4.30211 | -1.20956 | 2142 | 102 |
| good | boundary_confidence | 0.49253 | 0.748062 | 0.536919 | -0.211144 | 0.512211 | 1.07028 | 0.366267 | 0.718532 | 2142 | 102 |
| good | region_confidence | 0.49253 | 0.658295 | 0.472488 | -0.185806 | 0.450746 | 1.07028 | 0.322315 | 0.632308 | 2142 | 102 |
| good | qrs_prom_p90 | 0.488796 | 6.38608 | 5.92421 | -0.461869 | 5.89841 | 7.56548 | 5.00517 | 6.41741 | 2142 | 102 |
| good | qrs_prom_median | 0.466853 | 5.55246 | 4.97063 | -0.581827 | 1.56518 | 6.5701 | 2.41819 | 5.43565 | 2142 | 102 |
| good | knn_label_purity | 0.434174 | 1 | 0.883333 | -0.116667 | 0.866667 | 1 | 0.636667 | 1 | 2142 | 102 |
| good | band_1_5 | 0.427171 | 0.274727 | 0.206265 | -0.0684621 | 0.190481 | 0.349896 | 0.148075 | 0.304216 | 2142 | 102 |
| good | pca_nearest_other_distance | 0.419701 | 6.80326 | 4.67008 | -2.13319 | 4.31923 | 9.27551 | 3.40351 | 7.64645 | 2142 | 102 |
| good | lf_ratio | 0.383754 | 0.0151667 | 0.0344201 | 0.0192534 | 0.00284163 | 0.172084 | 0.0141488 | 0.133961 | 2142 | 102 |
| good | band_0p3_1 | 0.383754 | 0.0151667 | 0.0344201 | 0.0192534 | 0.00284163 | 0.172084 | 0.0141488 | 0.133961 | 2142 | 102 |
| good | rms | 0.378618 | 0.308712 | 0.253237 | -0.055475 | 0.24023 | 0.323399 | 0.190349 | 0.323399 | 2142 | 102 |
| good | sample_entropy_proxy | 0.37395 | 0.314889 | 0.365366 | 0.0504768 | 0.233763 | 0.374649 | 0.298574 | 0.430455 | 2142 | 102 |
| good | flatline_ratio | 0.366013 | 0.297038 | 0.188151 | -0.108887 | 0.15052 | 0.447558 | 0.116333 | 0.358287 | 2142 | 102 |
| good | pc6 | 0.349673 | -0.0489401 | -0.552696 | -0.503756 | -1.42201 | 1.47986 | -1.87753 | 0.119533 | 2142 | 102 |
| good | pc4 | 0.348739 | 0.159213 | -0.631615 | -0.790828 | -1.12498 | 4.57959 | -2.17022 | 1.20476 | 2142 | 102 |
| good | qrs_band_ratio | 0.34127 | 0.513146 | 0.58046 | 0.0673145 | 0.430781 | 0.607092 | 0.45627 | 0.638 | 2142 | 102 |
| good | std | 0.340803 | 0.296099 | 0.252141 | -0.0439584 | 0.231461 | 0.316135 | 0.190285 | 0.317165 | 2142 | 102 |
| good | hjorth_activity | 0.340803 | 0.0876745 | 0.0635748 | -0.0240997 | 0.0535741 | 0.099941 | 0.0362086 | 0.100594 | 2142 | 102 |
| good | ptp_p99_p01 | 0.323996 | 1.89676 | 1.59933 | -0.297427 | 1.4163 | 2.10137 | 1.14647 | 2.04749 | 2142 | 102 |
| good | band_5_15 | 0.322596 | 0.434245 | 0.482308 | 0.0480628 | 0.362146 | 0.513729 | 0.374001 | 0.532831 | 2142 | 102 |
| good | sqi_bSQI | 0.322129 | 0.9 | 0.916667 | 0.0166667 | 0.818182 | 0.923077 | 0.834615 | 0.944118 | 2142 | 102 |
| good | diff_abs_p95 | 0.314659 | 0.17351 | 0.260764 | 0.0872545 | 0.0806623 | 0.319355 | 0.0933323 | 0.427438 | 2142 | 102 |
| good | source_idx | 0.311391 | 17644 | 6594.5 | -11049.5 | 1341.4 | 29859.7 | 2345.1 | 30688.5 | 2142 | 102 |
| good | non_qrs_rms_ratio | 0.309524 | 0.310107 | 0.362723 | 0.0526161 | 0.12353 | 0.466421 | 0.278696 | 0.570613 | 2142 | 102 |
| good | rr_count_detector_c | 0.295985 | 15 | 19.5 | 4.5 | 11 | 23 | 13 | 25 | 2142 | 102 |
| good | diff_abs_median | 0.293651 | 0.0091201 | 0.0135405 | 0.00442041 | 0.0050061 | 0.0177761 | 0.00528272 | 0.0216548 | 2142 | 102 |
| good | pc7 | 0.291317 | -0.0805467 | 0.536877 | 0.617424 | -1.26052 | 1.04797 | -0.539902 | 1.99551 | 2142 | 102 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleanbad_combo_badguard_best_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleanbad_combo_badguard_best_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleanbad_combo_badguard_best_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleanbad_combo_badguard_best_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleanbad_combo_badguard_best_test_error_waveform_panels.png`
