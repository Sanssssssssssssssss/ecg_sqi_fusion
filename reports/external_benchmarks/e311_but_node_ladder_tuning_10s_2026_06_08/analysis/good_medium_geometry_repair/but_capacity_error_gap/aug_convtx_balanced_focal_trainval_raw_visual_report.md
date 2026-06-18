# BUT Capacity Error Gap: aug_convtx_balanced_focal_trainval_raw_visual

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 8477 | 0.814793 | 3640 | 0.963462 | 4426 | 0.734975 | 411 | 0.357664 | nan | nan | nan |
| but_test | 8018 | 0.81791 | 3336 | 0.993106 | 4390 | 0.732802 | 292 | 0.0958904 | 111001 | nan | nan |
| but_test | 4656 | 0.703823 | 2191 | 0.990872 | 2173 | 0.496088 | 292 | 0.0958904 | 111001 | outlier_low_confidence | nan |
| but_test | 2817 | 0.971601 | 1128 | 0.99734 | 1689 | 0.954411 | 0 | nan | 111001 | good_medium_overlap | nan |
| but_test | 2191 | 0.990872 | 2191 | 0.990872 | 0 | nan | 0 | nan | 111001 | outlier_low_confidence | good |
| but_test | 2173 | 0.496088 | 0 | nan | 2173 | 0.496088 | 0 | nan | 111001 | outlier_low_confidence | medium |
| but_test | 1689 | 0.954411 | 0 | nan | 1689 | 0.954411 | 0 | nan | 111001 | good_medium_overlap | medium |
| but_test | 1128 | 0.99734 | 1128 | 0.99734 | 0 | nan | 0 | nan | 111001 | good_medium_overlap | good |
| but_test | 540 | 0.998148 | 17 | 1 | 523 | 0.998088 | 0 | nan | 111001 | clean_core | nan |
| but_test | 523 | 0.998088 | 0 | nan | 523 | 0.998088 | 0 | nan | 111001 | clean_core | medium |
| but_test | 292 | 0.0958904 | 0 | nan | 0 | nan | 292 | 0.0958904 | 111001 | outlier_low_confidence | bad |
| but_test | 231 | 1 | 76 | 1 | 36 | 1 | 119 | 1 | 122001 | nan | nan |
| but_test | 228 | 0.517544 | 228 | 0.517544 | 0 | nan | 0 | nan | 125001 | nan | nan |
| but_test | 220 | 0.504545 | 220 | 0.504545 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | nan |
| but_test | 220 | 0.504545 | 220 | 0.504545 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | good |
| but_test | 119 | 1 | 0 | nan | 0 | nan | 119 | 1 | 122001 | near_bad_boundary | nan |
| but_test | 119 | 1 | 0 | nan | 0 | nan | 119 | 1 | 122001 | near_bad_boundary | bad |
| but_test | 77 | 1 | 55 | 1 | 22 | 1 | 0 | nan | 122001 | good_medium_overlap | nan |
| but_test | 55 | 1 | 55 | 1 | 0 | nan | 0 | nan | 122001 | good_medium_overlap | good |
| but_test | 29 | 1 | 21 | 1 | 8 | 1 | 0 | nan | 122001 | clean_core | nan |
| but_test | 22 | 1 | 0 | nan | 22 | 1 | 0 | nan | 122001 | good_medium_overlap | medium |
| but_test | 21 | 1 | 21 | 1 | 0 | nan | 0 | nan | 122001 | clean_core | good |
| but_test | 17 | 1 | 17 | 1 | 0 | nan | 0 | nan | 111001 | clean_core | good |
| but_test | 8 | 0.875 | 8 | 0.875 | 0 | nan | 0 | nan | 125001 | good_medium_overlap | good |
| but_test | 8 | 0.875 | 8 | 0.875 | 0 | nan | 0 | nan | 125001 | good_medium_overlap | nan |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 111001 | outlier_low_confidence | good | good | 2171 |
| but_test | 111001 | good_medium_overlap | medium | medium | 1612 |
| but_test | 111001 | good_medium_overlap | good | good | 1125 |
| but_test | 111001 | outlier_low_confidence | medium | good | 1078 |
| but_test | 111001 | outlier_low_confidence | medium | medium | 1078 |
| but_test | 111001 | clean_core | medium | medium | 522 |
| but_test | 111001 | outlier_low_confidence | bad | medium | 224 |
| but_test | 122001 | near_bad_boundary | bad | bad | 119 |
| but_test | 125001 | outlier_low_confidence | good | good | 111 |
| but_test | 125001 | outlier_low_confidence | good | medium | 109 |
| but_test | 111001 | good_medium_overlap | medium | good | 77 |
| but_test | 122001 | good_medium_overlap | good | good | 55 |
| but_test | 111001 | outlier_low_confidence | bad | good | 40 |
| but_test | 111001 | outlier_low_confidence | bad | bad | 28 |
| but_test | 122001 | good_medium_overlap | medium | medium | 22 |
| but_test | 122001 | clean_core | good | good | 21 |
| but_test | 111001 | outlier_low_confidence | good | medium | 20 |
| but_test | 111001 | clean_core | good | good | 17 |
| but_test | 111001 | outlier_low_confidence | medium | bad | 17 |
| but_test | 122001 | clean_core | medium | medium | 8 |
| but_test | 125001 | good_medium_overlap | good | good | 7 |
| but_test | 122001 | outlier_low_confidence | medium | medium | 6 |
| but_test | 111001 | medium_bad_overlap | medium | medium | 5 |
| but_test | 111001 | good_medium_overlap | good | medium | 3 |
| but_test | 111001 | clean_core | medium | bad | 1 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | boundary_confidence | 0.859385 | 0.389242 | 0.0227531 | -0.366488 | 0.0263217 | 0.394343 | 0.00946074 | 0.0311135 | 147 | 264 |
| bad | region_confidence | 0.859385 | 0.303608 | 0.00568827 | -0.29792 | 0.00658042 | 0.307587 | 0.00236518 | 0.00777838 | 147 | 264 |
| bad | knn_label_purity | 0.851809 | 0.966667 | 0 | -0.966667 | 0 | 0.966667 | 0 | 0 | 147 | 264 |
| bad | pca_margin | 0.817022 | 5.2176 | -6.31821 | -11.5358 | -5.32737 | 5.74293 | -7.81136 | -4.66108 | 147 | 264 |
| bad | class_margin_percentile | 0.817022 | 0.0775781 | 0.0344371 | -0.043141 | 0.0517692 | 0.0886282 | 0.00554399 | 0.0589782 | 147 | 264 |
| bad | pca_margin_rank | 0.817022 | 0.0775781 | 0.0344371 | -0.043141 | 0.0517692 | 0.0886282 | 0.00554399 | 0.0589782 | 147 | 264 |
| bad | lf_ratio | 0.816327 | 0.020664 | 0.389692 | 0.369028 | 0.0106256 | 0.373757 | 0.188909 | 0.760099 | 147 | 264 |
| bad | sqi_basSQI | 0.816327 | 0.978262 | 0.62252 | -0.355742 | 0.620942 | 0.98801 | 0.335854 | 0.809609 | 147 | 264 |
| bad | band_0p3_1 | 0.816327 | 0.020664 | 0.389692 | 0.369028 | 0.0106256 | 0.373757 | 0.188909 | 0.760099 | 147 | 264 |
| bad | wavelet_e0 | 0.816327 | 0.249833 | 0.8792 | 0.629366 | 0.196835 | 0.870647 | 0.716986 | 0.980466 | 147 | 264 |
| bad | detail_instability | 0.812539 | 0.286236 | 2.05125 | 1.76501 | 0.272099 | 2.52629 | 0.96195 | 3.67093 | 147 | 264 |
| bad | baseline_step | 0.811766 | 0.266604 | 1.39379 | 1.12719 | 0.14729 | 1.22923 | 0.867055 | 2.12382 | 147 | 264 |
| bad | sample_entropy_proxy | 0.810993 | 0.888902 | 0.405867 | -0.483035 | 0.400553 | 0.907385 | 0.227524 | 0.554494 | 147 | 264 |
| bad | flatline_ratio | 0.809524 | 0.00880705 | 0.397918 | 0.389111 | 0.00640512 | 0.605284 | 0.229784 | 0.602402 | 147 | 264 |
| bad | local_rms_cv | 0.809524 | 0.0636176 | 0.531653 | 0.468036 | 0.0461891 | 0.904313 | 0.343771 | 0.810296 | 147 | 264 |
| bad | diff_abs_median | 0.809524 | 0.152868 | 0.00387882 | -0.148989 | 0.00189018 | 0.161154 | 0.0017592 | 0.00794434 | 147 | 264 |
| bad | diff_abs_p95 | 0.809524 | 0.434436 | 0.0758234 | -0.358612 | 0.0437991 | 0.450912 | 0.0328479 | 0.163242 | 147 | 264 |
| bad | hf_ratio | 0.809524 | 0.482376 | 0.0245922 | -0.457783 | 0.0101121 | 0.523478 | 0.00561634 | 0.0921061 | 147 | 264 |
| bad | spectral_entropy | 0.809524 | 0.927222 | 0.633203 | -0.294018 | 0.607164 | 0.932413 | 0.435572 | 0.759038 | 147 | 264 |
| bad | qrs_width_median | 0.809524 | 0.0116011 | 0.0656686 | 0.0540675 | 0.0104047 | 0.0837689 | 0.0308047 | 0.115068 | 147 | 264 |
| bad | qrs_slope_median | 0.809524 | 2.75972 | 0.372055 | -2.38766 | 0.218332 | 3.05929 | 0.138694 | 1.05519 | 147 | 264 |
| bad | hjorth_mobility | 0.809524 | 1.38193 | 0.273113 | -1.10882 | 0.222817 | 1.41575 | 0.121694 | 0.507822 | 147 | 264 |
| bad | hjorth_complexity | 0.809524 | 1.25327 | 4.23053 | 2.97726 | 1.22602 | 4.7108 | 2.62332 | 9.777 | 147 | 264 |
| bad | zero_crossing_rate | 0.809524 | 0.485989 | 0.0336269 | -0.452362 | 0.0248199 | 0.511129 | 0.016253 | 0.071257 | 147 | 264 |
| bad | higuchi_fd_proxy | 0.809524 | 1.97921 | 1.33585 | -0.643352 | 1.21948 | 1.99898 | 1.2016 | 1.49685 | 147 | 264 |
| bad | rr_count_detector_c | 0.809524 | 26 | 13 | -13 | 10.6 | 28 | 9 | 18 | 147 | 264 |
| bad | non_qrs_diff_p95 | 0.809524 | 0.409694 | 0.0406081 | -0.369086 | 0.028519 | 0.44033 | 0.0186926 | 0.114079 | 147 | 264 |
| bad | band_30_45 | 0.809524 | 0.26541 | 0.0075363 | -0.257873 | 0.00333541 | 0.320139 | 0.00184719 | 0.0370921 | 147 | 264 |
| bad | wavelet_e3 | 0.809524 | 0.160864 | 0.00659226 | -0.154272 | 0.00300798 | 0.186931 | 0.00103586 | 0.0240113 | 147 | 264 |
| bad | wavelet_e4 | 0.809524 | 0.173829 | 0.00108669 | -0.172743 | 0.000474707 | 0.20371 | 0.000159785 | 0.00605301 | 147 | 264 |
| bad | pc1 | 0.809524 | 8.88322 | -4.04781 | -12.931 | -6.19024 | 9.41447 | -6.2145 | -1.14036 | 147 | 264 |
| bad | pc2 | 0.809524 | -0.900682 | 11.2171 | 12.1178 | -1.29959 | 13.6035 | 6.80991 | 17.7189 | 147 | 264 |
| bad | pca_own_distance | 0.809524 | 5.11959 | 18.8823 | 13.7627 | 4.81626 | 24.224 | 14.0733 | 25.9292 | 147 | 264 |
| bad | class_centrality_percentile | 0.809524 | 0.0775781 | 0.0301798 | -0.0473983 | 0.0107474 | 0.0886282 | 0.00824976 | 0.0529234 | 147 | 264 |
| bad | own_centrality_rank | 0.809524 | 0.0775781 | 0.0301798 | -0.0473983 | 0.0107474 | 0.0886282 | 0.00824976 | 0.0529234 | 147 | 264 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_raw_visual_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_raw_visual_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_raw_visual_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_raw_visual_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\aug_convtx_balanced_focal_trainval_raw_visual_test_error_waveform_panels.png`
