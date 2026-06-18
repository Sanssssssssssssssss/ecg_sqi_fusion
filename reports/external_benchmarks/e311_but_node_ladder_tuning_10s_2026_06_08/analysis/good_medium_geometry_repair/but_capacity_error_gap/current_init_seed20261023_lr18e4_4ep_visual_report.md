# BUT Capacity Error Gap: current_init_seed20261023_lr18e4_4ep_visual

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 8477 | 0.803232 | 3640 | 0.931319 | 4426 | 0.742431 | 411 | 0.323601 | nan | nan | nan |
| but_test | 8018 | 0.819032 | 3336 | 0.989808 | 4390 | 0.740319 | 292 | 0.0513699 | 111001 | nan | nan |
| but_test | 4656 | 0.711555 | 2191 | 0.985851 | 2173 | 0.5237 | 292 | 0.0513699 | 111001 | outlier_low_confidence | nan |
| but_test | 2817 | 0.961661 | 1128 | 0.99734 | 1689 | 0.937833 | 0 | nan | 111001 | good_medium_overlap | nan |
| but_test | 2191 | 0.985851 | 2191 | 0.985851 | 0 | nan | 0 | nan | 111001 | outlier_low_confidence | good |
| but_test | 2173 | 0.5237 | 0 | nan | 2173 | 0.5237 | 0 | nan | 111001 | outlier_low_confidence | medium |
| but_test | 1689 | 0.937833 | 0 | nan | 1689 | 0.937833 | 0 | nan | 111001 | good_medium_overlap | medium |
| but_test | 1128 | 0.99734 | 1128 | 0.99734 | 0 | nan | 0 | nan | 111001 | good_medium_overlap | good |
| but_test | 540 | 1 | 17 | 1 | 523 | 1 | 0 | nan | 111001 | clean_core | nan |
| but_test | 523 | 1 | 0 | nan | 523 | 1 | 0 | nan | 111001 | clean_core | medium |
| but_test | 292 | 0.0513699 | 0 | nan | 0 | nan | 292 | 0.0513699 | 111001 | outlier_low_confidence | bad |
| but_test | 231 | 0.995671 | 76 | 1 | 36 | 1 | 119 | 0.991597 | 122001 | nan | nan |
| but_test | 228 | 0.0526316 | 228 | 0.0526316 | 0 | nan | 0 | nan | 125001 | nan | nan |
| but_test | 220 | 0.0227273 | 220 | 0.0227273 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | nan |
| but_test | 220 | 0.0227273 | 220 | 0.0227273 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | good |
| but_test | 119 | 0.991597 | 0 | nan | 0 | nan | 119 | 0.991597 | 122001 | near_bad_boundary | nan |
| but_test | 119 | 0.991597 | 0 | nan | 0 | nan | 119 | 0.991597 | 122001 | near_bad_boundary | bad |
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
| but_test | 111001 | outlier_low_confidence | good | good | 2160 |
| but_test | 111001 | good_medium_overlap | medium | medium | 1584 |
| but_test | 111001 | outlier_low_confidence | medium | medium | 1138 |
| but_test | 111001 | good_medium_overlap | good | good | 1125 |
| but_test | 111001 | outlier_low_confidence | medium | good | 1008 |
| but_test | 111001 | clean_core | medium | medium | 523 |
| but_test | 125001 | outlier_low_confidence | good | medium | 215 |
| but_test | 111001 | outlier_low_confidence | bad | good | 180 |
| but_test | 122001 | near_bad_boundary | bad | bad | 118 |
| but_test | 111001 | good_medium_overlap | medium | good | 105 |
| but_test | 111001 | outlier_low_confidence | bad | medium | 97 |
| but_test | 122001 | good_medium_overlap | good | good | 55 |
| but_test | 111001 | outlier_low_confidence | good | medium | 29 |
| but_test | 111001 | outlier_low_confidence | medium | bad | 27 |
| but_test | 122001 | good_medium_overlap | medium | medium | 22 |
| but_test | 122001 | clean_core | good | good | 21 |
| but_test | 111001 | clean_core | good | good | 17 |
| but_test | 111001 | outlier_low_confidence | bad | bad | 15 |
| but_test | 122001 | clean_core | medium | medium | 8 |
| but_test | 125001 | good_medium_overlap | good | good | 7 |
| but_test | 122001 | outlier_low_confidence | medium | medium | 6 |
| but_test | 111001 | medium_bad_overlap | medium | medium | 5 |
| but_test | 125001 | outlier_low_confidence | good | good | 5 |
| but_test | 111001 | good_medium_overlap | good | medium | 3 |
| but_test | 111001 | outlier_low_confidence | good | bad | 2 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | pca_margin | 0.929383 | 5.27078 | -6.27673 | -11.5475 | -3.35363 | 5.78982 | -7.79172 | -4.74708 | 133 | 278 |
| bad | class_margin_percentile | 0.929383 | 0.0789026 | 0.0349101 | -0.0439924 | 0.0656954 | 0.0888931 | 0.00561968 | 0.0577673 | 133 | 278 |
| bad | pca_margin_rank | 0.929383 | 0.0789026 | 0.0349101 | -0.0439924 | 0.0656954 | 0.0888931 | 0.00561968 | 0.0577673 | 133 | 278 |
| bad | row_pos | 0.922838 | 32541 | 32816.5 | 275.5 | 32488.2 | 32649.8 | 32697.7 | 32927.3 | 133 | 278 |
| bad | boundary_confidence | 0.922838 | 0.389658 | 0.0228382 | -0.36682 | 0.064368 | 0.394534 | 0.00942952 | 0.0308515 | 133 | 278 |
| bad | region_confidence | 0.922838 | 0.303933 | 0.00570956 | -0.298224 | 0.016092 | 0.307736 | 0.00235738 | 0.00771287 | 133 | 278 |
| bad | knn_label_purity | 0.912046 | 0.966667 | 0 | -0.966667 | 0.1 | 0.966667 | 0 | 0 | 133 | 278 |
| bad | pc7 | 0.887191 | 3.25041 | -0.442154 | -3.69257 | 2.67491 | 3.90693 | -2.35927 | 2.0438 | 133 | 278 |
| bad | flatline_ratio | 0.883621 | 0.00880705 | 0.400721 | 0.391914 | 0.00640512 | 0.605765 | 0.228102 | 0.601761 | 133 | 278 |
| bad | local_rms_cv | 0.883621 | 0.0605833 | 0.536779 | 0.476196 | 0.0459193 | 0.615847 | 0.340772 | 0.849198 | 133 | 278 |
| bad | diff_abs_median | 0.883621 | 0.153506 | 0.00381301 | -0.149693 | 0.00158688 | 0.161309 | 0.00184714 | 0.00800694 | 133 | 278 |
| bad | diff_abs_p95 | 0.883621 | 0.435496 | 0.0754106 | -0.360086 | 0.0472138 | 0.452586 | 0.0346419 | 0.163084 | 133 | 278 |
| bad | hf_ratio | 0.883621 | 0.48374 | 0.0231576 | -0.460583 | 0.0225279 | 0.527005 | 0.00496557 | 0.0919721 | 133 | 278 |
| bad | spectral_entropy | 0.883621 | 0.927814 | 0.632924 | -0.294889 | 0.654829 | 0.932465 | 0.454076 | 0.760329 | 133 | 278 |
| bad | qrs_width_median | 0.883621 | 0.0114128 | 0.0667933 | 0.0553805 | 0.0103645 | 0.0269283 | 0.0308664 | 0.116268 | 133 | 278 |
| bad | qrs_slope_median | 0.883621 | 2.77205 | 0.373083 | -2.39897 | 0.246243 | 3.06496 | 0.138778 | 1.04998 | 133 | 278 |
| bad | sqi_basSQI | 0.883621 | 0.979089 | 0.626216 | -0.352873 | 0.770211 | 0.988121 | 0.355837 | 0.816055 | 133 | 278 |
| bad | hjorth_mobility | 0.883621 | 1.38556 | 0.265599 | -1.11996 | 0.287828 | 1.41709 | 0.126612 | 0.50577 | 133 | 278 |
| bad | hjorth_complexity | 0.883621 | 1.24903 | 4.23053 | 2.98151 | 1.22522 | 3.88869 | 2.60415 | 9.49913 | 133 | 278 |
| bad | zero_crossing_rate | 0.883621 | 0.488391 | 0.0332266 | -0.455164 | 0.0345877 | 0.511609 | 0.0173739 | 0.071257 | 133 | 278 |
| bad | sample_entropy_proxy | 0.883621 | 0.89143 | 0.409599 | -0.481832 | 0.443772 | 0.907815 | 0.223995 | 0.558176 | 133 | 278 |
| bad | higuchi_fd_proxy | 0.883621 | 1.98157 | 1.32212 | -0.659447 | 1.3659 | 1.99943 | 1.19925 | 1.49642 | 133 | 278 |
| bad | rr_count_detector_c | 0.883621 | 27 | 13 | -14 | 10.8 | 28 | 9 | 18 | 133 | 278 |
| bad | non_qrs_diff_p95 | 0.883621 | 0.412534 | 0.0406081 | -0.371926 | 0.0221642 | 0.443098 | 0.0190436 | 0.114042 | 133 | 278 |
| bad | band_30_45 | 0.883621 | 0.270143 | 0.00730754 | -0.262835 | 0.00768666 | 0.321561 | 0.00148141 | 0.0369855 | 133 | 278 |
| bad | wavelet_e3 | 0.883621 | 0.164882 | 0.00599473 | -0.158888 | 0.00963326 | 0.190308 | 0.00122679 | 0.0238157 | 133 | 278 |
| bad | wavelet_e4 | 0.883621 | 0.176381 | 0.00104728 | -0.175334 | 0.00173056 | 0.205163 | 0.000166152 | 0.00600161 | 133 | 278 |
| bad | pc1 | 0.883621 | 8.94414 | -4.16209 | -13.1062 | -5.99344 | 9.42315 | -6.30012 | -1.18762 | 133 | 278 |
| bad | pc2 | 0.883621 | -0.904496 | 11.2328 | 12.1373 | -1.31247 | 12.9592 | 6.79478 | 17.3824 | 133 | 278 |
| bad | pca_own_distance | 0.883621 | 5.09428 | 18.9325 | 13.8382 | 4.79875 | 27.887 | 14.081 | 25.5896 | 133 | 278 |
| bad | class_centrality_percentile | 0.883621 | 0.0789026 | 0.0298013 | -0.0491012 | 0.00613056 | 0.0888931 | 0.00877956 | 0.0528477 | 133 | 278 |
| bad | own_centrality_rank | 0.883621 | 0.0789026 | 0.0298013 | -0.0491012 | 0.00613056 | 0.0888931 | 0.00877956 | 0.0528477 | 133 | 278 |
| bad | detail_instability | 0.880024 | 0.28524 | 2.0655 | 1.78026 | 0.271097 | 1.17569 | 0.942228 | 3.67656 | 133 | 278 |
| bad | lf_ratio | 0.880024 | 0.0189602 | 0.387473 | 0.368513 | 0.0104676 | 0.280302 | 0.183933 | 0.724203 | 133 | 278 |
| bad | aggressive_peak_count | 0.880024 | 91 | 44 | -47 | 35.2 | 94 | 30 | 65 | 133 | 278 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\current_init_seed20261023_lr18e4_4ep_visual_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\current_init_seed20261023_lr18e4_4ep_visual_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\current_init_seed20261023_lr18e4_4ep_visual_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\current_init_seed20261023_lr18e4_4ep_visual_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\current_init_seed20261023_lr18e4_4ep_visual_test_error_waveform_panels.png`
