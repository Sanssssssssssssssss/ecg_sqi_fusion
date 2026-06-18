# BUT Capacity Error Gap: currentbest_init_rows

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 8477 | 0.801817 | 3640 | 0.928297 | 4426 | 0.741527 | 411 | 0.3309 | nan | nan | nan |
| but_test | 8018 | 0.818284 | 3336 | 0.988609 | 4390 | 0.739408 | 292 | 0.0582192 | 111001 | nan | nan |
| but_test | 4656 | 0.708763 | 2191 | 0.985395 | 2173 | 0.517257 | 292 | 0.0582192 | 111001 | outlier_low_confidence | nan |
| but_test | 2817 | 0.964501 | 1128 | 0.994681 | 1689 | 0.944346 | 0 | nan | 111001 | good_medium_overlap | nan |
| but_test | 2191 | 0.985395 | 2191 | 0.985395 | 0 | nan | 0 | nan | 111001 | outlier_low_confidence | good |
| but_test | 2173 | 0.517257 | 0 | nan | 2173 | 0.517257 | 0 | nan | 111001 | outlier_low_confidence | medium |
| but_test | 1689 | 0.944346 | 0 | nan | 1689 | 0.944346 | 0 | nan | 111001 | good_medium_overlap | medium |
| but_test | 1128 | 0.994681 | 1128 | 0.994681 | 0 | nan | 0 | nan | 111001 | good_medium_overlap | good |
| but_test | 540 | 0.998148 | 17 | 1 | 523 | 0.998088 | 0 | nan | 111001 | clean_core | nan |
| but_test | 523 | 0.998088 | 0 | nan | 523 | 0.998088 | 0 | nan | 111001 | clean_core | medium |
| but_test | 292 | 0.0582192 | 0 | nan | 0 | nan | 292 | 0.0582192 | 111001 | outlier_low_confidence | bad |
| but_test | 231 | 1 | 76 | 1 | 36 | 1 | 119 | 1 | 122001 | nan | nan |
| but_test | 228 | 0.0219298 | 228 | 0.0219298 | 0 | nan | 0 | nan | 125001 | nan | nan |
| but_test | 220 | 0 | 220 | 0 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | nan |
| but_test | 220 | 0 | 220 | 0 | 0 | nan | 0 | nan | 125001 | outlier_low_confidence | good |
| but_test | 119 | 1 | 0 | nan | 0 | nan | 119 | 1 | 122001 | near_bad_boundary | nan |
| but_test | 119 | 1 | 0 | nan | 0 | nan | 119 | 1 | 122001 | near_bad_boundary | bad |
| but_test | 77 | 1 | 55 | 1 | 22 | 1 | 0 | nan | 122001 | good_medium_overlap | nan |
| but_test | 55 | 1 | 55 | 1 | 0 | nan | 0 | nan | 122001 | good_medium_overlap | good |
| but_test | 29 | 1 | 21 | 1 | 8 | 1 | 0 | nan | 122001 | clean_core | nan |
| but_test | 22 | 1 | 0 | nan | 22 | 1 | 0 | nan | 122001 | good_medium_overlap | medium |
| but_test | 21 | 1 | 21 | 1 | 0 | nan | 0 | nan | 122001 | clean_core | good |
| but_test | 17 | 1 | 17 | 1 | 0 | nan | 0 | nan | 111001 | clean_core | good |
| but_test | 8 | 0.625 | 8 | 0.625 | 0 | nan | 0 | nan | 125001 | good_medium_overlap | good |
| but_test | 8 | 0.625 | 8 | 0.625 | 0 | nan | 0 | nan | 125001 | good_medium_overlap | nan |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 111001 | outlier_low_confidence | good | good | 2159 |
| but_test | 111001 | good_medium_overlap | medium | medium | 1595 |
| but_test | 111001 | outlier_low_confidence | medium | medium | 1124 |
| but_test | 111001 | good_medium_overlap | good | good | 1122 |
| but_test | 111001 | outlier_low_confidence | medium | good | 1019 |
| but_test | 111001 | clean_core | medium | medium | 522 |
| but_test | 125001 | outlier_low_confidence | good | medium | 220 |
| but_test | 111001 | outlier_low_confidence | bad | good | 176 |
| but_test | 122001 | near_bad_boundary | bad | bad | 119 |
| but_test | 111001 | outlier_low_confidence | bad | medium | 99 |
| but_test | 111001 | good_medium_overlap | medium | good | 94 |
| but_test | 122001 | good_medium_overlap | good | good | 55 |
| but_test | 111001 | outlier_low_confidence | medium | bad | 30 |
| but_test | 111001 | outlier_low_confidence | good | medium | 29 |
| but_test | 122001 | good_medium_overlap | medium | medium | 22 |
| but_test | 122001 | clean_core | good | good | 21 |
| but_test | 111001 | clean_core | good | good | 17 |
| but_test | 111001 | outlier_low_confidence | bad | bad | 17 |
| but_test | 122001 | clean_core | medium | medium | 8 |
| but_test | 111001 | good_medium_overlap | good | medium | 6 |
| but_test | 122001 | outlier_low_confidence | medium | medium | 6 |
| but_test | 111001 | medium_bad_overlap | medium | medium | 5 |
| but_test | 125001 | good_medium_overlap | good | good | 5 |
| but_test | 111001 | outlier_low_confidence | good | bad | 3 |
| but_test | 125001 | good_medium_overlap | good | medium | 3 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | pca_margin | 0.930267 | 5.26452 | -6.29764 | -11.5622 | -3.64565 | 5.78222 | -7.80645 | -4.77612 | 136 | 275 |
| bad | class_margin_percentile | 0.930267 | 0.0786187 | 0.0346263 | -0.0439924 | 0.0650899 | 0.0888363 | 0.00556291 | 0.0572564 | 136 | 275 |
| bad | pca_margin_rank | 0.930267 | 0.0786187 | 0.0346263 | -0.0439924 | 0.0650899 | 0.0888363 | 0.00556291 | 0.0572564 | 136 | 275 |
| bad | boundary_confidence | 0.923155 | 0.389615 | 0.0227531 | -0.366862 | 0.0642337 | 0.394469 | 0.00942668 | 0.0304011 | 136 | 275 |
| bad | region_confidence | 0.923155 | 0.3039 | 0.00568827 | -0.298212 | 0.0160584 | 0.307686 | 0.00235667 | 0.00760028 | 136 | 275 |
| bad | knn_label_purity | 0.912246 | 0.966667 | 0 | -0.966667 | 0.1 | 0.966667 | 0 | 0 | 136 | 275 |
| bad | pc7 | 0.886872 | 3.25088 | -0.449895 | -3.70078 | 2.673 | 3.95576 | -2.34546 | 1.97384 | 136 | 275 |
| bad | flatline_ratio | 0.875 | 0.00880705 | 0.40032 | 0.391513 | 0.00640512 | 0.618094 | 0.229784 | 0.598719 | 136 | 275 |
| bad | local_rms_cv | 0.875 | 0.061246 | 0.536881 | 0.475635 | 0.0460067 | 0.666755 | 0.34389 | 0.849065 | 136 | 275 |
| bad | diff_abs_median | 0.875 | 0.153456 | 0.00384781 | -0.149608 | 0.00153623 | 0.161265 | 0.00185272 | 0.00793781 | 136 | 275 |
| bad | diff_abs_p95 | 0.875 | 0.435459 | 0.0754695 | -0.35999 | 0.0415415 | 0.452124 | 0.0341585 | 0.162721 | 136 | 275 |
| bad | hf_ratio | 0.875 | 0.483629 | 0.0231242 | -0.460505 | 0.0216793 | 0.526319 | 0.0051567 | 0.0903384 | 136 | 275 |
| bad | spectral_entropy | 0.875 | 0.92779 | 0.632529 | -0.295261 | 0.643226 | 0.932446 | 0.4545 | 0.758755 | 136 | 275 |
| bad | qrs_width_median | 0.875 | 0.0115157 | 0.067025 | 0.0555093 | 0.0103745 | 0.0287108 | 0.0310829 | 0.116598 | 136 | 275 |
| bad | qrs_slope_median | 0.875 | 2.77188 | 0.373092 | -2.39879 | 0.195032 | 3.07116 | 0.139874 | 1.04322 | 136 | 275 |
| bad | sqi_basSQI | 0.875 | 0.978864 | 0.625354 | -0.353511 | 0.692923 | 0.988109 | 0.356406 | 0.814066 | 136 | 275 |
| bad | hjorth_mobility | 0.875 | 1.38545 | 0.264449 | -1.121 | 0.278459 | 1.41705 | 0.12779 | 0.503645 | 136 | 275 |
| bad | hjorth_complexity | 0.875 | 1.24926 | 4.23451 | 2.98525 | 1.22549 | 4.12557 | 2.62694 | 9.47322 | 136 | 275 |
| bad | zero_crossing_rate | 0.875 | 0.488391 | 0.0336269 | -0.454764 | 0.031225 | 0.511609 | 0.0171337 | 0.0709367 | 136 | 275 |
| bad | sample_entropy_proxy | 0.875 | 0.890997 | 0.408814 | -0.482183 | 0.405339 | 0.907807 | 0.224639 | 0.55727 | 136 | 275 |
| bad | higuchi_fd_proxy | 0.875 | 1.98086 | 1.32331 | -0.657549 | 1.33243 | 1.9993 | 1.19875 | 1.49522 | 136 | 275 |
| bad | rr_count_detector_c | 0.875 | 27 | 13 | -14 | 10 | 28 | 9.4 | 18 | 136 | 275 |
| bad | non_qrs_diff_p95 | 0.875 | 0.41245 | 0.0407184 | -0.371731 | 0.0201399 | 0.44213 | 0.019263 | 0.113132 | 136 | 275 |
| bad | band_30_45 | 0.875 | 0.269913 | 0.00733637 | -0.262576 | 0.00665396 | 0.321132 | 0.00150689 | 0.036891 | 136 | 275 |
| bad | wavelet_e3 | 0.875 | 0.163817 | 0.00591937 | -0.157898 | 0.00790322 | 0.189443 | 0.00118929 | 0.0236474 | 136 | 275 |
| bad | wavelet_e4 | 0.875 | 0.175906 | 0.00104707 | -0.174859 | 0.001515 | 0.204735 | 0.000165377 | 0.00591886 | 136 | 275 |
| bad | pc1 | 0.875 | 8.93597 | -4.15362 | -13.0896 | -6.10037 | 9.42091 | -6.26802 | -1.22736 | 136 | 275 |
| bad | pc2 | 0.875 | -0.903782 | 11.2267 | 12.1304 | -1.31089 | 14.3024 | 6.81492 | 17.3504 | 136 | 275 |
| bad | pca_own_distance | 0.875 | 5.10092 | 18.9288 | 13.8278 | 4.80426 | 29.7859 | 14.1032 | 25.4827 | 136 | 275 |
| bad | class_centrality_percentile | 0.875 | 0.0786187 | 0.0298959 | -0.0487228 | 0.0051088 | 0.0888363 | 0.00896878 | 0.0527152 | 136 | 275 |
| bad | own_centrality_rank | 0.875 | 0.0786187 | 0.0298959 | -0.0487228 | 0.0051088 | 0.0888363 | 0.00896878 | 0.0527152 | 136 | 275 |
| bad | detail_instability | 0.871364 | 0.285333 | 2.07768 | 1.79235 | 0.271383 | 1.31944 | 0.949155 | 3.67022 | 136 | 275 |
| bad | lf_ratio | 0.871364 | 0.0192587 | 0.386593 | 0.367335 | 0.0104805 | 0.325731 | 0.185538 | 0.722752 | 136 | 275 |
| bad | aggressive_peak_count | 0.871364 | 91 | 44 | -47 | 32.5 | 94 | 30 | 65 | 136 | 275 |
| bad | sqi_pSQI | 0.871364 | 0.323543 | 0.735925 | 0.412382 | 0.287234 | 0.672493 | 0.604751 | 0.838916 | 136 | 275 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\currentbest_init_rows_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\currentbest_init_rows_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\currentbest_init_rows_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\currentbest_init_rows_test_errors.csv`
