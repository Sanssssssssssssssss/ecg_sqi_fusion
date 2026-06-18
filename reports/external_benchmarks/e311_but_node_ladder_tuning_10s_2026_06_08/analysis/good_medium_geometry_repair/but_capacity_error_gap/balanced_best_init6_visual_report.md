# BUT Capacity Error Gap: balanced_best_init6_visual

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 9070 | 0.840353 | 6296 | 0.896283 | 2669 | 0.711502 | 105 | 0.761905 | nan | nan | nan |
| but_test | 8364 | 0.830703 | 5786 | 0.889043 | 2556 | 0.70579 | 22 | 0 | 100001 | nan | nan |
| but_test | 5083 | 0.914027 | 3692 | 0.951246 | 1391 | 0.815241 | 0 | nan | 100001 | good_medium_overlap | nan |
| but_test | 3692 | 0.951246 | 3692 | 0.951246 | 0 | nan | 0 | nan | 100001 | good_medium_overlap | good |
| but_test | 1793 | 0.4657 | 1030 | 0.551456 | 741 | 0.360324 | 22 | 0 | 100001 | outlier_low_confidence | nan |
| but_test | 1488 | 0.985887 | 1064 | 1 | 424 | 0.950472 | 0 | nan | 100001 | clean_core | nan |
| but_test | 1391 | 0.815241 | 0 | nan | 1391 | 0.815241 | 0 | nan | 100001 | good_medium_overlap | medium |
| but_test | 1064 | 1 | 1064 | 1 | 0 | nan | 0 | nan | 100001 | clean_core | good |
| but_test | 1030 | 0.551456 | 1030 | 0.551456 | 0 | nan | 0 | nan | 100001 | outlier_low_confidence | good |
| but_test | 741 | 0.360324 | 0 | nan | 741 | 0.360324 | 0 | nan | 100001 | outlier_low_confidence | medium |
| but_test | 424 | 0.950472 | 0 | nan | 424 | 0.950472 | 0 | nan | 100001 | clean_core | medium |
| but_test | 363 | 0.977961 | 229 | 1 | 51 | 0.901961 | 83 | 0.963855 | 114001 | nan | nan |
| but_test | 245 | 0.991837 | 220 | 1 | 25 | 0.92 | 0 | nan | 114001 | good_medium_overlap | nan |
| but_test | 220 | 1 | 220 | 1 | 0 | nan | 0 | nan | 114001 | good_medium_overlap | good |
| but_test | 180 | 0.966667 | 167 | 1 | 13 | 0.538462 | 0 | nan | 121001 | nan | nan |
| but_test | 163 | 0.889571 | 114 | 0.903509 | 49 | 0.857143 | 0 | nan | 100002 | nan | nan |
| but_test | 122 | 0.991803 | 118 | 1 | 4 | 0.75 | 0 | nan | 121001 | good_medium_overlap | nan |
| but_test | 118 | 1 | 118 | 1 | 0 | nan | 0 | nan | 121001 | good_medium_overlap | good |
| but_test | 108 | 0.944444 | 9 | 1 | 17 | 0.823529 | 82 | 0.963415 | 114001 | outlier_low_confidence | nan |
| but_test | 82 | 0.963415 | 0 | nan | 0 | nan | 82 | 0.963415 | 114001 | outlier_low_confidence | bad |
| but_test | 81 | 0.950617 | 58 | 0.982759 | 23 | 0.869565 | 0 | nan | 100002 | good_medium_overlap | nan |
| but_test | 58 | 0.982759 | 58 | 0.982759 | 0 | nan | 0 | nan | 100002 | good_medium_overlap | good |
| but_test | 49 | 0.979592 | 32 | 1 | 17 | 0.941176 | 0 | nan | 100002 | clean_core | nan |
| but_test | 44 | 0.886364 | 36 | 1 | 8 | 0.375 | 0 | nan | 121001 | outlier_low_confidence | nan |
| but_test | 36 | 1 | 36 | 1 | 0 | nan | 0 | nan | 121001 | outlier_low_confidence | good |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 100001 | good_medium_overlap | good | good | 3512 |
| but_test | 100001 | good_medium_overlap | medium | medium | 1134 |
| but_test | 100001 | clean_core | good | good | 1064 |
| but_test | 100001 | outlier_low_confidence | good | good | 568 |
| but_test | 100001 | outlier_low_confidence | medium | good | 474 |
| but_test | 100001 | outlier_low_confidence | good | medium | 462 |
| but_test | 100001 | clean_core | medium | medium | 403 |
| but_test | 100001 | outlier_low_confidence | medium | medium | 267 |
| but_test | 100001 | good_medium_overlap | medium | good | 256 |
| but_test | 114001 | good_medium_overlap | good | good | 220 |
| but_test | 100001 | good_medium_overlap | good | medium | 180 |
| but_test | 121001 | good_medium_overlap | good | good | 118 |
| but_test | 114001 | outlier_low_confidence | bad | bad | 79 |
| but_test | 100002 | good_medium_overlap | good | good | 57 |
| but_test | 121001 | outlier_low_confidence | good | good | 36 |
| but_test | 100002 | clean_core | good | good | 32 |
| but_test | 114001 | good_medium_overlap | medium | medium | 23 |
| but_test | 100001 | outlier_low_confidence | bad | medium | 22 |
| but_test | 100001 | clean_core | medium | good | 21 |
| but_test | 100002 | good_medium_overlap | medium | medium | 20 |
| but_test | 100002 | clean_core | medium | medium | 16 |
| but_test | 100002 | outlier_low_confidence | good | good | 14 |
| but_test | 114001 | outlier_low_confidence | medium | medium | 14 |
| but_test | 121001 | clean_core | good | good | 13 |
| but_test | 100002 | outlier_low_confidence | good | medium | 10 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | hjorth_mobility | 0.9875 | 1.88861 | 0.447535 | -1.44107 | 1.8703 | 1.89123 | 0.408332 | 0.556552 | 80 | 25 |
| bad | hjorth_complexity | 0.9875 | 1.00606 | 2.31634 | 1.31028 | 1.00471 | 1.01513 | 2.05741 | 2.72632 | 80 | 25 |
| bad | wavelet_e4 | 0.9875 | 0.862985 | 0.00312839 | -0.859857 | 0.596347 | 0.912543 | 0.00203008 | 0.00526197 | 80 | 25 |
| bad | zero_crossing_rate | 0.975 | 0.79984 | 0.10008 | -0.69976 | 0.790072 | 0.800641 | 0.082466 | 0.120096 | 80 | 25 |
| bad | pca_margin | 0.975 | 9.56066 | -7.17643 | -16.7371 | 8.52471 | 10.0118 | -7.58984 | -5.91924 | 80 | 25 |
| bad | class_margin_percentile | 0.975 | 0.101798 | 0.0162725 | -0.0855251 | 0.0917502 | 0.130104 | 0.00870388 | 0.0419678 | 80 | 25 |
| bad | pca_margin_rank | 0.975 | 0.101798 | 0.0162725 | -0.0855251 | 0.0917502 | 0.130104 | 0.00870388 | 0.0419678 | 80 | 25 |
| bad | row_pos | 0.9625 | 32437.5 | 32619 | 181.5 | 32310.1 | 32471.1 | 32602.4 | 32675.6 | 80 | 25 |
| bad | rr_count_detector_c | 0.9625 | 29 | 23 | -6 | 27 | 31 | 21.4 | 24 | 80 | 25 |
| bad | pc1 | 0.9625 | 11.2494 | 0.394732 | -10.8546 | 10.0634 | 11.7492 | -0.163994 | 0.912502 | 80 | 25 |
| bad | pca_own_distance | 0.9625 | 2.89482 | 11.9314 | 9.03655 | 2.48802 | 3.42801 | 11.137 | 12.4945 | 80 | 25 |
| bad | knn_label_purity | 0.9625 | 1 | 0.266667 | -0.733333 | 1 | 1 | 0.0333333 | 0.5 | 80 | 25 |
| bad | class_centrality_percentile | 0.9625 | 0.0977294 | 0.0637654 | -0.033964 | 0.0917502 | 0.103709 | 0.0603217 | 0.0659603 | 80 | 25 |
| bad | own_centrality_rank | 0.9625 | 0.0977294 | 0.0637654 | -0.033964 | 0.0917502 | 0.103709 | 0.0603217 | 0.0659603 | 80 | 25 |
| bad | boundary_confidence | 0.9625 | 0.414518 | 0.114459 | -0.300058 | 0.410331 | 0.429516 | 0.0457868 | 0.197893 | 80 | 25 |
| bad | region_confidence | 0.9625 | 0.103702 | 0.0286148 | -0.0750867 | 0.102813 | 0.107789 | 0.0114467 | 0.0494733 | 80 | 25 |
| bad | band_30_45 | 0.96 | 0.148111 | 0.00970913 | -0.138402 | 0.121381 | 0.181771 | 0.0072882 | 0.0162144 | 80 | 25 |
| bad | pca_nearest_other_distance | 0.96 | 12.4818 | 4.80794 | -7.67387 | 11.1737 | 13.0087 | 3.91644 | 6.44601 | 80 | 25 |
| bad | mean_abs | 0.95 | 0.165269 | 0.124055 | -0.0412138 | 0.160453 | 0.167348 | 0.11306 | 0.130669 | 80 | 25 |
| bad | diff_abs_median | 0.95 | 0.343555 | 0.0223202 | -0.321235 | 0.330076 | 0.355766 | 0.0206318 | 0.0274741 | 80 | 25 |
| bad | diff_abs_p95 | 0.9475 | 0.521443 | 0.163188 | -0.358255 | 0.517917 | 0.527773 | 0.146391 | 0.229546 | 80 | 25 |
| bad | higuchi_fd_proxy | 0.9475 | 2.40447 | 1.34506 | -1.05942 | 2.35271 | 2.43425 | 1.3117 | 1.4001 | 80 | 25 |
| bad | rr_count_detector_b | 0.9475 | 29 | 22 | -7 | 27 | 31 | 20 | 23.6 | 80 | 25 |
| bad | non_qrs_diff_p95 | 0.9475 | 0.495154 | 0.120581 | -0.374573 | 0.485865 | 0.506745 | 0.100424 | 0.163445 | 80 | 25 |
| bad | wavelet_e0 | 0.9475 | 0.0732815 | 0.595556 | 0.522274 | 0.0211125 | 0.351246 | 0.461286 | 0.699451 | 80 | 25 |
| bad | wavelet_e1 | 0.9475 | 0.00419414 | 0.25778 | 0.253586 | 0.00266733 | 0.0136253 | 0.180591 | 0.312726 | 80 | 25 |
| bad | qrs_slope_median | 0.9375 | 2.81793 | 1.67137 | -1.14655 | 2.64922 | 2.90717 | 0.943061 | 2.08344 | 80 | 25 |
| bad | sqi_kSQI | 0.9375 | 1.58636 | 4.36138 | 2.77503 | 1.56424 | 1.69512 | 3.41639 | 6.57148 | 80 | 25 |
| bad | band_15_30 | 0.9375 | 0.506906 | 0.0961999 | -0.410706 | 0.446659 | 0.561886 | 0.0680629 | 0.145091 | 80 | 25 |
| bad | hf_ratio | 0.935 | 0.217057 | 0.0506001 | -0.166457 | 0.187174 | 0.253158 | 0.0378323 | 0.0781555 | 80 | 25 |
| bad | qrs_prom_p90 | 0.935 | 1.77351 | 3.61121 | 1.8377 | 1.6816 | 1.82967 | 2.91591 | 3.9185 | 80 | 25 |
| bad | sqi_bSQI | 0.935 | 0 | 0.727273 | 0.727273 | 0 | 0 | 0.456667 | 0.857739 | 80 | 25 |
| bad | diff_zero_crossing_rate | 0.935 | 0.800481 | 0.294071 | -0.50641 | 0.799679 | 0.800481 | 0.273878 | 0.441506 | 80 | 25 |
| bad | band_1_5 | 0.925 | 0.0227139 | 0.507997 | 0.485283 | 0.018213 | 0.0464071 | 0.365848 | 0.558021 | 80 | 25 |
| bad | sqi_pSQI | 0.9225 | 0.332816 | 0.727895 | 0.395078 | 0.153635 | 0.389496 | 0.681881 | 0.788137 | 80 | 25 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\balanced_best_init6_visual_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\balanced_best_init6_visual_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\balanced_best_init6_visual_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\balanced_best_init6_visual_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\balanced_best_init6_visual_test_error_waveform_panels.png`
