# Original Gap In Waveform SQI-Plus Space

Report-only analysis of why the strict waveform-derived model stalls. Original rows are not used for training or model selection.

## Error Counts

| group | n | acc | good_n | medium_n | bad_n | pred_good | pred_medium | pred_bad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_test_all_10s+ | 8477 | 0.858794 | 3640 | 4426 | 411 | 3698 | 4655 | 124 |
| bad_core_nearboundary | 119 | 0.983193 | 0 | 0 | 119 | 0 | 2 | 117 |
| bad_outlier_stress | 292 | 0.00342466 | 0 | 0 | 292 | 218 | 73 | 1 |
| good_to_medium_errors | 529 | 0 | 529 | 0 | 0 | 0 | 529 | 0 |
| medium_to_good_errors | 369 | 0 | 0 | 369 | 0 | 369 | 0 | 0 |
| bad_to_medium_errors | 75 | 0 | 0 | 0 | 75 | 0 | 75 | 0 |
| good_correct | 3111 | 1 | 3111 | 0 | 0 | 3111 | 0 | 0 |
| medium_correct | 4051 | 1 | 0 | 4051 | 0 | 0 | 4051 | 0 |
| bad_correct | 118 | 1 | 0 | 0 | 118 | 0 | 0 | 118 |

## Nearest-Neighbor Coverage

| group | class | n | dist_p50 | dist_p90 | dist_p95 | train_self_p95 | outside_train_self_p95_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_test | good | 478 | 7.78877 | 11.8463 | 15.7542 | 10.3048 | 0.242678 |
| synthetic_test | medium | 1232 | 5.02416 | 9.21062 | 11.9125 | 6.20817 | 0.30763 |
| synthetic_test | bad | 241 | 4.89552 | 6.60315 | 7.50411 | 7.02632 | 0.0705394 |
| original_test_all_10s+ | good | 3640 | 12.5991 | 17.9888 | 21.1167 | 10.3048 | 0.741758 |
| original_test_all_10s+ | medium | 4426 | 8.10599 | 32.3582 | 46.1243 | 6.20817 | 0.684139 |
| original_test_all_10s+ | bad | 411 | 49.193 | 84.1139 | 88.2292 | 7.02632 | 1 |
| bad_core_nearboundary | bad | 119 | 79.7334 | 88.1222 | 89.8629 | 7.02632 | 1 |
| bad_outlier_stress | bad | 292 | 36.3166 | 71.6281 | 81.4749 | 7.02632 | 1 |
| good_to_medium_errors | good | 529 | 10.8184 | 18.0815 | 23.2437 | 10.3048 | 0.551985 |
| medium_to_good_errors | medium | 369 | 23.0354 | 69.9558 | 94.1637 | 6.20817 | 0.9729 |
| bad_to_medium_errors | bad | 75 | 26.441 | 56.6145 | 68.5016 | 7.02632 | 1 |
| good_correct | good | 3111 | 12.827 | 17.9797 | 20.8898 | 10.3048 | 0.774028 |
| medium_correct | medium | 4051 | 7.6507 | 27.6235 | 39.689 | 6.20817 | 0.657369 |
| bad_correct | bad | 118 | 79.9257 | 88.1319 | 89.8728 | 7.02632 | 1 |

## Top Feature Gaps By Group

### bad_core_nearboundary
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| bad | diff_p50 | 1 | 2.03182 | 0.955827 | 0.649947 | 119 |
| bad | diff_p75 | 1 | 2.08382 | 1.61215 | 1.1387 | 119 |
| bad | zero_crossing_rate | 1 | 1.86921 | 0.491593 | 0.297038 | 119 |
| bad | diff_zero_crossing_rate | 1 | 5.84364 | 0.675481 | 0.375801 | 119 |
| bad | band_30_45 | 1 | 2.39615 | 0.826371 | 0.0957749 | 119 |
| bad | hjorth_mobility_proxy | 1 | 2.12538 | 1.38832 | 0.965343 | 119 |
| bad | baseline_reversal_rate | 1 | 1.26388 | 0.510417 | 0.364583 | 119 |
| bad | sqi_band_30_45 | 1 | 1.63002 | 0.199533 | 0.0320152 | 119 |
| bad | sqi_band_45_55 | 1 | 2075.2 | 0.275968 | 8.36698e-05 | 119 |
| bad | diff_p90 | 0.993177 | 1.9021 | 2.28262 | 1.56015 | 119 |
| bad | nonqrs_diff_p95_2 | 0.991597 | 1.9134 | 2.67288 | 1.73224 | 119 |
| bad | band_5_15 | 0.986353 | 18.9135 | 1.17593 | 0.0630015 | 119 |

### bad_correct
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| bad | diff_p50 | 0.991525 | 2.02391 | 0.954638 | 0.649947 | 118 |
| bad | diff_p75 | 0.991525 | 2.0688 | 1.60874 | 1.1387 | 118 |
| bad | zero_crossing_rate | 0.991525 | 1.86152 | 0.490793 | 0.297038 | 118 |
| bad | diff_zero_crossing_rate | 0.991525 | 5.83582 | 0.67508 | 0.375801 | 118 |
| bad | band_30_45 | 0.991525 | 2.38888 | 0.824155 | 0.0957749 | 118 |
| bad | hjorth_mobility_proxy | 0.991525 | 2.12302 | 1.38785 | 0.965343 | 118 |
| bad | baseline_reversal_rate | 0.991525 | 1.26388 | 0.510417 | 0.364583 | 118 |
| bad | sqi_band_30_45 | 0.991525 | 1.6261 | 0.19913 | 0.0320152 | 118 |
| bad | sqi_band_45_55 | 0.991525 | 2075.46 | 0.276002 | 8.36698e-05 | 118 |
| bad | band_5_15 | 0.98787 | 18.9251 | 1.17661 | 0.0630015 | 118 |
| bad | sqi_band_5_15 | 0.987111 | 13.2261 | 0.188968 | 0.0141449 | 118 |
| bad | diff_p90 | 0.984702 | 1.89741 | 2.28084 | 1.56015 | 118 |

### bad_outlier_stress
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| bad | diff_p50 | 1 | -4.11235 | 0.030854 | 0.649947 | 292 |
| bad | flatline_ratio_0.005 | 1 | 33.9894 | 0.11289 | 0.0040032 | 292 |
| bad | flatline_ratio_0.01 | 1 | 65.3546 | 0.217374 | 0.00800641 | 292 |
| bad | flatline_ratio_0.015 | 1 | 61.6538 | 0.308247 | 0.0120096 | 292 |
| bad | flatline_ratio_0.025 | 1 | 75.2723 | 0.441954 | 0.020016 | 292 |
| bad | stress_diff_silence_ratio | 1 | 33.9894 | 0.11289 | 0.0040032 | 292 |
| bad | diff_p75 | 0.995451 | -4.62095 | 0.0887954 | 1.1387 | 292 |
| bad | nonqrs_hf_ratio2 | 0.993935 | -0.488846 | 0.0550992 | 0.88871 | 292 |
| bad | zero_crossing_rate | 0.992026 | -2.54613 | 0.0320256 | 0.297038 | 292 |
| bad | flatline_longest_0p02 | 0.99166 | 31.996 | 0.264 | 0.008 | 292 |
| bad | detail_hf_to_qrs2 | 0.989386 | -0.262715 | 0.36002 | 1.07215 | 292 |
| bad | hf_ratio_mean | 0.981046 | -25.0332 | 0.208262 | 0.912473 | 292 |

### bad_to_medium_errors
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| bad | sqi_band_15_30 | 0.974223 | -9.23109 | 0.0243498 | 0.920475 | 75 |
| bad | diff_p50 | 0.973333 | -4.00805 | 0.046556 | 0.649947 | 75 |
| bad | flatline_ratio_0.005 | 0.973333 | 19.7438 | 0.0672538 | 0.0040032 | 75 |
| bad | flatline_ratio_0.01 | 0.973333 | 39.2378 | 0.133707 | 0.00800641 | 75 |
| bad | flatline_ratio_0.015 | 0.973333 | 38.1587 | 0.195356 | 0.0120096 | 75 |
| bad | flatline_ratio_0.025 | 0.973333 | 53.5619 | 0.320256 | 0.020016 | 75 |
| bad | stress_diff_silence_ratio | 0.973333 | 19.7438 | 0.0672538 | 0.0040032 | 75 |
| bad | band_15_30 | 0.971948 | -9.08916 | 0.100395 | 2.73348 | 75 |
| bad | band_0.5_5 | 0.97119 | 118.204 | 8.16975 | 0.0898039 | 75 |
| bad | hjorth_complexity_proxy | 0.97119 | 87.3996 | 5.50735 | 1.09299 | 75 |
| bad | sqi_band_1_5 | 0.97119 | 42.3763 | 0.294685 | 0.00846868 | 75 |
| bad | diff_p75 | 0.968784 | -4.52796 | 0.109923 | 1.1387 | 75 |

### good_correct
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| good | diff_p95 | 0.869107 | -0.86643 | 0.536916 | 2.60712 | 3111 |
| good | det_diff_count | 0.861949 | -1.5 | 6 | 12 | 3111 |
| good | hf_ratio_p90 | 0.811651 | -3.35963 | 0.530969 | 1.5571 | 3111 |
| good | sqi_band_0.3_1 | 0.797411 | 4.6752 | 0.233161 | 0.0167818 | 3111 |
| good | detail_ratio_mean | 0.793095 | -1.77611 | 0.383561 | 0.583466 | 3111 |
| good | det_qrs_diff_agree | 0.759009 | -15.5384 | 0.75 | 1 | 3111 |
| good | diff_zero_crossing_rate | 0.757383 | 1.92451 | 0.403045 | 0.239583 | 3111 |
| good | hf_ratio_mean | 0.701314 | -1.36908 | 0.240573 | 0.370293 | 3111 |
| good | qrs_visibility_prom_p25 | 0.685778 | -0.880493 | 2.90485 | 16.3403 | 3111 |
| good | qrs_rr_cv2 | 0.679494 | 12.1765 | 0.537097 | 0.0245307 | 3111 |
| good | baseline_reversal_rate | 0.674739 | -1.10253 | 0.0320513 | 0.0665064 | 3111 |
| good | band_5_15 | 0.672206 | -1.11638 | 1.62311 | 2.07094 | 3111 |

### good_to_medium_errors
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| good | nonqrs_hf_ratio2 | 0.781838 | 3.21609 | 0.0269432 | 0.00974477 | 529 |
| good | diff_p95 | 0.764717 | -0.874104 | 0.518581 | 2.60712 | 529 |
| good | baseline_jump_p95 | 0.725919 | -0.875299 | 0.0124647 | 0.0363094 | 529 |
| good | ptp_z | 0.664188 | -1.07549 | 8.36414 | 26.1909 | 529 |
| good | high101_abs_mean | 0.658846 | -1.04382 | 0.541388 | 1.41808 | 529 |
| good | std_z | 0.653264 | -1.09657 | 1.18732 | 3.14318 | 529 |
| good | qrs_visibility_prom_p25 | 0.652968 | -0.898473 | 2.6305 | 16.3403 | 529 |
| good | mean_abs | 0.652165 | -1.10239 | 0.873414 | 1.3924 | 529 |
| good | rms_z | 0.651883 | -1.07917 | 1.19951 | 3.19457 | 529 |
| good | qrs_peak_slope_median | 0.647512 | -0.986463 | 2.54988 | 10.1307 | 529 |
| good | abs_p99 | 0.642832 | -1.04222 | 4.00398 | 17.1121 | 529 |
| good | baseline63_std_ratio | 0.634767 | 2.54569 | 0.655145 | 0.237451 | 529 |

### medium_correct
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| medium | sqi_band_0.3_1 | 0.514513 | 3.57473 | 0.24164 | 0.0310278 | 4051 |
| medium | baseline_low_swing2 | 0.486005 | 1.11222 | 2.6427 | 1.88853 | 4051 |
| medium | detail_ratio_mean | 0.480053 | -1.69584 | 0.528054 | 0.700319 | 4051 |
| medium | baseline_mid_ratio2 | 0.444734 | 1.99445 | 1.23311 | 0.5054 | 4051 |
| medium | hf_ratio_mean | 0.441559 | -1.15806 | 0.384204 | 0.514732 | 4051 |
| medium | sqi_band_1_5 | 0.431976 | -0.629596 | 0.201561 | 0.287009 | 4051 |
| medium | detector_union_rate | 0.427251 | -0.999999 | 2.9 | 3.7 | 4051 |
| medium | det_raw_count | 0.416214 | -1.2 | 16 | 22 | 4051 |
| medium | high101_abs_mean | 0.412563 | -1.13907 | 0.61857 | 0.970493 | 4051 |
| medium | det_qrs_diff_agree | 0.410996 | -0.618745 | 0.9 | 0.96875 | 4051 |
| medium | flatline_ratio_0.025 | 0.410116 | 1.619 | 0.120897 | 0.0664532 | 4051 |
| medium | diff_p50 | 0.407524 | -0.928968 | 0.127278 | 0.24995 | 4051 |

### medium_to_good_errors
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| medium | flatline_ratio_0.01 | 0.902286 | 12.9465 | 0.22498 | 0.0280224 | 369 |
| medium | diff_p50 | 0.900758 | -1.69612 | 0.0259737 | 0.24995 | 369 |
| medium | flatline_ratio_0.015 | 0.898787 | 13.1846 | 0.325861 | 0.0408327 | 369 |
| medium | flatline_ratio_0.025 | 0.898778 | 12.5234 | 0.48759 | 0.0664532 | 369 |
| medium | flatline_ratio_0.005 | 0.898129 | 12.6984 | 0.116093 | 0.0144115 | 369 |
| medium | stress_diff_silence_ratio | 0.898129 | 12.6984 | 0.116093 | 0.0144115 | 369 |
| medium | baseline_reversal_rate | 0.897835 | -1.59998 | 0.0232372 | 0.141827 | 369 |
| medium | diff_p75 | 0.897318 | -1.54941 | 0.0573804 | 0.508862 | 369 |
| medium | hf_ratio_mean | 0.895906 | -3.00612 | 0.175904 | 0.514732 | 369 |
| medium | detail_ratio_mean | 0.891975 | -3.98779 | 0.295236 | 0.700319 | 369 |
| medium | zero_crossing_rate | 0.882893 | -1.63216 | 0.0216173 | 0.135308 | 369 |
| medium | dropout_lowamp_longest_0p08 | 0.876611 | 10.4993 | 0.2 | 0.032 | 369 |

### original_test_all_10s+
| true_class | feature | ks | median_shift_iqr | group_median | train_class_median | n_group |
| --- | --- | --- | --- | --- | --- | --- |
| bad | sqi_band_15_30 | 0.971948 | -8.88653 | 0.0577991 | 0.920475 | 411 |
| bad | band_0.5_5 | 0.97119 | 98.6856 | 6.83554 | 0.0898039 | 411 |
| bad | band_15_30 | 0.97119 | -8.74088 | 0.201287 | 2.73348 | 411 |
| bad | hjorth_complexity_proxy | 0.97119 | 45.7302 | 3.40272 | 1.09299 | 411 |
| bad | sqi_band_1_5 | 0.97119 | 40.4462 | 0.281648 | 0.00846868 | 411 |
| bad | sqi_band_0.3_1 | 0.937479 | 274.924 | 0.298766 | 0.000667921 | 411 |
| bad | baseline63_std_ratio | 0.92854 | 22.5842 | 0.608061 | 0.0503688 | 411 |
| bad | band_5_15 | 0.915264 | 15.1993 | 0.957372 | 0.0630015 | 411 |
| bad | baseline_low_swing2 | 0.910468 | 27.5905 | 2.76814 | 0.142529 | 411 |
| bad | sqi_band_5_15 | 0.854066 | 11.8352 | 0.170583 | 0.0141449 | 411 |
| bad | sqi_band_45_55 | 0.846608 | 35.1778 | 0.00476032 | 8.36698e-05 | 411 |
| good | diff_p95 | 0.844409 | -0.867227 | 0.535011 | 2.60712 | 3640 |

## Files

- Counts: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_gap_waveform_sqi_plus_counts.csv`
- Distance: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_gap_waveform_sqi_plus_distance.csv`
- Feature gaps: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_gap_waveform_sqi_plus_feature_gaps.csv`
