# Original Record Error Deep Dive

This report isolates the largest remaining original BUT test errors after the simple wide-good `pc1 + qrs_prom_p90` rule. Original BUT remains report-only.

## Big Error Blocks
| record_id | class_name | pred_class | n | support | rate |
| --- | --- | --- | --- | --- | --- |
| 111001 | good | medium | 246 | 3336 | 0.0737 |
| 111001 | medium | good | 213 | 4390 | 0.0485 |
| 111001 | bad | medium | 140 | 292 | 0.4795 |
| 111001 | medium | bad | 80 | 4390 | 0.0182 |
| 125001 | good | medium | 75 | 228 | 0.3289 |
| 111001 | bad | good | 12 | 292 | 0.0411 |
| 111001 | good | bad | 10 | 3336 | 0.0030 |
| 122001 | good | medium | 3 | 76 | 0.0395 |

## Focus Block Counts
| focus_group | record_id | class_name | pred_class | n |
| --- | --- | --- | --- | --- |
| correct medium | 111001 | medium | medium | 4097 |
| correct good | 111001 | good | good | 3080 |
| 111001 good->medium | 111001 | good | medium | 246 |
| 111001 medium->good | 111001 | medium | good | 213 |
| 111001 bad->medium | 111001 | bad | medium | 140 |
| correct bad | 122001 | bad | bad | 119 |
| 125001 good->medium | 125001 | good | medium | 75 |

## Focus Block Feature Medians
| focus_group | n | pc1 | pc2 | pc3 | qrs_prom_p90 | qrs_visibility | qrs_band_ratio | baseline_step | flatline_ratio | non_qrs_rms_ratio | non_qrs_diff_p95 | diff_abs_p95 | amplitude_entropy | template_corr | detector_agreement | sqi_sSQI | sqi_kSQI | band_15_30 | band_30_45 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct medium | 4097 | -0.9224 | 4.9999 | 1.4455 | 5.5199 | 0.0640 | 0.3568 | 1.0406 | 0.1233 | 0.5683 | 0.0773 | 0.1211 | 0.7436 | 0.4747 | 0.2586 | 1.9033 | 11.6999 | 0.1676 | 0.0296 |
| correct good | 3080 | -5.4751 | 5.1005 | -1.4306 | 7.9475 | 0.2361 | 0.4122 | 0.7832 | 0.4035 | 0.2445 | 0.0192 | 0.0567 | 0.6522 | 0.5766 | 0.2586 | 4.5666 | 31.8951 | 0.1662 | 0.0193 |
| 111001 good->medium | 246 | -3.6826 | 10.9955 | -0.4459 | 4.0606 | 0.0644 | 0.2101 | 1.0905 | 0.2626 | 0.6267 | 0.0192 | 0.0375 | 0.8284 | 0.5192 | 0.2136 | 0.8460 | 4.9454 | 0.0573 | 0.0062 |
| 111001 medium->good | 213 | -3.4649 | 8.4275 | 0.4412 | 5.0746 | 0.0790 | 0.2673 | 1.0203 | 0.2298 | 0.5268 | 0.0300 | 0.0566 | 0.7855 | 0.5108 | 0.2344 | 1.4303 | 7.8151 | 0.0817 | 0.0093 |
| 111001 bad->medium | 140 | -3.4469 | 12.9947 | -0.6796 | 2.7721 | 0.0252 | 0.0986 | 1.5210 | 0.3667 | 0.7083 | 0.0403 | 0.0585 | 0.8258 | 0.2975 | 0.2599 | -0.3004 | 4.6721 | 0.0217 | 0.0057 |
| correct bad | 119 | 9.0054 | -0.9604 | 0.3024 | 2.9843 | 0.1041 | 0.3316 | 0.2475 | 0.0088 | 0.9217 | 0.4172 | 0.4374 | 0.8842 | 0.1962 | 0.5505 | 0.0068 | 3.0045 | 0.3210 | 0.2767 |
| 125001 good->medium | 75 | -2.2697 | 8.4315 | 1.1264 | 3.1799 | 0.1038 | 0.3540 | 1.2803 | 0.1689 | 0.7426 | 0.0297 | 0.0811 | 0.8546 | 0.7122 | 0.2058 | 0.7948 | 3.6605 | 0.1253 | 0.0076 |

## Top Feature Gaps
| comparison | feature | left_median | right_median | robust_effect | ks | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 111001 bad->medium vs correct bad | band_30_45 | 0.0057 | 0.2767 | -10.3310 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | non_qrs_diff_p95 | 0.0403 | 0.4172 | -9.2714 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | diff_abs_p95 | 0.0585 | 0.4374 | -8.8608 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | pc1 | -3.4469 | 9.0054 | -7.5434 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | band_15_30 | 0.0217 | 0.3210 | -7.5386 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | pc2 | 12.9947 | -0.9604 | 3.7641 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | flatline_ratio | 0.3667 | 0.0088 | 3.2509 | 1.0000 | 140 | 119 |
| 111001 bad->medium vs correct bad | baseline_step | 1.5210 | 0.2475 | 3.3003 | 0.9857 | 140 | 119 |
| 111001 good->medium vs correct good | qrs_visibility | 0.0644 | 0.2361 | -1.4188 | 0.6067 | 246 | 3080 |
| 111001 good->medium vs correct good | qrs_prom_p90 | 4.0606 | 7.9475 | -1.4538 | 0.6043 | 246 | 3080 |
| 111001 good->medium vs correct good | non_qrs_rms_ratio | 0.6267 | 0.2445 | 1.4407 | 0.5789 | 246 | 3080 |
| 111001 good->medium vs correct good | sqi_kSQI | 4.9454 | 31.8951 | -1.4509 | 0.5641 | 246 | 3080 |
| 111001 good->medium vs correct good | sqi_sSQI | 0.8460 | 4.5666 | -1.4325 | 0.5386 | 246 | 3080 |
| 111001 good->medium vs correct good | amplitude_entropy | 0.8284 | 0.6522 | 1.2002 | 0.5334 | 246 | 3080 |
| 111001 good->medium vs correct good | band_15_30 | 0.0573 | 0.1662 | -1.1260 | 0.5281 | 246 | 3080 |
| 111001 good->medium vs correct good | pc1 | -3.6826 | -5.4751 | 1.0007 | 0.5189 | 246 | 3080 |
| 111001 medium->good vs 111001 good->medium | qrs_prom_p90 | 5.0746 | 4.0606 | 0.2918 | 0.3555 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | non_qrs_diff_p95 | 0.0300 | 0.0192 | 0.4013 | 0.2928 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | non_qrs_rms_ratio | 0.5268 | 0.6267 | -0.3611 | 0.2839 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | flatline_ratio | 0.2298 | 0.2626 | -0.2021 | 0.2804 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | sqi_kSQI | 7.8151 | 4.9454 | 0.1541 | 0.2802 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | diff_abs_p95 | 0.0566 | 0.0375 | 0.4061 | 0.2757 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | sqi_sSQI | 1.4303 | 0.8460 | 0.2055 | 0.2580 | 213 | 246 |
| 111001 medium->good vs 111001 good->medium | qrs_visibility | 0.0790 | 0.0644 | 0.2455 | 0.2562 | 213 | 246 |
| 111001 medium->good vs correct medium | pc1 | -3.4649 | -0.9224 | -1.2581 | 0.6320 | 213 | 4097 |
| 111001 medium->good vs correct medium | flatline_ratio | 0.2298 | 0.1233 | 0.8261 | 0.5091 | 213 | 4097 |
| 111001 medium->good vs correct medium | non_qrs_diff_p95 | 0.0300 | 0.0773 | -0.7885 | 0.4837 | 213 | 4097 |
| 111001 medium->good vs correct medium | diff_abs_p95 | 0.0566 | 0.1211 | -0.4291 | 0.4046 | 213 | 4097 |
| 111001 medium->good vs correct medium | band_30_45 | 0.0093 | 0.0296 | -0.6408 | 0.3146 | 213 | 4097 |
| 111001 medium->good vs correct medium | qrs_visibility | 0.0790 | 0.0640 | 0.1411 | 0.2845 | 213 | 4097 |
| 111001 medium->good vs correct medium | qrs_prom_p90 | 5.0746 | 5.5199 | -0.1231 | 0.2837 | 213 | 4097 |
| 111001 medium->good vs correct medium | pc3 | 0.4412 | 1.4455 | -0.3450 | 0.2377 | 213 | 4097 |
| 125001 good->medium vs 111001 medium->good | template_corr | 0.7122 | 0.5108 | 1.6885 | 0.7681 | 75 | 213 |
| 125001 good->medium vs 111001 medium->good | pc1 | -2.2697 | -3.4649 | 0.9771 | 0.7031 | 75 | 213 |
| 125001 good->medium vs 111001 medium->good | qrs_prom_p90 | 3.1799 | 5.0746 | -0.8112 | 0.6038 | 75 | 213 |
| 125001 good->medium vs 111001 medium->good | non_qrs_rms_ratio | 0.7426 | 0.5268 | 1.0640 | 0.5801 | 75 | 213 |
| 125001 good->medium vs 111001 medium->good | sqi_kSQI | 3.6605 | 7.8151 | -0.3102 | 0.5489 | 75 | 213 |
| 125001 good->medium vs 111001 medium->good | amplitude_entropy | 0.8546 | 0.7855 | 0.5663 | 0.4485 | 75 | 213 |
| 125001 good->medium vs 111001 medium->good | band_30_45 | 0.0076 | 0.0093 | -0.1112 | 0.4460 | 75 | 213 |
| 125001 good->medium vs 111001 medium->good | diff_abs_p95 | 0.0811 | 0.0566 | 0.5622 | 0.4248 | 75 | 213 |
| 125001 good->medium vs correct good | pc1 | -2.2697 | -5.4751 | 3.0081 | 0.9285 | 75 | 3080 |
| 125001 good->medium vs correct good | flatline_ratio | 0.1689 | 0.4035 | -2.4519 | 0.8522 | 75 | 3080 |
| 125001 good->medium vs correct good | qrs_prom_p90 | 3.1799 | 7.9475 | -3.1083 | 0.8228 | 75 | 3080 |
| 125001 good->medium vs correct good | sqi_kSQI | 3.6605 | 31.8951 | -2.1155 | 0.7756 | 75 | 3080 |
| 125001 good->medium vs correct good | non_qrs_rms_ratio | 0.7426 | 0.2445 | 2.6020 | 0.7549 | 75 | 3080 |
| 125001 good->medium vs correct good | amplitude_entropy | 0.8546 | 0.6522 | 1.8344 | 0.7427 | 75 | 3080 |
| 125001 good->medium vs correct good | sqi_sSQI | 0.7948 | 4.5666 | -2.0740 | 0.7234 | 75 | 3080 |
| 125001 good->medium vs correct good | band_30_45 | 0.0076 | 0.0193 | -1.2692 | 0.6987 | 75 | 3080 |

## Visuals
![Record error waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_record_error_waveforms.png)

![Record error geometry](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_record_error_geometry.png)

![Record feature boxes](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_record_error_feature_boxes.png)

## Working Interpretation
- `125001 good->medium` is a record-level good-domain shift: almost the entire good block is being treated as medium by the clean-node-derived rule.
- `111001 bad->medium` is not the same as good/medium overlap; it is the controlled bad-outlier stress slice and needs a separate bad-stress rule or data block.
- `111001 medium->good` is the key counterexample for any wider good rescue: it has high QRS evidence but still belongs to medium, so a simple rule needs an additional medium guard if it tries to rescue more good.
- The next useful experiment should target these large blocks directly instead of tiny boundary bisection steps.
