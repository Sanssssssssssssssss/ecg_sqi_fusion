# N6800 Medium Coverage Gap Analysis

Selection note: original BUT metrics are not used here. This compares N6800 trim-bad medium target rows against PTB synthetic medium rows in the CleanBUT/SemiClean feature space.

## Consensus Feature Gaps
| feature | mean_ks | mean_median_gap_iqr | mean_high_tail_missing | mean_low_tail_missing | dominant_tail |
| --- | --- | --- | --- | --- | --- |
| sqi_iSQI | 1.0000 | -832528735.6322 | 0.0000 | 1.0000 | missing_low_tail |
| mean_abs | 0.5945 | 1.9404 | 0.0004 | 0.0000 | center_or_both_tails |
| sqi_bSQI | 0.4694 | 1.1948 | 0.0545 | 0.0357 | center_or_both_tails |
| hjorth_activity | 0.4323 | 0.6301 | 0.0821 | 0.0000 | missing_high_tail |
| std | 0.4323 | 0.7186 | 0.0821 | 0.0000 | missing_high_tail |
| rms | 0.4156 | 0.6889 | 0.0838 | 0.0000 | missing_high_tail |
| band_1_5 | 0.4064 | -0.9806 | 0.0053 | 0.1625 | missing_low_tail |
| sqi_pSQI | 0.4048 | -1.0249 | 0.0001 | 0.0080 | center_or_both_tails |
| hf_ratio | 0.3815 | 0.7865 | 0.0205 | 0.0126 | center_or_both_tails |
| diff_abs_p95 | 0.3789 | 0.6375 | 0.2457 | 0.0340 | missing_high_tail |
| ptp_p99_p01 | 0.3090 | 0.5436 | 0.1370 | 0.0000 | missing_high_tail |
| band_15_30 | 0.3077 | 0.7883 | 0.0298 | 0.0121 | center_or_both_tails |
| band_30_45 | 0.3058 | 0.5382 | 0.0447 | 0.0166 | center_or_both_tails |
| wavelet_e4 | 0.3044 | 0.5088 | 0.0597 | 0.0429 | center_or_both_tails |
| qrs_width_median | 0.3008 | -0.5161 | 0.0139 | 0.0169 | center_or_both_tails |
| non_qrs_diff_p95 | 0.3001 | 0.4849 | 0.1130 | 0.0215 | missing_high_tail |

## Best-Trained Base Region/Tier Concentration
| variant_label | group_type | group_value | n | mean_nearest64 | p90_nearest64 | uncovered64_top10_rate | uncovered64_top05_rate | mean_nearest_pca | uncovered_pca_top10_rate | variant_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trained_base_scan029 | ambiguous_type | medium_bad_boundary | 9 | 8.7592 | 10.4283 | 0.7778 | 0.7778 | 1.0043 | 0.4444 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | ambiguous_type | isolated_medium | 560 | 6.7526 | 8.5492 | 0.6036 | 0.4179 | 1.1505 | 0.6768 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | ambiguous_type | good_medium_boundary | 4141 | 4.6886 | 5.6647 | 0.0594 | 0.0169 | 0.2646 | 0.0488 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | ambiguous_type | clean_or_target | 1970 | 4.4044 | 5.3977 | 0.0426 | 0.0142 | 0.2350 | 0.0472 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | ambiguous_type | good_medium_low_purity | 120 | 4.5303 | 5.5461 | 0.0417 | 0.0083 | 0.2298 | 0.0167 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | clean_tier | ambiguous_boundary | 4830 | 4.9315 | 6.0514 | 0.1234 | 0.0646 | 0.3678 | 0.1215 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | clean_tier | clean_core_strict | 1137 | 4.3294 | 5.3119 | 0.0440 | 0.0167 | 0.2494 | 0.0554 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | clean_tier | clean_core_train_target | 833 | 4.5068 | 5.4771 | 0.0408 | 0.0108 | 0.2154 | 0.0360 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | original_region | medium_bad_overlap | 9 | 8.7592 | 10.4283 | 0.7778 | 0.7778 | 1.0043 | 0.4444 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | original_region | outlier_low_confidence | 680 | 6.3604 | 8.2089 | 0.5044 | 0.3456 | 0.9880 | 0.5603 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | original_region | good_medium_overlap | 4141 | 4.6886 | 5.6647 | 0.0594 | 0.0169 | 0.2646 | 0.0488 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |
| best_trained_base_scan029 | original_region | clean_core | 1970 | 4.4044 | 5.3977 | 0.0426 | 0.0142 | 0.2350 | 0.0472 | nl_n6800_gm_trim_bad_scan_029_sc_overlap_narrow_oscillato_18aa71dcf07e |

## Key Figures
- Consensus feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_refit\medium_feature_gap_consensus_top.png`
- Best-trained medium PCA uncovered: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_refit\best_trained_base_scan029_medium_uncovered_pca.png`
- Best-trained region concentration: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_refit\best_trained_medium_region_uncovered_bar.png`