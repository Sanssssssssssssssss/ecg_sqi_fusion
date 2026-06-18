# N6800 Medium Coverage Gap Analysis

Selection note: original BUT metrics are not used here. This compares N6800 trim-bad medium target rows against PTB synthetic medium rows in the CleanBUT/SemiClean feature space.

## Consensus Feature Gaps
| feature | mean_ks | mean_median_gap_iqr | mean_high_tail_missing | mean_low_tail_missing | dominant_tail |
| --- | --- | --- | --- | --- | --- |
| sqi_iSQI | 1.0000 | -831895639.4818 | 0.0000 | 1.0000 | missing_low_tail |
| mean_abs | 0.6459 | 2.2807 | 0.0418 | 0.0000 | center_or_both_tails |
| hjorth_activity | 0.5308 | 0.7705 | 0.2117 | 0.0000 | missing_high_tail |
| std | 0.5308 | 0.9602 | 0.2117 | 0.0000 | missing_high_tail |
| rms | 0.5176 | 0.9300 | 0.2153 | 0.0000 | missing_high_tail |
| band_1_5 | 0.5008 | -1.5197 | 0.0038 | 0.3228 | missing_low_tail |
| sqi_bSQI | 0.4561 | 1.1480 | 0.0650 | 0.0359 | center_or_both_tails |
| diff_abs_p95 | 0.4525 | 0.7288 | 0.3538 | 0.0281 | missing_high_tail |
| ptp_p99_p01 | 0.4319 | 0.7724 | 0.3006 | 0.0000 | missing_high_tail |
| band_15_30 | 0.3940 | 0.9647 | 0.1841 | 0.0115 | center_or_both_tails |
| detail_instability | 0.3936 | 0.6356 | 0.2756 | 0.0253 | missing_high_tail |
| hf_ratio | 0.3925 | 0.8153 | 0.0300 | 0.0151 | center_or_both_tails |
| sample_entropy_proxy | 0.3773 | -0.9083 | 0.0008 | 0.2451 | missing_low_tail |
| sqi_pSQI | 0.3772 | -0.9160 | 0.0018 | 0.0071 | center_or_both_tails |
| qrs_width_median | 0.3308 | -0.6829 | 0.0113 | 0.0335 | center_or_both_tails |
| qrs_slope_median | 0.3108 | 0.5036 | 0.1840 | 0.0598 | center_or_both_tails |

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
- Consensus feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_v2_refit\medium_feature_gap_consensus_top.png`
- Best-trained medium PCA uncovered: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_v2_refit\best_trained_base_scan029_medium_uncovered_pca.png`
- Best-trained region concentration: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_v2_refit\best_trained_medium_region_uncovered_bar.png`