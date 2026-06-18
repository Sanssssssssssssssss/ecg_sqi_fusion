# N6800 Medium Coverage Gap Analysis

Selection note: original BUT metrics are not used here. This compares N6800 trim-bad medium target rows against PTB synthetic medium rows in the CleanBUT/SemiClean feature space.

## Consensus Feature Gaps
| feature | mean_ks | mean_median_gap_iqr | mean_high_tail_missing | mean_low_tail_missing | dominant_tail |
| --- | --- | --- | --- | --- | --- |
| sqi_iSQI | 1.0000 | -833290655.5280 | 0.0000 | 1.0000 | missing_low_tail |
| mean_abs | 0.6156 | 2.0724 | 0.0184 | 0.0000 | center_or_both_tails |
| hjorth_activity | 0.4709 | 0.6870 | 0.1392 | 0.0000 | missing_high_tail |
| std | 0.4709 | 0.8179 | 0.1392 | 0.0000 | missing_high_tail |
| band_1_5 | 0.4623 | -1.2571 | 0.0030 | 0.2609 | missing_low_tail |
| rms | 0.4578 | 0.7898 | 0.1418 | 0.0000 | missing_high_tail |
| sqi_bSQI | 0.4489 | 1.1331 | 0.0600 | 0.0352 | center_or_both_tails |
| diff_abs_p95 | 0.4226 | 0.6896 | 0.3121 | 0.0280 | missing_high_tail |
| sqi_pSQI | 0.4057 | -1.0377 | 0.0008 | 0.0091 | center_or_both_tails |
| hf_ratio | 0.3958 | 0.8205 | 0.0339 | 0.0124 | center_or_both_tails |
| ptp_p99_p01 | 0.3586 | 0.6549 | 0.2181 | 0.0000 | missing_high_tail |
| detail_instability | 0.3564 | 0.5835 | 0.1920 | 0.0277 | missing_high_tail |
| band_15_30 | 0.3480 | 0.8638 | 0.0959 | 0.0093 | center_or_both_tails |
| qrs_width_median | 0.3221 | -0.6489 | 0.0104 | 0.0319 | center_or_both_tails |
| sample_entropy_proxy | 0.3161 | -0.7007 | 0.0007 | 0.1454 | missing_low_tail |
| band_30_45 | 0.3034 | 0.5044 | 0.0479 | 0.0194 | missing_high_tail |

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
- Consensus feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_v5_mound_refit\medium_feature_gap_consensus_top.png`
- Best-trained medium PCA uncovered: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_v5_mound_refit\best_trained_base_scan029_medium_uncovered_pca.png`
- Best-trained region concentration: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\medium_coverage_gap_n6800_subtype_v5_mound_refit\best_trained_medium_region_uncovered_bar.png`