# Bad Overlap Deep Dive

Analysis-only row-level audit. Existing thresholds are used as-is and original BUT is report-only.

## Controls

- Base model: `expertmix_medium_guard_expertonly`.
- Bad detectors: `flatbad_strict` and `highspec_bad_strict`.
- Detector thresholds were selected on synthetic/augmented validation in their original runs; this script does not tune thresholds.
- Original rows are used only to diagnose failure modes and draw visual panels.

## Bucket Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | good->bad | medium->bad | bad->medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_expertmix_medium_guard_expertonly | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 0 | 0 | 75 |
| base_expertmix_medium_guard_expertonly | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| base_expertmix_medium_guard_expertonly | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 73 |
| flatbad_strict_veto | original_test_all_10s+ | 0.809602 | 0.682469 | 0.829670 | 0.816765 | 0.554745 | 0 | 0 | 46 |
| flatbad_strict_veto | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict_veto | bad_outlier_stress | 0.380137 | 0.183623 | 0.000000 | 0.000000 | 0.380137 | 0 | 0 | 44 |
| highspec_bad_strict_veto | original_test_all_10s+ | 0.769258 | 0.641948 | 0.731593 | 0.817668 | 0.581509 | 0 | 0 | 53 |
| highspec_bad_strict_veto | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| highspec_bad_strict_veto | bad_outlier_stress | 0.417808 | 0.196457 | 0.000000 | 0.000000 | 0.417808 | 0 | 0 | 51 |

## Row Groups

| Group | Count |
|---|---:|
| nonbad_clean_ok | 5972 |
| false_bad_medium_any | 874 |
| not_in_focus | 714 |
| false_bad_good_any | 509 |
| bad_stress_rescued_any | 180 |
| bad_core_kept_bad | 117 |
| bad_stress_still_missed_both | 111 |

## Visuals

![PCA groups](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/bad_overlap_deep_dive/bad_overlap_pca_groups.png)

![Detector scores](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/bad_overlap_deep_dive/bad_overlap_detector_scores.png)

![Feature boxplots](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/bad_overlap_deep_dive/bad_overlap_feature_boxplots.png)

![Waveform panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/bad_overlap_deep_dive/bad_overlap_waveform_panels.png)

## Top Feature Gaps

### rescued_vs_false_good

| Feature | KS | Median A | Median B | nA | nB |
|---|---:|---:|---:|---:|---:|
| detail_ratio_p90 | 0.773 | 0.7766 | 1.165 | 180 | 509 |
| band_15_30 | 0.760 | 0.1087 | 0.4988 | 180 | 509 |
| qrs_template_corr_median | 0.742 | 0.6564 | 0.9284 | 180 | 509 |
| band_0.5_5 | 0.727 | 7.831 | 4.554 | 180 | 509 |
| band_5_15 | 0.701 | 0.6984 | 1.575 | 180 | 509 |
| hjorth_complexity_proxy | 0.663 | 4.439 | 2.171 | 180 | 509 |

### rescued_vs_false_medium

| Feature | KS | Median A | Median B | nA | nB |
|---|---:|---:|---:|---:|---:|
| peak_interval_cv | 0.461 | 0.8947 | 0.5865 | 180 | 874 |
| stress_lowfreq_detail_gap | 0.454 | 0.4784 | 1.075 | 180 | 874 |
| qrs_template_corr_median | 0.450 | 0.6564 | 0.862 | 180 | 874 |
| baseline251_std_ratio | 0.443 | 0.4105 | 0.6486 | 180 | 874 |
| baseline125_std_ratio | 0.433 | 0.5726 | 0.8078 | 180 | 874 |
| baseline63_std_ratio | 0.432 | 0.7104 | 0.8894 | 180 | 874 |

### rescued_vs_still_missed

| Feature | KS | Median A | Median B | nA | nB |
|---|---:|---:|---:|---:|---:|
| peak_interval_cv | 0.350 | 0.8947 | 0.7023 | 180 | 111 |
| longest_flat_run_0p01 | 0.328 | 0.112 | 0.064 | 180 | 111 |
| peak_interval_median | 0.315 | 0.404 | 0.568 | 180 | 111 |
| baseline125_std_ratio | 0.248 | 0.5726 | 0.4865 | 180 | 111 |
| stress_lowfreq_detail_gap | 0.244 | 0.4784 | 0.3649 | 180 | 111 |
| baseline251_std_ratio | 0.235 | 0.4105 | 0.3291 | 180 | 111 |

### false_nonbad_vs_clean_nonbad

| Feature | KS | Median A | Median B | nA | nB |
|---|---:|---:|---:|---:|---:|
| low_amp_ratio_0.2 | 0.528 | 0.2344 | 0.1712 | 509 | 5972 |
| longest_lowamp_run_0p05 | 0.519 | 0.152 | 0.048 | 509 | 5972 |
| low_amp_ratio_0.1 | 0.513 | 0.136 | 0.0872 | 509 | 5972 |
| stress_flat_lowamp_joint | 0.501 | 0.01601 | 0.004003 | 509 | 5972 |
| low_amp_ratio_0.05 | 0.501 | 0.0728 | 0.0448 | 509 | 5972 |
| low_amp_ratio_0.03 | 0.474 | 0.0448 | 0.0272 | 509 | 5972 |

### false_medium_vs_clean_nonbad

| Feature | KS | Median A | Median B | nA | nB |
|---|---:|---:|---:|---:|---:|
| baseline125_std_ratio | 0.545 | 0.8078 | 0.3534 | 874 | 5972 |
| stress_lowfreq_detail_gap | 0.541 | 1.075 | 0.2727 | 874 | 5972 |
| baseline63_std_ratio | 0.541 | 0.8894 | 0.477 | 874 | 5972 |
| baseline251_std_ratio | 0.529 | 0.6486 | 0.2648 | 874 | 5972 |
| hjorth_complexity_proxy | 0.515 | 6.422 | 2.017 | 874 | 5972 |
| detail_ratio_p90 | 0.505 | 0.6896 | 1.208 | 874 | 5972 |

### bad_core_vs_bad_stress_rescued

| Feature | KS | Median A | Median B | nA | nB |
|---|---:|---:|---:|---:|---:|
| diff_p50 | 1.000 | 0.9551 | 0.02821 | 117 | 180 |
| diff_p75 | 1.000 | 1.609 | 0.08809 | 117 | 180 |
| diff_p90 | 1.000 | 2.282 | 0.3239 | 117 | 180 |
| zero_crossing_rate | 1.000 | 0.4908 | 0.03243 | 117 | 180 |
| flatline_ratio_0.005 | 1.000 | 0.002402 | 0.1225 | 117 | 180 |
| flatline_ratio_0.01 | 1.000 | 0.005604 | 0.231 | 117 | 180 |


## Interpretation

- The current best waveform/stat model is strong on good/medium and bad core, but almost blind to original bad stress.
- The strict bad detectors recover a portion of bad stress, yet their false positives form a nearby non-bad shell. The next improvement must separate true flatline/baseline bad stress from visually similar low-amplitude or baseline-wandering good/medium.
- This should be handled with a controlled analysis-to-improvement step: generate synthetic analogs for both true bad stress and non-bad hard negatives, then re-evaluate before training.

## Files

- Row diagnostics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_overlap_deep_dive_rows.csv`
- Feature gaps: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_overlap_deep_dive_feature_gaps.csv`
- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_overlap_deep_dive_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_overlap_deep_dive_summary.json`
