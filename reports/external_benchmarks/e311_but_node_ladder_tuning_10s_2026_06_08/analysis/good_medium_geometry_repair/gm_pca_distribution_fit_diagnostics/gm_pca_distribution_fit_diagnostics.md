# Good/Medium PCA Distribution Fit Diagnostics

- Generated: 2026-06-22 20:19:13
- PCA fit: BUT train+val good/medium waveform-computable features.
- BUT test is diagnostic only; no synthetic protocol here used BUT test for generation.

## Key PCA Metrics vs BUT Test

| compare | scope | class_name | js_pc12 | target_to_synth_nn3 | synth_to_target_nn3 | pc1_median_gap | pc2_median_gap | pc3_median_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v23_quantile | target_test | good | 0.445834 | 0.532951 | 0.345445 | -0.811866 | 0.625845 | -0.093857 |
| v23_quantile | target_test | medium | 0.39422 | 0.66549 | 0.301851 | 0.631017 | 0.17906 | -0.526377 |
| v27_subtype | target_test | good | 0.448868 | 0.489892 | 0.376011 | -0.751715 | 1.0376 | 0.231811 |
| v27_subtype | target_test | medium | 0.444229 | 0.895463 | 0.287927 | 0.362912 | 0.156877 | -0.461608 |
| v28_anchor | target_test | good | 0.412327 | 0.451919 | 0.41184 | -0.733262 | 0.781754 | 0.113589 |
| v28_anchor | target_test | medium | 0.26731 | 0.304607 | 0.320525 | 0.439595 | -0.024785 | -0.278256 |
| v30_goodfeature | target_test | good | 0.478908 | 0.492262 | 0.413006 | -0.793045 | 0.806317 | 0.135954 |
| v30_goodfeature | target_test | medium | 0.451824 | 0.923391 | 0.289659 | 0.350419 | 0.151697 | -0.459459 |
| v31_mediumhf | target_test | good | 0.456736 | 0.517184 | 0.362683 | -0.727801 | 0.915215 | 0.310283 |
| v31_mediumhf | target_test | medium | 0.456443 | 0.875067 | 0.304051 | 0.383644 | 0.154979 | -0.578801 |
| v32_mediumtemplate | target_test | good | 0.442953 | 0.466223 | 0.36218 | -0.725275 | 0.909912 | 0.302611 |
| v32_mediumtemplate | target_test | medium | 0.462697 | 0.892907 | 0.318465 | 0.194423 | 0.101865 | -0.309366 |
| v33_naturalmedium | target_test | good | 0.444123 | 0.489892 | 0.376011 | -0.751715 | 1.0376 | 0.231811 |
| v33_naturalmedium | target_test | medium | 0.440308 | 0.820795 | 0.308607 | 0.305614 | 0.145117 | -0.454393 |

## PTB -> BUT Cross Summary

| compare | direction | candidate | bucket | runs | acc | acc_std | macro_f1 | macro_f1_std | good_recall | good_recall_std | medium_recall | medium_recall_std | bad_recall | bad_recall_std | record_macro_acc | record_macro_acc_std | record_macro_supported_f1 | record_macro_supported_f1_std | bad_containing_record_bad_recall_mean | bad_containing_record_bad_recall_mean_std | artifact_positive_nonbad_bad_fpr | artifact_positive_nonbad_bad_fpr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v23_quantile | ptb_to_but | E1_query_only | cross_all | 1 | 0.826883 | 0 | 0.836586 | 0 | 0.865515 | 0 | 0.645012 | 0 | 0.999755 | 0 | 0.852116 | 0 | 0.802227 | 0 | 0.666667 | 0 | 0.0103195 | 0 |
| v23_quantile | ptb_to_but | E1_query_only | cross_test | 1 | 0.629259 | 0 | 0.682077 | 0 | 0.993028 | 0 | 0.432354 | 0 | 1 | 0 | 0.716053 | 0 | 0.760703 | 0 | 1 | 0 | 0.0192698 | 0 |
| v27_subtype | ptb_to_but | E1_query_only | cross_all | 1 | 0.84482 | 0 | 0.863712 | 0 | 0.821696 | 0 | 0.785315 | 0 | 0.999755 | 0 | 0.838996 | 0 | 0.787708 | 0 | 0.666667 | 0 | 0.00104561 | 0 |
| v27_subtype | ptb_to_but | E1_query_only | cross_test | 1 | 0.743357 | 0 | 0.816983 | 0 | 0.988048 | 0 | 0.610496 | 0 | 1 | 0 | 0.61582 | 0 | 0.643522 | 0 | 1 | 0 | 0.00100874 | 0 |
| v28_anchor | ptb_to_but | E1_query_only | cross_all | 1 | 0.843152 | 0 | 0.8596 | 0 | 0.849216 | 0 | 0.730247 | 0 | 0.999755 | 0 | 0.882877 | 0 | 0.862745 | 0 | 0.666667 | 0 | 0.000267666 | 0 |
| v28_anchor | ptb_to_but | E1_query_only | cross_test | 1 | 0.690216 | 0 | 0.785359 | 0 | 0.993028 | 0 | 0.52624 | 0 | 1 | 0 | 0.785373 | 0 | 0.816513 | 0 | 1 | 0 | 0 | 0 |
| v31_mediumhf | ptb_to_but | E1_query_only | cross_all | 1 | 0.798656 | 0 | 0.820369 | 0 | 0.759352 | 0 | 0.738069 | 0 | 0.999755 | 0 | 0.798168 | 0 | 0.747692 | 0 | 0.666667 | 0 | 0.0130458 | 0 |
| v31_mediumhf | ptb_to_but | E1_query_only | cross_test | 1 | 0.698968 | 0 | 0.734964 | 0 | 0.978088 | 0 | 0.546943 | 0 | 1 | 0 | 0.585332 | 0 | 0.603446 | 0 | 1 | 0 | 0.0174908 | 0 |
| v32_mediumtemplate | ptb_to_but | E1_query_only | cross_all | 1 | 0.771727 | 0 | 0.765005 | 0 | 0.843516 | 0 | 0.494493 | 0 | 0.999755 | 0 | 0.774362 | 0 | 0.695081 | 0 | 0.666667 | 0 | 0.0473974 | 0 |
| v32_mediumtemplate | ptb_to_but | E1_query_only | cross_test | 1 | 0.500156 | 0 | 0.476447 | 0 | 0.984064 | 0 | 0.237843 | 0 | 1 | 0 | 0.563158 | 0 | 0.600224 | 0 | 1 | 0 | 0.09889 | 0 |
| v33_naturalmedium | ptb_to_but | E1_query_only | cross_all | 1 | 0.845098 | 0 | 0.859489 | 0 | 0.833986 | 0 | 0.764246 | 0 | 0.999755 | 0 | 0.850148 | 0 | 0.796283 | 0 | 0.666667 | 0 | 0.00914913 | 0 |
| v33_naturalmedium | ptb_to_but | E1_query_only | cross_test | 1 | 0.734605 | 0 | 0.759275 | 0 | 0.988048 | 0 | 0.597015 | 0 | 1 | 0 | 0.609767 | 0 | 0.636366 | 0 | 1 | 0 | 0.0174849 | 0 |

## Largest Feature Gaps vs BUT Test

| compare | scope | class_name | feature | target_median | synthetic_median | robust_z_gap | abs_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v23_quantile | target_test | good | non_qrs_diff_p95 | 0.680265 | 1.4105 | 1.77193 | 1.77193 |
| v23_quantile | target_test | good | qrs_band_ratio | 10 | 8.71627 | -1.66032 | 1.66032 |
| v23_quantile | target_test | good | sqi_basSQI | 0.438841 | 0.522263 | 1.03493 | 1.03493 |
| v23_quantile | target_test | good | baseline_step | 0.319683 | 0.228686 | -0.861228 | 0.861228 |
| v23_quantile | target_test | good | amplitude_entropy | 0.773808 | 0.81457 | 0.836978 | 0.836978 |
| v23_quantile | target_test | medium | band_30_45 | 0.0381995 | 0.022203 | -0.890424 | 0.890424 |
| v23_quantile | target_test | medium | amplitude_entropy | 0.795204 | 0.815881 | 0.676225 | 0.676225 |
| v23_quantile | target_test | medium | mean_abs | 1.01298 | 1.08427 | 0.392754 | 0.392754 |
| v23_quantile | target_test | medium | template_corr | 0.849193 | 0.928448 | 0.370615 | 0.370615 |
| v23_quantile | target_test | medium | band_15_30 | 0.204857 | 0.243032 | 0.319379 | 0.319379 |
| v27_subtype | target_test | good | non_qrs_diff_p95 | 0.680265 | 1.60853 | 2.25246 | 2.25246 |
| v27_subtype | target_test | good | qrs_band_ratio | 10 | 8.54934 | -1.87621 | 1.87621 |
| v27_subtype | target_test | good | sqi_basSQI | 0.438841 | 0.52383 | 1.05438 | 1.05438 |
| v27_subtype | target_test | good | baseline_step | 0.319683 | 0.227254 | -0.874785 | 0.874785 |
| v27_subtype | target_test | good | band_15_30 | 0.179661 | 0.238304 | 0.871159 | 0.871159 |
| v27_subtype | target_test | medium | band_30_45 | 0.0381995 | 0.0226043 | -0.868086 | 0.868086 |
| v27_subtype | target_test | medium | amplitude_entropy | 0.795204 | 0.817469 | 0.728171 | 0.728171 |
| v27_subtype | target_test | medium | band_15_30 | 0.204857 | 0.242516 | 0.315057 | 0.315057 |
| v27_subtype | target_test | medium | mean_abs | 1.01298 | 1.05879 | 0.252362 | 0.252362 |
| v27_subtype | target_test | medium | template_corr | 0.849193 | 0.890326 | 0.192347 | 0.192347 |
| v28_anchor | target_test | good | non_qrs_diff_p95 | 0.680265 | 1.5319 | 2.06653 | 2.06653 |
| v28_anchor | target_test | good | qrs_band_ratio | 10 | 8.75404 | -1.61146 | 1.61146 |
| v28_anchor | target_test | good | sqi_basSQI | 0.438841 | 0.521346 | 1.02355 | 1.02355 |
| v28_anchor | target_test | good | baseline_step | 0.319683 | 0.229528 | -0.853257 | 0.853257 |
| v28_anchor | target_test | good | detector_agreement | 0.833333 | 0.666667 | -0.77651 | 0.77651 |
| v28_anchor | target_test | medium | band_30_45 | 0.0381995 | 0.022969 | -0.847782 | 0.847782 |
| v28_anchor | target_test | medium | amplitude_entropy | 0.795204 | 0.816562 | 0.698484 | 0.698484 |
| v28_anchor | target_test | medium | mean_abs | 1.01298 | 1.07805 | 0.358476 | 0.358476 |
| v28_anchor | target_test | medium | band_15_30 | 0.204857 | 0.240465 | 0.297902 | 0.297902 |
| v28_anchor | target_test | medium | low_amp_ratio | 0.2048 | 0.2112 | 0.241514 | 0.241514 |
| v30_goodfeature | target_test | good | non_qrs_diff_p95 | 0.680265 | 1.55743 | 2.12847 | 2.12847 |
| v30_goodfeature | target_test | good | qrs_band_ratio | 10 | 8.36409 | -2.11581 | 2.11581 |
| v30_goodfeature | target_test | good | sqi_basSQI | 0.438841 | 0.52383 | 1.05438 | 1.05438 |
| v30_goodfeature | target_test | good | baseline_step | 0.319683 | 0.227254 | -0.874785 | 0.874785 |
| v30_goodfeature | target_test | good | amplitude_entropy | 0.773808 | 0.815575 | 0.857616 | 0.857616 |
| v30_goodfeature | target_test | medium | band_30_45 | 0.0381995 | 0.0225472 | -0.871262 | 0.871262 |
| v30_goodfeature | target_test | medium | amplitude_entropy | 0.795204 | 0.817533 | 0.730261 | 0.730261 |
| v30_goodfeature | target_test | medium | band_15_30 | 0.204857 | 0.241945 | 0.310279 | 0.310279 |
| v30_goodfeature | target_test | medium | mean_abs | 1.01298 | 1.05369 | 0.224245 | 0.224245 |
| v30_goodfeature | target_test | medium | template_corr | 0.849193 | 0.889533 | 0.188641 | 0.188641 |
| v31_mediumhf | target_test | good | non_qrs_diff_p95 | 0.680265 | 1.57697 | 2.17589 | 2.17589 |
| v31_mediumhf | target_test | good | qrs_band_ratio | 10 | 8.60505 | -1.80415 | 1.80415 |
| v31_mediumhf | target_test | good | detector_agreement | 0.833333 | 0.583333 | -1.16477 | 1.16477 |
| v31_mediumhf | target_test | good | sqi_basSQI | 0.438841 | 0.520689 | 1.0154 | 1.0154 |
| v31_mediumhf | target_test | good | band_15_30 | 0.179661 | 0.238508 | 0.874187 | 0.874187 |
| v31_mediumhf | target_test | medium | band_30_45 | 0.0381995 | 0.02226 | -0.887249 | 0.887249 |
| v31_mediumhf | target_test | medium | amplitude_entropy | 0.795204 | 0.817604 | 0.732586 | 0.732586 |
| v31_mediumhf | target_test | medium | detector_agreement | 0.5 | 0.666667 | 0.333333 | 0.333333 |
| v31_mediumhf | target_test | medium | band_15_30 | 0.204857 | 0.242168 | 0.312147 | 0.312147 |
| v31_mediumhf | target_test | medium | mean_abs | 1.01298 | 1.0671 | 0.298136 | 0.298136 |
| v32_mediumtemplate | target_test | good | non_qrs_diff_p95 | 0.680265 | 1.56813 | 2.15444 | 2.15444 |
| v32_mediumtemplate | target_test | good | qrs_band_ratio | 10 | 8.60142 | -1.80885 | 1.80885 |
| v32_mediumtemplate | target_test | good | detector_agreement | 0.833333 | 0.583333 | -1.16477 | 1.16477 |
| v32_mediumtemplate | target_test | good | sqi_basSQI | 0.438841 | 0.520583 | 1.01408 | 1.01408 |
| v32_mediumtemplate | target_test | good | band_15_30 | 0.179661 | 0.238157 | 0.868979 | 0.868979 |
| v32_mediumtemplate | target_test | medium | band_30_45 | 0.0381995 | 0.0218571 | -0.909674 | 0.909674 |
| v32_mediumtemplate | target_test | medium | amplitude_entropy | 0.795204 | 0.818218 | 0.752653 | 0.752653 |
| v32_mediumtemplate | target_test | medium | band_15_30 | 0.204857 | 0.238127 | 0.278342 | 0.278342 |
| v32_mediumtemplate | target_test | medium | low_amp_ratio | 0.2048 | 0.2096 | 0.181136 | 0.181136 |
| v32_mediumtemplate | target_test | medium | template_corr | 0.849193 | 0.886594 | 0.174896 | 0.174896 |
| v33_naturalmedium | target_test | good | non_qrs_diff_p95 | 0.680265 | 1.60853 | 2.25246 | 2.25246 |
| v33_naturalmedium | target_test | good | qrs_band_ratio | 10 | 8.54934 | -1.87621 | 1.87621 |
| v33_naturalmedium | target_test | good | sqi_basSQI | 0.438841 | 0.52383 | 1.05438 | 1.05438 |
| v33_naturalmedium | target_test | good | baseline_step | 0.319683 | 0.227254 | -0.874785 | 0.874785 |
| v33_naturalmedium | target_test | good | band_15_30 | 0.179661 | 0.238304 | 0.871159 | 0.871159 |
| v33_naturalmedium | target_test | medium | band_30_45 | 0.0381995 | 0.0222482 | -0.887908 | 0.887908 |
| v33_naturalmedium | target_test | medium | amplitude_entropy | 0.795204 | 0.817971 | 0.744584 | 0.744584 |
| v33_naturalmedium | target_test | medium | band_15_30 | 0.204857 | 0.24212 | 0.311745 | 0.311745 |
| v33_naturalmedium | target_test | medium | mean_abs | 1.01298 | 1.06187 | 0.269345 | 0.269345 |
| v33_naturalmedium | target_test | medium | template_corr | 0.849193 | 0.898938 | 0.232621 | 0.232621 |

## Figures

- PCA panels: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_pca_distribution_fit_diagnostics\gm_pca_distribution_panels.png`
- Metric bars: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_pca_distribution_fit_diagnostics\gm_pca_distribution_metric_bars.png`

## Interpretation Guardrail

A smaller PCA density distance is necessary but not sufficient.  The class boundary density and train/test target shift must also match; otherwise the model can learn a smoother but wrong good/medium decision surface.