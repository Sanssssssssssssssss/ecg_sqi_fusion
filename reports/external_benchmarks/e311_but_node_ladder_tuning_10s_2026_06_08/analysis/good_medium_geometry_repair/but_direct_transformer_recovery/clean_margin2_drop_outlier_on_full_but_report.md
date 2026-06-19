# BUT Direct Transformer Recovery: clean_margin2_drop_outlier_on_full_but

Diagnostic only. Model inference uses waveform only; feature columns are used here only as recovery targets.

## Test Metrics

| n | acc | confusion_3x3 | good_recall | good_precision | good_f1 | medium_recall | medium_precision | medium_f1 | bad_recall | bad_precision | bad_f1 | macro_f1 | good_to_medium | medium_to_good | bad_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8477 | 0.827415 | [[3179, 461, 0], [591, 3835, 0], [71, 340, 0]] | 0.873352 | 0.827649 | 0.849886 | 0.866471 | 0.827222 | 0.846392 | 0 |  | 0 | 0.565426 | 461 | 591 | 71 | 340 |

## Key Feature Recovery

| feature | corr_norm | mae_norm | target_median_norm | pred_median_norm | is_key_feature |
| --- | --- | --- | --- | --- | --- |
| detector_agreement | 0.47076 | 0.482041 | -0.829789 | -0.563153 | True |
| qrs_visibility | 0.474932 | 0.725985 | -1.40653 | -0.613779 | True |
| hjorth_complexity | 0.633107 | 4.57464 | 1.75584 | 1.21648 | True |
| baseline_step | 0.741841 | 1.0974 | 1.85592 | 1.33043 | True |
| flatline_ratio | 0.747335 | 0.490636 | 0.563254 | 0.476714 | True |
| band_30_45 | 0.749813 | 0.33842 | -0.531644 | -0.628635 | True |
| band_15_30 | 0.82756 | 0.226698 | -0.824099 | -0.702066 | True |
| wavelet_e4 | 0.832492 | 0.0842105 | -0.558933 | -0.609591 | True |
| qrs_band_ratio | 0.835181 | 0.918249 | -1.5287 | -0.957096 | True |
| sqi_basSQI | 0.844348 | 3.90867 | -3.92142 | -1.37644 | True |
| non_qrs_diff_p95 | 0.89251 | 0.147518 | -0.859586 | -0.854102 | True |
| diff_abs_p95 | 0.924762 | 0.400533 | -1.77805 | -1.41534 | True |
| wavelet_e0 | 0.939281 | 0.306307 | 1.63007 | 1.28675 | True |

## Worst Overall Feature Recovery

| feature | corr_norm | mae_norm | target_median_norm | pred_median_norm | is_key_feature |
| --- | --- | --- | --- | --- | --- |
| contact_loss_win_ratio | -0.154602 | 28.5161 | -0.00746022 | -0.0151844 | False |
| mean_abs | 0.263671 | 0.921734 | -0.260785 | -0.13535 | False |
| template_corr | 0.34982 | 0.697429 | -0.188649 | 0.249335 | False |
| sqi_bSQI | 0.374872 | 0.326629 | 0.362983 | 0.466508 | False |
| pc3 | 0.392741 | 0.754789 | 0.109003 | 0.30723 | False |
| knn_label_purity | 0.445709 | 2.44974 | -0.164693 | -0.0539147 | False |
| detector_agreement | 0.47076 | 0.482041 | -0.829789 | -0.563153 | True |
| qrs_visibility | 0.474932 | 0.725985 | -1.40653 | -0.613779 | True |
| pca_margin | 0.487933 | 0.292591 | -0.693117 | -0.524425 | False |
| sqi_fSQI | 0.513385 | 0.933819 | 0.657638 | 0.723244 | False |
| boundary_confidence | 0.584876 | 1.16704 | -1.12377 | -0.218137 | False |
| diff_zero_crossing_rate | 0.61625 | 0.457187 | -0.322836 | -0.627123 | False |
| region_confidence | 0.617963 | 1.58084 | -2.5775 | -0.336028 | False |
| hjorth_complexity | 0.633107 | 4.57464 | 1.75584 | 1.21648 | True |
| fatal_or_score | 0.64587 | 1.29017 | 0.363413 | -0.357584 | False |
| sqi_pSQI | 0.670633 | 0.219992 | 0.563614 | 0.60354 | False |
| baseline_step | 0.741841 | 1.0974 | 1.85592 | 1.33043 | True |
| flatline_ratio | 0.747335 | 0.490636 | 0.563254 | 0.476714 | True |
| band_30_45 | 0.749813 | 0.33842 | -0.531644 | -0.628635 | True |
| pc4 | 0.818804 | 0.775172 | 0.28017 | -0.458928 | False |
| higuchi_fd_proxy | 0.8231 | 0.327117 | -0.55051 | -0.629012 | False |
| band_15_30 | 0.82756 | 0.226698 | -0.824099 | -0.702066 | True |
| low_amp_ratio | 0.82928 | 0.375936 | -0.651122 | -0.752626 | False |
| wavelet_e4 | 0.832492 | 0.0842105 | -0.558933 | -0.609591 | True |
| qrs_band_ratio | 0.835181 | 0.918249 | -1.5287 | -0.957096 | True |
| sqi_basSQI | 0.844348 | 3.90867 | -3.92142 | -1.37644 | True |
| non_qrs_rms_ratio | 0.868342 | 0.306947 | -0.173037 | -0.120264 | False |
| qrs_prom_p90 | 0.87103 | 0.810139 | 0.432252 | -0.0387615 | False |
| rms | 0.879519 | 0.345777 | -0.662339 | -0.481051 | False |
| pc1 | 0.879664 | 0.18003 | -0.605171 | -0.487172 | False |

Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\clean_margin2_drop_outlier_on_full_but_predictions.csv`
Feature recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\clean_margin2_drop_outlier_on_full_but_feature_recovery.csv`
Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\clean_margin2_drop_outlier_on_full_but_metrics.csv`
