# BUT Direct Transformer Recovery: clean_margin5_drop_outlier_on_full_but

Diagnostic only. Model inference uses waveform only; feature columns are used here only as recovery targets.

## Test Metrics

| n | acc | confusion_3x3 | good_recall | good_precision | good_f1 | medium_recall | medium_precision | medium_f1 | bad_recall | bad_precision | bad_f1 | macro_f1 | good_to_medium | medium_to_good | bad_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8477 | 0.775392 | [[3416, 224, 0], [1388, 3038, 0], [146, 146, 119]] | 0.938462 | 0.690101 | 0.795343 | 0.686399 | 0.891432 | 0.775594 | 0.289538 | 1 | 0.449057 | 0.673331 | 224 | 1388 | 146 | 146 |

## Key Feature Recovery

| feature | corr_norm | mae_norm | target_median_norm | pred_median_norm | is_key_feature |
| --- | --- | --- | --- | --- | --- |
| detector_agreement | 0.476161 | 0.46327 | -0.832604 | -0.772117 | True |
| qrs_visibility | 0.489372 | 0.726035 | -1.40314 | -0.478089 | True |
| hjorth_complexity | 0.646015 | 4.51847 | 1.76147 | 1.20799 | True |
| baseline_step | 0.715846 | 1.23135 | 1.86483 | 1.10478 | True |
| flatline_ratio | 0.727419 | 0.580475 | 0.562282 | 0.914864 | True |
| qrs_band_ratio | 0.752473 | 0.847324 | -1.5308 | -1.20025 | True |
| band_15_30 | 0.759733 | 0.251076 | -0.826566 | -0.699261 | True |
| sqi_basSQI | 0.829061 | 3.68903 | -3.93369 | -1.84773 | True |
| band_30_45 | 0.844125 | 0.331623 | -0.535055 | -0.664581 | True |
| wavelet_e4 | 0.888071 | 0.0877025 | -0.563576 | -0.600891 | True |
| non_qrs_diff_p95 | 0.929424 | 0.130167 | -0.862213 | -0.866995 | True |
| diff_abs_p95 | 0.932131 | 0.307887 | -1.77529 | -1.79563 | True |
| wavelet_e0 | 0.936461 | 0.288067 | 1.63171 | 1.45581 | True |

## Worst Overall Feature Recovery

| feature | corr_norm | mae_norm | target_median_norm | pred_median_norm | is_key_feature |
| --- | --- | --- | --- | --- | --- |
| contact_loss_win_ratio | -0.000595463 | 28.3201 | -0.00751436 | -0.0232814 | False |
| knn_label_purity | 0.215632 | 2.51326 | -0.172659 | 0.107945 | False |
| pc3 | 0.337176 | 0.765843 | 0.115512 | -0.0408053 | False |
| template_corr | 0.337539 | 0.762329 | -0.184287 | 0.427538 | False |
| mean_abs | 0.406166 | 0.824576 | -0.256911 | -0.214177 | False |
| sqi_bSQI | 0.43082 | 0.291597 | 0.368897 | 0.414169 | False |
| detector_agreement | 0.476161 | 0.46327 | -0.832604 | -0.772117 | True |
| pca_margin | 0.47986 | 0.328156 | -0.699985 | -0.535666 | False |
| qrs_visibility | 0.489372 | 0.726035 | -1.40314 | -0.478089 | True |
| sqi_fSQI | 0.518324 | 0.956921 | 0.656299 | 1.34682 | False |
| region_confidence | 0.55355 | 1.66729 | -2.58536 | -0.344198 | False |
| sqi_pSQI | 0.565884 | 0.239025 | 0.567255 | 0.550287 | False |
| boundary_confidence | 0.571614 | 1.12026 | -1.12953 | -0.363817 | False |
| fatal_or_score | 0.608551 | 1.31856 | 0.365016 | -0.611596 | False |
| hjorth_complexity | 0.646015 | 4.51847 | 1.76147 | 1.20799 | True |
| diff_zero_crossing_rate | 0.648229 | 0.420447 | -0.328846 | -0.594532 | False |
| baseline_step | 0.715846 | 1.23135 | 1.86483 | 1.10478 | True |
| flatline_ratio | 0.727419 | 0.580475 | 0.562282 | 0.914864 | True |
| qrs_band_ratio | 0.752473 | 0.847324 | -1.5308 | -1.20025 | True |
| band_15_30 | 0.759733 | 0.251076 | -0.826566 | -0.699261 | True |
| low_amp_ratio | 0.806703 | 0.459775 | -0.646989 | -0.503687 | False |
| pc4 | 0.812322 | 0.635624 | 0.27622 | -0.12457 | False |
| higuchi_fd_proxy | 0.818878 | 0.318357 | -0.554315 | -0.590183 | False |
| sqi_basSQI | 0.829061 | 3.68903 | -3.93369 | -1.84773 | True |
| non_qrs_rms_ratio | 0.832303 | 0.390855 | -0.177333 | -0.269316 | False |
| qrs_prom_p90 | 0.842768 | 0.795009 | 0.436834 | 0.116808 | False |
| band_30_45 | 0.844125 | 0.331623 | -0.535055 | -0.664581 | True |
| pc1 | 0.857751 | 0.1857 | -0.607612 | -0.533099 | False |
| amplitude_entropy | 0.868693 | 0.3899 | 0.205507 | 0.0967855 | False |
| sqi_sSQI | 0.871194 | 0.463549 | -0.125798 | -0.0650798 | False |

Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\clean_margin5_drop_outlier_on_full_but_predictions.csv`
Feature recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\clean_margin5_drop_outlier_on_full_but_feature_recovery.csv`
Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\clean_margin5_drop_outlier_on_full_but_metrics.csv`
