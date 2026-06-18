# BUT Direct Transformer Recovery: aug_convtx_balanced_focal_trainval_badcal_t005

Diagnostic only. Model inference uses waveform only; feature columns are used here only as recovery targets.

## Test Metrics

| n | acc | confusion_3x3 | good_recall | good_precision | good_f1 | medium_recall | medium_precision | medium_f1 | bad_recall | bad_precision | bad_f1 | macro_f1 | good_to_medium | medium_to_good | bad_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8477 | 0.808659 | [[3506, 128, 6], [1100, 3049, 277], [20, 91, 300]] | 0.963187 | 0.75789 | 0.848294 | 0.688884 | 0.932987 | 0.792566 | 0.729927 | 0.51458 | 0.603622 | 0.748161 | 128 | 1100 | 20 | 91 |

## Key Feature Recovery

| feature | corr_norm | mae_norm | target_median_norm | pred_median_norm | is_key_feature |
| --- | --- | --- | --- | --- | --- |
| detector_agreement | 0.547042 | 0.445194 | -0.744255 | -0.805146 | True |
| qrs_visibility | 0.725149 | 0.349706 | -1.33756 | -0.881863 | True |
| baseline_step | 0.821738 | 0.621727 | 1.28322 | 1.41845 | True |
| band_15_30 | 0.854868 | 0.138073 | -0.782514 | -0.727599 | True |
| band_30_45 | 0.867312 | 0.264293 | -0.492789 | -0.546383 | True |
| flatline_ratio | 0.870761 | 0.525521 | 0.550125 | 1.47961 | True |
| hjorth_complexity | 0.877224 | 0.757458 | 0.418821 | 0.858318 | True |
| qrs_band_ratio | 0.898075 | 0.377115 | -1.28471 | -1.14587 | True |
| sqi_basSQI | 0.919925 | 0.693423 | -1.81822 | -1.52033 | True |
| diff_abs_p95 | 0.949437 | 0.30125 | -1.68979 | -1.40592 | True |
| non_qrs_diff_p95 | 0.951749 | 0.121856 | -0.834841 | -0.793331 | True |
| wavelet_e0 | 0.957832 | 0.224694 | 1.34313 | 1.17917 | True |
| wavelet_e4 | 0.973477 | 0.0464668 | -0.529868 | -0.482143 | True |

## Worst Overall Feature Recovery

| feature | corr_norm | mae_norm | target_median_norm | pred_median_norm | is_key_feature |
| --- | --- | --- | --- | --- | --- |
| contact_loss_win_ratio | 0.285118 | 1.61839 | -0.0317067 | -0.0225113 | False |
| knn_label_purity | 0.413364 | 1.23721 | 0.219747 | 0.587886 | False |
| detector_agreement | 0.547042 | 0.445194 | -0.744255 | -0.805146 | True |
| pca_margin | 0.56742 | 0.298823 | -0.514573 | -0.344586 | False |
| fatal_or_score | 0.592278 | 1.29105 | 0.347695 | -0.0365197 | False |
| template_corr | 0.648287 | 0.504694 | -0.181992 | 0.249275 | False |
| sqi_bSQI | 0.663832 | 0.287558 | 0.34496 | 0.245459 | False |
| sqi_pSQI | 0.675502 | 0.227683 | 0.542457 | 0.500414 | False |
| mean_abs | 0.707209 | 0.635014 | -0.289609 | -0.621582 | False |
| qrs_visibility | 0.725149 | 0.349706 | -1.33756 | -0.881863 | True |
| sqi_fSQI | 0.725484 | 0.930442 | 0.436099 | 1.61477 | False |
| pc3 | 0.726873 | 0.558566 | 0.0180814 | 0.124184 | False |
| boundary_confidence | 0.776551 | 0.723769 | -0.605117 | -0.111744 | False |
| diff_zero_crossing_rate | 0.799469 | 0.33782 | -0.2818 | -0.433303 | False |
| region_confidence | 0.799982 | 0.672107 | -1.4659 | -0.342053 | False |
| baseline_step | 0.821738 | 0.621727 | 1.28322 | 1.41845 | True |
| band_15_30 | 0.854868 | 0.138073 | -0.782514 | -0.727599 | True |
| low_amp_ratio | 0.86247 | 0.385505 | -0.588269 | -0.331648 | False |
| band_30_45 | 0.867312 | 0.264293 | -0.492789 | -0.546383 | True |
| sample_entropy_proxy | 0.869518 | 0.297843 | -0.464582 | -0.485435 | False |
| flatline_ratio | 0.870761 | 0.525521 | 0.550125 | 1.47961 | True |
| hjorth_complexity | 0.877224 | 0.757458 | 0.418821 | 0.858318 | True |
| qrs_band_ratio | 0.898075 | 0.377115 | -1.28471 | -1.14587 | True |
| pc4 | 0.903276 | 0.520261 | 0.365091 | -0.0600796 | False |
| wavelet_e1 | 0.907243 | 0.24558 | -0.491898 | -0.451606 | False |
| wavelet_e2 | 0.912545 | 0.240879 | -1.24029 | -1.02167 | False |
| sqi_basSQI | 0.919925 | 0.693423 | -1.81822 | -1.52033 | True |
| hjorth_activity | 0.920917 | 0.207042 | -0.710265 | -0.686515 | False |
| wavelet_e3 | 0.922866 | 0.136146 | -0.871159 | -0.73671 | False |
| std | 0.927002 | 0.205373 | -0.607722 | -0.568866 | False |

Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\aug_convtx_balanced_focal_trainval_badcal_t005_predictions.csv`
Feature recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\aug_convtx_balanced_focal_trainval_badcal_t005_feature_recovery.csv`
Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_direct_transformer_recovery\aug_convtx_balanced_focal_trainval_badcal_t005_metrics.csv`
