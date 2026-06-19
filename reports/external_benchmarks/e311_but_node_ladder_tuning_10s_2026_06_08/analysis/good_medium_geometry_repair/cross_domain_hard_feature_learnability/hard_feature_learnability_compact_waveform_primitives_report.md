# Cross-Domain Hard Feature Learnability

Diagnostic only. A source-trained linear ridge probe is fitted on waveform-derived primitive banks, then evaluated in-source and cross-domain.

- Bank mode: `compact`
- PTB target source: `waveform_primitives`
- Ridge alpha: `25`
- Targets: `qrs_visibility, detector_agreement, baseline_step, sqi_basSQI, flatline_ratio, qrs_band_ratio, template_corr, non_qrs_diff_p95, band_30_45, sqi_bSQI, sqi_pSQI`

## All-Class Recovery

| train_domain | eval_domain | feature | corr | mae |
| --- | --- | --- | --- | --- |
| clean_train | clean_test | detector_agreement | 0.5641 | 0.1832 |
| clean_train | clean_test | qrs_band_ratio | 0.7005 | 0.03717 |
| clean_train | clean_test | qrs_visibility | 0.704 | 0.1479 |
| clean_train | clean_test | sqi_bSQI | 0.831 | 0.065 |
| clean_train | clean_test | template_corr | 0.8723 | 0.1153 |
| clean_train | clean_test | baseline_step | 0.8893 | 0.1327 |
| clean_train | clean_test | sqi_pSQI | 0.91 | 0.02446 |
| clean_train | clean_test | band_30_45 | 0.925 | 0.01148 |
| clean_train | clean_test | sqi_basSQI | 0.963 | 0.01898 |
| clean_train | clean_test | non_qrs_diff_p95 | 0.977 | 0.01574 |
| clean_train | clean_test | flatline_ratio | 0.9966 | 0.01295 |
| clean_train | ptb_test | sqi_bSQI | -0.5244 | 0.3466 |
| clean_train | ptb_test | baseline_step | -0.4043 | 0.4486 |
| clean_train | ptb_test | sqi_basSQI | -0.3442 | 0.3201 |
| clean_train | ptb_test | non_qrs_diff_p95 | -0.04353 | 2.03 |
| clean_train | ptb_test | detector_agreement | -0.04314 | 0.3203 |
| clean_train | ptb_test | qrs_band_ratio | 0.03686 | 6.375 |
| clean_train | ptb_test | qrs_visibility | 0.291 | 1.486 |
| clean_train | ptb_test | sqi_pSQI | 0.407 | 0.4101 |
| clean_train | ptb_test | template_corr | 0.6442 | 0.244 |
| clean_train | ptb_test | flatline_ratio | 0.7785 | 0.08761 |
| clean_train | ptb_test | band_30_45 | 0.8552 | 0.01219 |
| ptb_train | clean_test | detector_agreement | -0.4732 | 0.398 |
| ptb_train | clean_test | baseline_step | -0.3523 | 0.4938 |
| ptb_train | clean_test | sqi_bSQI | -0.0351 | 0.3916 |
| ptb_train | clean_test | sqi_basSQI | -0.02111 | 0.334 |
| ptb_train | clean_test | sqi_pSQI | 0.02774 | 0.4271 |
| ptb_train | clean_test | qrs_visibility | 0.3658 | 1.709 |
| ptb_train | clean_test | template_corr | 0.4481 | 0.2381 |
| ptb_train | clean_test | non_qrs_diff_p95 | 0.5112 | 1.661 |
| ptb_train | clean_test | qrs_band_ratio | 0.5814 | 7.976 |
| ptb_train | clean_test | flatline_ratio | 0.8481 | 0.08767 |
| ptb_train | clean_test | band_30_45 | 0.9028 | 0.01552 |
| ptb_train | ptb_test | sqi_pSQI | 0.5894 | 0.01192 |
| ptb_train | ptb_test | detector_agreement | 0.7532 | 0.1332 |
| ptb_train | ptb_test | template_corr | 0.902 | 0.07381 |
| ptb_train | ptb_test | band_30_45 | 0.954 | 0.007766 |
| ptb_train | ptb_test | qrs_visibility | 0.9566 | 0.08647 |
| ptb_train | ptb_test | flatline_ratio | 0.9723 | 0.01082 |
| ptb_train | ptb_test | qrs_band_ratio | 0.9756 | 0.433 |
| ptb_train | ptb_test | sqi_bSQI | 0.9782 | 0.03569 |
| ptb_train | ptb_test | sqi_basSQI | 0.9915 | 0.01518 |
| ptb_train | ptb_test | baseline_step | 0.9972 | 0.008974 |
| ptb_train | ptb_test | non_qrs_diff_p95 | 1 | 0.007599 |

## Minimum Per-Class Correlation

| train_domain | eval_domain | feature | min_class_corr | mean_class_mae |
| --- | --- | --- | --- | --- |
| clean_train | clean_test | detector_agreement | -0.05688 | 0.171 |
| clean_train | clean_test | template_corr | -0.0417 | 0.09896 |
| clean_train | clean_test | sqi_bSQI | 0.02832 | 0.1296 |
| clean_train | clean_test | qrs_visibility | 0.3629 | 0.1631 |
| clean_train | clean_test | flatline_ratio | 0.4254 | 0.01132 |
| clean_train | clean_test | sqi_pSQI | 0.4377 | 0.08287 |
| clean_train | clean_test | band_30_45 | 0.4558 | 0.05952 |
| clean_train | clean_test | non_qrs_diff_p95 | 0.4638 | 0.01783 |
| clean_train | clean_test | qrs_band_ratio | 0.6063 | 0.1262 |
| clean_train | clean_test | baseline_step | 0.63 | 0.1115 |
| clean_train | clean_test | sqi_basSQI | 0.8912 | 0.02364 |
| clean_train | ptb_test | sqi_basSQI | -0.7451 | 0.3242 |
| clean_train | ptb_test | baseline_step | -0.6914 | 0.3435 |
| clean_train | ptb_test | qrs_band_ratio | -0.6539 | 5.719 |
| clean_train | ptb_test | sqi_bSQI | -0.5422 | 0.4733 |
| clean_train | ptb_test | sqi_pSQI | -0.3447 | 0.4765 |
| clean_train | ptb_test | qrs_visibility | -0.3273 | 1.274 |
| clean_train | ptb_test | detector_agreement | -0.1924 | 0.3038 |
| clean_train | ptb_test | non_qrs_diff_p95 | -0.03437 | 2.218 |
| clean_train | ptb_test | template_corr | 0.05192 | 0.2005 |
| clean_train | ptb_test | flatline_ratio | 0.349 | 0.09444 |
| clean_train | ptb_test | band_30_45 | 0.7905 | 0.0191 |
| ptb_train | clean_test | sqi_basSQI | -0.4427 | 0.3152 |
| ptb_train | clean_test | detector_agreement | -0.4304 | 0.3451 |
| ptb_train | clean_test | baseline_step | -0.4269 | 0.36 |
| ptb_train | clean_test | sqi_pSQI | -0.3674 | 0.5023 |
| ptb_train | clean_test | qrs_band_ratio | -0.1667 | 7.054 |
| ptb_train | clean_test | template_corr | -0.1264 | 0.1953 |
| ptb_train | clean_test | sqi_bSQI | -0.1163 | 0.4051 |
| ptb_train | clean_test | non_qrs_diff_p95 | -0.03581 | 1.694 |
| ptb_train | clean_test | qrs_visibility | 0.06692 | 1.499 |
| ptb_train | clean_test | flatline_ratio | 0.1584 | 0.08319 |
| ptb_train | clean_test | band_30_45 | 0.484 | 0.03942 |
| ptb_train | ptb_test | sqi_pSQI | 0.223 | 0.01692 |
| ptb_train | ptb_test | detector_agreement | 0.3049 | 0.1248 |
| ptb_train | ptb_test | flatline_ratio | 0.317 | 0.01054 |
| ptb_train | ptb_test | template_corr | 0.4439 | 0.07865 |
| ptb_train | ptb_test | qrs_band_ratio | 0.619 | 0.3793 |
| ptb_train | ptb_test | qrs_visibility | 0.667 | 0.07526 |
| ptb_train | ptb_test | sqi_bSQI | 0.8108 | 0.03579 |
| ptb_train | ptb_test | baseline_step | 0.8599 | 0.008317 |
| ptb_train | ptb_test | sqi_basSQI | 0.8978 | 0.01539 |
| ptb_train | ptb_test | band_30_45 | 0.9091 | 0.007985 |
| ptb_train | ptb_test | non_qrs_diff_p95 | 1 | 0.007395 |

CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\cross_domain_hard_feature_learnability\hard_feature_learnability_compact_waveform_primitives.csv`