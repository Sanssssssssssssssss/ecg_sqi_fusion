# Cross-Domain Hard Feature Learnability

Diagnostic only. A source-trained linear ridge probe is fitted on waveform-derived primitive banks, then evaluated in-source and cross-domain.

- Bank mode: `full`
- PTB target source: `waveform_primitives`
- Ridge alpha: `25`
- Targets: `qrs_visibility, detector_agreement, baseline_step, sqi_basSQI, flatline_ratio, qrs_band_ratio, template_corr, non_qrs_diff_p95, band_30_45, sqi_bSQI, sqi_pSQI`

## All-Class Recovery

| train_domain | eval_domain | feature | corr | mae |
| --- | --- | --- | --- | --- |
| clean_train | clean_test | detector_agreement | 0.5412 | 0.1865 |
| clean_train | clean_test | qrs_visibility | 0.6716 | 0.1448 |
| clean_train | clean_test | qrs_band_ratio | 0.7195 | 0.03516 |
| clean_train | clean_test | sqi_bSQI | 0.8271 | 0.06557 |
| clean_train | clean_test | template_corr | 0.8809 | 0.1199 |
| clean_train | clean_test | baseline_step | 0.8919 | 0.1299 |
| clean_train | clean_test | sqi_pSQI | 0.9118 | 0.02409 |
| clean_train | clean_test | band_30_45 | 0.9242 | 0.01127 |
| clean_train | clean_test | sqi_basSQI | 0.9644 | 0.01816 |
| clean_train | clean_test | non_qrs_diff_p95 | 0.9763 | 0.01598 |
| clean_train | clean_test | flatline_ratio | 0.9962 | 0.01207 |
| clean_train | ptb_test | sqi_bSQI | -0.5254 | 0.3568 |
| clean_train | ptb_test | baseline_step | -0.398 | 0.4311 |
| clean_train | ptb_test | sqi_basSQI | -0.3427 | 0.3204 |
| clean_train | ptb_test | non_qrs_diff_p95 | -0.06652 | 2.026 |
| clean_train | ptb_test | detector_agreement | -0.005314 | 0.3109 |
| clean_train | ptb_test | qrs_band_ratio | 0.05569 | 6.379 |
| clean_train | ptb_test | qrs_visibility | 0.3327 | 1.46 |
| clean_train | ptb_test | sqi_pSQI | 0.4118 | 0.4082 |
| clean_train | ptb_test | template_corr | 0.6691 | 0.2395 |
| clean_train | ptb_test | flatline_ratio | 0.7748 | 0.0896 |
| clean_train | ptb_test | band_30_45 | 0.8551 | 0.01209 |
| ptb_train | clean_test | detector_agreement | -0.4495 | 0.3763 |
| ptb_train | clean_test | baseline_step | -0.3681 | 0.4976 |
| ptb_train | clean_test | sqi_bSQI | -0.0658 | 0.3809 |
| ptb_train | clean_test | sqi_basSQI | -0.0607 | 0.3262 |
| ptb_train | clean_test | sqi_pSQI | 0.0276 | 0.4301 |
| ptb_train | clean_test | qrs_visibility | 0.3521 | 1.738 |
| ptb_train | clean_test | template_corr | 0.4622 | 0.2356 |
| ptb_train | clean_test | non_qrs_diff_p95 | 0.5108 | 1.661 |
| ptb_train | clean_test | qrs_band_ratio | 0.6157 | 7.924 |
| ptb_train | clean_test | flatline_ratio | 0.8545 | 0.08712 |
| ptb_train | clean_test | band_30_45 | 0.9262 | 0.01268 |
| ptb_train | ptb_test | sqi_pSQI | 0.5675 | 0.01292 |
| ptb_train | ptb_test | detector_agreement | 0.7153 | 0.1356 |
| ptb_train | ptb_test | template_corr | 0.8936 | 0.07794 |
| ptb_train | ptb_test | band_30_45 | 0.9416 | 0.007752 |
| ptb_train | ptb_test | qrs_visibility | 0.9566 | 0.08737 |
| ptb_train | ptb_test | flatline_ratio | 0.969 | 0.01108 |
| ptb_train | ptb_test | qrs_band_ratio | 0.9737 | 0.4615 |
| ptb_train | ptb_test | sqi_bSQI | 0.9809 | 0.03408 |
| ptb_train | ptb_test | sqi_basSQI | 0.993 | 0.01424 |
| ptb_train | ptb_test | baseline_step | 0.9975 | 0.008512 |
| ptb_train | ptb_test | non_qrs_diff_p95 | 1 | 0.009099 |

## Minimum Per-Class Correlation

| train_domain | eval_domain | feature | min_class_corr | mean_class_mae |
| --- | --- | --- | --- | --- |
| clean_train | clean_test | detector_agreement | -0.118 | 0.1754 |
| clean_train | clean_test | sqi_bSQI | 0.03766 | 0.1387 |
| clean_train | clean_test | template_corr | 0.0422 | 0.1009 |
| clean_train | clean_test | qrs_visibility | 0.2339 | 0.1575 |
| clean_train | clean_test | sqi_pSQI | 0.3619 | 0.0819 |
| clean_train | clean_test | band_30_45 | 0.3996 | 0.05982 |
| clean_train | clean_test | flatline_ratio | 0.4647 | 0.01098 |
| clean_train | clean_test | non_qrs_diff_p95 | 0.4767 | 0.0169 |
| clean_train | clean_test | qrs_band_ratio | 0.5508 | 0.1234 |
| clean_train | clean_test | baseline_step | 0.6062 | 0.1136 |
| clean_train | clean_test | sqi_basSQI | 0.8817 | 0.02215 |
| clean_train | ptb_test | sqi_basSQI | -0.7404 | 0.3245 |
| clean_train | ptb_test | baseline_step | -0.6919 | 0.326 |
| clean_train | ptb_test | qrs_band_ratio | -0.6606 | 5.723 |
| clean_train | ptb_test | sqi_bSQI | -0.5401 | 0.4822 |
| clean_train | ptb_test | sqi_pSQI | -0.3246 | 0.4757 |
| clean_train | ptb_test | qrs_visibility | -0.2206 | 1.263 |
| clean_train | ptb_test | detector_agreement | -0.2168 | 0.2961 |
| clean_train | ptb_test | non_qrs_diff_p95 | 0.01657 | 2.214 |
| clean_train | ptb_test | template_corr | 0.05285 | 0.1985 |
| clean_train | ptb_test | flatline_ratio | 0.3318 | 0.09571 |
| clean_train | ptb_test | band_30_45 | 0.8174 | 0.01885 |
| ptb_train | clean_test | sqi_basSQI | -0.4987 | 0.3042 |
| ptb_train | clean_test | baseline_step | -0.4276 | 0.3643 |
| ptb_train | clean_test | detector_agreement | -0.4098 | 0.3274 |
| ptb_train | clean_test | template_corr | -0.1982 | 0.2005 |
| ptb_train | clean_test | sqi_pSQI | -0.1653 | 0.5075 |
| ptb_train | clean_test | sqi_bSQI | -0.1586 | 0.4025 |
| ptb_train | clean_test | qrs_band_ratio | -0.1213 | 6.9 |
| ptb_train | clean_test | non_qrs_diff_p95 | -0.03792 | 1.693 |
| ptb_train | clean_test | flatline_ratio | 0.0971 | 0.08293 |
| ptb_train | clean_test | qrs_visibility | 0.122 | 1.518 |
| ptb_train | clean_test | band_30_45 | 0.6789 | 0.03744 |
| ptb_train | ptb_test | sqi_pSQI | 0.1248 | 0.01735 |
| ptb_train | ptb_test | detector_agreement | 0.2488 | 0.1289 |
| ptb_train | ptb_test | flatline_ratio | 0.3622 | 0.01067 |
| ptb_train | ptb_test | template_corr | 0.4052 | 0.08171 |
| ptb_train | ptb_test | qrs_band_ratio | 0.6259 | 0.3985 |
| ptb_train | ptb_test | qrs_visibility | 0.6349 | 0.0761 |
| ptb_train | ptb_test | band_30_45 | 0.7758 | 0.008126 |
| ptb_train | ptb_test | sqi_bSQI | 0.8166 | 0.0336 |
| ptb_train | ptb_test | baseline_step | 0.8474 | 0.007876 |
| ptb_train | ptb_test | sqi_basSQI | 0.9062 | 0.01444 |
| ptb_train | ptb_test | non_qrs_diff_p95 | 0.9999 | 0.008884 |

CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\cross_domain_hard_feature_learnability\hard_feature_learnability_full_waveform_primitives.csv`