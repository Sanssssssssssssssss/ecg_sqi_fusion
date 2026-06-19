# Cross-Domain Hard Feature Learnability

Diagnostic only. A source-trained linear ridge probe is fitted on waveform-derived primitive banks, then evaluated in-source and cross-domain.

- Bank mode: `compact`
- PTB target source: `waveform_primitives`
- Clean target source: `waveform_primitives`
- Ridge alpha: `25`
- Targets: `qrs_visibility, detector_agreement, baseline_step, sqi_basSQI, flatline_ratio, qrs_band_ratio, template_corr, non_qrs_diff_p95, band_30_45, sqi_bSQI, sqi_pSQI`

## All-Class Recovery

| train_domain | eval_domain | feature | corr | mae |
| --- | --- | --- | --- | --- |
| clean_train | clean_test | sqi_pSQI | 0.3112 | 0.006511 |
| clean_train | clean_test | detector_agreement | 0.8175 | 0.1353 |
| clean_train | clean_test | band_30_45 | 0.8487 | 0.008831 |
| clean_train | clean_test | template_corr | 0.9028 | 0.04975 |
| clean_train | clean_test | qrs_visibility | 0.9076 | 0.0652 |
| clean_train | clean_test | qrs_band_ratio | 0.9527 | 0.4993 |
| clean_train | clean_test | sqi_basSQI | 0.9687 | 0.01932 |
| clean_train | clean_test | sqi_bSQI | 0.9743 | 0.03676 |
| clean_train | clean_test | flatline_ratio | 0.9765 | 0.01297 |
| clean_train | clean_test | baseline_step | 0.9822 | 0.02097 |
| clean_train | clean_test | non_qrs_diff_p95 | 1 | 0.00833 |
| clean_train | ptb_test | detector_agreement | 0.326 | 0.2656 |
| clean_train | ptb_test | sqi_pSQI | 0.6157 | 0.01064 |
| clean_train | ptb_test | band_30_45 | 0.7661 | 0.01256 |
| clean_train | ptb_test | template_corr | 0.7816 | 0.1086 |
| clean_train | ptb_test | qrs_band_ratio | 0.9232 | 0.8746 |
| clean_train | ptb_test | qrs_visibility | 0.9351 | 0.09882 |
| clean_train | ptb_test | flatline_ratio | 0.9413 | 0.01481 |
| clean_train | ptb_test | sqi_bSQI | 0.9473 | 0.06085 |
| clean_train | ptb_test | sqi_basSQI | 0.9824 | 0.02229 |
| clean_train | ptb_test | baseline_step | 0.9942 | 0.0127 |
| clean_train | ptb_test | non_qrs_diff_p95 | 1 | 0.006644 |
| ptb_train | clean_test | sqi_pSQI | 0.04861 | 0.008096 |
| ptb_train | clean_test | template_corr | 0.8116 | 0.08221 |
| ptb_train | clean_test | detector_agreement | 0.8278 | 0.1303 |
| ptb_train | clean_test | qrs_visibility | 0.9026 | 0.0634 |
| ptb_train | clean_test | band_30_45 | 0.9026 | 0.01426 |
| ptb_train | clean_test | qrs_band_ratio | 0.9305 | 0.5806 |
| ptb_train | clean_test | sqi_bSQI | 0.9614 | 0.05175 |
| ptb_train | clean_test | sqi_basSQI | 0.9636 | 0.0362 |
| ptb_train | clean_test | flatline_ratio | 0.9718 | 0.01112 |
| ptb_train | clean_test | baseline_step | 0.9818 | 0.02035 |
| ptb_train | clean_test | non_qrs_diff_p95 | 0.9999 | 0.01472 |
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
| clean_train | clean_test | template_corr | -0.09757 | 0.06236 |
| clean_train | clean_test | detector_agreement | -0.08929 | 0.1319 |
| clean_train | clean_test | sqi_pSQI | 0.2433 | 0.01983 |
| clean_train | clean_test | flatline_ratio | 0.284 | 0.01073 |
| clean_train | clean_test | qrs_visibility | 0.3639 | 0.05482 |
| clean_train | clean_test | band_30_45 | 0.385 | 0.04292 |
| clean_train | clean_test | baseline_step | 0.4139 | 0.01735 |
| clean_train | clean_test | sqi_bSQI | 0.4795 | 0.02974 |
| clean_train | clean_test | qrs_band_ratio | 0.6216 | 0.4709 |
| clean_train | clean_test | sqi_basSQI | 0.6366 | 0.02179 |
| clean_train | clean_test | non_qrs_diff_p95 | 0.9999 | 0.009204 |
| clean_train | ptb_test | detector_agreement | -0.05611 | 0.2823 |
| clean_train | ptb_test | template_corr | 0.1592 | 0.1151 |
| clean_train | ptb_test | sqi_pSQI | 0.1642 | 0.01802 |
| clean_train | ptb_test | flatline_ratio | 0.2054 | 0.01468 |
| clean_train | ptb_test | qrs_band_ratio | 0.2143 | 0.778 |
| clean_train | ptb_test | baseline_step | 0.2853 | 0.01384 |
| clean_train | ptb_test | sqi_bSQI | 0.3294 | 0.06208 |
| clean_train | ptb_test | qrs_visibility | 0.4785 | 0.08569 |
| clean_train | ptb_test | sqi_basSQI | 0.505 | 0.02515 |
| clean_train | ptb_test | band_30_45 | 0.7493 | 0.02132 |
| clean_train | ptb_test | non_qrs_diff_p95 | 0.9999 | 0.007016 |
| ptb_train | clean_test | template_corr | -0.1146 | 0.06962 |
| ptb_train | clean_test | detector_agreement | -0.008954 | 0.2464 |
| ptb_train | clean_test | flatline_ratio | 0.1711 | 0.01056 |
| ptb_train | clean_test | qrs_visibility | 0.2086 | 0.0627 |
| ptb_train | clean_test | sqi_bSQI | 0.2561 | 0.05505 |
| ptb_train | clean_test | sqi_pSQI | 0.2888 | 0.00914 |
| ptb_train | clean_test | sqi_basSQI | 0.2919 | 0.04187 |
| ptb_train | clean_test | baseline_step | 0.4174 | 0.01755 |
| ptb_train | clean_test | band_30_45 | 0.4716 | 0.0164 |
| ptb_train | clean_test | qrs_band_ratio | 0.5226 | 0.8167 |
| ptb_train | clean_test | non_qrs_diff_p95 | 0.9996 | 0.01295 |
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

CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\cross_domain_hard_feature_learnability\hard_feature_learnability_compact_ptb-wf_clean-wf.csv`