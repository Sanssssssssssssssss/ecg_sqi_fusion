# Cross-Domain Hard Feature Learnability

Diagnostic only. A source-trained linear ridge probe is fitted on waveform-derived primitive banks, then evaluated in-source and cross-domain.

- Bank mode: `compact`
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
| clean_train | ptb_test | detector_agreement | 0.02304 | 0.17 |
| clean_train | ptb_test | qrs_visibility | 0.09943 | 0.2264 |
| clean_train | ptb_test | sqi_basSQI | 0.1101 | 0.1652 |
| clean_train | ptb_test | baseline_step | 0.1356 | 0.5362 |
| clean_train | ptb_test | template_corr | 0.2712 | 0.1889 |
| clean_train | ptb_test | qrs_band_ratio | 0.3555 | 0.1718 |
| clean_train | ptb_test | band_30_45 | 0.4136 | 0.0215 |
| clean_train | ptb_test | flatline_ratio | 0.5099 | 0.107 |
| clean_train | ptb_test | sqi_pSQI | 0.5391 | 0.118 |
| clean_train | ptb_test | non_qrs_diff_p95 | 0.5936 | 0.06151 |
| clean_train | ptb_test | sqi_bSQI | 0.6176 | 0.1593 |
| ptb_train | clean_test | qrs_band_ratio | 0.04768 | 0.0761 |
| ptb_train | clean_test | detector_agreement | 0.3293 | 0.1166 |
| ptb_train | clean_test | qrs_visibility | 0.3314 | 0.1534 |
| ptb_train | clean_test | template_corr | 0.3563 | 0.1694 |
| ptb_train | clean_test | baseline_step | 0.5547 | 0.2426 |
| ptb_train | clean_test | sqi_bSQI | 0.6692 | 0.09878 |
| ptb_train | clean_test | sqi_basSQI | 0.6994 | 0.05246 |
| ptb_train | clean_test | sqi_pSQI | 0.7267 | 0.05684 |
| ptb_train | clean_test | flatline_ratio | 0.789 | 0.09036 |
| ptb_train | clean_test | non_qrs_diff_p95 | 0.7987 | 0.04023 |
| ptb_train | clean_test | band_30_45 | 0.9029 | 0.02539 |
| ptb_train | ptb_test | qrs_visibility | 0.1441 | 0.1727 |
| ptb_train | ptb_test | sqi_basSQI | 0.158 | 0.1534 |
| ptb_train | ptb_test | detector_agreement | 0.2276 | 0.1251 |
| ptb_train | ptb_test | baseline_step | 0.2918 | 0.4451 |
| ptb_train | ptb_test | qrs_band_ratio | 0.4255 | 0.1557 |
| ptb_train | ptb_test | template_corr | 0.4584 | 0.1475 |
| ptb_train | ptb_test | flatline_ratio | 0.4995 | 0.1023 |
| ptb_train | ptb_test | sqi_bSQI | 0.6254 | 0.1539 |
| ptb_train | ptb_test | band_30_45 | 0.6895 | 0.01483 |
| ptb_train | ptb_test | non_qrs_diff_p95 | 0.7204 | 0.05332 |
| ptb_train | ptb_test | sqi_pSQI | 0.7466 | 0.07382 |

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
| clean_train | ptb_test | sqi_basSQI | -0.1777 | 0.1455 |
| clean_train | ptb_test | band_30_45 | -0.1332 | 0.02571 |
| clean_train | ptb_test | sqi_pSQI | -0.09065 | 0.1516 |
| clean_train | ptb_test | flatline_ratio | -0.07711 | 0.101 |
| clean_train | ptb_test | qrs_band_ratio | -0.06865 | 0.1645 |
| clean_train | ptb_test | non_qrs_diff_p95 | -0.03793 | 0.07248 |
| clean_train | ptb_test | qrs_visibility | -0.02767 | 0.2051 |
| clean_train | ptb_test | baseline_step | -0.02602 | 0.4825 |
| clean_train | ptb_test | detector_agreement | -0.01887 | 0.173 |
| clean_train | ptb_test | template_corr | -0.01317 | 0.1881 |
| clean_train | ptb_test | sqi_bSQI | 0.008501 | 0.194 |
| ptb_train | clean_test | band_30_45 | -0.6512 | 0.07305 |
| ptb_train | clean_test | sqi_pSQI | -0.5914 | 0.1158 |
| ptb_train | clean_test | template_corr | -0.3908 | 0.123 |
| ptb_train | clean_test | detector_agreement | -0.1778 | 0.1192 |
| ptb_train | clean_test | flatline_ratio | -0.1377 | 0.06636 |
| ptb_train | clean_test | qrs_band_ratio | -0.03546 | 0.1918 |
| ptb_train | clean_test | sqi_basSQI | 0.02216 | 0.04248 |
| ptb_train | clean_test | sqi_bSQI | 0.03482 | 0.2148 |
| ptb_train | clean_test | baseline_step | 0.08293 | 0.1944 |
| ptb_train | clean_test | non_qrs_diff_p95 | 0.112 | 0.04188 |
| ptb_train | clean_test | qrs_visibility | 0.1543 | 0.143 |
| ptb_train | ptb_test | non_qrs_diff_p95 | -0.1194 | 0.06134 |
| ptb_train | ptb_test | qrs_visibility | -0.06299 | 0.1477 |
| ptb_train | ptb_test | qrs_band_ratio | -0.05413 | 0.1422 |
| ptb_train | ptb_test | band_30_45 | -0.03833 | 0.01708 |
| ptb_train | ptb_test | baseline_step | -0.03301 | 0.3853 |
| ptb_train | ptb_test | sqi_bSQI | -0.02693 | 0.1642 |
| ptb_train | ptb_test | sqi_basSQI | -0.01641 | 0.129 |
| ptb_train | ptb_test | flatline_ratio | -0.01372 | 0.1003 |
| ptb_train | ptb_test | template_corr | 0.008178 | 0.1405 |
| ptb_train | ptb_test | sqi_pSQI | 0.0244 | 0.094 |
| ptb_train | ptb_test | detector_agreement | 0.03415 | 0.1162 |

CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\cross_domain_hard_feature_learnability\hard_feature_learnability_compact.csv`