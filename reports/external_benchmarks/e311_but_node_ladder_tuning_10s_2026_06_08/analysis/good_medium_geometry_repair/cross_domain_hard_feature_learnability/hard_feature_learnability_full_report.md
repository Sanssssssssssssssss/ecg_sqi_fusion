# Cross-Domain Hard Feature Learnability

Diagnostic only. A source-trained linear ridge probe is fitted on waveform-derived primitive banks, then evaluated in-source and cross-domain.

- Bank mode: `full`
- Ridge alpha: `50`
- Targets: `qrs_visibility, detector_agreement, baseline_step, sqi_basSQI, flatline_ratio, qrs_band_ratio, template_corr, non_qrs_diff_p95, band_30_45, sqi_bSQI, sqi_pSQI`

## All-Class Recovery

| train_domain | eval_domain | feature | corr | mae |
| --- | --- | --- | --- | --- |
| clean_train | clean_test | detector_agreement | 0.5411 | 0.1809 |
| clean_train | clean_test | qrs_visibility | 0.6669 | 0.1399 |
| clean_train | clean_test | qrs_band_ratio | 0.7125 | 0.03588 |
| clean_train | clean_test | sqi_bSQI | 0.8265 | 0.0648 |
| clean_train | clean_test | template_corr | 0.882 | 0.1127 |
| clean_train | clean_test | baseline_step | 0.8922 | 0.1295 |
| clean_train | clean_test | sqi_pSQI | 0.9089 | 0.02396 |
| clean_train | clean_test | band_30_45 | 0.9226 | 0.01144 |
| clean_train | clean_test | sqi_basSQI | 0.9669 | 0.01836 |
| clean_train | clean_test | non_qrs_diff_p95 | 0.9747 | 0.01583 |
| clean_train | clean_test | flatline_ratio | 0.9957 | 0.01275 |
| clean_train | ptb_test | detector_agreement | -0.05776 | 0.1782 |
| clean_train | ptb_test | qrs_visibility | 0.09145 | 0.2193 |
| clean_train | ptb_test | sqi_basSQI | 0.113 | 0.1648 |
| clean_train | ptb_test | baseline_step | 0.1409 | 0.535 |
| clean_train | ptb_test | template_corr | 0.2752 | 0.1903 |
| clean_train | ptb_test | qrs_band_ratio | 0.349 | 0.1706 |
| clean_train | ptb_test | band_30_45 | 0.4295 | 0.02178 |
| clean_train | ptb_test | flatline_ratio | 0.5076 | 0.1063 |
| clean_train | ptb_test | sqi_pSQI | 0.5506 | 0.1182 |
| clean_train | ptb_test | non_qrs_diff_p95 | 0.6018 | 0.06095 |
| clean_train | ptb_test | sqi_bSQI | 0.6177 | 0.16 |
| ptb_train | clean_test | qrs_band_ratio | -0.008258 | 0.08749 |
| ptb_train | clean_test | detector_agreement | 0.2601 | 0.1197 |
| ptb_train | clean_test | qrs_visibility | 0.2809 | 0.1521 |
| ptb_train | clean_test | template_corr | 0.2964 | 0.1703 |
| ptb_train | clean_test | baseline_step | 0.394 | 0.2683 |
| ptb_train | clean_test | sqi_basSQI | 0.4458 | 0.0644 |
| ptb_train | clean_test | sqi_bSQI | 0.6489 | 0.1091 |
| ptb_train | clean_test | non_qrs_diff_p95 | 0.6909 | 0.04832 |
| ptb_train | clean_test | sqi_pSQI | 0.6925 | 0.06262 |
| ptb_train | clean_test | flatline_ratio | 0.7082 | 0.09388 |
| ptb_train | clean_test | band_30_45 | 0.8593 | 0.02566 |
| ptb_train | ptb_test | sqi_basSQI | 0.05726 | 0.1594 |
| ptb_train | ptb_test | qrs_visibility | 0.07218 | 0.1797 |
| ptb_train | ptb_test | detector_agreement | 0.1563 | 0.1306 |
| ptb_train | ptb_test | baseline_step | 0.2367 | 0.4567 |
| ptb_train | ptb_test | qrs_band_ratio | 0.3808 | 0.1619 |
| ptb_train | ptb_test | template_corr | 0.3853 | 0.1581 |
| ptb_train | ptb_test | flatline_ratio | 0.4579 | 0.1054 |
| ptb_train | ptb_test | sqi_bSQI | 0.6017 | 0.1677 |
| ptb_train | ptb_test | band_30_45 | 0.6844 | 0.01508 |
| ptb_train | ptb_test | non_qrs_diff_p95 | 0.7105 | 0.05463 |
| ptb_train | ptb_test | sqi_pSQI | 0.746 | 0.07513 |

## Minimum Per-Class Correlation

| train_domain | eval_domain | feature | min_class_corr | mean_class_mae |
| --- | --- | --- | --- | --- |
| clean_train | clean_test | detector_agreement | -0.133 | 0.1706 |
| clean_train | clean_test | sqi_bSQI | 0.0258 | 0.1391 |
| clean_train | clean_test | template_corr | 0.04228 | 0.09819 |
| clean_train | clean_test | qrs_visibility | 0.1835 | 0.1635 |
| clean_train | clean_test | sqi_pSQI | 0.3547 | 0.08323 |
| clean_train | clean_test | band_30_45 | 0.3664 | 0.06043 |
| clean_train | clean_test | flatline_ratio | 0.4305 | 0.01075 |
| clean_train | clean_test | non_qrs_diff_p95 | 0.4675 | 0.01766 |
| clean_train | clean_test | qrs_band_ratio | 0.5256 | 0.126 |
| clean_train | clean_test | baseline_step | 0.6149 | 0.1101 |
| clean_train | clean_test | sqi_basSQI | 0.9008 | 0.01751 |
| clean_train | ptb_test | sqi_basSQI | -0.1867 | 0.1449 |
| clean_train | ptb_test | band_30_45 | -0.1395 | 0.02558 |
| clean_train | ptb_test | detector_agreement | -0.1041 | 0.1742 |
| clean_train | ptb_test | flatline_ratio | -0.1002 | 0.1009 |
| clean_train | ptb_test | sqi_pSQI | -0.09358 | 0.1512 |
| clean_train | ptb_test | qrs_band_ratio | -0.07738 | 0.1621 |
| clean_train | ptb_test | qrs_visibility | -0.05962 | 0.2031 |
| clean_train | ptb_test | non_qrs_diff_p95 | -0.049 | 0.07055 |
| clean_train | ptb_test | baseline_step | -0.02899 | 0.4778 |
| clean_train | ptb_test | template_corr | -0.02806 | 0.1865 |
| clean_train | ptb_test | sqi_bSQI | 0.0241 | 0.1931 |
| ptb_train | clean_test | band_30_45 | -0.3874 | 0.07292 |
| ptb_train | clean_test | sqi_pSQI | -0.3132 | 0.1185 |
| ptb_train | clean_test | template_corr | -0.2054 | 0.1213 |
| ptb_train | clean_test | baseline_step | -0.1233 | 0.2249 |
| ptb_train | clean_test | non_qrs_diff_p95 | -0.06872 | 0.04694 |
| ptb_train | clean_test | flatline_ratio | -0.06817 | 0.0699 |
| ptb_train | clean_test | detector_agreement | -0.04635 | 0.1165 |
| ptb_train | clean_test | sqi_basSQI | -0.02457 | 0.05411 |
| ptb_train | clean_test | qrs_band_ratio | 0.03197 | 0.2024 |
| ptb_train | clean_test | qrs_visibility | 0.05918 | 0.1407 |
| ptb_train | clean_test | sqi_bSQI | 0.06547 | 0.2227 |
| ptb_train | ptb_test | qrs_visibility | -0.1668 | 0.1527 |
| ptb_train | ptb_test | template_corr | -0.1659 | 0.1485 |
| ptb_train | ptb_test | baseline_step | -0.156 | 0.394 |
| ptb_train | ptb_test | qrs_band_ratio | -0.155 | 0.1484 |
| ptb_train | ptb_test | sqi_basSQI | -0.1351 | 0.1349 |
| ptb_train | ptb_test | flatline_ratio | -0.06499 | 0.1037 |
| ptb_train | ptb_test | non_qrs_diff_p95 | -0.06104 | 0.06296 |
| ptb_train | ptb_test | detector_agreement | -0.03406 | 0.121 |
| ptb_train | ptb_test | sqi_bSQI | -0.0185 | 0.1759 |
| ptb_train | ptb_test | band_30_45 | 0.02007 | 0.0174 |
| ptb_train | ptb_test | sqi_pSQI | 0.08984 | 0.09645 |

CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\cross_domain_hard_feature_learnability\hard_feature_learnability_full.csv`