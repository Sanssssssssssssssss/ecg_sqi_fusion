# Original Bad-Stress Failure: sqiquery_intermittentbad_balanced

Post-hoc report-only analysis. Original BUT is not used for training or model selection.

- test bad outlier stress: n=292, raw recall=0.0137, badcal recall=0.1747
- test bad core/near-boundary: n=119, raw recall=1.0000, badcal recall=1.0000

## Top Feature Gaps

| feature | group_a | group_b | median_a | median_b | robust_effect |
| --- | --- | --- | --- | --- | --- |
| non_qrs_diff_p95 | test_bad_outlier_stress | test_bad_core_nearboundary | -0.7877 | 2.3194 | -2.2399 |
| knn_label_purity | test_bad_outlier_stress | test_bad_core_nearboundary | -2.7837 | 0.5967 | -2.2215 |
| boundary_confidence | test_bad_outlier_stress | test_bad_core_nearboundary | -2.1728 | -0.8546 | -2.2063 |
| pca_margin | test_bad_outlier_stress | test_bad_core_nearboundary | -2.3703 | 0.5653 | -2.1448 |
| pc1 | test_bad_outlier_stress | test_bad_core_nearboundary | -0.8803 | 1.7217 | -2.1216 |
| pc2 | test_bad_outlier_stress | test_bad_core_nearboundary | 1.9707 | -0.5044 | 1.8538 |
| flatline_ratio | test_bad_outlier_stress | test_bad_core_nearboundary | 1.9321 | -1.0575 | 1.7534 |
| baseline_step | test_bad_outlier_stress | test_bad_core_nearboundary | 1.4859 | -0.5946 | 1.6611 |
| qrs_band_ratio | test_bad_outlier_stress | test_bad_core_nearboundary | -2.1009 | -1.1087 | -1.6016 |
| sqi_basSQI | test_bad_outlier_stress | test_bad_core_nearboundary | -1.6447 | 0.4668 | -1.5759 |
| non_qrs_rms_ratio | test_bad_outlier_stress | test_bad_core_nearboundary | 0.2304 | 1.3104 | -1.5306 |
| detector_agreement | test_bad_outlier_stress | test_bad_core_nearboundary | -0.5486 | 1.0242 | -1.5112 |
| qrs_visibility | test_bad_outlier_stress | test_bad_core_nearboundary | -1.4438 | -1.1118 | -1.4480 |
| amplitude_entropy | test_bad_outlier_stress | test_bad_core_nearboundary | 0.3705 | 1.3586 | -1.0223 |
| template_corr | test_bad_outlier_stress | test_bad_core_nearboundary | -1.1138 | -1.5228 | 0.9364 |
| pc3 | test_bad_outlier_stress | test_bad_core_nearboundary | -0.8275 | -0.1181 | -0.9233 |
| mean_abs | test_bad_outlier_stress | test_bad_core_nearboundary | -0.9494 | -0.1914 | -0.6815 |
| low_amp_ratio | test_bad_outlier_stress | test_bad_core_nearboundary | -0.5181 | -1.1329 | 0.6613 |

![waveform examples](E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\original_bad_stress_failure\sqiquery_intermittentbad_balanced\original_bad_stress_waveform_examples.png)
