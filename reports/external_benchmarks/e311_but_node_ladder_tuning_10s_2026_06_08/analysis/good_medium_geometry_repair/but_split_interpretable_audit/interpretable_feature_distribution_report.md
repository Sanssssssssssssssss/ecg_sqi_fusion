# Interpretable Feature Distribution Audit

This separates waveform-computable targets from weak geometry proxies and atlas-only label geometry.

## Feature Taxonomy Counts

| taxonomy | n |
| --- | --- |
| atlas_label_geometry_not_waveform_fact | 7 |
| stable_waveform_sqi_morph_rr | 42 |
| weak_target_distribution_proxy | 3 |

## Stable Waveform-Computable Targets

| feature | taxonomy |
| --- | --- |
| qrs_visibility | stable_waveform_sqi_morph_rr |
| qrs_band_ratio | stable_waveform_sqi_morph_rr |
| qrs_prom_p90 | stable_waveform_sqi_morph_rr |
| template_corr | stable_waveform_sqi_morph_rr |
| detector_agreement | stable_waveform_sqi_morph_rr |
| baseline_step | stable_waveform_sqi_morph_rr |
| flatline_ratio | stable_waveform_sqi_morph_rr |
| contact_loss_win_ratio | stable_waveform_sqi_morph_rr |
| non_qrs_rms_ratio | stable_waveform_sqi_morph_rr |
| non_qrs_diff_p95 | stable_waveform_sqi_morph_rr |
| diff_abs_p95 | stable_waveform_sqi_morph_rr |
| band_15_30 | stable_waveform_sqi_morph_rr |
| band_30_45 | stable_waveform_sqi_morph_rr |
| rms | stable_waveform_sqi_morph_rr |
| std | stable_waveform_sqi_morph_rr |
| mean_abs | stable_waveform_sqi_morph_rr |
| ptp_p99_p01 | stable_waveform_sqi_morph_rr |
| amplitude_entropy | stable_waveform_sqi_morph_rr |
| low_amp_ratio | stable_waveform_sqi_morph_rr |
| sqi_iSQI | stable_waveform_sqi_morph_rr |
| sqi_bSQI | stable_waveform_sqi_morph_rr |
| sqi_pSQI | stable_waveform_sqi_morph_rr |
| sqi_sSQI | stable_waveform_sqi_morph_rr |
| sqi_kSQI | stable_waveform_sqi_morph_rr |
| sqi_fSQI | stable_waveform_sqi_morph_rr |
| sqi_basSQI | stable_waveform_sqi_morph_rr |
| hjorth_activity | stable_waveform_sqi_morph_rr |
| hjorth_mobility | stable_waveform_sqi_morph_rr |
| hjorth_complexity | stable_waveform_sqi_morph_rr |
| zero_crossing_rate | stable_waveform_sqi_morph_rr |
| diff_zero_crossing_rate | stable_waveform_sqi_morph_rr |
| sample_entropy_proxy | stable_waveform_sqi_morph_rr |
| higuchi_fd_proxy | stable_waveform_sqi_morph_rr |
| wavelet_e0 | stable_waveform_sqi_morph_rr |
| wavelet_e1 | stable_waveform_sqi_morph_rr |
| wavelet_e2 | stable_waveform_sqi_morph_rr |
| wavelet_e3 | stable_waveform_sqi_morph_rr |
| wavelet_e4 | stable_waveform_sqi_morph_rr |
| fatal_or_score | stable_waveform_sqi_morph_rr |
| band_0p3_1 | stable_waveform_sqi_morph_rr |
| band_1_5 | stable_waveform_sqi_morph_rr |
| band_5_15 | stable_waveform_sqi_morph_rr |

## Diagnostic-Only Geometry Proxies

| feature | taxonomy |
| --- | --- |
| pc1 | weak_target_distribution_proxy |
| pc3 | weak_target_distribution_proxy |
| pca_margin | weak_target_distribution_proxy |

## Not Final Inference/Claim Targets

| feature | taxonomy |
| --- | --- |
| pc2 | atlas_label_geometry_not_waveform_fact |
| pc4 | atlas_label_geometry_not_waveform_fact |
| boundary_confidence | atlas_label_geometry_not_waveform_fact |
| region_confidence | atlas_label_geometry_not_waveform_fact |
| knn_label_purity | atlas_label_geometry_not_waveform_fact |
| class_margin_percentile | atlas_label_geometry_not_waveform_fact |
| class_centrality_percentile | atlas_label_geometry_not_waveform_fact |

## Current Split: Largest Stable Feature Train-vs-Test Gaps

| scheme | class_name | feature | taxonomy | ks_train_vs_test | train_median | test_median | delta_median_test_minus_train | train_p10 | train_p90 | test_p10 | test_p90 | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | all | band_0p3_1 | stable_waveform_sqi_morph_rr | 0.656979 | 0.0129168 | 0.242681 | 0.229765 | 0.000192086 | 0.152534 | 0.0383957 | 0.675903 | 23322 | 8477 |
| current | all | sqi_basSQI | stable_waveform_sqi_morph_rr | 0.645299 | 0.971646 | 0.779845 | -0.191801 | 0.879264 | 0.999684 | 0.399839 | 0.961133 | 23322 | 8477 |
| current | all | qrs_band_ratio | stable_waveform_sqi_morph_rr | 0.634392 | 0.552419 | 0.38391 | -0.16851 | 0.429148 | 0.811143 | 0.0847267 | 0.522526 | 23322 | 8477 |
| current | all | hjorth_complexity | stable_waveform_sqi_morph_rr | 0.614039 | 1.66217 | 2.1862 | 0.524028 | 1.21439 | 2.06504 | 1.65801 | 6.75803 | 23322 | 8477 |
| current | all | diff_abs_p95 | stable_waveform_sqi_morph_rr | 0.5625 | 0.264651 | 0.0663765 | -0.198275 | 0.098886 | 0.39757 | 0.0279353 | 0.375137 | 23322 | 8477 |
| current | all | wavelet_e0 | stable_waveform_sqi_morph_rr | 0.558664 | 0.39242 | 0.69068 | 0.298259 | 0.040719 | 0.674435 | 0.398755 | 0.96524 | 23322 | 8477 |
| current | all | band_15_30 | stable_waveform_sqi_morph_rr | 0.553848 | 0.250306 | 0.159504 | -0.0908012 | 0.184309 | 0.834313 | 0.0222854 | 0.268251 | 23322 | 8477 |
| current | all | baseline_step | stable_waveform_sqi_morph_rr | 0.549413 | 0.285214 | 0.915184 | 0.62997 | 0.0267381 | 0.922197 | 0.367883 | 1.83826 | 23322 | 8477 |
| current | all | qrs_visibility | stable_waveform_sqi_morph_rr | 0.52729 | 0.377941 | 0.106576 | -0.271366 | 0.168969 | 0.721767 | 0.0181091 | 0.373231 | 23322 | 8477 |
| current | all | wavelet_e2 | stable_waveform_sqi_morph_rr | 0.526748 | 0.207876 | 0.0936904 | -0.114186 | 0.105504 | 0.408895 | 0.00850614 | 0.210407 | 23322 | 8477 |
| current | all | zero_crossing_rate | stable_waveform_sqi_morph_rr | 0.500125 | 0.0968775 | 0.040032 | -0.0568455 | 0.0496397 | 0.57486 | 0.0176141 | 0.143315 | 23322 | 8477 |
| current | all | hjorth_mobility | stable_waveform_sqi_morph_rr | 0.47492 | 0.63537 | 0.499381 | -0.135989 | 0.52044 | 1.43415 | 0.155824 | 0.701889 | 23322 | 8477 |
| current | all | wavelet_e3 | stable_waveform_sqi_morph_rr | 0.469217 | 0.0753545 | 0.0344271 | -0.0409274 | 0.0369605 | 0.264514 | 0.00241217 | 0.0911433 | 23322 | 8477 |
| current | all | non_qrs_diff_p95 | stable_waveform_sqi_morph_rr | 0.446913 | 0.0750575 | 0.0300576 | -0.0449999 | 0.0336691 | 0.374999 | 0.0133347 | 0.133245 | 23322 | 8477 |
| current | all | band_5_15 | stable_waveform_sqi_morph_rr | 0.409194 | 0.422837 | 0.328551 | -0.0942863 | 0.0496385 | 0.519552 | 0.0765742 | 0.435555 | 23322 | 8477 |
| current | all | wavelet_e4 | stable_waveform_sqi_morph_rr | 0.369645 | 0.00685971 | 0.00366707 | -0.00319263 | 0.0031967 | 0.245229 | 0.000261671 | 0.0124463 | 23322 | 8477 |

## PTB Synthetic vs BUT Current Test: Largest Stable Feature Gaps

| class_name | but_bucket | feature | ks_ptb_synth_vs_but | ptb_median | but_median | delta_but_minus_ptb | ptb_nonmissing | but_nonmissing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | but_current_test | diff_abs_p95 | 0.454813 | 0.169026 | 0.0663765 | -0.102649 | 583 | 8477 |
| all | but_current_test | std | 0.438034 | 0.2714 | 0.204329 | -0.0670712 | 583 | 8477 |
| all | but_current_test | baseline_step | 0.422344 | 0.44448 | 0.915184 | 0.470704 | 583 | 8477 |
| all | but_current_test | rms | 0.412607 | 0.273906 | 0.207639 | -0.0662674 | 583 | 8477 |
| all | but_current_test | detector_agreement | 0.373652 | 0.327353 | 0.258569 | -0.0687843 | 583 | 8477 |
| all | but_current_test | qrs_band_ratio | 0.352931 | 0.490321 | 0.38391 | -0.106411 | 583 | 8477 |
| all | but_current_test | mean_abs | 0.305042 | 0.118542 | 0.127825 | 0.0092832 | 583 | 8477 |
| all | but_current_test | qrs_visibility | 0.289628 | 0.231873 | 0.106576 | -0.125297 | 583 | 8477 |
| all | but_current_test | ptp_p99_p01 | 0.278581 | 1.51104 | 1.20204 | -0.308996 | 583 | 8477 |
| all | but_current_test | low_amp_ratio | 0.278168 | 0.2112 | 0.1752 | -0.036 | 583 | 8477 |
| all | but_current_test | band_30_45 | 0.271388 | 0.0102327 | 0.0205357 | 0.010303 | 583 | 8477 |
| all | but_current_test | qrs_prom_p90 | 0.207867 | 5.60279 | 6.03394 | 0.43115 | 583 | 8477 |
| all | but_current_test | sqi_sSQI | 0.201706 | 2.87944 | 2.60335 | -0.276093 | 583 | 8477 |
| all | but_current_test | non_qrs_rms_ratio | 0.188262 | 0.37857 | 0.450596 | 0.0720257 | 583 | 8477 |
| all | but_current_test | amplitude_entropy | 0.176016 | 0.69483 | 0.723555 | 0.0287244 | 583 | 8477 |
| all | but_current_test | template_corr | 0.154027 | 0.573308 | 0.528815 | -0.0444921 | 583 | 8477 |

## PC Axis Interpretation

The PC columns are PCA coordinates in target-geometry/SQI space. The table below reports correlations with waveform-computable features, not physiological facts.

| axis | feature | pearson_corr | abs_corr | n |
| --- | --- | --- | --- | --- |
| pc1 | non_qrs_diff_p95 | 0.957156 | 0.957156 | 32956 |
| pc1 | sample_entropy_proxy | 0.956514 | 0.956514 | 32956 |
| pc1 | zero_crossing_rate | 0.953787 | 0.953787 | 32956 |
| pc1 | higuchi_fd_proxy | 0.948673 | 0.948673 | 32956 |
| pc1 | sqi_pSQI | -0.919441 | 0.919441 | 32956 |
| pc1 | band_15_30 | 0.90734 | 0.90734 | 32956 |
| pc1 | hjorth_mobility | 0.884357 | 0.884357 | 32956 |
| pc1 | wavelet_e4 | 0.877698 | 0.877698 | 32956 |
| pc2 | band_0p3_1 | 0.949247 | 0.949247 | 32956 |
| pc2 | sqi_basSQI | -0.939422 | 0.939422 | 32956 |
| pc2 | baseline_step | 0.806067 | 0.806067 | 32956 |
| pc2 | hjorth_complexity | 0.79193 | 0.79193 | 32956 |
| pc2 | qrs_band_ratio | -0.769057 | 0.769057 | 32956 |
| pc2 | wavelet_e0 | 0.766662 | 0.766662 | 32956 |
| pc2 | diff_abs_p95 | -0.660591 | 0.660591 | 32956 |
| pc2 | wavelet_e2 | -0.652274 | 0.652274 | 32956 |
| pc3 | flatline_ratio | -0.487422 | 0.487422 | 32956 |
| pc3 | sqi_fSQI | -0.447373 | 0.447373 | 32956 |
| pc3 | detector_agreement | -0.422595 | 0.422595 | 32956 |
| pc3 | qrs_visibility | -0.387998 | 0.387998 | 32956 |
| pc3 | fatal_or_score | 0.340055 | 0.340055 | 32956 |
| pc3 | band_5_15 | 0.299025 | 0.299025 | 32956 |
| pc3 | mean_abs | 0.297033 | 0.297033 | 32956 |
| pc3 | sqi_bSQI | 0.278541 | 0.278541 | 32956 |
| pc4 | sqi_kSQI | 0.440472 | 0.440472 | 32956 |
| pc4 | flatline_ratio | 0.430059 | 0.430059 | 32956 |
| pc4 | detector_agreement | -0.404634 | 0.404634 | 32956 |
| pc4 | qrs_prom_p90 | 0.366608 | 0.366608 | 32956 |
| pc4 | diff_abs_p95 | -0.359866 | 0.359866 | 32956 |
| pc4 | low_amp_ratio | 0.353054 | 0.353054 | 32956 |
| pc4 | fatal_or_score | -0.345136 | 0.345136 | 32956 |
| pc4 | sqi_sSQI | 0.304081 | 0.304081 | 32956 |
| pca_margin | sqi_pSQI | -0.891124 | 0.891124 | 32956 |
| pca_margin | band_15_30 | 0.880926 | 0.880926 | 32956 |
| pca_margin | zero_crossing_rate | 0.873345 | 0.873345 | 32956 |
| pca_margin | hjorth_mobility | 0.861202 | 0.861202 | 32956 |
| pca_margin | wavelet_e4 | 0.845161 | 0.845161 | 32956 |
| pca_margin | wavelet_e3 | 0.844007 | 0.844007 | 32956 |
| pca_margin | higuchi_fd_proxy | 0.839622 | 0.839622 | 32956 |
| pca_margin | non_qrs_diff_p95 | 0.838826 | 0.838826 | 32956 |

Feature KS CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_split_feature_ks.csv`
PTB/BUT gap CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\ptb_synthetic_vs_but_feature_gap.csv`
PC correlation CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\pc_axis_waveform_feature_correlations.csv`
