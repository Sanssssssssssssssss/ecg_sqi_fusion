# v44s Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 04:59:52`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\protocol_v44s_pc300_s20260629`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44s\v44s_but_keepoutlier_bad_subtype_waveforms.png`

## PTB Synthetic Counts

| split | class_name | display_subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 7 |
| test | bad | bad_contact_reset_flatline | 7 |
| test | bad | bad_dense_right_island | 7 |
| test | bad | bad_detector_template_disagree | 7 |
| test | bad | bad_highfreq_detail_noise | 7 |
| test | bad | bad_low_qrs_visibility | 7 |
| test | bad | bad_other_boundary | 7 |
| test | good | good_clean_core | 12 |
| test | good | good_isolated_low_purity | 12 |
| test | good | good_mild_artifact_outlier | 12 |
| test | good | good_overlap_boundary | 12 |
| test | medium | medium_clean_core | 9 |
| test | medium | medium_isolated_lowqrs | 9 |
| test | medium | medium_outlier_or_bad_boundary | 9 |
| test | medium | medium_overlap_boundary | 9 |
| test | medium | medium_visible_qrs_detail | 9 |
| train | bad | bad_baseline_wander_lowfreq | 30 |
| train | bad | bad_contact_reset_flatline | 30 |
| train | bad | bad_dense_right_island | 30 |
| train | bad | bad_detector_template_disagree | 30 |
| train | bad | bad_highfreq_detail_noise | 30 |
| train | bad | bad_low_qrs_visibility | 30 |
| train | bad | bad_other_boundary | 29 |
| train | good | good_clean_core | 52 |
| train | good | good_isolated_low_purity | 52 |
| train | good | good_mild_artifact_outlier | 52 |
| train | good | good_overlap_boundary | 52 |
| train | medium | medium_clean_core | 42 |
| train | medium | medium_isolated_lowqrs | 42 |
| train | medium | medium_outlier_or_bad_boundary | 42 |
| train | medium | medium_overlap_boundary | 42 |
| train | medium | medium_visible_qrs_detail | 42 |
| val | bad | bad_baseline_wander_lowfreq | 6 |
| val | bad | bad_contact_reset_flatline | 6 |
| val | bad | bad_dense_right_island | 6 |
| val | bad | bad_detector_template_disagree | 6 |
| val | bad | bad_highfreq_detail_noise | 6 |
| val | bad | bad_low_qrs_visibility | 6 |
| val | bad | bad_other_boundary | 6 |
| val | good | good_clean_core | 11 |
| val | good | good_isolated_low_purity | 11 |
| val | good | good_mild_artifact_outlier | 11 |
| val | good | good_overlap_boundary | 11 |
| val | medium | medium_clean_core | 9 |
| val | medium | medium_isolated_lowqrs | 9 |
| val | medium | medium_outlier_or_bad_boundary | 9 |
| val | medium | medium_overlap_boundary | 9 |
| val | medium | medium_visible_qrs_detail | 9 |

## BUT Keep-Outlier Counts

| split | class_name | display_subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 15 |
| test | bad | bad_contact_reset_flatline | 47 |
| test | bad | bad_dense_right_island | 202 |
| test | bad | bad_detector_template_disagree | 117 |
| test | bad | bad_highfreq_detail_noise | 14 |
| test | bad | bad_low_qrs_visibility | 10 |
| test | bad | bad_other_boundary | 85 |
| test | good | good_clean_core | 622 |
| test | good | good_isolated_low_purity | 219 |
| test | good | good_mild_artifact_outlier | 765 |
| test | good | good_overlap_boundary | 1402 |
| test | medium | medium_clean_core | 230 |
| test | medium | medium_outlier_or_bad_boundary | 580 |
| test | medium | medium_overlap_boundary | 487 |
| test | medium | medium_visible_qrs_detail | 545 |
| train | bad | bad_baseline_wander_lowfreq | 54 |
| train | bad | bad_contact_reset_flatline | 164 |
| train | bad | bad_dense_right_island | 682 |
| train | bad | bad_detector_template_disagree | 377 |
| train | bad | bad_highfreq_detail_noise | 48 |
| train | bad | bad_low_qrs_visibility | 33 |
| train | bad | bad_other_boundary | 298 |
| train | good | good_clean_core | 2222 |
| train | good | good_isolated_low_purity | 750 |
| train | good | good_mild_artifact_outlier | 2669 |
| train | good | good_overlap_boundary | 4889 |
| train | medium | medium_clean_core | 786 |
| train | medium | medium_outlier_or_bad_boundary | 2073 |
| train | medium | medium_overlap_boundary | 1743 |
| train | medium | medium_visible_qrs_detail | 1847 |
| val | bad | bad_baseline_wander_lowfreq | 8 |
| val | bad | bad_contact_reset_flatline | 24 |
| val | bad | bad_dense_right_island | 100 |
| val | bad | bad_detector_template_disagree | 59 |
| val | bad | bad_highfreq_detail_noise | 7 |
| val | bad | bad_low_qrs_visibility | 5 |
| val | bad | bad_other_boundary | 42 |
| val | good | good_clean_core | 320 |
| val | good | good_isolated_low_purity | 111 |
| val | good | good_mild_artifact_outlier | 380 |
| val | good | good_overlap_boundary | 693 |
| val | medium | medium_clean_core | 85 |
| val | medium | medium_outlier_or_bad_boundary | 303 |
| val | medium | medium_overlap_boundary | 257 |
| val | medium | medium_visible_qrs_detail | 276 |

## Distribution Metrics

Lower is better for robust gaps/MMD/Wasserstein/density gap. Discriminative AUC closer to 0.5 means synthetic and BUT are harder to tell apart.

| class_name | subtype | but_n_trainval | ptb_n | discriminative_auc | median_abs_robust_gap | mmd_rbf | pca_density_gap | quantile_loss | sliced_wasserstein |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | bad_baseline_wander_lowfreq | 62 | 43 | 1 | 3.69998 | 1.30606 | 1 | 255.121 | 7.00816 |
| bad | bad_contact_reset_flatline | 188 | 43 | 1 | 0.537479 | 0.894581 | 0.877907 | 0.581881 | 1.13519 |
| bad | bad_dense_right_island | 782 | 43 | 1 | 43.8228 | 0.432017 | 1 | 47.5822 | 126.787 |
| bad | bad_detector_template_disagree | 436 | 43 | 1 | 22.1315 | 0.432878 | 1 | 25.9885 | 48.0862 |
| bad | bad_highfreq_detail_noise | 55 | 43 | 1 | 29.5964 | 1.38012 | 0.972222 | 32.3424 | 67.4682 |
| bad | bad_low_qrs_visibility | 38 | 43 | 1 | 6.19913 | 1.19334 | 1 | 7.1014 | 13.1179 |
| bad | bad_other_boundary | 340 | 42 | 1 | 26.3409 | 0.452491 | 1 | 27.6889 | 45.8184 |
| good | good_clean_core | 2542 | 75 | 1 | 1.51155 | 1.09172 | 0.998747 | 1.51717 | 1.87824 |
| good | good_isolated_low_purity | 861 | 75 | 1 | 1.60165 | 0.846187 | 0.991474 | 1.71903 | 2.24083 |
| good | good_mild_artifact_outlier | 3049 | 75 | 1 | 0.640989 | 1.02997 | 0.870318 | 0.799243 | 1.2249 |
| good | good_overlap_boundary | 5582 | 75 | 1 | 0.993617 | 1.06322 | 0.97837 | 1.02461 | 1.78746 |
| medium | medium_clean_core | 871 | 60 | 1 | 0.743323 | 0.978893 | 0.858209 | 0.783112 | 1.43182 |
| medium | medium_outlier_or_bad_boundary | 2376 | 60 | 1 | 0.656857 | 1.0343 | 0.909215 | 0.649498 | 1.69087 |
| medium | medium_overlap_boundary | 2000 | 60 | 1 | 0.545341 | 0.882984 | 0.872225 | 0.616537 | 0.990968 |
| medium | medium_visible_qrs_detail | 2123 | 60 | 1 | 0.886627 | 0.944809 | 0.969559 | 0.900762 | 1.5139 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.