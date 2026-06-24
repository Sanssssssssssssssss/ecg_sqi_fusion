# v62s Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 14:16:02`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\protocol_v62s_pc330_s20260652`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v62s\v62s_but_keepoutlier_bad_subtype_waveforms.png`

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
| test | good | good_clean_core | 10 |
| test | good | good_hard_baseline_lowqrs | 10 |
| test | good | good_isolated_low_purity | 10 |
| test | good | good_mild_artifact_outlier | 10 |
| test | good | good_overlap_boundary | 10 |
| test | medium | medium_clean_core | 9 |
| test | medium | medium_hard_baseline_lowqrs | 9 |
| test | medium | medium_isolated_lowqrs | 9 |
| test | medium | medium_outlier_or_bad_boundary | 9 |
| test | medium | medium_overlap_boundary | 9 |
| test | medium | medium_visible_qrs_detail | 9 |
| train | bad | bad_baseline_wander_lowfreq | 33 |
| train | bad | bad_contact_reset_flatline | 33 |
| train | bad | bad_dense_right_island | 34 |
| train | bad | bad_detector_template_disagree | 33 |
| train | bad | bad_highfreq_detail_noise | 33 |
| train | bad | bad_low_qrs_visibility | 33 |
| train | bad | bad_other_boundary | 33 |
| train | good | good_clean_core | 46 |
| train | good | good_hard_baseline_lowqrs | 46 |
| train | good | good_isolated_low_purity | 46 |
| train | good | good_mild_artifact_outlier | 46 |
| train | good | good_overlap_boundary | 46 |
| train | medium | medium_clean_core | 38 |
| train | medium | medium_hard_baseline_lowqrs | 38 |
| train | medium | medium_isolated_lowqrs | 38 |
| train | medium | medium_outlier_or_bad_boundary | 38 |
| train | medium | medium_overlap_boundary | 38 |
| train | medium | medium_visible_qrs_detail | 38 |
| val | bad | bad_baseline_wander_lowfreq | 7 |
| val | bad | bad_contact_reset_flatline | 7 |
| val | bad | bad_dense_right_island | 7 |
| val | bad | bad_detector_template_disagree | 7 |
| val | bad | bad_highfreq_detail_noise | 7 |
| val | bad | bad_low_qrs_visibility | 7 |
| val | bad | bad_other_boundary | 7 |
| val | good | good_clean_core | 10 |
| val | good | good_hard_baseline_lowqrs | 10 |
| val | good | good_isolated_low_purity | 10 |
| val | good | good_mild_artifact_outlier | 10 |
| val | good | good_overlap_boundary | 10 |
| val | medium | medium_clean_core | 8 |
| val | medium | medium_hard_baseline_lowqrs | 8 |
| val | medium | medium_isolated_lowqrs | 8 |
| val | medium | medium_outlier_or_bad_boundary | 8 |
| val | medium | medium_overlap_boundary | 8 |
| val | medium | medium_visible_qrs_detail | 8 |

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
| test | good | good_clean_core | 620 |
| test | good | good_hard_baseline_lowqrs | 706 |
| test | good | good_isolated_low_purity | 205 |
| test | good | good_mild_artifact_outlier | 215 |
| test | good | good_overlap_boundary | 1262 |
| test | medium | medium_clean_core | 182 |
| test | medium | medium_hard_baseline_lowqrs | 483 |
| test | medium | medium_outlier_or_bad_boundary | 267 |
| test | medium | medium_overlap_boundary | 376 |
| test | medium | medium_visible_qrs_detail | 534 |
| train | bad | bad_baseline_wander_lowfreq | 54 |
| train | bad | bad_contact_reset_flatline | 164 |
| train | bad | bad_dense_right_island | 682 |
| train | bad | bad_detector_template_disagree | 377 |
| train | bad | bad_highfreq_detail_noise | 48 |
| train | bad | bad_low_qrs_visibility | 33 |
| train | bad | bad_other_boundary | 298 |
| train | good | good_clean_core | 2217 |
| train | good | good_hard_baseline_lowqrs | 2484 |
| train | good | good_isolated_low_purity | 690 |
| train | good | good_mild_artifact_outlier | 701 |
| train | good | good_overlap_boundary | 4438 |
| train | medium | medium_clean_core | 638 |
| train | medium | medium_hard_baseline_lowqrs | 1681 |
| train | medium | medium_outlier_or_bad_boundary | 949 |
| train | medium | medium_overlap_boundary | 1346 |
| train | medium | medium_visible_qrs_detail | 1835 |
| val | bad | bad_baseline_wander_lowfreq | 8 |
| val | bad | bad_contact_reset_flatline | 24 |
| val | bad | bad_dense_right_island | 100 |
| val | bad | bad_detector_template_disagree | 59 |
| val | bad | bad_highfreq_detail_noise | 7 |
| val | bad | bad_low_qrs_visibility | 5 |
| val | bad | bad_other_boundary | 42 |
| val | good | good_clean_core | 320 |
| val | good | good_hard_baseline_lowqrs | 346 |
| val | good | good_isolated_low_purity | 101 |
| val | good | good_mild_artifact_outlier | 101 |
| val | good | good_overlap_boundary | 636 |
| val | medium | medium_clean_core | 70 |
| val | medium | medium_hard_baseline_lowqrs | 239 |
| val | medium | medium_outlier_or_bad_boundary | 132 |
| val | medium | medium_overlap_boundary | 206 |
| val | medium | medium_visible_qrs_detail | 274 |

## Distribution Metrics

Lower is better for robust gaps/MMD/Wasserstein/density gap. Discriminative AUC closer to 0.5 means synthetic and BUT are harder to tell apart.

| class_name | subtype | but_n_trainval | ptb_n | discriminative_auc | median_abs_robust_gap | mmd_rbf | pca_density_gap | quantile_loss | sliced_wasserstein |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | bad_baseline_wander_lowfreq | 62 | 47 | 1 | 3.45071 | 1.78182 | 1 | 3.36836 | 6.26797 |
| bad | bad_contact_reset_flatline | 188 | 47 | 1 | 0.482642 | 0.876261 | 0.953488 | 0.535972 | 1.07023 |
| bad | bad_dense_right_island | 782 | 48 | 1 | 8.53495 | 0.455112 | 1 | 8.55937 | 10.9659 |
| bad | bad_detector_template_disagree | 436 | 47 | 1 | 7.74839 | 0.471869 | 1 | 7.83804 | 9.50976 |
| bad | bad_highfreq_detail_noise | 55 | 47 | 1 | 22.298 | 1.26832 | 1 | 22.179 | 31.458 |
| bad | bad_low_qrs_visibility | 38 | 47 | 1 | 2.66082 | 1.05823 | 1 | 3.57766 | 6.03471 |
| bad | bad_other_boundary | 340 | 47 | 1 | 7.83832 | 0.526451 | 1 | 8.00153 | 11.4488 |
| good | good_clean_core | 2537 | 66 | 1 | 1.31516 | 1.03723 | 0.99916 | 1.29852 | 2.20967 |
| good | good_hard_baseline_lowqrs | 2830 | 66 | 1 | 0.652222 | 1.01462 | 0.861017 | 0.75055 | 1.37371 |
| good | good_isolated_low_purity | 791 | 66 | 1 | 1.23828 | 1.07304 | 0.995913 | 1.25855 | 1.83053 |
| good | good_mild_artifact_outlier | 802 | 66 | 1 | 0.993854 | 1.07281 | 0.983849 | 1.0273 | 1.56647 |
| good | good_overlap_boundary | 5074 | 66 | 1 | 0.865425 | 1.08457 | 0.971234 | 0.912733 | 1.6915 |
| medium | medium_clean_core | 708 | 55 | 1 | 0.646597 | 0.97494 | 0.890769 | 0.710906 | 1.3147 |
| medium | medium_hard_baseline_lowqrs | 1920 | 55 | 1 | 0.551374 | 0.949451 | 0.839686 | 0.645046 | 1.23541 |
| medium | medium_outlier_or_bad_boundary | 1081 | 55 | 1 | 0.794324 | 0.967207 | 0.941235 | 0.837964 | 1.85509 |
| medium | medium_overlap_boundary | 1552 | 55 | 1 | 0.507468 | 0.795023 | 0.901114 | 0.566234 | 1.0397 |
| medium | medium_visible_qrs_detail | 2109 | 55 | 1 | 0.858975 | 0.939363 | 0.961637 | 0.887154 | 1.43031 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.