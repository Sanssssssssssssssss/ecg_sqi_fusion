# v70fix_naturalizer_nonbad Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 19:37:00`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\protocol_v70fix_naturalizer_nonbad_pc1500_s20260672`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_but_keepoutlier_bad_subtype_waveforms.png`

## PTB Synthetic Counts

| split | class_name | display_subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 32 |
| test | bad | bad_contact_reset_flatline | 32 |
| test | bad | bad_dense_right_island | 33 |
| test | bad | bad_detector_template_disagree | 33 |
| test | bad | bad_highfreq_detail_noise | 32 |
| test | bad | bad_low_qrs_visibility | 32 |
| test | bad | bad_other_boundary | 32 |
| test | good | good_clean_core | 45 |
| test | good | good_hard_baseline_lowqrs | 45 |
| test | good | good_isolated_low_purity | 45 |
| test | good | good_mild_artifact_outlier | 45 |
| test | good | good_overlap_boundary | 45 |
| test | medium | medium_clean_core | 37 |
| test | medium | medium_hard_baseline_lowqrs | 37 |
| test | medium | medium_isolated_lowqrs | 37 |
| test | medium | medium_outlier_or_bad_boundary | 37 |
| test | medium | medium_overlap_boundary | 37 |
| test | medium | medium_visible_qrs_detail | 37 |
| train | bad | bad_baseline_wander_lowfreq | 150 |
| train | bad | bad_contact_reset_flatline | 150 |
| train | bad | bad_dense_right_island | 150 |
| train | bad | bad_detector_template_disagree | 150 |
| train | bad | bad_highfreq_detail_noise | 150 |
| train | bad | bad_low_qrs_visibility | 150 |
| train | bad | bad_other_boundary | 150 |
| train | good | good_clean_core | 210 |
| train | good | good_hard_baseline_lowqrs | 210 |
| train | good | good_isolated_low_purity | 210 |
| train | good | good_mild_artifact_outlier | 210 |
| train | good | good_overlap_boundary | 210 |
| train | medium | medium_clean_core | 175 |
| train | medium | medium_hard_baseline_lowqrs | 175 |
| train | medium | medium_isolated_lowqrs | 175 |
| train | medium | medium_outlier_or_bad_boundary | 175 |
| train | medium | medium_overlap_boundary | 175 |
| train | medium | medium_visible_qrs_detail | 175 |
| val | bad | bad_baseline_wander_lowfreq | 32 |
| val | bad | bad_contact_reset_flatline | 32 |
| val | bad | bad_dense_right_island | 32 |
| val | bad | bad_detector_template_disagree | 32 |
| val | bad | bad_highfreq_detail_noise | 32 |
| val | bad | bad_low_qrs_visibility | 32 |
| val | bad | bad_other_boundary | 32 |
| val | good | good_clean_core | 45 |
| val | good | good_hard_baseline_lowqrs | 45 |
| val | good | good_isolated_low_purity | 45 |
| val | good | good_mild_artifact_outlier | 45 |
| val | good | good_overlap_boundary | 45 |
| val | medium | medium_clean_core | 38 |
| val | medium | medium_hard_baseline_lowqrs | 38 |
| val | medium | medium_isolated_lowqrs | 38 |
| val | medium | medium_outlier_or_bad_boundary | 38 |
| val | medium | medium_overlap_boundary | 38 |
| val | medium | medium_visible_qrs_detail | 38 |

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
| bad | bad_baseline_wander_lowfreq | 62 | 214 | 1 | 3.61895 | 1.26486 | 1 | 3.61717 | 6.77688 |
| bad | bad_contact_reset_flatline | 188 | 214 | 1 | 0.499954 | 0.821277 | 0.785859 | 0.510392 | 1.06437 |
| bad | bad_dense_right_island | 782 | 215 | 1 | 9.37343 | 0.882886 | 1 | 9.44115 | 11.054 |
| bad | bad_detector_template_disagree | 436 | 215 | 1 | 8.66732 | 1.22969 | 1 | 8.73716 | 11.6332 |
| bad | bad_highfreq_detail_noise | 55 | 214 | 1 | 23.6076 | 1.57962 | 1 | 23.6329 | 42.8146 |
| bad | bad_low_qrs_visibility | 38 | 214 | 1 | 3.22199 | 1.49439 | 1 | 3.90863 | 6.17321 |
| bad | bad_other_boundary | 340 | 214 | 1 | 8.71995 | 1.58248 | 1 | 8.78313 | 11.993 |
| good | good_clean_core | 2537 | 300 | 1 | 3.51416 | 0.968944 | 1 | 3.87909 | 5.05654 |
| good | good_hard_baseline_lowqrs | 2830 | 300 | 1 | 1.12643 | 0.964564 | 0.846518 | 1.11372 | 1.6352 |
| good | good_isolated_low_purity | 791 | 300 | 1 | 3.31959 | 1.39257 | 1 | 3.32717 | 5.36381 |
| good | good_mild_artifact_outlier | 802 | 300 | 1 | 1.81155 | 1.08299 | 0.994716 | 1.77295 | 2.1791 |
| good | good_overlap_boundary | 5074 | 300 | 1 | 2.33146 | 1.03108 | 0.997721 | 2.37123 | 3.28893 |
| medium | medium_clean_core | 708 | 250 | 1 | 0.776972 | 1.00898 | 0.742333 | 0.822122 | 1.35038 |
| medium | medium_hard_baseline_lowqrs | 1920 | 250 | 1 | 0.6356 | 0.936508 | 0.603358 | 0.739132 | 1.35856 |
| medium | medium_outlier_or_bad_boundary | 1081 | 250 | 1 | 1.10623 | 1.13306 | 0.941942 | 1.05328 | 1.59758 |
| medium | medium_overlap_boundary | 1552 | 250 | 1 | 0.682526 | 0.82651 | 0.783898 | 0.729293 | 1.13422 |
| medium | medium_visible_qrs_detail | 2109 | 250 | 1 | 1.2242 | 1.13966 | 0.969955 | 1.20648 | 1.43003 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.