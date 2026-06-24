# v63s Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 14:24:35`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\protocol_v63s_pc330_s20260653`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v63s\v63s_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 47 | 1 | 3.46319 | 1.79761 | 1 | 3.41824 | 6.79212 |
| bad | bad_contact_reset_flatline | 188 | 47 | 1 | 0.47876 | 0.867754 | 0.894737 | 0.522573 | 1.20758 |
| bad | bad_dense_right_island | 782 | 48 | 1 | 8.42643 | 0.469805 | 1 | 8.47491 | 10.5971 |
| bad | bad_detector_template_disagree | 436 | 47 | 1 | 7.78065 | 0.471596 | 1 | 7.89205 | 11.4219 |
| bad | bad_highfreq_detail_noise | 55 | 47 | 1 | 22.2995 | 1.28797 | 1 | 22.3353 | 36.4126 |
| bad | bad_low_qrs_visibility | 38 | 47 | 1 | 2.74765 | 1.04478 | 1 | 3.6674 | 5.61777 |
| bad | bad_other_boundary | 340 | 47 | 1 | 7.80986 | 0.527571 | 1 | 7.90497 | 9.72957 |
| good | good_clean_core | 2537 | 66 | 1 | 1.29961 | 1.04308 | 0.99958 | 1.2859 | 1.92699 |
| good | good_hard_baseline_lowqrs | 2830 | 66 | 1 | 0.698164 | 0.93598 | 0.860316 | 0.768394 | 1.49081 |
| good | good_isolated_low_purity | 791 | 66 | 1 | 1.25812 | 1.05119 | 0.997271 | 1.31613 | 2.14948 |
| good | good_mild_artifact_outlier | 802 | 66 | 1 | 1.01109 | 1.0392 | 0.979893 | 1.01235 | 1.75952 |
| good | good_overlap_boundary | 5074 | 66 | 1 | 0.888407 | 1.03314 | 0.979983 | 0.934892 | 1.65376 |
| medium | medium_clean_core | 708 | 55 | 1 | 0.590554 | 0.904752 | 0.903226 | 0.654493 | 1.25731 |
| medium | medium_hard_baseline_lowqrs | 1920 | 55 | 1 | 0.571255 | 0.938935 | 0.848094 | 0.644489 | 1.14592 |
| medium | medium_outlier_or_bad_boundary | 1081 | 55 | 1 | 0.845989 | 0.945088 | 0.947264 | 0.846734 | 1.7797 |
| medium | medium_overlap_boundary | 1552 | 55 | 1 | 0.499978 | 0.815555 | 0.892061 | 0.568043 | 0.997928 |
| medium | medium_visible_qrs_detail | 2109 | 55 | 1 | 0.834376 | 0.965895 | 0.969852 | 0.853291 | 1.41672 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.