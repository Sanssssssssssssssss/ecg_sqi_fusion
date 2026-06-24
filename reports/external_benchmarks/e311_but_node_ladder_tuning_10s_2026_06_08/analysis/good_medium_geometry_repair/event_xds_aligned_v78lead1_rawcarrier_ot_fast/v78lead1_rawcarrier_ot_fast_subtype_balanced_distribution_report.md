# v78lead1_rawcarrier_ot_fast Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 23:12:38`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\protocol_v78lead1_rawcarrier_ot_fast_pc1500_s20260683`
- BUT reference: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v78lead1_rawcarrier_ot_fast\v78lead1_rawcarrier_ot_fast_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 214 | 1 | 2.68346 | 1.40452 | 1 | 2.74658 | 5.80779 |
| bad | bad_contact_reset_flatline | 188 | 214 | 1 | 0.490133 | 0.833427 | 0.798311 | 0.512173 | 1.04347 |
| bad | bad_dense_right_island | 782 | 215 | 1 | 10.0758 | 0.968225 | 1 | 9.92879 | 14.6766 |
| bad | bad_detector_template_disagree | 436 | 215 | 1 | 9.721 | 1.30835 | 1 | 9.53447 | 12.6618 |
| bad | bad_highfreq_detail_noise | 55 | 214 | 1 | 22.4264 | 1.62386 | 1 | 22.4006 | 41.8093 |
| bad | bad_low_qrs_visibility | 38 | 214 | 1 | 2.52441 | 1.22177 | 1 | 2.57169 | 4.88411 |
| bad | bad_other_boundary | 340 | 214 | 1 | 9.63408 | 1.62928 | 1 | 9.50549 | 15.6029 |
| good | good_clean_core | 2537 | 300 | 1 | 1.56149 | 1.1936 | 0.980335 | 1.51161 | 2.17307 |
| good | good_hard_baseline_lowqrs | 2830 | 300 | 1 | 0.901164 | 0.928147 | 0.671887 | 0.890925 | 1.26535 |
| good | good_isolated_low_purity | 791 | 300 | 1 | 0.906606 | 1.20222 | 0.941816 | 0.928093 | 1.58993 |
| good | good_mild_artifact_outlier | 802 | 300 | 1 | 1.09973 | 0.978966 | 0.908232 | 1.09237 | 1.68409 |
| good | good_overlap_boundary | 5074 | 300 | 1 | 0.91439 | 1.16519 | 0.841568 | 0.896222 | 1.76864 |
| medium | medium_clean_core | 708 | 250 | 1 | 0.61269 | 0.954444 | 0.723653 | 0.637371 | 1.25145 |
| medium | medium_hard_baseline_lowqrs | 1920 | 250 | 1 | 0.573728 | 0.923565 | 0.566247 | 0.679745 | 1.37801 |
| medium | medium_outlier_or_bad_boundary | 1081 | 250 | 1 | 0.882214 | 1.06263 | 0.901272 | 0.839666 | 1.83123 |
| medium | medium_overlap_boundary | 1552 | 250 | 1 | 0.571577 | 0.742634 | 0.752351 | 0.577255 | 1.11132 |
| medium | medium_visible_qrs_detail | 2109 | 250 | 1 | 0.90577 | 1.02087 | 0.930624 | 0.869719 | 1.5045 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.