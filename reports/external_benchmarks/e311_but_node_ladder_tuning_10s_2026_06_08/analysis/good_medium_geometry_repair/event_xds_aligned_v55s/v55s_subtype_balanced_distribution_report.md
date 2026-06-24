# v55s Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 09:32:55`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\protocol_v55s_pc330_s20260645`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v55s\v55s_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 47 | 1 | 2.12358 | 1.75324 | 0.980769 | 2.36821 | 6.52855 |
| bad | bad_contact_reset_flatline | 188 | 47 | 1 | 0.513877 | 0.880041 | 0.894737 | 0.539302 | 1.25365 |
| bad | bad_dense_right_island | 782 | 48 | 1 | 12.476 | 0.447706 | 1 | 12.4921 | 16.7403 |
| bad | bad_detector_template_disagree | 436 | 47 | 1 | 11.4744 | 0.451433 | 1 | 11.4939 | 15.2546 |
| bad | bad_highfreq_detail_noise | 55 | 47 | 1 | 20.5872 | 1.34021 | 1 | 20.4226 | 32.725 |
| bad | bad_low_qrs_visibility | 38 | 47 | 1 | 2.11369 | 1.21533 | 0.977273 | 2.25328 | 4.93732 |
| bad | bad_other_boundary | 340 | 47 | 1 | 10.4155 | 0.512766 | 1 | 10.4477 | 13.5733 |
| good | good_clean_core | 2537 | 66 | 1 | 1.43904 | 0.954609 | 1 | 1.45281 | 1.78062 |
| good | good_hard_baseline_lowqrs | 2830 | 66 | 1 | 0.700968 | 1.01421 | 0.865537 | 0.76803 | 1.30489 |
| good | good_isolated_low_purity | 791 | 66 | 1 | 1.47747 | 0.729773 | 0.998665 | 1.68519 | 1.9236 |
| good | good_mild_artifact_outlier | 802 | 66 | 1 | 1.21551 | 1.03806 | 0.983914 | 1.22846 | 1.66716 |
| good | good_overlap_boundary | 5074 | 66 | 1 | 1.00263 | 1.11144 | 0.993827 | 1.06312 | 1.58152 |
| medium | medium_clean_core | 708 | 55 | 1 | 0.699377 | 0.893411 | 0.905054 | 0.698193 | 1.17626 |
| medium | medium_hard_baseline_lowqrs | 1920 | 55 | 1 | 0.592617 | 0.939848 | 0.871148 | 0.692034 | 1.37553 |
| medium | medium_outlier_or_bad_boundary | 1081 | 55 | 1 | 1.04267 | 1.02695 | 0.988048 | 1.01879 | 1.6285 |
| medium | medium_overlap_boundary | 1552 | 55 | 1 | 0.553092 | 0.812918 | 0.917884 | 0.635223 | 1.08931 |
| medium | medium_visible_qrs_detail | 2109 | 55 | 1 | 0.919991 | 0.954613 | 0.982125 | 0.963588 | 1.48677 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.