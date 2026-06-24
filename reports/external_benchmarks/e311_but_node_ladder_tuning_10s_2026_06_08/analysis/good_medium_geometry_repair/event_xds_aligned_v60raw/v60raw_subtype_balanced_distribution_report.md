# v60raw Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 13:29:31`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\protocol_v60raw_pc3000_s20260650`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\v60raw_but_keepoutlier_bad_subtype_waveforms.png`

## PTB Synthetic Counts

| split | class_name | display_subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 65 |
| test | bad | bad_contact_reset_flatline | 65 |
| test | bad | bad_dense_right_island | 65 |
| test | bad | bad_detector_template_disagree | 65 |
| test | bad | bad_highfreq_detail_noise | 64 |
| test | bad | bad_low_qrs_visibility | 64 |
| test | bad | bad_other_boundary | 64 |
| test | good | good_clean_core | 90 |
| test | good | good_hard_baseline_lowqrs | 90 |
| test | good | good_isolated_low_purity | 90 |
| test | good | good_mild_artifact_outlier | 90 |
| test | good | good_overlap_boundary | 90 |
| test | medium | medium_clean_core | 75 |
| test | medium | medium_hard_baseline_lowqrs | 75 |
| test | medium | medium_isolated_lowqrs | 75 |
| test | medium | medium_outlier_or_bad_boundary | 75 |
| test | medium | medium_overlap_boundary | 75 |
| test | medium | medium_visible_qrs_detail | 75 |
| train | bad | bad_baseline_wander_lowfreq | 300 |
| train | bad | bad_contact_reset_flatline | 300 |
| train | bad | bad_dense_right_island | 300 |
| train | bad | bad_detector_template_disagree | 300 |
| train | bad | bad_highfreq_detail_noise | 300 |
| train | bad | bad_low_qrs_visibility | 300 |
| train | bad | bad_other_boundary | 300 |
| train | good | good_clean_core | 420 |
| train | good | good_hard_baseline_lowqrs | 420 |
| train | good | good_isolated_low_purity | 420 |
| train | good | good_mild_artifact_outlier | 420 |
| train | good | good_overlap_boundary | 420 |
| train | medium | medium_clean_core | 350 |
| train | medium | medium_hard_baseline_lowqrs | 350 |
| train | medium | medium_isolated_lowqrs | 350 |
| train | medium | medium_outlier_or_bad_boundary | 350 |
| train | medium | medium_overlap_boundary | 350 |
| train | medium | medium_visible_qrs_detail | 350 |
| val | bad | bad_baseline_wander_lowfreq | 64 |
| val | bad | bad_contact_reset_flatline | 64 |
| val | bad | bad_dense_right_island | 64 |
| val | bad | bad_detector_template_disagree | 64 |
| val | bad | bad_highfreq_detail_noise | 64 |
| val | bad | bad_low_qrs_visibility | 64 |
| val | bad | bad_other_boundary | 64 |
| val | good | good_clean_core | 90 |
| val | good | good_hard_baseline_lowqrs | 90 |
| val | good | good_isolated_low_purity | 90 |
| val | good | good_mild_artifact_outlier | 90 |
| val | good | good_overlap_boundary | 90 |
| val | medium | medium_clean_core | 75 |
| val | medium | medium_hard_baseline_lowqrs | 75 |
| val | medium | medium_isolated_lowqrs | 75 |
| val | medium | medium_outlier_or_bad_boundary | 75 |
| val | medium | medium_overlap_boundary | 75 |
| val | medium | medium_visible_qrs_detail | 75 |

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
| bad | bad_baseline_wander_lowfreq | 62 | 429 | 1 | 3.44041 | 1.13545 | 1 | 3.40779 | 6.41919 |
| bad | bad_contact_reset_flatline | 188 | 429 | 1 | 0.470819 | 0.868337 | 0.794643 | 0.499542 | 1.06352 |
| bad | bad_dense_right_island | 782 | 429 | 1 | 8.39985 | 1.75236 | 1 | 8.53608 | 9.49151 |
| bad | bad_detector_template_disagree | 436 | 429 | 1 | 7.82 | 1.34327 | 1 | 7.87871 | 10.1686 |
| bad | bad_highfreq_detail_noise | 55 | 428 | 1 | 22.2348 | 1.51034 | 1 | 22.2948 | 40.0514 |
| bad | bad_low_qrs_visibility | 38 | 428 | 1 | 2.72923 | 1.21951 | 1 | 3.73986 | 5.39552 |
| bad | bad_other_boundary | 340 | 428 | 1 | 7.86065 | 1.75089 | 1 | 7.91791 | 8.99568 |
| good | good_clean_core | 2537 | 600 | 1 | 1.30359 | 1.17928 | 0.995795 | 1.2805 | 2.16468 |
| good | good_hard_baseline_lowqrs | 2830 | 600 | 1 | 0.676715 | 0.934364 | 0.667358 | 0.706039 | 1.2765 |
| good | good_isolated_low_purity | 791 | 600 | 1 | 1.25703 | 1.12062 | 0.98613 | 1.27859 | 1.83583 |
| good | good_mild_artifact_outlier | 802 | 600 | 1 | 1.00388 | 0.968908 | 0.924185 | 1.00396 | 1.63445 |
| good | good_overlap_boundary | 5074 | 600 | 1 | 0.873582 | 1.08963 | 0.926265 | 0.910897 | 1.83149 |
| medium | medium_clean_core | 708 | 500 | 1 | 0.58803 | 0.90087 | 0.603816 | 0.622126 | 1.2112 |
| medium | medium_hard_baseline_lowqrs | 1920 | 500 | 1 | 0.561697 | 0.878191 | 0.563643 | 0.624315 | 1.31307 |
| medium | medium_outlier_or_bad_boundary | 1081 | 500 | 1 | 0.834775 | 0.992566 | 0.868662 | 0.828888 | 1.67515 |
| medium | medium_overlap_boundary | 1552 | 500 | 1 | 0.499204 | 0.73483 | 0.653421 | 0.538698 | 1.03353 |
| medium | medium_visible_qrs_detail | 2109 | 500 | 1 | 0.807588 | 0.980118 | 0.84612 | 0.829024 | 1.58728 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.