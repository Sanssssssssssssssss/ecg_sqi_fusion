# v48raw Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 05:39:05`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\protocol_v48raw_pc3000_s20260637`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48raw\v48raw_but_keepoutlier_bad_subtype_waveforms.png`

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
| test | good | good_clean_core | 113 |
| test | good | good_isolated_low_purity | 113 |
| test | good | good_mild_artifact_outlier | 113 |
| test | good | good_overlap_boundary | 113 |
| test | medium | medium_clean_core | 90 |
| test | medium | medium_isolated_lowqrs | 90 |
| test | medium | medium_outlier_or_bad_boundary | 90 |
| test | medium | medium_overlap_boundary | 90 |
| test | medium | medium_visible_qrs_detail | 90 |
| train | bad | bad_baseline_wander_lowfreq | 300 |
| train | bad | bad_contact_reset_flatline | 300 |
| train | bad | bad_dense_right_island | 300 |
| train | bad | bad_detector_template_disagree | 300 |
| train | bad | bad_highfreq_detail_noise | 300 |
| train | bad | bad_low_qrs_visibility | 300 |
| train | bad | bad_other_boundary | 300 |
| train | good | good_clean_core | 525 |
| train | good | good_isolated_low_purity | 525 |
| train | good | good_mild_artifact_outlier | 525 |
| train | good | good_overlap_boundary | 525 |
| train | medium | medium_clean_core | 420 |
| train | medium | medium_isolated_lowqrs | 420 |
| train | medium | medium_outlier_or_bad_boundary | 420 |
| train | medium | medium_overlap_boundary | 420 |
| train | medium | medium_visible_qrs_detail | 420 |
| val | bad | bad_baseline_wander_lowfreq | 64 |
| val | bad | bad_contact_reset_flatline | 64 |
| val | bad | bad_dense_right_island | 64 |
| val | bad | bad_detector_template_disagree | 64 |
| val | bad | bad_highfreq_detail_noise | 64 |
| val | bad | bad_low_qrs_visibility | 64 |
| val | bad | bad_other_boundary | 64 |
| val | good | good_clean_core | 112 |
| val | good | good_isolated_low_purity | 112 |
| val | good | good_mild_artifact_outlier | 112 |
| val | good | good_overlap_boundary | 112 |
| val | medium | medium_clean_core | 90 |
| val | medium | medium_isolated_lowqrs | 90 |
| val | medium | medium_outlier_or_bad_boundary | 90 |
| val | medium | medium_overlap_boundary | 90 |
| val | medium | medium_visible_qrs_detail | 90 |

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
| bad | bad_baseline_wander_lowfreq | 62 | 429 | 1 | 3.49294 | 1.07337 | 1 | 3.46894 | 6.65856 |
| bad | bad_contact_reset_flatline | 188 | 429 | 1 | 0.49092 | 0.897411 | 0.839134 | 0.524815 | 1.07497 |
| bad | bad_dense_right_island | 782 | 429 | 1 | 24.5663 | 1.40646 | 1 | 23.7248 | 49.3254 |
| bad | bad_detector_template_disagree | 436 | 429 | 1 | 14.1547 | 1.27744 | 1 | 14.8305 | 24.0807 |
| bad | bad_highfreq_detail_noise | 55 | 428 | 1 | 23.836 | 1.49336 | 1 | 23.9726 | 49.2782 |
| bad | bad_low_qrs_visibility | 38 | 428 | 1 | 4.08208 | 1.1984 | 1 | 4.12888 | 5.59733 |
| bad | bad_other_boundary | 340 | 428 | 1 | 15.708 | 1.47821 | 1 | 15.7602 | 27.3254 |
| good | good_clean_core | 2542 | 750 | 1 | 1.33848 | 1.14713 | 0.993303 | 1.32113 | 1.81488 |
| good | good_isolated_low_purity | 861 | 750 | 1 | 1.27055 | 1.11649 | 0.973059 | 1.28362 | 1.96143 |
| good | good_mild_artifact_outlier | 3049 | 750 | 1 | 0.582995 | 0.888785 | 0.683742 | 0.702028 | 1.15866 |
| good | good_overlap_boundary | 5582 | 750 | 1 | 0.874115 | 1.09826 | 0.896475 | 0.898408 | 1.62727 |
| medium | medium_clean_core | 871 | 600 | 1 | 0.626089 | 0.941005 | 0.544709 | 0.669606 | 1.21651 |
| medium | medium_outlier_or_bad_boundary | 2376 | 600 | 1 | 0.55239 | 0.993407 | 0.800866 | 0.585983 | 1.2818 |
| medium | medium_overlap_boundary | 2000 | 600 | 1 | 0.488263 | 0.785156 | 0.598268 | 0.558034 | 1.18646 |
| medium | medium_visible_qrs_detail | 2123 | 600 | 1 | 0.831204 | 0.986485 | 0.849162 | 0.862066 | 1.27349 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.