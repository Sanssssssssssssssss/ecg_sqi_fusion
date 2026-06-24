# v52s Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 08:35:21`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\protocol_v52s_pc330_s20260641`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v52s\v52s_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 47 | 1 | 3.37648 | 1.80788 | 1 | 3.33625 | 5.681 |
| bad | bad_contact_reset_flatline | 188 | 47 | 1 | 0.517349 | 0.92019 | 0.97076 | 0.562571 | 1.1642 |
| bad | bad_dense_right_island | 782 | 48 | 1 | 31.4125 | 0.431715 | 1 | 30.3985 | 79.1855 |
| bad | bad_detector_template_disagree | 436 | 47 | 1 | 18.0077 | 0.433383 | 1 | 18.6099 | 37.2715 |
| bad | bad_highfreq_detail_noise | 55 | 47 | 1 | 25.1631 | 1.60947 | 1 | 25.5992 | 36.7057 |
| bad | bad_low_qrs_visibility | 38 | 47 | 1 | 4.0275 | 1.26276 | 1 | 4.01547 | 5.9611 |
| bad | bad_other_boundary | 340 | 47 | 1 | 17.6491 | 0.454305 | 1 | 18.0068 | 31.3714 |
| good | good_clean_core | 2537 | 66 | 1 | 1.4625 | 1.0068 | 1 | 1.45836 | 1.99773 |
| good | good_hard_baseline_lowqrs | 2830 | 66 | 1 | 0.710322 | 1.01122 | 0.883992 | 0.778473 | 1.42752 |
| good | good_isolated_low_purity | 791 | 66 | 1 | 1.56163 | 0.745164 | 1 | 1.81239 | 1.95161 |
| good | good_mild_artifact_outlier | 802 | 66 | 1 | 1.20612 | 1.04123 | 0.982574 | 1.2303 | 1.47178 |
| good | good_overlap_boundary | 5074 | 66 | 1 | 1.0213 | 1.09806 | 0.987872 | 1.03968 | 1.63213 |
| medium | medium_clean_core | 708 | 55 | 1 | 0.637907 | 0.95843 | 0.887692 | 0.703933 | 1.48253 |
| medium | medium_hard_baseline_lowqrs | 1920 | 55 | 1 | 0.609289 | 0.938067 | 0.82287 | 0.703428 | 1.17414 |
| medium | medium_outlier_or_bad_boundary | 1081 | 55 | 1 | 1.031 | 1.01796 | 0.974104 | 1.02567 | 1.78238 |
| medium | medium_overlap_boundary | 1552 | 55 | 1 | 0.524392 | 0.841596 | 0.915679 | 0.622739 | 1.16256 |
| medium | medium_visible_qrs_detail | 2109 | 55 | 1 | 0.891045 | 1.00785 | 0.974438 | 0.940622 | 1.34622 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.