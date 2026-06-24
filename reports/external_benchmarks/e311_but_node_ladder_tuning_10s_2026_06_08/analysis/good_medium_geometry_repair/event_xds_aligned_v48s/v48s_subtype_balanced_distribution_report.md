# v48s Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 05:28:29`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\protocol_v48s_pc300_s20260636`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v48s\v48s_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 43 | 1 | 3.43662 | 1.80337 | 1 | 3.36351 | 5.88634 |
| bad | bad_contact_reset_flatline | 188 | 43 | 1 | 0.512568 | 0.882472 | 0.912281 | 0.571115 | 1.10239 |
| bad | bad_dense_right_island | 782 | 43 | 1 | 30.8882 | 0.430008 | 1 | 29.1299 | 66.0698 |
| bad | bad_detector_template_disagree | 436 | 43 | 1 | 18.5955 | 0.431286 | 1 | 19.506 | 33.4923 |
| bad | bad_highfreq_detail_noise | 55 | 43 | 1 | 24.6252 | 1.68492 | 1 | 25.1022 | 44.4923 |
| bad | bad_low_qrs_visibility | 38 | 43 | 1 | 3.96425 | 1.28448 | 1 | 3.93937 | 6.23048 |
| bad | bad_other_boundary | 340 | 42 | 1 | 18.5641 | 0.449515 | 1 | 18.3845 | 34.5132 |
| good | good_clean_core | 2542 | 75 | 1 | 1.43734 | 1.00698 | 0.999582 | 1.42789 | 1.92808 |
| good | good_isolated_low_purity | 861 | 75 | 1 | 1.47462 | 0.751951 | 0.988971 | 1.70688 | 1.91146 |
| good | good_mild_artifact_outlier | 3049 | 75 | 1 | 0.622563 | 0.993048 | 0.877338 | 0.771844 | 1.18245 |
| good | good_overlap_boundary | 5582 | 75 | 1 | 0.960388 | 1.09708 | 0.981271 | 0.993957 | 1.61672 |
| medium | medium_clean_core | 871 | 60 | 1 | 0.702686 | 0.949065 | 0.861596 | 0.743269 | 1.52763 |
| medium | medium_outlier_or_bad_boundary | 2376 | 60 | 1 | 0.690025 | 1.09304 | 0.941871 | 0.705188 | 1.58797 |
| medium | medium_overlap_boundary | 2000 | 60 | 1 | 0.526702 | 0.906272 | 0.897019 | 0.636254 | 1.21912 |
| medium | medium_visible_qrs_detail | 2123 | 60 | 1 | 0.872465 | 0.992254 | 0.980213 | 0.931635 | 1.45309 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.