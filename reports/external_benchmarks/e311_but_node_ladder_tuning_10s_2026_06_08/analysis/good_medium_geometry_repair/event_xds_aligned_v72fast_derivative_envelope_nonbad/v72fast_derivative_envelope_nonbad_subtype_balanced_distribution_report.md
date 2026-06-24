# v72fast_derivative_envelope_nonbad Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 19:58:01`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\protocol_v72fast_derivative_envelope_nonbad_pc1500_s20260673`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v72fast_derivative_envelope_nonbad\v72fast_derivative_envelope_nonbad_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 214 | 1 | 3.81597 | 1.29835 | 1 | 3.81148 | 6.57742 |
| bad | bad_contact_reset_flatline | 188 | 214 | 1 | 0.518838 | 0.831419 | 0.777087 | 0.547898 | 1.26702 |
| bad | bad_dense_right_island | 782 | 215 | 1 | 9.88897 | 0.895001 | 1 | 9.94915 | 12.4651 |
| bad | bad_detector_template_disagree | 436 | 215 | 1 | 9.12242 | 1.22725 | 1 | 9.24613 | 13.1125 |
| bad | bad_highfreq_detail_noise | 55 | 214 | 1 | 24.5159 | 1.59503 | 1 | 24.493 | 40.7318 |
| bad | bad_low_qrs_visibility | 38 | 214 | 1 | 3.60283 | 1.48535 | 1 | 4.39286 | 7.84112 |
| bad | bad_other_boundary | 340 | 214 | 1 | 9.1697 | 1.56803 | 1 | 9.22637 | 13.0606 |
| good | good_clean_core | 2537 | 300 | 1 | 13.2336 | 1.62056 | 1 | 12.9431 | 45.8107 |
| good | good_hard_baseline_lowqrs | 2830 | 300 | 1 | 1.37976 | 0.990451 | 0.92508 | 1.33681 | 1.41039 |
| good | good_isolated_low_purity | 791 | 300 | 1 | 4.48669 | 1.45547 | 1 | 4.57334 | 6.39434 |
| good | good_mild_artifact_outlier | 802 | 300 | 1 | 2.02729 | 1.21639 | 1 | 2.08566 | 2.14153 |
| good | good_overlap_boundary | 5074 | 300 | 1 | 5.28325 | 1.47862 | 1 | 5.23488 | 14.1917 |
| medium | medium_clean_core | 708 | 250 | 1 | 0.986328 | 1.05489 | 0.868907 | 0.989253 | 1.428 |
| medium | medium_hard_baseline_lowqrs | 1920 | 250 | 1 | 0.692758 | 0.919064 | 0.66246 | 0.78448 | 1.4071 |
| medium | medium_outlier_or_bad_boundary | 1081 | 250 | 1 | 1.38604 | 1.11064 | 0.957916 | 1.23622 | 1.91009 |
| medium | medium_overlap_boundary | 1552 | 250 | 1 | 0.911999 | 0.851813 | 0.869263 | 0.885455 | 1.19963 |
| medium | medium_visible_qrs_detail | 2109 | 250 | 1 | 1.79286 | 1.08615 | 0.999008 | 1.75222 | 1.89255 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.