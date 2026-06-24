# V37 Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 03:48:52`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\protocol_v37_subtype_balanced_pc3000_s20260623`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_subtype_stratified_seed20260621`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v37_subtype_balanced_distribution\v37_but_keepoutlier_bad_subtype_waveforms.png`

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
| test | bad | bad_dense_right_island | 631 |
| test | bad | bad_detector_template_disagree | 229 |
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
| train | bad | bad_dense_right_island | 2208 |
| train | bad | bad_detector_template_disagree | 803 |
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
| val | bad | bad_dense_right_island | 316 |
| val | bad | bad_detector_template_disagree | 115 |
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
| bad | bad_baseline_wander_lowfreq | 62 | 429 | 1 | 48.5036 | 1.47305 | 0.992718 | 48.46 | 106.95 |
| bad | bad_contact_reset_flatline | 188 | 429 | 1 | 8.27973 | 1.57034 | 0.971077 | 8.18232 | 13.5768 |
| bad | bad_dense_right_island | 2524 | 429 | 1 | 112.004 | 1.56906 | 1 | 109.705 | 221.532 |
| bad | bad_detector_template_disagree | 918 | 429 | 1 | 114.648 | 1.73132 | 1 | 114.174 | 223.408 |
| bad | bad_highfreq_detail_noise | 55 | 428 | 1 | 362.833 | 1.48815 | 1 | 1.6298e+06 | 674.575 |
| bad | bad_low_qrs_visibility | 38 | 428 | 1 | 72.4514 | 1.45683 | 1 | 72.1404 | 159.653 |
| bad | bad_other_boundary | 340 | 428 | 1 | 85.1242 | 1.79864 | 1 | 84.9331 | 144.38 |
| good | good_clean_core | 2542 | 750 | 1 | 108.488 | 1.63446 | 0.987005 | 667854 | 450.17 |
| good | good_isolated_low_purity | 861 | 750 | 1 | 95.1219 | 1.68394 | 0.998473 | 870923 | 435.585 |
| good | good_mild_artifact_outlier | 3049 | 750 | 1 | 92.2374 | 1.4225 | 0.99658 | 725911 | 369.051 |
| good | good_overlap_boundary | 5582 | 750 | 1 | 95.0293 | 1.55856 | 0.996909 | 725913 | 369.729 |
| medium | medium_clean_core | 871 | 600 | 1 | 22.5578 | 1.4183 | 1 | 22.3379 | 67.3783 |
| medium | medium_outlier_or_bad_boundary | 2376 | 600 | 1 | 7.39103 | 1.4521 | 0.946415 | 7.18109 | 14.3851 |
| medium | medium_overlap_boundary | 2000 | 600 | 1 | 13.9003 | 1.43648 | 0.98419 | 13.7114 | 39.8285 |
| medium | medium_visible_qrs_detail | 2123 | 600 | 1 | 18.8113 | 1.4131 | 1 | 18.474 | 50.9846 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.