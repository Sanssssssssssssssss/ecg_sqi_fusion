# v42_unified_extractor_legacybad Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 04:19:46`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\protocol_v42_unified_extractor_legacybad_pc3000_s20260627`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_subtype_stratified_seed20260621`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v42_unified_extractor_legacybad\v42_unified_extractor_legacybad_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 429 | 1 | 3.90466 | 1.31481 | 0.980488 | 4.01226 | 5.95708 |
| bad | bad_contact_reset_flatline | 188 | 429 | 1 | 0.490211 | 0.929746 | 0.803861 | 0.526917 | 1.04658 |
| bad | bad_dense_right_island | 2524 | 429 | 1 | 16.8419 | 1.424 | 0.952917 | 17.0911 | 31.0273 |
| bad | bad_detector_template_disagree | 918 | 429 | 1 | 12.1436 | 1.4206 | 0.943466 | 11.1443 | 18.4236 |
| bad | bad_highfreq_detail_noise | 55 | 428 | 1 | 23.395 | 1.50758 | 1 | 23.6098 | 45.3419 |
| bad | bad_low_qrs_visibility | 38 | 428 | 1 | 4.95648 | 1.3188 | 1 | 5.01602 | 8.38519 |
| bad | bad_other_boundary | 340 | 428 | 1 | 13.0393 | 1.63723 | 0.978936 | 13.0861 | 18.6226 |
| good | good_clean_core | 2542 | 750 | 1 | 1.35979 | 1.16493 | 0.99303 | 1.33466 | 2.17103 |
| good | good_isolated_low_purity | 861 | 750 | 1 | 1.29663 | 1.13543 | 0.959967 | 1.26698 | 1.88756 |
| good | good_mild_artifact_outlier | 3049 | 750 | 1 | 0.591418 | 0.923087 | 0.66919 | 0.700812 | 1.34495 |
| good | good_overlap_boundary | 5582 | 750 | 1 | 0.857389 | 1.09555 | 0.885125 | 0.900966 | 1.80151 |
| medium | medium_clean_core | 871 | 600 | 1 | 0.648665 | 0.941869 | 0.602523 | 0.684486 | 1.70688 |
| medium | medium_outlier_or_bad_boundary | 2376 | 600 | 1 | 0.51107 | 0.992887 | 0.694845 | 0.554922 | 0.953315 |
| medium | medium_overlap_boundary | 2000 | 600 | 1 | 0.496508 | 0.804997 | 0.60596 | 0.537906 | 1.01512 |
| medium | medium_visible_qrs_detail | 2123 | 600 | 1 | 0.830778 | 0.976377 | 0.803637 | 0.830432 | 1.44849 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.