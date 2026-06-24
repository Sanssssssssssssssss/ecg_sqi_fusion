# v40_unified_extractor_anchor Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 03:59:15`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\protocol_v40_unified_extractor_anchor_pc90_s20260625`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_subtype_stratified_seed20260621`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v40_unified_extractor_anchor\v40_unified_extractor_anchor_but_keepoutlier_bad_subtype_waveforms.png`

## PTB Synthetic Counts

| split | class_name | display_subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 2 |
| test | bad | bad_contact_reset_flatline | 2 |
| test | bad | bad_dense_right_island | 2 |
| test | bad | bad_detector_template_disagree | 2 |
| test | bad | bad_highfreq_detail_noise | 2 |
| test | bad | bad_low_qrs_visibility | 2 |
| test | bad | bad_other_boundary | 2 |
| test | good | good_clean_core | 4 |
| test | good | good_isolated_low_purity | 4 |
| test | good | good_mild_artifact_outlier | 4 |
| test | good | good_overlap_boundary | 4 |
| test | medium | medium_clean_core | 2 |
| test | medium | medium_isolated_lowqrs | 2 |
| test | medium | medium_outlier_or_bad_boundary | 2 |
| test | medium | medium_overlap_boundary | 2 |
| test | medium | medium_visible_qrs_detail | 2 |
| train | bad | bad_baseline_wander_lowfreq | 9 |
| train | bad | bad_contact_reset_flatline | 9 |
| train | bad | bad_dense_right_island | 9 |
| train | bad | bad_detector_template_disagree | 9 |
| train | bad | bad_highfreq_detail_noise | 9 |
| train | bad | bad_low_qrs_visibility | 9 |
| train | bad | bad_other_boundary | 8 |
| train | good | good_clean_core | 16 |
| train | good | good_isolated_low_purity | 15 |
| train | good | good_mild_artifact_outlier | 15 |
| train | good | good_overlap_boundary | 16 |
| train | medium | medium_clean_core | 13 |
| train | medium | medium_isolated_lowqrs | 13 |
| train | medium | medium_outlier_or_bad_boundary | 13 |
| train | medium | medium_overlap_boundary | 13 |
| train | medium | medium_visible_qrs_detail | 13 |
| val | bad | bad_baseline_wander_lowfreq | 2 |
| val | bad | bad_contact_reset_flatline | 2 |
| val | bad | bad_dense_right_island | 2 |
| val | bad | bad_detector_template_disagree | 2 |
| val | bad | bad_highfreq_detail_noise | 2 |
| val | bad | bad_low_qrs_visibility | 2 |
| val | bad | bad_other_boundary | 2 |
| val | good | good_clean_core | 3 |
| val | good | good_isolated_low_purity | 3 |
| val | good | good_mild_artifact_outlier | 3 |
| val | good | good_overlap_boundary | 3 |
| val | medium | medium_clean_core | 3 |
| val | medium | medium_isolated_lowqrs | 3 |
| val | medium | medium_outlier_or_bad_boundary | 3 |
| val | medium | medium_overlap_boundary | 3 |
| val | medium | medium_visible_qrs_detail | 3 |

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
| bad | bad_baseline_wander_lowfreq | 62 | 13 | 1 | 3.87776 | 0.578457 | 1 | 115.739 | 7.0527 |
| bad | bad_contact_reset_flatline | 188 | 13 | 1 | 0.533002 | 0.880805 | 0.96 | 0.572884 | 0.93331 |
| bad | bad_dense_right_island | 2524 | 13 | 1 | 24.9946 | 0.455431 | 0.5 | 26.665 | 68.0983 |
| bad | bad_detector_template_disagree | 918 | 13 | 1 | 14.1656 | 0.455255 | 1 | 17.1383 | 29.0268 |
| bad | bad_highfreq_detail_noise | 55 | 13 | 1 | 23.6695 | 0.680919 | 1 | 24.7803 | 36.1398 |
| bad | bad_low_qrs_visibility | 38 | 13 | 1 | 5.18324 | 0.770148 | 1 | 5.84764 | 8.21733 |
| bad | bad_other_boundary | 340 | 12 | 1 | 14.467 | 0.473827 | 0.5 | 14.6586 | 22.5319 |
| good | good_clean_core | 2542 | 23 | 1 | 1.45656 | 0.911454 | 1 | 1.47288 | 2.05475 |
| good | good_isolated_low_purity | 861 | 22 | 1 | 1.60258 | 0.739025 | 0.996255 | 1.71836 | 2.40748 |
| good | good_mild_artifact_outlier | 3049 | 22 | 1 | 0.650324 | 0.99698 | 0.949562 | 0.825283 | 1.20031 |
| good | good_overlap_boundary | 5582 | 23 | 1 | 0.989329 | 1.05448 | 0.993627 | 1.02704 | 1.88975 |
| medium | medium_clean_core | 871 | 18 | 1 | 0.75529 | 0.860234 | 0.980124 | 0.782082 | 1.48333 |
| medium | medium_outlier_or_bad_boundary | 2376 | 18 | 1 | 0.64841 | 1.00681 | 0.973672 | 0.654653 | 1.72755 |
| medium | medium_overlap_boundary | 2000 | 18 | 1 | 0.581931 | 0.831293 | 0.970716 | 0.648479 | 1.20614 |
| medium | medium_visible_qrs_detail | 2123 | 18 | 1 | 0.879297 | 0.926664 | 0.984208 | 0.906697 | 1.58279 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.