# v39_subtype_anchor_mechanism Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 03:55:50`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\protocol_v39_subtype_anchor_mechanism_pc90_s20260625`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_subtype_stratified_seed20260621`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v39_subtype_anchor_mechanism\v39_subtype_anchor_mechanism_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 13 | 1 | 126.259 | 0.507046 | 1 | 581279 | 304.067 |
| bad | bad_contact_reset_flatline | 188 | 13 | 1 | 45.4507 | 0.520795 | 1 | 441352 | 267.174 |
| bad | bad_dense_right_island | 2524 | 13 | 1 | 154.406 | 0.456745 | 0.5 | 495849 | 358.719 |
| bad | bad_detector_template_disagree | 918 | 13 | 1 | 138.374 | 0.458894 | 0.5 | 116262 | 235.299 |
| bad | bad_highfreq_detail_noise | 55 | 13 | 1 | 359.987 | 0.609139 | 1 | 2.04292e+06 | 937.056 |
| bad | bad_low_qrs_visibility | 38 | 13 | 1 | 115.868 | 0.734783 | 1 | 413119 | 211.381 |
| bad | bad_other_boundary | 340 | 12 | 1 | 96.7651 | 0.469192 | 1 | 101713 | 158.8 |
| good | good_clean_core | 2542 | 23 | 1 | 148.129 | 0.457431 | 0.5 | 1.1121e+06 | 518.207 |
| good | good_isolated_low_purity | 861 | 22 | 1 | 156.457 | 0.458474 | 1 | 1.46768e+06 | 825.169 |
| good | good_mild_artifact_outlier | 3049 | 22 | 1 | 168.257 | 0.481269 | 0.5 | 1.53601e+06 | 709.163 |
| good | good_overlap_boundary | 5582 | 23 | 1 | 136.537 | 0.472569 | 0.5 | 1.23404e+06 | 554.583 |
| medium | medium_clean_core | 871 | 18 | 1 | 84.6796 | 0.455345 | 0.5 | 580744 | 331.282 |
| medium | medium_outlier_or_bad_boundary | 2376 | 18 | 1 | 55.1527 | 0.559621 | 0.5 | 432641 | 219.34 |
| medium | medium_overlap_boundary | 2000 | 18 | 1 | 92.3416 | 0.463541 | 0.5 | 869428 | 403.267 |
| medium | medium_visible_qrs_detail | 2123 | 18 | 1 | 98.394 | 0.442245 | 0.5 | 743339 | 325.202 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.