# v44d Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 05:04:14`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\protocol_v44d_pc300_s20260631`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44d\v44d_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 43 | 1 | 4.68575 | 1.43142 | 1 | 4.70448 | 6.5932 |
| bad | bad_contact_reset_flatline | 188 | 43 | 1 | 0.525034 | 0.857721 | 0.94152 | 0.577446 | 1.24143 |
| bad | bad_dense_right_island | 782 | 43 | 1 | 49.624 | 0.42941 | 1 | 49.783 | 153.249 |
| bad | bad_detector_template_disagree | 436 | 43 | 1 | 20.5845 | 0.431197 | 1 | 25.2277 | 46.6029 |
| bad | bad_highfreq_detail_noise | 55 | 43 | 1 | 28.7174 | 1.37348 | 1 | 31.9985 | 57.7295 |
| bad | bad_low_qrs_visibility | 38 | 43 | 1 | 6.77373 | 1.31286 | 1 | 6.96806 | 11.1422 |
| bad | bad_other_boundary | 340 | 42 | 1 | 26.2512 | 0.449512 | 1 | 28.0849 | 52.8698 |
| good | good_clean_core | 2542 | 75 | 1 | 1.42844 | 1.00909 | 1 | 1.40741 | 2.13305 |
| good | good_isolated_low_purity | 861 | 75 | 1 | 1.56292 | 0.761418 | 0.989011 | 1.77777 | 2.09223 |
| good | good_mild_artifact_outlier | 3049 | 75 | 1 | 0.604629 | 0.993618 | 0.830483 | 0.770413 | 1.13001 |
| good | good_overlap_boundary | 5582 | 75 | 1 | 0.947928 | 1.06255 | 0.975859 | 0.964757 | 1.71037 |
| medium | medium_clean_core | 871 | 60 | 1 | 0.695401 | 0.946037 | 0.88 | 0.734163 | 1.49475 |
| medium | medium_outlier_or_bad_boundary | 2376 | 60 | 1 | 0.594162 | 1.06319 | 0.932334 | 0.632121 | 1.44365 |
| medium | medium_overlap_boundary | 2000 | 60 | 1 | 0.517247 | 0.907196 | 0.877507 | 0.623799 | 0.950622 |
| medium | medium_visible_qrs_detail | 2123 | 60 | 1 | 0.919045 | 1.02006 | 0.975647 | 0.946987 | 1.6338 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.