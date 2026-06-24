# v44raw Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 05:12:06`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\protocol_v44raw_pc3000_s20260632`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v44raw\v44raw_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 429 | 1 | 4.0684 | 1.30229 | 0.982927 | 4.19896 | 5.58585 |
| bad | bad_contact_reset_flatline | 188 | 429 | 1 | 0.490287 | 0.881289 | 0.780619 | 0.521184 | 1.10605 |
| bad | bad_dense_right_island | 782 | 429 | 1 | 35.0721 | 1.32133 | 1 | 36.2284 | 105.24 |
| bad | bad_detector_template_disagree | 436 | 429 | 1 | 15.69 | 1.26486 | 1 | 17.1686 | 27.3401 |
| bad | bad_highfreq_detail_noise | 55 | 428 | 1 | 25.6298 | 1.47593 | 1 | 25.7831 | 45.457 |
| bad | bad_low_qrs_visibility | 38 | 428 | 1 | 5.58464 | 1.37886 | 1 | 5.9507 | 9.62805 |
| bad | bad_other_boundary | 340 | 428 | 1 | 19.6344 | 1.42809 | 1 | 20.8166 | 32.3739 |
| good | good_clean_core | 2542 | 750 | 1 | 1.33706 | 1.14859 | 0.997905 | 1.30915 | 1.81508 |
| good | good_isolated_low_purity | 861 | 750 | 1 | 1.31804 | 1.1359 | 0.977895 | 1.30161 | 1.78962 |
| good | good_mild_artifact_outlier | 3049 | 750 | 1 | 0.576607 | 0.875449 | 0.655325 | 0.690512 | 1.24257 |
| good | good_overlap_boundary | 5582 | 750 | 1 | 0.857836 | 1.07891 | 0.906188 | 0.890715 | 1.6315 |
| medium | medium_clean_core | 871 | 600 | 1 | 0.631039 | 0.929959 | 0.552919 | 0.667582 | 1.4109 |
| medium | medium_outlier_or_bad_boundary | 2376 | 600 | 1 | 0.536217 | 0.984252 | 0.765766 | 0.571856 | 1.25699 |
| medium | medium_overlap_boundary | 2000 | 600 | 1 | 0.478906 | 0.785171 | 0.615929 | 0.549896 | 1.04576 |
| medium | medium_visible_qrs_detail | 2123 | 600 | 1 | 0.838121 | 0.985583 | 0.841587 | 0.849642 | 1.38993 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.