# V37 Subtype-Balanced PTB Synthetic Distribution Report

- Created: `2026-06-23 03:51:58`
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\protocol_v38_subtype_balanced_ampcal_pc90_s20260624`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_subtype_stratified_seed20260621`
- Core rule: subtype-balanced PTB training rows; BUT train+val empirical feature distributions as selection targets; BUT test is audit-only.

## Key Outputs

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_shared_pca_but_vs_ptb.png`
- Feature gap heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_subtype_feature_gap_heatmap.png`
- PTB waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_ptb_synthetic_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_ptb_synthetic_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_ptb_synthetic_bad_subtype_waveforms.png`
- BUT waveform sheets: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_but_keepoutlier_good_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_but_keepoutlier_medium_subtype_waveforms.png`, `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v38_subtype_balanced_ampcal\v37_but_keepoutlier_bad_subtype_waveforms.png`

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
| bad | bad_baseline_wander_lowfreq | 62 | 13 | 1 | 68.354 | 0.507262 | 1 | 221384 | 179.253 |
| bad | bad_contact_reset_flatline | 188 | 13 | 1 | 27.947 | 0.507722 | 1 | 232296 | 123.898 |
| bad | bad_dense_right_island | 2524 | 13 | 1 | 114.793 | 0.456278 | 0.5 | 87224.1 | 231.745 |
| bad | bad_detector_template_disagree | 918 | 13 | 1 | 118.934 | 0.458798 | 0.5 | 226610 | 303.246 |
| bad | bad_highfreq_detail_noise | 55 | 13 | 1 | 342.115 | 0.609139 | 1 | 1.70603e+06 | 783.512 |
| bad | bad_low_qrs_visibility | 38 | 13 | 1 | 73.1646 | 0.735586 | 1 | 170785 | 205.064 |
| bad | bad_other_boundary | 340 | 12 | 1 | 96.2478 | 0.469192 | 1 | 101712 | 170.323 |
| good | good_clean_core | 2542 | 23 | 1 | 148.453 | 0.454761 | 0.5 | 1.13824e+06 | 564.62 |
| good | good_isolated_low_purity | 861 | 22 | 1 | 147.128 | 0.459426 | 1 | 1.25474e+06 | 759.936 |
| good | good_mild_artifact_outlier | 3049 | 22 | 0.938776 | 153.012 | 0.470462 | 0.5 | 1.33712e+06 | 706.21 |
| good | good_overlap_boundary | 5582 | 23 | 1 | 154.234 | 0.467189 | 0.5 | 1.41116e+06 | 746.109 |
| medium | medium_clean_core | 871 | 18 | 1 | 93.7815 | 0.453225 | 0.5 | 588002 | 374.703 |
| medium | medium_outlier_or_bad_boundary | 2376 | 18 | 1 | 88.4445 | 0.497435 | 1 | 630089 | 308.163 |
| medium | medium_overlap_boundary | 2000 | 18 | 1 | 54.2425 | 0.461949 | 0.5 | 493918 | 277.667 |
| medium | medium_visible_qrs_detail | 2123 | 18 | 1 | 89.9019 | 0.444118 | 0.5 | 586797 | 317.045 |

## Interpretation

- If PCA looks closer but discriminative AUC remains high, the waveform morphology still differs despite feature-space overlap.
- If rare bad subtype metrics remain high, the candidate generator did not produce enough natural-looking morphology for that mechanism.
- Training uses balanced subtype exposure; natural-prior counts are reported separately so dense right-island does not hide rare bad mechanisms.