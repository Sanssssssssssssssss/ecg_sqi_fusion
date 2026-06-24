# V28 PCA-Anchor Good/Medium Feature-Matched Protocol

- Generated: 2026-06-22 18:57:49
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v28_pca_anchor_gm_featurematched\protocol_v28_pca_anchor_pc3000_s20260622`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Good/medium target: BUT train+val robust PCA over waveform-computable features.
- Selection rule: sampled BUT PCA anchors plus nearest candidate coverage; BUT test is not used.
- PCA explained variance: PC1 0.378, PC2 0.202, PC3 0.136.

## Split Counts

| split | class_name | n |
| --- | --- | --- |
| test | bad | 450 |
| test | good | 452 |
| test | medium | 506 |
| train | bad | 2100 |
| train | good | 2105 |
| train | medium | 2014 |
| val | bad | 450 |
| val | good | 443 |
| val | medium | 480 |

## BUT Train/Val PCA Subtypes

| class_name | subtype_id | subtype_name | target_count | target_share | pc1_center | pc2_center | pc3_center |
| --- | --- | --- | --- | --- | --- | --- | --- |
| good | good_2_good_core_2 | good_core_2 | 4116 | 0.402582 | 0.715647 | -1.46481 | 0.198799 |
| good | good_1_good_core_1 | good_core_1 | 3221 | 0.315043 | -0.0229662 | 0.4994 | -0.56905 |
| good | good_3_good_pc3_morph_tail | good_pc3_morph_tail | 1796 | 0.175665 | 1.52925 | 0.203388 | 1.105 |
| good | good_0_good_low_pc1_shell | good_low_pc1_shell | 1091 | 0.10671 | -2.30602 | -0.928888 | -1.75259 |
| medium | medium_4_medium_high_pc2_detail | medium_high_pc2_detail | 1737 | 0.414756 | 0.199851 | 1.54225 | 0.125485 |
| medium | medium_2_medium_high_pc2_detail | medium_high_pc2_detail | 1057 | 0.252388 | -2.24748 | 0.965908 | -0.509438 |
| medium | medium_3_medium_high_pc2_detail | medium_high_pc2_detail | 658 | 0.157116 | -1.30999 | 2.21592 | 1.11593 |
| medium | medium_1_medium_goodlike_low_pc1 | medium_goodlike_low_pc1 | 500 | 0.119389 | -4.80457 | -0.277181 | 1.35409 |
| medium | medium_0_medium_goodlike_low_pc1 | medium_goodlike_low_pc1 | 236 | 0.0563515 | -7.4347 | -2.03741 | 3.86572 |

## Largest Remaining Feature Gaps

| class_name | feature | but_median | ptb_median | robust_z_gap | abs_gap |
| --- | --- | --- | --- | --- | --- |
| good | flatline_ratio | 0.111289 | 0.190552 | 1.03125 | 1.03125 |
| good | rms | 3.03507 | 2.26411 | -0.913265 | 0.913265 |
| good | amplitude_entropy | 0.758808 | 0.80945 | 0.739315 | 0.739315 |
| good | sqi_basSQI | 0.458062 | 0.521346 | 0.691176 | 0.691176 |
| good | mean_abs | 1.34355 | 1.18595 | -0.681732 | 0.681732 |
| good | qrs_band_ratio | 9.71146 | 8.75404 | -0.664915 | 0.664915 |
| good | baseline_step | 0.295778 | 0.229528 | -0.608522 | 0.608522 |
| good | band_30_45 | 0.0166204 | 0.0222828 | 0.543508 | 0.543508 |
| good | detector_agreement | 0.416667 | 0.666667 | 0.5 | 0.5 |
| good | template_corr | 0.883927 | 0.869664 | -0.338565 | 0.338565 |
| medium | amplitude_entropy | 0.806738 | 0.816562 | 0.305329 | 0.305329 |
| good | non_qrs_diff_p95 | 1.78239 | 1.5319 | -0.197996 | 0.197996 |
| medium | non_qrs_diff_p95 | 2.48983 | 2.25894 | -0.183715 | 0.183715 |
| medium | mean_abs | 1.11063 | 1.07805 | -0.141358 | 0.141358 |
| medium | flatline_ratio | 0.0440352 | 0.040032 | -0.139758 | 0.139758 |
| good | band_15_30 | 0.223651 | 0.217469 | -0.11548 | 0.11548 |
| medium | rms | 1.97361 | 1.90107 | -0.0825624 | 0.0825624 |
| medium | band_15_30 | 0.235968 | 0.240465 | 0.0586569 | 0.0586569 |
| medium | template_corr | 0.90111 | 0.895195 | -0.0463216 | 0.0463216 |
| medium | qrs_band_ratio | 8.10978 | 8.28059 | 0.0444247 | 0.0444247 |

## Figures

- PCA target vs synthetic: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v28_pca_anchor_gm_featurematched\v28_pca_anchor_pca_target_vs_synthetic.png`
- PCA V23 vs V27: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v28_pca_anchor_gm_featurematched\v28_pca_anchor_pca_v23_vs_v27.png`
- Good feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v28_pca_anchor_gm_featurematched\v28_pca_anchor_good_feature_gap.png`
- Medium feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v28_pca_anchor_gm_featurematched\v28_pca_anchor_medium_feature_gap.png`