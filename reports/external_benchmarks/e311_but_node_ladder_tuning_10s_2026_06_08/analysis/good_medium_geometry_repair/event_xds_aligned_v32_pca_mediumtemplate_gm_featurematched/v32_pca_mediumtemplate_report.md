# V32 PCA-Subtype Medium Template Feature-Matched Protocol

- Generated: 2026-06-22 19:50:17
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v32_pca_mediumtemplate_gm_featurematched\protocol_v32_pca_mediumtemplate_pc3000_s20260622`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Good/medium target: BUT train+val robust PCA over waveform-computable features.
- Selection rule: subtype quota in PCA space plus nearest candidate matching; BUT test is not used.
- V32 addition: 28% visible-QRS medium template/detail guard quota using BUT train+val waveform-computable targets.
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
| good | flatline_ratio | 0.111289 | 0.185749 | 0.96875 | 0.96875 |
| good | rms | 3.03507 | 2.3077 | -0.861631 | 0.861631 |
| good | amplitude_entropy | 0.758808 | 0.813366 | 0.796487 | 0.796487 |
| good | qrs_band_ratio | 9.71146 | 8.60142 | -0.770907 | 0.770907 |
| good | band_30_45 | 0.0166204 | 0.0238343 | 0.692428 | 0.692428 |
| good | sqi_basSQI | 0.458062 | 0.520583 | 0.682842 | 0.682842 |
| good | mean_abs | 1.34355 | 1.1977 | -0.630941 | 0.630941 |
| good | template_corr | 0.883927 | 0.858062 | -0.613974 | 0.613974 |
| good | baseline_step | 0.295778 | 0.230231 | -0.602065 | 0.602065 |
| medium | amplitude_entropy | 0.806738 | 0.818218 | 0.356806 | 0.356806 |
| good | detector_agreement | 0.416667 | 0.583333 | 0.333333 | 0.333333 |
| medium | mean_abs | 1.11063 | 1.0417 | -0.299108 | 0.299108 |
| good | band_15_30 | 0.223651 | 0.238157 | 0.271 | 0.271 |
| medium | non_qrs_diff_p95 | 2.48983 | 2.17114 | -0.253579 | 0.253579 |
| medium | rms | 1.97361 | 1.75572 | -0.24802 | 0.24802 |
| medium | sqi_basSQI | 0.596305 | 0.622541 | 0.175357 | 0.175357 |
| medium | baseline_step | 0.169249 | 0.15158 | -0.170402 | 0.170402 |
| good | non_qrs_diff_p95 | 1.78239 | 1.56813 | -0.169357 | 0.169357 |
| good | low_amp_ratio | 0.2424 | 0.2336 | -0.152778 | 0.152778 |
| medium | template_corr | 0.90111 | 0.886594 | -0.11367 | 0.11367 |

## Figures

- PCA target vs synthetic: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v32_pca_mediumtemplate_gm_featurematched\v32_pca_mediumtemplate_pca_target_vs_synthetic.png`
- PCA V23 vs V27: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v32_pca_mediumtemplate_gm_featurematched\v32_pca_mediumtemplate_pca_v23_vs_v27.png`
- Good feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v32_pca_mediumtemplate_gm_featurematched\v32_pca_mediumtemplate_good_feature_gap.png`
- Medium feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v32_pca_mediumtemplate_gm_featurematched\v32_pca_mediumtemplate_medium_feature_gap.png`