# V29 PCA-Subtype Good-Guard Feature-Matched Protocol

- Generated: 2026-06-22 19:06:43
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v29_pca_goodguard_gm_featurematched\protocol_v29_pca_goodguard_pc3000_s20260622`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Good/medium target: BUT train+val robust PCA over waveform-computable features.
- Selection rule: subtype quota in PCA space plus nearest candidate matching; BUT test is not used.
- V29 addition: narrow high-amplitude/QRS-visible/low-flatline good guard modes.
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
| good | rms | 3.03507 | 2.25262 | -0.926867 | 0.926867 |
| good | amplitude_entropy | 0.758808 | 0.813403 | 0.797024 | 0.797024 |
| good | qrs_band_ratio | 9.71146 | 8.571 | -0.79203 | 0.79203 |
| good | mean_abs | 1.34355 | 1.17086 | -0.747034 | 0.747034 |
| good | flatline_ratio | 0.111289 | 0.166934 | 0.723958 | 0.723958 |
| good | band_30_45 | 0.0166204 | 0.0241233 | 0.720169 | 0.720169 |
| good | sqi_basSQI | 0.458062 | 0.52383 | 0.718314 | 0.718314 |
| good | template_corr | 0.883927 | 0.853814 | -0.714806 | 0.714806 |
| good | baseline_step | 0.295778 | 0.227254 | -0.629414 | 0.629414 |
| good | detector_agreement | 0.416667 | 0.666667 | 0.5 | 0.5 |
| medium | amplitude_entropy | 0.806738 | 0.817533 | 0.335527 | 0.335527 |
| good | band_15_30 | 0.223651 | 0.238956 | 0.28591 | 0.28591 |
| good | low_amp_ratio | 0.2424 | 0.2264 | -0.277778 | 0.277778 |
| medium | mean_abs | 1.11063 | 1.05369 | -0.247098 | 0.247098 |
| medium | rms | 1.97361 | 1.80392 | -0.193154 | 0.193154 |
| medium | non_qrs_diff_p95 | 2.48983 | 2.27401 | -0.171726 | 0.171726 |
| good | non_qrs_diff_p95 | 1.78239 | 1.59166 | -0.150761 | 0.150761 |
| medium | flatline_ratio | 0.0440352 | 0.040032 | -0.139758 | 0.139758 |
| medium | template_corr | 0.90111 | 0.889533 | -0.0906522 | 0.0906522 |
| medium | band_15_30 | 0.235968 | 0.241945 | 0.0779526 | 0.0779526 |

## Figures

- PCA target vs synthetic: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v29_pca_goodguard_gm_featurematched\v29_pca_goodguard_pca_target_vs_synthetic.png`
- PCA V23 vs V27: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v29_pca_goodguard_gm_featurematched\v29_pca_goodguard_pca_v23_vs_v27.png`
- Good feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v29_pca_goodguard_gm_featurematched\v29_pca_goodguard_good_feature_gap.png`
- Medium feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v29_pca_goodguard_gm_featurematched\v29_pca_goodguard_medium_feature_gap.png`