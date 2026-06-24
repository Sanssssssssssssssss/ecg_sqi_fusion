# V35 PCA-Anchor Oracle All-BUT Good/Medium Feature-Matched Protocol

- Generated: 2026-06-22 20:40:10
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v35_pca_anchor_oracle_allbut_gm_featurematched\protocol_v35_pca_anchor_oracle_allbut_pc3000_s20260622`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Diagnostic only: target uses BUT train+val+test robust PCA over waveform-computable features.
- Selection rule: sampled BUT PCA anchors plus nearest candidate coverage; BUT test features are used, so this is not a formal result.
- PCA explained variance: PC1 0.363, PC2 0.205, PC3 0.124.

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
| good | good_3_good_pc3_morph_tail | good_pc3_morph_tail | 3981 | 0.35456 | 0.897391 | -1.60668 | 0.474001 |
| good | good_1_good_core_1 | good_core_1 | 3843 | 0.342269 | 0.0836897 | 0.00991666 | -0.469636 |
| good | good_2_good_pc3_morph_tail | good_pc3_morph_tail | 2254 | 0.200748 | 1.48567 | 0.0614068 | 1.03776 |
| good | good_0_good_low_pc1_shell | good_low_pc1_shell | 1150 | 0.102423 | -1.84461 | -1.31186 | -1.52012 |
| medium | medium_4_medium_high_pc2_detail | medium_high_pc2_detail | 1756 | 0.280287 | 0.317667 | 1.19677 | -0.0276434 |
| medium | medium_3_medium_high_pc2_detail | medium_high_pc2_detail | 1647 | 0.262889 | -0.597402 | 2.19473 | 0.645875 |
| medium | medium_2_medium_high_pc2_detail | medium_high_pc2_detail | 1595 | 0.254589 | -1.92828 | 0.593729 | -0.483363 |
| medium | medium_1_medium_goodlike_low_pc1 | medium_goodlike_low_pc1 | 912 | 0.145571 | -3.99248 | -0.242663 | 0.783225 |
| medium | medium_0_medium_goodlike_low_pc1 | medium_goodlike_low_pc1 | 355 | 0.056664 | -6.78758 | -1.55406 | 3.09475 |

## Largest Remaining Feature Gaps

| class_name | feature | but_median | ptb_median | robust_z_gap | abs_gap |
| --- | --- | --- | --- | --- | --- |
| good | flatline_ratio | 0.115292 | 0.192954 | 0.989796 | 0.989796 |
| good | rms | 3.00666 | 2.25823 | -0.877308 | 0.877308 |
| good | qrs_band_ratio | 9.84357 | 8.76475 | -0.77606 | 0.77606 |
| good | sqi_basSQI | 0.456514 | 0.522575 | 0.718329 | 0.718329 |
| good | amplitude_entropy | 0.760638 | 0.807972 | 0.71037 | 0.71037 |
| good | baseline_step | 0.297628 | 0.2284 | -0.627194 | 0.627194 |
| good | mean_abs | 1.33224 | 1.18402 | -0.621413 | 0.621413 |
| good | band_30_45 | 0.0173968 | 0.0217699 | 0.421917 | 0.421917 |
| medium | amplitude_entropy | 0.802852 | 0.815368 | 0.383563 | 0.383563 |
| good | detector_agreement | 0.5 | 0.666667 | 0.333333 | 0.333333 |
| good | template_corr | 0.883089 | 0.870742 | -0.287841 | 0.287841 |
| medium | detector_agreement | 0.5 | 0.583333 | 0.25 | 0.25 |
| good | band_15_30 | 0.221663 | 0.211902 | -0.174522 | 0.174522 |
| good | non_qrs_diff_p95 | 1.67403 | 1.501 | -0.132622 | 0.132622 |
| medium | band_30_45 | 0.026975 | 0.0245749 | -0.117118 | 0.117118 |
| medium | low_amp_ratio | 0.208 | 0.2112 | 0.111111 | 0.111111 |
| medium | flatline_ratio | 0.0440352 | 0.0408327 | -0.102094 | 0.102094 |
| medium | band_15_30 | 0.231507 | 0.241024 | 0.101141 | 0.101141 |
| medium | non_qrs_diff_p95 | 2.40152 | 2.30229 | -0.066197 | 0.066197 |
| medium | rms | 1.88391 | 1.841 | -0.0531545 | 0.0531545 |

## Figures

- PCA target vs synthetic: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v35_pca_anchor_oracle_allbut_gm_featurematched\v35_pca_anchor_oracle_allbut_pca_target_vs_synthetic.png`
- PCA V23 vs V27: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v35_pca_anchor_oracle_allbut_gm_featurematched\v35_pca_anchor_oracle_allbut_pca_v23_vs_v27.png`
- Good feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v35_pca_anchor_oracle_allbut_gm_featurematched\v35_pca_anchor_oracle_allbut_good_feature_gap.png`
- Medium feature gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v35_pca_anchor_oracle_allbut_gm_featurematched\v35_pca_anchor_oracle_allbut_medium_feature_gap.png`