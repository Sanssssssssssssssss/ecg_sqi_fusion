# V23 Good/Medium Feature-Matched Protocol

- Generated: 2026-06-21 02:31:22
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\protocol_v23_pc3000_s20260621`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Good/medium rows: PTB-only candidates selected against BUT train-split waveform-computable feature medians/IQRs.
- Generator profile: `v23_quantile`.
- No BUT waveform is copied; BUT is used as a distribution target.

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

## Largest Remaining Good/Medium Feature Gaps

| class_name | feature | but_median | ptb_median | robust_z_gap | abs_gap |
| --- | --- | --- | --- | --- | --- |
| good | rms | 3.00666 | 2.18855 | -0.958983 | 0.958983 |
| good | qrs_band_ratio | 9.84357 | 8.71627 | -0.810937 | 0.810937 |
| good | amplitude_entropy | 0.760638 | 0.81457 | 0.809383 | 0.809383 |
| good | mean_abs | 1.33224 | 1.14698 | -0.776655 | 0.776655 |
| good | sqi_basSQI | 0.456514 | 0.522263 | 0.714933 | 0.714933 |
| good | baseline_step | 0.297628 | 0.228686 | -0.624603 | 0.624603 |
| medium | amplitude_entropy | 0.802852 | 0.815881 | 0.399273 | 0.399273 |
| good | template_corr | 0.883089 | 0.868587 | -0.338096 | 0.338096 |
| good | detector_agreement | 0.5 | 0.666667 | 0.333333 | 0.333333 |
| good | band_15_30 | 0.221663 | 0.204468 | -0.307445 | 0.307445 |
| good | flatline_ratio | 0.115292 | 0.139311 | 0.306122 | 0.306122 |
| medium | template_corr | 0.887182 | 0.928448 | 0.260161 | 0.260161 |
| medium | detector_agreement | 0.5 | 0.583333 | 0.25 | 0.25 |
| medium | band_30_45 | 0.026975 | 0.022203 | -0.232856 | 0.232856 |
| good | band_30_45 | 0.0173968 | 0.0196251 | 0.214983 | 0.214983 |
| good | non_qrs_diff_p95 | 1.67403 | 1.4105 | -0.201989 | 0.201989 |
| good | low_amp_ratio | 0.2416 | 0.232 | -0.164384 | 0.164384 |
| medium | flatline_ratio | 0.0440352 | 0.040032 | -0.127617 | 0.127617 |
| medium | band_15_30 | 0.231507 | 0.243032 | 0.122481 | 0.122481 |
| medium | qrs_band_ratio | 8.12977 | 8.54775 | 0.119964 | 0.119964 |

## Figures

- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\v23_gm_medium_feature_boxplots.png`