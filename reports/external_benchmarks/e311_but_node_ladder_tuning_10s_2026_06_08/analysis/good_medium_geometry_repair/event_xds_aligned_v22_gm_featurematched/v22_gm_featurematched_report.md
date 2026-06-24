# V22 Good/Medium Feature-Matched Protocol

- Generated: 2026-06-21 02:17:28
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v22_gm_featurematched\protocol_v22_pc3000_s20260621`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Good/medium rows: PTB-only candidates selected against BUT train-split waveform-computable feature medians/IQRs.
- Generator profile: `v22_goodstrong`.
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
| good | rms | 3.00666 | 2.20079 | -0.944635 | 0.944635 |
| good | qrs_band_ratio | 9.84357 | 8.58286 | -0.906907 | 0.906907 |
| good | amplitude_entropy | 0.760638 | 0.820791 | 0.902748 | 0.902748 |
| good | mean_abs | 1.33224 | 1.15093 | -0.760107 | 0.760107 |
| good | sqi_basSQI | 0.456514 | 0.521491 | 0.706544 | 0.706544 |
| good | baseline_step | 0.297628 | 0.229394 | -0.618186 | 0.618186 |
| good | template_corr | 0.883089 | 0.862651 | -0.47648 | 0.47648 |
| good | band_30_45 | 0.0173968 | 0.0221262 | 0.456288 | 0.456288 |
| good | low_amp_ratio | 0.2416 | 0.2208 | -0.356165 | 0.356165 |
| good | detector_agreement | 0.5 | 0.666667 | 0.333333 | 0.333333 |
| good | flatline_ratio | 0.115292 | 0.139311 | 0.306122 | 0.306122 |
| medium | template_corr | 0.887182 | 0.931541 | 0.279661 | 0.279661 |
| medium | detector_agreement | 0.5 | 0.583333 | 0.25 | 0.25 |
| medium | band_15_30 | 0.231507 | 0.254454 | 0.243865 | 0.243865 |
| medium | qrs_band_ratio | 8.12977 | 8.66614 | 0.153944 | 0.153944 |
| medium | amplitude_entropy | 0.802852 | 0.807592 | 0.145264 | 0.145264 |
| medium | band_30_45 | 0.026975 | 0.024296 | -0.130724 | 0.130724 |
| medium | non_qrs_diff_p95 | 2.40152 | 2.55548 | 0.102721 | 0.102721 |
| good | non_qrs_diff_p95 | 1.67403 | 1.55983 | -0.0875313 | 0.0875313 |
| medium | low_amp_ratio | 0.208 | 0.2056 | -0.0833337 | 0.0833337 |

## Figures

- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v22_gm_featurematched\v22_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v22_gm_featurematched\v22_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v22_gm_featurematched\v22_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v22_gm_featurematched\v22_gm_medium_feature_boxplots.png`