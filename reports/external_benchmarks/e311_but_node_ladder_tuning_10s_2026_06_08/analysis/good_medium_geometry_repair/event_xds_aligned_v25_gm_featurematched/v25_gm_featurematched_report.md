# V25 Good/Medium Feature-Matched Protocol

- Generated: 2026-06-21 02:53:49
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\protocol_v25_pc3000_s20260621`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Good/medium rows: PTB-only candidates selected against BUT train-split waveform-computable feature medians/IQRs.
- Generator profile: `v25_forced_medium_modes`.
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
| good | rms | 3.00666 | 2.17873 | -0.970494 | 0.970494 |
| good | amplitude_entropy | 0.760638 | 0.81561 | 0.824991 | 0.824991 |
| good | sqi_basSQI | 0.456514 | 0.52922 | 0.79058 | 0.79058 |
| good | mean_abs | 1.33224 | 1.14698 | -0.776655 | 0.776655 |
| good | qrs_band_ratio | 9.84357 | 8.79233 | -0.756221 | 0.756221 |
| good | baseline_step | 0.297628 | 0.222393 | -0.681612 | 0.681612 |
| good | detector_agreement | 0.5 | 0.666667 | 0.333333 | 0.333333 |
| good | band_15_30 | 0.221663 | 0.203034 | -0.333079 | 0.333079 |
| medium | amplitude_entropy | 0.802852 | 0.813091 | 0.313763 | 0.313763 |
| good | flatline_ratio | 0.115292 | 0.138511 | 0.295918 | 0.295918 |
| medium | band_30_45 | 0.026975 | 0.0211606 | -0.283721 | 0.283721 |
| good | template_corr | 0.883089 | 0.871104 | -0.279408 | 0.279408 |
| medium | template_corr | 0.887182 | 0.927379 | 0.253423 | 0.253423 |
| medium | detector_agreement | 0.5 | 0.583333 | 0.25 | 0.25 |
| good | non_qrs_diff_p95 | 1.67403 | 1.3889 | -0.218545 | 0.218545 |
| good | low_amp_ratio | 0.2416 | 0.232 | -0.164384 | 0.164384 |
| good | band_30_45 | 0.0173968 | 0.0186912 | 0.124883 | 0.124883 |
| medium | sqi_basSQI | 0.605181 | 0.62237 | 0.120803 | 0.120803 |
| medium | baseline_step | 0.1631 | 0.15169 | -0.118342 | 0.118342 |
| medium | qrs_band_ratio | 8.12977 | 7.77737 | -0.101142 | 0.101142 |

## Figures

- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v25_gm_featurematched\v25_gm_medium_feature_boxplots.png`