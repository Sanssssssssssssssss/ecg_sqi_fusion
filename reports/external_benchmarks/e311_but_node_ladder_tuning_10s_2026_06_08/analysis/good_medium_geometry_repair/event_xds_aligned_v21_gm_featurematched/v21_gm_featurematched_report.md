# V21 Good/Medium Feature-Matched Protocol

- Generated: 2026-06-21 02:01:48
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v21_gm_featurematched\protocol_v21_pc3000_s20260621`
- Bad rows: inherited from V20 bad/poor distribution-matched protocol.
- Good/medium rows: PTB-only candidates selected against BUT train-split waveform-computable feature medians/IQRs.
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
| good | rms | 3.00666 | 2.09031 | -1.07415 | 1.07415 |
| good | qrs_band_ratio | 9.84357 | 8.44186 | -1.00833 | 1.00833 |
| good | mean_abs | 1.33224 | 1.12409 | -0.872623 | 0.872623 |
| good | amplitude_entropy | 0.760638 | 0.816941 | 0.844973 | 0.844973 |
| good | sqi_basSQI | 0.456514 | 0.5321 | 0.821899 | 0.821899 |
| good | baseline_step | 0.297628 | 0.219836 | -0.704778 | 0.704778 |
| good | template_corr | 0.883089 | 0.857803 | -0.589504 | 0.589504 |
| good | low_amp_ratio | 0.2416 | 0.2128 | -0.493151 | 0.493151 |
| good | band_30_45 | 0.0173968 | 0.0222828 | 0.471396 | 0.471396 |
| good | flatline_ratio | 0.115292 | 0.141713 | 0.336735 | 0.336735 |
| good | detector_agreement | 0.5 | 0.666667 | 0.333333 | 0.333333 |
| medium | template_corr | 0.887182 | 0.939212 | 0.328024 | 0.328024 |
| medium | detector_agreement | 0.5 | 0.583333 | 0.25 | 0.25 |
| medium | band_15_30 | 0.231507 | 0.254488 | 0.244219 | 0.244219 |
| medium | qrs_band_ratio | 8.12977 | 8.73942 | 0.174975 | 0.174975 |
| medium | amplitude_entropy | 0.802852 | 0.806591 | 0.114577 | 0.114577 |
| good | non_qrs_diff_p95 | 1.67403 | 1.52679 | -0.112858 | 0.112858 |
| medium | low_amp_ratio | 0.208 | 0.2048 | -0.111111 | 0.111111 |
| medium | flatline_ratio | 0.0440352 | 0.0408327 | -0.102094 | 0.102094 |
| medium | band_30_45 | 0.026975 | 0.0252051 | -0.0863671 | 0.0863671 |

## Figures

- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v21_gm_featurematched\v21_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v21_gm_featurematched\v21_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v21_gm_featurematched\v21_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v21_gm_featurematched\v21_gm_medium_feature_boxplots.png`