# V26_ALLTARGET_DIAG Good/Medium Feature-Matched Protocol

- Generated: 2026-06-21 03:02:49
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\protocol_v26_alltarget_diag_pc3000_s20260621`
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
| good | rms | 3.00666 | 2.18662 | -0.961244 | 0.961244 |
| good | amplitude_entropy | 0.760638 | 0.81457 | 0.809383 | 0.809383 |
| good | mean_abs | 1.33224 | 1.14698 | -0.776655 | 0.776655 |
| good | qrs_band_ratio | 9.84357 | 8.77907 | -0.765762 | 0.765762 |
| good | sqi_basSQI | 0.456514 | 0.521997 | 0.712044 | 0.712044 |
| good | baseline_step | 0.297628 | 0.22893 | -0.622395 | 0.622395 |
| good | template_corr | 0.883089 | 0.866795 | -0.379877 | 0.379877 |
| good | detector_agreement | 0.5 | 0.666667 | 0.333333 | 0.333333 |
| good | flatline_ratio | 0.115292 | 0.140112 | 0.316326 | 0.316326 |
| medium | amplitude_entropy | 0.802852 | 0.812853 | 0.306462 | 0.306462 |
| good | band_15_30 | 0.221663 | 0.204729 | -0.302773 | 0.302773 |
| medium | detector_agreement | 0.5 | 0.583333 | 0.25 | 0.25 |
| good | band_30_45 | 0.0173968 | 0.0196251 | 0.214983 | 0.214983 |
| good | non_qrs_diff_p95 | 1.67403 | 1.40056 | -0.209608 | 0.209608 |
| medium | template_corr | 0.887182 | 0.917856 | 0.193384 | 0.193384 |
| good | low_amp_ratio | 0.2416 | 0.232 | -0.164384 | 0.164384 |
| medium | band_15_30 | 0.231507 | 0.24533 | 0.146894 | 0.146894 |
| medium | flatline_ratio | 0.0440352 | 0.0408327 | -0.102094 | 0.102094 |
| medium | band_30_45 | 0.026975 | 0.0252339 | -0.0849586 | 0.0849586 |
| medium | mean_abs | 1.07184 | 1.0586 | -0.0593766 | 0.0593766 |

## Figures

- Good gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_good_feature_gap.png`
- Medium gap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_medium_feature_gap.png`
- Good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_good_feature_boxplots.png`
- Medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v26_alltarget_diag_gm_featurematched\v26_alltarget_diag_gm_medium_feature_boxplots.png`