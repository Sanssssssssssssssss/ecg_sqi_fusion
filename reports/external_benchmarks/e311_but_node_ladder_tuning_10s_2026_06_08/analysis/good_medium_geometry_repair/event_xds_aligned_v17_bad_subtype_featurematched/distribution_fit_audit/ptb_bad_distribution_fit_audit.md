# PTB Generated Bad Distribution Fit Audit

- PTB protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\protocol_v17_pc3000_s20260621`
- BUT target protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier`
- BUT usage here is diagnostic/target-distribution only; no BUT waveform is copied into PTB generated data.
- Current generator logic: match bad subtype proportions, over-generate candidates, compute waveform-computable SQI/morphology features, then keep candidates closest to BUT subtype feature medians/IQRs.

## Main Finding

The subtype mix is aligned, but the waveform feature distribution is only partially aligned. This audit recomputes BUT and PTB features with the same extractor before comparing distributions.

## Subtype Mix

| bad_subtype | but_share | ptb_share | share_gap |
| --- | --- | --- | --- |
| bad_other_boundary | 0.08243 | 0.08233 | -9.491e-05 |
| bad_baseline_wander_lowfreq | 0.01493 | 0.015 | 6.594e-05 |
| bad_low_qrs_visibility | 0.00931 | 0.009333 | 2.379e-05 |
| bad_dense_right_island | 0.6119 | 0.612 | 9.154e-05 |
| bad_detector_template_disagree | 0.2225 | 0.2223 | -0.0001259 |
| bad_contact_reset_flatline | 0.04558 | 0.04567 | 8.87e-05 |
| bad_highfreq_detail_noise | 0.01338 | 0.01333 | -4.913e-05 |

## Worst Feature Gaps

| feature | but_median | ptb_median | median_gap | robust_z_gap | but_iqr | ptb_iqr |
| --- | --- | --- | --- | --- | --- | --- |
| detector_agreement | 0 | 0.5833 | 0.5833 | 4.185 | 0 | 0.5 |
| amplitude_entropy | 0.8262 | 0.7714 | -0.05477 | -2.118 | 0.01012 | 0.03976 |
| band_30_45 | 0.05659 | 0.01504 | -0.04155 | -1.634 | 0.00835 | 0.02119 |
| low_amp_ratio | 0.2056 | 0.1936 | -0.012 | -0.828 | 0.0144 | 0.02 |
| sqi_basSQI | 0.8539 | 0.8866 | 0.0327 | 0.7236 | 0.01016 | 0.03609 |
| qrs_band_ratio | 3.097 | 2.764 | -0.3331 | -0.6164 | 0.2143 | 0.3903 |
| qrs_visibility | 0.9575 | 0.9207 | -0.0368 | -0.2376 | 0.06662 | 0.1594 |
| flatline_ratio | 0.006405 | 0.01521 | 0.008807 | 0.1334 | 0.003203 | 0.01361 |

## Output Figures

- Subtype mix: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\distribution_fit_audit\distfit_01_bad_subtype_mix.png`
- Feature boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\distribution_fit_audit\distfit_02_feature_boxplots.png`
- Feature gap bars: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\distribution_fit_audit\distfit_03_feature_gap_bars.png`
- Subtype-feature heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\distribution_fit_audit\distfit_04_subtype_feature_heatmap.png`
- Feature PCA overlay: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\distribution_fit_audit\distfit_05_feature_pca_overlay.png`

## Suggested Generator Improvement

- Keep subtype-proportion matching.
- Replace single median/IQR candidate scoring with quantile/CDF matching per subtype.
- Add a stronger candidate family for no-peak or low-peak continuous corruption, because the current candidate pool still cannot reach BUT-like qrs_visibility/qrs_band_ratio.
- Use feature-level acceptance quotas so selected candidates cover target quantile bins instead of clustering around one easy mode.