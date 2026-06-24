# PTB Generated Bad Distribution Fit Audit

- PTB protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\protocol_v20_pc3000_s20260621`
- BUT target protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier`
- BUT usage here is diagnostic/target-distribution only; no BUT waveform is copied into PTB generated data.
- Current generator logic: match bad subtype proportions, over-generate candidates, compute waveform-computable SQI/morphology features, then keep candidates closest to BUT subtype feature medians/IQRs.

## Main Finding

The subtype mix is aligned, but the waveform feature distribution is only partially aligned. This audit recomputes BUT and PTB features with the same extractor before comparing distributions.

## Subtype Mix

| bad_subtype | but_share | ptb_share | share_gap |
| --- | --- | --- | --- |
| bad_detector_template_disagree | 0.2225 | 0.2223 | -0.0001259 |
| bad_contact_reset_flatline | 0.04558 | 0.04567 | 8.87e-05 |
| bad_dense_right_island | 0.6119 | 0.612 | 9.154e-05 |
| bad_low_qrs_visibility | 0.00931 | 0.009333 | 2.379e-05 |
| bad_highfreq_detail_noise | 0.01338 | 0.01333 | -4.913e-05 |
| bad_other_boundary | 0.08243 | 0.08233 | -9.491e-05 |
| bad_baseline_wander_lowfreq | 0.01493 | 0.015 | 6.594e-05 |

## Worst Feature Gaps

| feature | but_median | ptb_median | median_gap | robust_z_gap | but_iqr | ptb_iqr |
| --- | --- | --- | --- | --- | --- | --- |
| band_30_45 | 0.05659 | 0.02958 | -0.027 | -1.062 | 0.00835 | 0.1969 |
| amplitude_entropy | 0.8262 | 0.8025 | -0.02366 | -0.9149 | 0.01012 | 0.01517 |
| low_amp_ratio | 0.2056 | 0.1968 | -0.0088 | -0.6072 | 0.0144 | 0.0152 |
| sqi_basSQI | 0.8539 | 0.835 | -0.01892 | -0.4187 | 0.01016 | 0.04064 |
| qrs_visibility | 0.9575 | 1.012 | 0.0543 | 0.3506 | 0.06662 | 0.1121 |
| qrs_band_ratio | 3.097 | 3.042 | -0.05512 | -0.102 | 0.2143 | 0.2424 |
| non_qrs_diff_p95 | 2.769 | 1.749 | -1.02 | -0.06623 | 0.1239 | 1.27 |
| flatline_ratio | 0.006405 | 0.01041 | 0.004003 | 0.06062 | 0.003203 | 0.006405 |

## Peak Mechanism Summary

| source | metric | median | iqr | n |
| --- | --- | --- | --- | --- |
| BUT target | peak_count | 25 | 2 | 5156 |
| BUT target | rr_cv | 0.328 | 0.07681 | 5156 |
| BUT target | peak_amp_p90 | 2.348 | 0.6067 | 5156 |
| BUT target | template_consistency | 0.1918 | 0.1434 | 5156 |
| BUT target | detector_agreement_recomputed | 0 | 0 | 5156 |
| PTB generated | peak_count | 24 | 3 | 3000 |
| PTB generated | rr_cv | 0.356 | 0.1299 | 2998 |
| PTB generated | peak_amp_p90 | 2.162 | 0.2779 | 3000 |
| PTB generated | template_consistency | 0.1843 | 0.0783 | 3000 |
| PTB generated | detector_agreement_recomputed | 0 | 0.1667 | 3000 |

## Output Figures

- Subtype mix: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_01_bad_subtype_mix.png`
- Feature boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_02_feature_boxplots.png`
- Feature gap bars: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_03_feature_gap_bars.png`
- Subtype-feature heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_04_subtype_feature_heatmap.png`
- Feature PCA overlay: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_05_feature_pca_overlay.png`
- Peak mechanism: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\distribution_fit_audit\distfit_06_peak_mechanism.png`

## Suggested Generator Improvement

- Keep subtype-proportion matching.
- Replace single median/IQR candidate scoring with quantile/CDF matching per subtype.
- Add a stronger candidate family for no-peak or low-peak continuous corruption, because the current candidate pool still cannot reach BUT-like qrs_visibility/qrs_band_ratio.
- Use feature-level acceptance quotas so selected candidates cover target quantile bins instead of clustering around one easy mode.