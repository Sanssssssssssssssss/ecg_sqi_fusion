# PTB Generated Bad Distribution Fit Audit

- PTB protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v19_bad_subtype_featurematched\protocol_v19_pc3000_s20260621`
- BUT target protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier`
- BUT usage here is diagnostic/target-distribution only; no BUT waveform is copied into PTB generated data.
- Current generator logic: match bad subtype proportions, over-generate candidates, compute waveform-computable SQI/morphology features, then keep candidates closest to BUT subtype feature medians/IQRs.

## Main Finding

The subtype mix is aligned, but the waveform feature distribution is only partially aligned. This audit recomputes BUT and PTB features with the same extractor before comparing distributions.

## Subtype Mix

| bad_subtype | but_share | ptb_share | share_gap |
| --- | --- | --- | --- |
| bad_contact_reset_flatline | 0.04558 | 0.04567 | 8.87e-05 |
| bad_low_qrs_visibility | 0.00931 | 0.009333 | 2.379e-05 |
| bad_other_boundary | 0.08243 | 0.08233 | -9.491e-05 |
| bad_highfreq_detail_noise | 0.01338 | 0.01333 | -4.913e-05 |
| bad_dense_right_island | 0.6119 | 0.612 | 9.154e-05 |
| bad_detector_template_disagree | 0.2225 | 0.2223 | -0.0001259 |
| bad_baseline_wander_lowfreq | 0.01493 | 0.015 | 6.594e-05 |

## Worst Feature Gaps

| feature | but_median | ptb_median | median_gap | robust_z_gap | but_iqr | ptb_iqr |
| --- | --- | --- | --- | --- | --- | --- |
| flatline_ratio | 0.006405 | 0.3775 | 0.3711 | 5.619 | 0.003203 | 0.9384 |
| amplitude_entropy | 0.8262 | 0.7393 | -0.08688 | -3.359 | 0.01012 | 0.1416 |
| band_30_45 | 0.05659 | 0.0006838 | -0.0559 | -2.198 | 0.00835 | 0.04842 |
| sqi_basSQI | 0.8539 | 0.8962 | 0.04234 | 0.937 | 0.01016 | 0.1063 |
| qrs_visibility | 0.9575 | 0.8193 | -0.1382 | -0.8924 | 0.06662 | 0.3482 |
| low_amp_ratio | 0.2056 | 0.1936 | -0.012 | -0.828 | 0.0144 | 0.0296 |
| qrs_band_ratio | 3.097 | 2.7 | -0.3966 | -0.7339 | 0.2143 | 0.7714 |
| detector_agreement | 0 | 0.08333 | 0.08333 | 0.5978 | 0 | 0.1667 |

## Peak Mechanism Summary

| source | metric | median | iqr | n |
| --- | --- | --- | --- | --- |
| BUT target | peak_count | 25 | 2 | 5156 |
| BUT target | rr_cv | 0.328 | 0.07681 | 5156 |
| BUT target | peak_amp_p90 | 2.348 | 0.6067 | 5156 |
| BUT target | template_consistency | 0.1918 | 0.1434 | 5156 |
| BUT target | detector_agreement_recomputed | 0 | 0 | 5156 |
| PTB generated | peak_count | 2 | 22 | 3000 |
| PTB generated | rr_cv | 0.2936 | 0.3954 | 2469 |
| PTB generated | peak_amp_p90 | 1.615 | 0.9675 | 3000 |
| PTB generated | template_consistency | 0.2064 | 0.4968 | 3000 |
| PTB generated | detector_agreement_recomputed | 0.08333 | 0.1667 | 3000 |

## Output Figures

- Subtype mix: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v19_bad_subtype_featurematched\distribution_fit_audit\distfit_01_bad_subtype_mix.png`
- Feature boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v19_bad_subtype_featurematched\distribution_fit_audit\distfit_02_feature_boxplots.png`
- Feature gap bars: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v19_bad_subtype_featurematched\distribution_fit_audit\distfit_03_feature_gap_bars.png`
- Subtype-feature heatmap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v19_bad_subtype_featurematched\distribution_fit_audit\distfit_04_subtype_feature_heatmap.png`
- Feature PCA overlay: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v19_bad_subtype_featurematched\distribution_fit_audit\distfit_05_feature_pca_overlay.png`
- Peak mechanism: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v19_bad_subtype_featurematched\distribution_fit_audit\distfit_06_peak_mechanism.png`

## Suggested Generator Improvement

- Keep subtype-proportion matching.
- Replace single median/IQR candidate scoring with quantile/CDF matching per subtype.
- Add a stronger candidate family for no-peak or low-peak continuous corruption, because the current candidate pool still cannot reach BUT-like qrs_visibility/qrs_band_ratio.
- Use feature-level acceptance quotas so selected candidates cover target quantile bins instead of clustering around one easy mode.