# P2 Seed-3 Cross Failure Audit

Scope: external-only analysis of saved waveform-only P2 predictions. No model selection uses target-domain labels.

## Key Files
- All aligned rows: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p2_seed3_cross_failure_audit\p2_seed3_cross_failure_rows.csv`
- Record summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p2_seed3_cross_failure_audit\p2_seed3_record_error_summary.csv`
- Subtype summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p2_seed3_cross_failure_audit\p2_seed3_subtype_error_summary.csv`
- Feature gap table: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p2_seed3_cross_failure_audit\p2_seed3_error_feature_gaps.csv`
- Confusion figure: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p2_seed3_cross_failure_audit\p2_seed3_cross_confusion.png`
- PTB->BUT waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p2_seed3_cross_failure_audit\p2_seed3_ptb_to_but_error_waveforms.png`
- BUT->PTB waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\p2_seed3_cross_failure_audit\p2_seed3_but_to_ptb_error_waveforms.png`

## Worst Records
| direction | record_id | n | acc | good_n | medium_n | bad_n | wrong_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | ptbgen_v37_bad_contact_reset_flatline | 195 | 0.353846 | 0 | 0 | 195 | 126 |
| but_to_ptb | ptbgen_v37_good_mild_artifact_outlier | 270 | 0.42963 | 270 | 0 | 0 | 154 |
| but_to_ptb | ptbgen_v37_good_isolated_low_purity | 270 | 0.466667 | 270 | 0 | 0 | 144 |
| but_to_ptb | ptbgen_v37_bad_other_boundary | 192 | 0.46875 | 0 | 0 | 192 | 102 |
| but_to_ptb | ptbgen_v37_bad_dense_right_island | 195 | 0.487179 | 0 | 0 | 195 | 100 |
| but_to_ptb | ptbgen_v37_bad_detector_template_disagree | 195 | 0.528205 | 0 | 0 | 195 | 92 |
| but_to_ptb | ptbgen_v37_good_hard_baseline_lowqrs | 270 | 0.644444 | 270 | 0 | 0 | 96 |
| but_to_ptb | ptbgen_v37_good_overlap_boundary | 270 | 0.877778 | 270 | 0 | 0 | 33 |
| but_to_ptb | ptbgen_v37_bad_low_qrs_visibility | 192 | 0.885417 | 0 | 0 | 192 | 22 |
| but_to_ptb | ptbgen_v37_bad_baseline_wander_lowfreq | 195 | 0.892308 | 0 | 0 | 195 | 21 |
| but_to_ptb | ptbgen_v37_medium_hard_baseline_lowqrs | 225 | 0.902222 | 0 | 225 | 0 | 22 |
| but_to_ptb | ptbgen_v37_bad_highfreq_detail_noise | 192 | 0.90625 | 0 | 0 | 192 | 18 |
| but_to_ptb | ptbgen_v37_good_clean_core | 270 | 0.974074 | 270 | 0 | 0 | 7 |
| but_to_ptb | ptbgen_v37_medium_overlap_boundary | 225 | 0.977778 | 0 | 225 | 0 | 5 |
| but_to_ptb | ptbgen_v37_medium_isolated_lowqrs | 225 | 0.982222 | 0 | 225 | 0 | 4 |
| but_to_ptb | ptbgen_v37_medium_outlier_or_bad_boundary | 225 | 0.986667 | 0 | 225 | 0 | 3 |
| but_to_ptb | ptbgen_v37_medium_clean_core | 225 | 1 | 0 | 225 | 0 | 0 |
| but_to_ptb | ptbgen_v37_medium_visible_qrs_detail | 225 | 1 | 0 | 225 | 0 | 0 |

## Worst Subtypes
| direction | subtype_for_split | true_name | n | acc | good_n | medium_n | bad_n | wrong_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | bad_contact_reset_flatline | bad | 195 | 0.353846 | 0 | 0 | 195 | 126 |
| but_to_ptb | good_mild_artifact_outlier | good | 270 | 0.42963 | 270 | 0 | 0 | 154 |
| but_to_ptb | good_isolated_low_purity | good | 270 | 0.466667 | 270 | 0 | 0 | 144 |
| but_to_ptb | bad_other_boundary | bad | 192 | 0.46875 | 0 | 0 | 192 | 102 |
| but_to_ptb | bad_dense_right_island | bad | 195 | 0.487179 | 0 | 0 | 195 | 100 |
| but_to_ptb | bad_detector_template_disagree | bad | 195 | 0.528205 | 0 | 0 | 195 | 92 |
| but_to_ptb | good_hard_baseline_lowqrs | good | 270 | 0.644444 | 270 | 0 | 0 | 96 |
| but_to_ptb | good_overlap_boundary | good | 270 | 0.877778 | 270 | 0 | 0 | 33 |
| but_to_ptb | bad_low_qrs_visibility | bad | 192 | 0.885417 | 0 | 0 | 192 | 22 |
| but_to_ptb | bad_baseline_wander_lowfreq | bad | 195 | 0.892308 | 0 | 0 | 195 | 21 |
| but_to_ptb | medium_hard_baseline_lowqrs | medium | 225 | 0.902222 | 0 | 225 | 0 | 22 |
| but_to_ptb | bad_highfreq_detail_noise | bad | 192 | 0.90625 | 0 | 0 | 192 | 18 |
| but_to_ptb | good_clean_core | good | 270 | 0.974074 | 270 | 0 | 0 | 7 |
| but_to_ptb | medium_overlap_boundary | medium | 225 | 0.977778 | 0 | 225 | 0 | 5 |
| but_to_ptb | medium_isolated_lowqrs | medium | 225 | 0.982222 | 0 | 225 | 0 | 4 |
| but_to_ptb | medium_outlier_or_bad_boundary | medium | 225 | 0.986667 | 0 | 225 | 0 | 3 |
| but_to_ptb | medium_clean_core | medium | 225 | 1 | 0 | 225 | 0 | 0 |
| but_to_ptb | medium_visible_qrs_detail | medium | 225 | 1 | 0 | 225 | 0 | 0 |

## Largest Error Feature Gaps
| direction | true_name | feature | correct_median | wrong_median | median_gap_wrong_minus_correct | correct_iqr | wrong_iqr | correct_n | wrong_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | medium | qrs_band_ratio | 6.59085 | 3.25456 | -3.33629 | 3.5438 | 0.768034 | 1316 | 34 |
| but_to_ptb | good | qrs_band_ratio | 7.46889 | 6.25305 | -1.21584 | 1.93191 | 1.74447 | 916 | 434 |
| but_to_ptb | medium | non_qrs_diff_p95 | 1.78595 | 0.63543 | -1.15052 | 1.36195 | 0.64241 | 1316 | 34 |
| but_to_ptb | medium | qrs_visibility | 2 | 1.08496 | -0.915039 | 0.450293 | 0.436413 | 1316 | 34 |
| ptb_to_but | medium | baseline_step | 0.518712 | 1.19535 | 0.676641 | 0.501241 | 1.03711 | 3953 | 1573 |
| ptb_to_but | bad | baseline_step | 0.0288019 | 0.453686 | 0.424884 | 0.0169142 | 0.786439 | 1441 | 29 |
| ptb_to_but | bad | qrs_band_ratio | 0.80632 | 0.393921 | -0.412399 | 0.0218957 | 0.150961 | 1441 | 29 |
| ptb_to_but | bad | template_corr | 0.164613 | 0.461263 | 0.29665 | 0.0963535 | 0.283804 | 1441 | 29 |
| ptb_to_but | medium | qrs_visibility | 0.338248 | 0.0592754 | -0.278972 | 0.227708 | 0.0563125 | 3953 | 1573 |
| but_to_ptb | medium | template_corr | 0.874001 | 0.605622 | -0.268379 | 0.188562 | 0.237795 | 1316 | 34 |
| ptb_to_but | bad | non_qrs_diff_p95 | 0.376467 | 0.117369 | -0.259098 | 0.0191345 | 0.0476466 | 1441 | 29 |
| ptb_to_but | medium | qrs_band_ratio | 0.543387 | 0.296268 | -0.247119 | 0.123256 | 0.294314 | 3953 | 1573 |
| ptb_to_but | medium | sqi_basSQI | 0.965526 | 0.719149 | -0.246377 | 0.0634946 | 0.428535 | 3953 | 1573 |
| but_to_ptb | medium | sqi_basSQI | 0.681732 | 0.858048 | 0.176316 | 0.152745 | 0.0819273 | 1316 | 34 |
| ptb_to_but | good | qrs_visibility | 0.377277 | 0.548182 | 0.170905 | 0.415339 | 0.264005 | 5183 | 3841 |
| but_to_ptb | good | non_qrs_diff_p95 | 1.4174 | 1.25129 | -0.16611 | 0.670064 | 0.720817 | 916 | 434 |
| ptb_to_but | good | flatline_ratio | 0.376301 | 0.211369 | -0.164932 | 0.143315 | 0.13771 | 5183 | 3841 |
| ptb_to_but | bad | detector_agreement | 0.514719 | 0.370415 | -0.144303 | 0.105516 | 0.238831 | 1441 | 29 |
| ptb_to_but | medium | template_corr | 0.594882 | 0.453497 | -0.141385 | 0.27214 | 0.120163 | 3953 | 1573 |
| ptb_to_but | bad | qrs_visibility | 0.244492 | 0.132898 | -0.111594 | 0.0172287 | 0.0432599 | 1441 | 29 |
| but_to_ptb | good | sqi_basSQI | 0.570769 | 0.670868 | 0.100098 | 0.160586 | 0.140856 | 916 | 434 |
| but_to_ptb | good | flatline_ratio | 0.144115 | 0.0464371 | -0.0976781 | 0.0904724 | 0.13811 | 916 | 434 |
| ptb_to_but | bad | amplitude_entropy | 0.896497 | 0.801093 | -0.0954038 | 0.0414949 | 0.217875 | 1441 | 29 |
| ptb_to_but | bad | sqi_basSQI | 0.999653 | 0.904538 | -0.0951157 | 0.000209462 | 0.0542171 | 1441 | 29 |

## Interpretation
- PTB->BUT remains dominated by good/medium natural-boundary errors on specific BUT records.
- BUT->PTB exposes synthetic bad subtype mismatch: several synthetic bad mechanisms are not consistently read as bad.
- Baseline and basSQI-like targets remain cross-domain unstable; they should be redefined with a shared physical extractor before adding more loss weight.