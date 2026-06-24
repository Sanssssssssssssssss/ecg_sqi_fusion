# PTB Bad Subtype Feature-Matched Protocol event_xds_aligned_v16_bad_subtype_featurematched

- Generated: 2026-06-21 00:51:11
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v16_bad_subtype_featurematched\protocol_v16_pc3000_s20260621`
- Good/medium rows: inherited from v11 PTB aligned protocol.
- Bad rows: PTB-only generated candidates selected by interpretable feature distance to BUT subtype statistics.
- v15 mode recomputes BUT target features from signals with the same extractor used for generated PTB rows.
- BUT usage: subtype proportions and feature medians/IQRs only; no BUT waveform is copied.

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

## Subtype Counts

| split | class_name | computed_subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 7 |
| test | bad | bad_contact_reset_flatline | 20 |
| test | bad | bad_dense_right_island | 276 |
| test | bad | bad_detector_template_disagree | 100 |
| test | bad | bad_highfreq_detail_noise | 6 |
| test | bad | bad_low_qrs_visibility | 4 |
| test | bad | bad_other_boundary | 37 |
| test | good | good_overlap_or_mild_artifact | 452 |
| test | medium | medium_overlap_or_detail | 506 |
| train | bad | bad_baseline_wander_lowfreq | 31 |
| train | bad | bad_contact_reset_flatline | 96 |
| train | bad | bad_dense_right_island | 1285 |
| train | bad | bad_detector_template_disagree | 467 |
| train | bad | bad_highfreq_detail_noise | 28 |
| train | bad | bad_low_qrs_visibility | 20 |
| train | bad | bad_other_boundary | 173 |
| train | good | good_overlap_or_mild_artifact | 2105 |
| train | medium | medium_overlap_or_detail | 2014 |
| val | bad | bad_baseline_wander_lowfreq | 7 |
| val | bad | bad_contact_reset_flatline | 21 |
| val | bad | bad_dense_right_island | 275 |
| val | bad | bad_detector_template_disagree | 100 |
| val | bad | bad_highfreq_detail_noise | 6 |
| val | bad | bad_low_qrs_visibility | 4 |
| val | bad | bad_other_boundary | 37 |
| val | good | good_overlap_or_mild_artifact | 443 |
| val | medium | medium_overlap_or_detail | 480 |

## Feature Median Sanity

| feature | group | median |
| --- | --- | --- |
| baseline_step | BUT keep bad | 0.027805033350936 |
| baseline_step | PTB v14 bad | 0.029962882399559 |
| qrs_visibility | BUT keep bad | 0.2462702383570051 |
| qrs_visibility | PTB v14 bad | 0.7646151781082153 |
| qrs_band_ratio | BUT keep bad | 0.8095484326642155 |
| qrs_band_ratio | PTB v14 bad | 2.409499406814575 |
| detector_agreement | BUT keep bad | 0.4449944320643648 |
| detector_agreement | PTB v14 bad | 0.6666666865348816 |
| non_qrs_diff_p95 | BUT keep bad | 0.3746937990188598 |
| non_qrs_diff_p95 | PTB v14 bad | 1.2704893350601196 |
| band_30_45 | BUT keep bad | 0.1037204155368285 |
| band_30_45 | PTB v14 bad | 0.002755892695859 |

## Candidate Score Summary

| subtype | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_baseline_wander_lowfreq | 45.0 | 0.43801864153809017 | 0.09794279559754455 | 0.15736716985702515 | 0.3724241256713867 | 0.48889362812042236 | 0.5113180875778198 | 0.5559592247009277 |
| bad_contact_reset_flatline | 137.0 | 0.7424850131038332 | 0.09667592868401856 | 0.4173794090747833 | 0.6857509613037109 | 0.7603979706764221 | 0.8125636577606201 | 0.887697160243988 |
| bad_dense_right_island | 1836.0 | 4.631542264830832 | 0.7231830294508158 | 1.2878702878952026 | 4.272057414054871 | 4.657950162887573 | 5.167681932449341 | 5.798629283905029 |
| bad_detector_template_disagree | 667.0 | 6.225981512974048 | 1.003695031930196 | 0.8676819205284119 | 6.281866073608398 | 6.621058940887451 | 6.76800537109375 | 6.891584396362305 |
| bad_highfreq_detail_noise | 40.0 | 13.88691692352295 | 0.8807443339917679 | 10.09644603729248 | 13.923510551452637 | 14.155747890472412 | 14.366625547409058 | 14.570151329040527 |
| bad_low_qrs_visibility | 28.0 | 0.5572844243475369 | 0.17049921224342318 | 0.19829896092414856 | 0.43892405927181244 | 0.5605869591236115 | 0.7210361361503601 | 0.7701572179794312 |
| bad_other_boundary | 247.0 | 5.115995644075185 | 0.9942959631047321 | 1.0497251749038696 | 4.401590824127197 | 5.303523540496826 | 5.977794647216797 | 6.353239059448242 |

## Figures

- Counts: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v16_bad_subtype_featurematched\ptb_v14_bad_subtype_counts.png`
- Waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v16_bad_subtype_featurematched\ptb_v14_bad_subtype_waveforms.png`
- Feature distributions: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v16_bad_subtype_featurematched\ptb_v14_vs_but_bad_feature_distributions.png`
- Candidate scores: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v16_bad_subtype_featurematched\ptb_v14_selected_score_by_subtype.png`