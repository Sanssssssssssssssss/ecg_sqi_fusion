# PTB Bad Subtype Feature-Matched Protocol event_xds_aligned_v17_bad_subtype_featurematched

- Generated: 2026-06-21 00:53:32
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\protocol_v17_pc3000_s20260621`
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
| baseline_step | PTB v14 bad | 0.0319836698472499 |
| qrs_visibility | BUT keep bad | 0.2462702383570051 |
| qrs_visibility | PTB v14 bad | 0.9207042455673218 |
| qrs_band_ratio | BUT keep bad | 0.8095484326642155 |
| qrs_band_ratio | PTB v14 bad | 2.763558387756348 |
| detector_agreement | BUT keep bad | 0.4449944320643648 |
| detector_agreement | PTB v14 bad | 0.5833333134651184 |
| non_qrs_diff_p95 | BUT keep bad | 0.3746937990188598 |
| non_qrs_diff_p95 | PTB v14 bad | 1.2728781700134275 |
| band_30_45 | BUT keep bad | 0.1037204155368285 |
| band_30_45 | PTB v14 bad | 0.0150371855124831 |

## Candidate Score Summary

| subtype | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_baseline_wander_lowfreq | 45.0 | 0.5704676873154111 | 0.14213907873143722 | 0.2527475655078888 | 0.4798082709312439 | 0.6132420897483826 | 0.6647499799728394 | 0.7565110325813293 |
| bad_contact_reset_flatline | 137.0 | 0.6593064667969725 | 0.11872027284262679 | 0.3473570942878723 | 0.5817108750343323 | 0.6863684058189392 | 0.7561392784118652 | 0.8145983815193176 |
| bad_dense_right_island | 1836.0 | 4.131757520707344 | 0.7083996255408332 | 1.6751654148101807 | 3.711918890476227 | 4.297621726989746 | 4.68937873840332 | 5.045522689819336 |
| bad_detector_template_disagree | 667.0 | 4.290362888547792 | 1.1558421373796641 | 0.827252984046936 | 3.5941792726516724 | 4.533823490142822 | 5.205885887145996 | 5.875337600708008 |
| bad_highfreq_detail_noise | 40.0 | 11.807141327857972 | 1.1205538652010907 | 8.211465835571289 | 11.559569120407104 | 12.093196868896484 | 12.598190784454346 | 13.031407356262207 |
| bad_low_qrs_visibility | 28.0 | 0.8656784530196872 | 0.31579593490214874 | 0.40511831641197205 | 0.5764381438493729 | 0.8490868806838989 | 1.1035042107105255 | 1.4715338945388794 |
| bad_other_boundary | 247.0 | 4.4064685000099155 | 0.8350182319491322 | 1.7305971384048462 | 3.930976986885071 | 4.6539812088012695 | 5.0853986740112305 | 5.401329517364502 |

## Figures

- Counts: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\ptb_v14_bad_subtype_counts.png`
- Waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\ptb_v14_bad_subtype_waveforms.png`
- Feature distributions: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\ptb_v14_vs_but_bad_feature_distributions.png`
- Candidate scores: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v17_bad_subtype_featurematched\ptb_v14_selected_score_by_subtype.png`