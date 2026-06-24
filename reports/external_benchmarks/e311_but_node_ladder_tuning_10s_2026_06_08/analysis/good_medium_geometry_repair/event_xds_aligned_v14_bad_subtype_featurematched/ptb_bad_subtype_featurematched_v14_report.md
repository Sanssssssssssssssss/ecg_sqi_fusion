# PTB Bad Subtype Feature-Matched Protocol v14

- Generated: 2026-06-21 00:37:29
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v14_bad_subtype_featurematched\protocol_v14_pc3000_s20260621`
- Good/medium rows: inherited from v11 PTB aligned protocol.
- Bad rows: PTB-only generated candidates selected by interpretable feature distance to BUT subtype statistics.
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
| baseline_step | PTB v14 bad | 0.0157503634691238 |
| qrs_visibility | BUT keep bad | 0.2462702383570051 |
| qrs_visibility | PTB v14 bad | 0.5799652338027954 |
| qrs_band_ratio | BUT keep bad | 0.8095484326642155 |
| qrs_band_ratio | PTB v14 bad | 2.008800983428955 |
| detector_agreement | BUT keep bad | 0.4449944320643648 |
| detector_agreement | PTB v14 bad | 0.5 |
| non_qrs_diff_p95 | BUT keep bad | 0.3746937990188598 |
| non_qrs_diff_p95 | PTB v14 bad | 0.3407809138298034 |
| band_30_45 | BUT keep bad | 0.1037204155368285 |
| band_30_45 | PTB v14 bad | 0.0334339365363121 |

## Candidate Score Summary

| subtype | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_baseline_wander_lowfreq | 45.0 | 7.41778482860989 | 0.15301531142745103 | 6.80795955657959 | 7.384620666503906 | 7.45025110244751 | 7.523067474365234 | 7.585261344909668 |
| bad_contact_reset_flatline | 137.0 | 5.855264679358823 | 0.5151325611696874 | 3.968573808670044 | 5.550140857696533 | 5.966197967529297 | 6.263853549957275 | 6.507004737854004 |
| bad_dense_right_island | 1836.0 | 8.905094300739647 | 0.3246422596778633 | 7.836389064788818 | 8.66571569442749 | 8.934061050415039 | 9.178285360336304 | 9.399807929992676 |
| bad_detector_template_disagree | 667.0 | 8.781111457001144 | 0.3175654371698264 | 7.614259243011475 | 8.554182529449463 | 8.82016372680664 | 9.056457042694092 | 9.241132736206055 |
| bad_highfreq_detail_noise | 40.0 | 11.008204221725464 | 0.11394576206107582 | 10.733445167541504 | 10.88598108291626 | 11.031903266906738 | 11.093578338623047 | 11.1585111618042 |
| bad_low_qrs_visibility | 28.0 | 8.559832572937012 | 0.1365326093210205 | 8.31398868560791 | 8.481244325637817 | 8.551653861999512 | 8.688640832901001 | 8.721694946289062 |
| bad_other_boundary | 247.0 | 8.840098313474462 | 0.3634792309009729 | 7.712372779846191 | 8.55916166305542 | 8.898965835571289 | 9.162603378295898 | 9.331758499145508 |

## Figures

- Counts: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v14_bad_subtype_featurematched\ptb_v14_bad_subtype_counts.png`
- Waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v14_bad_subtype_featurematched\ptb_v14_bad_subtype_waveforms.png`
- Feature distributions: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v14_bad_subtype_featurematched\ptb_v14_vs_but_bad_feature_distributions.png`
- Candidate scores: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v14_bad_subtype_featurematched\ptb_v14_selected_score_by_subtype.png`