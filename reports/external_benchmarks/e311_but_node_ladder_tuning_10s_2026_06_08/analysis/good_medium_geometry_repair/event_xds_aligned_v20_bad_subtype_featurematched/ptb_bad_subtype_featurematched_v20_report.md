# PTB Bad Subtype Feature-Matched Protocol event_xds_aligned_v20_bad_subtype_featurematched

- Generated: 2026-06-21 01:28:20
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\protocol_v20_pc3000_s20260621`
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
| baseline_step | PTB v14 bad | 0.0494143329560756 |
| qrs_visibility | BUT keep bad | 0.2462702383570051 |
| qrs_visibility | PTB v14 bad | 1.0118048191070557 |
| qrs_band_ratio | BUT keep bad | 0.8095484326642155 |
| qrs_band_ratio | PTB v14 bad | 3.04151463508606 |
| detector_agreement | BUT keep bad | 0.4449944320643648 |
| detector_agreement | PTB v14 bad | 0.0 |
| non_qrs_diff_p95 | BUT keep bad | 0.3746937990188598 |
| non_qrs_diff_p95 | PTB v14 bad | 1.7488391399383545 |
| band_30_45 | BUT keep bad | 0.1037204155368285 |
| band_30_45 | PTB v14 bad | 0.0295825153589248 |

## Candidate Score Summary

| subtype | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_baseline_wander_lowfreq | 45.0 | 1.9532843867937724 | 1.2154768163227867 | 0.33788537979125977 | 0.8639204502105713 | 1.6231369972229004 | 2.7550206184387207 | 4.407502174377441 |
| bad_contact_reset_flatline | 137.0 | 0.9113937642452491 | 0.2698842474799329 | 0.29900699853897095 | 0.7485464811325073 | 0.9197894334793091 | 1.1605496406555176 | 1.289284348487854 |
| bad_dense_right_island | 1836.0 | 9.650746427005672 | 3.7281408143470607 | 0.9906195402145386 | 6.6395299434661865 | 11.693602561950684 | 12.572218179702759 | 13.44105339050293 |
| bad_detector_template_disagree | 667.0 | 9.045702173613359 | 3.729936231964403 | 1.0437028408050537 | 5.795720100402832 | 11.114890098571777 | 12.134357452392578 | 12.928147315979004 |
| bad_highfreq_detail_noise | 40.0 | 11.689648628234863 | 1.139089764288446 | 8.38205623626709 | 10.83171820640564 | 11.922186374664307 | 12.56871509552002 | 13.044597625732422 |
| bad_low_qrs_visibility | 28.0 | 1.3698730915784836 | 0.40246959663300913 | 0.6872589588165283 | 1.1231703460216522 | 1.3106626272201538 | 1.660810798406601 | 2.3595809936523438 |
| bad_other_boundary | 247.0 | 9.351631332022942 | 3.258603072278787 | 1.0583181381225586 | 7.304795980453491 | 10.567201614379883 | 12.03071641921997 | 12.601819038391113 |

## Figures

- Counts: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\ptb_v14_bad_subtype_counts.png`
- Waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\ptb_v14_bad_subtype_waveforms.png`
- Feature distributions: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\ptb_v14_vs_but_bad_feature_distributions.png`
- Candidate scores: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\ptb_v14_selected_score_by_subtype.png`