# PTB Bad Subtype Feature-Matched Protocol event_xds_aligned_v18_bad_subtype_featurematched

- Generated: 2026-06-21 00:55:56
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v18_bad_subtype_featurematched\protocol_v18_pc3000_s20260621`
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
| baseline_step | PTB v14 bad | 0.0429193452000618 |
| qrs_visibility | BUT keep bad | 0.2462702383570051 |
| qrs_visibility | PTB v14 bad | 0.9078477621078492 |
| qrs_band_ratio | BUT keep bad | 0.8095484326642155 |
| qrs_band_ratio | PTB v14 bad | 2.7744736671447754 |
| detector_agreement | BUT keep bad | 0.4449944320643648 |
| detector_agreement | PTB v14 bad | 0.6666666865348816 |
| non_qrs_diff_p95 | BUT keep bad | 0.3746937990188598 |
| non_qrs_diff_p95 | PTB v14 bad | 0.6526693105697632 |
| band_30_45 | BUT keep bad | 0.1037204155368285 |
| band_30_45 | PTB v14 bad | 0.0051269680261611 |

## Candidate Score Summary

| subtype | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_baseline_wander_lowfreq | 45.0 | 1.1473233653439416 | 0.6171154309122527 | 0.43133294582366943 | 0.6531720757484436 | 0.8984276056289673 | 1.819962978363037 | 2.2279279232025146 |
| bad_contact_reset_flatline | 137.0 | 0.4671342279354151 | 0.08405589130740618 | 0.26852846145629883 | 0.40355396270751953 | 0.4637816250324249 | 0.529839277267456 | 0.6079681515693665 |
| bad_dense_right_island | 1836.0 | 3.8606864363279736 | 0.4575962881729692 | 1.4093165397644043 | 3.670601487159729 | 3.9768747091293335 | 4.202765226364136 | 4.367788314819336 |
| bad_detector_template_disagree | 667.0 | 3.7290456077863072 | 0.6621772175334257 | 0.7855349183082581 | 3.549475073814392 | 3.9208686351776123 | 4.180191993713379 | 4.379420757293701 |
| bad_highfreq_detail_noise | 40.0 | 5.939092773199081 | 1.0022835967949217 | 3.411475419998169 | 5.363800764083862 | 6.17700982093811 | 6.751182794570923 | 7.377030372619629 |
| bad_low_qrs_visibility | 28.0 | 1.2427064308098383 | 0.7442395430405967 | 0.2778123617172241 | 0.6608107686042786 | 1.0675612092018127 | 1.4774261713027954 | 2.7107479572296143 |
| bad_other_boundary | 247.0 | 3.8507629005532515 | 0.4208527867840028 | 1.9840511083602905 | 3.6731300354003906 | 3.943484306335449 | 4.176828622817993 | 4.360053539276123 |

## Figures

- Counts: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v18_bad_subtype_featurematched\ptb_v14_bad_subtype_counts.png`
- Waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v18_bad_subtype_featurematched\ptb_v14_bad_subtype_waveforms.png`
- Feature distributions: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v18_bad_subtype_featurematched\ptb_v14_vs_but_bad_feature_distributions.png`
- Candidate scores: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v18_bad_subtype_featurematched\ptb_v14_selected_score_by_subtype.png`