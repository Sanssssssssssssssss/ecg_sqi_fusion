# PTB Bad Subtype Feature-Matched Protocol event_xds_aligned_v15_bad_subtype_featurematched

- Generated: 2026-06-21 00:48:48
- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v15_bad_subtype_featurematched\protocol_v15_pc3000_s20260621`
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
| baseline_step | PTB v14 bad | 0.0541030652821064 |
| qrs_visibility | BUT keep bad | 0.2462702383570051 |
| qrs_visibility | PTB v14 bad | 0.991968274116516 |
| qrs_band_ratio | BUT keep bad | 0.8095484326642155 |
| qrs_band_ratio | PTB v14 bad | 3.045927047729492 |
| detector_agreement | BUT keep bad | 0.4449944320643648 |
| detector_agreement | PTB v14 bad | 0.0 |
| non_qrs_diff_p95 | BUT keep bad | 0.3746937990188598 |
| non_qrs_diff_p95 | PTB v14 bad | 2.6922411918640137 |
| band_30_45 | BUT keep bad | 0.1037204155368285 |
| band_30_45 | PTB v14 bad | 0.3018146753311157 |

## Candidate Score Summary

| subtype | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_baseline_wander_lowfreq | 45.0 | 0.3481297966506746 | 0.0727129490069026 | 0.12219156324863434 | 0.3125794231891632 | 0.3734899163246155 | 0.4033437669277191 | 0.4364091157913208 |
| bad_contact_reset_flatline | 137.0 | 0.6811962145088363 | 0.11500619836181004 | 0.30417725443840027 | 0.6142906546592712 | 0.7098895907402039 | 0.7695388197898865 | 0.8135421872138977 |
| bad_dense_right_island | 1836.0 | 2.4325752861489396 | 0.277615453028424 | 1.1302374601364136 | 2.2511550188064575 | 2.4878894090652466 | 2.6549859642982483 | 2.8083267211914062 |
| bad_detector_template_disagree | 667.0 | 2.400685355312999 | 0.31598621411404304 | 0.8676819205284119 | 2.243808627128601 | 2.461005210876465 | 2.644791841506958 | 2.781153917312622 |
| bad_highfreq_detail_noise | 40.0 | 4.895503222942352 | 0.583488495521117 | 3.6269969940185547 | 4.553444147109985 | 4.767561674118042 | 5.195167183876038 | 6.316628456115723 |
| bad_low_qrs_visibility | 28.0 | 0.37735690655452864 | 0.059484670582296076 | 0.24565373361110687 | 0.34824106842279434 | 0.385039746761322 | 0.4235348328948021 | 0.45977064967155457 |
| bad_other_boundary | 247.0 | 2.3854282949617516 | 0.24097795997084778 | 0.8402178287506104 | 2.2551894187927246 | 2.425748348236084 | 2.569351553916931 | 2.6905577182769775 |

## Figures

- Counts: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v15_bad_subtype_featurematched\ptb_v14_bad_subtype_counts.png`
- Waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v15_bad_subtype_featurematched\ptb_v14_bad_subtype_waveforms.png`
- Feature distributions: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v15_bad_subtype_featurematched\ptb_v14_vs_but_bad_feature_distributions.png`
- Candidate scores: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v15_bad_subtype_featurematched\ptb_v14_selected_score_by_subtype.png`