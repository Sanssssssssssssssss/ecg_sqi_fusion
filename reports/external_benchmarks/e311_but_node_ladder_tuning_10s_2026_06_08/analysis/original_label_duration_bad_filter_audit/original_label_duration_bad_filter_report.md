# Original 10s Label-Duration Bad-Filter Audit

Report-only audit. This does not change selection, promotion, training data, or mainline checkpoints.

## Main Finding

- The current `p1_current_10s_center` original set is already built from full 10s windows inside continuous consensus label segments.
- Protocol windows failing the explicit 10s-labeled scope check: `0`.
- Raw consensus annotation segments shorter than 10s do exist and were dropped before protocol generation: `4468` segments (good `1685`, medium `2459`, bad `324`).

## Counts Under 10s+ Labeled Scope

- Original all 10s+ labeled: `32956` windows = good `17043`, medium `10628`, bad `5285`.
- Original test 10s+ labeled: `8477` windows = good `3640`, medium `4426`, bad `411`.
- Original test 10s+ after dropping only `bad/outlier_low_confidence`: `8185` windows = good `3640`, medium `4426`, bad core `119`.

## Bad Filter Severity

- In original test 10s+ labeled, `bad/outlier_low_confidence` removes `292` of `411` bad windows (71.0%).
- Removed bad/outlier windows all come from record(s) `{'111001': 292}` across `76` continuous label segments; their median label-segment duration is `101.0s`.
- Kept bad core windows come from record(s) `{'122001': 119}` across `1` continuous label segment(s); their median label-segment duration is `1200.0s`.
- So yes: the current bad outlier filter is very aggressive for bad coverage. It is not caused by short labels being included; those short segments were already excluded. The filter is instead removing a subject/record-specific low-confidence bad cluster.
- If we want original-filtered to still test bad meaningfully, a softer bad policy should keep at least `near_bad_boundary` plus `right_bad_island`, and only exclude bad outliers as a separate report-only stress bucket rather than deleting them from the main filtered score.

## Artifacts

- `scope_bad_filter_summary.csv`
- `region_counts_10s_labeled_by_split_class.csv`
- `raw_consensus_segment_duration_counts.csv`
- `short_consensus_segments_by_record_class.csv`
- `test_bad_10s_region_counts_by_record.csv`
- `test_bad_region_filter_counts.png`
- `test_bad_segment_duration_hist.png`
