# Clean BUT Window Policy Experiment Design

This report is a data/protocol design artifact. It does not change model selection, training data, or checkpoints.

## Sources

- metadata: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_protocol_adaptation_2026_06_03\protocols\p1_current_10s_center\metadata.csv`
- atlas: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\original_region_boundary\original_region_atlas.csv`
- ANN root: `E:\GPTProject2\ecg\data\external\butqdb_1_0_0`

## Audit Result

- Current protocol windows: `32956`
- Matched to consensus segments: `32956`
- Unmatched windows: `0`

The previous 10s-duration audit showed that short consensus segments were already dropped. This stricter audit asks a different question: how far each fixed 10s window is from the start/end of its consensus-label segment, and how much data remains if we require a clean interior margin.

## Fixed 10s Clean-Window Policies

| policy | n | retention_rate | class_counts | split_class_counts | record_count | bad_outlier_low_confidence | good_medium_overlap | clean_core | good_n | medium_n | bad_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_all_10s | 32956 | 1 | {"bad": 5285, "good": 17043, "medium": 10628} | {"test": {"bad": 411, "good": 3640, "medium": 4426}, "train": {"bad": 4791, "good": 12434, "medium": 6097}, "val": {"bad": 83, "good": 969, "medium": 105}} | 18 | 1201 | 14370 | 5234 | 17043 | 10628 | 5285 |
| margin_ge_1s | 30151 | 0.914887 | {"bad": 5174, "good": 15488, "medium": 9489} | {"test": {"bad": 326, "good": 3204, "medium": 4017}, "train": {"bad": 4768, "good": 11407, "medium": 5403}, "val": {"bad": 80, "good": 877, "medium": 69}} | 18 | 1092 | 12974 | 4967 | 15488 | 9489 | 5174 |
| margin_ge_2s | 29959 | 0.909061 | {"bad": 5167, "good": 15381, "medium": 9411} | {"test": {"bad": 320, "good": 3174, "medium": 3984}, "train": {"bad": 4767, "good": 11335, "medium": 5361}, "val": {"bad": 80, "good": 872, "medium": 66}} | 18 | 1085 | 12879 | 4944 | 15381 | 9411 | 5167 |
| margin_ge_5s | 29410 | 0.892402 | {"bad": 5156, "good": 15042, "medium": 9212} | {"test": {"bad": 311, "good": 3080, "medium": 3911}, "train": {"bad": 4765, "good": 11111, "medium": 5241}, "val": {"bad": 80, "good": 851, "medium": 60}} | 18 | 1074 | 12607 | 4877 | 15042 | 9212 | 5156 |
| margin_ge_10s | 28588 | 0.86746 | {"bad": 5132, "good": 14574, "medium": 8882} | {"test": {"bad": 291, "good": 2944, "medium": 3782}, "train": {"bad": 4762, "good": 10803, "medium": 5046}, "val": {"bad": 79, "good": 827, "medium": 54}} | 18 | 1051 | 12223 | 4774 | 14574 | 8882 | 5132 |
| margin_ge_2s_drop_outlier | 21914 | 0.664947 | {"bad": 4082, "good": 11458, "medium": 6374} | {"test": {"bad": 118, "good": 1041, "medium": 2102}, "train": {"bad": 3963, "good": 9779, "medium": 4227}, "val": {"bad": 1, "good": 638, "medium": 45}} | 18 | 0 | 12879 | 4944 | 11458 | 6374 | 4082 |
| margin_ge_5s_drop_outlier | 21575 | 0.654661 | {"bad": 4082, "good": 11228, "medium": 6265} | {"test": {"bad": 118, "good": 1004, "medium": 2077}, "train": {"bad": 3963, "good": 9603, "medium": 4145}, "val": {"bad": 1, "good": 621, "medium": 43}} | 18 | 0 | 12607 | 4877 | 11228 | 6265 | 4082 |
| clean_core_plus_overlap_margin2 | 17823 | 0.540812 | {"good": 11458, "medium": 6365} | {"test": {"good": 1041, "medium": 2097}, "train": {"good": 9779, "medium": 4223}, "val": {"good": 638, "medium": 45}} | 18 | 0 | 12879 | 4944 | 11458 | 6365 | 0 |
| clean_core_only_margin2 | 4944 | 0.150018 | {"good": 3196, "medium": 1748} | {"test": {"good": 32, "medium": 488}, "train": {"good": 3054, "medium": 1241}, "val": {"good": 110, "medium": 19}} | 16 | 0 | 0 | 4944 | 3196 | 1748 | 0 |

## Regenerated-Center Window Capacity

These rows estimate how many windows could be generated from the original consensus segments if we stop anchoring windows at segment starts and instead sample only from the clean interior.

| regen_policy | min_margin_sec | mode | segment_count | candidate_window_count | record_count | good_segments | good_candidate_windows | medium_segments | medium_candidate_windows | bad_segments | bad_candidate_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| center_one_per_segment_margin0 | 0 | one_center | 2636 | 2636 | 18 | 1469 | 1469 | 1067 | 1067 | 100 | 100 |
| center_one_per_segment_margin2 | 2 | one_center | 2164 | 2164 | 18 | 1239 | 1239 | 854 | 854 | 71 | 71 |
| center_one_per_segment_margin5 | 5 | one_center | 1732 | 1732 | 18 | 1000 | 1000 | 679 | 679 | 53 | 53 |
| center_stride10_margin2 | 2 | stride10 | 2164 | 31765 | 18 | 1239 | 16396 | 854 | 10138 | 71 | 5231 |
| center_stride10_margin5 | 5 | stride10 | 1732 | 30320 | 18 | 1000 | 15574 | 679 | 9561 | 53 | 5185 |
| center_stride5_margin5 | 5 | stride5 | 1732 | 59730 | 18 | 1000 | 30616 | 679 | 18773 | 53 | 10341 |

## Edge-Risk Buckets With Smallest Margins

| split | record_id | y_class | original_region | n | segment_duration_median | min_margin_median | min_margin_p10 | min_margin_p25 | min_margin_p75 | margin_ge_2s_rate | margin_ge_5s_rate | margin_ge_10s_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 100002 | good | outlier_low_confidence | 24 | 50.427 | 0 | 0 | 0 | 12.8202 | 0.375 | 0.375 | 0.333333 |
| train | 118001 | good | outlier_low_confidence | 3 | 11.887 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | 121001 | medium | clean_core | 1 | 10.494 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | 113001 | bad | outlier_low_confidence | 4 | 14.1935 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | 115001 | good | outlier_low_confidence | 10 | 20.6145 | 0 | 0 | 0 | 3.024 | 0.3 | 0.2 | 0 |
| val | 103002 | medium | good_medium_overlap | 4 | 14.6135 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | 103002 | good | outlier_low_confidence | 5 | 23.132 | 0 | 0 | 0 | 3.132 | 0.4 | 0.2 | 0.2 |
| val | 103002 | medium | outlier_low_confidence | 3 | 12.535 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | 103001 | medium | outlier_low_confidence | 13 | 23.07 | 0 | 0 | 0 | 0 | 0.230769 | 0 | 0 |
| train | 124001 | good | good_medium_overlap | 3 | 10.821 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | 124001 | medium | outlier_low_confidence | 23 | 25.234 | 0 | 0 | 0 | 10 | 0.434783 | 0.391304 | 0.304348 |
| train | 124001 | good | outlier_low_confidence | 67 | 30.862 | 0 | 0 | 0 | 10 | 0.41791 | 0.402985 | 0.313433 |

## Recommended Experiment Sequence

1. **P0 protocol sanity:** keep current test as legacy stress, but introduce grouped CV and record-balanced validation. Do not calibrate on a validation set that has been merged into training.
2. **P1 strict-clean 10s:** train/evaluate on `margin_ge_2s_drop_outlier` and `clean_core_plus_overlap_margin2` as learnable-body diagnostics. Keep outliers as stress buckets, not deleted final evidence.
3. **P2 regenerated center windows:** create a new protocol from raw consensus segments using `center_one_per_segment_margin5` and `center_stride10_margin5`; this avoids first-10s-at-boundary artifacts.
4. **P3 variable-length / MIL:** for long segments, feed multiple clean 10s crops or variable-length chunks with masks; aggregate segment-level decisions. This preserves event context better than a single arbitrary 10s crop.
5. **P4 model input repair:** dual-view input: physical/global-normalized waveform, robust waveform, derivative recomputed after augmentation, long baseline, and local envelope/log-RMS. Augment raw waveform first, then recompute derived channels.
6. **P5 model head repair:** intrinsic-only SQI/event targets, explicit SQI late fusion, QRS/RR query, contact/baseline query, detail query, bad-stress query, and hierarchical bad-vs-nonbad then good-vs-medium classification.

## Design Decision

The clean-data path should not simply remove everything difficult. The clean interior subset is for learning and ablation. The full BUT and bad/outlier stress buckets must remain report-only stress evaluations so we can see whether the model is genuinely improving or only becoming cleaner by filtering.
