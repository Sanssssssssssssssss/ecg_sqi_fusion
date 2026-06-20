# Joint PTB+BUT Split-Roll Diagnostic

- Generated: 2026-06-20 22:49:32
- Model: EventFactorizedSQIConformer, waveform-derived channels only at inference.
- Teacher signals: interpretable SQI/factor targets used only during training/diagnostics.
- Purpose: test whether the current cleaned BUT test score is dominated by split skew.

## Current Clean BUT Split

Class counts:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 118 |
| test | good | 1004 |
| test | medium | 2077 |
| train | bad | 3963 |
| train | good | 9603 |
| train | medium | 4145 |
| val | bad | 1 |
| val | good | 621 |
| val | medium | 43 |

Top record/class split counts:

| split | record_id | class_name | n |
| --- | --- | --- | --- |
| test | 111001 | medium | 2053 |
| test | 111001 | good | 927 |
| test | 122001 | bad | 118 |
| test | 122001 | good | 70 |
| test | 122001 | medium | 24 |
| test | 125001 | good | 7 |
| train | 105001 | good | 4554 |
| train | 100001 | good | 4457 |
| train | 105001 | bad | 3963 |
| train | 105001 | medium | 2250 |
| train | 100001 | medium | 1669 |
| train | 113001 | good | 125 |
| train | 115001 | good | 116 |
| train | 118001 | good | 116 |
| train | 123001 | good | 90 |
| train | 121001 | good | 89 |
| train | 124001 | medium | 79 |
| train | 100002 | good | 56 |
| train | 104001 | medium | 42 |
| train | 113001 | medium | 29 |
| train | 118001 | medium | 29 |
| train | 100002 | medium | 23 |
| train | 115001 | medium | 22 |
| train | 123001 | medium | 2 |
| val | 114001 | good | 209 |
| val | 103002 | good | 175 |
| val | 126001 | good | 105 |
| val | 103001 | good | 95 |
| val | 103003 | good | 37 |
| val | 114001 | medium | 30 |

Interpretation: the current test split is mostly record `111001` good/medium overlap, while validation has almost no bad. This makes it a hard record/region-shift test, not a balanced clean-BUT test.

## Split-Roll Results

| split_mode | split_seed | candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window_stratified | 4101 | E1_query_only | but_all | 0.876941 | 0.883188 | 0.885109 | 0.782123 | 1 | 0.900662 | 0.0477697 |
| window_stratified | 4101 | E1_query_only | but_test | 0.872064 | 0.877466 | 0.880712 | 0.773163 | 1 | 0.931936 | 0.0559284 |
| window_stratified | 4101 | E1_query_only | joint_test | 0.908454 | 0.912202 | 0.904539 | 0.843599 | 0.999112 | 0.991436 | 0.0446321 |
| window_stratified | 4101 | E1_query_only | ptb_test | 0.988451 | 0.988753 | 0.993363 | 0.974308 | 0.998054 | 0.994577 | 0.0314136 |
| window_stratified | 4102 | E1_query_only | but_all | 0.877173 | 0.879948 | 0.887068 | 0.779409 | 1 | 0.89236 | 0.0741672 |
| window_stratified | 4102 | E1_query_only | but_test | 0.872991 | 0.874603 | 0.886647 | 0.765708 | 1 | 0.910466 | 0.0807453 |
| window_stratified | 4102 | E1_query_only | joint_test | 0.910365 | 0.912732 | 0.910154 | 0.841522 | 0.999112 | 0.992176 | 0.0543353 |
| window_stratified | 4102 | E1_query_only | ptb_test | 0.992527 | 0.992765 | 0.997788 | 0.982213 | 0.998054 | 0.996249 | 0.0209424 |
| window_stratified | 4103 | E1_query_only | but_all | 0.866651 | 0.875642 | 0.831493 | 0.842777 | 1 | 0.824764 | 0.064628 |
| window_stratified | 4103 | E1_query_only | but_test | 0.860939 | 0.870038 | 0.826706 | 0.831736 | 1 | 0.875911 | 0.0767635 |
| window_stratified | 4103 | E1_query_only | joint_test | 0.901444 | 0.906859 | 0.861956 | 0.883045 | 1 | 0.992451 | 0.0532407 |
| window_stratified | 4103 | E1_query_only | ptb_test | 0.990489 | 0.990713 | 0.993363 | 0.978261 | 1 | 0.998602 | 0.0235602 |


## Split-Roll Mean Across 3 Seeds

| split_mode | candidate | bucket | acc_mean | acc_std | macro_f1_mean | good_recall_mean | medium_recall_mean | bad_recall_mean | record_macro_supported_f1_mean | artifact_positive_nonbad_bad_fpr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window_stratified | E1_query_only | but_all | 0.873588 | 0.00600877 | 0.879593 | 0.86789 | 0.801437 | 1 | 0.872595 | 0.0621883 |
| window_stratified | E1_query_only | but_test | 0.868665 | 0.00670659 | 0.874036 | 0.864688 | 0.790202 | 1 | 0.906104 | 0.0711457 |
| window_stratified | E1_query_only | joint_test | 0.906754 | 0.00469697 | 0.910598 | 0.892217 | 0.856055 | 0.999408 | 0.992021 | 0.050736 |
| window_stratified | E1_query_only | ptb_test | 0.990489 | 0.00203804 | 0.990744 | 0.994838 | 0.978261 | 0.998703 | 0.996476 | 0.0253054 |

## Caveat

`window_stratified` is diagnostic only because it allows record overlap across train/val/test. If it improves sharply, the low current BUT test score is at least partly a split/record-shift issue. It is not a replacement for strict external validation.

## Output Files

- Aggregate summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\split_roll_aggregate_summary.csv`
- Source split audit: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll`
- Per-run reports: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_joint_ptb_but_split_roll_<mode>_<seed>.md`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_joint_ptb_but_split_roll`