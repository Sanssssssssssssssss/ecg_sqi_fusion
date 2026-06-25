# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 21:18:16
- Scope: PTB-only self-test and BUT-only self-test.
- No joint training and no cross-dataset evaluation in this report.
- Every model is initialized from scratch; checkpoints are saved outputs only.
- Inference input: waveform-derived channels only.
- SQI/factor columns: training teacher/diagnostic targets only.

## Protocol Counts

PTB protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 896 |
| test | good | 572 |
| test | medium | 624 |
| train | bad | 4200 |
| train | good | 2640 |
| train | medium | 2920 |
| val | bad | 896 |
| val | good | 572 |
| val | medium | 632 |

BUT protocol:

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

## Results

| domain | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ptb_only | E1_query_only_unified_subtype_low_lr15e4 | ptb_only_all | 1 | 0.938575 | 0.933043 | 0.955867 | 0.883621 | 0.965955 | 0.939511 | 0.0311526 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_all | 1 | 0.937858 | 0.932953 | 0.949524 | 0.902059 | 0.955441 | 0.936631 | 0.0176532 |
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_all | 1 | 0.929472 | 0.923634 | 0.94556 | 0.878352 | 0.95494 | 0.928398 | 0.0297681 |
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_test | 1 | 0.913958 | 0.907437 | 0.935315 | 0.850962 | 0.944196 | 0.914889 | 0.0450237 |
| ptb_only | E1_query_only_unified_subtype_low_lr15e4 | ptb_only_test | 1 | 0.910134 | 0.902407 | 0.931818 | 0.833333 | 0.949777 | 0.91205 | 0.0450237 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_test | 1 | 0.909656 | 0.903248 | 0.921329 | 0.858974 | 0.9375 | 0.909756 | 0.0379147 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v103natural_lrgrid_24e\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v103natural_lrgrid_24e\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v103natural_lrgrid_24e\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v103natural_lrgrid_24e\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v103natural_lrgrid_24e`