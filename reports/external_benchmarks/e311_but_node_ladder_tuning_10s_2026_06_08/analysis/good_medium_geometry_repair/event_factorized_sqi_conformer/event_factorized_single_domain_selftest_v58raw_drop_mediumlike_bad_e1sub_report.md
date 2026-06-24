# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-23 12:05:31
- Scope: PTB-only self-test and BUT-only self-test.
- No joint training and no cross-dataset evaluation in this report.
- Every model is initialized from scratch; checkpoints are saved outputs only.
- Inference input: waveform-derived channels only.
- SQI/factor columns: training teacher/diagnostic targets only.

## Protocol Counts

PTB protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 452 |
| test | good | 450 |
| test | medium | 450 |
| train | bad | 2100 |
| train | good | 2100 |
| train | medium | 2100 |
| val | bad | 448 |
| val | good | 450 |
| val | medium | 450 |

BUT protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 490 |
| test | good | 3008 |
| test | medium | 1842 |
| train | bad | 1656 |
| train | good | 10530 |
| train | medium | 6449 |
| val | bad | 245 |
| val | good | 1504 |
| val | medium | 921 |

## Results

| domain | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_only | E1_query_only_subtype_aux | but_only_all | 1 | 0.91383 | 0.907596 | 0.929797 | 0.865936 | 0.997909 | 0.966421 | 0.0573266 |
| but_only | E1_query_only_subtype_aux | but_only_test | 1 | 0.909176 | 0.903075 | 0.926862 | 0.858306 | 0.991837 | 0.957819 | 0.0603799 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.993778 | 0.993778 | 0.997 | 0.997667 | 0.986667 | 0.996475 | 0.00278996 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.984467 | 0.984459 | 0.995556 | 0.988889 | 0.969027 | 0.9908 | 0.0108108 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v58raw_drop_mediumlike_bad_e1sub\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v58raw_drop_mediumlike_bad_e1sub\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v58raw_drop_mediumlike_bad_e1sub\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v58raw_drop_mediumlike_bad_e1sub\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v58raw_drop_mediumlike_bad_e1sub`