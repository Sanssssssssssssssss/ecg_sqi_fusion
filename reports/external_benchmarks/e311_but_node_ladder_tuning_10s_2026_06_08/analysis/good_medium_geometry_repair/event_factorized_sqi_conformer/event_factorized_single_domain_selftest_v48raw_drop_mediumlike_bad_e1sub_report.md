# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-23 05:42:53
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
| test | good | 452 |
| test | medium | 450 |
| train | bad | 2100 |
| train | good | 2100 |
| train | medium | 2100 |
| val | bad | 448 |
| val | good | 448 |
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
| but_only | E1_query_only_subtype_aux | but_only_all | 1 | 0.920961 | 0.930991 | 0.904335 | 0.928571 | 0.996236 | 0.969289 | 0.0146395 |
| but_only | E1_query_only_subtype_aux | but_only_test | 1 | 0.915169 | 0.926131 | 0.896609 | 0.92671 | 0.985714 | 0.966504 | 0.0108548 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.997667 | 0.997667 | 0.999 | 0.997333 | 0.996667 | 0.998719 | 0.00453104 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.997046 | 0.997047 | 0.997788 | 0.995556 | 0.997788 | 0.998535 | 0.0096463 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v48raw_drop_mediumlike_bad_e1sub\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v48raw_drop_mediumlike_bad_e1sub\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v48raw_drop_mediumlike_bad_e1sub\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v48raw_drop_mediumlike_bad_e1sub\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v48raw_drop_mediumlike_bad_e1sub`