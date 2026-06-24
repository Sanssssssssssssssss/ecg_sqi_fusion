# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-23 23:58:15
- Scope: PTB-only self-test and BUT-only self-test.
- No joint training and no cross-dataset evaluation in this report.
- Every model is initialized from scratch; checkpoints are saved outputs only.
- Inference input: waveform-derived channels only.
- SQI/factor columns: training teacher/diagnostic targets only.

## Protocol Counts

PTB protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 226 |
| test | good | 225 |
| test | medium | 222 |
| train | bad | 1050 |
| train | good | 1050 |
| train | medium | 1050 |
| val | bad | 224 |
| val | good | 225 |
| val | medium | 228 |

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
| but_only | E1_query_only | but_only_all | 1 | 0.929743 | 0.93963 | 0.929464 | 0.914459 | 0.990381 | 0.976698 | 0.00670406 |
| but_only | E1_query_only | but_only_test | 1 | 0.92191 | 0.929198 | 0.922872 | 0.907166 | 0.971429 | 0.967546 | 0.00746269 |
| ptb_only | E1_query_only | ptb_only_all | 1 | 0.837556 | 0.838218 | 0.836 | 0.724 | 0.952667 | 0.906405 | 0.00724638 |
| ptb_only | E1_query_only | ptb_only_test | 1 | 0.830609 | 0.830525 | 0.826667 | 0.707207 | 0.955752 | 0.900168 | 0.0101523 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e1\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e1\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e1\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e1\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e1`