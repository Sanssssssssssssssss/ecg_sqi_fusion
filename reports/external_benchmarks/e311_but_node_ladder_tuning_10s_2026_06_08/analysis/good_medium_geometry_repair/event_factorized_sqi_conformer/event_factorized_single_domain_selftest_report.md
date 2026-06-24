# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-20 23:12:36
- Scope: PTB-only self-test and BUT-only self-test.
- No joint training and no cross-dataset evaluation in this report.
- Every model is initialized from scratch; checkpoints are saved outputs only.
- Inference input: waveform-derived channels only.
- SQI/factor columns: training teacher/diagnostic targets only.

## Protocol Counts

PTB protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 514 |
| test | good | 452 |
| test | medium | 506 |
| train | bad | 2047 |
| train | good | 2105 |
| train | medium | 2014 |
| val | bad | 439 |
| val | good | 443 |
| val | medium | 480 |

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
| but_only | E1_query_only | but_only_all | 3 | 0.973627 | 0.975067 | 0.995725 | 0.922533 | 0.991262 | 0.962047 | 0 |
| but_only | E1_query_only | but_only_test | 3 | 0.937793 | 0.871073 | 0.999004 | 0.921361 | 0.706215 | 0.908112 | 0 |
| ptb_only | E1_query_only | ptb_only_all | 3 | 0.996593 | 0.996591 | 0.999556 | 0.992556 | 0.997667 | 0.997645 | 0.00482921 |
| ptb_only | E1_query_only | ptb_only_test | 3 | 0.991168 | 0.991456 | 0.998525 | 0.981555 | 0.994163 | 0.990605 | 0.0157171 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest`