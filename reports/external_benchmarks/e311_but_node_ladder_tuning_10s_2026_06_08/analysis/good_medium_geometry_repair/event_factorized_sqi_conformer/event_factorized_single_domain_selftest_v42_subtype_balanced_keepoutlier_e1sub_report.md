# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-23 04:24:41
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
| test | bad | 1031 |
| test | good | 3008 |
| test | medium | 1842 |
| train | bad | 3608 |
| train | good | 10530 |
| train | medium | 6449 |
| val | bad | 517 |
| val | good | 1504 |
| val | medium | 921 |

## Results

| domain | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_only | E1_query_only_subtype_aux | but_only_all | 1 | 0.938694 | 0.94309 | 0.951602 | 0.884064 | 0.998642 | 0.975561 | 0.0187483 |
| but_only | E1_query_only_subtype_aux | but_only_test | 1 | 0.932154 | 0.936738 | 0.944814 | 0.876221 | 0.99515 | 0.969959 | 0.0188299 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.997111 | 0.997111 | 0.999333 | 0.994333 | 0.997667 | 0.998513 | 0.00670241 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.990399 | 0.990399 | 0.997788 | 0.982222 | 0.99115 | 0.994932 | 0.0229226 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v42_subtype_balanced_keepoutlier_e1sub\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v42_subtype_balanced_keepoutlier_e1sub\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v42_subtype_balanced_keepoutlier_e1sub\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v42_subtype_balanced_keepoutlier_e1sub\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v42_subtype_balanced_keepoutlier_e1sub`