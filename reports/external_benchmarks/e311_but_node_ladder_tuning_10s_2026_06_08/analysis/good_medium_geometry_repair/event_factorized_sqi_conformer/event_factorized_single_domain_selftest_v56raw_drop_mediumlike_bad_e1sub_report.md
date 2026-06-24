# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-23 10:48:11
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
| but_only | E1_query_only_subtype_aux | but_only_all | 1 | 0.921336 | 0.929576 | 0.955525 | 0.846939 | 0.99289 | 0.971201 | 0.0124504 |
| but_only | E1_query_only_subtype_aux | but_only_test | 1 | 0.917603 | 0.923324 | 0.952793 | 0.842562 | 0.983673 | 0.956264 | 0.0149254 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.997667 | 0.997667 | 0.999 | 0.998333 | 0.995667 | 0.998727 | 0.000805153 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.992604 | 0.992607 | 0.997778 | 0.995556 | 0.984513 | 0.995777 | 0.00268817 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v56raw_drop_mediumlike_bad_e1sub\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v56raw_drop_mediumlike_bad_e1sub\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v56raw_drop_mediumlike_bad_e1sub\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v56raw_drop_mediumlike_bad_e1sub\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v56raw_drop_mediumlike_bad_e1sub`