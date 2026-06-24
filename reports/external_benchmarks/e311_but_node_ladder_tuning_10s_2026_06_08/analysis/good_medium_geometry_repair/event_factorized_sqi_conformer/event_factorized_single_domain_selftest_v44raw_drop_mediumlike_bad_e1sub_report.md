# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-23 05:17:03
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
| but_only | E1_query_only_subtype_aux | but_only_all | 1 | 0.928054 | 0.938019 | 0.926805 | 0.913048 | 0.993726 | 0.97363 | 0.00875633 |
| but_only | E1_query_only_subtype_aux | but_only_test | 1 | 0.922846 | 0.932115 | 0.92387 | 0.904995 | 0.983673 | 0.966416 | 0.00881954 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.998889 | 0.998889 | 0.999333 | 0.998667 | 0.998667 | 0.999415 | 0.00221435 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.997046 | 0.997044 | 1 | 0.995556 | 0.995575 | 0.998321 | 0.0060423 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v44raw_drop_mediumlike_bad_e1sub\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v44raw_drop_mediumlike_bad_e1sub\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v44raw_drop_mediumlike_bad_e1sub\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v44raw_drop_mediumlike_bad_e1sub\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v44raw_drop_mediumlike_bad_e1sub`