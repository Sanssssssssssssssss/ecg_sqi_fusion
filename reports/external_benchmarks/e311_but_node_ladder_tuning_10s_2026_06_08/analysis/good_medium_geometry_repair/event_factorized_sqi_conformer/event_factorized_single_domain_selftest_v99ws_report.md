# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 18:43:13
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
| test | good | 165 |
| test | medium | 178 |
| train | bad | 1050 |
| train | good | 770 |
| train | medium | 840 |
| val | bad | 224 |
| val | good | 165 |
| val | medium | 182 |

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
| ptb_only | E1_query_only_unified_subtype_mid | ptb_only_all | 1 | 0.909474 | 0.90415 | 0.882727 | 0.821667 | 0.999333 | 0.911203 | 0.130531 |
| ptb_only | E1_query_only_unified_subtype_tiny | ptb_only_all | 1 | 0.908421 | 0.902685 | 0.897273 | 0.805 | 0.999333 | 0.909773 | 0.130531 |
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_all | 1 | 0.905263 | 0.899791 | 0.838182 | 0.849167 | 0.999333 | 0.905612 | 0.130531 |
| ptb_only | E1_query_only_unified_subtype_tiny | ptb_only_test | 1 | 0.912127 | 0.905375 | 0.90303 | 0.808989 | 1 | 0.910562 | 0.110294 |
| ptb_only | E1_query_only_unified_subtype_mid | ptb_only_test | 1 | 0.908612 | 0.901557 | 0.884848 | 0.814607 | 1 | 0.906899 | 0.110294 |
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_test | 1 | 0.903339 | 0.89599 | 0.824242 | 0.853933 | 1 | 0.90232 | 0.110294 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99ws\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99ws\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99ws\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99ws\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v99ws`