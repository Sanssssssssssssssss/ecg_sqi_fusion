# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 18:54:17
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
| test | good | 151 |
| test | medium | 164 |
| train | bad | 1050 |
| train | good | 700 |
| train | medium | 770 |
| val | bad | 224 |
| val | good | 151 |
| val | medium | 168 |

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
| ptb_only | E1_query_only_unified_subtype_tiny | ptb_only_all | 1 | 0.939512 | 0.934117 | 0.901198 | 0.893829 | 0.998667 | 0.940965 | 0.071608 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.936737 | 0.930952 | 0.876248 | 0.906534 | 0.999333 | 0.937044 | 0.071608 |
| ptb_only | E1_query_only | ptb_only_all | 1 | 0.934517 | 0.92813 | 0.912176 | 0.866606 | 0.999333 | 0.936975 | 0.071608 |
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_all | 1 | 0.922863 | 0.914863 | 0.867265 | 0.869328 | 0.999333 | 0.924684 | 0.0728643 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.933457 | 0.926205 | 0.860927 | 0.908537 | 1 | 0.932946 | 0.0650407 |
| ptb_only | E1_query_only_unified_subtype_tiny | ptb_only_test | 1 | 0.927911 | 0.92026 | 0.86755 | 0.890244 | 0.995575 | 0.9294 | 0.0650407 |
| ptb_only | E1_query_only | ptb_only_test | 1 | 0.924214 | 0.91533 | 0.89404 | 0.847561 | 1 | 0.924242 | 0.0650407 |
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_test | 1 | 0.922366 | 0.913245 | 0.874172 | 0.859756 | 1 | 0.923598 | 0.0650407 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v100bh\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v100bh\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v100bh\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v100bh\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v100bh`