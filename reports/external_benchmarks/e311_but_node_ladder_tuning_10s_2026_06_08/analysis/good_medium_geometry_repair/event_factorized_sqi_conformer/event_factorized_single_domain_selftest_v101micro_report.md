# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 19:01:32
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
| test | good | 143 |
| test | medium | 156 |
| train | bad | 1050 |
| train | good | 660 |
| train | medium | 730 |
| val | bad | 224 |
| val | good | 143 |
| val | medium | 160 |

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
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_all | 1 | 0.949313 | 0.942603 | 0.910148 | 0.913002 | 0.999333 | 0.949279 | 0.033557 |
| ptb_only | E1_query_only_unified_subtype_tiny | ptb_only_all | 1 | 0.94559 | 0.938363 | 0.89852 | 0.913002 | 0.998 | 0.945799 | 0.033557 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.936999 | 0.928001 | 0.853066 | 0.924474 | 0.998667 | 0.937058 | 0.033557 |
| ptb_only | E1_query_only | ptb_only_all | 1 | 0.928981 | 0.918013 | 0.79704 | 0.947419 | 0.999333 | 0.927935 | 0.033557 |
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_test | 1 | 0.935238 | 0.926334 | 0.881119 | 0.891026 | 1 | 0.934193 | 0.0353982 |
| ptb_only | E1_query_only_unified_subtype_tiny | ptb_only_test | 1 | 0.933333 | 0.924646 | 0.881119 | 0.891026 | 0.995575 | 0.932209 | 0.0353982 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.931429 | 0.922378 | 0.86014 | 0.903846 | 0.995575 | 0.930225 | 0.0353982 |
| ptb_only | E1_query_only | ptb_only_test | 1 | 0.929524 | 0.919261 | 0.811189 | 0.935897 | 1 | 0.926587 | 0.0353982 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v101micro`