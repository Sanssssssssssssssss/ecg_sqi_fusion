# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 19:05:19
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
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_all | 1 | 0.962772 | 0.958597 | 0.937632 | 0.934034 | 0.998667 | 0.963172 | 0.033557 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.955613 | 0.950183 | 0.900634 | 0.942639 | 0.999333 | 0.955211 | 0.033557 |
| ptb_only | E1_query_only_unified_subtype_low | ptb_only_test | 1 | 0.940952 | 0.933696 | 0.888112 | 0.910256 | 0.995575 | 0.940476 | 0.0353982 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.937143 | 0.92855 | 0.853147 | 0.923077 | 1 | 0.935185 | 0.0353982 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro16\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro16\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro16\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101micro16\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v101micro16`