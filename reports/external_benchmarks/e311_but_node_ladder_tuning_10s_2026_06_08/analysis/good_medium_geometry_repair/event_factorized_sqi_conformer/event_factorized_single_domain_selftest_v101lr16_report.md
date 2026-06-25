# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 19:11:18
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
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_all | 1 | 0.959622 | 0.95478 | 0.938689 | 0.921606 | 0.999333 | 0.961145 | 0.033557 |
| ptb_only | E1_query_only_unified_subtype_low_lr15e4 | ptb_only_all | 1 | 0.955613 | 0.950109 | 0.926004 | 0.919694 | 0.999333 | 0.957579 | 0.033557 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_all | 1 | 0.947881 | 0.941018 | 0.893235 | 0.924474 | 0.998667 | 0.948822 | 0.033557 |
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_test | 1 | 0.944762 | 0.937581 | 0.923077 | 0.884615 | 1 | 0.943122 | 0.0353982 |
| ptb_only | E1_query_only_unified_subtype_low_lr15e4 | ptb_only_test | 1 | 0.942857 | 0.935358 | 0.909091 | 0.891026 | 1 | 0.94246 | 0.0353982 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_test | 1 | 0.937143 | 0.929152 | 0.895105 | 0.891026 | 0.995575 | 0.936177 | 0.0353982 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101lr16\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101lr16\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101lr16\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v101lr16\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v101lr16`