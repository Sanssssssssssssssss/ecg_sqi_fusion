# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 23:33:30
- Scope: PTB-only self-test and BUT-only self-test.
- No joint training and no cross-dataset evaluation in this report.
- Every model is initialized from scratch; checkpoints are saved outputs only.
- Inference input: waveform-derived channels only.
- SQI/factor columns: training teacher/diagnostic targets only.

## Protocol Counts

PTB protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 904 |
| test | good | 572 |
| test | medium | 624 |
| train | bad | 4200 |
| train | good | 2640 |
| train | medium | 2920 |
| val | bad | 896 |
| val | good | 572 |
| val | medium | 640 |

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
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_all | 1 | 0.932417 | 0.927848 | 0.935254 | 0.870937 | 0.9735 | 0.928442 | 0.0583997 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_all | 1 | 0.925401 | 0.919084 | 0.950846 | 0.84369 | 0.966333 | 0.920997 | 0.0436102 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_test | 1 | 0.804286 | 0.790982 | 0.723776 | 0.6875 | 0.935841 | 0.805683 | 0.157428 |
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_test | 1 | 0.791429 | 0.77383 | 0.681818 | 0.642628 | 0.963496 | 0.791056 | 0.206208 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v106pcashell_lrgrid_24e\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v106pcashell_lrgrid_24e\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v106pcashell_lrgrid_24e\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v106pcashell_lrgrid_24e\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v106pcashell_lrgrid_24e`