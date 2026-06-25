# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 23:03:10
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
| ptb_only | E1_query_only_unified_subtype_low_lr15e4 | ptb_only_all | 1 | 0.948525 | 0.943564 | 0.94556 | 0.911329 | 0.976333 | 0.949165 | 0.0253737 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_all | 1 | 0.946377 | 0.941476 | 0.940011 | 0.907744 | 0.977333 | 0.947605 | 0.0382343 |
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_all | 1 | 0.940793 | 0.935311 | 0.91649 | 0.901769 | 0.983333 | 0.94121 | 0.0545707 |
| ptb_only | E1_query_only_unified_subtype_low_lr75e5 | ptb_only_test | 1 | 0.912381 | 0.903899 | 0.879371 | 0.855769 | 0.972345 | 0.911808 | 0.0703297 |
| ptb_only | E1_query_only_unified_subtype_low_lr15e4 | ptb_only_test | 1 | 0.911905 | 0.904432 | 0.914336 | 0.849359 | 0.95354 | 0.910673 | 0.0505495 |
| ptb_only | E1_query_only_unified_subtype_low_lr1e4 | ptb_only_test | 1 | 0.908571 | 0.900263 | 0.893357 | 0.849359 | 0.959071 | 0.908384 | 0.0615385 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v105natural_brutal_lrgrid_24e\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v105natural_brutal_lrgrid_24e\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v105natural_brutal_lrgrid_24e\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v105natural_brutal_lrgrid_24e\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v105natural_brutal_lrgrid_24e`