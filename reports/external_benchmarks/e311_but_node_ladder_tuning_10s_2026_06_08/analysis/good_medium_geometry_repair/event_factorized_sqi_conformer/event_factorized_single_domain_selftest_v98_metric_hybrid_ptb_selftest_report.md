# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 10:25:02
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
| test | good | 225 |
| test | medium | 222 |
| train | bad | 1050 |
| train | good | 1050 |
| train | medium | 1050 |
| val | bad | 224 |
| val | good | 225 |
| val | medium | 228 |

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
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.874222 | 0.875538 | 0.909333 | 0.832 | 0.881333 | 0.87441 | 0.00909091 |
| ptb_only | E0_noquery_nohi_nolocal_noart | ptb_only_all | 1 | 0.862444 | 0.864449 | 0.858667 | 0.852 | 0.876667 | 0.864126 | 0.00530303 |
| ptb_only | E0_noquery_nohi_nolocal_noart | ptb_only_test | 1 | 0.843982 | 0.846075 | 0.813333 | 0.833333 | 0.884956 | 0.846086 | 0.0103627 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.835067 | 0.836986 | 0.826667 | 0.779279 | 0.89823 | 0.832245 | 0.0103627 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v98_metric_hybrid_ptb_selftest\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v98_metric_hybrid_ptb_selftest\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v98_metric_hybrid_ptb_selftest\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v98_metric_hybrid_ptb_selftest\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v98_metric_hybrid_ptb_selftest`