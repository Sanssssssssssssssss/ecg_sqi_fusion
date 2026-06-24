# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 03:21:52
- Scope: PTB-only self-test and BUT-only self-test.
- No joint training and no cross-dataset evaluation in this report.
- Every model is initialized from scratch; checkpoints are saved outputs only.
- Inference input: waveform-derived channels only.
- SQI/factor columns: training teacher/diagnostic targets only.

## Protocol Counts

PTB protocol:

| split | class_name | n |
| --- | --- | --- |
| holdout | bad | 150 |
| holdout | good | 600 |
| holdout | medium | 250 |
| test | bad | 448 |
| test | good | 135 |
| test | medium | 362 |
| train | bad | 700 |
| train | good | 630 |
| train | medium | 700 |
| val | bad | 202 |
| val | good | 135 |
| val | medium | 188 |

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
| ptb_only | E4_query_highres_local_art_subtype_aux_lr1e4 | ptb_only_all | 1 | 0.804667 | 0.802636 | 0.572 | 0.902 | 0.94 | 0.857255 | 0.00386399 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr1e4 | ptb_only_test | 1 | 0.948148 | 0.925066 | 0.896296 | 0.911602 | 0.993304 | 0.966245 | 0 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80ptb_lr1e4_30e\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80ptb_lr1e4_30e\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80ptb_lr1e4_30e\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80ptb_lr1e4_30e\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v80ptb_lr1e4_30e`