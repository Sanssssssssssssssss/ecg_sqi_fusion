# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 02:19:26
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
| but_only | E4_query_highres_local_art_subtype_aux_lr1e4 | but_only_all | 1 | 0.929105 | 0.939108 | 0.91557 | 0.934216 | 0.994563 | 0.974489 | 0.0109454 |
| but_only | E4_query_highres_local_art_subtype_aux_lr1e4 | but_only_test | 1 | 0.918539 | 0.927319 | 0.908577 | 0.916938 | 0.985714 | 0.967491 | 0.0142469 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr1e4 | ptb_only_all | 1 | 0.803556 | 0.800881 | 0.553333 | 0.928 | 0.929333 | 0.854607 | 0.00309119 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr1e4 | ptb_only_test | 1 | 0.953439 | 0.935055 | 0.881481 | 0.939227 | 0.986607 | 0.960958 | 0.0169492 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80lr\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80lr\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80lr\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80lr\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v80lr`