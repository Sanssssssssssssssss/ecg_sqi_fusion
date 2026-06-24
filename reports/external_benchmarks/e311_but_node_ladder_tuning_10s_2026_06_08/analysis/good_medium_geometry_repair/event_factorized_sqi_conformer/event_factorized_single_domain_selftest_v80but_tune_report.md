# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 03:09:00
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
| but_only | E4_query_highres_local_art_subtype_aux_lowaux_lr15e4 | but_only_all | 1 | 0.927454 | 0.936892 | 0.910983 | 0.93617 | 0.997491 | 0.976445 | 0.0128609 |
| but_only | E4_query_highres_local_art_subtype_aux_lr15e4 | but_only_all | 1 | 0.910527 | 0.914699 | 0.909919 | 0.888949 | 0.997491 | 0.962663 | 0.0326994 |
| but_only | E4_query_highres_local_art_subtype_aux_lowaux_lr15e4 | but_only_test | 1 | 0.920037 | 0.928934 | 0.904588 | 0.927253 | 0.987755 | 0.962294 | 0.0122117 |
| but_only | E4_query_highres_local_art_subtype_aux_lr15e4 | but_only_test | 1 | 0.905993 | 0.908513 | 0.907247 | 0.881107 | 0.991837 | 0.958405 | 0.0352782 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80but_tune\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80but_tune\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80but_tune\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v80but_tune\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v80but_tune`