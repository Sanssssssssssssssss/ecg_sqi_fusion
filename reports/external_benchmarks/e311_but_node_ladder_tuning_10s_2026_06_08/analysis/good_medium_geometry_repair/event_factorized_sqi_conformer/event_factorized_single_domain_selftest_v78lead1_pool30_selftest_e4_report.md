# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 00:02:28
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
| but_only | E4_query_highres_local_art | but_only_all | 1 | 0.925014 | 0.933692 | 0.91537 | 0.922058 | 0.997072 | 0.970963 | 0.0143659 |
| but_only | E4_query_highres_local_art | but_only_test | 1 | 0.916667 | 0.92484 | 0.906582 | 0.913138 | 0.991837 | 0.968307 | 0.0169607 |
| ptb_only | E4_query_highres_local_art | ptb_only_all | 1 | 0.843556 | 0.84302 | 0.702 | 0.854667 | 0.974 | 0.917269 | 0.00805153 |
| ptb_only | E4_query_highres_local_art | ptb_only_test | 1 | 0.821694 | 0.820657 | 0.675556 | 0.810811 | 0.977876 | 0.903344 | 0.0101523 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4`