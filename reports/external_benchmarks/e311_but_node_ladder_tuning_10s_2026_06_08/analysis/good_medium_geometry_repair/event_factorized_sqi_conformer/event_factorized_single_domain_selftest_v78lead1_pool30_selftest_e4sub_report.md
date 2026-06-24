# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 00:13:05
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
| but_only | E4_query_highres_local_art_subtype_aux | but_only_all | 1 | 0.917395 | 0.928967 | 0.922151 | 0.889709 | 0.994145 | 0.97576 | 0.00998769 |
| but_only | E4_query_highres_local_art_subtype_aux | but_only_test | 1 | 0.913109 | 0.92395 | 0.916888 | 0.886536 | 0.989796 | 0.97261 | 0.0128901 |
| ptb_only | E4_query_highres_local_art_subtype_aux | ptb_only_all | 1 | 0.863556 | 0.862929 | 0.835333 | 0.770667 | 0.984667 | 0.926947 | 0.0161031 |
| ptb_only | E4_query_highres_local_art_subtype_aux | ptb_only_test | 1 | 0.845468 | 0.844346 | 0.813333 | 0.738739 | 0.982301 | 0.914815 | 0.0152284 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4sub\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4sub\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4sub\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4sub\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v78lead1_pool30_selftest_e4sub`