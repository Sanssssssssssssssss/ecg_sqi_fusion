# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 08:40:41
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
| ptb_only | E4_query_highres_local_art_subtype_aux | ptb_only_all | 1 | 0.889556 | 0.889555 | 0.847333 | 0.822 | 0.999333 | 0.891109 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr15e4 | ptb_only_all | 1 | 0.886889 | 0.886861 | 0.850667 | 0.810667 | 0.999333 | 0.889522 | 0 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.884889 | 0.884845 | 0.804 | 0.851333 | 0.999333 | 0.885938 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr75e5 | ptb_only_all | 1 | 0.881111 | 0.881075 | 0.844 | 0.8 | 0.999333 | 0.88187 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lowaux_lr15e4 | ptb_only_all | 1 | 0.880667 | 0.88062 | 0.797333 | 0.845333 | 0.999333 | 0.882003 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr1e4 | ptb_only_all | 1 | 0.878444 | 0.878465 | 0.818 | 0.818 | 0.999333 | 0.880322 | 0 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.870728 | 0.870119 | 0.768889 | 0.842342 | 1 | 0.869645 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux | ptb_only_test | 1 | 0.867756 | 0.867253 | 0.804444 | 0.797297 | 1 | 0.866771 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr15e4 | ptb_only_test | 1 | 0.867756 | 0.867261 | 0.8 | 0.801802 | 1 | 0.8686 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lowaux_lr15e4 | ptb_only_test | 1 | 0.864785 | 0.86406 | 0.751111 | 0.842342 | 1 | 0.865726 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr75e5 | ptb_only_test | 1 | 0.857355 | 0.856817 | 0.786667 | 0.783784 | 1 | 0.857628 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr1e4 | ptb_only_test | 1 | 0.851412 | 0.850731 | 0.742222 | 0.810811 | 1 | 0.85162 | 0 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_subtypefixed_grid_ptb\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_subtypefixed_grid_ptb\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_subtypefixed_grid_ptb\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_subtypefixed_grid_ptb\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v94_subtypefixed_grid_ptb`