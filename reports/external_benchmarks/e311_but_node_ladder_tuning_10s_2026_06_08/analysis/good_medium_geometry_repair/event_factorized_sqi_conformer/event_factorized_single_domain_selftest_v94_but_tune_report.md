# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 09:06:48
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
| but_only | E4_query_highres_local_art_subtype_aux_lowaux_lr15e4 | but_only_all | 1 | 0.937136 | 0.943217 | 0.948744 | 0.902953 | 0.995818 | 0.976721 | 0.0124504 |
| but_only | E1_query_only_subtype_aux | but_only_all | 1 | 0.933046 | 0.941076 | 0.957585 | 0.877768 | 0.991635 | 0.977653 | 0.00834587 |
| but_only | E4_query_highres_local_art | but_only_all | 1 | 0.932858 | 0.940618 | 0.941165 | 0.902844 | 0.996236 | 0.977798 | 0.012724 |
| but_only | E0_noquery_nohi_nolocal_noart | but_only_all | 1 | 0.931619 | 0.936946 | 0.942827 | 0.896439 | 0.996654 | 0.974041 | 0.0176495 |
| but_only | E4_query_highres_local_art_subtype_aux | but_only_all | 1 | 0.912817 | 0.919491 | 0.917232 | 0.884498 | 0.994145 | 0.953531 | 0.0224381 |
| but_only | E4_query_highres_local_art_subtype_aux_lowaux_lr15e4 | but_only_test | 1 | 0.927903 | 0.934291 | 0.939162 | 0.893594 | 0.987755 | 0.964603 | 0.0128901 |
| but_only | E1_query_only_subtype_aux | but_only_test | 1 | 0.925281 | 0.931857 | 0.952793 | 0.86645 | 0.977551 | 0.97165 | 0.00881954 |
| but_only | E0_noquery_nohi_nolocal_noart | but_only_test | 1 | 0.925094 | 0.929432 | 0.936835 | 0.889251 | 0.987755 | 0.961581 | 0.0203528 |
| but_only | E4_query_highres_local_art | but_only_test | 1 | 0.92397 | 0.931935 | 0.931184 | 0.895223 | 0.987755 | 0.97043 | 0.0135685 |
| but_only | E4_query_highres_local_art_subtype_aux | but_only_test | 1 | 0.906929 | 0.914233 | 0.90891 | 0.88165 | 0.989796 | 0.96863 | 0.0223881 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_but_tune\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_but_tune\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_but_tune\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_but_tune\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v94_but_tune`