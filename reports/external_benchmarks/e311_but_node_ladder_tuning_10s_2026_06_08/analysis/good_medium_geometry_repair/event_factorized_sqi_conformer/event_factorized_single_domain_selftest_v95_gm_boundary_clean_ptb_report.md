# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 09:25:50
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
| test | good | 135 |
| test | medium | 148 |
| train | bad | 1050 |
| train | good | 630 |
| train | medium | 700 |
| val | bad | 224 |
| val | good | 135 |
| val | medium | 152 |

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
| ptb_only | E4_query_highres_local_art_subtype_aux_lr15e4 | ptb_only_all | 1 | 0.987059 | 0.984527 | 0.981111 | 0.973 | 1 | 0.987483 | 0 |
| ptb_only | E1_query_only | ptb_only_all | 1 | 0.982941 | 0.979609 | 0.976667 | 0.963 | 1 | 0.983839 | 0 |
| ptb_only | E3_query_highres_local | ptb_only_all | 1 | 0.973824 | 0.968906 | 0.961111 | 0.946 | 1 | 0.975619 | 0.00142857 |
| ptb_only | E1_query_only | ptb_only_test | 1 | 0.962672 | 0.955186 | 0.940741 | 0.925676 | 1 | 0.961749 | 0 |
| ptb_only | E4_query_highres_local_art_subtype_aux_lr15e4 | ptb_only_test | 1 | 0.960707 | 0.952786 | 0.925926 | 0.932432 | 1 | 0.959699 | 0 |
| ptb_only | E3_query_highres_local | ptb_only_test | 1 | 0.956778 | 0.948122 | 0.933333 | 0.912162 | 1 | 0.956967 | 0 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v95_gm_boundary_clean_ptb\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v95_gm_boundary_clean_ptb\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v95_gm_boundary_clean_ptb\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v95_gm_boundary_clean_ptb\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v95_gm_boundary_clean_ptb`