# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 08:49:56
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
| ptb_only | E1_query_only | ptb_only_all | 1 | 0.893111 | 0.893107 | 0.825333 | 0.854667 | 0.999333 | 0.894841 | 0 |
| ptb_only | E3_query_highres_local | ptb_only_all | 1 | 0.891111 | 0.890966 | 0.892 | 0.788 | 0.993333 | 0.894085 | 0 |
| ptb_only | E2_query_highres | ptb_only_all | 1 | 0.889333 | 0.889298 | 0.862667 | 0.807333 | 0.998 | 0.890992 | 0 |
| ptb_only | E0_noquery_nohi_nolocal_noart | ptb_only_all | 1 | 0.888222 | 0.888114 | 0.803333 | 0.862 | 0.999333 | 0.889442 | 0 |
| ptb_only | E4_query_highres_local_art | ptb_only_all | 1 | 0.884222 | 0.884184 | 0.800667 | 0.853333 | 0.998667 | 0.884427 | 0 |
| ptb_only | E0_noquery_nohi_nolocal_noart | ptb_only_test | 1 | 0.870728 | 0.869999 | 0.768889 | 0.842342 | 1 | 0.869906 | 0 |
| ptb_only | E4_query_highres_local_art | ptb_only_test | 1 | 0.870728 | 0.869986 | 0.755556 | 0.855856 | 1 | 0.870167 | 0 |
| ptb_only | E1_query_only | ptb_only_test | 1 | 0.869242 | 0.868754 | 0.795556 | 0.810811 | 1 | 0.869906 | 0 |
| ptb_only | E3_query_highres_local | ptb_only_test | 1 | 0.861813 | 0.861175 | 0.844444 | 0.747748 | 0.99115 | 0.860502 | 0 |
| ptb_only | E2_query_highres | ptb_only_test | 1 | 0.860327 | 0.859726 | 0.826667 | 0.756757 | 0.995575 | 0.858934 | 0 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_phase1_baselines_ptb\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_phase1_baselines_ptb\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_phase1_baselines_ptb\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v94_phase1_baselines_ptb\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v94_phase1_baselines_ptb`