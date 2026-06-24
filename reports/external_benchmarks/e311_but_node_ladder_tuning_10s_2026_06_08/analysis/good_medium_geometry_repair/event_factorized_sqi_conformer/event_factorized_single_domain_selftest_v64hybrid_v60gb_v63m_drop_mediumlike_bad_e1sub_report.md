# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-23 15:36:15
- Scope: PTB-only self-test and BUT-only self-test.
- No joint training and no cross-dataset evaluation in this report.
- Every model is initialized from scratch; checkpoints are saved outputs only.
- Inference input: waveform-derived channels only.
- SQI/factor columns: training teacher/diagnostic targets only.

## Protocol Counts

PTB protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 452 |
| test | good | 450 |
| test | medium | 450 |
| train | bad | 2100 |
| train | good | 2100 |
| train | medium | 2100 |
| val | bad | 448 |
| val | good | 450 |
| val | medium | 450 |

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
| but_only | E1_query_only_subtype_aux | but_only_all | 1 | 0.916795 | 0.925341 | 0.959846 | 0.825879 | 0.996236 | 0.973047 | 0.0138186 |
| but_only | E1_query_only_subtype_aux | but_only_test | 1 | 0.911236 | 0.918476 | 0.953457 | 0.820847 | 0.991837 | 0.970138 | 0.0169607 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.996444 | 0.996442 | 1 | 0.998333 | 0.991 | 0.997917 | 0.000404694 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.993343 | 0.993331 | 1 | 1 | 0.980088 | 0.995868 | 0 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub`