# Event-Factorized SQI Conformer Single-Domain Self-Test

- Generated: 2026-06-24 18:46:25
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
| test | good | 165 |
| test | medium | 178 |
| train | bad | 1050 |
| train | good | 770 |
| train | medium | 840 |
| val | bad | 224 |
| val | good | 165 |
| val | medium | 182 |

BUT protocol:

| split | class_name | n |
| --- | --- | --- |
| test | bad | 118 |
| test | good | 1004 |
| test | medium | 2077 |
| train | bad | 3963 |
| train | good | 9603 |
| train | medium | 4145 |
| val | bad | 1 |
| val | good | 621 |
| val | medium | 43 |

## Results

| domain | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ptb_only | E1_query_only_subtype_aux | ptb_only_all | 1 | 0.897632 | 0.891009 | 0.834545 | 0.828333 | 0.999333 | 0.899159 | 0.130531 |
| ptb_only | E1_query_only | ptb_only_all | 1 | 0.895 | 0.887765 | 0.862727 | 0.795833 | 0.998 | 0.896651 | 0.130531 |
| ptb_only | E1_query_only_subtype_aux | ptb_only_test | 1 | 0.899824 | 0.891899 | 0.836364 | 0.831461 | 1 | 0.898962 | 0.110294 |
| ptb_only | E1_query_only | ptb_only_test | 1 | 0.898067 | 0.889913 | 0.878788 | 0.792135 | 0.995575 | 0.898962 | 0.110294 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99e1base\single_domain_selftest_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99e1base\single_domain_selftest_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99e1base\single_domain_selftest_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v99e1base\single_domain_selftest_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_single_domain_selftest_v99e1base`