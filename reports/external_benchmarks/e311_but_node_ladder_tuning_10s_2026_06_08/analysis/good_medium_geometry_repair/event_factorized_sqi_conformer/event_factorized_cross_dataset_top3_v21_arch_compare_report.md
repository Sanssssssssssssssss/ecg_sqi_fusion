# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-21 02:12:48
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v21_gm_featurematched\protocol_v21_pc3000_s20260621`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E2_query_highres | cross_all | 1 | 0.861111 | 0.864158 | 0.851 | 1 | 0.732333 | 0.925246 |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.841111 | 0.843945 | 0.844333 | 1 | 0.679 | 0.921833 |
| but_to_ptb | E2_query_highres | cross_test | 1 | 0.840199 | 0.84357 | 0.80531 | 1 | 0.695556 | 0.906414 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.823864 | 0.826732 | 0.79646 | 1 | 0.653333 | 0.901894 |
| ptb_to_but | E2_query_highres | cross_all | 1 | 0.709896 | 0.712796 | 0.748397 | 0.451875 | 1 | 0.62709 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.682642 | 0.711363 | 0.626559 | 0.576536 | 0.999755 | 0.560951 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.593936 | 0.569476 | 0.967131 | 0.390467 | 1 | 0.79298 |
| ptb_to_but | E2_query_highres | cross_test | 1 | 0.510785 | 0.480216 | 0.987052 | 0.252768 | 1 | 0.77219 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v21_arch_compare\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v21_arch_compare\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v21_arch_compare\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v21_arch_compare\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v21_arch_compare`