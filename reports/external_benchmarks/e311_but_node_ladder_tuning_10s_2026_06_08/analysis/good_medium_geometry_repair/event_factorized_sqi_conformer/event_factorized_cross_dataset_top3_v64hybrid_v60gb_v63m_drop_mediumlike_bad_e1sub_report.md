# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 15:41:35
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v64hybrid_v60gb_v63m\protocol_v64hybrid_v60gb_v63m_pc3000_s20260654`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.599222 | 0.564633 | 0.623333 | 0.994 | 0.180333 | 0.645385 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.613905 | 0.582939 | 0.651111 | 0.993333 | 0.199115 | 0.660251 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.635166 | 0.67956 | 0.536564 | 0.705493 | 0.984525 | 0.608569 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.638764 | 0.685668 | 0.533577 | 0.719327 | 0.981633 | 0.670382 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v64hybrid_v60gb_v63m_drop_mediumlike_bad_e1sub`