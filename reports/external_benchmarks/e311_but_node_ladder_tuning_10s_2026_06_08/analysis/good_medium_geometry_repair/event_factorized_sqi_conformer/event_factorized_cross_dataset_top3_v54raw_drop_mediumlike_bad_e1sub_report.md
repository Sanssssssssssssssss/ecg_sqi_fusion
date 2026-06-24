# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 09:28:38
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v54raw\protocol_v54raw_pc3000_s20260644`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.750444 | 0.753056 | 0.737667 | 0.966333 | 0.547333 | 0.829796 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.764053 | 0.765394 | 0.775556 | 0.975556 | 0.542035 | 0.836564 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.638206 | 0.664707 | 0.534237 | 0.720256 | 0.976161 | 0.626464 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.643446 | 0.672251 | 0.532247 | 0.737242 | 0.973469 | 0.695679 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v54raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v54raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v54raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v54raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v54raw_drop_mediumlike_bad_e1sub`