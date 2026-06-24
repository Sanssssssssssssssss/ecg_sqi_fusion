# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 16:08:45
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\protocol_v60raw_pc3000_s20260650`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.783222 | 0.788234 | 0.751333 | 0.943667 | 0.654667 | 0.858636 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.778107 | 0.782355 | 0.764444 | 0.937778 | 0.632743 | 0.852142 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.640495 | 0.677725 | 0.580907 | 0.647199 | 0.989544 | 0.663079 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.64588 | 0.684959 | 0.580452 | 0.662324 | 0.985714 | 0.737651 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_e1sub_recordbalanced\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_e1sub_recordbalanced\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_e1sub_recordbalanced\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_e1sub_recordbalanced\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v60raw_drop_mediumlike_bad_e1sub_recordbalanced`