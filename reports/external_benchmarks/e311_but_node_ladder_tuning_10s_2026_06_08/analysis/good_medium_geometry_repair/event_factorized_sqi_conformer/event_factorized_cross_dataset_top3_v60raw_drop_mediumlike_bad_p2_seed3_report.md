# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 17:18:05
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
| but_to_ptb | P2_ecg_beat_rhythm_mask | cross_all | 3 | 0.765111 | 0.765296 | 0.661222 | 0.978667 | 0.655444 | 0.828704 |
| but_to_ptb | P2_ecg_beat_rhythm_mask | cross_test | 3 | 0.766026 | 0.765615 | 0.678519 | 0.974815 | 0.64528 | 0.827518 |
| ptb_to_but | P2_ecg_beat_rhythm_mask | cross_all | 3 | 0.656959 | 0.686682 | 0.575899 | 0.705095 | 0.981458 | 0.639487 |
| ptb_to_but | P2_ecg_beat_rhythm_mask | cross_test | 3 | 0.660237 | 0.69138 | 0.574357 | 0.715346 | 0.980272 | 0.700082 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2_seed3\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2_seed3\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2_seed3\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2_seed3\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v60raw_drop_mediumlike_bad_p2_seed3`