# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 16:46:12
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
| but_to_ptb | P3_clean_noisy_physio_distill | cross_all | 1 | 0.844667 | 0.845047 | 0.653667 | 0.974 | 0.906333 | 0.9057 |
| but_to_ptb | P2_ecg_beat_rhythm_mask | cross_all | 1 | 0.717889 | 0.722318 | 0.63 | 0.975667 | 0.548 | 0.797502 |
| but_to_ptb | P3_clean_noisy_physio_distill | cross_test | 1 | 0.846154 | 0.846882 | 0.671111 | 0.962222 | 0.904867 | 0.907295 |
| but_to_ptb | P2_ecg_beat_rhythm_mask | cross_test | 1 | 0.719675 | 0.724225 | 0.646667 | 0.968889 | 0.544248 | 0.798117 |
| ptb_to_but | P2_ecg_beat_rhythm_mask | cross_all | 1 | 0.719685 | 0.736923 | 0.680162 | 0.716565 | 0.980343 | 0.704747 |
| ptb_to_but | P3_clean_noisy_physio_distill | cross_all | 1 | 0.584537 | 0.632595 | 0.37741 | 0.819692 | 0.981598 | 0.470463 |
| ptb_to_but | P2_ecg_beat_rhythm_mask | cross_test | 1 | 0.725843 | 0.743867 | 0.686503 | 0.723127 | 0.977551 | 0.766956 |
| ptb_to_but | P3_clean_noisy_physio_distill | cross_test | 1 | 0.579213 | 0.629434 | 0.365691 | 0.821933 | 0.977551 | 0.526178 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2p3\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2p3\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2p3\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2p3\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v60raw_drop_mediumlike_bad_p2p3`