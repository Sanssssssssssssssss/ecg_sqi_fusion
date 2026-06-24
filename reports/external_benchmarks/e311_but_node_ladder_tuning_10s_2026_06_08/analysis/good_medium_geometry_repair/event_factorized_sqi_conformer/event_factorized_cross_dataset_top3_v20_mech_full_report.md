# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-21 01:42:50
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\protocol_v20_pc3000_s20260621`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E4_query_highres_local_art | cross_all | 1 | 0.845222 | 0.84925 | 0.729667 | 0.997667 | 0.808333 | 0.916335 |
| but_to_ptb | P2_ecg_beat_rhythm_mask | cross_all | 1 | 0.823778 | 0.825949 | 0.835667 | 0.999 | 0.636667 | 0.948889 |
| but_to_ptb | E4_query_highres_local_art | cross_test | 1 | 0.851562 | 0.855532 | 0.75 | 0.998024 | 0.788889 | 0.920385 |
| but_to_ptb | P2_ecg_beat_rhythm_mask | cross_test | 1 | 0.819602 | 0.819173 | 0.84292 | 1 | 0.593333 | 0.91815 |
| ptb_to_but | P2_ecg_beat_rhythm_mask | cross_all | 1 | 0.784936 | 0.770204 | 0.897132 | 0.443895 | 0.999755 | 0.755253 |
| ptb_to_but | E4_query_highres_local_art | cross_all | 1 | 0.783778 | 0.74527 | 0.981831 | 0.288109 | 0.999755 | 0.757581 |
| ptb_to_but | P2_ecg_beat_rhythm_mask | cross_test | 1 | 0.532354 | 0.520221 | 0.994024 | 0.282619 | 1 | 0.762335 |
| ptb_to_but | E4_query_highres_local_art | cross_test | 1 | 0.435136 | 0.490601 | 0.998008 | 0.130958 | 1 | 0.657938 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_mech_full\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_mech_full\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_mech_full\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_mech_full\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v20_mech_full`