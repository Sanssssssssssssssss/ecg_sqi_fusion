# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 04:55:14
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\v43p`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\butdmb`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.560333 | 0.487444 | 0.0386667 | 0.970333 | 0.672 | 0.657722 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.558346 | 0.480032 | 0.0243363 | 0.977778 | 0.674779 | 0.657556 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.595459 | 0.571381 | 0.546669 | 0.572297 | 0.991635 | 0.579334 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.595318 | 0.573022 | 0.543218 | 0.575461 | 0.989796 | 0.636007 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v43_drop_mediumlike_bad_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v43_drop_mediumlike_bad_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v43_drop_mediumlike_bad_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v43_drop_mediumlike_bad_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v43_drop_mediumlike_bad_e1sub`