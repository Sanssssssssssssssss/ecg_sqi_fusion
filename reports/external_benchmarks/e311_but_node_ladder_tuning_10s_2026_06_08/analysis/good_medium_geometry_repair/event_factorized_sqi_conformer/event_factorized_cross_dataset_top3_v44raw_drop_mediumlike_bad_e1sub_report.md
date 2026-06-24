# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 05:20:51
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\v44raw`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\butdmb`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.732556 | 0.708576 | 0.903 | 0.973 | 0.321667 | 0.753885 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.717873 | 0.689894 | 0.900442 | 0.962222 | 0.292035 | 0.73488 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.653518 | 0.656136 | 0.558835 | 0.726987 | 0.966123 | 0.621738 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.658614 | 0.664589 | 0.558843 | 0.740499 | 0.963265 | 0.683839 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v44raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v44raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v44raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v44raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v44raw_drop_mediumlike_bad_e1sub`