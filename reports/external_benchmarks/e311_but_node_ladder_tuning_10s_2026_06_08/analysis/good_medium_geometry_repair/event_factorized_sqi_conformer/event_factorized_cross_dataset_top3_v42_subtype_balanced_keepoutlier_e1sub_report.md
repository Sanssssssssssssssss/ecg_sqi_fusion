# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 04:28:19
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\v42p`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\butko`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.581556 | 0.485893 | 0.008 | 0.871 | 0.865667 | 0.695134 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.578287 | 0.48269 | 0.00663717 | 0.877778 | 0.85177 | 0.692127 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.581877 | 0.589088 | 0.437774 | 0.584781 | 0.997091 | 0.484603 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.580514 | 0.588552 | 0.430519 | 0.593377 | 0.99515 | 0.533969 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v42_subtype_balanced_keepoutlier_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v42_subtype_balanced_keepoutlier_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v42_subtype_balanced_keepoutlier_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v42_subtype_balanced_keepoutlier_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v42_subtype_balanced_keepoutlier_e1sub`