# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 05:46:32
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\v48raw`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\butdmb`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.846556 | 0.850263 | 0.795667 | 0.969333 | 0.774667 | 0.899864 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.843427 | 0.846985 | 0.809735 | 0.96 | 0.761062 | 0.895731 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.639182 | 0.639467 | 0.563156 | 0.672492 | 0.989126 | 0.617172 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.644382 | 0.644484 | 0.563497 | 0.686211 | 0.983673 | 0.682011 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v48raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v48raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v48raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v48raw_drop_mediumlike_bad_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v48raw_drop_mediumlike_bad_e1sub`