# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 15:47:49
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v65hybrid_v60gb_v63m25\protocol_v65hybrid_v60gb_v63m25_pc3000_s20260654`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.650222 | 0.623428 | 0.739 | 0.974 | 0.237667 | 0.688951 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.649408 | 0.623859 | 0.742222 | 0.962222 | 0.245575 | 0.690914 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.51961 | 0.566679 | 0.274166 | 0.798415 | 0.989544 | 0.403749 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.521536 | 0.569051 | 0.277926 | 0.795874 | 0.985714 | 0.457326 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v65hybrid_v60gb_v63m25_drop_mediumlike_bad_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v65hybrid_v60gb_v63m25_drop_mediumlike_bad_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v65hybrid_v60gb_v63m25_drop_mediumlike_bad_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v65hybrid_v60gb_v63m25_drop_mediumlike_bad_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v65hybrid_v60gb_v63m25_drop_mediumlike_bad_e1sub`