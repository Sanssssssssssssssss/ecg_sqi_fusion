# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 15:53:15
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v66hybrid_v60gb_v63m50\protocol_v66hybrid_v60gb_v63m50_pc3000_s20260654`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only_subtype_aux | cross_all | 1 | 0.810889 | 0.809819 | 0.572 | 0.982667 | 0.878 | 0.879131 |
| but_to_ptb | E1_query_only_subtype_aux | cross_test | 1 | 0.817308 | 0.816414 | 0.586667 | 0.986667 | 0.878319 | 0.884825 |
| ptb_to_but | E1_query_only_subtype_aux | cross_all | 1 | 0.571364 | 0.620841 | 0.401409 | 0.740556 | 0.988708 | 0.519307 |
| ptb_to_but | E1_query_only_subtype_aux | cross_test | 1 | 0.57603 | 0.626209 | 0.40359 | 0.748643 | 0.985714 | 0.56933 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v66hybrid_v60gb_v63m50_drop_mediumlike_bad_e1sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v66hybrid_v60gb_v63m50_drop_mediumlike_bad_e1sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v66hybrid_v60gb_v63m50_drop_mediumlike_bad_e1sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v66hybrid_v60gb_v63m50_drop_mediumlike_bad_e1sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v66hybrid_v60gb_v63m50_drop_mediumlike_bad_e1sub`