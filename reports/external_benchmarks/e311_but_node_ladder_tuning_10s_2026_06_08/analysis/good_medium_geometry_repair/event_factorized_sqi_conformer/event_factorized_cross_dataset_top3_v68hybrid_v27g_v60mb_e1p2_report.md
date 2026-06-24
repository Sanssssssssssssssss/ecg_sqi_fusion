# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 17:32:51
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v68hybrid_v27g_v60mb\protocol_v68hybrid_v27g_v60mb_pc3000_s20260654`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.668889 | 0.620308 | 0.857667 | 0.987667 | 0.161333 | 0.856527 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.666174 | 0.621168 | 0.847345 | 0.98 | 0.172566 | 0.840363 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.72006 | 0.683672 | 0.879138 | 0.404581 | 0.934755 | 0.632632 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.719288 | 0.681647 | 0.878324 | 0.400651 | 0.940816 | 0.680835 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v68hybrid_v27g_v60mb_e1p2\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v68hybrid_v27g_v60mb_e1p2\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v68hybrid_v27g_v60mb_e1p2\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v68hybrid_v27g_v60mb_e1p2\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v68hybrid_v27g_v60mb_e1p2`