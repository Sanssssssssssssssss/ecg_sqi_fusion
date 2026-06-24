# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-21 02:40:52
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v23_gm_featurematched\protocol_v23_pc3000_s20260621`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E2_query_highres | cross_all | 1 | 0.878889 | 0.88204 | 0.831 | 0.997 | 0.808667 | 0.913829 |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.860222 | 0.863706 | 0.833 | 0.997667 | 0.75 | 0.915082 |
| but_to_ptb | E2_query_highres | cross_test | 1 | 0.863636 | 0.867436 | 0.780973 | 1 | 0.793333 | 0.89542 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.84233 | 0.846308 | 0.780973 | 0.998024 | 0.728889 | 0.894059 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.826883 | 0.836586 | 0.865515 | 0.645012 | 0.999755 | 0.802227 |
| ptb_to_but | E2_query_highres | cross_all | 1 | 0.769455 | 0.791458 | 0.741628 | 0.669274 | 0.999755 | 0.75793 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.629259 | 0.682077 | 0.993028 | 0.432354 | 1 | 0.760703 |
| ptb_to_but | E2_query_highres | cross_test | 1 | 0.604251 | 0.617013 | 0.982072 | 0.399133 | 1 | 0.775784 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v23_arch_compare\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v23_arch_compare\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v23_arch_compare\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v23_arch_compare\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v23_arch_compare`