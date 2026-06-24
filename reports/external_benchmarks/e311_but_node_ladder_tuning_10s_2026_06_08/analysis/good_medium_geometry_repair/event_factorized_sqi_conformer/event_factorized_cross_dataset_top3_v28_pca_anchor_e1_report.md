# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-22 19:02:22
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v28_pca_anchor_gm_featurematched\protocol_v28_pca_anchor_pc3000_s20260622`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.870889 | 0.874345 | 0.822 | 0.996667 | 0.794 | 0.909147 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.858665 | 0.862559 | 0.783186 | 1 | 0.775556 | 0.896406 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.843152 | 0.8596 | 0.849216 | 0.730247 | 0.999755 | 0.862745 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.690216 | 0.785359 | 0.993028 | 0.52624 | 1 | 0.816513 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v28_pca_anchor_e1\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v28_pca_anchor_e1\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v28_pca_anchor_e1\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v28_pca_anchor_e1\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v28_pca_anchor_e1`