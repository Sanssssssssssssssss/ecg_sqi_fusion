# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-22 20:26:28
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v34_pca_mixture_gm_featurematched\protocol_v34_pca_mixture_pc3000_s20260622`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.750444 | 0.742337 | 0.834667 | 0.994 | 0.422667 | 0.750444 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.75 | 0.736594 | 0.831858 | 0.994071 | 0.393333 | 0.75 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.792213 | 0.802407 | 0.777788 | 0.682841 | 0.999755 | 0.730786 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.659581 | 0.6374 | 0.981076 | 0.484834 | 1 | 0.744272 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v34_pca_mixture_e1\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v34_pca_mixture_e1\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v34_pca_mixture_e1\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v34_pca_mixture_e1\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v34_pca_mixture_e1`