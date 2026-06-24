# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-24 02:27:31
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v80lead1_boundary_clean\protocol_v80lead1_boundary_clean_pc700_s20260686`
- Clean BUT protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E4_query_highres_local_art_subtype_aux_lr1e4 | cross_all | 1 | 0.737778 | 0.743877 | 0.649333 | 0.754667 | 0.809333 | 0.840253 |
| but_to_ptb | E4_query_highres_local_art_subtype_aux_lr1e4 | cross_test | 1 | 0.793651 | 0.761286 | 0.77037 | 0.743094 | 0.841518 | 0.868477 |
| ptb_to_but | E4_query_highres_local_art_subtype_aux_lr1e4 | cross_all | 1 | 0.739688 | 0.751442 | 0.717724 | 0.720039 | 0.953576 | 0.58724 |
| ptb_to_but | E4_query_highres_local_art_subtype_aux_lr1e4 | cross_test | 1 | 0.74382 | 0.756084 | 0.72008 | 0.726384 | 0.955102 | 0.636689 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v80lr_cross\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v80lr_cross\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v80lr_cross\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v80lr_cross\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v80lr_cross`