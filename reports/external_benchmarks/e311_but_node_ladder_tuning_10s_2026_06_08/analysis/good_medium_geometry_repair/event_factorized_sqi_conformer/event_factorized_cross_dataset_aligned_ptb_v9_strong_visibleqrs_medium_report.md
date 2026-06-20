# Event-Factorized Cross-Dataset Aligned-PTB Report

- Generated: 2026-06-20 21:02:21
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- PTB protocol is selected to match clean BUT train-split waveform-computable feature distribution.
- Clean BUT test rows are not used for alignment.
- Formal model input remains waveform-derived channels only.

## Protocols

- Aligned PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v9_strong_visibleqrs_medium\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb_aligned | E1_query_only | cross_all | 1 | 0.591667 | 0.52805 | 0.0763333 | 0.986 | 0.712667 | 0.694048 |
| but_to_ptb_aligned | E2_query_highres | cross_all | 1 | 0.518 | 0.482701 | 0.268333 | 0.999667 | 0.286 | 0.385868 |
| but_to_ptb_aligned | E1_query_only | cross_test | 1 | 0.601223 | 0.522861 | 0.0685841 | 0.990119 | 0.68677 | 0.67857 |
| but_to_ptb_aligned | E2_query_highres | cross_test | 1 | 0.559103 | 0.533743 | 0.396018 | 0.998024 | 0.270428 | 0.388425 |
| ptb_aligned_to_but | E1_query_only | cross_all | 1 | 0.755319 | 0.689815 | 0.99813 | 0.160734 | 1 | 0.690564 |
| ptb_aligned_to_but | E2_query_highres | cross_all | 1 | 0.753187 | 0.675091 | 0.996794 | 0.155786 | 1 | 0.658018 |
| ptb_aligned_to_but | E2_query_highres | cross_test | 1 | 0.432635 | 0.507098 | 0.999004 | 0.126625 | 1 | 0.655999 |
| ptb_aligned_to_but | E1_query_only | cross_test | 1 | 0.429509 | 0.558613 | 0.999004 | 0.12181 | 1 | 0.644504 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v9_strong_visibleqrs_medium\aligned_cross_dataset_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v9_strong_visibleqrs_medium\aligned_cross_dataset_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v9_strong_visibleqrs_medium\aligned_cross_dataset_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v9_strong_visibleqrs_medium\aligned_cross_dataset_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_aligned_v9_strong_visibleqrs_medium`