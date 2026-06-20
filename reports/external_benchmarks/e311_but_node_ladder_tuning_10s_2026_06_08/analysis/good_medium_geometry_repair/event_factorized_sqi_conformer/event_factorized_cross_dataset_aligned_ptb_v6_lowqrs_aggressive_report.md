# Event-Factorized Cross-Dataset Aligned-PTB Report

- Generated: 2026-06-20 20:27:59
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- PTB protocol is selected to match clean BUT train-split waveform-computable feature distribution.
- Clean BUT test rows are not used for alignment.
- Formal model input remains waveform-derived channels only.

## Protocols

- Aligned PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v6_lowqrs_aggressive\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb_aligned | E2_query_highres | cross_all | 1 | 0.364222 | 0.229523 | 0.0276667 | 0.999667 | 0.0653333 | 0.171848 |
| but_to_ptb_aligned | E1_query_only | cross_all | 1 | 0.358222 | 0.216287 | 0.00133333 | 1 | 0.0733333 | 0.173527 |
| but_to_ptb_aligned | E2_query_highres | cross_test | 1 | 0.375 | 0.233368 | 0.0265487 | 1 | 0.0661479 | 0.173651 |
| but_to_ptb_aligned | E1_query_only | cross_test | 1 | 0.367527 | 0.216116 | 0 | 1 | 0.0680934 | 0.171429 |
| ptb_aligned_to_but | E2_query_highres | cross_all | 1 | 0.709618 | 0.587931 | 1 | 0 | 1 | 0.532377 |
| ptb_aligned_to_but | E1_query_only | cross_all | 1 | 0.7092 | 0.582385 | 0.999198 | 0 | 1 | 0.530828 |
| ptb_aligned_to_but | E2_query_highres | cross_test | 1 | 0.350735 | 0.41607 | 1 | 0 | 1 | 0.620037 |
| ptb_aligned_to_but | E1_query_only | cross_test | 1 | 0.349797 | 0.361237 | 0.997012 | 0 | 1 | 0.62196 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v6_lowqrs_aggressive\aligned_cross_dataset_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v6_lowqrs_aggressive\aligned_cross_dataset_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v6_lowqrs_aggressive\aligned_cross_dataset_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v6_lowqrs_aggressive\aligned_cross_dataset_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_aligned_v6_lowqrs_aggressive`