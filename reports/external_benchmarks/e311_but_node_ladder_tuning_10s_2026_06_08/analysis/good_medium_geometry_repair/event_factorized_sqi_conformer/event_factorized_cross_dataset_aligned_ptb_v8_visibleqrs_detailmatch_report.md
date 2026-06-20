# Event-Factorized Cross-Dataset Aligned-PTB Report

- Generated: 2026-06-20 20:51:00
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- PTB protocol is selected to match clean BUT train-split waveform-computable feature distribution.
- Clean BUT test rows are not used for alignment.
- Formal model input remains waveform-derived channels only.

## Protocols

- Aligned PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v8_visibleqrs_detailmatch\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb_aligned | E1_query_only | cross_all | 1 | 0.524778 | 0.469126 | 0.0976667 | 1 | 0.476667 | 0.513794 |
| but_to_ptb_aligned | E2_query_highres | cross_all | 1 | 0.502889 | 0.461186 | 0.267333 | 1 | 0.241333 | 0.347978 |
| but_to_ptb_aligned | E2_query_highres | cross_test | 1 | 0.542799 | 0.510914 | 0.382743 | 1 | 0.233463 | 0.349933 |
| but_to_ptb_aligned | E1_query_only | cross_test | 1 | 0.538043 | 0.478954 | 0.123894 | 1 | 0.447471 | 0.498371 |
| ptb_aligned_to_but | E2_query_highres | cross_all | 1 | 0.756895 | 0.694723 | 0.997328 | 0.167598 | 1 | 0.695157 |
| ptb_aligned_to_but | E1_query_only | cross_all | 1 | 0.751611 | 0.683583 | 0.996883 | 0.1502 | 1 | 0.65473 |
| ptb_aligned_to_but | E1_query_only | cross_test | 1 | 0.455142 | 0.555059 | 0.998008 | 0.161772 | 1 | 0.657358 |
| ptb_aligned_to_but | E2_query_highres | cross_test | 1 | 0.447952 | 0.555348 | 0.998008 | 0.150698 | 1 | 0.662482 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v8_visibleqrs_detailmatch\aligned_cross_dataset_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v8_visibleqrs_detailmatch\aligned_cross_dataset_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v8_visibleqrs_detailmatch\aligned_cross_dataset_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v8_visibleqrs_detailmatch\aligned_cross_dataset_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_aligned_v8_visibleqrs_detailmatch`