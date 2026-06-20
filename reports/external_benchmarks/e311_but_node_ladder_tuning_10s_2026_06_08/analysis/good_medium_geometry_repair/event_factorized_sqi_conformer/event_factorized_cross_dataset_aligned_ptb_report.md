# Event-Factorized Cross-Dataset Aligned-PTB Report

- Generated: 2026-06-20 18:42:20
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- PTB protocol is selected to match clean BUT train-split waveform-computable feature distribution.
- Clean BUT test rows are not used for alignment.
- Formal model input remains waveform-derived channels only.

## Protocols

- Aligned PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v1\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb_aligned | E1_query_only | cross_all | 1 | 0.685111 | 0.594626 | 1 | 1 | 0.0553333 | 0.238924 |
| but_to_ptb_aligned | E2_query_highres | cross_all | 1 | 0.666778 | 0.555802 | 1 | 1 | 0.000333333 | 0.195972 |
| but_to_ptb_aligned | E0_noquery_nohi_nolocal_noart | cross_all | 1 | 0.666667 | 0.555556 | 1 | 1 | 0 | 0.195531 |
| but_to_ptb_aligned | E1_query_only | cross_test | 1 | 0.671196 | 0.595371 | 1 | 1 | 0.0583658 | 0.237079 |
| but_to_ptb_aligned | E0_noquery_nohi_nolocal_noart | cross_test | 1 | 0.650815 | 0.554391 | 1 | 1 | 0 | 0.193509 |
| but_to_ptb_aligned | E2_query_highres | cross_test | 1 | 0.650815 | 0.554391 | 1 | 1 | 0 | 0.193509 |
| ptb_aligned_to_but | E2_query_highres | cross_all | 1 | 0.723105 | 0.613826 | 0.993855 | 0.0574621 | 1 | 0.588871 |
| ptb_aligned_to_but | E1_query_only | cross_all | 1 | 0.71657 | 0.590942 | 0.995547 | 0.0319234 | 1 | 0.560118 |
| ptb_aligned_to_but | E0_noquery_nohi_nolocal_noart | cross_all | 1 | 0.713279 | 0.579206 | 0.990648 | 0.0293695 | 1 | 0.54561 |
| ptb_aligned_to_but | E2_query_highres | cross_test | 1 | 0.392623 | 0.391524 | 0.997012 | 0.0659605 | 1 | 0.621553 |
| ptb_aligned_to_but | E1_query_only | cross_test | 1 | 0.379494 | 0.356424 | 0.998008 | 0.0452576 | 1 | 0.616219 |
| ptb_aligned_to_but | E0_noquery_nohi_nolocal_noart | cross_test | 1 | 0.371991 | 0.318274 | 0.991036 | 0.0370727 | 1 | 0.618372 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v1\aligned_cross_dataset_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v1\aligned_cross_dataset_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v1\aligned_cross_dataset_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v1\aligned_cross_dataset_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_aligned_v1`