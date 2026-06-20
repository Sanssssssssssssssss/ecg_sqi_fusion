# Event-Factorized Cross-Dataset Aligned-PTB Report

- Generated: 2026-06-20 18:59:23
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- PTB protocol is selected to match clean BUT train-split waveform-computable feature distribution.
- Clean BUT test rows are not used for alignment.
- Formal model input remains waveform-derived channels only.

## Protocols

- Aligned PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v3_detectorfail\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb_aligned | E1_query_only | cross_all | 1 | 0.480556 | 0.413594 | 0.079 | 1 | 0.362667 | 0.433761 |
| but_to_ptb_aligned | E2_query_highres | cross_all | 1 | 0.456333 | 0.378092 | 0.057 | 1 | 0.312 | 0.391339 |
| but_to_ptb_aligned | E1_query_only | cross_test | 1 | 0.506793 | 0.432688 | 0.079646 | 1 | 0.396887 | 0.475144 |
| but_to_ptb_aligned | E2_query_highres | cross_test | 1 | 0.484375 | 0.400359 | 0.0575221 | 1 | 0.35214 | 0.436851 |
| ptb_aligned_to_but | E1_query_only | cross_all | 1 | 0.709571 | 0.593399 | 1 | 0 | 0.999755 | 0.522746 |
| ptb_aligned_to_but | E2_query_highres | cross_all | 1 | 0.709571 | 0.59362 | 1 | 0 | 0.999755 | 0.522743 |
| ptb_aligned_to_but | E1_query_only | cross_test | 1 | 0.350735 | 0.495818 | 1 | 0 | 1 | 0.618404 |
| ptb_aligned_to_but | E2_query_highres | cross_test | 1 | 0.350735 | 0.497185 | 1 | 0 | 1 | 0.618384 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v3_detectorfail\aligned_cross_dataset_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v3_detectorfail\aligned_cross_dataset_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v3_detectorfail\aligned_cross_dataset_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v3_detectorfail\aligned_cross_dataset_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_aligned_v3_detectorfail`