# Event-Factorized Cross-Dataset Aligned-PTB Report

- Generated: 2026-06-20 21:18:43
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- PTB protocol is selected to match clean BUT train-split waveform-computable feature distribution.
- Clean BUT test rows are not used for alignment.
- Formal model input remains waveform-derived channels only.

## Protocols

- Aligned PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v10_butmedium_extreme\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb_aligned | E1_query_only | cross_all | 1 | 0.614111 | 0.57518 | 0.166667 | 1 | 0.675667 | 0.674312 |
| but_to_ptb_aligned | E2_query_highres | cross_all | 1 | 0.504778 | 0.459078 | 0.171667 | 1 | 0.342667 | 0.418887 |
| but_to_ptb_aligned | E1_query_only | cross_test | 1 | 0.669158 | 0.650983 | 0.329646 | 1 | 0.642023 | 0.659685 |
| but_to_ptb_aligned | E2_query_highres | cross_test | 1 | 0.533288 | 0.494421 | 0.247788 | 1 | 0.324903 | 0.407188 |
| ptb_aligned_to_but | E2_query_highres | cross_all | 1 | 0.752908 | 0.687289 | 0.997239 | 0.15419 | 0.999755 | 0.656097 |
| ptb_aligned_to_but | E1_query_only | cross_all | 1 | 0.743546 | 0.662249 | 0.998931 | 0.118755 | 1 | 0.631061 |
| ptb_aligned_to_but | E2_query_highres | cross_test | 1 | 0.429197 | 0.539941 | 0.996016 | 0.122773 | 1 | 0.653597 |
| ptb_aligned_to_but | E1_query_only | cross_test | 1 | 0.422319 | 0.52691 | 0.998008 | 0.111218 | 1 | 0.64156 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v10_butmedium_extreme\aligned_cross_dataset_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v10_butmedium_extreme\aligned_cross_dataset_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v10_butmedium_extreme\aligned_cross_dataset_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v10_butmedium_extreme\aligned_cross_dataset_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_aligned_v10_butmedium_extreme`