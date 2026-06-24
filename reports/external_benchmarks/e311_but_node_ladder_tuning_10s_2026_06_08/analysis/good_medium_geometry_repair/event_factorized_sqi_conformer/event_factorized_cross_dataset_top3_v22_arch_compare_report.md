# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-21 02:26:50
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v22_gm_featurematched\protocol_v22_pc3000_s20260621`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E2_query_highres | cross_all | 1 | 0.863 | 0.865561 | 0.867667 | 1 | 0.721333 | 0.933566 |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.853667 | 0.856467 | 0.855 | 1 | 0.706 | 0.92718 |
| but_to_ptb | E2_query_highres | cross_test | 1 | 0.84446 | 0.84705 | 0.834071 | 1 | 0.68 | 0.919847 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.838068 | 0.841147 | 0.811947 | 1 | 0.682222 | 0.909238 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.753326 | 0.753787 | 0.792928 | 0.521788 | 0.999755 | 0.644858 |
| ptb_to_but | E2_query_highres | cross_all | 1 | 0.722503 | 0.73682 | 0.69487 | 0.591381 | 0.999755 | 0.580276 |
| ptb_to_but | E2_query_highres | cross_test | 1 | 0.612691 | 0.563513 | 0.97012 | 0.41791 | 1 | 0.819053 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.587684 | 0.553988 | 0.996016 | 0.366875 | 1 | 0.816984 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v22_arch_compare\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v22_arch_compare\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v22_arch_compare\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v22_arch_compare\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v22_arch_compare`