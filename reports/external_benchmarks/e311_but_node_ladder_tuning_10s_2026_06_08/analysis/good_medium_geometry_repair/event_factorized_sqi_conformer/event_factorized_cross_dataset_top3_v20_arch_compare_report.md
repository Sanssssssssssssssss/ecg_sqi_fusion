# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-21 01:53:08
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v20_bad_subtype_featurematched\protocol_v20_pc3000_s20260621`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E2_query_highres | cross_all | 1 | 0.886556 | 0.889335 | 0.854667 | 0.993667 | 0.811333 | 0.961882 |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.877222 | 0.880511 | 0.827333 | 0.999333 | 0.805 | 0.9575 |
| but_to_ptb | E2_query_highres | cross_test | 1 | 0.884943 | 0.888005 | 0.853982 | 0.994071 | 0.793333 | 0.940703 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.87358 | 0.877087 | 0.818584 | 1 | 0.786667 | 0.933988 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.818355 | 0.829233 | 0.835055 | 0.670231 | 0.999755 | 0.809019 |
| ptb_to_but | E2_query_highres | cross_all | 1 | 0.799676 | 0.800605 | 0.856965 | 0.56664 | 0.999755 | 0.797386 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.690841 | 0.698043 | 0.987052 | 0.530091 | 1 | 0.769276 |
| ptb_to_but | E2_query_highres | cross_test | 1 | 0.577681 | 0.573636 | 0.996016 | 0.351468 | 1 | 0.792732 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_arch_compare\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_arch_compare\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_arch_compare\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v20_arch_compare\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v20_arch_compare`