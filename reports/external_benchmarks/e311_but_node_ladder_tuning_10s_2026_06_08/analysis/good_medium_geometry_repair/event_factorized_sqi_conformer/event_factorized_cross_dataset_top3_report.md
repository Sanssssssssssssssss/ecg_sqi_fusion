# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-20 18:11:26
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Top-3 definition: pooled clean BUT internal ranking from Event-Factorized full report: `E1`, `E0`, `E2`.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3\protocol_ptb_bal2500_blocks_v3_s20260620`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | E0_noquery_nohi_nolocal_noart | cross_all | 1 | 0.6576 | 0.53885 | 1 | 0.9728 | 0 | 0.559011 |
| but_to_ptb | E2_query_highres | cross_all | 1 | 0.6496 | 0.531907 | 1 | 0.9484 | 0.0004 | 0.549408 |
| but_to_ptb | E1_query_only | cross_all | 1 | 0.581733 | 0.469021 | 0.9968 | 0.7484 | 0 | 0.465609 |
| but_to_ptb | E0_noquery_nohi_nolocal_noart | cross_test | 1 | 0.651695 | 0.534437 | 1 | 0.979328 | 0 | 0.559606 |
| but_to_ptb | E2_query_highres | cross_test | 1 | 0.636441 | 0.521561 | 1 | 0.932817 | 0 | 0.542584 |
| but_to_ptb | E1_query_only | cross_test | 1 | 0.574576 | 0.465495 | 1 | 0.744186 | 0 | 0.455399 |
| ptb_to_but | E2_query_highres | cross_all | 1 | 0.790915 | 0.750276 | 0.891877 | 0.473743 | 1 | 0.745678 |
| ptb_to_but | E1_query_only | cross_all | 1 | 0.787856 | 0.722537 | 0.954043 | 0.351796 | 1 | 0.693932 |
| ptb_to_but | E0_noquery_nohi_nolocal_noart | cross_all | 1 | 0.655249 | 0.559103 | 0.811097 | 0.151317 | 1 | 0.536318 |
| ptb_to_but | E2_query_highres | cross_test | 1 | 0.656143 | 0.576713 | 0.948207 | 0.495426 | 1 | 0.744938 |
| ptb_to_but | E1_query_only | cross_test | 1 | 0.592685 | 0.531959 | 0.977092 | 0.383727 | 1 | 0.641868 |
| ptb_to_but | E0_noquery_nohi_nolocal_noart | cross_test | 1 | 0.389809 | 0.395645 | 0.88247 | 0.116996 | 1 | 0.635037 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3`