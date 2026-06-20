# Joint PTB+BUT Interpretable Waveform Model

- Generated: 2026-06-20 22:15:19
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Training protocol: PTB v11 style replay + clean BUT train/val + controlled PTB bad morphology extras.
- Formal model input remains waveform-derived channels only.
- SQI/factor columns are teacher targets and diagnostics only.

## Protocol

- Joint protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\protocol_joint_v11_badextra0_s20260621`
- PTB eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v11_buttrain_style_replay\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- BUT eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Held-Out Results

| candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only | but_all | 1 | 0.827161 | 0.813783 | 0.982633 | 0.435914 | 1 | 0.848594 | 0.0184573 |
| E1_query_only | but_test | 1 | 0.529853 | 0.620139 | 0.998008 | 0.276842 | 1 | 0.755661 | 0.0293283 |
| E1_query_only | joint_test | 1 | 0.674588 | 0.735836 | 0.998626 | 0.41386 | 0.993671 | 0.98562 | 0.0271022 |
| E1_query_only | ptb_all | 1 | 0.994444 | 0.994438 | 1 | 0.986667 | 0.996667 | 0.996044 | 0.00516351 |
| E1_query_only | ptb_test | 1 | 0.98913 | 0.98939 | 1 | 0.976285 | 0.992218 | 0.987643 | 0.0209424 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_joint_ptb_but_interpretable`