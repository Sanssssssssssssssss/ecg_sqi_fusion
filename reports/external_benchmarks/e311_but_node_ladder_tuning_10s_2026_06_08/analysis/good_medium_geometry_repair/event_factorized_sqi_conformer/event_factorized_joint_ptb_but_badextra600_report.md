# Joint PTB+BUT Interpretable Waveform Model

- Generated: 2026-06-20 22:19:17
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Training protocol: PTB v11 style replay + clean BUT train/val + controlled PTB bad morphology extras.
- Formal model input remains waveform-derived channels only.
- SQI/factor columns are teacher targets and diagnostics only.

## Protocol

- Joint protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\protocol_joint_v11_badextra600_s20260622`
- PTB eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v11_buttrain_style_replay\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- BUT eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Held-Out Results

| candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only | but_all | 1 | 0.853163 | 0.868383 | 0.827396 | 0.803831 | 0.999755 | 0.872333 | 0.0225201 |
| E1_query_only | but_test | 1 | 0.778681 | 0.766064 | 0.987052 | 0.665383 | 1 | 0.707657 | 0.0538321 |
| E1_query_only | joint_test | 1 | 0.845215 | 0.863019 | 0.990385 | 0.726287 | 0.996835 | 0.990437 | 0.0473613 |
| E1_query_only | ptb_all | 1 | 0.997111 | 0.997111 | 0.999333 | 0.993333 | 0.998667 | 0.998181 | 0.00730868 |
| E1_query_only | ptb_test | 1 | 0.98981 | 0.99015 | 0.997788 | 0.976285 | 0.996109 | 0.992925 | 0.0287958 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_joint_ptb_but_interpretable`