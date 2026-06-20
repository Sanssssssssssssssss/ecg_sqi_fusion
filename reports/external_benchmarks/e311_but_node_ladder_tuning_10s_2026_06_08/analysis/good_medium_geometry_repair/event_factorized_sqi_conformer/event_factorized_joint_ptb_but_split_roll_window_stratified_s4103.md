# Joint PTB+BUT Interpretable Waveform Model

- Generated: 2026-06-20 22:49:32
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Training protocol: PTB v11 style replay + clean BUT train/val + controlled PTB bad morphology extras.
- Formal model input remains waveform-derived channels only.
- SQI/factor columns are teacher targets and diagnostics only.

## Protocol

- Joint protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4103\protocol_joint_v11_badextra1200_s20264103`
- PTB eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v11_buttrain_style_replay\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- BUT eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\but_protocol_window_stratified_s4103`

## Held-Out Results

| candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only | but_all | 1 | 0.866651 | 0.875642 | 0.831493 | 0.842777 | 1 | 0.824764 | 0.064628 |
| E1_query_only | but_test | 1 | 0.860939 | 0.870038 | 0.826706 | 0.831736 | 1 | 0.875911 | 0.0767635 |
| E1_query_only | joint_test | 1 | 0.901444 | 0.906859 | 0.861956 | 0.883045 | 1 | 0.992451 | 0.0532407 |
| E1_query_only | ptb_all | 1 | 0.990556 | 0.990555 | 0.996667 | 0.975333 | 0.999667 | 0.997853 | 0.0211116 |
| E1_query_only | ptb_test | 1 | 0.990489 | 0.990713 | 0.993363 | 0.978261 | 1 | 0.998602 | 0.0235602 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4103\joint_ptb_but_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4103\joint_ptb_but_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4103\joint_ptb_but_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4103\joint_ptb_but_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_joint_ptb_but_split_roll\window_stratified_s4103`