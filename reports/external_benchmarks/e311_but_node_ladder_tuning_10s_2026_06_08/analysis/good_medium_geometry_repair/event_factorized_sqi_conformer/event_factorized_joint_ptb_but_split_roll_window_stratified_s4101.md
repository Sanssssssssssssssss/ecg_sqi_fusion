# Joint PTB+BUT Interpretable Waveform Model

- Generated: 2026-06-20 22:43:17
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Training protocol: PTB v11 style replay + clean BUT train/val + controlled PTB bad morphology extras.
- Formal model input remains waveform-derived channels only.
- SQI/factor columns are teacher targets and diagnostics only.

## Protocol

- Joint protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4101\protocol_joint_v11_badextra1200_s20264101`
- PTB eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v11_buttrain_style_replay\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- BUT eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\but_protocol_window_stratified_s4101`

## Held-Out Results

| candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only | but_all | 1 | 0.876941 | 0.883188 | 0.885109 | 0.782123 | 1 | 0.900662 | 0.0477697 |
| E1_query_only | but_test | 1 | 0.872064 | 0.877466 | 0.880712 | 0.773163 | 1 | 0.931936 | 0.0559284 |
| E1_query_only | joint_test | 1 | 0.908454 | 0.912202 | 0.904539 | 0.843599 | 0.999112 | 0.991436 | 0.0446321 |
| E1_query_only | ptb_all | 1 | 0.991889 | 0.991887 | 0.997333 | 0.979 | 0.999333 | 0.997511 | 0.0219733 |
| E1_query_only | ptb_test | 1 | 0.988451 | 0.988753 | 0.993363 | 0.974308 | 0.998054 | 0.994577 | 0.0314136 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4101\joint_ptb_but_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4101\joint_ptb_but_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4101\joint_ptb_but_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4101\joint_ptb_but_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_joint_ptb_but_split_roll\window_stratified_s4101`