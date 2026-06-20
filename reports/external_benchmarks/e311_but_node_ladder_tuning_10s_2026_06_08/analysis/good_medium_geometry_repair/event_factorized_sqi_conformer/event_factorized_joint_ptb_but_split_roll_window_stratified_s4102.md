# Joint PTB+BUT Interpretable Waveform Model

- Generated: 2026-06-20 22:46:26
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Training protocol: PTB v11 style replay + clean BUT train/val + controlled PTB bad morphology extras.
- Formal model input remains waveform-derived channels only.
- SQI/factor columns are teacher targets and diagnostics only.

## Protocol

- Joint protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4102\protocol_joint_v11_badextra1200_s20264102`
- PTB eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v11_buttrain_style_replay\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- BUT eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\but_protocol_window_stratified_s4102`

## Held-Out Results

| candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only | but_all | 1 | 0.877173 | 0.879948 | 0.887068 | 0.779409 | 1 | 0.89236 | 0.0741672 |
| E1_query_only | but_test | 1 | 0.872991 | 0.874603 | 0.886647 | 0.765708 | 1 | 0.910466 | 0.0807453 |
| E1_query_only | joint_test | 1 | 0.910365 | 0.912732 | 0.910154 | 0.841522 | 0.999112 | 0.992176 | 0.0543353 |
| E1_query_only | ptb_all | 1 | 0.997444 | 0.997444 | 0.998 | 0.994667 | 0.999667 | 0.999192 | 0.00560103 |
| E1_query_only | ptb_test | 1 | 0.992527 | 0.992765 | 0.997788 | 0.982213 | 0.998054 | 0.996249 | 0.0209424 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4102\joint_ptb_but_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4102\joint_ptb_but_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4102\joint_ptb_but_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_split_roll\joint_window_stratified_s4102\joint_ptb_but_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_joint_ptb_but_split_roll\window_stratified_s4102`