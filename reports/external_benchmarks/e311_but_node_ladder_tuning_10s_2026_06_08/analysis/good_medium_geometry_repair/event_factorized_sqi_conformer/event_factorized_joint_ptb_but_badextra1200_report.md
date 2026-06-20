# Joint PTB+BUT Interpretable Waveform Model

- Generated: 2026-06-20 22:23:23
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Training protocol: PTB v11 style replay + clean BUT train/val + controlled PTB bad morphology extras.
- Formal model input remains waveform-derived channels only.
- SQI/factor columns are teacher targets and diagnostics only.

## Protocol

- Joint protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\protocol_joint_v11_badextra1200_s20260620`
- PTB eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v11_buttrain_style_replay\protocol_ptb_buttrain_aligned_pc3000_s20260620`
- BUT eval protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_drop_outlier`

## Held-Out Results

| candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only | but_all | 1 | 0.893441 | 0.897704 | 0.949857 | 0.723065 | 0.999755 | 0.881844 | 0.0172865 |
| E1_query_only | but_test | 1 | 0.770866 | 0.781395 | 0.998008 | 0.64805 | 1 | 0.836452 | 0.0364444 |
| E1_query_only | joint_test | 1 | 0.840505 | 0.864077 | 0.998626 | 0.713124 | 0.996835 | 0.992187 | 0.0331126 |
| E1_query_only | ptb_all | 1 | 0.997444 | 0.997444 | 1 | 0.993 | 0.999333 | 0.998831 | 0.00643777 |
| E1_query_only | ptb_test | 1 | 0.991848 | 0.992155 | 1 | 0.980237 | 0.996109 | 0.993557 | 0.0233766 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_joint_ptb_but_interpretable\joint_ptb_but_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_joint_ptb_but_interpretable`