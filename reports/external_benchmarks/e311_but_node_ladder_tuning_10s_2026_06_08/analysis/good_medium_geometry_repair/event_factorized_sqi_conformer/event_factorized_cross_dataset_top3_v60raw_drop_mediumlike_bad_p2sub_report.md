# Event-Factorized Cross-Dataset Top-3 Report

- Generated: 2026-06-23 16:59:40
- Scope: external-only experiment; no `src/sqi_pipeline` changes.
- Formal input: waveform-derived channels only.
- Factor/SQI targets: training teacher/diagnostic only, not inference input.
- Candidate definition: supplied phase/candidate list; checkpoints are trained from scratch.

## Protocols

- PTB synthetic protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\protocol_v60raw_pc3000_s20260650`
- Clean BUT protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`

## Cross-Dataset Summary

| direction | candidate | bucket | runs | acc | macro_f1 | good_recall | medium_recall | bad_recall | record_macro_supported_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_to_ptb | P2_ecg_beat_rhythm_mask_lowfactor_subtype | cross_all | 1 | 0.735667 | 0.737043 | 0.735667 | 0.967 | 0.504333 | 0.80169 |
| but_to_ptb | P2_ecg_beat_rhythm_mask_subtype_aux | cross_all | 1 | 0.727444 | 0.73086 | 0.673667 | 0.978333 | 0.530333 | 0.799632 |
| but_to_ptb | P2_ecg_beat_rhythm_mask_lowfactor_subtype | cross_test | 1 | 0.742604 | 0.743807 | 0.755556 | 0.962222 | 0.511062 | 0.803797 |
| but_to_ptb | P2_ecg_beat_rhythm_mask_subtype_aux | cross_test | 1 | 0.735947 | 0.739505 | 0.693333 | 0.975556 | 0.539823 | 0.807396 |
| ptb_to_but | P2_ecg_beat_rhythm_mask_subtype_aux | cross_all | 1 | 0.653181 | 0.694398 | 0.549594 | 0.736648 | 0.983271 | 0.619882 |
| ptb_to_but | P2_ecg_beat_rhythm_mask_lowfactor_subtype | cross_all | 1 | 0.558454 | 0.617785 | 0.351615 | 0.785931 | 0.983271 | 0.467955 |
| ptb_to_but | P2_ecg_beat_rhythm_mask_subtype_aux | cross_test | 1 | 0.656929 | 0.700094 | 0.549867 | 0.744843 | 0.983673 | 0.689879 |
| ptb_to_but | P2_ecg_beat_rhythm_mask_lowfactor_subtype | cross_test | 1 | 0.559176 | 0.619792 | 0.345745 | 0.794788 | 0.983673 | 0.505952 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2sub\cross_dataset_top3_metrics.csv`
- Summary: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2sub\cross_dataset_top3_summary.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2sub\cross_dataset_top3_record_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_top3_v60raw_drop_mediumlike_bad_p2sub\cross_dataset_top3_feature_recovery.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_xds_top3_v60raw_drop_mediumlike_bad_p2sub`