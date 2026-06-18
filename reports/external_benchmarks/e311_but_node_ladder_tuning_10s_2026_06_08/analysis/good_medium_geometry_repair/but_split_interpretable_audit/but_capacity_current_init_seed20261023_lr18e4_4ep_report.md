# BUT-Only Capacity Diagnostic: current_init_seed20261023_lr18e4_4ep

Diagnostic only. This uses BUT train rows and must not be mixed into PTB->BUT claims.
Split scheme: `current`.

| acc | macro_f1 | good_recall | medium_recall | bad_recall | confusion_3x3 | bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 0.914201 | 0.918392 | 0.910568 | 0.863867 | 0.987685 | [[11322, 1112, 0], [829, 5267, 1], [2, 57, 4732]] | but_train |
| 0.921348 | 0.863846 | 0.927761 | 0.828571 | 0.963855 | [[899, 70, 0], [18, 87, 0], [0, 3, 80]] | but_val |
| 0.803232 | 0.69814 | 0.931319 | 0.742431 | 0.323601 | [[3390, 248, 2], [1113, 3286, 27], [180, 98, 133]] | but_test |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_current_init_seed20261023_lr18e4_4ep_metrics.csv`
Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_current_init_seed20261023_lr18e4_4ep_predictions.csv`
Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_current_init_seed20261023_lr18e4_4ep_best.pt`
