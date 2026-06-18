# BUT-Only Capacity Diagnostic: window_random_init4

Diagnostic only. This uses BUT train rows and must not be mixed into PTB->BUT claims.

| acc | macro_f1 | good_recall | medium_recall | bad_recall | confusion_3x3 | bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 0.921105 | 0.926766 | 0.939957 | 0.85712 | 0.988962 | [[9612, 613, 1], [767, 5465, 144], [2, 33, 3136]] | but_train |
| 0.923836 | 0.928323 | 0.941625 | 0.861176 | 0.992431 | [[3210, 196, 3], [238, 1830, 57], [1, 7, 1049]] | but_val |
| 0.917476 | 0.922753 | 0.941021 | 0.845322 | 0.986755 | [[3207, 201, 0], [276, 1798, 53], [0, 14, 1043]] | but_test |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_window_random_init4_metrics.csv`
Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_window_random_init4_predictions.csv`
Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_window_random_init4_best.pt`
