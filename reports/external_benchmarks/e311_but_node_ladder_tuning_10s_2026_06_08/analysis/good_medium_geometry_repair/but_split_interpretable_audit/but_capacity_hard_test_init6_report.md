# BUT-Only Capacity Diagnostic: hard_test_init6

Diagnostic only. This uses BUT train rows and must not be mixed into PTB->BUT claims.

| acc | macro_f1 | good_recall | medium_recall | bad_recall | confusion_3x3 | bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 0.956322 | 0.9521 | 0.929608 | 0.952844 | 0.998583 | [[6854, 519, 0], [160, 3233, 0], [2, 5, 4933]] | but_train |
| 0.794114 | 0.522691 | 0.759129 | 0.884449 | 0 | [[4532, 1438, 0], [321, 2457, 0], [0, 53, 0]] | but_val |
| 0.846727 | 0.601288 | 0.909459 | 0.847207 | 0.0445205 | [[3365, 334, 1], [670, 3776, 11], [165, 114, 13]] | but_test |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_hard_test_init6_metrics.csv`
Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_hard_test_init6_predictions.csv`
Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_hard_test_init6_best.pt`
