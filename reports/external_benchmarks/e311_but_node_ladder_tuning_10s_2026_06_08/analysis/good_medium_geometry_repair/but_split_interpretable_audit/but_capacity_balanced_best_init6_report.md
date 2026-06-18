# BUT-Only Capacity Diagnostic: balanced_best_init6

Diagnostic only. This uses BUT train rows and must not be mixed into PTB->BUT claims.

| acc | macro_f1 | good_recall | medium_recall | bad_recall | confusion_3x3 | bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 0.934845 | 0.939987 | 0.973906 | 0.852145 | 0.984692 | [[9368, 251, 0], [1053, 6455, 67], [13, 64, 4953]] | but_train |
| 0.938026 | 0.910253 | 0.975177 | 0.877604 | 0.813333 | [[1100, 28, 0], [46, 337, 1], [3, 25, 122]] | but_val |
| 0.840353 | 0.824099 | 0.896283 | 0.711502 | 0.761905 | [[5643, 653, 0], [769, 1899, 1], [0, 25, 80]] | but_test |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_balanced_best_init6_metrics.csv`
Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_balanced_best_init6_predictions.csv`
Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_balanced_best_init6_best.pt`
