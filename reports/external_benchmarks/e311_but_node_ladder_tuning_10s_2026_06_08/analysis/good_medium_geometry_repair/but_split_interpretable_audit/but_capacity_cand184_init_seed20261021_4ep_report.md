# BUT-Only Capacity Diagnostic: cand184_init_seed20261021_4ep

Diagnostic only. This uses BUT train rows and must not be mixed into PTB->BUT claims.
Split scheme: `candidate_seed_184`.

| acc | macro_f1 | good_recall | medium_recall | bad_recall | confusion_3x3 | bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 0.946754 | 0.951149 | 0.950865 | 0.918812 | 0.9807 | [[9734, 503, 0], [569, 6994, 49], [23, 74, 4929]] | but_train |
| 0.933198 | 0.91357 | 0.953789 | 0.972696 | 0.785714 | [[516, 25, 0], [6, 285, 2], [0, 33, 121]] | but_val |
| 0.82668 | 0.826274 | 0.819154 | 0.846493 | 0.761905 | [[5132, 1133, 0], [418, 2305, 0], [0, 25, 80]] | but_test |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_cand184_init_seed20261021_4ep_metrics.csv`
Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_cand184_init_seed20261021_4ep_predictions.csv`
Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_cand184_init_seed20261021_4ep_best.pt`
