# BUT-Only Capacity Diagnostic: cand6529_init_seed20261022_4ep

Diagnostic only. This uses BUT train rows and must not be mixed into PTB->BUT claims.
Split scheme: `candidate_seed_6529`.

| acc | macro_f1 | good_recall | medium_recall | bad_recall | confusion_3x3 | bucket |
| --- | --- | --- | --- | --- | --- | --- |
| 0.94561 | 0.949598 | 0.950853 | 0.922183 | 0.970553 | [[9422, 487, 0], [555, 6980, 34], [9, 139, 4878]] | but_train |
| 0.808915 | 0.792961 | 0.789766 | 0.86014 | 0.666667 | [[5124, 1364, 0], [380, 2337, 0], [0, 35, 70]] | but_val |
| 0.845884 | 0.728523 | 0.899381 | 0.997076 | 0.285714 | [[581, 65, 0], [1, 341, 0], [0, 110, 44]] | but_test |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_cand6529_init_seed20261022_4ep_metrics.csv`
Predictions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_cand6529_init_seed20261022_4ep_predictions.csv`
Checkpoint: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_interpretable_audit\but_capacity_cand6529_init_seed20261022_4ep_best.pt`
