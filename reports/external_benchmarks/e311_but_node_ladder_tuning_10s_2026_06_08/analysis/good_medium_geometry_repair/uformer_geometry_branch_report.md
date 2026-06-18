# UFormer + SQI/Geometry Branch Experiment

Experiment-only normal checkpoint path. Mainline UFormer code/checkpoints are not overwritten.

## Best Node-Test Result

- Variant: `nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal`
- Node-test acc: `0.965318`
- Node-test good/medium/bad recall: `0.961264` / `0.975825` / `0.888078`
- Useful >=0.94 gate: `False`
- Promotion >=0.95 gate: `False`

## Metrics

| created_at | node_id | variant_id | model_kind | prediction_mode | node_test_acc | node_test_macro_f1 | node_test_good_recall | node_test_medium_recall | node_test_bad_recall | node_all_acc | node_all_good_recall | node_all_medium_recall | node_all_bad_recall | original_test_all_10s_plus_acc | original_test_all_10s_plus_good_recall | original_test_all_10s_plus_medium_recall | original_test_all_10s_plus_bad_recall | original_test_main_without_bad_stress_acc | original_test_main_without_bad_stress_good_recall | original_test_main_without_bad_stress_medium_recall | original_test_main_without_bad_stress_bad_recall | eligible_useful_094 | eligible_promotion_095 | probs_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-14T19:15:50 | N17043_gm_probe | nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal | uformer_geometry_branch | raw | 0.965318 | 0.935857 | 0.961264 | 0.975825 | 0.888078 | 0.985769 | 0.989908 | 0.976571 | 0.990918 | 0.965318 | 0.961264 | 0.975825 | 0.888078 | 0.969701 | 0.961264 | 0.975825 | 1.000000 | False | False | E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\predictions\nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal_probs.npz |

## Notes

- Feature normalization is fit on the declared train split only.
- Original BUT is bucketed report-only and never used for selection.
- The branch uses 47 SQI/geometry columns and saves schema/stats inside the checkpoint.
