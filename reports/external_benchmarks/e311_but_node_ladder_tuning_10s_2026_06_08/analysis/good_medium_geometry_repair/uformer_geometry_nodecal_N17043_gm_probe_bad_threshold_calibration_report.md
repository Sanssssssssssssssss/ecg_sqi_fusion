# UFormer Geometry Bad-Threshold Calibration

Selection uses node train+val only. Test/original buckets are report-only diagnostics.

## Metrics

| variant_id | config_name | bucket | threshold | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal | geom_tabular_dualbad_internal | trainval_selection | 0.130000 | 24479 | 0.992933 | 0.993145 | 0.997687 | 0.977104 | 1.000000 | 0.989566 | 0.994911 | 0.999795 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal | geom_tabular_dualbad_internal | node_test_original_test_all_10s+ | 0.130000 | 8477 | 0.963548 | 0.930683 | 0.956319 | 0.972887 | 0.927007 | 0.976712 | 0.971790 | 0.790456 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal | geom_tabular_dualbad_internal | original_test_main_without_bad_stress | 0.130000 | 8185 | 0.965913 | 0.881427 | 0.956319 | 0.972887 | 1.000000 | 0.978634 | 0.976860 | 0.540909 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal | geom_tabular_dualbad_internal | bad_core_nearboundary | 0.130000 | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal | geom_tabular_dualbad_internal | bad_outlier_stress | 0.130000 | 292 | 0.897260 | 0.315283 | 0.000000 | 0.000000 | 0.897260 | 0.000000 | 0.000000 | 1.000000 |
| nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal | geom_tabular_dualbad_internal | node_all_original_all_10s+ | 0.130000 | 32956 | 0.985374 | 0.985233 | 0.988852 | 0.975348 | 0.994324 | 0.986883 | 0.985174 | 0.980959 |

## Notes

- A single bad posterior threshold is selected from train+val probabilities.
- No original test bucket is used to choose the threshold.
- The underlying checkpoint remains a normal UFormer geometry-branch checkpoint.
