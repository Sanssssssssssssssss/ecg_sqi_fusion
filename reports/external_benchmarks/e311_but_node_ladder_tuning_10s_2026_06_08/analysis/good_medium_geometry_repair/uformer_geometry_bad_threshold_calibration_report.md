# UFormer Geometry Bad-Threshold Calibration

Selection uses node train+val only. Original/test buckets are report-only diagnostics.

- Base normal checkpoint: `nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6_uformergeom_N17043_gm_probe_nodecal_geom_tabular_dualbad_internal`
- Chosen bad posterior threshold: `p_bad >= 0.13`
- Mechanism: normal model probabilities first, then a single posterior threshold for bad; no original rows are used to choose the threshold.

## Metrics
| bucket | threshold | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trainval_selection | 0.130000 | 24479 | 0.992933 | 0.993145 | 0.997687 | 0.977104 | 1.000000 | 0.989566 | 0.994911 | 0.999795 |
| node_test_original_test_all_10s+ | 0.130000 | 8477 | 0.963548 | 0.930683 | 0.956319 | 0.972887 | 0.927007 | 0.976712 | 0.971790 | 0.790456 |
| original_test_main_without_bad_stress | 0.130000 | 8185 | 0.965913 | 0.881427 | 0.956319 | 0.972887 | 1.000000 | 0.978634 | 0.976860 | 0.540909 |
| bad_core_nearboundary | 0.130000 | 119 | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| bad_outlier_stress | 0.130000 | 292 | 0.897260 | 0.315283 | 0.000000 | 0.000000 | 0.897260 | 0.000000 | 0.000000 | 1.000000 |
| node_all_original_all_10s+ | 0.130000 | 32956 | 0.985374 | 0.985233 | 0.988852 | 0.975348 | 0.994324 | 0.986883 | 0.985174 | 0.980959 |

## Confusion Matrices
### trainval_selection
```
[[13372, 31, 0], [141, 6060, 1], [0, 0, 4874]]
```
### node_test_original_test_all_10s+
```
[[3481, 102, 57], [76, 4306, 44], [7, 23, 381]]
```
### original_test_main_without_bad_stress
```
[[3481, 102, 57], [76, 4306, 44], [0, 0, 119]]
```
### bad_core_nearboundary
```
[[0, 0, 0], [0, 0, 0], [0, 0, 119]]
```
### bad_outlier_stress
```
[[0, 0, 0], [0, 0, 0], [7, 23, 262]]
```
### node_all_original_all_10s+
```
[[16853, 133, 57], [217, 10366, 45], [7, 23, 5255]]
```