# Bad Specificity QRS-Aware Hard-Negative Search

Synthetic-only bad-priority repair using low-detail good/medium hard negatives plus QRS visibility/detector-agreement features for the binary bad detector. Original BUT remains report-only.

## Best Original Acc Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---:|---|
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.360 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.365 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.370 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.375 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.380 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.385 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.390 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880382 | 0.798274 | 0.878297 | 0.915951 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.395 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880264 | 0.797574 | 0.878297 | 0.915951 | 0.513382 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.400 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.880264 | 0.797574 | 0.878297 | 0.915951 | 0.513382 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.405 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879792 | 0.796604 | 0.878022 | 0.915047 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.390 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879792 | 0.796604 | 0.878022 | 0.915047 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.395 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879792 | 0.796604 | 0.878022 | 0.915047 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.400 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879792 | 0.796604 | 0.878022 | 0.915047 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.405 soft_logit |
| hardneg_badplus_base_bw3p4_hgb_spec | 0.879792 | 0.794003 | 0.878846 | 0.915951 | 0.498783 | 0.304795 | hardneg_badplus bw=3.4 score=hgb_spec thr=0.570 soft_logit |
| hardneg_badplus_base_bw3p4_hgb_spec | 0.879792 | 0.794003 | 0.878846 | 0.915951 | 0.498783 | 0.304795 | hardneg_badplus bw=3.4 score=hgb_spec thr=0.575 soft_logit |
| hardneg_badplus_base_bw3p4_hgb_spec | 0.879792 | 0.794003 | 0.878846 | 0.915951 | 0.498783 | 0.304795 | hardneg_badplus bw=3.4 score=hgb_spec thr=0.580 soft_logit |
| hardneg_badplus_base_bw3p4_hgb_spec | 0.879792 | 0.794003 | 0.878846 | 0.915951 | 0.498783 | 0.304795 | hardneg_badplus bw=3.4 score=hgb_spec thr=0.585 soft_logit |
| hardneg_badplus_base_bw3p4_hgb_spec | 0.879792 | 0.794003 | 0.878846 | 0.915951 | 0.498783 | 0.304795 | hardneg_badplus bw=3.4 score=hgb_spec thr=0.590 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.796274 | 0.878022 | 0.914822 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.375 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.796274 | 0.878022 | 0.914822 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.380 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.796274 | 0.878022 | 0.914822 | 0.515815 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.385 soft_logit |
| hardneg_badplus_base_bw3p4_tri_mean | 0.879674 | 0.795916 | 0.878297 | 0.914822 | 0.513382 | 0.315068 | hardneg_badplus bw=3.4 score=tri_mean thr=0.405 soft_logit |
| hardneg_badplus_base_bw3p4_hgb_spec | 0.879674 | 0.793288 | 0.878846 | 0.915951 | 0.496350 | 0.304795 | hardneg_badplus bw=3.4 score=hgb_spec thr=0.595 soft_logit |
| hardneg_badplus_base_bw3p4_hgb_spec | 0.879674 | 0.793288 | 0.878846 | 0.915951 | 0.496350 | 0.304795 | hardneg_badplus bw=3.4 score=hgb_spec thr=0.600 soft_logit |

## Best Original Acc With Bad Recall >= 0.55

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---:|---|

## Best Original Bad Recall Report-Only

| Candidate | Acc | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---|
| hardneg_balanced_base_bw2p6_tri_mean | 0.869765 | 0.860165 | 0.908269 | 0.540146 | 0.352740 | hardneg_balanced bw=2.6 score=tri_mean thr=0.465 hard |
| hardneg_balanced_base_bw2p6_tri_mean | 0.869175 | 0.858791 | 0.908269 | 0.540146 | 0.352740 | hardneg_balanced bw=2.6 score=tri_mean thr=0.460 hard |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877197 | 0.867033 | 0.917533 | 0.532847 | 0.342466 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.360 soft_logit |
| hardneg_balanced_base_bw2p6_tri_mean | 0.874838 | 0.872802 | 0.908269 | 0.532847 | 0.342466 | hardneg_balanced bw=2.6 score=tri_mean thr=0.460 protect_strong_good |
| hardneg_badplus_base_bw4p2_hgb_spec | 0.878849 | 0.868132 | 0.920018 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=hgb_spec thr=0.530 soft_logit |
| hardneg_badplus_base_bw4p2_hgb_spec | 0.878849 | 0.868132 | 0.920018 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=hgb_spec thr=0.535 soft_logit |
| hardneg_badplus_base_bw4p2_hgb_spec | 0.878849 | 0.868132 | 0.920018 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=hgb_spec thr=0.540 soft_logit |
| hardneg_badplus_base_bw4p2_hgb_spec | 0.878849 | 0.868132 | 0.920018 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=hgb_spec thr=0.545 soft_logit |
| hardneg_badplus_base_bw4p2_hgb_spec | 0.878849 | 0.868132 | 0.920018 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=hgb_spec thr=0.550 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877433 | 0.867308 | 0.917985 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.390 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877433 | 0.867308 | 0.917985 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.395 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877433 | 0.867308 | 0.917985 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.400 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877433 | 0.867308 | 0.917985 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.405 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877197 | 0.867033 | 0.917759 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.365 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877197 | 0.867033 | 0.917759 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.370 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877197 | 0.867033 | 0.917759 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.375 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877197 | 0.867033 | 0.917759 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.380 soft_logit |
| hardneg_badplus_base_bw4p2_mlp_balanced | 0.877197 | 0.867033 | 0.917759 | 0.530414 | 0.339041 | hardneg_badplus bw=4.2 score=mlp_balanced thr=0.385 soft_logit |
| hardneg_balanced_base_bw2p6_tri_mean | 0.876017 | 0.875824 | 0.908269 | 0.530414 | 0.339041 | hardneg_balanced bw=2.6 score=tri_mean thr=0.460 medium_or_uncertain |
| hardneg_balanced_base_bw2p6_mlp_balanced | 0.867052 | 0.855220 | 0.908043 | 0.530414 | 0.339041 | hardneg_balanced bw=2.6 score=mlp_balanced thr=0.575 protect_strong_nonbad |
| hardneg_balanced_base_bw2p6_mlp_balanced | 0.867052 | 0.855220 | 0.908043 | 0.530414 | 0.339041 | hardneg_balanced bw=2.6 score=mlp_balanced thr=0.580 protect_strong_nonbad |
| hardneg_balanced_base_bw2p6_mlp_balanced | 0.867052 | 0.855220 | 0.908043 | 0.530414 | 0.339041 | hardneg_balanced bw=2.6 score=mlp_balanced thr=0.585 protect_strong_nonbad |
| hardneg_balanced_base_bw2p6_mlp_balanced | 0.866934 | 0.854945 | 0.908043 | 0.530414 | 0.339041 | hardneg_balanced bw=2.6 score=mlp_balanced thr=0.570 protect_strong_nonbad |
| hardneg_badplus_base_bw4p2_hgb_spec | 0.878731 | 0.868132 | 0.920018 | 0.527981 | 0.335616 | hardneg_badplus bw=4.2 score=hgb_spec thr=0.555 soft_logit |
| hardneg_badplus_base_bw4p2_hgb_recall | 0.878495 | 0.867857 | 0.919792 | 0.527981 | 0.335616 | hardneg_badplus bw=4.2 score=hgb_recall thr=0.430 soft_logit |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_specificity_qrsaware_metrics.csv`
- Grid CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_specificity_qrsaware_grid.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_specificity_qrsaware_summary.json`
