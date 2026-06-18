# Bad Specificity Hard-Negative Search

Synthetic-only bad-priority repair using low-detail good/medium hard negatives for the binary bad detector. Original BUT remains report-only.

## Best Original Acc Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---:|---|
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879910 | 0.796124 | 0.878846 | 0.915273 | 0.508516 | 0.321918 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.575 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879910 | 0.796124 | 0.878846 | 0.915273 | 0.508516 | 0.321918 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.580 soft_logit |
| hardneg_badplus_base_bw3p4_tri_mean | 0.879792 | 0.796133 | 0.878297 | 0.915273 | 0.510949 | 0.321918 | hardneg_badplus bw=3.4 score=tri_mean thr=0.530 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879792 | 0.795408 | 0.878846 | 0.915273 | 0.506083 | 0.318493 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.585 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.879792 | 0.794271 | 0.878846 | 0.915951 | 0.498783 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.675 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.879792 | 0.794271 | 0.878846 | 0.915951 | 0.498783 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.680 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.879792 | 0.794271 | 0.878846 | 0.915951 | 0.498783 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.685 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.879792 | 0.794271 | 0.878846 | 0.915951 | 0.498783 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.690 soft_logit |
| hardneg_badplus_base_bw3p4_tri_mean | 0.879674 | 0.795430 | 0.878297 | 0.915273 | 0.508516 | 0.321918 | hardneg_badplus bw=3.4 score=tri_mean thr=0.535 soft_logit |
| hardneg_badplus_base_bw3p4_tri_mean | 0.879674 | 0.795430 | 0.878297 | 0.915273 | 0.508516 | 0.321918 | hardneg_badplus bw=3.4 score=tri_mean thr=0.540 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.794320 | 0.878846 | 0.915499 | 0.501217 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.600 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.794320 | 0.878846 | 0.915499 | 0.501217 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.605 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.794320 | 0.878846 | 0.915499 | 0.501217 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.610 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.794320 | 0.878846 | 0.915499 | 0.501217 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.615 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_mean | 0.879674 | 0.794320 | 0.878846 | 0.915499 | 0.501217 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_mean thr=0.620 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.879674 | 0.793555 | 0.878846 | 0.915951 | 0.496350 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.695 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.879674 | 0.793555 | 0.878846 | 0.915951 | 0.496350 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.700 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_hgb_minboost | 0.879674 | 0.793555 | 0.878846 | 0.915951 | 0.496350 | 0.315068 | hardneg_badplus bw=3.4 score=mlp_hgb_minboost thr=0.705 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_plus_base | 0.879556 | 0.796587 | 0.878297 | 0.914144 | 0.518248 | 0.325342 | hardneg_badplus bw=3.4 score=mlp_plus_base thr=0.395 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_balanced | 0.879556 | 0.796220 | 0.878297 | 0.914370 | 0.515815 | 0.325342 | hardneg_badplus bw=3.4 score=mlp_balanced thr=0.625 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_balanced | 0.879556 | 0.796220 | 0.878297 | 0.914370 | 0.515815 | 0.325342 | hardneg_badplus bw=3.4 score=mlp_balanced thr=0.630 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_balanced | 0.879556 | 0.796220 | 0.878297 | 0.914370 | 0.515815 | 0.325342 | hardneg_badplus bw=3.4 score=mlp_balanced thr=0.635 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_plus_base | 0.879556 | 0.796220 | 0.878297 | 0.914370 | 0.515815 | 0.325342 | hardneg_badplus bw=3.4 score=mlp_plus_base thr=0.430 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_plus_base | 0.879556 | 0.796220 | 0.878297 | 0.914370 | 0.515815 | 0.325342 | hardneg_badplus bw=3.4 score=mlp_plus_base thr=0.435 soft_logit |
| hardneg_badplus_base_bw3p4_mlp_plus_base | 0.879556 | 0.796220 | 0.878297 | 0.914370 | 0.515815 | 0.325342 | hardneg_badplus bw=3.4 score=mlp_plus_base thr=0.440 soft_logit |

## Best Original Acc With Bad Recall >= 0.55

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---:|---|
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.874130 | 0.797093 | 0.877473 | 0.898328 | 0.583942 | 0.414384 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.390 medium_or_uncertain |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.873776 | 0.796177 | 0.877473 | 0.897650 | 0.583942 | 0.414384 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.385 medium_or_uncertain |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.873658 | 0.795873 | 0.877473 | 0.897424 | 0.583942 | 0.414384 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.380 medium_or_uncertain |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.873540 | 0.795349 | 0.877473 | 0.897198 | 0.583942 | 0.414384 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.375 medium_or_uncertain |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.873422 | 0.795047 | 0.877473 | 0.896972 | 0.583942 | 0.414384 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.370 medium_or_uncertain |
| hardneg_balanced_base_bw3p8_tri_mean | 0.870827 | 0.789026 | 0.877747 | 0.891324 | 0.588808 | 0.421233 | hardneg_balanced bw=3.8 score=tri_mean thr=0.560 medium_or_uncertain |
| hardneg_balanced_base_bw2p6_recall_with_spec_gate | 0.870827 | 0.784546 | 0.875275 | 0.896746 | 0.552311 | 0.369863 | hardneg_balanced bw=2.6 score=recall_with_spec_gate thr=0.825 protect_strong_good |
| hardneg_balanced_base_bw3p2_recall_with_spec_gate | 0.870827 | 0.782744 | 0.868681 | 0.902169 | 0.552311 | 0.369863 | hardneg_balanced bw=3.2 score=recall_with_spec_gate thr=0.660 protect_strong_nonbad |
| hardneg_balanced_base_bw3p2_recall_with_spec_gate | 0.870709 | 0.782458 | 0.868681 | 0.901943 | 0.552311 | 0.369863 | hardneg_balanced bw=3.2 score=recall_with_spec_gate thr=0.650 protect_strong_nonbad |
| hardneg_balanced_base_bw3p2_recall_with_spec_gate | 0.870709 | 0.782458 | 0.868681 | 0.901943 | 0.552311 | 0.369863 | hardneg_balanced bw=3.2 score=recall_with_spec_gate thr=0.655 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_tri_mean | 0.870591 | 0.788453 | 0.877747 | 0.890872 | 0.588808 | 0.421233 | hardneg_balanced bw=3.8 score=tri_mean thr=0.555 medium_or_uncertain |
| hardneg_balanced_base_bw2p6_recall_with_spec_gate | 0.870473 | 0.784470 | 0.873626 | 0.896746 | 0.559611 | 0.380137 | hardneg_balanced bw=2.6 score=recall_with_spec_gate thr=0.825 hard |
| hardneg_balanced_base_bw3p8_tri_mean | 0.869883 | 0.783906 | 0.864835 | 0.900813 | 0.581509 | 0.410959 | hardneg_balanced bw=3.8 score=tri_mean thr=0.450 protect_strong_nonbad |
| hardneg_balanced_base_bw3p2_recall_with_spec_gate | 0.869883 | 0.782475 | 0.877198 | 0.892906 | 0.557178 | 0.376712 | hardneg_balanced bw=3.2 score=recall_with_spec_gate thr=0.750 medium_or_uncertain |
| hardneg_balanced_base_bw3p2_recall_with_spec_gate | 0.869883 | 0.780437 | 0.867857 | 0.901039 | 0.552311 | 0.369863 | hardneg_balanced bw=3.2 score=recall_with_spec_gate thr=0.645 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_tri_mean | 0.869293 | 0.783126 | 0.863462 | 0.900362 | 0.586375 | 0.417808 | hardneg_balanced bw=3.8 score=tri_mean thr=0.445 protect_strong_nonbad |
| hardneg_balanced_base_bw3p2_recall_with_spec_gate | 0.869175 | 0.781153 | 0.874725 | 0.892906 | 0.564477 | 0.386986 | hardneg_balanced bw=3.2 score=recall_with_spec_gate thr=0.750 protect_strong_good |
| hardneg_balanced_base_bw3p8_tri_mean | 0.868114 | 0.782606 | 0.870330 | 0.891776 | 0.593674 | 0.428082 | hardneg_balanced bw=3.8 score=tri_mean thr=0.565 protect_strong_good |
| hardneg_balanced_base_bw3p8_tri_mean | 0.867878 | 0.782375 | 0.870055 | 0.891324 | 0.596107 | 0.431507 | hardneg_balanced bw=3.8 score=tri_mean thr=0.560 protect_strong_good |
| hardneg_balanced_base_bw3p8_tri_mean | 0.867642 | 0.781835 | 0.870055 | 0.890872 | 0.596107 | 0.431507 | hardneg_balanced bw=3.8 score=tri_mean thr=0.555 protect_strong_good |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.867406 | 0.780659 | 0.860989 | 0.898328 | 0.591241 | 0.424658 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.390 protect_strong_good |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.867052 | 0.779852 | 0.860989 | 0.897650 | 0.591241 | 0.424658 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.385 protect_strong_good |
| hardneg_balanced_base_bw3p8_recall_with_spec_gate | 0.866934 | 0.777504 | 0.861813 | 0.897198 | 0.586375 | 0.417808 | hardneg_balanced bw=3.8 score=recall_with_spec_gate thr=0.595 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.866816 | 0.779305 | 0.860714 | 0.897424 | 0.591241 | 0.424658 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.380 protect_strong_good |
| hardneg_balanced_base_bw3p8_mlp_hgb_minboost | 0.866580 | 0.778578 | 0.860440 | 0.897198 | 0.591241 | 0.424658 | hardneg_balanced bw=3.8 score=mlp_hgb_minboost thr=0.375 protect_strong_good |

## Best Original Bad Recall Report-Only

| Candidate | Acc | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---|
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.853722 | 0.835714 | 0.891550 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.450 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.853132 | 0.835165 | 0.890872 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.445 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.852542 | 0.834615 | 0.890194 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.440 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.851952 | 0.833516 | 0.889968 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.435 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.851480 | 0.832967 | 0.889516 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.430 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.850537 | 0.831593 | 0.888839 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.425 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.850301 | 0.831044 | 0.888839 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.420 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.849829 | 0.830495 | 0.888387 | 0.605839 | 0.445205 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.415 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_hgb_mean | 0.852542 | 0.835714 | 0.889516 | 0.603406 | 0.441781 | hardneg_balanced bw=3.8 score=mlp_hgb_mean thr=0.390 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.854312 | 0.836813 | 0.892228 | 0.600973 | 0.438356 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.460 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_plus_base | 0.853958 | 0.835989 | 0.892228 | 0.600973 | 0.438356 | hardneg_balanced bw=3.8 score=mlp_plus_base thr=0.455 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_hgb_mean | 0.853368 | 0.837363 | 0.889968 | 0.600973 | 0.438356 | hardneg_balanced bw=3.8 score=mlp_hgb_mean thr=0.400 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_hgb_mean | 0.853132 | 0.836813 | 0.889968 | 0.600973 | 0.438356 | hardneg_balanced bw=3.8 score=mlp_hgb_mean thr=0.395 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.849829 | 0.830495 | 0.888839 | 0.600973 | 0.438356 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.630 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_tri_mean | 0.862451 | 0.857143 | 0.891324 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=tri_mean thr=0.560 hard |
| hardneg_balanced_base_bw3p8_tri_mean | 0.861508 | 0.855495 | 0.890872 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=tri_mean thr=0.555 hard |
| hardneg_balanced_base_bw3p8_mlp_hgb_mean | 0.854312 | 0.838736 | 0.890872 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_hgb_mean thr=0.405 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.853604 | 0.835440 | 0.892228 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.675 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.853250 | 0.835165 | 0.891776 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.670 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.852542 | 0.834615 | 0.890872 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.665 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.852188 | 0.834066 | 0.890646 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.660 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.851834 | 0.833242 | 0.890646 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.655 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.851245 | 0.832692 | 0.889968 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.650 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.850655 | 0.831868 | 0.889516 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.645 protect_strong_nonbad |
| hardneg_balanced_base_bw3p8_mlp_balanced | 0.850301 | 0.831319 | 0.889291 | 0.598540 | 0.434932 | hardneg_balanced bw=3.8 score=mlp_balanced thr=0.640 protect_strong_nonbad |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_specificity_hardneg_metrics.csv`
- Grid CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_specificity_hardneg_grid.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_specificity_hardneg_summary.json`
