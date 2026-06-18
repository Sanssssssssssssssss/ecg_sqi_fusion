# Bad-Priority Structures

Synthetic-only search focused on learning bad-specific flatline/dropout/low-detail waveform features. Original BUT is report-only.

## Best Original Acc Report-Only

| Candidate | Acc | Macro-F1 | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---:|---|
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.881444 | 0.794415 | 0.899451 | 0.901491 | 0.506083 | 0.311644 | bw=3.8 bad_features bad_hgb23 thr=0.960 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.881444 | 0.794415 | 0.899451 | 0.901491 | 0.506083 | 0.311644 | bw=3.8 bad_features bad_hgb23 thr=0.960 non_good_confident |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.881326 | 0.793703 | 0.899451 | 0.901491 | 0.503650 | 0.308219 | bw=3.8 bad_features bad_hgb23 thr=0.960 medium_only |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.881208 | 0.792621 | 0.899451 | 0.901717 | 0.498783 | 0.301370 | bw=3.8 bad_features bad_hgb23 thr=0.970 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.881208 | 0.792621 | 0.899451 | 0.901717 | 0.498783 | 0.301370 | bw=3.8 bad_features bad_hgb23 thr=0.970 non_good_confident |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.881090 | 0.791902 | 0.899451 | 0.901717 | 0.496350 | 0.297945 | bw=3.8 bad_features bad_hgb23 thr=0.970 medium_only |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.880736 | 0.789767 | 0.899451 | 0.901717 | 0.489051 | 0.287671 | bw=3.8 bad_features bad_hgb23 thr=0.980 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb23 | 0.880618 | 0.789041 | 0.899451 | 0.901717 | 0.486618 | 0.284247 | bw=3.8 bad_features bad_hgb23 thr=0.980 medium_only |
| badprio_balanced_base_bw3p0_bad_features_bad_hgb23 | 0.880264 | 0.792167 | 0.893681 | 0.905106 | 0.493917 | 0.321918 | bw=3.0 bad_features bad_hgb23 thr=0.960 hard |
| badprio_balanced_base_bw3p0_bad_features_bad_hgb23 | 0.880146 | 0.791057 | 0.893681 | 0.905332 | 0.489051 | 0.315068 | bw=3.0 bad_features bad_hgb23 thr=0.970 hard |
| badprio_balanced_base_bw3p0_bad_features_bad_hgb23 | 0.880028 | 0.790714 | 0.893681 | 0.905106 | 0.489051 | 0.315068 | bw=3.0 bad_features bad_hgb23 thr=0.960 medium_only |
| badprio_balanced_base_bw3p0_bad_features_bad_hgb23 | 0.880028 | 0.790714 | 0.893681 | 0.905106 | 0.489051 | 0.315068 | bw=3.0 bad_features bad_hgb23 thr=0.960 non_good_confident |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.880028 | 0.784593 | 0.899451 | 0.902169 | 0.469586 | 0.260274 | bw=3.8 bad_features bad_rf thr=0.910 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.880028 | 0.784593 | 0.899451 | 0.902169 | 0.469586 | 0.260274 | bw=3.8 bad_features bad_rf thr=0.910 medium_only |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.880028 | 0.784593 | 0.899451 | 0.902169 | 0.469586 | 0.260274 | bw=3.8 bad_features bad_rf thr=0.910 non_good_confident |
| badprio_balanced_base_bw3p0_bad_features_bad_hgb23 | 0.879910 | 0.789594 | 0.893681 | 0.905332 | 0.484185 | 0.308219 | bw=3.0 bad_features bad_hgb23 thr=0.970 medium_only |
| badprio_balanced_base_bw3p0_bad_features_bad_hgb23 | 0.879910 | 0.789594 | 0.893681 | 0.905332 | 0.484185 | 0.308219 | bw=3.0 bad_features bad_hgb23 thr=0.970 non_good_confident |
| badprio_balanced_base_bw3p0_bad_features_bad_hgb23 | 0.879910 | 0.789611 | 0.893681 | 0.905332 | 0.484185 | 0.308219 | bw=3.0 bad_features bad_hgb23 thr=0.980 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.879910 | 0.783858 | 0.899451 | 0.902169 | 0.467153 | 0.256849 | bw=3.8 bad_features bad_rf thr=0.920 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.879910 | 0.783858 | 0.899451 | 0.902169 | 0.467153 | 0.256849 | bw=3.8 bad_features bad_rf thr=0.920 medium_only |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.879910 | 0.783858 | 0.899451 | 0.902169 | 0.467153 | 0.256849 | bw=3.8 bad_features bad_rf thr=0.920 non_good_confident |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb31 | 0.879792 | 0.787613 | 0.899451 | 0.899684 | 0.491484 | 0.291096 | bw=3.8 bad_features bad_hgb31 thr=0.980 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.879792 | 0.783120 | 0.899451 | 0.902169 | 0.464720 | 0.253425 | bw=3.8 bad_features bad_rf thr=0.930 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_rf | 0.879792 | 0.783120 | 0.899451 | 0.902169 | 0.464720 | 0.253425 | bw=3.8 bad_features bad_rf thr=0.930 medium_only |
| badprio_balanced_base_bw3p8_bad_features_bad_hgb31 | 0.879674 | 0.786894 | 0.899451 | 0.899684 | 0.489051 | 0.287671 | bw=3.8 bad_features bad_hgb31 thr=0.980 medium_only |

## Best Original Bad Recall Report-Only

| Candidate | Acc | Good R | Medium R | Bad R | Bad outlier R | Params |
|---|---:|---:|---:|---:|---:|---|
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.863867 | 0.853297 | 0.895617 | 0.615572 | 0.458904 | bw=3.0 bad_features bad_mlp thr=0.460 hard |
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.863513 | 0.852747 | 0.895391 | 0.615572 | 0.458904 | bw=3.0 bad_features bad_mlp thr=0.450 hard |
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.862923 | 0.852747 | 0.894261 | 0.615572 | 0.458904 | bw=3.0 bad_features bad_mlp thr=0.440 hard |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.865636 | 0.864011 | 0.890420 | 0.613139 | 0.455479 | bw=3.8 bad_features bad_mlp thr=0.460 hard |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.865283 | 0.863462 | 0.890194 | 0.613139 | 0.455479 | bw=3.8 bad_features bad_mlp thr=0.450 hard |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.864693 | 0.863462 | 0.889065 | 0.613139 | 0.455479 | bw=3.8 bad_features bad_mlp thr=0.440 hard |
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.867052 | 0.861813 | 0.895391 | 0.608273 | 0.448630 | bw=3.0 bad_features bad_mlp thr=0.450 non_good_confident |
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.866462 | 0.861813 | 0.894261 | 0.608273 | 0.448630 | bw=3.0 bad_features bad_mlp thr=0.440 non_good_confident |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.869057 | 0.873352 | 0.890194 | 0.603406 | 0.441781 | bw=3.8 bad_features bad_mlp thr=0.450 non_good_confident |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.868468 | 0.873352 | 0.889065 | 0.603406 | 0.441781 | bw=3.8 bad_features bad_mlp thr=0.440 non_good_confident |
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.867760 | 0.863736 | 0.895617 | 0.603406 | 0.441781 | bw=3.0 bad_features bad_mlp thr=0.460 medium_only |
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.867642 | 0.863736 | 0.895391 | 0.603406 | 0.441781 | bw=3.0 bad_features bad_mlp thr=0.450 medium_only |
| badprio_badplus_base_bw3p0_bad_features_bad_mlp | 0.867052 | 0.863736 | 0.894261 | 0.603406 | 0.441781 | bw=3.0 bad_features bad_mlp thr=0.440 medium_only |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.869411 | 0.874725 | 0.890420 | 0.596107 | 0.431507 | bw=3.8 bad_features bad_mlp thr=0.460 medium_only |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.869293 | 0.874725 | 0.890194 | 0.596107 | 0.431507 | bw=3.8 bad_features bad_mlp thr=0.450 medium_only |
| badprio_badplus_base_bw3p8_bad_features_bad_mlp | 0.868704 | 0.874725 | 0.889065 | 0.596107 | 0.431507 | bw=3.8 bad_features bad_mlp thr=0.440 medium_only |
| badprio_balanced_base_bw3p8_bad_features_bad_mlp | 0.877669 | 0.898077 | 0.889516 | 0.569343 | 0.393836 | bw=3.8 bad_features bad_mlp thr=0.870 hard |
| badprio_balanced_base_bw3p0_bad_features_bad_mlp | 0.876961 | 0.892033 | 0.893131 | 0.569343 | 0.393836 | bw=3.0 bad_features bad_mlp thr=0.870 hard |
| badprio_balanced_base_bw2p4_bad_features_bad_mlp | 0.874130 | 0.880220 | 0.897876 | 0.564477 | 0.386986 | bw=2.4 bad_features bad_mlp thr=0.870 hard |
| badprio_balanced_base_bw3p8_bad_features_bad_mlp | 0.877669 | 0.899176 | 0.889516 | 0.559611 | 0.380137 | bw=3.8 bad_features bad_mlp thr=0.870 non_good_confident |
| badprio_balanced_base_bw3p8_bad_features_bad_mlp | 0.877551 | 0.898352 | 0.889968 | 0.559611 | 0.380137 | bw=3.8 bad_features bad_mlp thr=0.880 hard |
| badprio_balanced_base_bw3p0_bad_features_bad_mlp | 0.877197 | 0.893681 | 0.893131 | 0.559611 | 0.380137 | bw=3.0 bad_features bad_mlp thr=0.870 non_good_confident |
| badprio_balanced_base_bw3p0_bad_features_bad_mlp | 0.876843 | 0.892308 | 0.893583 | 0.559611 | 0.380137 | bw=3.0 bad_features bad_mlp thr=0.880 hard |
| badprio_badplus_base_bw2p4_bad_features_bad_hgb23 | 0.875664 | 0.875549 | 0.905332 | 0.557178 | 0.376712 | bw=2.4 bad_features bad_hgb23 thr=0.950 hard |
| badprio_balanced_base_bw2p4_bad_features_bad_mlp | 0.874484 | 0.881868 | 0.897876 | 0.557178 | 0.376712 | bw=2.4 bad_features bad_mlp thr=0.870 non_good_confident |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_priority_structures_metrics.csv`
- Grid CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_priority_structures_grid.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_priority_structures_summary.json`
