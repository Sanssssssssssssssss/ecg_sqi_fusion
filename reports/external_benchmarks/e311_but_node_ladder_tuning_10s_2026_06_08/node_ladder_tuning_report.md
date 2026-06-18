# Original-Aware SemiCleanBUT Node Ladder

This package freezes each SemiClean boundary as a node, fits PTB synthetic data to that node in 64D/PCA space, then trains and promotes only after the node passes. It is a diagnostic/generator workflow, not a replacement for formal BUT original test.

## Node Registry

| node_id | level | role | status | existing_acc | existing_medium_recall | existing_bad_recall | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N3600_anchor | 3600 | frozen_anchor | promoted | 0.9544 | 0.9147 | 1.0000 | Current feasible 0.95 node; freeze as sanity baseline. |
| N4200_bridge | 4200 | active_bridge | promoted | 0.9356 | 0.8957 | 0.9714 | First failing widening node; add good/medium overlap and controlled near-bad shell. |
| N4800_wide | 4800 | wide_after_bridge | needs_generator_refine | 0.9279 | 0.8808 | 0.9744 | Wider target after N4200 is stable; increases medium overlap and bad near-boundary coverage. |
| N5200_stress | 5200 | stress_diagnostic_only | needs_generator_refine | 0.9049 | 0.8719 | 0.9208 | Stress node with outlier shell; do not promote until N4800 succeeds. |
| N5600_probe | 5600 | lower_confidence_probe | no_metrics |  |  |  | Boundary probe after N5200: more good/medium overlap plus controlled low-confidence shell; fit/visual before training. |
| N6000_probe | 6000 | extreme_boundary_probe | needs_generator_refine |  |  |  | Extreme stress probe for finding the widest still-learnable 0.95-ish boundary; fit/visual only by default. |
| N6000_goodcover_retry | 6000 | good_outer_shell_retry | promoted |  |  |  | Same N6000 boundary, but generator/score focus on good 64D outer-shell coverage before widening further. |
| N6400_gm_probe | 6400 | gm_low_confidence_probe | no_metrics |  |  |  | Good/medium low-confidence expansion after bad-side saturation; bad remains all available island/near-boundary rows. |

## Best Node 64D Fits

| node_id | rank | variant_id | node_score | node_region_score | good_64d_KS | medium_64d_KS | bad_64d_KS | good_gm_overlap_coverage | medium_gm_overlap_coverage | bad_near_boundary_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N7000_gm_trim_bad | 1 | nl_n7000_gm_trim_bad_scan_014_sc_overlap_compact_pca_core_8b486652852a | 0.1767 | 0.0883 | 0.2585 | 0.2072 | 0.6010 | 1.0000 | 0.9900 | 0.9917 |
| N7000_gm_trim_bad | 2 | nl_n7000_gm_trim_bad_scan_012_sc_overlap_compact_pca_core_a80628065b3b | 0.1790 | 0.0969 | 0.2614 | 0.2053 | 0.6017 | 1.0000 | 0.9900 | 0.9917 |
| N7000_gm_trim_bad | 3 | nl_n7000_gm_trim_bad_scan_024_sc_overlap_1530_spike_core__a5187589709c | 0.1791 | 0.1082 | 0.2614 | 0.2072 | 0.5661 | 1.0000 | 0.9900 | 0.9917 |
| N7000_gm_trim_bad | 4 | nl_n7000_gm_trim_bad_scan_011_sc_overlap_compact_pca_core_2a88470eba03 | 0.1795 | 0.0872 | 0.2684 | 0.2053 | 0.6037 | 1.0000 | 0.9900 | 0.9917 |
| N7000_gm_trim_bad | 5 | nl_n7000_gm_trim_bad_scan_013_sc_overlap_compact_pca_core_951d66bb4975 | 0.1805 | 0.1009 | 0.2629 | 0.2072 | 0.6050 | 1.0000 | 0.9900 | 0.9917 |
| N7000_gm_trim_bad | 6 | nl_n7000_gm_trim_bad_scan_009_sc_overlap_1530_locked_lowp_4e977948afea | 0.1815 | 0.1061 | 0.2629 | 0.2099 | 0.5989 | 1.0000 | 0.9956 | 0.9917 |
| N7000_gm_trim_bad | 7 | nl_n7000_gm_trim_bad_scan_015_sc_overlap_compact_pca_core_2a9e4663814f | 0.1821 | 0.1001 | 0.2684 | 0.2072 | 0.6266 | 1.0000 | 0.9900 | 0.9917 |
| N7000_gm_trim_bad | 8 | nl_n7000_gm_trim_bad_scan_004_sc_overlap_qrs_visible_comp_e8fe8036f1ff | 0.1835 | 0.1225 | 0.2614 | 0.2072 | 0.5973 | 1.0000 | 0.9900 | 0.9083 |
| N7000_gm_trim_bad | 9 | nl_n7000_gm_trim_bad_scan_019_sc_overlap_1530_hfedge_spik_216fb77cfac4 | 0.1969 | 0.1693 | 0.2684 | 0.2053 | 0.6006 | 1.0000 | 0.9900 | 0.5250 |
| N7000_gm_trim_bad | 10 | nl_n7000_gm_trim_bad_scan_010_sc_overlap_1530_locked_lowp_fd452d0d185f | 0.1975 | 0.1987 | 0.2585 | 0.2053 | 0.6501 | 1.0000 | 0.9900 | 0.9833 |
| N7000_gm_trim_bad | 11 | nl_n7000_gm_trim_bad_scan_006_sc_overlap_1530_locked_lowp_1acdcd110a0f | 0.1996 | 0.2112 | 0.2585 | 0.2072 | 0.6041 | 1.0000 | 0.9900 | 0.9917 |
| N7000_gm_trim_bad | 12 | nl_n7000_gm_trim_bad_scan_008_sc_overlap_1530_locked_lowp_0b5234ad1700 | 0.2090 | 0.2336 | 0.2614 | 0.2099 | 0.6065 | 1.0000 | 0.9956 | 0.9917 |
| N5200_stress | 13 | nl_n5200_stress_scan_054_sc_overlap_quiet_dead_core_053_5c478f3f094d | 0.2090 | 0.0859 | 0.2875 | 0.2331 | 0.5788 | 1.0000 | 0.9856 | 0.9917 |
| N7000_gm_trim_bad | 14 | nl_n7000_gm_trim_bad_scan_007_sc_overlap_1530_locked_lowp_4bc9f41c4c28 | 0.2104 | 0.2339 | 0.2684 | 0.2099 | 0.6065 | 1.0000 | 0.9956 | 0.9917 |
| N6800_gm_probe | 15 | nl_n6800_gm_probe_scan_038_sc_overlap_bandlimited_disagre_3985466cbeab | 0.2115 | 0.0988 | 0.2559 | 0.2033 | 0.5604 | 0.9978 | 0.9933 | 0.9917 |
| N6800_gm_probe | 16 | nl_n6800_gm_probe_scan_118_sc_overlap_bandlimited_disagre_94edf0e8e084 | 0.2119 | 0.0998 | 0.2559 | 0.2022 | 0.5604 | 0.9978 | 0.9944 | 0.9917 |
| N6800_gm_probe | 17 | nl_n6800_gm_probe_scan_037_sc_overlap_bandlimited_disagre_303d089a9d1f | 0.2125 | 0.0997 | 0.2565 | 0.2033 | 0.5604 | 0.9978 | 0.9933 | 0.9917 |
| N6800_gm_probe | 18 | nl_n6800_gm_probe_scan_198_sc_overlap_bandlimited_disagre_a6538daf310e | 0.2126 | 0.0982 | 0.2559 | 0.2067 | 0.5682 | 0.9978 | 0.9889 | 0.9917 |

## Node Diagnostics

| node_id | variant_id | cls_loss_variant | prediction_mode | acc | macro_f1 | good_recall | medium_recall | bad_recall | original_but_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N4200_bridge | nl_n4200_bridge_scan_094_sc_overlap_compact_pca_core_093__2d25051030f7 | hard_ce | raw | 0.9656 | 0.9656 | 0.9329 | 0.9643 | 0.9998 | 0.6090 |
| N4200_bridge | nl_n4200_bridge_scan_094_sc_overlap_compact_pca_core_093__2d25051030f7 | hard_ce | calibrated | 0.9663 | 0.9663 | 0.9548 | 0.9443 | 0.9998 | 0.6090 |
| N4800_wide | nl_n4800_wide_scan_094_sc_overlap_compact_pca_core_093_ad_0ca34d107697 | soft_sample_weighted_ce | raw | 0.9460 | 0.9460 | 0.9708 | 0.8927 | 0.9744 | 0.4733 |
| N4800_wide | nl_n4800_wide_scan_094_sc_overlap_compact_pca_core_093_ad_0ca34d107697 | soft_sample_weighted_ce | calibrated | 0.9424 | 0.9424 | 0.9742 | 0.8771 | 0.9758 | 0.4733 |
| N6000_probe | nl_n6000_probe_scan_037_sc_overlap_bandlimited_disagree_c_46a888c88ffc | soft_sample_weighted_ce | raw | 0.9052 | 0.9085 | 0.8377 | 0.9520 | 0.9287 | 0.5953 |
| N6000_probe | nl_n6000_probe_scan_037_sc_overlap_bandlimited_disagree_c_46a888c88ffc | soft_sample_weighted_ce | calibrated | 0.9091 | 0.9122 | 0.8800 | 0.9203 | 0.9292 | 0.5953 |
| N6000_goodcover_retry | nl_n6000_goodcover_retry_scan_037_sc_overlap_bandlimited__1d6f869eb41f | soft_sample_weighted_ce | raw | 0.9498 | 0.9506 | 0.9695 | 0.9478 | 0.9298 | 0.6526 |
| N6000_goodcover_retry | nl_n6000_goodcover_retry_scan_037_sc_overlap_bandlimited__1d6f869eb41f | soft_sample_weighted_ce | calibrated | 0.9520 | 0.9525 | 0.9802 | 0.9432 | 0.9300 | 0.6526 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_037_sc_overlap_bandlimited_disagre_2640f362b234 | soft_sample_weighted_ce | raw | 0.9415 | 0.9421 | 0.9696 | 0.9460 | 0.8997 | 0.5308 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_037_sc_overlap_bandlimited_disagre_2640f362b234 | soft_sample_weighted_ce | calibrated | 0.9417 | 0.9422 | 0.9785 | 0.9360 | 0.9016 | 0.5308 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_037_sc_overlap_bandlimited_disagre_91a6fe4919d0 | soft_sample_weighted_ce | raw | 0.8876 | 0.8916 | 0.9984 | 0.7629 | 0.9056 | 0.4892 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_037_sc_overlap_bandlimited_disagre_91a6fe4919d0 | soft_sample_weighted_ce | calibrated | 0.8854 | 0.8894 | 0.9985 | 0.7559 | 0.9063 | 0.4892 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_037_sc_overlap_bandlimited_disagre_2938e1399eed | soft_sample_weighted_ce | raw | 0.8993 | 0.9029 | 0.9975 | 0.7934 | 0.9092 | 0.4728 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_037_sc_overlap_bandlimited_disagre_2938e1399eed | soft_sample_weighted_ce | calibrated | 0.8812 | 0.8857 | 0.9987 | 0.7419 | 0.9092 | 0.4728 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_197_sc_overlap_bandlimited_disagre_16e79b054a7c | sample_weighted_ce | raw | 0.9317 | 0.9310 | 0.9587 | 0.9246 | 0.9063 | 0.5129 |
| N6800_gm_probe | nl_n6800_gm_probe_scan_197_sc_overlap_bandlimited_disagre_16e79b054a7c | sample_weighted_ce | calibrated | 0.9338 | 0.9334 | 0.9681 | 0.9212 | 0.9060 | 0.5129 |
| N17043_gm_trim_bad | nl_n17043_gm_trim_bad_boundaryblocks_large_badcore_balanc_678e0e2e2775 | soft_sample_weighted_ce | raw | 0.7898 | 0.8225 | 0.7265 | 0.8219 | 0.9706 | 0.5033 |
| N17043_gm_trim_bad | nl_n17043_gm_trim_bad_boundaryblocks_large_badcore_balanc_678e0e2e2775 | soft_sample_weighted_ce | calibrated | 0.7961 | 0.8283 | 0.7446 | 0.8116 | 0.9706 | 0.5033 |

## Promotion Decisions

| node_id | node_level | best_variant | metric_source | prediction_mode | acc | macro_f1 | good_recall | medium_recall | bad_recall | original_but_macro_f1 | target_acc | target_good | target_medium | target_bad | promoted | status | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N3600_anchor | 3600.0000 | sc_overlap_compact_pca_core_013_2f3509c08cf4 | existing | raw | 0.9544 | 0.9543 | 0.9483 | 0.9147 | 1.0000 | 0.4442 | 0.9500 | 0.9400 | 0.9100 | 0.9800 | True | promoted | Node meets promotion gates. |
| N4200_bridge | 4200.0000 | nl_n4200_bridge_scan_094_sc_overlap_compact_pca_core_093__2d25051030f7 | trained | calibrated | 0.9663 | 0.9663 | 0.9548 | 0.9443 | 0.9998 | 0.6090 | 0.9500 | 0.9400 | 0.9100 | 0.9800 | True | promoted | Node meets promotion gates. |
| N4800_wide | 4800.0000 | nl_n4800_wide_scan_094_sc_overlap_compact_pca_core_093_ad_0ca34d107697 | trained | raw | 0.9460 | 0.9460 | 0.9708 | 0.8927 | 0.9744 | 0.4733 | 0.9500 | 0.9400 | 0.9000 | 0.9800 | False | needs_generator_refine | Do not widen yet; refine this node generator/loss first. |
| N5200_stress | 5200.0000 | sc_overlap_compact_pca_core_013_2f3509c08cf4 | existing | raw | 0.9049 | 0.9058 | 0.9219 | 0.8719 | 0.9208 | 0.4442 | 0.9500 | 0.9200 | 0.8800 | 0.9500 | False | needs_generator_refine | Do not widen yet; refine this node generator/loss first. |
| N5600_probe |  | nan | nan | nan |  |  |  |  |  |  |  |  |  |  | False | no_metrics | No trained or existing diagnostic metrics available. |
| N6000_probe | 6000.0000 | nl_n6000_probe_scan_037_sc_overlap_bandlimited_disagree_c_46a888c88ffc | trained | calibrated | 0.9091 | 0.9122 | 0.8800 | 0.9203 | 0.9292 | 0.5953 | 0.9500 | 0.8800 | 0.8000 | 0.9000 | False | needs_generator_refine | Do not widen yet; refine this node generator/loss first. |
| N6000_goodcover_retry | 6000.0000 | nl_n6000_goodcover_retry_scan_037_sc_overlap_bandlimited__1d6f869eb41f | trained | calibrated | 0.9520 | 0.9525 | 0.9802 | 0.9432 | 0.9300 | 0.6526 | 0.9500 | 0.9200 | 0.8600 | 0.9000 | True | promoted | Node meets promotion gates. |
| N6400_gm_probe |  | nan | nan | nan |  |  |  |  |  |  |  |  |  |  | False | no_metrics | No trained or existing diagnostic metrics available. |

## Files

- `node_registry.json` / `node_registry.csv`: immutable node definitions plus current status.
- `nodes/<node_id>/node_boundary_manifest.csv`: selected BUT windows for that node.
- `nodes/<node_id>/node_target_distributions.json`: 64D target distribution and region mix.
- `nodes/<node_id>/node64_distance_leaderboard.csv`: no-training synthetic fit ranking.
- `nodes/<node_id>/figures/best_rule_node_overlay.png`: node target vs best PTB synthetic.
- `node_ladder_training_summary.jsonl` and `node_ladder_diagnostic_metrics.csv`: training and node-filtered diagnostics.

## Current Recommendation

Treat `N3600_anchor` as frozen. Work on `N4200_bridge` until it reaches the promotion gates; only then move to `N4800_wide`. If `N4200` misses, inspect medium good/medium-overlap coverage and bad near-boundary coverage before changing network size.
