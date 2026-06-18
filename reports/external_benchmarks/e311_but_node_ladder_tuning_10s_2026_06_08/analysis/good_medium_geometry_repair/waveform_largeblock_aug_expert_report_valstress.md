# Waveform Large-Block Augmentation Expert

This run converts broad boundary diagnostics into PTB/synthetic-only waveform augmentation blocks. Original BUT remains report-only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| largeblock_medium93_short_stable | synthetic_test | 0.980523 | 0.979997 | 0.979079 | 0.981331 | 0.979253 | 10 | 23 | 0 | 5 |
| largeblock_medium93_short_stable_fullonly | synthetic_test | 0.980523 | 0.979977 | 0.976987 | 0.982143 | 0.979253 | 11 | 22 | 0 | 5 |
| largeblock_medium93_short_stable_expertonly | synthetic_test | 0.984623 | 0.983889 | 0.993724 | 0.981331 | 0.983402 | 3 | 22 | 0 | 4 |
| largeblock_medium93_short_stable | original_test_all_10s+ | 0.755338 | 0.604299 | 0.904121 | 0.679169 | 0.257908 | 344 | 1093 | 0 | 78 |
| largeblock_medium93_short_stable_fullonly | original_test_all_10s+ | 0.754394 | 0.588354 | 0.906044 | 0.682784 | 0.182482 | 341 | 1172 | 0 | 104 |
| largeblock_medium93_short_stable_expertonly | original_test_all_10s+ | 0.768904 | 0.604328 | 0.895604 | 0.713285 | 0.245742 | 346 | 833 | 0 | 106 |
| largeblock_medium93_short_stable | original_all_10s+ | 0.804193 | 0.826788 | 0.756322 | 0.817557 | 0.931693 | 4148 | 1585 | 0 | 132 |
| largeblock_medium93_short_stable_fullonly | original_all_10s+ | 0.806651 | 0.830242 | 0.763187 | 0.817463 | 0.925071 | 4035 | 1693 | 0 | 162 |
| largeblock_medium93_short_stable_expertonly | original_all_10s+ | 0.796941 | 0.819334 | 0.730094 | 0.837505 | 0.930937 | 4566 | 1275 | 0 | 161 |
| largeblock_medium93_short_stable | bad_core_nearboundary | 0.638655 | 0.259829 | 0.000000 | 0.000000 | 0.638655 | 0 | 0 | 0 | 43 |
| largeblock_medium93_short_stable_fullonly | bad_core_nearboundary | 0.436975 | 0.202729 | 0.000000 | 0.000000 | 0.436975 | 0 | 0 | 0 | 67 |
| largeblock_medium93_short_stable_expertonly | bad_core_nearboundary | 0.495798 | 0.220974 | 0.000000 | 0.000000 | 0.495798 | 0 | 0 | 0 | 60 |
| largeblock_medium93_short_stable | bad_outlier_stress | 0.102740 | 0.062112 | 0.000000 | 0.000000 | 0.102740 | 0 | 0 | 0 | 35 |
| largeblock_medium93_short_stable_fullonly | bad_outlier_stress | 0.078767 | 0.048677 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 0 | 37 |
| largeblock_medium93_short_stable_expertonly | bad_outlier_stress | 0.143836 | 0.083832 | 0.000000 | 0.000000 | 0.143836 | 0 | 0 | 0 | 46 |
| largeblock_gm_slim_short_stable | synthetic_test | 0.983598 | 0.982579 | 0.993724 | 0.980519 | 0.979253 | 3 | 23 | 0 | 5 |
| largeblock_gm_slim_short_stable_fullonly | synthetic_test | 0.985136 | 0.984251 | 0.979079 | 0.988636 | 0.979253 | 10 | 14 | 0 | 5 |
| largeblock_gm_slim_short_stable_expertonly | synthetic_test | 0.984111 | 0.983049 | 0.993724 | 0.981331 | 0.979253 | 3 | 22 | 0 | 5 |
| largeblock_gm_slim_short_stable | original_test_all_10s+ | 0.804176 | 0.652886 | 0.903571 | 0.777451 | 0.211679 | 349 | 938 | 0 | 97 |
| largeblock_gm_slim_short_stable_fullonly | original_test_all_10s+ | 0.788369 | 0.659643 | 0.908791 | 0.740850 | 0.233577 | 332 | 1144 | 0 | 80 |
| largeblock_gm_slim_short_stable_expertonly | original_test_all_10s+ | 0.807361 | 0.628230 | 0.903297 | 0.790330 | 0.141119 | 350 | 908 | 0 | 128 |
| largeblock_gm_slim_short_stable | original_all_10s+ | 0.809868 | 0.838893 | 0.737546 | 0.866861 | 0.928477 | 4471 | 1362 | 0 | 151 |
| largeblock_gm_slim_short_stable_fullonly | original_all_10s+ | 0.812781 | 0.841712 | 0.754621 | 0.848419 | 0.928666 | 4182 | 1605 | 0 | 141 |
| largeblock_gm_slim_short_stable_expertonly | original_all_10s+ | 0.809291 | 0.838669 | 0.734143 | 0.873353 | 0.922800 | 4529 | 1323 | 0 | 183 |
| largeblock_gm_slim_short_stable | bad_core_nearboundary | 0.638655 | 0.259829 | 0.000000 | 0.000000 | 0.638655 | 0 | 0 | 0 | 43 |
| largeblock_gm_slim_short_stable_fullonly | bad_core_nearboundary | 0.806723 | 0.297674 | 0.000000 | 0.000000 | 0.806723 | 0 | 0 | 0 | 23 |
| largeblock_gm_slim_short_stable_expertonly | bad_core_nearboundary | 0.428571 | 0.200000 | 0.000000 | 0.000000 | 0.428571 | 0 | 0 | 0 | 68 |
| largeblock_gm_slim_short_stable | bad_outlier_stress | 0.037671 | 0.024202 | 0.000000 | 0.000000 | 0.037671 | 0 | 0 | 0 | 54 |
| largeblock_gm_slim_short_stable_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 57 |
| largeblock_gm_slim_short_stable_expertonly | bad_outlier_stress | 0.023973 | 0.015608 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 0 | 60 |
| largeblock_bad_cautious_short_stable | synthetic_test | 0.977447 | 0.972533 | 0.993724 | 0.970779 | 0.979253 | 3 | 22 | 0 | 5 |
| largeblock_bad_cautious_short_stable_fullonly | synthetic_test | 0.980523 | 0.974807 | 0.993724 | 0.974838 | 0.983402 | 3 | 14 | 0 | 4 |
| largeblock_bad_cautious_short_stable_expertonly | synthetic_test | 0.972322 | 0.965386 | 0.991632 | 0.962662 | 0.983402 | 4 | 23 | 0 | 4 |
| largeblock_bad_cautious_short_stable | original_test_all_10s+ | 0.728442 | 0.562187 | 0.879396 | 0.654089 | 0.192214 | 326 | 572 | 0 | 143 |
| largeblock_bad_cautious_short_stable_fullonly | original_test_all_10s+ | 0.720420 | 0.561390 | 0.889835 | 0.626525 | 0.231144 | 333 | 639 | 0 | 118 |
| largeblock_bad_cautious_short_stable_expertonly | original_test_all_10s+ | 0.699540 | 0.564791 | 0.818681 | 0.635563 | 0.333333 | 320 | 389 | 0 | 117 |
| largeblock_bad_cautious_short_stable | original_all_10s+ | 0.790326 | 0.800359 | 0.738661 | 0.807490 | 0.922422 | 4338 | 1018 | 0 | 221 |
| largeblock_bad_cautious_short_stable_fullonly | original_all_10s+ | 0.791722 | 0.799816 | 0.752039 | 0.785849 | 0.931504 | 4153 | 1107 | 0 | 162 |
| largeblock_bad_cautious_short_stable_expertonly | original_all_10s+ | 0.780617 | 0.786538 | 0.720589 | 0.797798 | 0.939640 | 4412 | 829 | 0 | 162 |
| largeblock_bad_cautious_short_stable | bad_core_nearboundary | 0.025210 | 0.016393 | 0.000000 | 0.000000 | 0.025210 | 0 | 0 | 0 | 116 |
| largeblock_bad_cautious_short_stable_fullonly | bad_core_nearboundary | 0.176471 | 0.100000 | 0.000000 | 0.000000 | 0.176471 | 0 | 0 | 0 | 98 |
| largeblock_bad_cautious_short_stable_expertonly | bad_core_nearboundary | 0.193277 | 0.107981 | 0.000000 | 0.000000 | 0.193277 | 0 | 0 | 0 | 96 |
| largeblock_bad_cautious_short_stable | bad_outlier_stress | 0.260274 | 0.137681 | 0.000000 | 0.000000 | 0.260274 | 0 | 0 | 0 | 27 |
| largeblock_bad_cautious_short_stable_fullonly | bad_outlier_stress | 0.253425 | 0.134791 | 0.000000 | 0.000000 | 0.253425 | 0 | 0 | 0 | 20 |
| largeblock_bad_cautious_short_stable_expertonly | bad_outlier_stress | 0.390411 | 0.187192 | 0.000000 | 0.000000 | 0.390411 | 0 | 0 | 0 | 21 |
| largeblock_bad_outlier_bridge_stable | synthetic_test | 0.974372 | 0.968192 | 0.993724 | 0.965097 | 0.983402 | 3 | 23 | 0 | 4 |
| largeblock_bad_outlier_bridge_stable_fullonly | synthetic_test | 0.973347 | 0.966642 | 0.993724 | 0.963474 | 0.983402 | 3 | 23 | 0 | 4 |
| largeblock_bad_outlier_bridge_stable_expertonly | synthetic_test | 0.968734 | 0.959500 | 0.991632 | 0.956981 | 0.983402 | 4 | 21 | 0 | 4 |
| largeblock_bad_outlier_bridge_stable | original_test_all_10s+ | 0.730093 | 0.594975 | 0.879396 | 0.634885 | 0.433090 | 324 | 573 | 0 | 46 |
| largeblock_bad_outlier_bridge_stable_fullonly | original_test_all_10s+ | 0.718061 | 0.590082 | 0.872253 | 0.613873 | 0.474453 | 304 | 550 | 0 | 33 |
| largeblock_bad_outlier_bridge_stable_expertonly | original_test_all_10s+ | 0.702843 | 0.594859 | 0.805769 | 0.625169 | 0.627737 | 330 | 337 | 0 | 26 |
| largeblock_bad_outlier_bridge_stable | original_all_10s+ | 0.799430 | 0.807094 | 0.757789 | 0.791682 | 0.949290 | 4010 | 1042 | 0 | 79 |
| largeblock_bad_outlier_bridge_stable_fullonly | original_all_10s+ | 0.799187 | 0.803359 | 0.765065 | 0.777569 | 0.952696 | 3837 | 1027 | 0 | 66 |
| largeblock_bad_outlier_bridge_stable_expertonly | original_all_10s+ | 0.782832 | 0.787855 | 0.721411 | 0.791024 | 0.964428 | 4362 | 751 | 0 | 59 |
| largeblock_bad_outlier_bridge_stable | bad_core_nearboundary | 0.789916 | 0.294210 | 0.000000 | 0.000000 | 0.789916 | 0 | 0 | 0 | 25 |
| largeblock_bad_outlier_bridge_stable_fullonly | bad_core_nearboundary | 0.857143 | 0.307692 | 0.000000 | 0.000000 | 0.857143 | 0 | 0 | 0 | 17 |
| largeblock_bad_outlier_bridge_stable_expertonly | bad_core_nearboundary | 0.974790 | 0.329078 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 0 | 3 |
| largeblock_bad_outlier_bridge_stable | bad_outlier_stress | 0.287671 | 0.148936 | 0.000000 | 0.000000 | 0.287671 | 0 | 0 | 0 | 21 |
| largeblock_bad_outlier_bridge_stable_fullonly | bad_outlier_stress | 0.318493 | 0.161039 | 0.000000 | 0.000000 | 0.318493 | 0 | 0 | 0 | 16 |
| largeblock_bad_outlier_bridge_stable_expertonly | bad_outlier_stress | 0.486301 | 0.218126 | 0.000000 | 0.000000 | 0.486301 | 0 | 0 | 0 | 23 |

## Block Counts

| candidate | block | label | n |
| --- | --- | --- | --- |
| largeblock_bad_cautious_short_stable | bad_controlled_outlier | 2 | 1098 |
| largeblock_bad_cautious_short_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 1952 |
| largeblock_bad_cautious_short_stable | gm_medium_qrslow_hardneg | 1 | 2318 |
| largeblock_bad_cautious_short_stable | gm_visible_qrs_medium_detail | 1 | 1586 |
| largeblock_bad_cautious_short_stable | nonbad_irregular_spike_hardneg | 0 | 1866 |
| largeblock_bad_cautious_short_stable | nonbad_irregular_spike_hardneg | 1 | 4478 |
| largeblock_bad_outlier_bridge_stable | bad_controlled_outlier | 2 | 1586 |
| largeblock_bad_outlier_bridge_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 1708 |
| largeblock_bad_outlier_bridge_stable | gm_medium_qrslow_hardneg | 1 | 2074 |
| largeblock_bad_outlier_bridge_stable | gm_visible_qrs_medium_detail | 1 | 1464 |
| largeblock_bad_outlier_bridge_stable | nonbad_irregular_spike_hardneg | 0 | 2513 |
| largeblock_bad_outlier_bridge_stable | nonbad_irregular_spike_hardneg | 1 | 6271 |
| largeblock_gm_slim_short_stable | bad_controlled_outlier | 2 | 268 |
| largeblock_gm_slim_short_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 2196 |
| largeblock_gm_slim_short_stable | gm_medium_qrslow_hardneg | 1 | 2684 |
| largeblock_gm_slim_short_stable | gm_visible_qrs_medium_detail | 1 | 1708 |
| largeblock_gm_slim_short_stable | nonbad_irregular_spike_hardneg | 0 | 843 |
| largeblock_gm_slim_short_stable | nonbad_irregular_spike_hardneg | 1 | 2085 |
| largeblock_medium93_short_stable | bad_controlled_outlier | 2 | 427 |
| largeblock_medium93_short_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 2684 |
| largeblock_medium93_short_stable | gm_medium_qrslow_hardneg | 1 | 3172 |
| largeblock_medium93_short_stable | gm_visible_qrs_medium_detail | 1 | 2196 |
| largeblock_medium93_short_stable | nonbad_irregular_spike_hardneg | 0 | 1046 |
| largeblock_medium93_short_stable | nonbad_irregular_spike_hardneg | 1 | 2614 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_largeblock_aug_expert_metrics_valstress.csv`
- Manifest CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_largeblock_aug_manifest_valstress.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_largeblock_aug_expert_summary_valstress.json`
