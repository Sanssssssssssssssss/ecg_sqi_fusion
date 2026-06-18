# Waveform Large-Block Augmentation Expert

This run converts broad boundary diagnostics into PTB/synthetic-only waveform augmentation blocks. Original BUT remains report-only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| largeblock_medium93_short_stable | synthetic_test | 0.981035 | 0.980486 | 0.981172 | 0.981331 | 0.979253 | 9 | 23 | 0 | 5 |
| largeblock_medium93_short_stable_fullonly | synthetic_test | 0.980523 | 0.979977 | 0.976987 | 0.982143 | 0.979253 | 11 | 22 | 0 | 5 |
| largeblock_medium93_short_stable_expertonly | synthetic_test | 0.984623 | 0.983877 | 0.993724 | 0.982143 | 0.979253 | 3 | 22 | 0 | 5 |
| largeblock_medium93_short_stable | original_test_all_10s+ | 0.775510 | 0.641876 | 0.902747 | 0.715545 | 0.294404 | 352 | 1072 | 0 | 65 |
| largeblock_medium93_short_stable_fullonly | original_test_all_10s+ | 0.764657 | 0.604582 | 0.901648 | 0.704474 | 0.199513 | 357 | 1121 | 0 | 99 |
| largeblock_medium93_short_stable_expertonly | original_test_all_10s+ | 0.796036 | 0.665072 | 0.904670 | 0.758925 | 0.233577 | 347 | 1065 | 0 | 87 |
| largeblock_medium93_short_stable | original_all_10s+ | 0.808047 | 0.834058 | 0.751159 | 0.835341 | 0.936613 | 4239 | 1543 | 0 | 108 |
| largeblock_medium93_short_stable_fullonly | original_all_10s+ | 0.807592 | 0.832506 | 0.757672 | 0.828284 | 0.926963 | 4129 | 1622 | 0 | 154 |
| largeblock_medium93_short_stable_expertonly | original_all_10s+ | 0.804497 | 0.836051 | 0.732031 | 0.856887 | 0.932829 | 4567 | 1511 | 0 | 127 |
| largeblock_medium93_short_stable | bad_core_nearboundary | 0.848739 | 0.306061 | 0.000000 | 0.000000 | 0.848739 | 0 | 0 | 0 | 18 |
| largeblock_medium93_short_stable_fullonly | bad_core_nearboundary | 0.546218 | 0.235507 | 0.000000 | 0.000000 | 0.546218 | 0 | 0 | 0 | 54 |
| largeblock_medium93_short_stable_expertonly | bad_core_nearboundary | 0.789916 | 0.294210 | 0.000000 | 0.000000 | 0.789916 | 0 | 0 | 0 | 25 |
| largeblock_medium93_short_stable | bad_outlier_stress | 0.068493 | 0.042735 | 0.000000 | 0.000000 | 0.068493 | 0 | 0 | 0 | 47 |
| largeblock_medium93_short_stable_fullonly | bad_outlier_stress | 0.058219 | 0.036677 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 0 | 45 |
| largeblock_medium93_short_stable_expertonly | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 0 | 62 |
| largeblock_gm_slim_short_stable | synthetic_test | 0.984111 | 0.983407 | 0.993724 | 0.981331 | 0.979253 | 3 | 23 | 0 | 5 |
| largeblock_gm_slim_short_stable_fullonly | synthetic_test | 0.985136 | 0.984251 | 0.979079 | 0.988636 | 0.979253 | 10 | 14 | 0 | 5 |
| largeblock_gm_slim_short_stable_expertonly | synthetic_test | 0.984623 | 0.983877 | 0.993724 | 0.982143 | 0.979253 | 3 | 22 | 0 | 5 |
| largeblock_gm_slim_short_stable | original_test_all_10s+ | 0.812198 | 0.696997 | 0.904121 | 0.785585 | 0.284672 | 349 | 949 | 0 | 66 |
| largeblock_gm_slim_short_stable_fullonly | original_test_all_10s+ | 0.788369 | 0.659643 | 0.908791 | 0.740850 | 0.233577 | 332 | 1144 | 0 | 80 |
| largeblock_gm_slim_short_stable_expertonly | original_test_all_10s+ | 0.815855 | 0.696648 | 0.903846 | 0.793493 | 0.277372 | 350 | 914 | 0 | 70 |
| largeblock_gm_slim_short_stable | original_all_10s+ | 0.812022 | 0.842414 | 0.737722 | 0.870437 | 0.934153 | 4470 | 1373 | 0 | 120 |
| largeblock_gm_slim_short_stable_fullonly | original_all_10s+ | 0.812781 | 0.841712 | 0.754621 | 0.848419 | 0.928666 | 4182 | 1605 | 0 | 141 |
| largeblock_gm_slim_short_stable_expertonly | original_all_10s+ | 0.811324 | 0.841826 | 0.734260 | 0.874671 | 0.932450 | 4529 | 1329 | 0 | 130 |
| largeblock_gm_slim_short_stable | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| largeblock_gm_slim_short_stable_fullonly | bad_core_nearboundary | 0.806723 | 0.297674 | 0.000000 | 0.000000 | 0.806723 | 0 | 0 | 0 | 23 |
| largeblock_gm_slim_short_stable_expertonly | bad_core_nearboundary | 0.957983 | 0.326180 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 0 | 5 |
| largeblock_gm_slim_short_stable | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 64 |
| largeblock_gm_slim_short_stable_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 57 |
| largeblock_gm_slim_short_stable_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 65 |
| largeblock_bad_cautious_short_stable | synthetic_test | 0.982573 | 0.981982 | 0.991632 | 0.979708 | 0.979253 | 4 | 25 | 0 | 5 |
| largeblock_bad_cautious_short_stable_fullonly | synthetic_test | 0.957970 | 0.959917 | 0.983264 | 0.943994 | 0.979253 | 8 | 69 | 0 | 5 |
| largeblock_bad_cautious_short_stable_expertonly | synthetic_test | 0.955407 | 0.941327 | 0.991632 | 0.935065 | 0.987552 | 4 | 23 | 0 | 3 |
| largeblock_bad_cautious_short_stable | original_test_all_10s+ | 0.837325 | 0.651191 | 0.904121 | 0.847040 | 0.141119 | 349 | 675 | 0 | 149 |
| largeblock_bad_cautious_short_stable_fullonly | original_test_all_10s+ | 0.846290 | 0.622685 | 0.860989 | 0.905784 | 0.075426 | 506 | 417 | 0 | 214 |
| largeblock_bad_cautious_short_stable_expertonly | original_test_all_10s+ | 0.773741 | 0.642292 | 0.900000 | 0.698373 | 0.467153 | 320 | 716 | 0 | 30 |
| largeblock_bad_cautious_short_stable | original_all_10s+ | 0.820882 | 0.848388 | 0.743120 | 0.894806 | 0.922990 | 4378 | 1110 | 0 | 203 |
| largeblock_bad_cautious_short_stable_fullonly | original_all_10s+ | 0.787080 | 0.794455 | 0.731913 | 0.915600 | 0.706528 | 4569 | 895 | 0 | 1383 |
| largeblock_bad_cautious_short_stable_expertonly | original_all_10s+ | 0.800218 | 0.818596 | 0.738133 | 0.824050 | 0.952507 | 4414 | 1170 | 0 | 62 |
| largeblock_bad_cautious_short_stable | bad_core_nearboundary | 0.478992 | 0.215909 | 0.000000 | 0.000000 | 0.478992 | 0 | 0 | 0 | 62 |
| largeblock_bad_cautious_short_stable_fullonly | bad_core_nearboundary | 0.260504 | 0.137778 | 0.000000 | 0.000000 | 0.260504 | 0 | 0 | 0 | 88 |
| largeblock_bad_cautious_short_stable_expertonly | bad_core_nearboundary | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 0 | 4 |
| largeblock_bad_cautious_short_stable | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 0 | 87 |
| largeblock_bad_cautious_short_stable_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 126 |
| largeblock_bad_cautious_short_stable_expertonly | bad_outlier_stress | 0.263699 | 0.139115 | 0.000000 | 0.000000 | 0.263699 | 0 | 0 | 0 | 26 |
| largeblock_bad_outlier_bridge_stable | synthetic_test | 0.981548 | 0.981027 | 0.989540 | 0.978896 | 0.979253 | 5 | 26 | 0 | 5 |
| largeblock_bad_outlier_bridge_stable_fullonly | synthetic_test | 0.962583 | 0.963122 | 0.979079 | 0.952922 | 0.979253 | 10 | 56 | 0 | 5 |
| largeblock_bad_outlier_bridge_stable_expertonly | synthetic_test | 0.941056 | 0.923041 | 0.991632 | 0.910714 | 0.995851 | 4 | 24 | 0 | 1 |
| largeblock_bad_outlier_bridge_stable | original_test_all_10s+ | 0.832016 | 0.712926 | 0.901099 | 0.825350 | 0.291971 | 360 | 772 | 0 | 74 |
| largeblock_bad_outlier_bridge_stable_fullonly | original_test_all_10s+ | 0.841807 | 0.726164 | 0.876923 | 0.858789 | 0.347932 | 442 | 575 | 0 | 96 |
| largeblock_bad_outlier_bridge_stable_expertonly | original_test_all_10s+ | 0.772679 | 0.639756 | 0.906319 | 0.697018 | 0.403893 | 319 | 879 | 0 | 33 |
| largeblock_bad_outlier_bridge_stable | original_all_10s+ | 0.813752 | 0.844130 | 0.732911 | 0.882386 | 0.936424 | 4552 | 1240 | 0 | 119 |
| largeblock_bad_outlier_bridge_stable_fullonly | original_all_10s+ | 0.806924 | 0.836995 | 0.717949 | 0.881633 | 0.943614 | 4800 | 1137 | 0 | 124 |
| largeblock_bad_outlier_bridge_stable_expertonly | original_all_10s+ | 0.795879 | 0.816611 | 0.734554 | 0.818216 | 0.948723 | 4497 | 1311 | 0 | 58 |
| largeblock_bad_outlier_bridge_stable | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| largeblock_bad_outlier_bridge_stable_fullonly | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 0 | 1 |
| largeblock_bad_outlier_bridge_stable_expertonly | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | 0 |
| largeblock_bad_outlier_bridge_stable | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 0 | 72 |
| largeblock_bad_outlier_bridge_stable_fullonly | bad_outlier_stress | 0.085616 | 0.052576 | 0.000000 | 0.000000 | 0.085616 | 0 | 0 | 0 | 95 |
| largeblock_bad_outlier_bridge_stable_expertonly | bad_outlier_stress | 0.160959 | 0.092429 | 0.000000 | 0.000000 | 0.160959 | 0 | 0 | 0 | 33 |

## Block Counts

| candidate | block | label | n |
| --- | --- | --- | --- |
| largeblock_bad_cautious_short_stable | bad_controlled_outlier | 2 | 900 |
| largeblock_bad_cautious_short_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 1600 |
| largeblock_bad_cautious_short_stable | gm_medium_qrslow_hardneg | 1 | 1900 |
| largeblock_bad_cautious_short_stable | gm_visible_qrs_medium_detail | 1 | 1300 |
| largeblock_bad_cautious_short_stable | nonbad_irregular_spike_hardneg | 0 | 1520 |
| largeblock_bad_cautious_short_stable | nonbad_irregular_spike_hardneg | 1 | 3680 |
| largeblock_bad_outlier_bridge_stable | bad_controlled_outlier | 2 | 1300 |
| largeblock_bad_outlier_bridge_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 1400 |
| largeblock_bad_outlier_bridge_stable | gm_medium_qrslow_hardneg | 1 | 1700 |
| largeblock_bad_outlier_bridge_stable | gm_visible_qrs_medium_detail | 1 | 1200 |
| largeblock_bad_outlier_bridge_stable | nonbad_irregular_spike_hardneg | 0 | 2111 |
| largeblock_bad_outlier_bridge_stable | nonbad_irregular_spike_hardneg | 1 | 5089 |
| largeblock_gm_slim_short_stable | bad_controlled_outlier | 2 | 220 |
| largeblock_gm_slim_short_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 1800 |
| largeblock_gm_slim_short_stable | gm_medium_qrslow_hardneg | 1 | 2200 |
| largeblock_gm_slim_short_stable | gm_visible_qrs_medium_detail | 1 | 1400 |
| largeblock_gm_slim_short_stable | nonbad_irregular_spike_hardneg | 0 | 673 |
| largeblock_gm_slim_short_stable | nonbad_irregular_spike_hardneg | 1 | 1727 |
| largeblock_medium93_short_stable | bad_controlled_outlier | 2 | 350 |
| largeblock_medium93_short_stable | gm_good_rescue_pc1flat_qrsvisible | 0 | 2200 |
| largeblock_medium93_short_stable | gm_medium_qrslow_hardneg | 1 | 2600 |
| largeblock_medium93_short_stable | gm_visible_qrs_medium_detail | 1 | 1800 |
| largeblock_medium93_short_stable | nonbad_irregular_spike_hardneg | 0 | 881 |
| largeblock_medium93_short_stable | nonbad_irregular_spike_hardneg | 1 | 2119 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_largeblock_aug_expert_metrics_stable.csv`
- Manifest CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_largeblock_aug_manifest_stable.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_largeblock_aug_expert_summary_stable.json`
