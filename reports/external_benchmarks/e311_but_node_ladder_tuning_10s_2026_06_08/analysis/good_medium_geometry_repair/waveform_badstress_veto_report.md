# Waveform Bad-Stress Veto Evaluation

A strict bad-stress veto is selected on synthetic validation only and evaluated on original BUT buckets.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_expertmix_medium_guard_expertonly | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_expertmix_medium_guard_expertonly | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 529 | 369 | 0 | 75 |
| base_expertmix_medium_guard_expertonly | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_expertmix_medium_guard_expertonly | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 0 | 73 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_hard | synthetic_test | 0.988211 | 0.986495 | 0.983264 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_hard | original_test_all_10s+ | 0.810900 | 0.674564 | 0.783242 | 0.863534 | 0.489051 | 516 | 287 | 0 | 57 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_hard | bad_outlier_stress | 0.287671 | 0.148936 | 0.000000 | 0.000000 | 0.287671 | 0 | 0 | 0 | 55 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_hard | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_hard | original_test_all_10s+ | 0.794857 | 0.662883 | 0.762912 | 0.844329 | 0.545012 | 508 | 268 | 0 | 50 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_hard | bad_outlier_stress | 0.366438 | 0.178780 | 0.000000 | 0.000000 | 0.366438 | 0 | 0 | 0 | 48 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_hard | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_hard | original_test_all_10s+ | 0.794857 | 0.662883 | 0.762912 | 0.844329 | 0.545012 | 508 | 268 | 0 | 50 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_hard | bad_outlier_stress | 0.366438 | 0.178780 | 0.000000 | 0.000000 | 0.366438 | 0 | 0 | 0 | 48 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_soft | synthetic_test | 0.988211 | 0.986495 | 0.983264 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_soft | original_test_all_10s+ | 0.810900 | 0.674564 | 0.783242 | 0.863534 | 0.489051 | 516 | 287 | 0 | 57 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_strict_nonbad_soft | bad_outlier_stress | 0.287671 | 0.148936 | 0.000000 | 0.000000 | 0.287671 | 0 | 0 | 0 | 55 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_soft | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_soft | original_test_all_10s+ | 0.794857 | 0.662883 | 0.762912 | 0.844329 | 0.545012 | 508 | 268 | 0 | 50 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_balanced_guard_soft | bad_outlier_stress | 0.366438 | 0.178780 | 0.000000 | 0.000000 | 0.366438 | 0 | 0 | 0 | 48 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_soft | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_soft | original_test_all_10s+ | 0.794857 | 0.662883 | 0.762912 | 0.844329 | 0.545012 | 508 | 268 | 0 | 50 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_bad_balanced_bad_recovery_soft | bad_outlier_stress | 0.366438 | 0.178780 | 0.000000 | 0.000000 | 0.366438 | 0 | 0 | 0 | 48 |
| base_plus_stressaug_medium_heavy_strict_nonbad_hard | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_heavy_strict_nonbad_hard | original_test_all_10s+ | 0.798514 | 0.655610 | 0.778297 | 0.847040 | 0.454988 | 514 | 283 | 0 | 59 |
| base_plus_stressaug_medium_heavy_strict_nonbad_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_heavy_strict_nonbad_hard | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 0 | 57 |
| base_plus_stressaug_medium_heavy_balanced_guard_hard | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_heavy_balanced_guard_hard | original_test_all_10s+ | 0.798514 | 0.655610 | 0.778297 | 0.847040 | 0.454988 | 514 | 283 | 0 | 59 |
| base_plus_stressaug_medium_heavy_balanced_guard_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_heavy_balanced_guard_hard | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 0 | 57 |
| base_plus_stressaug_medium_heavy_bad_recovery_hard | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_heavy_bad_recovery_hard | original_test_all_10s+ | 0.798514 | 0.655610 | 0.778297 | 0.847040 | 0.454988 | 514 | 283 | 0 | 59 |
| base_plus_stressaug_medium_heavy_bad_recovery_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_heavy_bad_recovery_hard | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 0 | 57 |
| base_plus_stressaug_medium_heavy_strict_nonbad_soft | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_heavy_strict_nonbad_soft | original_test_all_10s+ | 0.798514 | 0.655610 | 0.778297 | 0.847040 | 0.454988 | 514 | 283 | 0 | 59 |
| base_plus_stressaug_medium_heavy_strict_nonbad_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_heavy_strict_nonbad_soft | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 0 | 57 |
| base_plus_stressaug_medium_heavy_balanced_guard_soft | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_heavy_balanced_guard_soft | original_test_all_10s+ | 0.798514 | 0.655610 | 0.778297 | 0.847040 | 0.454988 | 514 | 283 | 0 | 59 |
| base_plus_stressaug_medium_heavy_balanced_guard_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_heavy_balanced_guard_soft | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 0 | 57 |
| base_plus_stressaug_medium_heavy_bad_recovery_soft | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_medium_heavy_bad_recovery_soft | original_test_all_10s+ | 0.798514 | 0.655610 | 0.778297 | 0.847040 | 0.454988 | 514 | 283 | 0 | 59 |
| base_plus_stressaug_medium_heavy_bad_recovery_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_medium_heavy_bad_recovery_soft | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 0 | 57 |
| base_plus_stressaug_bad_heavy_strict_nonbad_hard | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_bad_heavy_strict_nonbad_hard | original_test_all_10s+ | 0.811490 | 0.679258 | 0.782143 | 0.862630 | 0.520681 | 518 | 274 | 0 | 56 |
| base_plus_stressaug_bad_heavy_strict_nonbad_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_bad_heavy_strict_nonbad_hard | bad_outlier_stress | 0.332192 | 0.166238 | 0.000000 | 0.000000 | 0.332192 | 0 | 0 | 0 | 54 |
| base_plus_stressaug_bad_heavy_balanced_guard_hard | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_bad_heavy_balanced_guard_hard | original_test_all_10s+ | 0.773859 | 0.647231 | 0.735714 | 0.821509 | 0.598540 | 500 | 238 | 0 | 49 |
| base_plus_stressaug_bad_heavy_balanced_guard_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_bad_heavy_balanced_guard_hard | bad_outlier_stress | 0.441781 | 0.204276 | 0.000000 | 0.000000 | 0.441781 | 0 | 0 | 0 | 47 |
| base_plus_stressaug_bad_heavy_bad_recovery_hard | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_bad_heavy_bad_recovery_hard | original_test_all_10s+ | 0.773859 | 0.647231 | 0.735714 | 0.821509 | 0.598540 | 500 | 238 | 0 | 49 |
| base_plus_stressaug_bad_heavy_bad_recovery_hard | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_bad_heavy_bad_recovery_hard | bad_outlier_stress | 0.441781 | 0.204276 | 0.000000 | 0.000000 | 0.441781 | 0 | 0 | 0 | 47 |
| base_plus_stressaug_bad_heavy_strict_nonbad_soft | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_bad_heavy_strict_nonbad_soft | original_test_all_10s+ | 0.811490 | 0.679258 | 0.782143 | 0.862630 | 0.520681 | 518 | 274 | 0 | 56 |
| base_plus_stressaug_bad_heavy_strict_nonbad_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_bad_heavy_strict_nonbad_soft | bad_outlier_stress | 0.332192 | 0.166238 | 0.000000 | 0.000000 | 0.332192 | 0 | 0 | 0 | 54 |
| base_plus_stressaug_bad_heavy_balanced_guard_soft | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_bad_heavy_balanced_guard_soft | original_test_all_10s+ | 0.773859 | 0.647231 | 0.735714 | 0.821509 | 0.598540 | 500 | 238 | 0 | 49 |
| base_plus_stressaug_bad_heavy_balanced_guard_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_bad_heavy_balanced_guard_soft | bad_outlier_stress | 0.441781 | 0.204276 | 0.000000 | 0.000000 | 0.441781 | 0 | 0 | 0 | 47 |
| base_plus_stressaug_bad_heavy_bad_recovery_soft | synthetic_test | 0.987699 | 0.985464 | 0.981172 | 0.989448 | 0.991701 | 8 | 8 | 0 | 2 |
| base_plus_stressaug_bad_heavy_bad_recovery_soft | original_test_all_10s+ | 0.773859 | 0.647231 | 0.735714 | 0.821509 | 0.598540 | 500 | 238 | 0 | 49 |
| base_plus_stressaug_bad_heavy_bad_recovery_soft | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_plus_stressaug_bad_heavy_bad_recovery_soft | bad_outlier_stress | 0.441781 | 0.204276 | 0.000000 | 0.000000 | 0.441781 | 0 | 0 | 0 | 47 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_badstress_veto_metrics.csv`
- Thresholds CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_badstress_veto_thresholds.csv`
