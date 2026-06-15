# Fixed Waveform-Only Ensemble Report

No 47-feature inference input. Ensemble weights are fixed from model roles, not tuned on original BUT.

| candidate | bucket | acc | macro-F1 | good | medium | bad | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ens_medium_multiscale_equal | synthetic_test | 0.995387 | 0.993892 | 0.992 | 1.000 | 0.979 | 4 | 0 | 5 |
| ens_medium_multiscale_equal_badcal | synthetic_test | 0.995387 | 0.993892 | 0.992 | 1.000 | 0.979 | 4 | 0 | 5 |
| ens_medium_multiscale_equal | original_test_all_10s+ | 0.809013 | 0.694793 | 0.899 | 0.784 | 0.285 | 368 | 957 | 82 |
| ens_medium_multiscale_equal_badcal | original_test_all_10s+ | 0.809249 | 0.696818 | 0.899 | 0.784 | 0.290 | 368 | 957 | 80 |
| ens_medium_multiscale_equal | original_all_10s+ | 0.819942 | 0.848736 | 0.742 | 0.889 | 0.932 | 4391 | 1184 | 143 |
| ens_medium_multiscale_equal_badcal | original_all_10s+ | 0.819972 | 0.848766 | 0.742 | 0.888 | 0.933 | 4391 | 1184 | 140 |
| ens_medium_multiscale_equal | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000 | 0.000 | 0.983 | 0 | 0 | 2 |
| ens_medium_multiscale_equal_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| ens_medium_multiscale_equal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 80 |
| ens_medium_multiscale_equal_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 80 |
| ens_balanced_three_equal | synthetic_test | 0.995387 | 0.993894 | 0.994 | 0.999 | 0.979 | 3 | 1 | 5 |
| ens_balanced_three_equal_badcal | synthetic_test | 0.995387 | 0.993894 | 0.994 | 0.999 | 0.979 | 3 | 1 | 5 |
| ens_balanced_three_equal | original_test_all_10s+ | 0.794739 | 0.686972 | 0.905 | 0.751 | 0.290 | 347 | 1101 | 76 |
| ens_balanced_three_equal_badcal | original_test_all_10s+ | 0.794503 | 0.686312 | 0.905 | 0.751 | 0.290 | 347 | 1101 | 76 |
| ens_balanced_three_equal | original_all_10s+ | 0.829652 | 0.855624 | 0.773 | 0.869 | 0.933 | 3864 | 1392 | 139 |
| ens_balanced_three_equal_badcal | original_all_10s+ | 0.829561 | 0.855462 | 0.773 | 0.868 | 0.933 | 3864 | 1392 | 135 |
| ens_balanced_three_equal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| ens_balanced_three_equal_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| ens_balanced_three_equal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 76 |
| ens_balanced_three_equal_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 76 |
| ens_stress_guard_mix | synthetic_test | 0.995387 | 0.993892 | 0.992 | 1.000 | 0.979 | 4 | 0 | 5 |
| ens_stress_guard_mix_badcal | synthetic_test | 0.995387 | 0.993892 | 0.992 | 1.000 | 0.979 | 4 | 0 | 5 |
| ens_stress_guard_mix | original_test_all_10s+ | 0.806535 | 0.668382 | 0.886 | 0.795 | 0.224 | 414 | 907 | 107 |
| ens_stress_guard_mix_badcal | original_test_all_10s+ | 0.807833 | 0.680400 | 0.886 | 0.795 | 0.251 | 414 | 907 | 96 |
| ens_stress_guard_mix | original_all_10s+ | 0.804436 | 0.836464 | 0.709 | 0.897 | 0.926 | 4956 | 1099 | 171 |
| ens_stress_guard_mix_badcal | original_all_10s+ | 0.804709 | 0.836841 | 0.709 | 0.896 | 0.929 | 4956 | 1099 | 159 |
| ens_stress_guard_mix | bad_core_nearboundary | 0.773109 | 0.290679 | 0.000 | 0.000 | 0.773 | 0 | 0 | 27 |
| ens_stress_guard_mix_badcal | bad_core_nearboundary | 0.865546 | 0.309309 | 0.000 | 0.000 | 0.866 | 0 | 0 | 16 |
| ens_stress_guard_mix | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 80 |
| ens_stress_guard_mix_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 80 |
| ens_all_profiles_equal | synthetic_test | 0.995900 | 0.994378 | 0.994 | 1.000 | 0.979 | 3 | 0 | 5 |
| ens_all_profiles_equal_badcal | synthetic_test | 0.995900 | 0.994378 | 0.994 | 1.000 | 0.979 | 3 | 0 | 5 |
| ens_all_profiles_equal | original_test_all_10s+ | 0.793677 | 0.669888 | 0.897 | 0.760 | 0.248 | 376 | 1064 | 85 |
| ens_all_profiles_equal_badcal | original_test_all_10s+ | 0.795447 | 0.686675 | 0.897 | 0.759 | 0.290 | 376 | 1063 | 68 |
| ens_all_profiles_equal | original_all_10s+ | 0.809534 | 0.840259 | 0.729 | 0.880 | 0.929 | 4622 | 1279 | 150 |
| ens_all_profiles_equal_badcal | original_all_10s+ | 0.809959 | 0.840742 | 0.729 | 0.879 | 0.934 | 4622 | 1275 | 125 |
| ens_all_profiles_equal | bad_core_nearboundary | 0.857143 | 0.307692 | 0.000 | 0.000 | 0.857 | 0 | 0 | 17 |
| ens_all_profiles_equal_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000 | 0.000 | 1.000 | 0 | 0 | 0 |
| ens_all_profiles_equal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 68 |
| ens_all_profiles_equal_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 68 |

## Takeaway

A fixed waveform-only ensemble tests complementarity. If it improves original without feature inputs, the next architecture should be a learned multi-expert waveform Transformer rather than another single-head weight sweep.
