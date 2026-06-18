# Predicted-Axis Augmented Bad Cascade

Base model is the best waveform expert mixture.  The augmented predicted-axis model is used only as a bad-stress veto.  Thresholds are ranked by synthetic validation; original BUT is report-only.

## Synthetic-Selected Top 20

| Candidate | Thr | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | Veto n |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.25 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.45 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.70 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m1.01 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.25 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.45 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.70 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m1.01 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.25 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.45 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.70 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m1.01 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.25 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.45 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.70 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m1.01 | 0.980 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.25 | 0.929 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 18 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.45 | 0.929 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 24 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.70 | 0.929 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 25 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m1.01 | 0.929 | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 25 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.45 | 0.929 | bad_outlier_stress | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 24 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.70 | 0.929 | bad_outlier_stress | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 25 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m1.01 | 0.929 | bad_outlier_stress | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 25 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.45 | 0.980 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.70 | 0.980 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m1.01 | 0.980 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 2 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.25 | 0.929 | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 18 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.25 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.25 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.45 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.70 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m1.01 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.25 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.45 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.70 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m1.01 | 0.980 | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.25 | 0.980 | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.45 | 0.980 | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.70 | 0.980 | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m1.01 | 0.980 | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.45 | 0.980 | original_test_all_10s+ | 0.859030 | 0.729089 | 0.854670 | 0.915273 | 0.291971 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.70 | 0.980 | original_test_all_10s+ | 0.859030 | 0.729089 | 0.854670 | 0.915273 | 0.291971 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m1.01 | 0.980 | original_test_all_10s+ | 0.859030 | 0.729089 | 0.854670 | 0.915273 | 0.291971 | 2 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.25 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.25 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.45 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.70 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m1.01 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.25 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.45 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.70 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m1.01 | 0.980 | original_test_all_10s+ | 0.858912 | 0.728084 | 0.854670 | 0.915273 | 0.289538 | 1 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.25 | 0.980 | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.45 | 0.980 | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.70 | 0.980 | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m1.01 | 0.980 | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.25 | 0.929 | original_test_all_10s+ | 0.857261 | 0.724030 | 0.850549 | 0.915273 | 0.291971 | 18 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.45 | 0.929 | original_test_all_10s+ | 0.857025 | 0.723848 | 0.849725 | 0.915273 | 0.294404 | 24 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.70 | 0.929 | original_test_all_10s+ | 0.857025 | 0.723622 | 0.849725 | 0.915273 | 0.294404 | 25 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m1.01 | 0.929 | original_test_all_10s+ | 0.857025 | 0.723622 | 0.849725 | 0.915273 | 0.294404 | 25 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.25 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.45 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.70 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m1.01 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.25 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.45 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.70 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m1.01 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.25 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.45 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.70 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m1.01 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.25 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.45 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.70 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m1.01 | 0.980 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.25 | 0.929 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.45 | 0.929 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.70 | 0.929 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m1.01 | 0.929 | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.25 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.45 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m0.70 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.45_m1.01 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.25 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.45 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m0.70 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.60_m1.01 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.25 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.45 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m0.70 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.75_m1.01 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.25 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.45 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m0.70 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_cascade_g0.85_m1.01 | 0.980 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.25 | 0.929 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.45 | 0.929 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m0.70 | 0.929 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_cascade_g0.45_m1.01 | 0.929 | synthetic_val | 0.976731 | 0.972100 | 0.982801 | 0.973485 | 0.980469 | 0 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\predaxis_aug_bad_cascade_metrics.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\predaxis_aug_bad_cascade_summary.json`
