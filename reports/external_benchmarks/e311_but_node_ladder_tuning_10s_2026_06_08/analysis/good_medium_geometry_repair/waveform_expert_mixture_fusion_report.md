# Waveform Expert Mixture Fusion

Inference uses waveform-derived morphology/statistics only.  The experiment explicitly separates good/medium and bad-stress modules, then selects the fusion on synthetic validation only.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expertmix_balanced | synthetic_test | 0.984111 | 0.983286 | 0.976987 | 0.987825 | 0.979253 | 11 | 15 | 0 | 5 |
| expertmix_balanced_fullonly | synthetic_test | 0.984111 | 0.983297 | 0.976987 | 0.987013 | 0.983402 | 11 | 15 | 0 | 4 |
| expertmix_balanced_expertonly | synthetic_test | 0.964634 | 0.966493 | 0.981172 | 0.952922 | 0.991701 | 9 | 57 | 0 | 2 |
| expertmix_balanced | original_test_all_10s+ | 0.832960 | 0.706449 | 0.867582 | 0.856304 | 0.274939 | 482 | 635 | 0 | 62 |
| expertmix_balanced_fullonly | original_test_all_10s+ | 0.833550 | 0.711550 | 0.867582 | 0.856304 | 0.287105 | 482 | 635 | 0 | 57 |
| expertmix_balanced_expertonly | original_test_all_10s+ | 0.832370 | 0.580084 | 0.836813 | 0.904202 | 0.019465 | 594 | 410 | 0 | 235 |
| expertmix_balanced | bad_core_nearboundary | 0.949580 | 0.324713 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 0 | 6 |
| expertmix_balanced_fullonly | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 0 | 1 |
| expertmix_balanced_expertonly | bad_core_nearboundary | 0.067227 | 0.041995 | 0.000000 | 0.000000 | 0.067227 | 0 | 0 | 0 | 111 |
| expertmix_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 56 |
| expertmix_balanced_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 56 |
| expertmix_balanced_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 124 |
| expertmix_medium_guard | synthetic_test | 0.987186 | 0.986204 | 0.987448 | 0.988636 | 0.979253 | 6 | 14 | 0 | 5 |
| expertmix_medium_guard_fullonly | synthetic_test | 0.987699 | 0.987043 | 0.987448 | 0.988636 | 0.983402 | 6 | 14 | 0 | 4 |
| expertmix_medium_guard_expertonly | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| expertmix_medium_guard | original_test_all_10s+ | 0.841453 | 0.705331 | 0.850824 | 0.887935 | 0.257908 | 543 | 496 | 0 | 83 |
| expertmix_medium_guard_fullonly | original_test_all_10s+ | 0.842515 | 0.715393 | 0.850824 | 0.887709 | 0.282238 | 543 | 496 | 0 | 73 |
| expertmix_medium_guard_expertonly | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 529 | 369 | 0 | 75 |
| expertmix_medium_guard | bad_core_nearboundary | 0.890756 | 0.314074 | 0.000000 | 0.000000 | 0.890756 | 0 | 0 | 0 | 13 |
| expertmix_medium_guard_fullonly | bad_core_nearboundary | 0.974790 | 0.329078 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 0 | 3 |
| expertmix_medium_guard_expertonly | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| expertmix_medium_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 70 |
| expertmix_medium_guard_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 70 |
| expertmix_medium_guard_expertonly | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 0 | 73 |
| expertmix_bad_stress_guard | synthetic_test | 0.980010 | 0.979527 | 0.979079 | 0.980519 | 0.979253 | 10 | 24 | 0 | 5 |
| expertmix_bad_stress_guard_fullonly | synthetic_test | 0.980523 | 0.980365 | 0.979079 | 0.980519 | 0.983402 | 10 | 24 | 0 | 4 |
| expertmix_bad_stress_guard_expertonly | synthetic_test | 0.981548 | 0.981733 | 0.985356 | 0.978084 | 0.991701 | 7 | 26 | 0 | 2 |
| expertmix_bad_stress_guard | original_test_all_10s+ | 0.812316 | 0.603905 | 0.883516 | 0.821509 | 0.082725 | 424 | 790 | 0 | 129 |
| expertmix_bad_stress_guard_fullonly | original_test_all_10s+ | 0.816916 | 0.654799 | 0.883516 | 0.821509 | 0.177616 | 424 | 790 | 0 | 90 |
| expertmix_bad_stress_guard_expertonly | original_test_all_10s+ | 0.841925 | 0.647628 | 0.815385 | 0.929507 | 0.133820 | 672 | 299 | 0 | 143 |
| expertmix_bad_stress_guard | bad_core_nearboundary | 0.285714 | 0.148148 | 0.000000 | 0.000000 | 0.285714 | 0 | 0 | 0 | 85 |
| expertmix_bad_stress_guard_fullonly | bad_core_nearboundary | 0.613445 | 0.253472 | 0.000000 | 0.000000 | 0.613445 | 0 | 0 | 0 | 46 |
| expertmix_bad_stress_guard_expertonly | bad_core_nearboundary | 0.462185 | 0.210728 | 0.000000 | 0.000000 | 0.462185 | 0 | 0 | 0 | 64 |
| expertmix_bad_stress_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 44 |
| expertmix_bad_stress_guard_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 44 |
| expertmix_bad_stress_guard_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 79 |

## Selection Contract

- Synthetic train/val only for training and mixture selection.
- Original BUT is bucketed report-only.
- No sidecar 47 SQI/geometry columns are used as inference input.

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_expert_mixture_fusion_metrics.csv`
- Feature schema: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_expert_mixture_feature_schema.json`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_expert_mixture_fusion_summary.json`
