# Waveform Bad High-Specificity Detector

A bad-stress veto is trained with synthetic bad positives plus synthetic hard non-bad negatives. Thresholds are selected on synthetic augmented validation only.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_expertmix_medium_guard_expertonly | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_expertmix_medium_guard_expertonly | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 529 | 369 | 0 | 75 |
| base_expertmix_medium_guard_expertonly | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_expertmix_medium_guard_expertonly | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 0 | 73 |
| highspec_bad_strict | synthetic_test | 0.988724 | 0.986653 | 0.976987 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| highspec_bad_strict | original_test_all_10s+ | 0.769258 | 0.641948 | 0.731593 | 0.817668 | 0.581509 | 494 | 294 | 0 | 53 |
| highspec_bad_strict | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| highspec_bad_strict | bad_outlier_stress | 0.417808 | 0.196457 | 0.000000 | 0.000000 | 0.417808 | 0 | 0 | 0 | 51 |
| highspec_bad_balanced | synthetic_test | 0.989749 | 0.988731 | 0.981172 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| highspec_bad_balanced | original_test_all_10s+ | 0.790610 | 0.653105 | 0.747253 | 0.853366 | 0.498783 | 510 | 308 | 0 | 59 |
| highspec_bad_balanced | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| highspec_bad_balanced | bad_outlier_stress | 0.301370 | 0.154386 | 0.000000 | 0.000000 | 0.301370 | 0 | 0 | 0 | 57 |
| highspec_bad_recall | synthetic_test | 0.989236 | 0.987691 | 0.979079 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| highspec_bad_recall | original_test_all_10s+ | 0.759585 | 0.631719 | 0.703022 | 0.822865 | 0.579075 | 502 | 281 | 0 | 53 |
| highspec_bad_recall | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| highspec_bad_recall | bad_outlier_stress | 0.414384 | 0.195319 | 0.000000 | 0.000000 | 0.414384 | 0 | 0 | 0 | 51 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_highspec_detector_metrics.csv`
- Manifest CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_highspec_detector_manifest.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_highspec_detector_summary.json`
