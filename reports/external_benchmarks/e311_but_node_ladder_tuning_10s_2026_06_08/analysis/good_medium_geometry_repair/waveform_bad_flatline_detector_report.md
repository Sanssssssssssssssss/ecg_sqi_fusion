# Waveform Bad Flatline Detector

Targets flatline/baseline bad stress with synthetic positives and short-contact non-bad hard negatives. Threshold selection is synthetic-only.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_expertmix_medium_guard_expertonly | synthetic_test | 0.990261 | 0.989773 | 0.983264 | 0.992695 | 0.991701 | 8 | 8 | 0 | 2 |
| base_expertmix_medium_guard_expertonly | original_test_all_10s+ | 0.858794 | 0.727076 | 0.854670 | 0.915273 | 0.287105 | 529 | 369 | 0 | 75 |
| base_expertmix_medium_guard_expertonly | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| base_expertmix_medium_guard_expertonly | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 0 | 73 |
| flatbad_ultrastrict | synthetic_test | 0.976935 | 0.969275 | 0.983264 | 0.970779 | 0.995851 | 8 | 8 | 0 | 1 |
| flatbad_ultrastrict | original_test_all_10s+ | 0.796626 | 0.659063 | 0.824451 | 0.802756 | 0.484185 | 500 | 281 | 0 | 49 |
| flatbad_ultrastrict | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| flatbad_ultrastrict | bad_outlier_stress | 0.280822 | 0.146168 | 0.000000 | 0.000000 | 0.280822 | 0 | 0 | 0 | 47 |
| flatbad_strict | synthetic_test | 0.976935 | 0.969275 | 0.983264 | 0.970779 | 0.995851 | 8 | 8 | 0 | 1 |
| flatbad_strict | original_test_all_10s+ | 0.809602 | 0.682469 | 0.829670 | 0.816765 | 0.554745 | 506 | 263 | 0 | 46 |
| flatbad_strict | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| flatbad_strict | bad_outlier_stress | 0.380137 | 0.183623 | 0.000000 | 0.000000 | 0.380137 | 0 | 0 | 0 | 44 |
| flatbad_recall | synthetic_test | 0.990774 | 0.990603 | 0.983264 | 0.992695 | 0.995851 | 8 | 8 | 0 | 1 |
| flatbad_recall | original_test_all_10s+ | 0.809838 | 0.669791 | 0.837912 | 0.819702 | 0.454988 | 509 | 284 | 0 | 52 |
| flatbad_recall | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 0 | 2 |
| flatbad_recall | bad_outlier_stress | 0.239726 | 0.128913 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 0 | 50 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_flatline_detector_metrics.csv`
- Manifest CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_flatline_detector_manifest.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_flatline_detector_summary.json`
