# Waveform Bad Detector Threshold Sweep

Threshold candidates are synthetic augmented validation-derived or fixed ultra-strict values. Original BUT remains report-only.

| Candidate | Thr | Bucket | Acc | Good R | Medium R | Bad R | g->b | m->b | Bad out b->m |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| flatbad_ultrastrict | ckpt_selected | original_test_all_10s+ | 0.796626 | 0.824451 | 0.802756 | 0.484185 | 0 | 0 | 49 |
| flatbad_ultrastrict | ckpt_selected | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_ultrastrict | ckpt_selected | bad_outlier_stress | 0.280822 | 0.000000 | 0.000000 | 0.280822 | 0 | 0 | 47 |
| flatbad_ultrastrict | nonbad_max_plus | original_test_all_10s+ | 0.807597 | 0.834615 | 0.819928 | 0.435523 | 0 | 0 | 56 |
| flatbad_ultrastrict | nonbad_max_plus | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_ultrastrict | nonbad_max_plus | bad_outlier_stress | 0.212329 | 0.000000 | 0.000000 | 0.212329 | 0 | 0 | 54 |
| flatbad_ultrastrict | nonbad_q999 | original_test_all_10s+ | 0.795800 | 0.824176 | 0.801401 | 0.484185 | 0 | 0 | 49 |
| flatbad_ultrastrict | nonbad_q999 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_ultrastrict | nonbad_q999 | bad_outlier_stress | 0.280822 | 0.000000 | 0.000000 | 0.280822 | 0 | 0 | 47 |
| flatbad_ultrastrict | nonbad_q9995 | original_test_all_10s+ | 0.797570 | 0.825275 | 0.804338 | 0.479319 | 0 | 0 | 49 |
| flatbad_ultrastrict | nonbad_q9995 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_ultrastrict | nonbad_q9995 | bad_outlier_stress | 0.273973 | 0.000000 | 0.000000 | 0.273973 | 0 | 0 | 47 |
| flatbad_ultrastrict | fixed_0p95 | original_test_all_10s+ | 0.853368 | 0.853297 | 0.903525 | 0.313869 | 0 | 0 | 72 |
| flatbad_ultrastrict | fixed_0p95 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_ultrastrict | fixed_0p95 | bad_outlier_stress | 0.041096 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 70 |
| flatbad_ultrastrict | fixed_0p98 | original_test_all_10s+ | 0.856907 | 0.853846 | 0.912110 | 0.289538 | 0 | 0 | 75 |
| flatbad_ultrastrict | fixed_0p98 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_ultrastrict | fixed_0p98 | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 73 |
| flatbad_ultrastrict | fixed_0p995 | original_test_all_10s+ | 0.858676 | 0.854670 | 0.915047 | 0.287105 | 0 | 0 | 75 |
| flatbad_ultrastrict | fixed_0p995 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_ultrastrict | fixed_0p995 | bad_outlier_stress | 0.003425 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 73 |
| flatbad_strict | ckpt_selected | original_test_all_10s+ | 0.809602 | 0.829670 | 0.816765 | 0.554745 | 0 | 0 | 46 |
| flatbad_strict | ckpt_selected | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict | ckpt_selected | bad_outlier_stress | 0.380137 | 0.000000 | 0.000000 | 0.380137 | 0 | 0 | 44 |
| flatbad_strict | nonbad_max_plus | original_test_all_10s+ | 0.825056 | 0.838187 | 0.846136 | 0.481752 | 0 | 0 | 56 |
| flatbad_strict | nonbad_max_plus | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict | nonbad_max_plus | bad_outlier_stress | 0.277397 | 0.000000 | 0.000000 | 0.277397 | 0 | 0 | 54 |
| flatbad_strict | nonbad_q999 | original_test_all_10s+ | 0.824466 | 0.838187 | 0.845007 | 0.481752 | 0 | 0 | 56 |
| flatbad_strict | nonbad_q999 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict | nonbad_q999 | bad_outlier_stress | 0.277397 | 0.000000 | 0.000000 | 0.277397 | 0 | 0 | 54 |
| flatbad_strict | nonbad_q9995 | original_test_all_10s+ | 0.824938 | 0.838187 | 0.845911 | 0.481752 | 0 | 0 | 56 |
| flatbad_strict | nonbad_q9995 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict | nonbad_q9995 | bad_outlier_stress | 0.277397 | 0.000000 | 0.000000 | 0.277397 | 0 | 0 | 54 |
| flatbad_strict | fixed_0p95 | original_test_all_10s+ | 0.854901 | 0.853297 | 0.906010 | 0.318735 | 0 | 0 | 72 |
| flatbad_strict | fixed_0p95 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict | fixed_0p95 | bad_outlier_stress | 0.047945 | 0.000000 | 0.000000 | 0.047945 | 0 | 0 | 70 |
| flatbad_strict | fixed_0p98 | original_test_all_10s+ | 0.857851 | 0.854396 | 0.913692 | 0.287105 | 0 | 0 | 75 |
| flatbad_strict | fixed_0p98 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict | fixed_0p98 | bad_outlier_stress | 0.003425 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 73 |
| flatbad_strict | fixed_0p995 | original_test_all_10s+ | 0.858794 | 0.854670 | 0.915273 | 0.287105 | 0 | 0 | 75 |
| flatbad_strict | fixed_0p995 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_strict | fixed_0p995 | bad_outlier_stress | 0.003425 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 73 |
| flatbad_recall | ckpt_selected | original_test_all_10s+ | 0.809838 | 0.837912 | 0.819702 | 0.454988 | 0 | 0 | 52 |
| flatbad_recall | ckpt_selected | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_recall | ckpt_selected | bad_outlier_stress | 0.239726 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 50 |
| flatbad_recall | nonbad_max_plus | original_test_all_10s+ | 0.809367 | 0.837637 | 0.819024 | 0.454988 | 0 | 0 | 52 |
| flatbad_recall | nonbad_max_plus | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_recall | nonbad_max_plus | bad_outlier_stress | 0.239726 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 50 |
| flatbad_recall | nonbad_q999 | original_test_all_10s+ | 0.793913 | 0.827747 | 0.792815 | 0.506083 | 0 | 0 | 48 |
| flatbad_recall | nonbad_q999 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_recall | nonbad_q999 | bad_outlier_stress | 0.311644 | 0.000000 | 0.000000 | 0.311644 | 0 | 0 | 46 |
| flatbad_recall | nonbad_q9995 | original_test_all_10s+ | 0.796154 | 0.829670 | 0.796204 | 0.498783 | 0 | 0 | 48 |
| flatbad_recall | nonbad_q9995 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_recall | nonbad_q9995 | bad_outlier_stress | 0.301370 | 0.000000 | 0.000000 | 0.301370 | 0 | 0 | 46 |
| flatbad_recall | fixed_0p95 | original_test_all_10s+ | 0.841689 | 0.851648 | 0.878220 | 0.360097 | 0 | 0 | 64 |
| flatbad_recall | fixed_0p95 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_recall | fixed_0p95 | bad_outlier_stress | 0.106164 | 0.000000 | 0.000000 | 0.106164 | 0 | 0 | 62 |
| flatbad_recall | fixed_0p98 | original_test_all_10s+ | 0.851009 | 0.853297 | 0.898554 | 0.318735 | 0 | 0 | 71 |
| flatbad_recall | fixed_0p98 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_recall | fixed_0p98 | bad_outlier_stress | 0.047945 | 0.000000 | 0.000000 | 0.047945 | 0 | 0 | 69 |
| flatbad_recall | fixed_0p995 | original_test_all_10s+ | 0.857497 | 0.854121 | 0.912788 | 0.291971 | 0 | 0 | 75 |
| flatbad_recall | fixed_0p995 | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| flatbad_recall | fixed_0p995 | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 73 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_detector_threshold_sweep_metrics.csv`
- Thresholds CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_bad_detector_threshold_sweep_thresholds.csv`
