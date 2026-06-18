# Waveform Stress-Augmented Expert Mixture

Training adds PTB/synthetic-only medium low-confidence shell and controlled bad-stress blocks, then uses the waveform expert mixture. Original BUT is report-only.

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stressaug_medium_bad_balanced | synthetic_test | 0.988724 | 0.987653 | 0.991632 | 0.989448 | 0.979253 | 4 | 13 | 0 | 5 |
| stressaug_medium_bad_balanced_fullonly | synthetic_test | 0.988724 | 0.987653 | 0.991632 | 0.989448 | 0.979253 | 4 | 13 | 0 | 5 |
| stressaug_medium_bad_balanced_expertonly | synthetic_test | 0.990774 | 0.989934 | 0.989540 | 0.991071 | 0.991701 | 5 | 9 | 0 | 2 |
| stressaug_medium_bad_balanced | original_test_all_10s+ | 0.807715 | 0.551116 | 0.859890 | 0.839810 | 0.000000 | 510 | 709 | 0 | 159 |
| stressaug_medium_bad_balanced_fullonly | original_test_all_10s+ | 0.807715 | 0.551116 | 0.859890 | 0.839810 | 0.000000 | 510 | 709 | 0 | 159 |
| stressaug_medium_bad_balanced_expertonly | original_test_all_10s+ | 0.837914 | 0.689311 | 0.853571 | 0.881383 | 0.231144 | 533 | 512 | 0 | 84 |
| stressaug_medium_bad_balanced | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 119 |
| stressaug_medium_bad_balanced_fullonly | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 119 |
| stressaug_medium_bad_balanced_expertonly | bad_core_nearboundary | 0.798319 | 0.295950 | 0.000000 | 0.000000 | 0.798319 | 0 | 0 | 0 | 24 |
| stressaug_medium_bad_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 40 |
| stressaug_medium_bad_balanced_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 40 |
| stressaug_medium_bad_balanced_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 60 |
| stressaug_medium_heavy | synthetic_test | 0.994362 | 0.992926 | 0.991632 | 0.998377 | 0.979253 | 4 | 2 | 0 | 5 |
| stressaug_medium_heavy_fullonly | synthetic_test | 0.995387 | 0.994601 | 0.991632 | 0.998377 | 0.987552 | 4 | 2 | 0 | 3 |
| stressaug_medium_heavy_expertonly | synthetic_test | 0.995900 | 0.995434 | 0.991632 | 0.998377 | 0.991701 | 4 | 2 | 0 | 2 |
| stressaug_medium_heavy | original_test_all_10s+ | 0.797688 | 0.544643 | 0.872527 | 0.810212 | 0.000000 | 464 | 839 | 0 | 162 |
| stressaug_medium_heavy_fullonly | original_test_all_10s+ | 0.797806 | 0.546289 | 0.872527 | 0.810212 | 0.002433 | 464 | 839 | 0 | 161 |
| stressaug_medium_heavy_expertonly | original_test_all_10s+ | 0.834729 | 0.621650 | 0.841209 | 0.898554 | 0.090024 | 578 | 442 | 0 | 153 |
| stressaug_medium_heavy | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 119 |
| stressaug_medium_heavy_fullonly | bad_core_nearboundary | 0.008403 | 0.005556 | 0.000000 | 0.000000 | 0.008403 | 0 | 0 | 0 | 118 |
| stressaug_medium_heavy_expertonly | bad_core_nearboundary | 0.310924 | 0.158120 | 0.000000 | 0.000000 | 0.310924 | 0 | 0 | 0 | 82 |
| stressaug_medium_heavy | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 43 |
| stressaug_medium_heavy_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 43 |
| stressaug_medium_heavy_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 71 |
| stressaug_bad_heavy | synthetic_test | 0.989749 | 0.989318 | 0.989540 | 0.990260 | 0.987552 | 5 | 12 | 0 | 3 |
| stressaug_bad_heavy_fullonly | synthetic_test | 0.988724 | 0.988004 | 0.987448 | 0.989448 | 0.987552 | 6 | 12 | 0 | 3 |
| stressaug_bad_heavy_expertonly | synthetic_test | 0.989749 | 0.989314 | 0.987448 | 0.990260 | 0.991701 | 6 | 11 | 0 | 2 |
| stressaug_bad_heavy | original_test_all_10s+ | 0.821753 | 0.673756 | 0.835989 | 0.866245 | 0.216545 | 597 | 588 | 0 | 82 |
| stressaug_bad_heavy_fullonly | original_test_all_10s+ | 0.822697 | 0.690074 | 0.835440 | 0.864889 | 0.255474 | 599 | 592 | 0 | 66 |
| stressaug_bad_heavy_expertonly | original_test_all_10s+ | 0.826472 | 0.632507 | 0.836264 | 0.883642 | 0.124088 | 596 | 499 | 0 | 117 |
| stressaug_bad_heavy | bad_core_nearboundary | 0.747899 | 0.285256 | 0.000000 | 0.000000 | 0.747899 | 0 | 0 | 0 | 30 |
| stressaug_bad_heavy_fullonly | bad_core_nearboundary | 0.882353 | 0.312500 | 0.000000 | 0.000000 | 0.882353 | 0 | 0 | 0 | 14 |
| stressaug_bad_heavy_expertonly | bad_core_nearboundary | 0.428571 | 0.200000 | 0.000000 | 0.000000 | 0.428571 | 0 | 0 | 0 | 68 |
| stressaug_bad_heavy | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 52 |
| stressaug_bad_heavy_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 52 |
| stressaug_bad_heavy_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 49 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stress_aug_expert_mixture_metrics.csv`
- Augment manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stress_aug_manifest.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_stress_aug_expert_mixture_summary.json`
