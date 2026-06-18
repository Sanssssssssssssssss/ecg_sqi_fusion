# Waveform Predicted-Axis Student

Inference input is waveform-derived statistics only.  The true 47-column SQI/geometry sidecar is used only as a synthetic training teacher for the axis predictor, never as original input.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_stats_axis_interact_predaxis_balanced | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | synthetic_val | 0.993019 | 0.991965 | 0.995086 | 0.995265 | 0.980469 | 2 | 5 | 0 | 5 |
| core_stats_axis_interact_predaxis_balanced_expertonly | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | synthetic_test | 0.992312 | 0.990994 | 0.989540 | 0.995942 | 0.979253 | 5 | 5 | 0 | 5 |
| core_stats_axis_interact_predaxis_balanced_expertonly | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | original_test_all_10s+ | 0.810546 | 0.662573 | 0.882692 | 0.807501 | 0.204380 | 427 | 852 | 0 | 71 |
| core_stats_axis_interact_predaxis_balanced_expertonly | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | original_all_10s+ | 0.834082 | 0.858331 | 0.783665 | 0.869496 | 0.925449 | 3687 | 1387 | 0 | 134 |
| core_stats_axis_interact_predaxis_balanced_expertonly | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | bad_core_nearboundary | 0.705882 | 0.275862 | 0.000000 | 0.000000 | 0.705882 | 0 | 0 | 0 | 35 |
| core_stats_axis_interact_predaxis_balanced_expertonly | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 36 |
| core_stats_axis_interact_predaxis_balanced_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard | synthetic_val | 0.990692 | 0.989727 | 0.997543 | 0.990530 | 0.980469 | 1 | 10 | 0 | 5 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | synthetic_val | 0.990692 | 0.989727 | 0.997543 | 0.990530 | 0.980469 | 1 | 10 | 0 | 5 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | synthetic_val | 0.969750 | 0.965733 | 0.972973 | 0.962121 | 0.996094 | 11 | 19 | 0 | 1 |
| core_stats_axis_interact_predaxis_medium_guard | synthetic_test | 0.982060 | 0.981496 | 0.989540 | 0.979708 | 0.979253 | 5 | 25 | 0 | 5 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | synthetic_test | 0.982060 | 0.981496 | 0.989540 | 0.979708 | 0.979253 | 5 | 25 | 0 | 5 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | synthetic_test | 0.975397 | 0.975969 | 0.970711 | 0.974026 | 0.991701 | 14 | 31 | 0 | 2 |
| core_stats_axis_interact_predaxis_medium_guard | original_test_all_10s+ | 0.838858 | 0.688475 | 0.855495 | 0.882512 | 0.221411 | 526 | 520 | 0 | 80 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | original_test_all_10s+ | 0.839212 | 0.691815 | 0.855495 | 0.882512 | 0.228710 | 526 | 520 | 0 | 77 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | original_test_all_10s+ | 0.812670 | 0.684685 | 0.690934 | 0.962720 | 0.274939 | 1125 | 150 | 0 | 145 |
| core_stats_axis_interact_predaxis_medium_guard | original_all_10s+ | 0.848404 | 0.870187 | 0.794285 | 0.895653 | 0.927909 | 3506 | 1107 | 0 | 137 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | original_all_10s+ | 0.848556 | 0.870413 | 0.794285 | 0.895653 | 0.928855 | 3506 | 1107 | 0 | 132 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | original_all_10s+ | 0.770391 | 0.810964 | 0.612979 | 0.942887 | 0.931126 | 6596 | 592 | 0 | 208 |
| core_stats_axis_interact_predaxis_medium_guard | bad_core_nearboundary | 0.764706 | 0.288889 | 0.000000 | 0.000000 | 0.764706 | 0 | 0 | 0 | 28 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | bad_core_nearboundary | 0.789916 | 0.294210 | 0.000000 | 0.000000 | 0.789916 | 0 | 0 | 0 | 25 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | bad_core_nearboundary | 0.949580 | 0.324713 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 0 | 6 |
| core_stats_axis_interact_predaxis_medium_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 52 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 52 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 139 |
| core_stats_axis_interact_predaxis_bad_guard | synthetic_val | 0.994183 | 0.993853 | 0.995086 | 0.994318 | 0.992188 | 2 | 6 | 0 | 2 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | synthetic_val | 0.993601 | 0.992566 | 0.997543 | 0.992424 | 0.992188 | 1 | 5 | 0 | 2 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | synthetic_val | 0.949971 | 0.941357 | 0.972973 | 0.930871 | 0.992188 | 11 | 23 | 0 | 2 |
| core_stats_axis_interact_predaxis_bad_guard | synthetic_test | 0.995387 | 0.994956 | 0.993724 | 0.996753 | 0.991701 | 3 | 4 | 0 | 2 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | synthetic_test | 0.995387 | 0.994956 | 0.993724 | 0.996753 | 0.991701 | 3 | 4 | 0 | 2 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | synthetic_test | 0.979498 | 0.980227 | 0.985356 | 0.974026 | 0.995851 | 7 | 31 | 0 | 1 |
| core_stats_axis_interact_predaxis_bad_guard | original_test_all_10s+ | 0.824112 | 0.688126 | 0.879396 | 0.832354 | 0.245742 | 439 | 737 | 0 | 56 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | original_test_all_10s+ | 0.813613 | 0.678125 | 0.886813 | 0.806597 | 0.240876 | 412 | 846 | 0 | 53 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | original_test_all_10s+ | 0.840864 | 0.709580 | 0.767033 | 0.953005 | 0.287105 | 848 | 187 | 0 | 115 |
| core_stats_axis_interact_predaxis_bad_guard | original_all_10s+ | 0.826981 | 0.853681 | 0.757144 | 0.888408 | 0.928666 | 4139 | 1181 | 0 | 119 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | original_all_10s+ | 0.822521 | 0.849777 | 0.757261 | 0.874577 | 0.928288 | 4137 | 1323 | 0 | 116 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | original_all_10s+ | 0.806105 | 0.838298 | 0.689491 | 0.930467 | 0.932072 | 5292 | 710 | 0 | 179 |
| core_stats_axis_interact_predaxis_bad_guard | bad_core_nearboundary | 0.848739 | 0.306061 | 0.000000 | 0.000000 | 0.848739 | 0 | 0 | 0 | 18 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | bad_core_nearboundary | 0.831933 | 0.302752 | 0.000000 | 0.000000 | 0.831933 | 0 | 0 | 0 | 20 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 0 | 1 |
| core_stats_axis_interact_predaxis_bad_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 38 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 33 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 114 |

## Axis Recovery

| Axis set | Feature | Corr | MAE-z |
|---|---|---:|---:|
| core | `pca_margin` | 0.7527 | 0.3032 |
| core | `pc1` | 0.7199 | 0.4005 |
| core | `non_qrs_diff_p95` | 0.7139 | 0.4086 |
| core | `band_15_30` | 0.7129 | 0.3901 |
| core | `band_30_45` | 0.6903 | 0.4067 |
| core | `sqi_bSQI` | 0.6284 | 0.3711 |
| core | `flatline_ratio` | 0.4906 | 0.7622 |
| core | `template_corr` | 0.4650 | 0.6484 |
| core | `amplitude_entropy` | 0.4470 | 0.7178 |
| core | `sqi_sSQI` | 0.3975 | 0.8092 |
| core | `diff_abs_p95` | 0.3680 | 1.0654 |
| core | `pc3` | 0.3676 | 0.7706 |
| core | `baseline_step` | 0.3030 | 0.8284 |
| core | `sqi_basSQI` | 0.2007 | 0.8918 |
| core | `mean_abs` | 0.0482 | 0.9071 |
| core | `pc2` | 0.0152 | 1.0001 |

## Contract

- Axis teacher targets are synthetic train/val/test only.
- Original BUT is bucketed report-only and is never used for training, validation, feature selection, or threshold selection.
- This is a normal waveform-derived student path, not a rule artifact using original-side features.

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_predicted_axis_student_metrics_core_interact.csv`
- Axis recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_predicted_axis_student_recovery_core_interact.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_predicted_axis_student_summary_core_interact.json`
