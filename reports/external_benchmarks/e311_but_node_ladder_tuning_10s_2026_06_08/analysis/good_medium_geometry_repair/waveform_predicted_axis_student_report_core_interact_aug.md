# Waveform Predicted-Axis Student

Inference input is waveform-derived statistics only.  The true 47-column SQI/geometry sidecar is used only as a synthetic training teacher for the axis predictor, never as original input.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_stats_axis_interact_predaxis_balanced | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | synthetic_val | 0.977894 | 0.970943 | 0.990172 | 0.972538 | 0.980469 | 1 | 6 | 0 | 5 |
| core_stats_axis_interact_predaxis_balanced_expertonly | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | synthetic_test | 0.975397 | 0.961444 | 0.964435 | 0.978084 | 0.983402 | 2 | 6 | 0 | 4 |
| core_stats_axis_interact_predaxis_balanced_expertonly | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | original_test_all_10s+ | 0.644568 | 0.542346 | 0.580769 | 0.708992 | 0.515815 | 540 | 101 | 0 | 111 |
| core_stats_axis_interact_predaxis_balanced_expertonly | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | original_all_10s+ | 0.738045 | 0.731129 | 0.637857 | 0.808525 | 0.919395 | 4005 | 473 | 0 | 336 |
| core_stats_axis_interact_predaxis_balanced_expertonly | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | bad_core_nearboundary | 0.226891 | 0.123288 | 0.000000 | 0.000000 | 0.226891 | 0 | 0 | 0 | 92 |
| core_stats_axis_interact_predaxis_balanced_expertonly | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_balanced_fullonly | bad_outlier_stress | 0.633562 | 0.258560 | 0.000000 | 0.000000 | 0.633562 | 0 | 0 | 0 | 19 |
| core_stats_axis_interact_predaxis_balanced_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | synthetic_val | 0.971495 | 0.962040 | 0.980344 | 0.965909 | 0.980469 | 2 | 5 | 0 | 5 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | synthetic_test | 0.974885 | 0.959281 | 0.956067 | 0.980519 | 0.983402 | 3 | 2 | 0 | 4 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | original_test_all_10s+ | 0.613660 | 0.516037 | 0.487912 | 0.726390 | 0.513382 | 602 | 53 | 0 | 137 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | original_all_10s+ | 0.691437 | 0.674865 | 0.588570 | 0.825555 | 0.753453 | 4347 | 336 | 0 | 1239 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | bad_core_nearboundary | 0.016807 | 0.011019 | 0.000000 | 0.000000 | 0.016807 | 0 | 0 | 0 | 117 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_medium_guard_fullonly | bad_outlier_stress | 0.715753 | 0.278110 | 0.000000 | 0.000000 | 0.715753 | 0 | 0 | 0 | 20 |
| core_stats_axis_interact_predaxis_medium_guard_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | synthetic_val | 0.966841 | 0.955011 | 0.963145 | 0.965909 | 0.976562 | 4 | 4 | 0 | 6 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | synthetic_val | 0.236766 | 0.127626 | 1.000000 | 0.000000 | 0.000000 | 0 | 1056 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | synthetic_test | 0.977447 | 0.966834 | 0.968619 | 0.980519 | 0.979253 | 8 | 3 | 0 | 5 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | synthetic_test | 0.245003 | 0.131193 | 1.000000 | 0.000000 | 0.000000 | 0 | 1232 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | original_test_all_10s+ | 0.618615 | 0.520698 | 0.529670 | 0.702440 | 0.503650 | 581 | 129 | 0 | 82 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | original_test_all_10s+ | 0.429397 | 0.200270 | 1.000000 | 0.000000 | 0.000000 | 0 | 4426 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | original_all_10s+ | 0.724967 | 0.724056 | 0.596081 | 0.823956 | 0.941533 | 4871 | 351 | 0 | 185 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | original_all_10s+ | 0.517144 | 0.227245 | 1.000000 | 0.000000 | 0.000000 | 0 | 10628 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | bad_core_nearboundary | 0.436975 | 0.202729 | 0.000000 | 0.000000 | 0.436975 | 0 | 0 | 0 | 67 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |
| core_stats_axis_interact_predaxis_bad_guard_fullonly | bad_outlier_stress | 0.530822 | 0.231171 | 0.000000 | 0.000000 | 0.530822 | 0 | 0 | 0 | 15 |
| core_stats_axis_interact_predaxis_bad_guard_expertonly | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 |

## Axis Recovery

| Axis set | Feature | Corr | MAE-z |
|---|---|---:|---:|
| core | `pca_margin` | 0.7423 | 0.3392 |
| core | `pc1` | 0.7152 | 0.4168 |
| core | `non_qrs_diff_p95` | 0.7077 | 0.4332 |
| core | `band_15_30` | 0.7059 | 0.4042 |
| core | `band_30_45` | 0.6872 | 0.4217 |
| core | `sqi_bSQI` | 0.6128 | 0.4068 |
| core | `flatline_ratio` | 0.4940 | 0.7665 |
| core | `template_corr` | 0.4487 | 0.6286 |
| core | `amplitude_entropy` | 0.4214 | 0.7337 |
| core | `diff_abs_p95` | 0.3834 | 1.0571 |
| core | `sqi_sSQI` | 0.3771 | 0.8253 |
| core | `pc3` | 0.3496 | 0.7688 |
| core | `baseline_step` | 0.3105 | 0.8188 |
| core | `sqi_basSQI` | 0.2071 | 0.8884 |
| core | `pc2` | 0.0821 | 0.9796 |
| core | `mean_abs` | 0.0415 | 0.8985 |

## Contract

- Axis teacher targets are synthetic train/val/test only.
- Original BUT is bucketed report-only and is never used for training, validation, feature selection, or threshold selection.
- This is a normal waveform-derived student path, not a rule artifact using original-side features.

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_predicted_axis_student_metrics_core_interact_aug.csv`
- Axis recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_predicted_axis_student_recovery_core_interact_aug.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_predicted_axis_student_summary_core_interact_aug.json`
