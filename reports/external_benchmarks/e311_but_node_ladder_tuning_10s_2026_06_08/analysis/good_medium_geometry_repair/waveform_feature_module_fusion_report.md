# Waveform Feature Module Fusion

Final inference is waveform-only: ECG waveform -> learned feature modules -> fusion MLP. SQI/geometry columns are training targets only.

## Key Result

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| modulefusion_balanced | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_balanced_badcal | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_balanced | original_test_all_10s+ | 0.742834 | 0.651134 | 0.903297 | 0.652960 | 0.289538 | 352 | 1536 | 40 |
| modulefusion_balanced_badcal | original_test_all_10s+ | 0.742834 | 0.651134 | 0.903297 | 0.652960 | 0.289538 | 352 | 1536 | 40 |
| modulefusion_balanced | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_balanced_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 40 |
| modulefusion_balanced_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 40 |
| modulefusion_medium_guard | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_medium_guard_badcal | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_medium_guard | original_test_all_10s+ | 0.743659 | 0.651786 | 0.896154 | 0.660416 | 0.289538 | 378 | 1503 | 38 |
| modulefusion_medium_guard_badcal | original_test_all_10s+ | 0.743659 | 0.651786 | 0.896154 | 0.660416 | 0.289538 | 378 | 1503 | 38 |
| modulefusion_medium_guard | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_medium_guard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_medium_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 38 |
| modulefusion_medium_guard_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 38 |
| modulefusion_bad_guard | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_bad_guard_badcal | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_bad_guard | original_test_all_10s+ | 0.740710 | 0.649640 | 0.903297 | 0.648893 | 0.289538 | 352 | 1554 | 40 |
| modulefusion_bad_guard_badcal | original_test_all_10s+ | 0.740710 | 0.649640 | 0.903297 | 0.648893 | 0.289538 | 352 | 1554 | 40 |
| modulefusion_bad_guard | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_bad_guard_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_bad_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 40 |
| modulefusion_bad_guard_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 40 |
| modulefusion_medium_bad | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_medium_bad_badcal | synthetic_test | 0.991287 | 0.990027 | 0.987448 | 0.995130 | 0.979253 | 6 | 6 | 5 |
| modulefusion_medium_bad | original_test_all_10s+ | 0.736227 | 0.646435 | 0.908516 | 0.636014 | 0.289538 | 333 | 1611 | 33 |
| modulefusion_medium_bad_badcal | original_test_all_10s+ | 0.736227 | 0.646435 | 0.908516 | 0.636014 | 0.289538 | 333 | 1611 | 33 |
| modulefusion_medium_bad | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_medium_bad_badcal | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| modulefusion_medium_bad | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 33 |
| modulefusion_medium_bad_badcal | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 33 |

## Module Recovery

| Module | Split | Mean MAE-z | Mean Corr |
|---|---|---:|---:|
| baseline_flatline | original_all_10s+ | 0.7062 | 0.4376 |
| baseline_flatline | synthetic_test | 0.7903 | 0.3077 |
| baseline_flatline | synthetic_val | 0.7238 | 0.4192 |
| detail_hf | original_all_10s+ | 0.3578 | 0.8468 |
| detail_hf | synthetic_test | 0.4993 | 0.6229 |
| detail_hf | synthetic_val | 0.4165 | 0.7612 |
| geometry_shell | original_all_10s+ | 0.6361 | 0.4535 |
| geometry_shell | synthetic_test | 0.6114 | 0.4707 |
| geometry_shell | synthetic_val | 0.6653 | 0.5229 |
| qrs_shape | original_all_10s+ | 0.6755 | 0.4974 |
| qrs_shape | synthetic_test | 0.7980 | 0.3120 |
| qrs_shape | synthetic_val | 0.6672 | 0.4906 |
| sqi_shape | original_all_10s+ | 0.5984 | 0.5294 |
| sqi_shape | synthetic_test | 0.7622 | 0.3706 |
| sqi_shape | synthetic_val | 0.6749 | 0.4289 |

## References

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls `0.956/0.973/0.927`. tabular feature input; upper-bound reference, not waveform-only.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls `0.973/0.885/0.696`. tabular feature input; target level for learned modules.
- `best_waveform_transformer_before_modules` original_test_all_10s+: acc `0.811018`, recalls `0.961/0.726/0.401`. aug_convtx_balanced_focal raw waveform-only baseline.

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_feature_module_fusion_metrics.csv`
- Module recovery CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_feature_module_recovery.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_feature_module_fusion_summary.json`

## Notes

- Selection uses synthetic train/val only.
- Original BUT is report-only and bucketed.
- A strong result requires both module recovery and original test transfer; high synthetic acc alone is not enough.
