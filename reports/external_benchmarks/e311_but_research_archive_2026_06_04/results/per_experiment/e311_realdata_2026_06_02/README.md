# E3.11f External Real-Data Benchmarks

External validation for the Uformer SQI mainline. Raw data and processed NPZ files stay under ignored local directories.

## Datasets

- BUT QDB: expert consensus ECG quality labels, mapped `1/2/3 -> good/medium/bad`.
- CinC2017: `~` label is treated as noisy/unusable; `N/A/O` are usable.

## Current Artifacts

- `results/butqdb/frozen_probe_suite/bottleneck_only_linear_svm_report.json`
- `results/butqdb/frozen_probe_suite/bottleneck_only_logreg_report.json`
- `results/butqdb/frozen_probe_suite/bottleneck_only_small_mlp_report.json`
- `results/butqdb/frozen_probe_suite/full_tokens_linear_svm_report.json`
- `results/butqdb/frozen_probe_suite/full_tokens_logreg_report.json`
- `results/butqdb/frozen_probe_suite/full_tokens_small_mlp_report.json`
- `results/butqdb/frozen_probe_suite/handcrafted_sqi_linear_svm_report.json`
- `results/butqdb/frozen_probe_suite/handcrafted_sqi_logreg_report.json`
- `results/butqdb/frozen_probe_suite/handcrafted_sqi_small_mlp_report.json`
- `results/butqdb/frozen_probe_suite/probe_summary.json`
- `results/butqdb/frozen_probe_suite/residual_summary_only_linear_svm_report.json`
- `results/butqdb/frozen_probe_suite/residual_summary_only_logreg_report.json`
- `results/butqdb/frozen_probe_suite/residual_summary_only_small_mlp_report.json`
- `results/butqdb/frozen_probe_suite/summary_only_linear_svm_report.json`
- `results/butqdb/frozen_probe_suite/summary_only_logreg_report.json`
- `results/butqdb/frozen_probe_suite/summary_only_small_mlp_report.json`
- `results/butqdb/zero_shot_current_uformer/runtime_hardware.json`
- `results/butqdb/zero_shot_current_uformer/test_report.json`
- `results/butqdb/zero_shot_current_uformer/threshold_calibration.json`
- `results/cinc2017/frozen_probe_suite/bottleneck_only_linear_svm_report.json`
- `results/cinc2017/frozen_probe_suite/bottleneck_only_logreg_report.json`
- `results/cinc2017/frozen_probe_suite/bottleneck_only_small_mlp_report.json`
- `results/cinc2017/frozen_probe_suite/full_tokens_linear_svm_report.json`
- `results/cinc2017/frozen_probe_suite/full_tokens_logreg_report.json`
- `results/cinc2017/frozen_probe_suite/full_tokens_small_mlp_report.json`
- `results/cinc2017/frozen_probe_suite/handcrafted_sqi_linear_svm_report.json`
- `results/cinc2017/frozen_probe_suite/handcrafted_sqi_logreg_report.json`
- `results/cinc2017/frozen_probe_suite/handcrafted_sqi_small_mlp_report.json`
- `results/cinc2017/frozen_probe_suite/probe_summary.json`
- `results/cinc2017/frozen_probe_suite/residual_summary_only_linear_svm_report.json`
- `results/cinc2017/frozen_probe_suite/residual_summary_only_logreg_report.json`
- `results/cinc2017/frozen_probe_suite/residual_summary_only_small_mlp_report.json`
- `results/cinc2017/frozen_probe_suite/summary_only_linear_svm_report.json`
- `results/cinc2017/frozen_probe_suite/summary_only_logreg_report.json`
- `results/cinc2017/frozen_probe_suite/summary_only_small_mlp_report.json`
- `results/cinc2017/zero_shot_current_uformer/runtime_hardware.json`
- `results/cinc2017/zero_shot_current_uformer/test_report.json`
- `results/cinc2017/zero_shot_current_uformer/threshold_calibration.json`

## Notes

- Thresholds are calibrated only on validation splits.
- External data has no clean waveform, so denoise visuals are no-reference audits, not supervised denoise scores.
- `src/sqi_pipeline` is intentionally untouched.
