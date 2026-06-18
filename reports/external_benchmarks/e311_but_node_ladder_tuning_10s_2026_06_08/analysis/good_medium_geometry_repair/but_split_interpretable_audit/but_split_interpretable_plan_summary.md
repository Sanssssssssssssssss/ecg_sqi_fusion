# BUT Split Roll + Interpretable Waveform-SQI Transformer Summary

This is an external-only diagnostic summary. It does not modify `src/sqi_pipeline`, does not overwrite mainline checkpoints, and does not use BUT for PTB->BUT model selection.

## What Changed

- Added `run_but_split_interpretable_plan.py` under the external analysis area.
- Generated strict subject-level split candidates:
  - `current_split`: original baseline.
  - `balanced_best_split`: seed `4403`, subject-level, test `9070` rows with good/medium/bad `6296/2669/105`.
  - `hard_test_split`: seed `4819`, subject-level, test `8449` rows with good/medium/bad `3700/4457/292`, intentionally record111-like and hard.
  - `window_random_diagnostic_split`: leaky window-random upper bound only, not an external claim.
- Separated features into:
  - `stable_waveform_sqi_morph_rr`: 42 waveform-computable SQI/morphology/RR/frequency targets.
  - `weak_target_distribution_proxy`: `pc1`, `pc3`, `pca_margin`, diagnostic only.
  - `atlas_label_geometry_not_waveform_fact`: `pc2`, `pc4`, `boundary_confidence`, `region_confidence`, `knn_label_purity`, and centrality/margin ranks.

## Key Findings

- The current BUT split is genuinely shifted. Current test has major train-vs-test gaps in interpretable waveform features:
  - `sqi_basSQI`: train median `0.9716`, test median `0.7798`, KS `0.6453`.
  - `band_0p3_1`: train median `0.0129`, test median `0.2427`, KS `0.6570`.
  - `baseline_step`: train median `0.2852`, test median `0.9152`, KS `0.5494`.
  - `qrs_visibility`: train median `0.3779`, test median `0.1066`, KS `0.5273`.
  - `qrs_band_ratio`: train median `0.5524`, test median `0.3839`, KS `0.6344`.
- PTB synthetic vs BUT current test still has visible waveform-computable gaps:
  - `baseline_step`, `detector_agreement`, `qrs_band_ratio`, `qrs_visibility`, `diff_abs_p95`, `std/rms`, and `band_30_45`.
- PC interpretation:
  - `pc1` is mostly a detail/noise/high-frequency axis: strongest correlations are `non_qrs_diff_p95`, entropy/zero-crossing, `band_15_30`, inverse `sqi_pSQI`.
  - `pc2` is mostly a baseline/low-frequency axis in BUT: strongest correlations are `band_0p3_1`, inverse `sqi_basSQI`, `baseline_step`, `hjorth_complexity`, inverse `qrs_band_ratio`.
  - We should not train or claim on PC coordinates directly. We should train on the underlying waveform-computable SQI/frequency/QRS features.

## Capacity Diagnostics

BUT-only waveform Transformer capacity, initialized from the strongest normal waveform checkpoint:

| Split scheme | Test acc | Macro-F1 | Good recall | Medium recall | Bad recall | Meaning |
|---|---:|---:|---:|---:|---:|---|
| balanced strict subject split | 0.8404 | 0.8241 | 0.8963 | 0.7115 | 0.7619 | Better bad, but medium transfer still weak |
| hard strict subject split | 0.8467 | 0.6013 | 0.9095 | 0.8472 | 0.0445 | hard split exposes bad-stress/domain failure |
| window-random diagnostic | 0.9175 | 0.9228 | 0.9410 | 0.8453 | 0.9868 | Model capacity exists when distribution leakage removes subject shift |

Conclusion: the Transformer is not hopeless. It can exceed 0.90 in a leaky same-distribution diagnostic, but strict subject-level BUT generalization is still blocked by baseline/SQI/QRS distribution shift and bad-stress specificity.

## Next Research Direction

- Keep the final model waveform-only. No route/rule artifacts, no MLP/tree/47-feature classifier as the official result.
- Make the next Transformer explicitly learn waveform-computable targets:
  - mandatory: `sqi_basSQI`, `band_0p3_1`, `baseline_step`, `qrs_visibility`, `qrs_band_ratio`, `detector_agreement`, `flatline_ratio`;
  - secondary: `non_qrs_diff_p95`, `diff_abs_p95`, `band_15_30`, `band_30_45`, entropy/Hjorth/wavelet.
- Use PC axes only to explain gaps:
  - PC1 says detail/high-frequency shell;
  - PC2 says baseline/basSQI shell;
  - neither should be a final input or official training target.
- For the next model iteration, prioritize a baseline/low-frequency branch plus QRS/detector-agreement auxiliary heads, then evaluate on:
  - synthetic/node diagnostic;
  - current BUT report-only;
  - balanced strict split capacity;
  - hard strict split capacity;
  - window-random capacity only as upper-bound sanity.

## Artifacts

- Split report: `reports/.../but_split_interpretable_audit/but_split_roll_report.md`
- Feature report: `reports/.../but_split_interpretable_audit/interpretable_feature_distribution_report.md`
- Split assignments: `outputs/.../but_split_interpretable_audit/but_split_roll_row_assignments.csv`
- Feature gaps: `outputs/.../but_split_interpretable_audit/but_split_feature_ks.csv`
- PTB/BUT gaps: `outputs/.../but_split_interpretable_audit/ptb_synthetic_vs_but_feature_gap.csv`
- PC correlations: `outputs/.../but_split_interpretable_audit/pc_axis_waveform_feature_correlations.csv`
- Capacity reports:
  - `but_capacity_balanced_best_init6_report.md`
  - `but_capacity_hard_test_init6_report.md`
  - `but_capacity_window_random_init4_report.md`
