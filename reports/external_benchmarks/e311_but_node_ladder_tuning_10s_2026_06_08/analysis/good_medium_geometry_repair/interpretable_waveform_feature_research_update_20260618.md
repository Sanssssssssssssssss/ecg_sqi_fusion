# Interpretable Waveform Feature Research Update - 2026-06-18

This is an external-only research note. It does not modify `src/sqi_pipeline`, does not use BUT for PTB->BUT selection, and does not promote route/rule/MLP/tree artifacts as the final model.

## Literature-Aligned Feature Groups

Recent ECG quality papers support organizing the model around waveform-computable physiology and signal quality facts, not dataset-neighborhood geometry.

- QRS/RR reliability: BUT-style classes are naturally tied to whether QRS and RR can be reliably extracted. The CinC wearable-quality paper defines class A as P/T/QRS visible, class B as QRS-reliable only, and class C as unusable.
- Detector robustness: Pan-Tompkins++ motivates explicit event timing, missing-beat search, post-peak false-beat suppression, and morphology checks around detected peaks.
- Baseline/contact/low-frequency quality: textile/wearable SQA work emphasizes template matching, physiological feasibility checks, and RR accuracy under sensor/contact conditions.
- Frequency/detail/artifact energy: systematic SQA reviews and wavelet-scattering work support band power, wavelet/detail, entropy/Hjorth, and transient artifact localization as interpretable signal-quality axes.

References used: CinC 2022 wearable ECG quality preprint, Pan-Tompkins++, automatic ECG quality assessment systematic review, wearable ECG wavelet scattering SQA, and textile-electrode SQA/RR-quality work.

## What The Current Data Audit Says

Current BUT split is shifted in stable waveform-computable features:

- `sqi_basSQI`: train median `0.9716`, test median `0.7798`, KS `0.6453`.
- `band_0p3_1`: train median `0.0129`, test median `0.2427`, KS `0.6570`.
- `baseline_step`: train median `0.2852`, test median `0.9152`, KS `0.5494`.
- `qrs_visibility`: train median `0.3779`, test median `0.1066`, KS `0.5273`.
- `qrs_band_ratio`: train median `0.5524`, test median `0.3839`, KS `0.6344`.

PC interpretation is useful but should stay explanatory:

- `pc1` mostly tracks detail/high-frequency complexity: `non_qrs_diff_p95`, entropy/zero-crossing, `band_15_30`, inverse `sqi_pSQI`.
- `pc2` mostly tracks baseline/low-frequency quality: `band_0p3_1`, inverse `sqi_basSQI`, `baseline_step`, inverse `qrs_band_ratio`.
- Dataset-neighborhood proxies (`knn_label_purity`, `region_confidence`, `boundary_confidence`) remain diagnostic-only, not official waveform facts.

## New Experiments Run

Two small experiment families were added to `run_waveform_geometry_student.py` and run for 2 epochs from the `stattoken_v2_badstress` checkpoint.

### 1. StatToken + PhysioSQI Context

Candidates:

- `stattoken_v2_badstress_physioctx_guard_a050`
- `stattoken_v2_badstress_physioctx_recall_a050`

Best report-only original_test_all_10s+:

- guard: acc `0.8350`, good/medium/bad `0.790/0.923/0.285`.
- recall: acc `0.8361`, good/medium/bad `0.811/0.908/0.287`.

Result: not a breakthrough. It preserves the current waveform frontier but does not recover bad outlier stress. original_all good collapses badly, so this should not be long-trained as-is.

### 2. StatToken + Differentiable Primitive Thresholds

Candidates:

- `stattoken_v2_badstress_threshold_guard_a050`
- `stattoken_v2_badstress_threshold_recall_a050`

Best report-only original_test_all_10s+:

- guard badcal: acc `0.8173`, good/medium/bad `0.729/0.940/0.277`.
- recall badcal: acc `0.8356`, good/medium/bad `0.799/0.916/0.290`.

Result: threshold capacity alone is not the blocker. It still misses bad_outlier_stress completely.

## Hard Feature Recovery Status

The model can learn:

- `pc1`, `pca_margin`, `non_qrs_diff_p95`, `sqi_bSQI`, `sqi_pSQI`, flatline/detail axes.

Still weak:

- `qrs_visibility`, `detector_agreement`, `baseline_step`, `sqi_basSQI`, and `pc2`/baseline shell.

This matches the literature: these hard features require reliable event timing, baseline/contact tracking, and QRS morphology consistency, not just global patch embeddings or late feature fusion.

## Current Decision

Do not keep sweeping late PhysioSQI fusion or threshold heads. They improve synthetic/node metrics but do not close the real BUT gap.

The next useful experiment should change the data/teacher alignment, not just the head:

- Build a PTB synthetic stress block that explicitly matches BUT `record111` bad_outlier morphology at waveform level.
- Recompute/validate QRS/RR/baseline targets from waveform facts, especially `sqi_basSQI`, `baseline_step`, detector agreement, and QRS visibility.
- Train a waveform-only Transformer with these targets as auxiliary supervision, but evaluate whether each feature is actually recovered before trusting class accuracy.
- Keep PC axes as interpretable audit coordinates only; avoid KNN/region/boundary proxies as official model features.

## Key Artifacts

- Feature audit: `reports/.../but_split_interpretable_audit/interpretable_feature_distribution_report.md`
- Split/capacity summary: `reports/.../but_split_interpretable_audit/but_split_interpretable_plan_summary.md`
- Latest waveform metrics: `outputs/.../waveform_geometry_student_search_metrics.csv`
- Latest runner: `outputs/.../run_waveform_geometry_student.py`
