# Waveform Feature Feasibility Audit

Purpose: explain why 47-feature/reduced MLP can beat waveform Transformers, and which targets are actually learnable from a single ECG waveform.

## Feature Source Split

- `label_neighborhood_geometry`: 4 features
  - pca_margin, boundary_confidence, region_confidence, knn_label_purity
- `target_distribution_pca_geometry`: 4 features
  - pc1, pc2, pc3, pc4
- `waveform_computable_primitive`: 39 features
  - qrs_visibility, qrs_band_ratio, qrs_prom_p90, template_corr, detector_agreement, baseline_step, flatline_ratio, fatal_or_score, contact_loss_win_ratio, non_qrs_rms_ratio, non_qrs_diff_p95, diff_abs_p95, band_15_30, band_30_45, rms, std, mean_abs, ptp_p99_p01, amplitude_entropy, low_amp_ratio, sqi_iSQI, sqi_bSQI, sqi_pSQI, sqi_sSQI, sqi_kSQI, sqi_fSQI, sqi_basSQI, hjorth_activity, hjorth_mobility, hjorth_complexity, zero_crossing_rate, diff_zero_crossing_rate, sample_entropy_proxy, higuchi_fd_proxy, wavelet_e0, wavelet_e1, wavelet_e2, wavelet_e3, wavelet_e4

## Result Table

| Model | Formal waveform-only? | Input | Original test acc | Good | Medium | Bad |
|---|---:|---|---:|---:|---:|---:|
| `47_feature_tabular_upper_bound` | False | all 47 features incl. target geometry | 0.963548 | 0.956 | 0.973 | 0.927 |
| `reduced_top14_balanced_mlp` | False | 14 features incl. pca_margin/region_confidence | 0.898431 | 0.892 | 0.897 | 0.973 |
| `waveform_proxy_logreg_l2` | True | 279 deterministic waveform stats only | 0.827415 | 0.921 | 0.800 | 0.299 |
| `p20_waveform_transformer` | True | waveform + teacher loss only | 0.822225 | 0.858 | 0.844 | 0.270 |
| `eventqrs_qrsheavy_transformer` | True | waveform event-QRS tokens + teacher loss | 0.819394 | 0.894 | 0.804 | 0.324 |
| `top14_teacher_transformer` | True | waveform, predicted top14 teacher fusion | 0.781998 | 0.853 | 0.768 | 0.311 |

## Interpretation

- The 47-feature and top14 MLP results are not pure waveform-inference results: they use target/PCA/label-neighborhood geometry such as `pca_margin`, `region_confidence`, `boundary_confidence`, and `knn_label_purity`.
- The best deterministic waveform-computable proxy upper bound is about `0.827` on original_test, close to the best waveform Transformer (`~0.822`). This means the current bottleneck is not just Transformer architecture; it is missing target-geometry information and BUT split/domain shift.
- For formal waveform-only claims, the realistic current frontier is around `0.82-0.83` unless we improve PTB synthetic domain randomization or find waveform-computable proxies that close the target-geometry gap.

## Next Research Target

- Stop trying to force `region_confidence/knn_label_purity` into the Transformer as if they were recoverable waveform properties.
- Build a cleaner waveform-computable proxy-token Transformer around features that are actually computable from the signal: band/detail, flatline, entropy, baseline, QRS prominence/visibility proxies.
- Treat top14/47-feature MLP as a diagnostic oracle for dataset geometry, not the target performance of a waveform-only model.