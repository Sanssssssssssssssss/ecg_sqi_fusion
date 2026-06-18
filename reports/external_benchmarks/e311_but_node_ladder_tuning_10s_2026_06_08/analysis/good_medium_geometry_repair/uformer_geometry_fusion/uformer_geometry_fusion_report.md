# UFormer + SQI/Geometry Late-Fusion Diagnostic

Experiment-only. The mainline UFormer and checkpoints were not modified.

- UFormer probability source: `nl_n17043_gm_trim_bad_boundaryblocks_teacherwall_goodsafe_5e30e8c689a6`
- Node target: `N17043_gm_trim_bad`
- Explicit feature count: 28
- Transfer-safe feature count: 28
- Node-full feature count: 47
- Transfer-safe feature columns: `pc1, pc2, pc3, pc4, qrs_visibility, qrs_band_ratio, qrs_prom_p90, template_corr, detector_agreement, baseline_step, flatline_ratio, fatal_or_score, contact_loss_win_ratio, non_qrs_rms_ratio, non_qrs_diff_p95, diff_abs_p95, band_15_30, band_30_45, rms, std, mean_abs, ptp_p99_p01, amplitude_entropy, low_amp_ratio, sqi_bSQI, sqi_sSQI, sqi_kSQI, sqi_basSQI`
- Node-full extra columns: `pca_margin, boundary_confidence, region_confidence, knn_label_purity, sqi_iSQI, sqi_pSQI, sqi_fSQI, hjorth_activity, hjorth_mobility, hjorth_complexity, zero_crossing_rate, diff_zero_crossing_rate, sample_entropy_proxy, higuchi_fd_proxy, wavelet_e0, wavelet_e1, wavelet_e2, wavelet_e3, wavelet_e4`

## Best Node-Test Fusion

- `nodefull_uformer_geometry_rf`: acc=0.980574, macro-F1=0.921254, good/medium/bad=0.989560/0.981699/0.663866

## Best Original Main-Bucket Fusion

- `geometry_only_logreg`: acc=0.868051, macro-F1=0.886290, good/medium/bad=0.935989/0.812020/0.873950

## Full Metrics

| fusion | scope | n | acc | macro-F1 | good | medium | bad |
|---|---|---:|---:|---:|---:|---:|---:|
| `geometry_only_logreg` | `node_all` | 31755 | 0.902976 | 0.921360 | 0.903538 | 0.866202 | 0.996327 |
| `geometry_only_logreg` | `node_test` | 8185 | 0.868051 | 0.886290 | 0.935989 | 0.812020 | 0.873950 |
| `geometry_only_logreg` | `node_trainval` | 23570 | 0.915104 | 0.923897 | 0.894725 | 0.904869 | 1.000000 |
| `geometry_only_logreg` | `original_bad_core_nearboundary` | 119 | 0.873950 | 0.310912 | 0.000000 | 0.000000 | 0.873950 |
| `geometry_only_logreg` | `original_bad_outlier_stress` | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `geometry_only_logreg` | `original_test_all_10s+` | 8477 | 0.838150 | 0.702030 | 0.935989 | 0.812020 | 0.253041 |
| `geometry_only_logreg` | `original_test_main_without_bad_stress` | 8185 | 0.868051 | 0.886290 | 0.935989 | 0.812020 | 0.873950 |
| `nodefull_geometry_logreg` | `node_all` | 31755 | 0.964289 | 0.967652 | 0.959455 | 0.969514 | 0.970862 |
| `nodefull_geometry_logreg` | `node_test` | 8185 | 0.944411 | 0.634343 | 0.938187 | 0.974921 | 0.000000 |
| `nodefull_geometry_logreg` | `node_trainval` | 23570 | 0.971192 | 0.973594 | 0.965232 | 0.965656 | 1.000000 |
| `nodefull_uformer_geometry_hgb` | `node_all` | 31755 | 0.992915 | 0.990517 | 0.999061 | 0.991532 | 0.970862 |
| `nodefull_uformer_geometry_hgb` | `node_test` | 8185 | 0.973122 | 0.653847 | 0.995879 | 0.980569 | 0.000000 |
| `nodefull_uformer_geometry_hgb` | `node_trainval` | 23570 | 0.999788 | 0.999803 | 0.999925 | 0.999355 | 1.000000 |
| `nodefull_uformer_geometry_logreg` | `node_all` | 31755 | 0.964384 | 0.967725 | 0.959690 | 0.969420 | 0.970862 |
| `nodefull_uformer_geometry_logreg` | `node_test` | 8185 | 0.943677 | 0.633851 | 0.940110 | 0.971984 | 0.000000 |
| `nodefull_uformer_geometry_logreg` | `node_trainval` | 23570 | 0.971574 | 0.973960 | 0.965008 | 0.967591 | 1.000000 |
| `nodefull_uformer_geometry_mlp` | `node_all` | 31755 | 0.986742 | 0.985499 | 0.998181 | 0.974501 | 0.970862 |
| `nodefull_uformer_geometry_mlp` | `node_test` | 8185 | 0.950397 | 0.638445 | 0.993681 | 0.940352 | 0.000000 |
| `nodefull_uformer_geometry_mlp` | `node_trainval` | 23570 | 0.999364 | 0.999410 | 0.999403 | 0.998871 | 1.000000 |
| `nodefull_uformer_geometry_rf` | `node_all` | 31755 | 0.991938 | 0.992243 | 0.993076 | 0.990779 | 0.990206 |
| `nodefull_uformer_geometry_rf` | `node_test` | 8185 | 0.980574 | 0.921254 | 0.989560 | 0.981699 | 0.663866 |
| `nodefull_uformer_geometry_rf` | `node_trainval` | 23570 | 0.995885 | 0.996197 | 0.994031 | 0.997259 | 1.000000 |
| `uformer_geometry_hgb` | `node_all` | 31755 | 0.924295 | 0.933640 | 0.970897 | 0.831671 | 0.970862 |
| `uformer_geometry_hgb` | `node_test` | 8185 | 0.794013 | 0.532694 | 0.953297 | 0.684365 | 0.000000 |
| `uformer_geometry_hgb` | `node_trainval` | 23570 | 0.969538 | 0.971696 | 0.975677 | 0.936795 | 1.000000 |
| `uformer_geometry_hgb` | `original_bad_core_nearboundary` | 119 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_geometry_hgb` | `original_bad_outlier_stress` | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_geometry_hgb` | `original_test_all_10s+` | 8477 | 0.766663 | 0.523398 | 0.953297 | 0.684365 | 0.000000 |
| `uformer_geometry_hgb` | `original_test_main_without_bad_stress` | 8185 | 0.794013 | 0.532694 | 0.953297 | 0.684365 | 0.000000 |
| `uformer_geometry_logreg` | `node_all` | 31755 | 0.901527 | 0.920393 | 0.902247 | 0.863474 | 0.997551 |
| `uformer_geometry_logreg` | `node_test` | 8185 | 0.863653 | 0.893592 | 0.937912 | 0.801175 | 0.915966 |
| `uformer_geometry_logreg` | `node_trainval` | 23570 | 0.914680 | 0.923652 | 0.892561 | 0.907933 | 1.000000 |
| `uformer_geometry_logreg` | `original_bad_core_nearboundary` | 119 | 0.915966 | 0.318713 | 0.000000 | 0.000000 | 0.915966 |
| `uformer_geometry_logreg` | `original_bad_outlier_stress` | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_geometry_logreg` | `original_test_all_10s+` | 8477 | 0.833904 | 0.704441 | 0.937912 | 0.801175 | 0.265207 |
| `uformer_geometry_logreg` | `original_test_main_without_bad_stress` | 8185 | 0.863653 | 0.893592 | 0.937912 | 0.801175 | 0.915966 |
| `uformer_geometry_mlp` | `node_all` | 31755 | 0.908235 | 0.923899 | 0.950302 | 0.805890 | 0.999021 |
| `uformer_geometry_mlp` | `node_test` | 8185 | 0.819426 | 0.872077 | 0.956319 | 0.702892 | 0.966387 |
| `uformer_geometry_mlp` | `node_trainval` | 23570 | 0.939075 | 0.943406 | 0.948668 | 0.879394 | 1.000000 |
| `uformer_geometry_mlp` | `original_bad_core_nearboundary` | 119 | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 |
| `uformer_geometry_mlp` | `original_bad_outlier_stress` | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_geometry_mlp` | `original_test_all_10s+` | 8477 | 0.791200 | 0.680719 | 0.956319 | 0.702892 | 0.279805 |
| `uformer_geometry_mlp` | `original_test_main_without_bad_stress` | 8185 | 0.819426 | 0.872077 | 0.956319 | 0.702892 | 0.966387 |
| `uformer_geometry_rf` | `node_all` | 31755 | 0.903480 | 0.917775 | 0.926949 | 0.840045 | 0.970617 |
| `uformer_geometry_rf` | `node_test` | 8185 | 0.795480 | 0.533878 | 0.940934 | 0.697244 | 0.000000 |
| `uformer_geometry_rf` | `node_trainval` | 23570 | 0.940984 | 0.946768 | 0.923152 | 0.941954 | 0.999748 |
| `uformer_geometry_rf` | `original_bad_core_nearboundary` | 119 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_geometry_rf` | `original_bad_outlier_stress` | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_geometry_rf` | `original_test_all_10s+` | 8477 | 0.768078 | 0.524546 | 0.940934 | 0.697244 | 0.000000 |
| `uformer_geometry_rf` | `original_test_main_without_bad_stress` | 8185 | 0.795480 | 0.533878 | 0.940934 | 0.697244 | 0.000000 |
| `uformer_logits_logreg` | `node_all` | 31755 | 0.822201 | 0.857106 | 0.767412 | 0.853030 | 0.970617 |
| `uformer_logits_logreg` | `node_test` | 8185 | 0.791203 | 0.530283 | 0.751648 | 0.845007 | 0.000000 |
| `uformer_logits_logreg` | `node_trainval` | 23570 | 0.832966 | 0.856663 | 0.771693 | 0.858755 | 0.999748 |
| `uformer_logits_logreg` | `original_bad_core_nearboundary` | 119 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_logits_logreg` | `original_bad_outlier_stress` | 292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `uformer_logits_logreg` | `original_test_all_10s+` | 8477 | 0.763950 | 0.521586 | 0.751648 | 0.845007 | 0.000000 |
| `uformer_logits_logreg` | `original_test_main_without_bad_stress` | 8185 | 0.791203 | 0.530283 | 0.751648 | 0.845007 | 0.000000 |

## Interpretation

- If `uformer_geometry_*` beats `uformer_logits_logreg`, the UFormer representation benefits from an explicit SQI/geometry feature branch.
- If `geometry_only_*` is already strongest, the missing signal is mostly tabular morphology/SQI logic rather than waveform denoising capacity.
- Original buckets remain report-only; they are included to reveal transfer gaps, not to select checkpoints.
