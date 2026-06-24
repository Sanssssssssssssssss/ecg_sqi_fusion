# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 09:54:18
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v96_feature_weighted_mmd_transport_pc1500_cpa10_med16_bad24_s20260720`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8535              8.3494         5.7075      1.0000               0.0158
good         0.3898              1.4949         1.2465      1.0000               0.3042
medium       0.2714              0.8459         0.7142      1.0000               0.3415
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.138
good      0.074
medium    0.150
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.199
good      0.490
medium    0.384
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.249
good      0.279
medium    0.295
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr  abs_gap
v81        bad             bad_other_boundary detector_agreement        8333.334 8333.334
v81        bad         bad_dense_right_island detector_agreement        8333.334 8333.334
v81        bad bad_detector_template_disagree detector_agreement        8333.334 8333.334
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -13.833   13.833
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -13.389   13.389
v81        bad             bad_other_boundary   raw_diff_abs_p95         -12.225   12.225
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -8.942    8.942
v81        bad             bad_other_boundary   non_qrs_diff_p95          -8.884    8.884
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -8.567    8.567
v81        bad         bad_dense_right_island     qrs_visibility           8.070    8.070
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.861    7.861
v81        bad bad_detector_template_disagree      baseline_step           7.816    7.816
v81        bad         bad_dense_right_island      baseline_step           7.667    7.667
v81        bad             bad_other_boundary      baseline_step           7.622    7.622
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio          -7.601    7.601
v81        bad bad_detector_template_disagree         sqi_basSQI          -7.254    7.254
v81        bad         bad_dense_right_island         sqi_basSQI          -7.130    7.130
v81        bad         bad_low_qrs_visibility   non_qrs_diff_p95          -7.099    7.099
v81        bad             bad_other_boundary         sqi_basSQI          -7.069    7.069
v81        bad         bad_dense_right_island         band_15_30           7.024    7.024
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.