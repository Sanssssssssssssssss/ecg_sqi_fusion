# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 07:40:04
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v93_bad_param_family_transport_pc1500_cpa8_s20260693`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8608              8.1599         5.3539      0.9978               0.0193
good         0.3916              1.4276         1.2837      1.0000               0.3062
medium       0.2734              0.8117         0.6852      1.0000               0.3460
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.126
good      0.069
medium    0.167
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.202
good      0.541
medium    0.216
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.194
good      0.258
medium    0.173
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr  abs_gap
v81        bad             bad_other_boundary detector_agreement        8333.334 8333.334
v81        bad         bad_dense_right_island detector_agreement        8333.334 8333.334
v81        bad bad_detector_template_disagree detector_agreement        8333.334 8333.334
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -14.131   14.131
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -13.302   13.302
v81        bad             bad_other_boundary   raw_diff_abs_p95         -12.594   12.594
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.369    9.369
v81        bad bad_detector_template_disagree         band_15_30           9.025    9.025
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -8.893    8.893
v81        bad         bad_dense_right_island         band_15_30           8.797    8.797
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -8.765    8.765
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio          -8.656    8.656
v81        bad             bad_other_boundary         band_15_30           8.474    8.474
v81        bad         bad_dense_right_island     qrs_visibility           8.264    8.264
v81        bad         bad_dense_right_island      baseline_step           8.153    8.153
v81       good      good_hard_baseline_lowqrs     qrs_band_ratio          -7.865    7.865
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.847    7.847
v81        bad         bad_dense_right_island         sqi_basSQI          -7.548    7.548
v81        bad bad_detector_template_disagree      baseline_step           7.227    7.227
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95           7.130    7.130
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.