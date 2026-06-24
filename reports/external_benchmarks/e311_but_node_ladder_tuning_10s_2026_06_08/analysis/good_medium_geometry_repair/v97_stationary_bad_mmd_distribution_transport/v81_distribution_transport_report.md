# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 10:13:59
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v97_stationary_bad_mmd_transport_pc1500_cpa10_med16_bad24_s20260721`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8667             11.4815         6.4957      1.0000               0.0742
good         0.4134              1.4025         1.3504      1.0000               0.2675
medium       0.3055              0.7951         0.7366      1.0000               0.3254
```

Mean rbf_mmd_reduction:
```
class_name
bad      0.046
good     0.018
medium   0.159
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.082
good      0.427
medium    0.532
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.019
good      0.205
medium    0.557
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81       good     good_mild_artifact_outlier     qrs_visibility      -33399.069 33399.069
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise         band_15_30         168.289   168.289
v81        bad      bad_highfreq_detail_noise         band_30_45         162.379   162.379
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -34.036    34.036
v81        bad      bad_highfreq_detail_noise      baseline_step          16.435    16.435
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -14.870    14.870
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          14.513    14.513
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          13.135    13.135
v81        bad      bad_highfreq_detail_noise     qrs_visibility          12.354    12.354
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -12.244    12.244
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -11.698    11.698
v81        bad             bad_other_boundary   raw_diff_abs_p95         -10.873    10.873
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.581     9.581
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.247     9.247
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.110     9.110
v81        bad      bad_highfreq_detail_noise  amplitude_entropy           8.370     8.370
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.720     7.720
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.