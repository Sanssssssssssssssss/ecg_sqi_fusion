# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 23:16:25
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v106_pca_shell_cover_v101x4_from_v104_s20260731`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          1.0017             10.2946         6.5059      1.0000               0.0737
good         0.4509              1.0707         0.9524      1.0000               0.2879
medium       0.3338              0.7757         0.6476      1.0000               0.3200
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.178
good     -0.076
medium    0.075
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.001
good      0.610
medium    0.542
```

Mean quantile_loss_reduction:
```
class_name
bad      0.002
good     0.452
medium   0.591
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise         band_15_30         169.046   169.046
v81        bad      bad_highfreq_detail_noise         band_30_45         158.968   158.968
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -34.552    34.552
v81        bad      bad_highfreq_detail_noise      baseline_step          17.205    17.205
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -15.495    15.495
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          14.762    14.762
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          13.531    13.531
v81        bad      bad_highfreq_detail_noise     qrs_visibility          12.505    12.505
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -12.233    12.233
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -11.560    11.560
v81        bad             bad_other_boundary   raw_diff_abs_p95         -10.742    10.742
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.485     9.485
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.079     9.079
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -8.986     8.986
v81        bad    bad_baseline_wander_lowfreq         band_15_30           8.953     8.953
v81        bad      bad_highfreq_detail_noise  amplitude_entropy           8.554     8.554
v81     medium medium_outlier_or_bad_boundary     qrs_band_ratio          -6.857     6.857
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.