# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 20:32:34
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v102_large_transport_pc6000_cpa4_s20260727`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.9942             10.5412         6.6263      1.0000               0.0760
good         0.4318              2.4925         1.8789      1.0000               0.2916
medium       0.3371              0.9422         0.8171      1.0000               0.4031
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.162
good     -0.039
medium    0.061
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.014
good      0.346
medium    0.498
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.026
good      0.070
medium    0.504
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81       good     good_mild_artifact_outlier     qrs_visibility      -60775.089 60775.089
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise         band_15_30         168.900   168.900
v81        bad      bad_highfreq_detail_noise         band_30_45         167.701   167.701
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -33.795    33.795
v81        bad      bad_highfreq_detail_noise      baseline_step          17.585    17.585
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -15.802    15.802
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          15.153    15.153
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          13.689    13.689
v81        bad      bad_highfreq_detail_noise     qrs_visibility          12.840    12.840
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -12.438    12.438
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -11.748    11.748
v81        bad             bad_other_boundary   raw_diff_abs_p95         -10.947    10.947
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.468     9.468
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.043     9.043
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.011     9.011
v81        bad      bad_highfreq_detail_noise  amplitude_entropy           8.616     8.616
v81     medium medium_outlier_or_bad_boundary     qrs_band_ratio          -8.268     8.268
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.