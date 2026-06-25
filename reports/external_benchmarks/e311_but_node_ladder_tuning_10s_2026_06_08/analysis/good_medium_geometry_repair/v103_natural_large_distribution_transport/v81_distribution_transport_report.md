# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 20:46:08
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v103_natural_large_v101x4_from_v102_s20260728`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.9963             10.3502         6.6461      1.0000               0.0788
good         0.4214              2.4244         1.9041      1.0000               0.2251
medium       0.3354              0.9267         0.8253      1.0000               0.3176
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.155
good     -0.006
medium    0.072
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.088
good      0.404
medium    0.476
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.031
good      0.048
medium    0.497
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81       good     good_mild_artifact_outlier     qrs_visibility      -66385.758 66385.758
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise         band_15_30         168.925   168.925
v81        bad      bad_highfreq_detail_noise         band_30_45         167.754   167.754
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -33.789    33.789
v81        bad      bad_highfreq_detail_noise      baseline_step          17.585    17.585
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -15.802    15.802
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          15.152    15.152
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          13.689    13.689
v81        bad      bad_highfreq_detail_noise     qrs_visibility          12.837    12.837
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -12.438    12.438
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -11.749    11.749
v81        bad             bad_other_boundary   raw_diff_abs_p95         -10.947    10.947
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.468     9.468
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.042     9.042
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.011     9.011
v81        bad      bad_highfreq_detail_noise  amplitude_entropy           8.617     8.617
v81     medium medium_outlier_or_bad_boundary     qrs_band_ratio          -8.268     8.268
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.