# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 22:34:38
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v104_brutal_cover_pc9000_cpa8_s20260729`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.9842             11.2762         6.6032      0.9997               0.0734
good         0.4341              1.6860         1.5184      1.0000               0.3299
medium       0.3308              0.8588         0.7721      1.0000               0.4113
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.142
good     -0.039
medium    0.079
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.098
good      0.397
medium    0.494
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.032
good      0.168
medium    0.535
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81       good     good_mild_artifact_outlier     qrs_visibility      -52628.189 52628.189
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise         band_15_30         169.522   169.522
v81        bad      bad_highfreq_detail_noise         band_30_45         163.989   163.989
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -34.097    34.097
v81        bad      bad_highfreq_detail_noise      baseline_step          17.142    17.142
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -15.444    15.444
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          14.907    14.907
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          13.544    13.544
v81        bad      bad_highfreq_detail_noise     qrs_visibility          12.708    12.708
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -12.415    12.415
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -11.756    11.756
v81        bad             bad_other_boundary   raw_diff_abs_p95         -10.919    10.919
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.576     9.576
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.162     9.162
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.103     9.103
v81        bad      bad_highfreq_detail_noise  amplitude_entropy           8.556     8.556
v81     medium medium_outlier_or_bad_boundary     qrs_band_ratio          -7.913     7.913
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.