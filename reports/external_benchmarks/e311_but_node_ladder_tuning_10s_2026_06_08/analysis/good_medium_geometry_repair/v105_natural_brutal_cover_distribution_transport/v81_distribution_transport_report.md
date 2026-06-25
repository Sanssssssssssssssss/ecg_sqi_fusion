# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 22:38:58
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v105_natural_large_v101x4_from_v104_s20260730`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.9874             10.7459         6.6032      1.0000               0.0733
good         0.4341              1.5898         1.4999      1.0000               0.2490
medium       0.3300              0.8534         0.7740      1.0000               0.3114
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.140
good     -0.023
medium    0.084
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.070
good      0.424
medium    0.510
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.030
good      0.186
medium    0.532
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81       good     good_mild_artifact_outlier     qrs_visibility      -57627.517 57627.517
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise         band_15_30         169.696   169.696
v81        bad      bad_highfreq_detail_noise         band_30_45         164.049   164.049
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -34.055    34.055
v81        bad      bad_highfreq_detail_noise      baseline_step          17.147    17.147
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -15.448    15.448
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          14.878    14.878
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          13.581    13.581
v81        bad      bad_highfreq_detail_noise     qrs_visibility          12.698    12.698
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -12.417    12.417
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -11.750    11.750
v81        bad             bad_other_boundary   raw_diff_abs_p95         -10.887    10.887
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.568     9.568
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.163     9.163
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.117     9.117
v81        bad      bad_highfreq_detail_noise  amplitude_entropy           8.550     8.550
v81     medium medium_outlier_or_bad_boundary     qrs_band_ratio          -7.819     7.819
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.