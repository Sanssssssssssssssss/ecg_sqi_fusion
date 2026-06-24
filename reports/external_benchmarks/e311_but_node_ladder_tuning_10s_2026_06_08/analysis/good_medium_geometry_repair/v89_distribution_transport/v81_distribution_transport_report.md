# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 06:34:06
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v89_midband_dense_transport_pc1500_cpa8_s20260689`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8893             14.1553         8.8135      1.0000               0.0210
good         0.3960              1.4494         1.2526      1.0000               0.3085
medium       0.2782              2.2058         0.6941      1.0000               0.3421
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.158
good      0.072
medium    0.133
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.759
good      0.520
medium   -1.170
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.631
good      0.273
medium    0.164
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       41666.666 41666.666
v81        bad         bad_dense_right_island detector_agreement       41666.666 41666.666
v81        bad bad_detector_template_disagree detector_agreement       41666.666 41666.666
v81        bad bad_detector_template_disagree         band_15_30         -28.826    28.826
v81        bad         bad_dense_right_island         band_15_30         -27.388    27.388
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -26.673    26.673
v81        bad             bad_other_boundary         band_15_30         -26.432    26.432
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -25.502    25.502
v81        bad             bad_other_boundary   raw_diff_abs_p95         -23.649    23.649
v81        bad             bad_other_boundary   non_qrs_diff_p95         -16.495    16.495
v81        bad         bad_dense_right_island   non_qrs_diff_p95         -15.884    15.884
v81        bad bad_detector_template_disagree   non_qrs_diff_p95         -15.856    15.856
v81        bad bad_detector_template_disagree      baseline_step           9.472     9.472
v81        bad         bad_dense_right_island      baseline_step           9.445     9.445
v81        bad             bad_other_boundary      baseline_step           9.322     9.322
v81        bad bad_detector_template_disagree         sqi_basSQI          -8.659     8.659
v81        bad         bad_dense_right_island         sqi_basSQI          -8.642     8.642
v81        bad             bad_other_boundary         sqi_basSQI          -8.509     8.509
v81        bad         bad_dense_right_island         band_30_45          -8.473     8.473
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.730     7.730
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.