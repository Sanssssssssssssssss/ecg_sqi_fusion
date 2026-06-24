# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 10:18:50
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v98_metric_hybrid_transport_pc1500_s20260722`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8258              5.8961         3.9272      1.0000               0.0737
good         0.4038              1.5956         1.4899      1.0000               0.2607
medium       0.3041              0.8783         0.7586      1.0000               0.3048
```

Mean rbf_mmd_reduction:
```
class_name
bad      0.077
good     0.043
medium   0.170
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      0.062
good     0.448
medium   0.508
```

Mean quantile_loss_reduction:
```
class_name
bad      0.111
good     0.163
medium   0.537
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81       good     good_mild_artifact_outlier     qrs_visibility      -34049.666 34049.666
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -12.244    12.244
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -11.698    11.698
v81        bad             bad_other_boundary   raw_diff_abs_p95         -10.873    10.873
v81        bad             bad_other_boundary   non_qrs_diff_p95          -9.581     9.581
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.247     9.247
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.110     9.110
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.861     7.861
v81     medium medium_outlier_or_bad_boundary     qrs_band_ratio          -7.822     7.822
v81        bad         bad_low_qrs_visibility   non_qrs_diff_p95          -7.099     7.099
v81        bad         bad_low_qrs_visibility         sqi_basSQI           6.990     6.990
v81        bad    bad_baseline_wander_lowfreq   non_qrs_diff_p95          -6.917     6.917
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95           6.805     6.805
v81        bad         bad_low_qrs_visibility      baseline_step          -6.547     6.547
v81       good      good_hard_baseline_lowqrs   raw_diff_abs_p95           6.358     6.358
v81        bad         bad_low_qrs_visibility   raw_diff_abs_p95          -6.220     6.220
v81        bad    bad_baseline_wander_lowfreq         sqi_basSQI           5.645     5.645
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.