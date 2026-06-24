# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 07:04:14
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v91_dense_derivative_repair_transport_pc1500_cpa8_s20260691`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8743              8.1088         5.5783      1.0000               0.0183
good         0.3883              1.5562         1.3153      1.0000               0.3038
medium       0.2754              1.8412         0.6771      1.0000               0.3621
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.155
good      0.077
medium    0.151
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.312
good      0.498
medium   -0.722
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.226
good      0.259
medium    0.178
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -17.759    17.759
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -16.966    16.966
v81        bad             bad_other_boundary   raw_diff_abs_p95         -15.569    15.569
v81        bad             bad_other_boundary   non_qrs_diff_p95         -11.438    11.438
v81        bad bad_detector_template_disagree   non_qrs_diff_p95         -11.064    11.064
v81        bad         bad_dense_right_island   non_qrs_diff_p95         -10.884    10.884
v81        bad    bad_baseline_wander_lowfreq         band_15_30           9.614     9.614
v81        bad         bad_dense_right_island         band_30_45          -8.211     8.211
v81        bad             bad_other_boundary         band_30_45          -7.424     7.424
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.409     7.409
v81        bad bad_detector_template_disagree         band_30_45          -7.382     7.382
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio          -7.198     7.198
v81        bad         bad_dense_right_island      baseline_step           7.170     7.170
v81        bad bad_detector_template_disagree      baseline_step           6.947     6.947
v81        bad         bad_low_qrs_visibility   non_qrs_diff_p95          -6.939     6.939
v81        bad         bad_low_qrs_visibility         sqi_basSQI           6.905     6.905
v81        bad         bad_dense_right_island         sqi_basSQI          -6.698     6.698
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.