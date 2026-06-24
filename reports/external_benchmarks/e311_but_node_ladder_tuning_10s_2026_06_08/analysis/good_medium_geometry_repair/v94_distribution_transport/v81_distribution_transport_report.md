# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 08:14:57
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v94_bad_impulse_edge_transport_pc1500_cpa8_s20260694`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8530              7.7172         5.3977      1.0000               0.0225
good         0.3928              1.4825         1.2968      1.0000               0.2957
medium       0.3396             87.7784        31.3540      1.0000               0.2845
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.116
good      0.071
medium   -0.028
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.151
good      0.495
medium   -0.978
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.209
good      0.247
medium   -0.392
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81     medium medium_outlier_or_bad_boundary     qrs_visibility      -84155.512 84155.512
v81        bad bad_detector_template_disagree detector_agreement        8333.334  8333.334
v81        bad         bad_dense_right_island detector_agreement        8333.334  8333.334
v81        bad             bad_other_boundary detector_agreement        8333.334  8333.334
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -13.750    13.750
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -13.220    13.220
v81        bad             bad_other_boundary   raw_diff_abs_p95         -11.773    11.773
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -10.237    10.237
v81        bad bad_detector_template_disagree         band_15_30           8.875     8.875
v81        bad             bad_other_boundary   non_qrs_diff_p95          -8.866     8.866
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -8.835     8.835
v81       good      good_hard_baseline_lowqrs     qrs_band_ratio          -8.738     8.738
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -8.566     8.566
v81        bad         bad_dense_right_island         band_15_30           8.198     8.198
v81        bad         bad_dense_right_island     qrs_visibility           7.834     7.834
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.796     7.796
v81        bad             bad_other_boundary         band_15_30           7.741     7.741
v81        bad         bad_dense_right_island      baseline_step           7.308     7.308
v81        bad bad_detector_template_disagree      baseline_step           7.148     7.148
v81        bad         bad_low_qrs_visibility         sqi_basSQI           7.141     7.141
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.