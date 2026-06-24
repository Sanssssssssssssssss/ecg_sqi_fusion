# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 09:29:14
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v95_gm_boundary_clean_drop4_from_v94_s20260710`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8529              8.0511         5.4148      0.9978               0.0222
good         0.4089              1.4996         1.3555      1.0000               0.2865
medium       0.3371             74.1597        31.5137      1.0000               0.3113
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.116
good      0.018
medium   -0.026
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.160
good      0.467
medium   -0.980
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.212
good      0.155
medium   -0.617
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
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -12.952    12.952
v81        bad             bad_other_boundary   raw_diff_abs_p95         -11.773    11.773
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