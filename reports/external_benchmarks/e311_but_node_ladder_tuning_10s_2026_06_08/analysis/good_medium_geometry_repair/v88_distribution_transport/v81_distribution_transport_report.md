# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 06:16:15
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v88_eventtrain_highfreq_transport_pc1500_cpa8_s20260688`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8755              7.2697         5.1104      0.9978               0.0286
good         0.3943              1.4417         1.3070      1.0000               0.3095
medium       0.2983              3.3228         0.7426      1.0000               0.3225
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.162
good      0.057
medium    0.095
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.098
good      0.497
medium   -2.235
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.161
good      0.253
medium    0.092
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr  abs_gap
v81        bad         bad_dense_right_island detector_agreement        8333.334 8333.334
v81        bad         bad_dense_right_island         band_30_45          17.078   17.078
v81        bad bad_detector_template_disagree         band_30_45          16.190   16.190
v81        bad             bad_other_boundary         band_30_45          15.779   15.779
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio          -9.670    9.670
v81        bad         bad_dense_right_island         band_15_30          -8.427    8.427
v81        bad    bad_baseline_wander_lowfreq         band_15_30           8.134    8.134
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.283    7.283
v81        bad bad_detector_template_disagree         band_15_30          -7.241    7.241
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95           7.120    7.120
v81        bad         bad_low_qrs_visibility   non_qrs_diff_p95          -6.617    6.617
v81        bad    bad_baseline_wander_lowfreq   non_qrs_diff_p95          -6.394    6.394
v81        bad bad_detector_template_disagree      baseline_step           6.372    6.372
v81        bad             bad_other_boundary         band_15_30          -6.276    6.276
v81        bad         bad_low_qrs_visibility         sqi_basSQI           6.252    6.252
v81        bad         bad_dense_right_island      baseline_step           6.227    6.227
v81        bad bad_detector_template_disagree         sqi_basSQI          -5.994    5.994
v81        bad         bad_low_qrs_visibility      baseline_step          -5.895    5.895
v81        bad         bad_dense_right_island         sqi_basSQI          -5.869    5.869
v81        bad         bad_low_qrs_visibility   raw_diff_abs_p95          -5.777    5.777
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.