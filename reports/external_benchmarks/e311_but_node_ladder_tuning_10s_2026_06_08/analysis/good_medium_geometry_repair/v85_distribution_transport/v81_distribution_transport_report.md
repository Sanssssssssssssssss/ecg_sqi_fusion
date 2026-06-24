# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 05:28:08
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v85_bad_quasiperiodic_transport_pc1500_cpa8_s20260685`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          1.1504             23.3874         8.5909      1.0000               0.0156
good         0.6879              2.3748         1.7359      1.0000               0.1980
medium       0.6087              1.9469         1.0496      1.0000               0.2224
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.016
good      0.060
medium    0.055
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.193
good      0.476
medium   -0.127
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.049
good      0.207
medium    0.099
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr  abs_gap
v81        bad bad_detector_template_disagree detector_agreement        8333.334 8333.334
v81        bad             bad_other_boundary detector_agreement        8333.334 8333.334
v81        bad      bad_highfreq_detail_noise         band_30_45         900.420  900.420
v81        bad         bad_dense_right_island         band_30_45          12.588   12.588
v81        bad bad_detector_template_disagree         band_30_45          11.905   11.905
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -11.565   11.565
v81        bad             bad_other_boundary         band_30_45          11.067   11.067
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95          -7.457    7.457
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01           5.253    5.253
v81       good      good_hard_baseline_lowqrs     qrs_band_ratio          -4.729    4.729
v81        bad             bad_other_boundary   non_qrs_diff_p95          -4.321    4.321
v81     medium    medium_hard_baseline_lowqrs     flatline_ratio           4.173    4.173
v81        bad         bad_dense_right_island         band_15_30           4.091    4.091
v81        bad             bad_other_boundary         band_15_30           4.080    4.080
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -3.995    3.995
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -3.981    3.981
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -3.855    3.855
v81        bad bad_detector_template_disagree         band_15_30           3.741    3.741
v81        bad      bad_highfreq_detail_noise         sqi_basSQI           3.733    3.733
v81        bad      bad_highfreq_detail_noise      baseline_step          -3.632    3.632
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.