# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 05:12:47
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v84_mmd_herding_transport_pc1500_cpa8_s20260684`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          1.0509             15.2786         9.0106      1.0000               0.0161
good         0.7054              2.3230         1.6760      1.0000               0.1756
medium       0.6124              1.5572         1.0433      1.0000               0.2345
```

Mean rbf_mmd_reduction:
```
class_name
bad      0.054
good     0.035
medium   0.051
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.064
good      0.528
medium    0.191
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.194
good      0.236
medium    0.100
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -67.582    67.582
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95         -18.980    18.980
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -16.901    16.901
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -16.031    16.031
v81        bad      bad_highfreq_detail_noise      baseline_step          15.667    15.667
v81        bad    bad_baseline_wander_lowfreq   non_qrs_diff_p95         -15.343    15.343
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          15.294    15.294
v81        bad             bad_other_boundary   raw_diff_abs_p95         -14.536    14.536
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -14.240    14.240
v81        bad bad_detector_template_disagree         band_15_30         -12.116    12.116
v81        bad             bad_other_boundary   non_qrs_diff_p95         -11.897    11.897
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          11.791    11.791
v81        bad bad_detector_template_disagree   non_qrs_diff_p95         -11.510    11.510
v81        bad      bad_highfreq_detail_noise   non_qrs_diff_p95         -11.381    11.381
v81        bad         bad_dense_right_island         band_15_30         -11.375    11.375
v81        bad         bad_dense_right_island   non_qrs_diff_p95         -11.345    11.345
v81        bad             bad_other_boundary         band_15_30         -10.696    10.696
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.