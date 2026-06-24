# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 05:40:54
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v86_bad_hybrid_transport_pc1500_cpa8_s20260686`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          1.1472             12.4478         7.4520      1.0000               0.0153
good         0.6742              2.3472         1.7888      1.0000               0.1973
medium       0.6085              7.6457         3.4241      1.0000               0.2109
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.012
good      0.079
medium    0.049
```

Mean sliced_wasserstein_reduction:
```
class_name
bad       0.067
good      0.480
medium   -4.263
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.039
good      0.180
medium   -2.335
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr  abs_gap
v81        bad             bad_other_boundary detector_agreement        8333.334 8333.334
v81        bad         bad_dense_right_island detector_agreement        8333.334 8333.334
v81        bad bad_detector_template_disagree detector_agreement        8333.334 8333.334
v81        bad      bad_highfreq_detail_noise         band_15_30         276.546  276.546
v81        bad      bad_highfreq_detail_noise         band_30_45          36.503   36.503
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -25.458   25.458
v81        bad         bad_dense_right_island         band_15_30          21.140   21.140
v81        bad bad_detector_template_disagree         band_15_30          20.912   20.912
v81        bad             bad_other_boundary         band_15_30          19.073   19.073
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          15.365   15.365
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -10.713   10.713
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -10.207   10.207
v81        bad             bad_other_boundary   non_qrs_diff_p95         -10.128   10.128
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.996    9.996
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.506    9.506
v81        bad      bad_highfreq_detail_noise     qrs_visibility           9.309    9.309
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio           9.255    9.255
v81        bad bad_detector_template_disagree   raw_diff_abs_p95          -9.086    9.086
v81        bad             bad_other_boundary   raw_diff_abs_p95          -8.625    8.625
v81        bad         bad_dense_right_island  amplitude_entropy          -7.795    7.795
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.