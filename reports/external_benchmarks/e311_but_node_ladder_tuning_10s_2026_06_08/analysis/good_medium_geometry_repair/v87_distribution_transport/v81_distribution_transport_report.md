# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 05:59:30
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v87_consistent_primitive_transport_pc1500_cpa8_s20260687`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8589             10.6805         6.1511      0.9963               0.0167
good         0.3945              1.5013         1.2764      1.0000               0.3068
medium       0.2971              0.8214         0.7357      1.0000               0.3429
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.141
good      0.046
medium    0.083
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.147
good      0.471
medium    0.244
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.141
good      0.261
medium    0.101
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr  abs_gap
v81        bad             bad_other_boundary detector_agreement        8333.334 8333.334
v81        bad         bad_dense_right_island detector_agreement        8333.334 8333.334
v81        bad bad_detector_template_disagree detector_agreement        8333.334 8333.334
v81        bad      bad_highfreq_detail_noise         band_15_30         245.163  245.163
v81        bad      bad_highfreq_detail_noise         band_30_45          47.076   47.076
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -23.445   23.445
v81        bad bad_detector_template_disagree         band_15_30          21.512   21.512
v81        bad         bad_dense_right_island         band_15_30          20.022   20.022
v81        bad             bad_other_boundary         band_15_30          19.792   19.792
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          17.857   17.857
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          11.172   11.172
v81        bad      bad_highfreq_detail_noise     qrs_visibility          10.699   10.699
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -10.663   10.663
v81        bad             bad_other_boundary   non_qrs_diff_p95         -10.166   10.166
v81        bad         bad_dense_right_island   raw_diff_abs_p95          -9.838    9.838
v81        bad         bad_dense_right_island   non_qrs_diff_p95          -9.661    9.661
v81        bad bad_detector_template_disagree   non_qrs_diff_p95          -9.631    9.631
v81        bad bad_detector_template_disagree   raw_diff_abs_p95          -9.177    9.177
v81        bad             bad_other_boundary   raw_diff_abs_p95          -8.633    8.633
v81        bad      bad_highfreq_detail_noise      baseline_step           8.095    8.095
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.