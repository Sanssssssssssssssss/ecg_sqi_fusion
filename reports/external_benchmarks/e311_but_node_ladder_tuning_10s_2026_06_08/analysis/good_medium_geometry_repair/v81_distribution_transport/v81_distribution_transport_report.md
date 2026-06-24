# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 04:47:07
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v82_targeted_transport_pc1500_cpa6_s20260682`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          1.0495             15.2989         9.1446      1.0000               0.0167
good         0.7506              2.4228         1.8226      1.0000               0.1457
medium       0.6753             13.1888         8.7837      1.0000               0.2021
```

Mean rbf_mmd_reduction:
```
class_name
bad       0.055
good     -0.030
medium   -0.056
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.194
good      0.473
medium   -8.766
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.226
good      0.168
medium   -7.805
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -61.681    61.681
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95         -19.168    19.168
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -16.853    16.853
v81        bad      bad_highfreq_detail_noise      baseline_step          16.774    16.774
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -16.124    16.124
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          15.922    15.922
v81        bad    bad_baseline_wander_lowfreq   non_qrs_diff_p95         -15.486    15.486
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -15.146    15.146
v81       good      good_hard_baseline_lowqrs     qrs_band_ratio         -14.972    14.972
v81        bad             bad_other_boundary   raw_diff_abs_p95         -14.915    14.915
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          13.018    13.018
v81        bad bad_detector_template_disagree         band_15_30         -12.059    12.059
v81        bad             bad_other_boundary   non_qrs_diff_p95         -11.919    11.919
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -11.727    11.727
v81        bad bad_detector_template_disagree   non_qrs_diff_p95         -11.543    11.543
v81        bad         bad_dense_right_island   non_qrs_diff_p95         -11.420    11.420
v81        bad         bad_dense_right_island         band_15_30         -11.351    11.351
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.