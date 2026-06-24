# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 04:54:10
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v83_detector_edge_transport_pc1500_cpa6_s20260683`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          1.0566             15.0425         9.2515      1.0000               0.0137
good         0.7537              2.4155         1.7677      1.0000               0.1512
medium       0.6731             19.0804        12.6582      1.0000               0.2243
```

Mean rbf_mmd_reduction:
```
class_name
bad       0.048
good     -0.031
medium   -0.051
```

Mean sliced_wasserstein_reduction:
```
class_name
bad       -0.149
good       0.515
medium   -11.827
```

Mean quantile_loss_reduction:
```
class_name
bad       -0.229
good       0.213
medium   -11.769
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -67.637    67.637
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95         -19.081    19.081
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -17.092    17.092
v81        bad      bad_highfreq_detail_noise      baseline_step          16.543    16.543
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -16.107    16.107
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          15.489    15.489
v81        bad    bad_baseline_wander_lowfreq   non_qrs_diff_p95         -15.410    15.410
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -14.958    14.958
v81        bad             bad_other_boundary   raw_diff_abs_p95         -14.788    14.788
v81        bad bad_detector_template_disagree         band_15_30         -12.396    12.396
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          12.092    12.092
v81        bad             bad_other_boundary   non_qrs_diff_p95         -12.076    12.076
v81        bad         bad_dense_right_island         band_15_30         -11.684    11.684
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -11.662    11.662
v81        bad bad_detector_template_disagree   non_qrs_diff_p95         -11.498    11.498
v81        bad         bad_dense_right_island   non_qrs_diff_p95         -11.465    11.465
v81        bad      bad_highfreq_detail_noise   non_qrs_diff_p95         -11.388    11.388
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.