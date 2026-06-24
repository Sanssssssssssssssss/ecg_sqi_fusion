# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 05:02:45
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v84_smoke_mmd_transport_pc90_cpa4_s20260684`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.7332             15.6481         9.3394         NaN               0.0094
good         0.5621              3.0548         1.9678         NaN               0.0103
medium       0.5631            125.9836        47.7097         NaN               0.0213
```

Mean rbf_mmd_reduction:
```
class_name
bad      0.282
good     0.225
medium   0.119
```

Mean sliced_wasserstein_reduction:
```
class_name
bad       -0.165
good       0.398
medium   -84.635
```

Mean quantile_loss_reduction:
```
class_name
bad       -0.249
good       0.142
medium   -47.586
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement        8333.334  8333.334
v81        bad         bad_dense_right_island detector_agreement        8333.334  8333.334
v81       good          good_overlap_boundary     qrs_visibility       -5935.264  5935.264
v81        bad      bad_highfreq_detail_noise   raw_diff_abs_p95         -68.249    68.249
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95         -18.730    18.730
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -16.693    16.693
v81        bad      bad_highfreq_detail_noise      baseline_step          16.521    16.521
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -15.854    15.854
v81       good      good_hard_baseline_lowqrs     qrs_band_ratio         -15.481    15.481
v81        bad             bad_other_boundary   raw_diff_abs_p95         -15.224    15.224
v81        bad      bad_highfreq_detail_noise    raw_ptp_p99_p01          15.224    15.224
v81        bad    bad_baseline_wander_lowfreq   non_qrs_diff_p95         -15.012    15.012
v81        bad      bad_highfreq_detail_noise         sqi_basSQI         -14.940    14.940
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio         -13.389    13.389
v81        bad bad_detector_template_disagree         band_15_30         -12.837    12.837
v81        bad      bad_highfreq_detail_noise     qrs_band_ratio          12.479    12.479
v81        bad             bad_other_boundary         band_15_30         -11.952    11.952
v81        bad             bad_other_boundary   non_qrs_diff_p95         -11.625    11.625
v81        bad      bad_highfreq_detail_noise   non_qrs_diff_p95         -11.596    11.596
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.