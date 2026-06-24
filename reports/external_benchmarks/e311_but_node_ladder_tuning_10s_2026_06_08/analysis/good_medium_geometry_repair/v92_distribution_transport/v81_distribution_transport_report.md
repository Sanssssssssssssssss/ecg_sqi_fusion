# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 07:18:57
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v92_rawdiff_target_transport_pc1500_cpa8_s20260692`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8778              8.1891         5.5528      0.9978               0.0226
good         0.3958              1.5018         1.2766      1.0000               0.2926
medium       0.2728              2.4343         0.7010      1.0000               0.3549
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.157
good      0.059
medium    0.144
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.302
good      0.508
medium   -1.404
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.225
good      0.256
medium    0.155
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -17.827    17.827
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -16.966    16.966
v81        bad             bad_other_boundary   raw_diff_abs_p95         -15.494    15.494
v81        bad             bad_other_boundary   non_qrs_diff_p95         -11.465    11.465
v81        bad bad_detector_template_disagree   non_qrs_diff_p95         -10.909    10.909
v81        bad         bad_dense_right_island   non_qrs_diff_p95         -10.849    10.849
v81        bad         bad_dense_right_island         band_30_45          -8.240     8.240
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.706     7.706
v81        bad bad_detector_template_disagree      baseline_step           7.532     7.532
v81        bad         bad_dense_right_island      baseline_step           7.488     7.488
v81        bad             bad_other_boundary         band_30_45          -7.414     7.414
v81        bad bad_detector_template_disagree         band_30_45          -7.399     7.399
v81        bad         bad_dense_right_island     qrs_visibility           7.195     7.195
v81        bad         bad_low_qrs_visibility   non_qrs_diff_p95          -7.070     7.070
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio          -7.015     7.015
v81        bad bad_detector_template_disagree         sqi_basSQI          -7.009     7.009
v81        bad         bad_dense_right_island         sqi_basSQI          -6.975     6.975
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.