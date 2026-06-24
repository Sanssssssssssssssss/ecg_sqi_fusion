# V81 PTB to BUT Feature Distribution Transport Audit

- Created: 2026-06-24 06:49:35
- V81 protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v90_midband_dense_repair_transport_pc1500_cpa8_s20260690`
- Contract: BUT train+val features are targets; BUT waveforms are never copied.

## Distribution Metrics

```
            rbf_mmd  sliced_wasserstein  quantile_loss  domain_auc  pca_density_overlap
class_name                                                                             
bad          0.8707              7.3675         5.3532      1.0000               0.0277
good         0.3969              1.4627         1.2766      1.0000               0.3018
medium       0.2738              1.7435         0.6796      1.0000               0.3439
```

Mean rbf_mmd_reduction:
```
class_name
bad      -0.142
good      0.054
medium    0.160
```

Mean sliced_wasserstein_reduction:
```
class_name
bad      -0.083
good      0.508
medium   -0.592
```

Mean quantile_loss_reduction:
```
class_name
bad      -0.187
good      0.268
medium    0.176
```

## Worst Key Feature Gaps

```
tag class_name                        subtype            feature  median_gap_iqr   abs_gap
v81        bad             bad_other_boundary detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island detector_agreement       16666.667 16666.667
v81        bad bad_detector_template_disagree detector_agreement       16666.667 16666.667
v81        bad         bad_dense_right_island   raw_diff_abs_p95         -17.434    17.434
v81        bad bad_detector_template_disagree   raw_diff_abs_p95         -16.550    16.550
v81        bad             bad_other_boundary   raw_diff_abs_p95         -15.213    15.213
v81        bad             bad_other_boundary   non_qrs_diff_p95         -11.348    11.348
v81        bad         bad_dense_right_island   non_qrs_diff_p95         -10.971    10.971
v81        bad bad_detector_template_disagree   non_qrs_diff_p95         -10.943    10.943
v81        bad         bad_dense_right_island         band_30_45          -8.343     8.343
v81        bad    bad_baseline_wander_lowfreq   raw_diff_abs_p95          -7.705     7.705
v81     medium    medium_hard_baseline_lowqrs     qrs_band_ratio          -7.650     7.650
v81        bad             bad_other_boundary         band_30_45          -7.493     7.493
v81        bad bad_detector_template_disagree         band_30_45          -7.489     7.489
v81        bad bad_detector_template_disagree         band_15_30           7.203     7.203
v81        bad         bad_dense_right_island         band_15_30           7.121     7.121
v81        bad         bad_dense_right_island      baseline_step           7.098     7.098
v81        bad         bad_low_qrs_visibility   non_qrs_diff_p95          -7.006     7.006
v81        bad             bad_other_boundary         band_15_30           6.896     6.896
v81        bad    bad_baseline_wander_lowfreq   non_qrs_diff_p95          -6.829     6.829
```

## Figures

- `v81_shared_pca.png`
- `v81_key_feature_cdf_overlay.png`
- `v81_rbf_mmd_heatmap.png`
- `v81_good_subtype_waveforms.png`, `v81_medium_subtype_waveforms.png`, `v81_bad_subtype_waveforms.png`

## Interpretation Rule

If MMD/SW/domain-AUC do not improve enough, the next step is not model training. Increase candidate count or add a targeted transform for the worst key feature/subtype reported above.