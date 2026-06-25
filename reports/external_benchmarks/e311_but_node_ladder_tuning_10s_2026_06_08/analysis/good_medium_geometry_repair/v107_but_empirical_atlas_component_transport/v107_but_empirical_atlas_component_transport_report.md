# v107 BUT Empirical Atlas Component Transport

Generated: 2026-06-25 00:00:28

## What changed

- BUT train+val is treated as a multi-component empirical distribution, not a Gaussian blob.
- Each `class x subtype` is split into `core / upper_shell / lower_shell / left_ridge / right_ridge / sparse_tail` using shared PCA geometry and local support density.
- PTB rows are selected from the v104 large waveform-real candidate pool to cover those components; no BUT waveform is copied.
- This is still support-limited: component shortages are reported explicitly instead of hidden by nearest-centroid fill.

## Protocol

- Protocol path: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v107_but_empirical_atlas_component_transport_v101x4_from_v104_s20260732`
- Rows: `13968`
- Source pool: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v104_brutal_cover_pc9000_cpa8_s20260729`
- BUT target: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed`
- Reference counts: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v99_boundary_repair\protocol_v101_boundary_anchor_v95plus_hardgm_s20260726`
- Component shortage total: `10787`

## Class-Level Median Metrics

| version | class_name | median_rbf_mmd | median_sliced_wasserstein | median_quantile_loss | median_domain_auc | median_pca_density_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| v101 | bad | 0.6367 | 7.0642 | 4.0061 | 1.0000 | 0.0000 |
| v101 | good | 0.3425 | 1.1820 | 1.3051 | 1.0000 | 0.1284 |
| v101 | medium | 0.3056 | 0.7785 | 0.6612 | 1.0000 | 0.1892 |
| v105 | bad | 1.3263 | 9.0142 | 6.3406 | 1.0000 | 0.0000 |
| v105 | good | 0.3967 | 1.3039 | 1.2797 | 1.0000 | 0.1919 |
| v105 | medium | 0.3028 | 0.7352 | 0.6425 | 1.0000 | 0.3386 |
| v106 | bad | 1.3207 | 8.8750 | 6.2697 | 1.0000 | 0.0000 |
| v106 | good | 0.4538 | 0.9680 | 0.9398 | 1.0000 | 0.2677 |
| v106 | medium | 0.3165 | 0.7199 | 0.5887 | 1.0000 | 0.3524 |
| v107 | bad | 1.3195 | 8.9023 | 6.2144 | 1.0000 | 0.0000 |
| v107 | good | 0.4486 | 0.8364 | 0.6931 | 1.0000 | 0.3347 |
| v107 | medium | 0.3112 | 0.6822 | 0.5489 | 1.0000 | 0.3533 |

## v107 Subtype Metric Summary

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 1.3152 | 8.2482 | 6.2144 | 1.0000 | 0.0000 |
| good | 0.4479 | 0.8299 | 0.6931 | 1.0000 | 0.3347 |
| medium | 0.3100 | 0.7016 | 0.5489 | 1.0000 | 0.3533 |

## v107 Component Metric Summary

| class_name | component | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_occupancy_recall |
| --- | --- | --- | --- | --- | --- | --- |
| bad | core | 1.2852 | 9.2632 | 6.8451 | 1.0000 | 0.1111 |
| bad | left_ridge | 1.3388 | 11.9654 | 8.0229 | 1.0000 | 0.0312 |
| bad | lower_shell | 1.1743 | 9.4542 | 6.9339 | 1.0000 | 0.0857 |
| bad | right_ridge | 1.2991 | 35.6138 | 25.5683 | 1.0000 | 0.0308 |
| bad | sparse_tail | 1.1247 | 8.2824 | 5.1270 | 1.0000 | 0.0893 |
| bad | upper_shell | 1.2923 | 11.1238 | 7.5988 | 1.0000 | 0.0303 |
| good | core | 0.6120 | 1.3121 | 1.0424 | 1.0000 | 0.3333 |
| good | left_ridge | 0.6049 | 2.6084 | 1.2728 | 1.0000 | 0.3056 |
| good | lower_shell | 0.6401 | 3.7721 | 1.2769 | 1.0000 | 0.2000 |
| good | right_ridge | 0.5611 | 1.2755 | 1.2824 | 1.0000 | 0.1667 |
| good | sparse_tail | 0.4831 | 0.8773 | 0.7122 | 1.0000 | 0.3333 |
| good | upper_shell | 0.4204 | 1.0628 | 0.9904 | 1.0000 | 0.3030 |
| medium | core | 0.4219 | 0.9409 | 0.7711 | 1.0000 | 0.6667 |
| medium | left_ridge | 0.5505 | 1.3733 | 1.0203 | 1.0000 | 0.3468 |
| medium | lower_shell | 0.3733 | 0.7807 | 0.6740 | 1.0000 | 0.4857 |
| medium | right_ridge | 0.6059 | 13.8612 | 10.3543 | 1.0000 | 0.1716 |
| medium | sparse_tail | 0.2871 | 0.5812 | 0.5585 | 1.0000 | 0.6190 |
| medium | upper_shell | 0.4052 | 0.9695 | 0.8674 | 1.0000 | 0.4167 |

## Key Figures

- `v107_empirical_component_component_shared_pca.png`
- `v107_empirical_component_shared_pca.png`
- `v107_empirical_component_key_feature_cdf_overlay.png`
- `v107_empirical_component_rbf_mmd_heatmap.png`
- `v107_empirical_component_component_pca_occupancy_recall_heatmap.png`
- `v107_empirical_component_good_subtype_waveforms.png`
- `v107_empirical_component_medium_subtype_waveforms.png`
- `v107_empirical_component_bad_subtype_waveforms.png`

## Interpretation

If subtype domain AUC stays near 1 while component occupancy improves, the remaining issue is waveform mechanism support rather than selection. In that case the next generator change should create missing mechanisms directly, especially bad detector/template disagreement, low-QRS visibility, high-frequency detail, and contact/reset/flatline events.
