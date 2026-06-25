# v108 Anchor-Direct Component Transport

Generated: 2026-06-25 03:04:37

## Why this exists

v107 reused the v104 synthetic candidate pool, so it inherited the old round/blob geometry. v108 samples BUT train+val anchors directly and generates PTB-carrier candidates for each anchor.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v109_large_anchor_coverage_repair_v101x4_cpa8m10b48_s20260738`
- Rows: `13968`
- Carrier protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\raw_ptbxl_carrier_protocols\raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680`
- BUT target: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed`
- Candidate policy: good `8`, medium `10`, bad `48` per anchor, sparse tails x1.5.

## v101/v106/v107/v108 class median comparison

| version | class_name | median_rbf_mmd | median_sliced_wasserstein | median_quantile_loss | median_domain_auc | median_pca_density_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| v101 | bad | 0.6367 | 6.5309 | 4.0061 | 1.0000 | 0.0000 |
| v101 | good | 0.3422 | 1.2684 | 1.3051 | 1.0000 | 0.1284 |
| v101 | medium | 0.3055 | 0.8493 | 0.6612 | 1.0000 | 0.1892 |
| v106 | bad | 1.3171 | 8.2466 | 6.2697 | 1.0000 | 0.0000 |
| v106 | good | 0.4541 | 0.9756 | 0.9398 | 1.0000 | 0.2677 |
| v106 | medium | 0.3158 | 0.7199 | 0.5887 | 1.0000 | 0.3524 |
| v107 | bad | 1.3149 | 8.1956 | 6.2144 | 1.0000 | 0.0000 |
| v107 | good | 0.4490 | 0.8810 | 0.6931 | 1.0000 | 0.3347 |
| v107 | medium | 0.3094 | 0.6879 | 0.5489 | 1.0000 | 0.3533 |
| v108 | bad | 0.7089 | 12.5652 | 10.1763 | 1.0000 | 0.0000 |
| v108 | good | 0.3682 | 1.2287 | 1.1591 | 1.0000 | 0.2835 |
| v108 | medium | 0.2469 | 0.6429 | 0.5410 | 1.0000 | 0.3584 |

## v108 subtype summary

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 0.7287 | 13.1287 | 10.1763 | 1.0000 | 0.0000 |
| good | 0.3655 | 1.1592 | 1.1591 | 1.0000 | 0.2835 |
| medium | 0.2456 | 0.6417 | 0.5410 | 1.0000 | 0.3584 |

## v108 component summary

| class_name | component | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_occupancy_recall |
| --- | --- | --- | --- | --- | --- | --- |
| bad | core | 0.8272 | 14.5279 | 11.6373 | 1.0000 | 0.1765 |
| bad | left_ridge | 0.8127 | 14.5705 | 11.5887 | 1.0000 | 0.0938 |
| bad | lower_shell | 0.7670 | 12.3948 | 10.2145 | 1.0000 | 0.1667 |
| bad | right_ridge | 0.7851 | 16.5544 | 11.9860 | 1.0000 | 0.1458 |
| bad | sparse_tail | 0.8465 | 10.0132 | 8.5475 | 1.0000 | 0.1414 |
| bad | upper_shell | 0.6797 | 15.1288 | 12.5862 | 1.0000 | 0.1471 |
| good | core | 0.4836 | 1.9385 | 1.5378 | 1.0000 | 0.2222 |
| good | left_ridge | 0.5278 | 79.9358 | 42.6242 | 1.0000 | 0.1964 |
| good | lower_shell | 0.5434 | 1066.5286 | 391.4873 | 1.0000 | 0.1714 |
| good | right_ridge | 0.5258 | 512.1835 | 305.4526 | 1.0000 | 0.1389 |
| good | sparse_tail | 0.3595 | 0.8954 | 0.6449 | 1.0000 | 0.3000 |
| good | upper_shell | 0.4549 | 1.2986 | 0.9729 | 1.0000 | 0.1944 |
| medium | core | 0.3184 | 0.8742 | 0.7273 | 1.0000 | 0.3750 |
| medium | left_ridge | 0.5023 | 1.3260 | 1.1696 | 1.0000 | 0.3056 |
| medium | lower_shell | 0.2961 | 0.9095 | 0.8070 | 1.0000 | 0.4273 |
| medium | right_ridge | 0.4646 | 14.3713 | 7.1813 | 1.0000 | 0.1690 |
| medium | sparse_tail | 0.2243 | 0.5695 | 0.5622 | 1.0000 | 0.5512 |
| medium | upper_shell | 0.3395 | 0.9174 | 0.7851 | 1.0000 | 0.3587 |

## Figures

- `v108_anchor_direct_shared_pca.png`
- `v108_anchor_direct_component_shared_pca.png`
- `v108_anchor_direct_key_feature_cdf_overlay.png`
- `v108_anchor_direct_rbf_mmd_heatmap.png`
- `v108_anchor_direct_good_subtype_waveforms.png`
- `v108_anchor_direct_medium_subtype_waveforms.png`
- `v108_anchor_direct_bad_subtype_waveforms.png`