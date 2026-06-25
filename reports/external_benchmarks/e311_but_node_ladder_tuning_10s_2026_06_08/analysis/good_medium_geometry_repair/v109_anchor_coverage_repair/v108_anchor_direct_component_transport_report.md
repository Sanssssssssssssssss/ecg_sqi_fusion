# v108 Anchor-Direct Component Transport

Generated: 2026-06-25 02:13:01

## Why this exists

v107 reused the v104 synthetic candidate pool, so it inherited the old round/blob geometry. v108 samples BUT train+val anchors directly and generates PTB-carrier candidates for each anchor.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v109_anchor_coverage_repair_v101x1_cpa8m10b48_s20260736`
- Rows: `3492`
- Carrier protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\raw_ptbxl_carrier_protocols\raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680`
- BUT target: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed`
- Candidate policy: good `8`, medium `10`, bad `48` per anchor, sparse tails x1.5.

## v101/v106/v107/v108 class median comparison

| version | class_name | median_rbf_mmd | median_sliced_wasserstein | median_quantile_loss | median_domain_auc | median_pca_density_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| v101 | bad | 0.6367 | 6.7859 | 4.0061 | 1.0000 | 0.0000 |
| v101 | good | 0.3433 | 1.1729 | 1.3051 | 1.0000 | 0.1284 |
| v101 | medium | 0.3059 | 0.8614 | 0.6612 | 1.0000 | 0.1892 |
| v106 | bad | 1.3160 | 9.3087 | 6.2697 | 1.0000 | 0.0000 |
| v106 | good | 0.4549 | 1.0921 | 0.9398 | 1.0000 | 0.2677 |
| v106 | medium | 0.3169 | 0.7708 | 0.5887 | 1.0000 | 0.3524 |
| v107 | bad | 1.3116 | 9.2663 | 6.2144 | 1.0000 | 0.0000 |
| v107 | good | 0.4495 | 0.8787 | 0.6931 | 1.0000 | 0.3347 |
| v107 | medium | 0.3088 | 0.7391 | 0.5489 | 1.0000 | 0.3533 |
| v108 | bad | 0.5619 | 11.7595 | 9.9509 | 1.0000 | 0.0000 |
| v108 | good | 0.3592 | 1.0638 | 0.9885 | 1.0000 | 0.1750 |
| v108 | medium | 0.2642 | 0.7346 | 0.5763 | 1.0000 | 0.2165 |

## v108 subtype summary

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 0.5619 | 11.9960 | 9.9509 | 1.0000 | 0.0000 |
| good | 0.3560 | 1.1555 | 0.9885 | 1.0000 | 0.1750 |
| medium | 0.2638 | 0.7552 | 0.5763 | 1.0000 | 0.2165 |

## v108 component summary

| class_name | component | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_occupancy_recall |
| --- | --- | --- | --- | --- | --- | --- |
| bad | core | 0.8290 | 15.3246 | 11.7222 | 1.0000 | 0.1471 |
| bad | left_ridge | 0.8596 | 12.5586 | 9.6249 | 1.0000 | 0.0714 |
| bad | lower_shell | 0.8413 | 10.4222 | 8.6337 | 1.0000 | 0.1111 |
| bad | right_ridge | 0.8181 | 12.7162 | 9.9877 | 1.0000 | 0.0764 |
| bad | sparse_tail | 0.7133 | 9.2854 | 7.7452 | 1.0000 | 0.0655 |
| bad | upper_shell | 0.6756 | 12.9962 | 9.8698 | 1.0000 | 0.0909 |
| good | core | 0.4390 | 532.9807 | 299.8362 | 1.0000 | 0.2222 |
| good | left_ridge | 0.4602 | 147.0136 | 75.1630 | 1.0000 | 0.1944 |
| good | lower_shell | 0.4679 | 1464.4217 | 468.5225 | 1.0000 | 0.1714 |
| good | right_ridge | 0.4383 | 787.7495 | 390.4962 | 1.0000 | 0.1389 |
| good | sparse_tail | 0.3387 | 0.8406 | 0.6099 | 1.0000 | 0.2963 |
| good | upper_shell | 0.4472 | 87.4346 | 45.8848 | 1.0000 | 0.1667 |
| medium | core | 0.3825 | 16.6577 | 9.0323 | 1.0000 | 0.3536 |
| medium | left_ridge | 0.3728 | 1.1883 | 1.0391 | 1.0000 | 0.1713 |
| medium | lower_shell | 0.3189 | 1.0434 | 0.9465 | 1.0000 | 0.3345 |
| medium | right_ridge | 0.4618 | 159.2905 | 85.9244 | 1.0000 | 0.1528 |
| medium | sparse_tail | 0.2346 | 0.5706 | 0.4963 | 1.0000 | 0.3252 |
| medium | upper_shell | 0.3899 | 1.5832 | 1.3342 | 1.0000 | 0.3194 |

## Figures

- `v108_anchor_direct_shared_pca.png`
- `v108_anchor_direct_component_shared_pca.png`
- `v108_anchor_direct_key_feature_cdf_overlay.png`
- `v108_anchor_direct_rbf_mmd_heatmap.png`
- `v108_anchor_direct_good_subtype_waveforms.png`
- `v108_anchor_direct_medium_subtype_waveforms.png`
- `v108_anchor_direct_bad_subtype_waveforms.png`