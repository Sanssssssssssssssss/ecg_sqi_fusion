# v108 Anchor-Direct Component Transport

Generated: 2026-06-25 00:34:55

## Why this exists

v107 reused the v104 synthetic candidate pool, so it inherited the old round/blob geometry. v108 samples BUT train+val anchors directly and generates PTB-carrier candidates for each anchor.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v108_smoke_anchor_direct_component_transport_v101x1_cpa4m5b12_s20260734`
- Rows: `3492`
- Carrier protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\raw_ptbxl_carrier_protocols\raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680`
- BUT target: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed`
- Candidate policy: good `4`, medium `5`, bad `12` per anchor, sparse tails x1.5.

## v101/v106/v107/v108 class median comparison

| version | class_name | median_rbf_mmd | median_sliced_wasserstein | median_quantile_loss | median_domain_auc | median_pca_density_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| v101 | bad | 0.6367 | 6.4310 | 4.0061 | 1.0000 | 0.0000 |
| v101 | good | 0.3410 | 1.1757 | 1.3051 | 1.0000 | 0.1284 |
| v101 | medium | 0.3055 | 0.8370 | 0.6612 | 1.0000 | 0.1892 |
| v106 | bad | 1.3184 | 9.5381 | 6.2697 | 1.0000 | 0.0000 |
| v106 | good | 0.4528 | 0.9586 | 0.9398 | 1.0000 | 0.2677 |
| v106 | medium | 0.3167 | 0.6753 | 0.5887 | 1.0000 | 0.3524 |
| v107 | bad | 1.3164 | 9.4650 | 6.2144 | 1.0000 | 0.0000 |
| v107 | good | 0.4487 | 0.9147 | 0.6931 | 1.0000 | 0.3347 |
| v107 | medium | 0.3094 | 0.6442 | 0.5489 | 1.0000 | 0.3533 |
| v108 | bad | 0.7121 | 9.3609 | 6.3160 | 1.0000 | 0.0000 |
| v108 | good | 0.3781 | 1.3597 | 1.3768 | 1.0000 | 0.1383 |
| v108 | medium | 0.2833 | 0.7847 | 0.6377 | 1.0000 | 0.1586 |

## v108 subtype summary

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 0.7067 | 9.3596 | 6.3160 | 1.0000 | 0.0000 |
| good | 0.3832 | 1.3190 | 1.3768 | 1.0000 | 0.1383 |
| medium | 0.2833 | 0.8526 | 0.6377 | 1.0000 | 0.1586 |

## v108 component summary

| class_name | component | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_occupancy_recall |
| --- | --- | --- | --- | --- | --- | --- |
| bad | core | 0.9132 | 9.8547 | 7.2424 | 1.0000 | 0.0294 |
| bad | left_ridge | 1.2430 | 11.6332 | 7.9723 | 1.0000 | 0.0312 |
| bad | lower_shell | 0.8778 | 9.8005 | 7.0200 | 1.0000 | 0.0286 |
| bad | right_ridge | 0.9817 | 11.5332 | 7.5463 | 1.0000 | 0.0308 |
| bad | sparse_tail | 0.9284 | 8.2538 | 5.1963 | 1.0000 | 0.0208 |
| bad | upper_shell | 0.8242 | 11.3839 | 7.7049 | 1.0000 | 0.0294 |
| good | core | 0.4681 | 1067.1967 | 482.1174 | 1.0000 | 0.1944 |
| good | left_ridge | 0.4578 | 902.0054 | 429.3737 | 1.0000 | 0.1389 |
| good | lower_shell | 0.4648 | 2044.5476 | 664.9069 | 1.0000 | 0.1667 |
| good | right_ridge | 0.4382 | 373.5600 | 170.9046 | 1.0000 | 0.1111 |
| good | sparse_tail | 0.3534 | 0.8516 | 0.7381 | 1.0000 | 0.2917 |
| good | upper_shell | 0.4539 | 623.7833 | 294.0206 | 1.0000 | 0.1389 |
| medium | core | 0.4040 | 152.4938 | 78.0053 | 1.0000 | 0.2778 |
| medium | left_ridge | 0.4017 | 1.3328 | 1.0847 | 1.0000 | 0.1250 |
| medium | lower_shell | 0.3189 | 1.1639 | 0.9970 | 1.0000 | 0.3929 |
| medium | right_ridge | 0.4645 | 294.1794 | 132.8179 | 1.0000 | 0.1389 |
| medium | sparse_tail | 0.2699 | 0.6516 | 0.6353 | 1.0000 | 0.2957 |
| medium | upper_shell | 0.4375 | 1.8279 | 1.5854 | 1.0000 | 0.2639 |

## Figures

- `v108_anchor_direct_shared_pca.png`
- `v108_anchor_direct_component_shared_pca.png`
- `v108_anchor_direct_key_feature_cdf_overlay.png`
- `v108_anchor_direct_rbf_mmd_heatmap.png`
- `v108_anchor_direct_good_subtype_waveforms.png`
- `v108_anchor_direct_medium_subtype_waveforms.png`
- `v108_anchor_direct_bad_subtype_waveforms.png`