# v108 Anchor-Direct Component Transport

Generated: 2026-06-25 00:42:57

## Why this exists

v107 reused the v104 synthetic candidate pool, so it inherited the old round/blob geometry. v108 samples BUT train+val anchors directly and generates PTB-carrier candidates for each anchor.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v108_smoke_smoothlowqrs_component_transport_v101x1_cpa4m5b12_s20260735`
- Rows: `3492`
- Carrier protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\raw_ptbxl_carrier_protocols\raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680`
- BUT target: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed`
- Candidate policy: good `4`, medium `5`, bad `12` per anchor, sparse tails x1.5.

## v101/v106/v107/v108 class median comparison

| version | class_name | median_rbf_mmd | median_sliced_wasserstein | median_quantile_loss | median_domain_auc | median_pca_density_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| v101 | bad | 0.6367 | 7.3510 | 4.0061 | 1.0000 | 0.0000 |
| v101 | good | 0.3439 | 1.2435 | 1.3051 | 1.0000 | 0.1284 |
| v101 | medium | 0.3054 | 0.8132 | 0.6612 | 1.0000 | 0.1892 |
| v106 | bad | 1.3146 | 9.5569 | 6.2697 | 1.0000 | 0.0000 |
| v106 | good | 0.4523 | 0.9681 | 0.9398 | 1.0000 | 0.2677 |
| v106 | medium | 0.3154 | 0.7348 | 0.5887 | 1.0000 | 0.3524 |
| v107 | bad | 1.3140 | 9.5423 | 6.2144 | 1.0000 | 0.0000 |
| v107 | good | 0.4478 | 0.8937 | 0.6931 | 1.0000 | 0.3347 |
| v107 | medium | 0.3101 | 0.7213 | 0.5489 | 1.0000 | 0.3533 |
| v108 | bad | 1.0580 | 39.2584 | 17.0820 | 1.0000 | 0.0000 |
| v108 | good | 0.3823 | 1.2845 | 1.2867 | 1.0000 | 0.1365 |
| v108 | medium | 0.2835 | 0.7297 | 0.6178 | 1.0000 | 0.1710 |

## v108 subtype summary

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 1.0580 | 36.3708 | 17.0820 | 1.0000 | 0.0000 |
| good | 0.3820 | 1.3558 | 1.2867 | 1.0000 | 0.1365 |
| medium | 0.2837 | 0.8010 | 0.6178 | 1.0000 | 0.1710 |

## v108 component summary

| class_name | component | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_occupancy_recall |
| --- | --- | --- | --- | --- | --- | --- |
| bad | core | 1.3687 | 36.3861 | 18.1501 | 1.0000 | 0.0278 |
| bad | left_ridge | 1.3016 | 39.7906 | 19.5125 | 1.0000 | 0.0385 |
| bad | lower_shell | 1.2655 | 39.2266 | 16.3663 | 1.0000 | 0.0286 |
| bad | right_ridge | 1.2719 | 48.3606 | 20.5697 | 1.0000 | 0.0518 |
| bad | sparse_tail | 0.9962 | 37.1165 | 16.8365 | 1.0000 | 0.0446 |
| bad | upper_shell | 1.3182 | 42.3396 | 19.8206 | 1.0000 | 0.0294 |
| good | core | 0.4708 | 799.4821 | 439.4837 | 1.0000 | 0.1944 |
| good | left_ridge | 0.4717 | 701.2047 | 312.0762 | 1.0000 | 0.1111 |
| good | lower_shell | 0.4675 | 3280.8581 | 758.6402 | 1.0000 | 0.1429 |
| good | right_ridge | 0.4384 | 511.9607 | 236.8273 | 1.0000 | 0.1111 |
| good | sparse_tail | 0.3745 | 0.9794 | 0.7286 | 1.0000 | 0.2414 |
| good | upper_shell | 0.4409 | 412.1769 | 185.9970 | 1.0000 | 0.1667 |
| medium | core | 0.4207 | 152.7703 | 77.7698 | 1.0000 | 0.3333 |
| medium | left_ridge | 0.4107 | 1.3372 | 1.2064 | 1.0000 | 0.2315 |
| medium | lower_shell | 0.3284 | 0.9684 | 0.8995 | 1.0000 | 0.3851 |
| medium | right_ridge | 0.4849 | 514.2170 | 213.4598 | 1.0000 | 0.1250 |
| medium | sparse_tail | 0.2701 | 0.6697 | 0.5845 | 1.0000 | 0.3191 |
| medium | upper_shell | 0.4178 | 1.7905 | 1.5282 | 1.0000 | 0.2500 |

## Figures

- `v108_anchor_direct_shared_pca.png`
- `v108_anchor_direct_component_shared_pca.png`
- `v108_anchor_direct_key_feature_cdf_overlay.png`
- `v108_anchor_direct_rbf_mmd_heatmap.png`
- `v108_anchor_direct_good_subtype_waveforms.png`
- `v108_anchor_direct_medium_subtype_waveforms.png`
- `v108_anchor_direct_bad_subtype_waveforms.png`