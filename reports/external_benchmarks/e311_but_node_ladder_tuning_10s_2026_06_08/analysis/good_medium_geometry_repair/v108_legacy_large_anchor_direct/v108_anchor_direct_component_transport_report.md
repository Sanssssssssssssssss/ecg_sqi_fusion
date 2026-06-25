# v108 Anchor-Direct Component Transport

Generated: 2026-06-25 03:37:39

## Why this exists

v107 reused the v104 synthetic candidate pool, so it inherited the old round/blob geometry. v108 samples BUT train+val anchors directly and generates PTB-carrier candidates for each anchor.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v108_legacy_large_anchor_direct_v101x4_cpa8m10b24_s20260739`
- Rows: `13968`
- Carrier protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\raw_ptbxl_carrier_protocols\raw_ptbxl_lead1_clean_noise_notes_max9000_seed20260680`
- BUT target: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed`
- Candidate policy: good `8`, medium `10`, bad `24` per anchor, sparse tails x1.5.

## v101/v106/v107/v108 class median comparison

| version | class_name | median_rbf_mmd | median_sliced_wasserstein | median_quantile_loss | median_domain_auc | median_pca_density_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| v101 | bad | 0.6367 | 6.8596 | 4.0061 | 1.0000 | 0.0000 |
| v101 | good | 0.3423 | 1.1678 | 1.3051 | 1.0000 | 0.1284 |
| v101 | medium | 0.3058 | 0.8474 | 0.6612 | 1.0000 | 0.1892 |
| v106 | bad | 1.3160 | 9.0691 | 6.2697 | 1.0000 | 0.0000 |
| v106 | good | 0.4531 | 1.0260 | 0.9398 | 1.0000 | 0.2677 |
| v106 | medium | 0.3158 | 0.7375 | 0.5887 | 1.0000 | 0.3524 |
| v107 | bad | 1.3161 | 9.1383 | 6.2144 | 1.0000 | 0.0000 |
| v107 | good | 0.4481 | 0.9283 | 0.6931 | 1.0000 | 0.3347 |
| v107 | medium | 0.3100 | 0.7246 | 0.5489 | 1.0000 | 0.3533 |
| v108 | bad | 1.3592 | 9.1576 | 6.2280 | 1.0000 | 0.0000 |
| v108 | good | 0.3685 | 1.0509 | 0.9185 | 1.0000 | 0.2784 |
| v108 | medium | 0.2543 | 0.6399 | 0.5343 | 1.0000 | 0.3574 |

## v108 subtype summary

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 1.3556 | 9.2738 | 6.2280 | 1.0000 | 0.0000 |
| good | 0.3678 | 1.0264 | 0.9185 | 1.0000 | 0.2784 |
| medium | 0.2537 | 0.6364 | 0.5343 | 1.0000 | 0.3574 |

## v108 component summary

| class_name | component | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_occupancy_recall |
| --- | --- | --- | --- | --- | --- | --- |
| bad | core | 1.3270 | 10.1673 | 7.1343 | 1.0000 | 0.0294 |
| bad | left_ridge | 1.3394 | 10.7352 | 7.7066 | 1.0000 | 0.0312 |
| bad | lower_shell | 1.2872 | 9.0145 | 6.8684 | 1.0000 | 0.0667 |
| bad | right_ridge | 1.3510 | 11.4853 | 7.3936 | 1.0000 | 0.0308 |
| bad | sparse_tail | 1.2257 | 7.7131 | 5.0766 | 1.0000 | 0.1101 |
| bad | upper_shell | 1.3358 | 11.4376 | 7.6028 | 1.0000 | 0.0303 |
| good | core | 0.4913 | 1.7786 | 1.3235 | 1.0000 | 0.2222 |
| good | left_ridge | 0.5088 | 116.5755 | 70.2102 | 1.0000 | 0.1719 |
| good | lower_shell | 0.5388 | 489.4069 | 177.3020 | 1.0000 | 0.1714 |
| good | right_ridge | 0.5732 | 35.8195 | 14.5548 | 1.0000 | 0.1667 |
| good | sparse_tail | 0.3672 | 0.7386 | 0.6188 | 1.0000 | 0.2857 |
| good | upper_shell | 0.4045 | 1.3323 | 0.9576 | 1.0000 | 0.2222 |
| medium | core | 0.3314 | 0.8650 | 0.7572 | 1.0000 | 0.2778 |
| medium | left_ridge | 0.4092 | 1.2063 | 0.9878 | 1.0000 | 0.3351 |
| medium | lower_shell | 0.2920 | 0.8672 | 0.7868 | 1.0000 | 0.4000 |
| medium | right_ridge | 0.4684 | 28.8535 | 19.2727 | 1.0000 | 0.1806 |
| medium | sparse_tail | 0.2318 | 0.5922 | 0.5025 | 1.0000 | 0.4972 |
| medium | upper_shell | 0.3431 | 0.9502 | 0.7901 | 1.0000 | 0.3301 |

## Figures

- `v108_anchor_direct_shared_pca.png`
- `v108_anchor_direct_component_shared_pca.png`
- `v108_anchor_direct_key_feature_cdf_overlay.png`
- `v108_anchor_direct_rbf_mmd_heatmap.png`
- `v108_anchor_direct_good_subtype_waveforms.png`
- `v108_anchor_direct_medium_subtype_waveforms.png`
- `v108_anchor_direct_bad_subtype_waveforms.png`