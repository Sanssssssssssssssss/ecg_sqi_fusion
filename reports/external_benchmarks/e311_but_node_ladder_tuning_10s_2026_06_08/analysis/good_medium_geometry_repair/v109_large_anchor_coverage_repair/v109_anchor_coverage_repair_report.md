# v109 Anchor Coverage Repair

Generated: 2026-06-25 03:04:37

## What changed

- Reuses v108 anchor-direct protocol and BUT empirical components.
- Replaces the single bad transform with an 8-family bad mechanism candidate pool.
- Adds light component-tail jitter for good/medium so sparse shells are not always pulled back to the center.
- BUT waveforms are still never copied; BUT train+val is used only as a feature/component target.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v109_large_anchor_coverage_repair_v101x4_cpa8m10b48_s20260738`
- Rows: `13968`
- Candidate policy: good `8`, medium `10`, bad `48` per anchor.

## Class-Level Median Metrics

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 0.7287 | 13.1287 | 10.1763 | 1.0000 | 0.0000 |
| good | 0.3655 | 1.1592 | 1.1591 | 1.0000 | 0.2835 |
| medium | 0.2456 | 0.6417 | 0.5410 | 1.0000 | 0.3584 |

## Figures

- `v108_anchor_direct_shared_pca.png`
- `v108_anchor_direct_key_feature_cdf_overlay.png`
- `v108_anchor_direct_good_subtype_waveforms.png`
- `v108_anchor_direct_medium_subtype_waveforms.png`
- `v108_anchor_direct_bad_subtype_waveforms.png`

## Interpretation

This is a coverage-repair smoke/versioned build. It should be accepted only if the shared PCA shows PTB covering the BUT shell and the class/subtype CDF gaps fall relative to v108.