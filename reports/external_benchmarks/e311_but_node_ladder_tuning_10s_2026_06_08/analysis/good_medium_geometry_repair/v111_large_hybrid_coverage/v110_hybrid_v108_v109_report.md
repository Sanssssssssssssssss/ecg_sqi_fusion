# v110 Hybrid v108/v109 Distribution Audit

Generated: 2026-06-25 03:48:55

## Logic

- good/medium use v109 coverage-repair because it improves tail coverage over v108 smoke.
- bad_contact_reset_flatline and bad_low_qrs_visibility use v109.
- dense/detector/baseline/highfreq/other bad use v108 because v109 made their Wasserstein/quantile gaps worse.
- This is an audit protocol to confirm subtype-wise mechanism selection, not a trained model result.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v111_large_hybrid_coverage_v108legacy_v109coverage_s20260740`
- Rows: `13968`
- v108 rows: `4288`
- v109 rows: `9680`

## Class-Level Median Metrics

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 1.3303 | 9.0639 | 6.2280 | 1.0000 | 0.0000 |
| good | 0.3678 | 1.1476 | 1.1591 | 1.0000 | 0.2835 |
| medium | 0.2454 | 0.6224 | 0.5410 | 1.0000 | 0.3584 |

## Figures

- `v110_hybrid_shared_pca.png`
- `v110_hybrid_key_feature_cdf_overlay.png`
- `v110_hybrid_good_subtype_waveforms.png`
- `v110_hybrid_medium_subtype_waveforms.png`
- `v110_hybrid_bad_subtype_waveforms.png`
