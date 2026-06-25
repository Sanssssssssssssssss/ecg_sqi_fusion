# v110 Hybrid v108/v109 Distribution Audit

Generated: 2026-06-25 02:19:25

## Logic

- good/medium use v109 coverage-repair because it improves tail coverage over v108 smoke.
- bad_contact_reset_flatline and bad_low_qrs_visibility use v109.
- dense/detector/baseline/highfreq/other bad use v108 because v109 made their Wasserstein/quantile gaps worse.
- This is an audit protocol to confirm subtype-wise mechanism selection, not a trained model result.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v110_hybrid_v108_v109_s20260737`
- Rows: `3492`
- v108 rows: `1072`
- v109 rows: `2420`

## Class-Level Median Metrics

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 0.7072 | 9.1606 | 6.3160 | 1.0000 | 0.0000 |
| good | 0.3589 | 1.1731 | 0.9885 | 1.0000 | 0.1750 |
| medium | 0.2642 | 0.7870 | 0.5763 | 1.0000 | 0.2165 |

## Figures

- `v110_hybrid_shared_pca.png`
- `v110_hybrid_key_feature_cdf_overlay.png`
- `v110_hybrid_good_subtype_waveforms.png`
- `v110_hybrid_medium_subtype_waveforms.png`
- `v110_hybrid_bad_subtype_waveforms.png`
