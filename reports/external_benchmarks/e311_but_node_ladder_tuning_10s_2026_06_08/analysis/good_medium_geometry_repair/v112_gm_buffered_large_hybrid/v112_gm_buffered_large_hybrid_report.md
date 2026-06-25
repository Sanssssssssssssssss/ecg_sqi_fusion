# v112 GM-Buffered Large Hybrid

Generated: 2026-06-25 04:02:20

## Logic

- Input is v111 large hybrid.
- Bad rows are kept unchanged.
- Good/medium rows are removed only when a BUT train+val good-vs-medium discriminator strongly predicts the opposite class.
- The discriminator is a generation/audit filter only; it is not a model input and not a final classifier.

## Protocol

- Protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v81_feature_transport\protocol_v112_gm_buffered_large_hybrid_s20260741`
- Rows before: `13968`
- Rows after: `12074`
- Removed rows: `1894`
- Good/medium contradiction threshold: `0.85`
- BUT train+val GM audit AUC: `0.966959`

## Class-Level Median Metrics

| class_name | rbf_mmd | sliced_wasserstein | quantile_loss | domain_auc | pca_density_overlap |
| --- | --- | --- | --- | --- | --- |
| bad | 1.3561 | 8.7570 | 6.2280 | 1.0000 | 0.0000 |
| good | 0.3868 | 0.9999 | 0.9760 | 1.0000 | 0.2686 |
| medium | 0.2449 | 0.6327 | 0.5405 | 1.0000 | 0.3464 |

## Figures

- `v112_gm_buffered_shared_pca.png`
- `v112_gm_buffered_key_feature_cdf_overlay.png`
- `v112_gm_buffered_good_subtype_waveforms.png`
- `v112_gm_buffered_medium_subtype_waveforms.png`
- `v112_gm_buffered_bad_subtype_waveforms.png`