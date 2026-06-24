# v88-v92 Distribution Matching Decision

## Scope

This note summarizes the latest PTB synthetic -> BUT keep-outlier subtype distribution matching loop. The acceptance target is visual similarity plus 47 waveform/SQI feature distribution similarity. No model training was launched in this loop.

## Protocols

- v87: `protocol_v87_consistent_primitive_transport_pc1500_cpa8_s20260687`
- v88: `protocol_v88_eventtrain_highfreq_transport_pc1500_cpa8_s20260688`
- v89: `protocol_v89_midband_dense_transport_pc1500_cpa8_s20260689`
- v90: `protocol_v90_midband_dense_repair_transport_pc1500_cpa8_s20260690`
- v91: `protocol_v91_dense_derivative_repair_transport_pc1500_cpa8_s20260691`
- v92: `protocol_v92_rawdiff_target_transport_pc1500_cpa8_s20260692`

## Class-Level Metrics

Lower is better for RBF-MMD / sliced Wasserstein / quantile loss / domain AUC. Higher is better for PCA density overlap.

| version | class | RBF-MMD | sliced Wasserstein | quantile loss | domain AUC | PCA overlap |
|---|---:|---:|---:|---:|---:|---:|
| v87 | bad | 0.8589 | 10.6805 | 6.1511 | 0.9963 | 0.0167 |
| v88 | bad | 0.8755 | 7.2697 | 5.1104 | 0.9978 | 0.0286 |
| v89 | bad | 0.8893 | 14.1553 | 8.8135 | 1.0000 | 0.0210 |
| v90 | bad | 0.8707 | 7.3675 | 5.3532 | 1.0000 | 0.0277 |
| v91 | bad | 0.8743 | 8.1088 | 5.5783 | 1.0000 | 0.0183 |
| v92 | bad | 0.8778 | 8.1891 | 5.5528 | 0.9978 | 0.0226 |
| v88 | good | 0.3944 | 1.4417 | 1.3070 | 1.0000 | 0.3095 |
| v90 | medium | 0.2738 | 1.7435 | 0.6796 | 1.0000 | 0.3439 |
| v92 | medium | 0.2728 | 2.4343 | 0.7010 | 1.0000 | 0.3549 |

## Decision

v88 remains the best temporary bad-distribution candidate by sliced Wasserstein, quantile loss, and PCA density overlap. v90-v92 improved some medium-hard and high-frequency details, but they did not improve bad overall. v89-v92 should not be used as the main synthetic protocol.

The main remaining bad mismatch is not fixed by tuning the random generator:

- dense/detector/other bad still have near-zero PCA overlap.
- `detector_agreement` is still separable.
- v88 has too much `band_30_45`; v90-v92 over-correct and lose `raw_diff_abs_p95` / `non_qrs_diff_p95`.
- v92 proves that simply passing `raw_diff_abs_p95` as the target does not make selected waveforms match it.

## Next Step

Stop tuning fixed random bad generators. The next distribution-matching step should be a per-anchor black-box waveform-parameter optimization for bad subtypes:

1. Start from v88-style event-train morphology.
2. For each BUT bad anchor, optimize a small parameter vector: event width, event jitter, edge amplitude, midband gain, highband gain, baseline gain, contact severity.
3. Score each candidate directly on the recomputed 47-feature distance plus prototype waveform distance.
4. Keep subtype-balanced herding after the per-anchor optimization.

This is a support-set expansion, not another class-weight or model-training experiment.

## Key Outputs

- v88 report: `reports/.../good_medium_geometry_repair/v88_distribution_transport/v81_distribution_transport_report.md`
- v88 PCA: `reports/.../good_medium_geometry_repair/v88_distribution_transport/v88_shared_pca.png`
- v88 bad waveforms: `reports/.../good_medium_geometry_repair/v88_distribution_transport/v88_bad_subtype_waveforms.png`
- v92 report: `reports/.../good_medium_geometry_repair/v92_distribution_transport/v81_distribution_transport_report.md`
- v92 PCA: `reports/.../good_medium_geometry_repair/v92_distribution_transport/v92_shared_pca.png`
- v92 bad waveforms: `reports/.../good_medium_geometry_repair/v92_distribution_transport/v92_bad_subtype_waveforms.png`
