# Current Waveform Transformer Iteration Compare

Selection remains synthetic/node only; BUT rows below are report-only.

## Metric Snapshot

### synthetic_test
| candidate | acc | good | medium | bad | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p20_sqiquery_primctx_v5_light` | 0.995900 | 0.994 | 1.000 | 0.979 | 2 | 0 | 5 |
| `p20_sqiquery_primctx_v5_badguard` | 0.995387 | 0.994 | 0.999 | 0.979 | 3 | 1 | 5 |

### original_test_all_10s+
| candidate | acc | good | medium | bad | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p20_sqiquery_primctx_v5_light` | 0.808069 | 0.894 | 0.784 | 0.309 | 375 | 897 | 68 |
| `p20_sqiquery_primctx_v5_badguard` | 0.794857 | 0.896 | 0.757 | 0.304 | 376 | 1048 | 65 |

### original_all_10s+
| candidate | acc | good | medium | bad | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p20_sqiquery_primctx_v5_light` | 0.850983 | 0.820 | 0.859 | 0.935 | 3043 | 1423 | 128 |
| `p20_sqiquery_primctx_v5_badguard` | 0.834446 | 0.785 | 0.864 | 0.934 | 3657 | 1404 | 127 |

### bad_core_nearboundary
| candidate | acc | good | medium | bad | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p20_sqiquery_primctx_v5_badguard` | 0.991597 | 0.000 | 0.000 | 0.992 | 0 | 0 | 1 |
| `p20_sqiquery_primctx_v5_light` | 0.983193 | 0.000 | 0.000 | 0.983 | 0 | 0 | 2 |

### bad_outlier_stress
| candidate | acc | good | medium | bad | g->m | m->g | b->m |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p20_sqiquery_primctx_v5_light` | 0.034247 | 0.000 | 0.000 | 0.034 | 0 | 0 | 66 |
| `p20_sqiquery_primctx_v5_badguard` | 0.023973 | 0.000 | 0.000 | 0.024 | 0 | 0 | 64 |

## Feature Recovery Correlation On Original All

| candidate | qrs_visibility | detector_agreement | baseline_step | flatline_ratio | sqi_basSQI | non_qrs_diff_p95 | pca_margin | pc1 | pc2 | boundary_confidence | knn_label_purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p20_sqiquery_primctx_v5_badguard` | 0.317 | 0.314 | 0.389 | 0.847 | 0.155 | 0.948 | 0.881 | 0.970 | -0.185 | 0.158 | 0.315 |
| `p20_sqiquery_primctx_v5_light` | 0.394 | 0.310 | 0.423 | 0.821 | 0.185 | 0.941 | 0.878 | 0.961 | -0.065 | 0.167 | 0.316 |
| `predtop20_eventqrs_impulsebad_dual_p20_qrsheavy` | 0.427 | 0.319 | 0.447 | 0.821 | 0.264 | 0.940 | 0.888 | 0.965 | -0.009 | 0.208 | 0.328 |
| `predtop20_qrsbank_impulsebad_dual_p20` | 0.367 | 0.314 | 0.394 | 0.824 | 0.153 | 0.944 | 0.883 | 0.960 | -0.120 | 0.149 | 0.293 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20` | 0.397 | 0.292 | 0.440 | 0.827 | 0.167 | 0.941 | 0.881 | 0.964 | -0.131 | 0.193 | 0.358 |
| `wavecomp_sqiquery_p20` | 0.313 | 0.297 | 0.306 | 0.829 | 0.093 | 0.929 | 0.860 | 0.952 | -0.208 | 0.123 | 0.290 |
| `wavecomp_sqiquery_v5_p20_badguard` | 0.360 | 0.304 | 0.346 | 0.825 | 0.110 | 0.931 | 0.861 | 0.956 | -0.161 | 0.124 | 0.303 |
| `waveprimtoken_v5_p20` | 0.221 | 0.320 | 0.355 | 0.855 | 0.165 | 0.954 | 0.880 | 0.972 | -0.230 | 0.151 | 0.292 |
| `waveprimtoken_v5_p20_badguard` | 0.337 | 0.296 | 0.316 | 0.805 | 0.084 | 0.935 | 0.863 | 0.952 | -0.206 | 0.112 | 0.256 |

## Decision Notes

- Full primitive-token encoder improved controlled bad stress in some cases but hurt good/medium transfer; it should not replace patch morphology.
- Removing atlas-like geometry targets from the SQI-query p20 backbone made synthetic stay high but reduced BUT original_test, so target-geometry supervision is acting as useful training regularization even when not used as raw input.
- Light primitive-context fusion improves original_all but not original_test; the blocker is the held-out test record/domain slice rather than broad BUT capacity.
- The best formal waveform-only original_test candidate in this table remains p20 SQI-query patch Transformer at about 0.822 accuracy.