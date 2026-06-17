# Waveform SQI Module Probe

Created: 2026-06-17 17:39:36

This is a module-first diagnostic. It uses PTB synthetic train only for fitting diagnostic probes; original BUT remains report-only.

## Feature Taxonomy

- `atlas_label_geometry_not_waveform_fact`: 5 features
- `stable_waveform_sqi_morph_rr`: 39 features
- `weak_target_distribution_proxy`: 3 features

## Module Feature Dimensions

- `qrs_rr`: 252 waveform-derived dimensions
- `baseline_band`: 144 waveform-derived dimensions
- `local_artifact`: 168 waveform-derived dimensions
- `morph_detail`: 93 waveform-derived dimensions
- `stable_all`: 657 waveform-derived dimensions
- `qrs_stress_v5_all`: 510 waveform-derived dimensions

## Best Original-Test Classification Probes

| model | group | acc | good | medium | bad |
|---|---|---:|---:|---:|---:|
| `extratrees` | `qrs_rr` | 0.790020 | 0.912 | 0.736 | 0.290 |
| `extratrees` | `stable_all` | 0.789076 | 0.914 | 0.732 | 0.290 |
| `hgb` | `qrs_rr` | 0.788369 | 0.916 | 0.729 | 0.290 |
| `extratrees` | `qrs_stress_v5_all` | 0.781880 | 0.897 | 0.733 | 0.290 |
| `extratrees` | `morph_detail` | 0.771617 | 0.898 | 0.712 | 0.290 |
| `extratrees` | `baseline_band` | 0.770674 | 0.895 | 0.713 | 0.290 |
| `hgb` | `stable_all` | 0.769022 | 0.927 | 0.683 | 0.290 |
| `extratrees` | `local_artifact` | 0.768078 | 0.910 | 0.695 | 0.290 |
| `hgb` | `morph_detail` | 0.761826 | 0.889 | 0.701 | 0.290 |
| `hgb` | `qrs_stress_v5_all` | 0.746726 | 0.920 | 0.647 | 0.290 |
| `hgb` | `local_artifact` | 0.729149 | 0.927 | 0.607 | 0.290 |
| `hgb` | `baseline_band` | 0.690574 | 0.921 | 0.537 | 0.302 |

## Bad Stress Buckets

| bucket | model | group | acc / bad recall |
|---|---|---|---:|
| `bad_core_nearboundary` | `hgb` | `qrs_rr` | 1.000000 |
| `bad_core_nearboundary` | `extratrees` | `qrs_rr` | 1.000000 |
| `bad_core_nearboundary` | `extratrees` | `baseline_band` | 1.000000 |
| `bad_core_nearboundary` | `hgb` | `local_artifact` | 1.000000 |
| `bad_core_nearboundary` | `extratrees` | `local_artifact` | 1.000000 |
| `bad_core_nearboundary` | `hgb` | `morph_detail` | 1.000000 |
| `bad_core_nearboundary` | `extratrees` | `morph_detail` | 1.000000 |
| `bad_core_nearboundary` | `hgb` | `stable_all` | 1.000000 |
| `bad_outlier_stress` | `hgb` | `baseline_band` | 0.020548 |
| `bad_outlier_stress` | `hgb` | `qrs_rr` | 0.000000 |
| `bad_outlier_stress` | `extratrees` | `qrs_rr` | 0.000000 |
| `bad_outlier_stress` | `extratrees` | `baseline_band` | 0.000000 |
| `bad_outlier_stress` | `hgb` | `local_artifact` | 0.000000 |
| `bad_outlier_stress` | `extratrees` | `local_artifact` | 0.000000 |
| `bad_outlier_stress` | `hgb` | `morph_detail` | 0.000000 |
| `bad_outlier_stress` | `extratrees` | `morph_detail` | 0.000000 |

## Feature Recovery Snapshot

_Feature recovery skipped for this run._

## Interpretation

- If `qrs_rr` does not recover `qrs_visibility` / `detector_agreement`, the next Transformer needs true event/RR sequence tokens rather than global QRS summaries.
- If `baseline_band` does not recover `baseline_step` / `sqi_basSQI`, fixed filterbank/baseline token streams should become first-class inputs.
- `boundary_confidence`, `knn_label_purity`, and much of `pc2` are atlas/label geometry diagnostics, not single-waveform facts; poor recovery there should not drive architecture capacity.
- The formal model must still be waveform-only. These probes are a map for what the waveform encoder must learn, not a final classifier claim.
