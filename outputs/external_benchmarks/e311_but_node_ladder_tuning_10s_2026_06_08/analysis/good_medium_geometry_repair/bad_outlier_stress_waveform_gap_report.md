# Bad Outlier Stress Waveform Gap

Computed on the same `robust3` waveform channels used by waveform-only models.

## Counts

- `synthetic_bad_all`: 1816
- `synthetic_medium_all`: 7867
- `synthetic_good_all`: 3216
- `original_test_bad_core`: 119
- `original_test_bad_outlier_stress`: 292
- `original_test_good`: 3640
- `original_test_medium`: 4426

## Top Stress-vs-Synthetic-Bad Gaps

| feature | KS | stress median | synth bad median | core bad median | stress-synth IQR units |
|---|---:|---:|---:|---:|---:|
| wf_diff_flat_015 | 1.000 | 0.3823 | 0.0104 | 0.0048 | 77.40 |
| wf_z_flat_015 | 1.000 | 0.3082 | 0.0120 | 0.0080 | 61.65 |
| wf_diff_contact_015 | 1.000 | 0.3088 | 0.0128 | 0.0088 | 61.65 |
| wf_diff_lowamp_050 | 1.000 | 0.6272 | 0.0408 | 0.0280 | 56.38 |
| wf_z_zcr | 0.992 | 0.0320 | 0.2986 | 0.4916 | -2.58 |
| wf_diff_diff_abs | 0.977 | 0.1396 | 0.8111 | 1.9424 | -2.00 |
| wf_diff_mean_abs | 0.977 | 0.1471 | 0.7555 | 1.1166 | -3.30 |
| wf_z_diff_abs | 0.977 | 0.1473 | 0.7561 | 1.1174 | -3.30 |
| wf_baseline_ptp | 0.975 | 4.3969 | 0.2776 | 0.6715 | 37.01 |
| wf_baseline_mean_abs | 0.975 | 0.7372 | 0.0416 | 0.1041 | 35.38 |
| wf_baseline_std | 0.974 | 0.9568 | 0.0484 | 0.1282 | 38.10 |
| wf_baseline_rms | 0.974 | 0.9739 | 0.0519 | 0.1301 | 37.97 |

## Takeaway

Original bad outlier stress is not solved by more ordinary bad class weight. It occupies a different robust-waveform-stat shell, so the next experiment should either synthesize that shell explicitly or keep it as an external stress bucket.
