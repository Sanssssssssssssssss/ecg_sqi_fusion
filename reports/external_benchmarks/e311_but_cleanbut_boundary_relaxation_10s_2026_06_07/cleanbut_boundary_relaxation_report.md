# CleanBUT Filtered Boundary Relaxation

This is a filtered diagnostic benchmark, not the formal BUT original test. Selection uses CleanBUT atlas label-confidence features only; model predictions are used only after the subset is fixed.

## Main Finding

- Widest raw subset at/above target: `cc_bad_narrow_oscillatory_core_quiet_core_tight_sec0p00_c_1b585bae6df3`
- Selected per class: `4800`; total `14400`
- Filtered acc: `0.9553`; macro-F1 `0.9553`
- Recalls good/medium/bad: `0.913/0.953/1.000`
- Ambiguous fraction: `0.553`

## Top Raw Operating Points

| rank | variant | n/class | acc | macro-F1 | recalls good/medium/bad | ambiguous fraction | original BUT macro |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | `cc_bad_narrow_oscillatory_core_quiet_core_tight_sec0p00_c_1b585bae6df3` | 4800 | 0.9553 | 0.9553 | 0.913/0.953/1.000 | 0.553 | 0.4604 |
| 2 | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d7274ab2f04d` | 3000 | 0.9534 | 0.9534 | 0.984/0.884/0.992 | 0.315 | 0.4314 |
| 3 | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_d8774356567f` | 2500 | 0.9651 | 0.9650 | 0.987/0.908/1.000 | 0.244 | 0.4173 |
| 4 | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_238aae411c7e` | 2500 | 0.9524 | 0.9522 | 0.994/0.863/1.000 | 0.244 | 0.4249 |
| 5 | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_6bb3561abb9a` | 2500 | 0.9505 | 0.9503 | 0.861/0.990/1.000 | 0.244 | 0.5624 |
| 6 | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_adf610ec036d` | 1198 | 0.9569 | 0.9567 | 1.000/0.871/1.000 | 0.001 | 0.4195 |
| 7 | `cc_bad_1530_spike_core_quiet_core_tight_sec0p00_cw1p00_1p_8e39b9c324f4` | 500 | 0.9527 | 0.9524 | 1.000/0.858/1.000 | 0.000 | 0.6594 |
| 8 | `cc_bad_1530_locked_lowpc2_core_quiet_core_tight_sec0p00_c_9683ffc33626` | 200 | 0.9500 | 0.9497 | 1.000/0.850/1.000 | 0.000 | 0.4999 |

## Outputs

- `boundary_relaxation_metrics.csv`: every model, raw/calibrated mode, and boundary level.
- `boundary_relaxation_best_at_0p95.csv`: widest subset per model/mode at the target accuracy.
- `figures/boundary_relaxation_curve.png`: accuracy as the boundary is relaxed.
- `figures/best_0p95_boundary_pca.png`: selected subset, ambiguity tiers, and current boundary shell.
- `figures/best_0p95_boundary_composition.png`: clean vs ambiguous composition.
- `boundary_waveform_galleries/`: low-confidence included samples at the selected boundary.
