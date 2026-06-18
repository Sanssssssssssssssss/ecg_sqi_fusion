# N7110 QRS-Low Single-Checkpoint Conversion

This report records the attempt to convert the N7110 qrs-low transparent rule insight into a normal single checkpoint.

## Result

The conversion did not promote. The best normal N7110 checkpoint remains:

- `nl_n7110_gm_trim_bad_geom_directrule_n7100base_g003_m008__69ab5b71cf7d` / raw
- acc `0.949628`, macro-F1 `0.954632`
- good/medium/bad recall `0.960338 / 0.926864 / 0.970617`

The transparent qrs-low rule-mode remains the only N7110 artifact above gate:

- `rule_n7110_qrslow_gate_trainval_base_g003_alt_g004_pmed0005`
- acc `0.952906`, macro-F1 `0.957450`
- good/medium/bad recall `0.960338 / 0.935302 / 0.970617`
- fixes 63 medium->good errors with 0 good lost

## Single-Checkpoint Attempts

The new synthetic-only picker targeted very-low-qrs medium hard negatives. It did not use node test rows or original BUT.

| Variant | Best mode | acc | good | medium | bad | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `geom_qrslowmed_n7110best_g000_m003...` | medium_guarded_pmed0005 | 0.931818 | 0.990577 | 0.850774 | 0.970617 | too good-heavy |
| `geom_qrslowmed_n7110best_g000_m005...` | calibrated | 0.821187 | 0.586217 | 0.970323 | 0.970617 | too medium-heavy |
| `geom_qrslowmed_n7110best_g000_m007...` | medium_guarded_pmed0005 | 0.917177 | 0.995921 | 0.807736 | 0.970617 | too good-heavy |

## Conclusion

This boundary is not behaving like a simple data-capacity gap. A narrow qrs-low rule fixes the local geometry, but forcing that pattern into one checkpoint creates unstable global class bias. Do not continue broad qrs-low or class-weight sweeps.

Next useful options:

- Keep the transparent qrs-low rule-mode as an explicit rule-engine artifact and test small boundary bisections with endpoint+threshold manifests.
- If a single checkpoint is required, use a distillation-style objective around endpoint disagreement probabilities rather than adding more synthetic qrs-low rows.
- Continue raw waveform and overlap-only analysis before any wider N7125/N7150/N7200 attempt.
