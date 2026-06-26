# E29/E30 Final-Score GM Repair Report

Date: 2026-06-26

Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`

Split: 3-fold record-heldout, `split_seed=20260901`

Training seed: `20260951`

Runner: `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_gm_mechanism_repair_suite.py`

## 1. What changed

This run implemented the minimal repair proposed after E24:

- `pairrank_loss` can now use the final fused GM log-odds:
  `log(P_final_medium) - log(P_final_good)`.
- E29 adds final-score pairrank, hard-GM loss, medium-specific bad guard, and GM-oriented checkpoint selection.
- E30 adds reliable-factor-only GM fusion on top of E29:
  unstable factors remain supervised/reported, but no longer directly drive the GM boundary.
- A posthoc factor ablation stage was added for E24.
- Error-route and error-conditioned factor-residual reports were added.

The encoder, data policy, waveform-only inference contract, and v112 distribution are unchanged.

## 2. Clean-test summary

| candidate | acc | macro-F1 | good | medium | bad | GM balanced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E30 reliable detached fusion | 0.917508 | 0.900581 | 0.869858 | 0.852103 | 0.976008 | 0.860980 |
| E24 same-split baseline | 0.917177 | 0.898747 | 0.849749 | 0.855553 | 0.981994 | 0.852651 |
| E29 final-score guard | 0.903761 | 0.882930 | 0.822191 | 0.841516 | 0.976139 | 0.831854 |

Result:

- E30 is the best candidate in this run, but only by a small margin.
- E30 improves mean acc by `+0.00033` and GM balanced by `+0.00833` versus the same-split E24 baseline.
- E30 does not meet the planned useful threshold of `+0.003 acc` or `medium >= 0.875`.
- E29 is clearly worse; the final-score pairrank + medium bad guard combination is too restrictive/unstable as configured.

## 3. Error-route summary

| candidate | good->medium | good->bad | medium->good | medium->bad | bad->good | bad->medium |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E24 same-split baseline | 110.7 | 20.0 | 97.7 | 69.0 | 11.7 | 24.3 |
| E29 final-score guard | 142.7 | 14.3 | 104.0 | 78.7 | 11.3 | 36.3 |
| E30 reliable detached fusion | 99.3 | 14.3 | 102.3 | 68.0 | 13.3 | 34.7 |

Interpretation:

- E30 mainly helps by reducing `good->medium` errors by about 11 rows/fold.
- It does not solve `medium->good`; that route slightly increases versus baseline.
- `medium->bad` is almost unchanged.
- Bad recall drops from `0.98199` to `0.97601`, still high but worse than baseline.

So E30 improves one side of the GM boundary, but it does not truly separate the two classes yet.

## 4. Factor ablation findings

Posthoc E24 factor masking showed:

- Masking `detector_agreement`, `contact_loss_win_ratio`, and `amplitude_entropy` has essentially no effect on clean-test metrics.
- Masking `template_corr` is slightly harmful.
- Masking `qrs_visibility` is nearly neutral.
- Masking `baseline_step` reduces `medium->good` by about 6 rows/fold, but hurts good recall enough that aggregate GM does not improve.

Conclusion:

The weak factors are mostly not being used effectively by the current GM head, rather than actively poisoning it. `baseline_step` is the only masked factor with a visible decision effect, but it is a two-edged signal: it helps medium but can demote good.

## 5. Feature recovery

E30 preserves or slightly improves several strong factors:

| feature | E24 corr_all | E30 corr_all | E24 min-class | E30 min-class |
| --- | ---: | ---: | ---: | ---: |
| baseline_step | 0.972493 | 0.974630 | 0.895197 | 0.890044 |
| qrs_visibility | 0.964617 | 0.963761 | 0.655241 | 0.678170 |
| sqi_basSQI | 0.967722 | 0.969640 | 0.810553 | 0.811301 |
| qrs_band_ratio | 0.955430 | 0.954715 | 0.838545 | 0.839022 |
| detector_agreement | 0.709902 | 0.716076 | 0.060416 | 0.044153 |
| template_corr | 0.947434 | 0.947430 | 0.279932 | 0.282954 |
| contact_loss_win_ratio | 0.267122 | 0.285242 | -0.023155 | -0.002506 |

The recovery story is consistent with the metric story:

- Strong waveform-computable SQI factors remain strong.
- E30 slightly improves some weak factors, but not enough to change the medium boundary materially.
- `detector_agreement` and `contact_loss_win_ratio` remain poor class-wise signals.

## 6. Decision

Do not promote E29.

E30 is a useful diagnostic candidate and the best same-split candidate in this run, but it is not a strong replacement for E24 because:

- the acc gain is too small;
- medium recall is still only `0.852`;
- medium->good is not reduced;
- bad recall is lower than baseline.

The most important lesson is that final-score alignment alone is not enough. Reliable detached factor fusion helps modestly, which supports the idea that weak factors should not directly control GM, but the model still needs a better mechanism for the actual hard boundary rows.

## 7. Recommended next step

Keep E24/E30 as the two reference mechanisms:

- E24: stronger bad recall and slightly better medium recall.
- E30: better good recall, better GM balanced, slightly cleaner factor path.

Next experiment should not increase alpha/lr/pairrank again. It should target the remaining hard boundary directly:

1. Use E24 vs E30 disagreement rows to build a hard-GM analysis set.
2. For `medium->good`, inspect whether medium rows have systematically overestimated `qrs_visibility`, `sqi_basSQI`, or underestimated `baseline_step/non_qrs_rms_ratio`.
3. Add a family-specific GM expert only for those rows, or add a small boundary-family classifier whose output is supervised by the existing hard subtype labels.
4. Keep medium bad guard weaker or separate from GM repair, because E29 showed that tying the two too strongly hurts overall performance.

## 8. Output files

Metrics:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/candidate_metrics_e29e30_finalscore_guard_20260626.csv`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/error_route_metrics_e29e30_finalscore_guard_20260626.csv`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/error_conditioned_factor_residual_e29e30_finalscore_guard_20260626.csv`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/factor_recovery_decoded_e29e30_finalscore_guard_20260626.csv`

Factor ablation:

- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/factor_ablation_posthoc_e24_20260626.csv`
- `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/factor_ablation_posthoc_deltas_e24_20260626.csv`

Report:

- `reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/gm_mechanism_repair_e29e30_finalscore_report_20260626.md`
