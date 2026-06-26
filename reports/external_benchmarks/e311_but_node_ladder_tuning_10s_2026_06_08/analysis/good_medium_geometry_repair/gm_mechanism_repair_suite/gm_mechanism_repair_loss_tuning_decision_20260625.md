# GM Mechanism Repair Loss/Fusion Tuning Decision

Run date: 2026-06-25

Policy fixed: `ptb_v112_gm_buffered_large_hybrid_s20260741`

Scope: external experiment only. No `src/sqi_pipeline` changes. All candidates were trained from scratch with waveform-derived inputs only. SQI/factor targets were used only as training supervision and diagnostics.

## Decision

Keep `E6_factor_fused_gm` as the mainline normal checkpoint architecture.

Use `E14_e6_medguard_lowgain` only as a balanced good/medium ablation, not as the mainline. The E15-E20 loss tuning run did not beat E6, and none of the CE/focal/smoothing/low-LR changes moved the model toward the 0.96 target.

The strongest observed mechanism remains:

```text
EventFactorizedSQIConformer
  + fixed FactorSpec decoding
  + factor-fused GM decision
  + local/factor auxiliary supervision retained
```

## Clean Test Comparison

Mean over 3 folds.

| group | candidate | acc | macro-F1 | good recall | medium recall | bad recall | GM mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4-E9 | E6_factor_fused_gm | 0.921981 | 0.905553 | 0.894739 | 0.853715 | 0.972295 | 0.874227 |
| E15-E20 | E16_e14_classheavy_lowaux | 0.921650 | 0.904851 | 0.883797 | 0.854335 | 0.976943 | 0.869066 |
| E10-E14 | E14_e6_medguard_lowgain | 0.921401 | 0.904896 | 0.890435 | 0.860098 | 0.970151 | 0.875266 |
| E15-E20 | E15_e6_classheavy_lowaux | 0.920573 | 0.903566 | 0.876837 | 0.860963 | 0.973608 | 0.868900 |
| E4-E9 | E5_factor_contract_only | 0.919910 | 0.902502 | 0.863471 | 0.864026 | 0.976273 | 0.863748 |
| E10-E14 | E13_e6_boundary_aux_stronger | 0.919662 | 0.902140 | 0.847494 | 0.878839 | 0.974661 | 0.863166 |
| E15-E20 | E17_e6_focal_gm | 0.919496 | 0.902764 | 0.890513 | 0.851704 | 0.970332 | 0.871108 |
| E10-E14 | E12_e6_pairrank_only | 0.919496 | 0.902442 | 0.888113 | 0.852856 | 0.971127 | 0.870485 |
| E4-E9 | E4_v112_lowaux_lr15e4 | 0.918917 | 0.901144 | 0.859228 | 0.864086 | 0.976414 | 0.861657 |
| E15-E20 | E18_e6_smooth_ce | 0.918751 | 0.901078 | 0.878496 | 0.850248 | 0.975561 | 0.864372 |
| E10-E14 | E10_e6_medguard | 0.917177 | 0.899133 | 0.866349 | 0.857489 | 0.972449 | 0.861919 |
| E15-E20 | E20_e6_lowlr_balanced | 0.916514 | 0.898298 | 0.877002 | 0.837557 | 0.978382 | 0.857280 |
| E4-E9 | E8_pairrank_hardsampler | 0.916018 | 0.897255 | 0.878287 | 0.837610 | 0.977224 | 0.857948 |
| E10-E14 | E11_e6_lowgain | 0.915521 | 0.897354 | 0.858777 | 0.848165 | 0.979110 | 0.853471 |
| E4-E9 | E7_family_moe_condsubtype | 0.913285 | 0.894334 | 0.856891 | 0.851413 | 0.973255 | 0.854152 |
| E15-E20 | E19_e6_nolocal_classfocus | 0.912953 | 0.894165 | 0.855318 | 0.854732 | 0.971590 | 0.855025 |
| E4-E9 | E9_beat_background_tokens | 0.863757 | 0.831859 | 0.878617 | 0.691933 | 0.956356 | 0.785275 |

`GM mean = (good recall + medium recall) / 2`.

## What Worked

1. `E6_factor_fused_gm` is still the best overall mean-accuracy candidate.
2. `E14_e6_medguard_lowgain` slightly improves the good/medium balance, mainly by lifting medium recall, but it does not improve overall accuracy.
3. Factor recovery is no longer the obvious blocker for the major waveform-computable SQI targets.

For E15-E20, decoded factor correlations were already strong:

| candidate | qrs_visibility | qrs_band_ratio | sqi_basSQI | baseline_step | detector_agreement | non_qrs_diff_p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E15 | 0.960 | 0.943 | 0.958 | 0.955 | 0.773 | 0.864 |
| E16 | 0.960 | 0.944 | 0.960 | 0.958 | 0.781 | 0.874 |
| E17 | 0.965 | 0.959 | 0.973 | 0.977 | 0.759 | 0.918 |
| E18 | 0.960 | 0.945 | 0.959 | 0.958 | 0.757 | 0.885 |
| E19 | 0.959 | 0.938 | 0.949 | 0.954 | 0.808 | 0.854 |
| E20 | 0.958 | 0.940 | 0.949 | 0.961 | 0.767 | 0.879 |

This means the current encoder/head can recover many interpretable SQI factors. The remaining ceiling is the mapping from factor evidence to stable good/medium class decisions, especially for boundary subtypes.

## What Did Not Work

1. Heavier class loss with lower auxiliary loss did not help.
   - E15: 0.920573
   - E16: 0.921650
2. Focal loss improved some factor recovery but did not improve classification.
   - E17: 0.919496
3. Label smoothing was worse.
   - E18: 0.918751
4. Removing local supervision was clearly worse.
   - E19: 0.912953
   - Local/event supervision should stay in the model.
5. Lower learning rate made training smoother but did not improve the ceiling.
   - E20: 0.916514

## Interpretation

The 0.96 target was not reached by loss-ratio tuning. The current result is not a simple CE/aux-weight problem.

The strongest evidence:

- E17 learns factors very well, but its classification accuracy is lower than E6.
- E19 removes local supervision and degrades strongly, so local evidence is useful.
- E20 lowers LR and stabilizes training, but does not unlock higher accuracy.
- E14/E15 can trade good recall against medium recall, but the trade is small and does not reduce the overall boundary ceiling.

So the next useful work should not be another broad CE/focal/smoothing sweep. The next meaningful change should target class-boundary composition:

1. stronger subtype-to-class calibration,
2. better boundary subtype sampling,
3. boundary-pair ranking using verified same-family pairs,
4. or a cleaner class-label/subtype-label audit around the good/medium boundary.

## Files

Primary runner:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_gm_mechanism_repair_suite.py`

E15-E20 outputs:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\candidate_metrics_e15e20_losstune_20260625.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\factor_recovery_decoded_e15e20_losstune_20260625.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\confusion_by_candidate_e15e20_losstune_20260625.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\local_event_metrics_e15e20_losstune_20260625.csv`

Reports:

- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\gm_mechanism_repair_loss_tuning_decision_20260625.md`
- `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_mechanism_repair_suite\gm_mechanism_repair_suite_report_e15e20_losstune_20260625.md`

