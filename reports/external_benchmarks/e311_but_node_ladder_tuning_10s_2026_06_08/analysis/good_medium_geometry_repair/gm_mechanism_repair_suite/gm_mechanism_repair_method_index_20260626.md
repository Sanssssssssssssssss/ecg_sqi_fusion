# GM Mechanism Repair Method Index

Date: 2026-06-26

This note is a compact handoff map for the current GM mechanism repair line. It explains which candidate means what, what was actually tested, and what should be adjusted next.

## Fixed context

- Data policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`
- Model family: waveform-only `EventFactorizedSQIConformer` plus `GMMechanismConformer` repair heads.
- Inference input: waveform-derived channels only.
- SQI/factor columns: teacher targets and internal predicted evidence only; they are not external inference inputs.
- No `src/sqi_pipeline` changes.
- Key runner: `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_gm_mechanism_repair_suite.py`

## Candidate map

| Candidate | Meaning | Main mechanism | Outcome |
| --- | --- | --- | --- |
| `E4_v112_lowaux_lr15e4` | Old low-aux reference | Legacy factor contract, legacy GM path | Historical reference only |
| `E6_factor_fused_gm` | First useful GM repair base | Mechanism factor contract + GM reads decoded factors/local stats | Became base for later variants |
| `E21_e6_subcls_consistency` | Subtype consistency ablation | Adds subtype-to-class consistency to E6 | Slightly useful but not best |
| `E22_e6_subtype_class_fusion` | Subtype class fusion | Fuses subtype-derived class probability into final class probability | Useful, but weaker than E24 |
| `E23_e14_subtype_class_fusion` | Medium-weighted subtype fusion | More medium-favoring class weights with subtype fusion | Better medium/bad, worse good |
| `E24_e6_subtype_fusion_pairrank` | Current main reference before E29/E30 | E6 + subtype class fusion + pairrank | Best balanced historical candidate |
| `E25_e24_lowalpha` | E24 with weaker subtype fusion | Alpha 0.14 instead of 0.20 | Worse than E24 |
| `E26_e24_lowlr` | E24 with lower LR | LR 1e-4 | Higher good, worse medium/acc |
| `E27_e24_lowalpha_lowlr` | Low alpha + low LR | Conservative combined variant | Worse |
| `E28_e24_softpair_lowalpha` | Softer pairrank | Pairrank 0.04, alpha 0.16 | Worse |
| `E24_e29e30_baseline` | Same-split rerun of E24 | E24 config rerun on the E29/E30 comparison seed | Baseline for this exact run |
| `E29_e24_finalscore_pairrank_mediumguard` | Final-score loss alignment | Pairrank uses final fused GM logit; adds hard-GM loss and medium-bad guard | Failed, too restrictive |
| `E30_e29_reliable_detached_factor_fusion` | Reliable detached factor fusion | E29 plus only stable factors in GM fusion, detached evidence, gated factor residual | Small diagnostic gain, not enough to promote |

## Current results

Latest same-split clean-test summary:

| Candidate | Acc | Macro-F1 | Good | Medium | Bad | GM balanced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E30 reliable detached fusion | 0.9175 | 0.9006 | 0.8699 | 0.8521 | 0.9760 | 0.8610 |
| E24 same-split baseline | 0.9172 | 0.8987 | 0.8497 | 0.8556 | 0.9820 | 0.8527 |
| E29 final-score guard | 0.9038 | 0.8829 | 0.8222 | 0.8415 | 0.9761 | 0.8319 |

Interpretation:

- E29 should not be used: final-score pairrank + hard-GM + medium-bad guard made the model too constrained.
- E30 is the best in the exact E29/E30 comparison, but only slightly. It improves good recall and GM balanced, but does not fix medium recall.
- E24 remains the safer main reference; E30 is a useful diagnostic branch showing that weak factors should not directly control the GM boundary.

## What each repair tested

### E24

Purpose: make good/medium classification read predicted mechanism evidence, not only global embedding.

Mechanism:

- `medium_logit = gm_direct_head(GM_BOUNDARY) + gm_factor_head([GM_BOUNDARY, decoded_factor_pred, local_stats])`
- subtype leaf probabilities are summed into class probabilities;
- final class probability is log-space fusion of hierarchy and subtype class evidence;
- pairrank was originally applied to pre-fusion `medium_logit`.

Known problem:

- It moves the good/medium boundary, but does not truly separate the representation.
- Medium still has two major error routes: `medium->good` and `medium->bad`.

### E29

Purpose: align pairrank and hard-GM losses with the final fused class probability.

Mechanism:

- `gm_final_logit = log(P_final_medium) - log(P_final_good)`;
- pairrank and hard-GM BCE use this final GM logit;
- `bad_final_logit` is guarded for medium rows so moderate artifacts are not over-pushed into bad;
- checkpoint selection rejects epochs with bad recall below 0.965.

Outcome:

- Failed. It over-constrained training and reduced acc/GM balance.
- It did create some high-medium epochs, but those did not generalize as best checkpoints.

### E30

Purpose: test whether unstable factors are polluting GM fusion.

Mechanism:

- Stable GM factors kept: `baseline_step`, `non_qrs_rms_ratio`, `qrs_visibility`, `sqi_basSQI`, `qrs_band_ratio`, `non_qrs_diff_p95`.
- Unstable factors excluded from direct GM fusion but still supervised/reported: `detector_agreement`, `template_corr`, `contact_loss_win_ratio`, `amplitude_entropy`, `flatline_ratio`.
- Factor/local evidence is detached before GM residual.
- A small gated residual controls how strongly factor evidence can move the direct GM score.

Outcome:

- Small positive diagnostic result.
- Reduced `good->medium` errors, but did not reduce `medium->good`.
- Good signal: factor detachment/reliability filtering is directionally useful.
- Bad signal: this alone is far from a real 0.95 fix.

## Factor ablation findings

Posthoc E24 factor masking showed:

- Masking `detector_agreement`, `contact_loss_win_ratio`, and `amplitude_entropy` had nearly zero effect.
- Masking `template_corr` was slightly harmful.
- Masking `qrs_visibility` was nearly neutral.
- Masking `baseline_step` reduced `medium->good`, but hurt good recall.

This means the weak factors are mostly not being used effectively by the GM head. The next step should not be "make detector/contact bigger"; it should first determine whether hard GM rows need a separate mechanism.

## Recommended next question for another reviewer/AI

Ask specifically:

> We have a waveform-only EventFactorized SQI Conformer. E24 uses factor-fused GM + subtype fusion + pairrank. E30 detaches/restricts reliable factor evidence and slightly improves GM balanced but does not reduce medium->good. Which architecture/loss should separate hard good/medium rows without just moving the threshold? Should we add family-specific GM experts, a contrastive boundary embedding, or a supervised boundary-family query?

Useful files for review:

- Method index: this file.
- E24 mechanism report: `gm_mechanism_repair_e24_mainline_mechanism_report_20260625.md`
- E29/E30 report: `gm_mechanism_repair_e29e30_finalscore_report_20260626.md`
- Metrics: `candidate_metrics_e29e30_finalscore_guard_20260626.csv`
- Error routes: `error_route_metrics_e29e30_finalscore_guard_20260626.csv`
- Factor residuals: `error_conditioned_factor_residual_e29e30_finalscore_guard_20260626.csv`
- Runner code: `run_gm_mechanism_repair_suite.py`
