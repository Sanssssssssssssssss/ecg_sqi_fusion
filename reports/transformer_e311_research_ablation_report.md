# E3.11 Isolated SQI Research Ablation

This report is generated from the isolated package `src/experiment/e311_sqi_research/`.

## Fixed References

- Strong baseline clone target: `0.9464`.
- Current highest candidate: local-mask result `0.9505`.
- Dataset: `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph`.
- Warm-start: D1 CLS checkpoint.

## Results

| group | run | status | acc | good | medium | bad | best epoch | decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| loss_conflict | lc_ce_only | pending |  |  |  |  |  |  |
| loss_conflict | lc_ce_level | pending |  |  |  |  |  |  |
| loss_conflict | lc_ce_denoise | pending |  |  |  |  |  |  |
| loss_conflict | lc_fixed_multitask | pending |  |  |  |  |  |  |
| loss_conflict | lc_uncert_multitask | pending |  |  |  |  |  |  |
| head_reimpl | hr_baseline_clone | pending |  |  |  |  |  |  |
| head_reimpl | hr_sqi_interpretable | pending |  |  |  |  |  |  |
| head_reimpl | hr_local_quality_v2 | pending |  |  |  |  |  |  |
| head_reimpl | hr_sqi_local_combo | pending |  |  |  |  |  |  |
| head_reimpl | hr_multiscale_10_20_40 | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_clean_rr_level | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_bad_fallback_level | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_patch_residual_level | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_abs_topq_gate | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_qrs_gate | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_level_weight_gate | pending |  |  |  |  |  |  |
| generalization_loss | gl_label_smooth_005 | pending |  |  |  |  |  |  |
| generalization_loss | gl_label_smooth_020 | pending |  |  |  |  |  |  |
| generalization_loss | gl_focal_15 | pending |  |  |  |  |  |  |
| generalization_loss | gl_focal_20 | pending |  |  |  |  |  |  |
| generalization_loss | gl_ordinal_ce | pending |  |  |  |  |  |  |
| generalization_loss | gl_rdrop | pending |  |  |  |  |  |  |
| generalization_loss | gl_sam | pending |  |  |  |  |  |  |

## Gradient-Norm Snapshot

| group | run | cls | denoise | level |
|---|---:|---:|---:|---:|

## Reading Guide

- `loss_conflict` tests whether auxiliary dense losses help or fight classification.
- `head_reimpl` tests whether classification improves when it sees local residual/level summaries instead of a token alone.
- `target_gate_reimpl` checks whether RR targets and denoise gates cover bad samples more reliably.
- `generalization_loss` checks whether boundary-friendly losses improve medium without sacrificing bad.

Runs only become candidates for seed expansion if they beat `0.9505`, improve medium recall with minimal bad loss, or match the strong baseline while reducing task-conflict evidence.
