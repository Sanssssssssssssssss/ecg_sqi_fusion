# E3.11 Isolated SQI Research Ablation

This report is generated from the isolated package `src/experiment/e311_sqi_research/`.

## Fixed References

- Strong baseline clone target: `0.9464`.
- Current highest candidate: mainline local-mask+rank result `0.9519`.
- Dataset: `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph`.
- Warm-start: D1 CLS checkpoint.

## Queue And Promotion Plan

Submitted on 2026-05-28 after the loss-scale fix:

| job id | group | array | purpose |
|---:|---|---|---|
| `29751790` | `focused_tuning` | `0-5%1` | low-weight local/level follow-up around the strongest mainline signal |
| `29751791` | `head_reimpl` | `0-4%1` | baseline, interpretable SQI head, local quality v2, combo head, multiscale |
| `29754081` | `sqi_head_tuning` | `0-15%2` | projected deterministic/predicted SQI stats injected into the classifier head |
| `29756103` | `sqi_head_mlp_tuning` | `0-11%2` | SQI-injected head plus a tiny one-hidden-layer classifier MLP |
| `29767663` | `sqi_residual_tuning` | `0-9%2` | zero-init SQI residual correction on top of the warm-started CLS logits |
| `29752152` | `loss_conflict` | `0-4%1` | CE-only and multi-task weighting conflict screen |
| `29752153` | `target_gate_reimpl` | `0-5%1` | clean-RR targets, bad fallback target, denoise gates |
| `29752154` | `generalization_loss` | `0-6%1` | label smoothing, focal, ordinal, R-Drop, SAM |

A recipe can move into the mainline sweep only when it:

- Beats the current candidate (`0.9519` test acc), or
- Improves medium recall by at least `0.01` while keeping bad recall within `0.005` of `0.9809`, or
- Stays near the strong baseline (`>= 0.9464`) and clearly reduces multi-task gradient conflict, in which case it must be rerun with seeds `0/1/2/3` before promotion.

Do not promote recipes that only improve auxiliary denoise/level metrics while lowering classification accuracy.

## Implementation Audit Notes

- `baseline_clone` and all fixed-weight recipes now match the mainline loss scale (`cls_weight=10`) and cosine LR schedule; earlier pre-fix research jobs over-weighted SNR/level/local auxiliaries relative to classification.
- `lc_uncert_multitask` now follows Kendall-style uncertainty weighting more closely: after warmup it learns from raw CE/denoise/level losses instead of pre-scaled fixed weights.
- `tg_patch_residual_level` now uses squared residual with a dataset-level p99 scale, so the target keeps absolute severity and no longer inverts bad-vs-medium supervision.
- `sqi_interpretable` and `sqi_local` now include an estimated SNR-style feature from signal/residual power instead of a duplicated residual mean.
- `multiscale_sqi_transformer` now uses PatchTST-style `unfold -> Linear(patch)` tokenizers for extra scales instead of padded Conv1d tokenizers.
- `gl_sam` now freezes BatchNorm running-stat updates during the second SAM forward pass, matching common SAM practice for models with BatchNorm layers.
- `sqi_head_tuning` keeps the model simple: it projects a small deterministic/predicted SQI-stat vector with one `LayerNorm -> Linear(32) -> GELU` block, then concatenates it with the CLS token.

## Current Best

- `target_gate_reimpl/tg_clean_rr_level`: acc `0.9432`, recall good/medium/bad `0.9319/0.9251/0.9728`.
- Decision: stop unless curve/grad norms explain a useful failure.

## Results

| group | run | status | acc | good | medium | bad | best epoch | decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| loss_conflict | lc_ce_only | done | 0.9391 | 0.9264 | 0.9210 | 0.9700 | 21 | stop unless curve/grad norms explain a useful failure |
| loss_conflict | lc_ce_level | pending |  |  |  |  |  |  |
| loss_conflict | lc_ce_denoise | pending |  |  |  |  |  |  |
| loss_conflict | lc_fixed_multitask | pending |  |  |  |  |  |  |
| loss_conflict | lc_uncert_multitask | pending |  |  |  |  |  |  |
| head_reimpl | hr_baseline_clone | done | 0.9391 | 0.9346 | 0.9142 | 0.9687 | 22 | stop unless curve/grad norms explain a useful failure |
| head_reimpl | hr_sqi_interpretable | pending |  |  |  |  |  |  |
| head_reimpl | hr_local_quality_v2 | pending |  |  |  |  |  |  |
| head_reimpl | hr_sqi_local_combo | pending |  |  |  |  |  |  |
| head_reimpl | hr_multiscale_10_20_40 | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_clean_rr_level | done | 0.9432 | 0.9319 | 0.9251 | 0.9728 | 24 | stop unless curve/grad norms explain a useful failure |
| target_gate_reimpl | tg_bad_fallback_level | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_patch_residual_level | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_abs_topq_gate | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_qrs_gate | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_level_weight_gate | pending |  |  |  |  |  |  |
| focused_tuning | ft_local_l0025_nodense | done | 0.9369 | 0.9033 | 0.9360 | 0.9714 | 20 | stop unless curve/grad norms explain a useful failure |
| focused_tuning | ft_local_l005_nodense | pending |  |  |  |  |  |  |
| focused_tuning | ft_local_l0075_nodense | pending |  |  |  |  |  |  |
| focused_tuning | ft_cleanrr_l025 | pending |  |  |  |  |  |  |
| focused_tuning | ft_badfallback_l025 | pending |  |  |  |  |  |  |
| focused_tuning | ft_local_cleanrr_l005_l025 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_input_cls | done | 0.9355 | 0.9428 | 0.9074 | 0.9564 | 17 | stop unless curve/grad norms explain a useful failure |
| sqi_head_tuning | sqi_input_attn | done | 0.9373 | 0.9292 | 0.9128 | 0.9700 | 24 | stop unless curve/grad norms explain a useful failure |
| sqi_head_tuning | sqi_input_cls_drop005 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_input_cls_snr010 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_pred_nodense | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_pred_den5 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_pred_den10_lvl025 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_pred_detach_den10_lvl025 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_pred_detach_cleanrr_l025 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_mil_nodense | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_mil_den5 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_mil_detach_den10_lvl025 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_mil_detach_cleanrr_l025 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_mil_detach_cleanrr_l05 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_mil_detach_patchres_l025 | pending |  |  |  |  |  |  |
| sqi_head_tuning | sqi_mil_detach_ls005 | pending |  |  |  |  |  |  |
| sqi_head_mlp_tuning | sqi_input_mlp64 | done | 0.9342 | 0.9210 | 0.9128 | 0.9687 | 18 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_input_mlp128 | done | 0.9378 | 0.9346 | 0.9128 | 0.9659 | 23 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_input_attn_mlp64 | done | 0.9401 | 0.9360 | 0.9251 | 0.9591 | 23 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_input_mlp64_drop005 | done | 0.9405 | 0.9319 | 0.9196 | 0.9700 | 19 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_input_mlp64_snr010 | done | 0.9382 | 0.9087 | 0.9305 | 0.9755 | 21 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_pred_detach_mlp64_nodense | done | 0.9360 | 0.9469 | 0.8992 | 0.9619 | 13 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_pred_detach_mlp64_den5 | done | 0.9391 | 0.9278 | 0.9210 | 0.9687 | 22 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_pred_detach_mlp64_den10_l025 | done | 0.9332 | 0.9196 | 0.9128 | 0.9673 | 20 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_nodense | done | 0.9410 | 0.9346 | 0.9210 | 0.9673 | 22 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_den5 | done | 0.9378 | 0.9114 | 0.9332 | 0.9687 | 15 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_cleanrr_l025 | done | 0.9414 | 0.9332 | 0.9237 | 0.9673 | 21 | stop unless curve/grad norms explain a useful failure |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_ls005 | done | 0.9414 | 0.9292 | 0.9237 | 0.9714 | 22 | stop unless curve/grad norms explain a useful failure |
| sqi_residual_tuning | sqi_resid_input | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_input_drop005 | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_input_snr010 | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_pred_detach_nodense | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_pred_detach_den5 | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_pred_detach_den10_l025 | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_mil_detach_nodense | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_mil_detach_den5 | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_mil_detach_cleanrr_l025 | pending |  |  |  |  |  |  |
| sqi_residual_tuning | sqi_resid_mil_detach_ls005 | pending |  |  |  |  |  |  |
| generalization_loss | gl_label_smooth_005 | done | 0.9387 | 0.9264 | 0.9441 | 0.9455 | 15 | stop unless curve/grad norms explain a useful failure |
| generalization_loss | gl_label_smooth_020 | pending |  |  |  |  |  |  |
| generalization_loss | gl_focal_15 | pending |  |  |  |  |  |  |
| generalization_loss | gl_focal_20 | pending |  |  |  |  |  |  |
| generalization_loss | gl_ordinal_ce | pending |  |  |  |  |  |  |
| generalization_loss | gl_rdrop | pending |  |  |  |  |  |  |
| generalization_loss | gl_sam | pending |  |  |  |  |  |  |

## Gradient-Norm Snapshot

| group | run | cls | denoise | level |
|---|---:|---:|---:|---:|
| loss_conflict | lc_ce_only | 0.0573 | 0.0000 | 0.0000 |
| head_reimpl | hr_baseline_clone | 38.5965 | 0.0000 | 0.0000 |
| target_gate_reimpl | tg_clean_rr_level | 24.9120 | 0.0000 | 0.8119 |
| focused_tuning | ft_local_l0025_nodense | 29.0549 | 0.0000 | 0.0000 |
| sqi_head_tuning | sqi_input_cls | 27.0821 | 0.0000 | 0.0000 |
| sqi_head_tuning | sqi_input_attn | 79.1725 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_input_mlp64 | 0.0580 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_input_mlp128 | 8.3363 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_input_attn_mlp64 | 1.3306 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_input_mlp64_drop005 | 107.3120 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_input_mlp64_snr010 | 1.5186 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_pred_detach_mlp64_nodense | 35.2848 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_pred_detach_mlp64_den5 | 17.8370 | 0.2493 | 0.0000 |
| sqi_head_mlp_tuning | sqi_pred_detach_mlp64_den10_l025 | 18.9009 | 0.5423 | 0.1114 |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_nodense | 0.1349 | 0.0000 | 0.0000 |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_den5 | 0.0201 | 0.2827 | 0.0000 |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_cleanrr_l025 | 14.0799 | 0.6189 | 0.0887 |
| sqi_head_mlp_tuning | sqi_mil_detach_mlp64_ls005 | 5.0719 | 0.5661 | 0.1074 |
| generalization_loss | gl_label_smooth_005 | 0.2286 | 0.0000 | 0.0000 |

## Reading Guide

- `loss_conflict` tests whether auxiliary dense losses help or fight classification.
- `head_reimpl` tests whether classification improves when it sees local residual/level summaries instead of a token alone.
- `sqi_head_tuning` is the focused version of that idea: deterministic input SQI stats, predicted residual/level stats, detached variants, and top-k patch MIL summaries.
- `target_gate_reimpl` checks whether RR targets and denoise gates cover bad samples more reliably.
- `generalization_loss` checks whether boundary-friendly losses improve medium without sacrificing bad.
- `focused_tuning` follows up on early results by lowering local/level supervision weights instead of changing data or the mainline model.

Runs only become candidates for seed expansion if they beat the current mainline best, improve medium recall with minimal bad loss, or match the strong baseline while reducing task-conflict evidence.
