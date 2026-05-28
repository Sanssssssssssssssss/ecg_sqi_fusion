# E3.11 Isolated SQI Research Ablation

This report is generated from the isolated package `src/experiment/e311_sqi_research/`.

## Fixed References

- Strong baseline clone target: `0.9464`.
- Current highest candidate: local-mask result `0.9505`.
- Dataset: `outputs/transformer_e311_mainline_strict/e311f_lite_e310_morph`.
- Warm-start: D1 CLS checkpoint.

## Queue And Promotion Plan

Submitted on 2026-05-28:

| job id | group | array | purpose |
|---:|---|---|---|
| `29750889` | `head_reimpl` | `0` | baseline clone sanity run |
| `29751043` | `head_reimpl` | `1-4%1` | interpretable SQI head, local quality v2, combo head, multiscale |
| `29751044` | `loss_conflict` | `0-4%1` | CE-only and multi-task weighting conflict screen |
| `29751045` | `target_gate_reimpl` | `0-5%1` | clean-RR targets, bad fallback target, denoise gates |
| `29751046` | `generalization_loss` | `0-6%1` | label smoothing, focal, ordinal, R-Drop, SAM |

A recipe can move into the mainline sweep only when it:

- Beats the current candidate (`0.9505` test acc), or
- Improves medium recall by at least `0.01` while keeping bad recall within `0.005` of `0.9741`, or
- Stays near the strong baseline (`>= 0.9464`) and clearly reduces multi-task gradient conflict, in which case it must be rerun with seeds `0/1/2/3` before promotion.

Do not promote recipes that only improve auxiliary denoise/level metrics while lowering classification accuracy.

## Implementation Audit Notes

- `lc_uncert_multitask` now follows Kendall-style uncertainty weighting more closely: after warmup it learns from raw CE/denoise/level losses instead of pre-scaled fixed weights.
- `tg_patch_residual_level` now uses squared residual with a dataset-level p99 scale, so the target keeps absolute severity and no longer inverts bad-vs-medium supervision.
- `sqi_interpretable` and `sqi_local` now include an estimated SNR-style feature from signal/residual power instead of a duplicated residual mean.
- `multiscale_sqi_transformer` now uses PatchTST-style `unfold -> Linear(patch)` tokenizers for extra scales instead of padded Conv1d tokenizers.

## Current Best

- `target_gate_reimpl/tg_clean_rr_level`: acc `0.9437`, recall good/medium/bad `0.9319/0.9387/0.9605`.
- Decision: stop unless curve/grad norms explain a useful failure.

## Results

| group | run | status | acc | good | medium | bad | best epoch | decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| loss_conflict | lc_ce_only | done | 0.9360 | 0.9183 | 0.9223 | 0.9673 | 12 | stop unless curve/grad norms explain a useful failure |
| loss_conflict | lc_ce_level | pending |  |  |  |  |  |  |
| loss_conflict | lc_ce_denoise | pending |  |  |  |  |  |  |
| loss_conflict | lc_fixed_multitask | pending |  |  |  |  |  |  |
| loss_conflict | lc_uncert_multitask | pending |  |  |  |  |  |  |
| head_reimpl | hr_baseline_clone | done | 0.9382 | 0.9183 | 0.9183 | 0.9782 | 21 | stop unless curve/grad norms explain a useful failure |
| head_reimpl | hr_sqi_interpretable | pending |  |  |  |  |  |  |
| head_reimpl | hr_local_quality_v2 | pending |  |  |  |  |  |  |
| head_reimpl | hr_sqi_local_combo | pending |  |  |  |  |  |  |
| head_reimpl | hr_multiscale_10_20_40 | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_clean_rr_level | done | 0.9437 | 0.9319 | 0.9387 | 0.9605 | 15 | stop unless curve/grad norms explain a useful failure |
| target_gate_reimpl | tg_bad_fallback_level | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_patch_residual_level | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_abs_topq_gate | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_qrs_gate | pending |  |  |  |  |  |  |
| target_gate_reimpl | tg_level_weight_gate | pending |  |  |  |  |  |  |
| generalization_loss | gl_label_smooth_005 | done | 0.9382 | 0.9373 | 0.9196 | 0.9578 | 20 | stop unless curve/grad norms explain a useful failure |
| generalization_loss | gl_label_smooth_020 | pending |  |  |  |  |  |  |
| generalization_loss | gl_focal_15 | pending |  |  |  |  |  |  |
| generalization_loss | gl_focal_20 | pending |  |  |  |  |  |  |
| generalization_loss | gl_ordinal_ce | pending |  |  |  |  |  |  |
| generalization_loss | gl_rdrop | pending |  |  |  |  |  |  |
| generalization_loss | gl_sam | pending |  |  |  |  |  |  |

## Gradient-Norm Snapshot

| group | run | cls | denoise | level |
|---|---:|---:|---:|---:|
| loss_conflict | lc_ce_only | 0.4180 | 0.0000 | 0.0000 |
| head_reimpl | hr_baseline_clone | 1.7354 | 0.0000 | 0.0000 |
| target_gate_reimpl | tg_clean_rr_level | 2.2559 | 0.0000 | 0.2389 |
| generalization_loss | gl_label_smooth_005 | 5.2081 | 0.0000 | 0.0000 |

## Reading Guide

- `loss_conflict` tests whether auxiliary dense losses help or fight classification.
- `head_reimpl` tests whether classification improves when it sees local residual/level summaries instead of a token alone.
- `target_gate_reimpl` checks whether RR targets and denoise gates cover bad samples more reliably.
- `generalization_loss` checks whether boundary-friendly losses improve medium without sacrificing bad.

Runs only become candidates for seed expansion if they beat `0.9505`, improve medium recall with minimal bad loss, or match the strong baseline while reducing task-conflict evidence.
