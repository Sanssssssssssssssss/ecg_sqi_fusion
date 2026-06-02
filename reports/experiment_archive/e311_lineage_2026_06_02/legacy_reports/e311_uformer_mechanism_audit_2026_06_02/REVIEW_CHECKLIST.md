# Review Checklist For The Uformer Mechanism Audit

Use this checklist when reviewing the Uformer/TransUNet results. The goal is to decide whether the architecture is mechanism-solid enough to become the next mainline candidate.

## Do Not Misread The Experiments

- Do not treat the old U-Net teacher as the final mainline. It is an oracle/reference showing that denoiser latent representations can support classification.
- Do not treat residual-only failures as failed denoising. They are classification-feature ablations showing that residual energy alone is not sufficient.
- Do not promote a run based only on acc. Promotion needs denoise quality, visual plausibility, class balance, and clean mechanism.
- Do not claim "pure Transformer" for the old FiLM/token/cross re-entry results. Those used a frozen U-Net denoiser.
- Do not claim "deep morphology only" if `summary_only` remains very strong. Say that waveform-level summaries are informative, while full/bottleneck tokens are the more defensible latent representation.

## Core Questions

1. Does `uformer1d_hier` remain the best denoiser family after the full runner completes?
2. Does `detach_full_tokens` stay above the freeze and summary-only variants across seeds?
3. Are `full_tokens` and `bottleneck_only` both strong enough to justify the denoiser-latent mechanism?
4. Does `residual_summary_only` stay weak enough to rule out a simple residual/SNR shortcut?
5. Do visual galleries support the metrics for:
   - good-class safety
   - QRS amplitude preservation
   - QRS timing preservation
   - T-wave preservation
   - reduced baseline drift
   - reduced high-frequency noise
6. Is medium recall stable enough? The current best full-token run has excellent acc and bad recall, but medium recall is lower than the bottleneck variant.

## Candidate Ranking Logic

Prefer:

1. `detach_full_tokens` if it keeps the best acc/denoise and visuals look clean.
2. `detach_bottleneck_only` if it is more balanced or much simpler.
3. `freeze_full_tokens` if detach appears to overfit or alters denoise too aggressively.
4. `summary_only` only as an ablation/auxiliary feature, not as the main claimed mechanism.
5. `residual_summary_only` only as a negative control.

## Promotion Bar

A promotion candidate should satisfy:

- acc >= 0.985 on the main split
- bad recall >= 0.995
- medium recall preferably >= 0.98, or a clear reason why the chosen candidate favors good/bad
- denoise_score >= 4.0
- SNR gain >= 11 dB
- no severe visual good-overfiltering
- no obvious QRS/T distortion in hard and worst-residual galleries
- no clean/true mask/label leakage as classifier input

## Follow-Up Experiments To Schedule

If the current runner confirms the snapshot:

1. Seed stability:
   - `detach_full_tokens_ns0p9_morph_soft_guard`
   - `detach_bottleneck_only_ns0p9_morph_soft_guard`
   - `freeze_full_tokens_ns0p9_morph_soft_guard`
2. Cross-gap robustness:
   - gap5
   - gap6.5
   - gap7
3. Visual audit expansion:
   - fixed same-sample comparison across U-Net teacher, Uformer full_tokens, Uformer bottleneck, residual-only control
4. Mainline simplification study:
   - full tokens vs bottleneck token
   - detached denoiser vs frozen denoiser
   - whether summary features should be included or only reported as analysis

## Suggested Mainline Wording

If confirmed, describe the model as:

> A 1D U-shaped hierarchical Transformer residual denoiser that predicts noise_hat and produces both a denoised ECG and denoiser-derived SQI latent features. A lightweight detached SQI classifier reads multi-scale/bottleneck Transformer denoiser representations to classify good/medium/bad quality while preserving the denoising objective.

Avoid describing it as:

> A U-Net classifier.

Avoid describing it as:

> A plain Transformer classifier.

The key mechanism is:

> Transformer-based restoration first; detached denoiser-latent SQI classification second.
