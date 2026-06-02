# E3.11f Uformer Mechanism Audit

Date: 2026-06-02

This report is a snapshot of the Transformer-denoiser audit while the runner is still active (`transunet1d_state: 31/49` at the time of capture). It is meant to preserve the important control groups and the mechanistic interpretation, not to declare the final promoted mainline yet.

## Executive Takeaway

The current evidence has shifted strongly toward a Transformer-based mainline:

```text
noisy ECG
  -> Uformer1D residual denoiser
  -> denoise = noisy - scale * noise_hat
  -> Uformer multi-scale latent / bottleneck / waveform summaries
  -> SQI classifier head
  -> good / medium / bad
```

The previous high-accuracy path used a frozen warm U-Net denoiser plus a Transformer classifier adapter. That was useful as an oracle/teacher, but it was not architecturally acceptable as the final mainline because U-Net provided the denoiser and most of the latent SQI signal.

The new Uformer1D results are different: the denoiser itself is Transformer-based, denoise quality exceeds the old U-Net reference, and the latent tokens support very strong classification.

Current best snapshot:

| Candidate | Architecture | Feature input to SQI head | Mode | Acc | Good | Medium | Bad | Denoise score | SNR gain |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `r2_detach_full_tokens_r1_uformer1d_hier_ns0p9_morph_soft_guard` | Uformer1D denoiser + SQI head | full Transformer tokens + summaries | detach | **0.98955** | **0.99455** | 0.97684 | **0.99728** | **4.303** | **12.41 dB** |
| `r2_detach_bottleneck_only_r1_uformer1d_hier_ns0p9_morph_soft_guard` | Uformer1D denoiser + SQI head | bottleneck latent | detach | 0.98910 | 0.98774 | **0.98774** | 0.99183 | 4.298 | 12.39 dB |
| `r2_freeze_bottleneck_only_r1_uformer1d_hier_ns0p9_morph_soft_guard` | Uformer1D denoiser + SQI head | bottleneck latent | freeze | 0.98865 | 0.98774 | 0.98229 | 0.99591 | 3.998 | 11.45 dB |
| `r2_freeze_full_tokens_r1_uformer1d_hier_ns0p9_morph_soft_guard` | Uformer1D denoiser + SQI head | full Transformer tokens + summaries | freeze | 0.98819 | 0.98774 | 0.97956 | 0.99728 | 3.998 | 11.45 dB |

This is the first line that looks like a credible "new king" for the user goal: Transformer-based denoising, visually strong residual noise prediction, and SQI/classification using the denoiser representation.

## Why The Architecture Is Reasonable

The best current architecture is not a generic Transformer dropped onto a waveform. It keeps the parts that made the old U-Net teacher work, but expresses them in a Transformer-based restoration model.

The model needs four behaviors:

1. Preserve local ECG morphology, especially QRS/T amplitude and timing.
2. See long-range 10-second noise structure such as baseline drift and mixed artifacts.
3. Predict residual/noise rather than directly hallucinating a clean ECG.
4. Expose a meaningful SQI representation to the classifier.

Uformer1D fits this because it is U-shaped and hierarchical:

```text
noisy ECG
  -> Conv1D local stem
  -> stage tokens / local features
  -> downsample
  -> hierarchical Transformer blocks
  -> bottleneck tokens
  -> upsample + skip
  -> decoder
  -> noise_hat
  -> denoise = noisy - scale * noise_hat
```

This is the 1D ECG analogue of restoration models such as Uformer / TransUNet: use convolutional/local structure for signal fidelity, Transformer blocks for long-range quality and morphology context, and skip connections for reconstruction detail.

## Control Group Map

The control groups are not incidental. They explain why the final model should have this shape.

| Control | What changes | Result | Interpretation |
|---|---|---:|---|
| Pure Transformer baseline `hr_baseline_clone` | No warm denoiser, raw Transformer training line | acc 0.94460 | A plain Transformer classifier is not enough on this dataset and noise regime. It learns bad recall, but good/medium are much weaker. |
| Pure/SQI Transformer `sqi_mil_detach_den10_lvl025` | Raw Transformer with SQI/MIL-style recipe | acc 0.94596 | SQI heads alone do not recover the missing denoising representation. |
| Old U-Net teacher + Transformer FiLM/token/cross | Frozen U-Net denoiser, Transformer classifier adapter | acc 0.976-0.981 | Demonstrates that mature denoiser latent features can support strong classification, but U-Net is too dominant for final mainline. |
| `conv_transunet_lite` denoise-only | Lightweight Conv stem + Transformer bottleneck + U decoder | denoise 3.74-3.85 | Good minimum Transformer-denoiser baseline; proves the idea is feasible but not best. |
| `conformer_unet1d` denoise-only | Conformer local conv + MHSA blocks + U decoder | denoise 3.80-3.92 | Strong local/global ECG inductive bias; useful ablation but currently below Uformer. |
| `uformer1d_hier` denoise-only | Hierarchical U-shaped Transformer | denoise 3.87-4.08 | Best denoise-only family; strongest restoration architecture so far. |
| Uformer `full_tokens` SQI | Multi-scale Uformer tokens + summaries | acc 0.988-0.990 | Best mainline candidate: token-level denoiser representation supports classification. |
| Uformer `bottleneck_only` SQI | Only bottleneck latent | acc 0.985-0.989 | Bottleneck is highly informative; quality/morphology signal is concentrated in high-level denoiser representation. |
| Uformer `summary_only` SQI | No latent tokens, waveform summaries only | acc 0.981-0.987 | Summary statistics are powerful, so the dataset has observable quality structure; this is a warning not to overclaim "deep morphology only." |
| Uformer `residual_summary_only` SQI | Residual/noise summary only | acc 0.54-0.59 | Critical negative control: the model is not merely using residual energy/SNR shortcut. |
| Freeze SQI head | Denoiser frozen, only SQI head trained | acc up to 0.98865 | The pretrained Uformer representation itself is already class-discriminative. |
| Detach SQI head | CE cannot backprop through encoder features; denoiser may continue denoise-only updates | acc up to 0.98955 | Best tradeoff: classification reads denoiser latent without CE corrupting it, while denoise continues improving. |

## Denoiser Family Comparison

Round1 denoise-only results show the architectural trend clearly.

| Model family | Best run | Denoise score | SNR gain | MSE ratio | Qualitative read |
|---|---|---:|---:|---:|---|
| `conv_transunet_lite` | `ns0p955_morph_soft_guard` | 3.847 | 10.96 dB | 0.0607 | Strong lightweight baseline. It proves a small Transformer bottleneck is enough to learn residual denoising. |
| `conformer_unet1d` | `ns0p955_morph_full_identity` | 3.922 | 11.19 dB | 0.0566 | Local conv + attention is well matched to ECG, but current implementation is a little below Uformer. |
| `uformer1d_hier` | `ns0p955_morph_soft_guard` | 4.077 | 11.69 dB | 0.0510 | Strongest denoiser family. U-shaped hierarchy and Transformer stages appear to preserve morphology while removing drift/HF noise. |

The Uformer result matters because it breaks the old worry that only U-Net can make the denoise visually credible. A Transformer-based denoiser now beats the U-Net teacher range on denoise score and SNR gain.

## SQI Feature Ablation

The Round2 SQI-head ablations are the strongest mechanism evidence.

| Feature set | Best acc | Best bad recall | Best denoise score | Interpretation |
|---|---:|---:|---:|---|
| `full_tokens` | **0.98955** | 0.99728 | **4.303** | Best mainline candidate. Multi-scale Uformer token representation is useful and class-discriminative. |
| `bottleneck_only` | 0.98910 | 0.99591 | 4.298 | Bottleneck has strong global quality/morphology representation. It may be a simpler deployable variant. |
| `summary_only` | 0.98728 | **0.99864** | 4.296 | Global summaries are powerful. Useful as ablation and auxiliary feature, but too simple to be the whole story. |
| `residual_summary_only` | 0.58765 | 0.95095 | 4.281 | Very important negative control. Residual energy alone does not solve classification and heavily collapses good recall. |

This strongly supports the architecture:

- Full/bottleneck Uformer latent features solve the task.
- Residual-only shortcuts do not.
- Summary-only is surprisingly strong, so we should be careful: part of the task is observable from global quality statistics, but the best and most defensible representation remains token/bottleneck latent.

## Freeze vs Detach

Two modes were used to avoid classification loss damaging the denoiser:

`freeze`:

```text
Uformer denoiser checkpoint frozen
SQI/classifier head trains on fixed features
```

`detach`:

```text
CE reads detached encoder features
denoiser can still receive denoise loss
classification cannot directly distort denoiser latent
```

Detach currently wins:

| Mode | Representative run | Acc | Denoise score | Read |
|---|---|---:|---:|---|
| Freeze | `freeze_full_tokens_ns0p9_soft_guard` | 0.98819 | 3.998 | Very strong fixed-representation evidence. |
| Detach | `detach_full_tokens_ns0p9_soft_guard` | **0.98955** | **4.303** | Better because denoiser continues denoise-only refinement while CE is blocked from corrupting features. |

This is a clean mechanism: do not let CE freely reshape the denoiser; let classifier read denoiser representation, while denoise loss keeps the representation faithful to waveform restoration.

## Why Not The Old U-Net Teacher As Mainline

The old U-Net teacher remains important, but as evidence and a reference.

It showed:

- Mature residual denoising can support high classification.
- Denoiser latent features are better than raw waveform alone.
- A small SQI adapter can dramatically help a Transformer classifier.

But it also had a structural problem: the denoiser and latent representation were U-Net-based. The user explicitly rejected a U-Net-only or U-Net-dominant mainline.

The new Uformer results solve that objection:

- Denoiser is Transformer-based.
- Reconstruction quality is better than the old teacher range.
- SQI/classification can use Transformer denoiser latent tokens.
- Negative controls show this is not just residual-energy leakage.

## Current New-King Candidate

Primary candidate:

```text
r2_detach_full_tokens_r1_uformer1d_hier_ns0p9_morph_soft_guard
```

Architecture:

```text
noisy ECG
  -> Uformer1D residual denoiser
  -> denoise = noisy - 0.90 * noise_hat
  -> full Uformer tokens + waveform summaries
  -> SQI classifier head
  -> good / medium / bad
```

Metrics:

| Metric | Value |
|---|---:|
| Test acc | **0.98955** |
| Good recall | **0.99455** |
| Medium recall | 0.97684 |
| Bad recall | **0.99728** |
| Denoise score | **4.303** |
| SNR gain | **12.41 dB** |
| MSE ratio | **0.0441** |

Why this one is more mainline-worthy than the raw top table alone:

- It uses `full_tokens`, which is the clearest Transformer-denoiser representation.
- It uses `detach`, which matches the loss-conflict lesson: classification reads representation but does not damage denoising.
- It beats the U-Net teacher family in both classification and denoise score in this snapshot.
- It has a strong negative control: residual-only is far worse.

Secondary candidate:

```text
r2_detach_bottleneck_only_r1_uformer1d_hier_ns0p9_morph_soft_guard
```

This has slightly lower acc but much more balanced medium recall:

| Metric | Value |
|---|---:|
| Test acc | 0.98910 |
| Good recall | 0.98774 |
| Medium recall | **0.98774** |
| Bad recall | 0.99183 |
| Denoise score | 4.298 |

This may become the "clean/simple" variant if we decide full-token features are too complex.

## Visual Audit Pointers

Representative figures are copied under `figures/`.

Important galleries:

- `figures/r2_detach_full_tokens_r1_uformer1d_hier_ns0p9_morph_soft_guard_balanced_gallery.png`
- `figures/r2_detach_full_tokens_r1_uformer1d_hier_ns0p9_morph_soft_guard_hard_bad_gallery.png`
- `figures/r2_detach_full_tokens_r1_uformer1d_hier_ns0p9_morph_soft_guard_good_safety_gallery.png`
- `figures/r2_detach_full_tokens_r1_uformer1d_hier_ns0p9_morph_soft_guard_worst_residual_gallery.png`
- `figures/r2_detach_residual_summary_only_r1_uformer1d_hier_ns0p9_morph_soft_guard_balanced_gallery.png`

Visual review should focus on:

1. Good-class safety: no excessive filtering of already clean morphology.
2. QRS amplitude/timing: no peak shift or flattening.
3. T-wave shape: no over-smoothing.
4. Bad/high-noise recovery: denoise should be closer to clean, not merely smoother.
5. Worst residual cases: identify remaining morphology failures before promotion.

## What Still Needs To Finish

This report is not final. The runner was at `31/49` when captured.

Remaining items:

- Finish remaining Round2 Uformer variants.
- Check whether any Round3 light joint fine-tune improves or harms the detach/freeze result.
- Manually inspect galleries for the new king and the failure controls.
- Run seed stability for:
  - `detach_full_tokens_ns0p9_soft_guard`
  - `detach_bottleneck_only_ns0p9_soft_guard`
  - one freeze baseline
  - one residual-only negative control, if needed.
- Re-run leakage/split audit in the final report.

## Promotion Criteria For Tomorrow

Do not promote just because of the high acc. Promote only if the following hold after completion:

1. Full-token or bottleneck Uformer stays strong across seeds.
2. Denoise visual galleries are clinically plausible enough for the project goal.
3. Good-class safety is acceptable.
4. Residual-only / summary-only controls are clearly documented.
5. The mainline implementation is simple:

```text
Uformer1D residual denoiser
  + detached/frozen SQI classifier head over full_tokens or bottleneck
```

6. No clean, true mask, true morph score, or true label is used as classifier input.

## Files In This Report

- `metrics/transunet1d_summary_snapshot.jsonl`: raw snapshot of TransUNet/Uformer/Conformer runs.
- `metrics/transformer_reentry_summary_snapshot.jsonl`: raw snapshot of previous Transformer re-entry controls.
- `metrics/<run_id>/test_report.json`: selected run classification reports.
- `metrics/<run_id>/denoise_metrics.json`: selected run denoise metrics.
- `figures/*gallery.png`: selected visual galleries.

Checkpoints and NPZ files are intentionally not included.
