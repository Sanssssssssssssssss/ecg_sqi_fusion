# E3.11f Encoder-Head Audit Package

This package is a GitHub-readable snapshot of the E3.11f denoise/classification experiments.
It intentionally excludes raw datasets, checkpoints, and NPZ tensors.

## Why This Exists

The current anomaly is:

- A denoiser-warm / classifier-He run reached very high classification accuracy.
- Its denoise metrics were bad enough that it should not be treated as a clean denoiser.
- We needed to check whether the classifier is learning from a true denoised waveform, or whether the denoiser/encoder is acting as a label shortcut.

## Initialization Clarification

The two encoder-head audit runs here are **not from zero**.

| run | denoiser init | classifier init | new head init | trainable parts |
|---|---|---|---|---|
| `encoder_head_only_e8_seed0` | warm mature denoiser checkpoint | none | new MLP head from random/default init | denoiser + encoder MLP head |
| `dual_branch_encoder_delta_e8_seed0` | warm mature denoiser checkpoint | warm Transformer classifier checkpoint | zero-init gated encoder delta | denoiser + Transformer classifier + encoder delta |

Both runs use the full split:

- train: `10935`
- val: `2184`
- test: `2202`

The warm denoiser checkpoint is the mature morph-aware residual U-Net baseline:

`outputs/experiment/e311_morph_denoise_gap5_7_grid/baseline/baseline_best_med6p25_badgap7_scale0p955/checkpoints/ckpt_best_denoise.pt`

## Key Results

| experiment | acc | good | medium | bad | denoise score | SNR gain | note |
|---|---:|---:|---:|---:|---:|---:|---|
| Mature denoise baseline | `0.9614` | `0.9441` | `0.9496` | `0.9905` | `2.8193` | `+7.69 dB` | frozen/warm baseline package |
| Denoiser-warm classifier-He | `0.9882` | `0.9864` | `0.9809` | `0.9973` | `-0.0689` | `-0.337 dB` | high acc, bad denoise morphology |
| `encoder_head_only_e8_seed0` | `0.9864` | `0.9809` | `0.9796` | `0.9986` | `3.6118` | `+10.21 dB` | denoiser encoder latent -> MLP classifier |
| `dual_branch_encoder_delta_e8_seed0` | `0.9628` | `0.9523` | `0.9441` | `0.9918` | `3.1657` | `+8.77 dB` | denoised waveform Transformer + encoder delta |
| Best PCGrad/cap run | `0.9728` | `0.9510` | `0.9741` | `0.9932` | `2.2825` | `+6.04 dB` | conflict controlled, denoise weaker |

## Current Interpretation

The `encoder_head_only` result is the most important audit signal. A classifier trained on the mature denoiser encoder latent reaches `0.9864` test accuracy while preserving strong denoise metrics. This means the denoiser encoder contains highly separable quality/class information.

That can be good, but it is also a shortcut warning:

- The labels are synthetic good/medium/bad labels derived from SNR and morphology damage rules.
- The denoiser encoder is trained exactly to understand and remove those same perturbations.
- A head on this latent may learn the data-generation rule rather than a clinically robust ECG quality concept.

The `dual_branch` result is less impressive. Its final accuracy is `0.9628`, and the base waveform Transformer inside that run has only `0.7225` accuracy at the selected checkpoint. The encoder delta repairs many errors, but this looks unstable and needs a cleaner design before promotion.

## What Another Chat Should Analyze

Ask the reviewer to focus on these questions:

1. Is `encoder_head_only` a valid architecture direction, or mostly evidence that the denoiser encoder encodes synthetic label rules?
2. Why does the warm denoiser make both classification and denoise train so well, while scratch runs learned classification but not good denoise?
3. Does the dataset construction make bad/medium too easy because class labels are tied to visible SNR/morphology perturbations?
4. Should the final publishable architecture allow classification directly from denoiser encoder features, or should encoder features only enter through a small residual SQI head?
5. Which audit is still missing: frozen denoiser + scratch classifier, CE stop-gradient through denoiser, cross-gap evaluation, patient-level split audit, residual-only/noisy-only classifiers?

Suggested next experiments:

- Freeze mature denoiser and train only `encoder_head_only` from scratch.
- Train `encoder_head_only` with `detach_encoder_features=True` to prevent CE modifying the denoiser.
- Run cross-gap evaluation on gap5/gap6.5/gap7 for the encoder-head model.
- Train residual-only, noisy-only, clean-only, and denoised-only classifiers to measure shortcut strength.
- If patient metadata is recoverable, repeat with patient-level split.

## Package Contents

- `code/encoder_head_audit.py`: experiment-only audit implementation.
- `code/run_encoder_head_audit.py`: two-run sequential runner.
- `metrics/*.json`: selected test reports and denoise metrics.
- `figures/*.png`: representative denoise galleries and conflict plots.
- `copied_reports/*.md`: previous full-joint, conflict, and mature-denoise reports.

No `src/transformer_pipeline` files are changed by this package.
