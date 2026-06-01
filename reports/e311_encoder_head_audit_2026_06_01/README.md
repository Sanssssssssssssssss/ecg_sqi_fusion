# E3.11f Encoder-Head Audit Package

This package is a GitHub-readable snapshot of the E3.11f denoise/classification experiments.
It intentionally excludes raw datasets, checkpoints, and NPZ tensors.

## 2026-06-01 Update: Do Not Promote U-Net-Only As Mainline

The later clean-confirm audit found a very strong balanced result, but it is **not** the desired publishable mainline because the classifier reads mature residual U-Net denoiser latent features rather than using a Transformer classification backbone.

Clean-confirm 3-seed results:

| group | acc mean/std | recall good/medium/bad mean | denoise score mean | interpretation |
|---|---:|---|---:|---|
| `encoder_latent_plus_explicit_sqi` | `0.98411 / 0.00162` | `0.9827 / 0.9732 / 0.9964` | `3.742` | U-Net latent oracle/teacher evidence |
| `encoder_latent_sqi` | `0.98274 / 0.00074` | `0.9787 / 0.9759 / 0.9936` | `3.735` | U-Net latent oracle/teacher evidence |
| `scratch_full` | `0.89177 / 0.00413` | `0.7793 / 0.9101 / 0.9859` | `1.501` | warm denoiser is mechanism-critical |
| `waveform_only_no_sqi` | `0.71223 / 0.02457` | `0.4918 / 0.7729 / 0.8719` | `3.515` | weak lightweight CNN lower bound, not a Transformer baseline |

This result is useful because it tells us the warm denoiser latent contains highly separable SQI/morphology information. It should be treated as an oracle/teacher probe, not as the main project architecture.

The next direction is therefore **Transformer re-entry**:

`noisy ECG -> warm denoiser / denoised ECG / SQI latent -> Transformer classifier -> optional small SQI adapter`

The mainline requirement is now explicit: classification must pass through a Transformer path. U-Net may remain a denoiser, teacher, or feature provider, but it cannot replace the Transformer classifier.

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
| Full no-SQI Transformer | `0.9623` | `0.9496` | `0.9510` | `0.9864` | `2.9036` | about `+8 dB` | denoised waveform -> MTLTransformerPTBXL |
| Best full SQI Transformer delta | `0.9628` | `0.9537` | `0.9455` | `0.9891` | `2.9655` | `+8.14 dB` | SQI helps but does not recover U-Net latent advantage |
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

The clean-confirm audit strengthens the mechanism story but also tightens the mainline constraint: a U-Net encoder MLP can be a teacher/oracle, but the final architecture must bring the Transformer back into the classification path.

## What Another Chat Should Analyze

Ask the reviewer to focus on these questions:

1. Is `encoder_head_only` a valid architecture direction, or mostly evidence that the denoiser encoder encodes synthetic label rules?
2. Why does the warm denoiser make both classification and denoise train so well, while scratch runs learned classification but not good denoise?
3. Does the dataset construction make bad/medium too easy because class labels are tied to visible SNR/morphology perturbations?
4. Should the final publishable architecture allow classification directly from denoiser encoder features, or should encoder features only enter through a small residual SQI head?
5. Which audit is still missing: frozen denoiser + scratch classifier, CE stop-gradient through denoiser, cross-gap evaluation, patient-level split audit, residual-only/noisy-only classifiers?
6. How should the warm denoiser latent be injected into a Transformer: SQI token prefix, FiLM adapter, CLS cross-attention, or residual SQI delta?
7. Can a pure Transformer denoiser/classifier reproduce enough of the denoise visual quality, or is warm denoiser pretraining still needed?

Suggested next experiments:

- Freeze mature denoiser and train only `encoder_head_only` from scratch.
- Train `encoder_head_only` with `detach_encoder_features=True` to prevent CE modifying the denoiser.
- Run cross-gap evaluation on gap5/gap6.5/gap7 for the encoder-head model.
- Train residual-only, noisy-only, clean-only, and denoised-only classifiers to measure shortcut strength.
- If patient metadata is recoverable, repeat with patient-level split.
- Run Transformer re-entry experiments where the final logits come from `MTLTransformerPTBXL`, with U-Net latent used only as SQI token/adapter/teacher.
- Run a pure Transformer denoise/classification control with no U-Net to determine whether the architecture can be made fully Transformer-based.

## Package Contents

- `code/encoder_head_audit.py`: experiment-only audit implementation.
- `code/run_encoder_head_audit.py`: two-run sequential runner.
- `metrics/*.json`: selected test reports and denoise metrics.
- `figures/*.png`: representative denoise galleries and conflict plots.
- `copied_reports/*.md`: previous full-joint, conflict, and mature-denoise reports.

No `src/transformer_pipeline` files are changed by this package.
