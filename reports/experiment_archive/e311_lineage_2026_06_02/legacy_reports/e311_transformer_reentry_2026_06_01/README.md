# E3.11f Transformer Re-Entry Audit Package

This package records the transition from the strong U-Net latent oracle result back to a Transformer-centered architecture.

Raw data, checkpoints, and NPZ tensors are intentionally excluded. Experiment code is copied as a snapshot under `code/`; live runs remain under `outputs/experiment/e311_sqi_denoise_classifier_grid`.

## Why This Exists

The clean-confirm audit showed that warm residual U-Net denoiser latent features classify the synthetic SQI labels very well:

- `encoder_latent_plus_explicit_sqi`: mean acc `0.9841`, recalls `0.9827 / 0.9732 / 0.9964`, denoise score `3.742`.
- `encoder_latent_sqi`: mean acc `0.9827`, recalls `0.9787 / 0.9759 / 0.9936`, denoise score `3.735`.
- `scratch_full`: mean acc `0.8918`, denoise score `1.501`.

That is a useful mechanism/oracle result, but it is **not** an acceptable mainline architecture because the classifier is not Transformer-based. The new goal is to keep the denoise quality and SQI insight while restoring a Transformer classifier.

## Current Baselines

| model family | acc | good | medium | bad | denoise score | note |
|---|---:|---:|---:|---:|---:|---|
| Full no-SQI Transformer | `0.9623` | `0.9496` | `0.9510` | `0.9864` | `2.9036` | warm denoiser -> denoised ECG -> `MTLTransformerPTBXL` |
| Best full SQI Transformer delta | `0.9628` | `0.9537` | `0.9455` | `0.9891` | `2.9655` | SQI residual helps a little, medium not stable |
| Best PCGrad/cap Transformer | `0.9728` | `0.9510` | `0.9741` | `0.9932` | `2.2825` | conflict controlled, denoise weaker |
| U-Net latent oracle + explicit SQI | `0.9841` mean | `0.9827` | `0.9732` | `0.9964` | `3.742` | teacher/oracle only, not mainline |

## Implemented Transformer Re-Entry Variants

All implemented variants keep final classification logits on a Transformer path and are experiment-only.

- `dual_branch`: denoised ECG goes through `MTLTransformerPTBXL`; optional zero-init denoiser-latent residual delta adjusts logits.
- `transformer_film_sqi`: denoised ECG goes through `MTLTransformerPTBXL`; warm denoiser latent applies zero-init FiLM-style modulation to the Transformer pooled feature.
- `transformer_cross_sqi`: denoised ECG goes through `MTLTransformerPTBXL`; Transformer pooled feature cross-attends to the warm denoiser latent through a small zero-init adapter.
- `transformer_sqi_token_prefix`: warm denoiser latent is projected into an SQI token inserted after CLS before Transformer encoder blocks.
- Round 3 pure Transformer controls use `src/experiment/e311_sqi_research/train.py` on the same sweep artifact, without the U-Net denoiser path.

Smoke checks passed on the full split `10935 / 2184 / 2202`:

- `transformer_film_sqi` with warm denoiser + He-scratch Transformer completed 1 epoch / 1 batch and generated reports.
- `transformer_sqi_token_prefix` with warm denoiser + warm Transformer completed 1 epoch / 1 batch; base Transformer test acc was `0.9600`, token-path acc was `0.9573`, confirming the token path is active without catastrophic breakage.

## Runner

Live experiment runner:

`outputs/experiment/e311_sqi_denoise_classifier_grid/run_transformer_reentry_audit.py`

Expected state files:

- `transformer_reentry_state.json`
- `transformer_reentry_specs.jsonl`
- `transformer_reentry_summary.jsonl`
- logs under `logs/transformer_reentry_audit`

Main command:

```powershell
.\.venv\Scripts\python.exe outputs\experiment\e311_sqi_denoise_classifier_grid\run_transformer_reentry_audit.py
```

The first full wave runs:

- Round 1: warm denoiser + Transformer baseline, no SQI, warm classifier and He-scratch classifier, seeds `0/1/2`.
- Round 1: warm denoiser + Transformer + latent residual delta, seeds `0/1/2`.
- Round 2: FiLM / cross-attention / SQI-token Transformer adapters, warm classifier and He-scratch classifier, seeds `0/1`.
- Round 3: pure Transformer controls from the existing E3.11 research trainer.

## Acceptance For Mainline Consideration

- Final classification logits must come from a Transformer path.
- Initial mechanism target: acc `>= 0.97`, bad recall near `0.99`, no severe good/medium collapse.
- Denoise target: denoise score `>= 3.0` or visual denoise clearly closer to clean than noisy.
- If Transformer re-entry remains far below U-Net oracle, the U-Net result stays a teacher/oracle and the mainline should not be promoted yet.

## Files

- `metrics/clean_confirm_aggregate.json`: 3-seed U-Net oracle/proxy/scratch aggregate.
- `metrics/data_leakage_audit.json`: split and duplicate audit.
- `metrics/smoke_*`: Transformer adapter smoke outputs.
- `figures/same_samples_clean_confirm_gallery.png`: fixed-sample U-Net oracle comparison.
- `code/encoder_head_audit.py`: experiment-only model/training implementation snapshot.
- `code/run_transformer_reentry_audit.py`: experiment runner snapshot.
