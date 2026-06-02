# E3.11f Experiment Lineage Archive

Date: 2026-06-02

This archive records the path that led to the current thesis/mainline method:
`a_repr_detach_full_tokens`, implemented as a Uformer1D residual denoiser plus a detached full-token SQI/classifier head.

The important warning is explicit: the earlier U-Net encoder-head result is an oracle/teacher result, not the final mainline. The final candidate must contain a real Transformer-based denoiser path.

## Current Mainline

```text
noisy ECG
  -> Conv1D local stem
  -> hierarchical Uformer/Transformer encoder
  -> U-shaped decoder with skips
  -> noise_hat
  -> denoise = noisy - 0.9 * noise_hat

multi-scale Uformer tokens + bottleneck + noisy/denoised/residual summaries
  -> detached feature vector
  -> small MLP SQI/classifier head
  -> good / medium / bad
```

Clean rerun output:

- Source code: `src/transformer_pipeline/train_uformer_mainline.py`
- Output folder: `outputs/mainline/e311_uformer_full_tokens_detach_seed0/`
- Split: train `10935`, val `2184`, test `2202`
- Test acc: `0.98819`
- Good/medium/bad recall: `0.98910 / 0.97956 / 0.99591`
- Denoise score: `4.293`
- SNR gain: `12.386 dB`
- MSE ratio: `0.0445`

The audit winner that motivated promotion:

- `a_repr_detach_full_tokens`
- Acc `0.99001`
- Good/medium/bad `0.98501 / 0.98910 / 0.99591`
- Denoise score `4.282`
- SNR gain `12.342 dB`

## Why This Is The Mainline

The ablations show the mechanism is not accidental:

- `residual_summary_only` collapses, so the classifier is not just reading a simple SNR/residual proxy.
- Removing the U-shaped skip connections weakens denoise and classification.
- Removing Transformer blocks hurts representation quality even when denoise remains decent.
- Letting CE backprop through the denoiser keeps classification high but destroys denoise, so detach is a core design decision.
- Scratch/warm audits show mature denoiser representation is the useful intermediate object.

## Registry

Use `registry.jsonl` as the canonical experiment registry. Each line contains:

- `tag`
- `status`
- `method`
- `key_result`
- `value`
- `artifacts`
- `recommendation`

Older tracked E3.11 reports were moved into `legacy_reports/` so the root
`reports/` folder stays readable. Non-E3.11 reports remain in place.

## Figures

- `figures/mainline_same_sample_stage1_stage2.png`: clean/noisy/stage1/stage2 same-sample view.
- `figures/mainline_balanced_gallery.png`: current mainline balanced test gallery.
- `figures/mainline_train_curves.png`: Stage 2 validation curves.
- `figures/uformer_mechanism_grid.png`: mechanism comparison grid from the Uformer audit.

## What Is Not In Git

Large files stay local:

- Checkpoints
- `test_denoise_outputs.npz`
- Raw experiment logs
- Full `outputs/experiment/...` folders

The local mirror is under:

```text
outputs/experiment_archive/e311_lineage_2026_06_02/
```

## Recommendation

Proceed with this as the thesis/mainline architecture. The next clean confirmations should be seed `0/1/2` reruns and a compact paper table comparing:

- current Uformer mainline
- highest-score Uformer backup
- U-Net oracle
- Transformer adapter
- pure Transformer control
- residual/summary-only negative controls
