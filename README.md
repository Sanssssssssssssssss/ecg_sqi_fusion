# ECG SQI Fusion

Research code for ECG signal-quality assessment, noise-aware ECG denoising, and Transformer-based SQI classification.

The repository now has two deliberately separated lines:

1. `src/sqi_pipeline/`: classical SQI baselines. This line is preserved as-is.
2. `src/transformer_pipeline/`: PTB-XL Lead I Transformer/Uformer research and the current E3.11f mainline.

## Current Mainline

The thesis/mainline method is E3.11f `a_repr_detach_full_tokens`:

```text
noisy ECG
  -> Conv1D local stem
  -> hierarchical Uformer/Transformer encoder
  -> U-shaped decoder
  -> noise_hat
  -> denoise = noisy - 0.9 * noise_hat

multi-scale Uformer tokens + bottleneck + noisy/denoised/residual summaries
  -> detached feature vector
  -> small MLP SQI/classifier head
  -> good / medium / bad
```

Why detach: previous loss-conflict audits showed CE can keep classification high while damaging denoise. The classifier therefore reads the mature denoising representation, while denoiser updates are governed by denoise loss or very small continuation loss.

Mainline command:

```bash
python -m src.transformer_pipeline.train_uformer_mainline --stage all
```

Useful checks:

```bash
python -m compileall src/transformer_pipeline
python -m src.transformer_pipeline.train_uformer_mainline --stage dry_run
python -m src.transformer_pipeline.train_uformer_mainline --stage split_audit
```

Default output:

```text
outputs/mainline/e311_uformer_full_tokens_detach_seed0/
  ckpt_best.pt
  test_report.json
  train_log.json
  split_audit.json
  denoise_eval/
    denoise_metrics.json
    test_denoise_outputs.npz
  visuals/
    balanced_gallery.png
    hard_bad_gallery.png
    good_safety_gallery.png
    worst_residual_gallery.png
    qrs_tst_focus_gallery.png
    same_sample_stage1_stage2_gallery.png
    train_curves.png
```

## Mainline Rerun Snapshot

Clean source rerun, seed `0`, full split `10935 / 2184 / 2202`:

- Test acc: `0.98819`
- Good/medium/bad recall: `0.98910 / 0.97956 / 0.99591`
- Denoise score: `4.293`
- SNR gain: `12.386 dB`
- MSE ratio: `0.0445`

The earlier Uformer ablation winner that selected this architecture reached acc `0.99001`, bad recall `0.99591`, denoise score `4.282`.

## Repository Layout

`src/sqi_pipeline/`
Classical SQI/ML pipeline and baseline command line entrypoint. Do not modify this when working on the Uformer mainline.

`src/transformer_pipeline/models/mtl_transformer.py`
Legacy E3.11 multi-task Transformer baseline, retained for comparison.

`src/transformer_pipeline/models/uformer1d.py`
Current Uformer1D residual denoiser and detached SQI classifier head.

`src/transformer_pipeline/train_uformer_mainline.py`
Two-stage E3.11f mainline trainer.

`reports/experiment_archive/e311_lineage_2026_06_02/`
GitHub-readable experiment lineage archive: registry, metadata snapshots, selected figures, and reference experiment scripts.

`outputs/experiment_archive/e311_lineage_2026_06_02/`
Local mirror of archive metadata and pointers to full raw outputs. Large checkpoints and NPZ files stay outside git.

## Environment

Use Python 3.11 if possible.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If PyTorch is installed separately for CUDA or a cluster, `req.txt` is a lighter dependency list without `torch`.

## Classical SQI Pipeline

Run the full classical SQI line:

```bash
python -m src.sqi_pipeline.run_all --verbose
```

Useful flags:

```bash
python -m src.sqi_pipeline.cli --fresh
python -m src.sqi_pipeline.cli --only manifest_raw,split_seta,record84
python -m src.sqi_pipeline.cli --force
python -m src.sqi_pipeline.validate_outputs --write outputs/sqi/validation/current_seed0.json
```

## Legacy PTB-XL Transformer Workflow

The old preprocessing and legacy MTL Transformer workflow are retained for reproduction and ablation context:

```bash
python -m src.transformer_pipeline.run_preprocess_all --verbose
python -m src.transformer_pipeline.run_transformer_all --verbose
```

The old train step writes under `outputs/transformer/`; the current Uformer mainline writes under `outputs/mainline/`.
