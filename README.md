# ECG SQI Fusion

Research code for ECG signal-quality assessment and waveform Transformer-based SQI classification.

`main` is the active official line. Use the v116 data pipeline and E31 model below unless you are intentionally reproducing an old baseline.

The repository has two deliberately separated lines:

1. `src/sqi_pipeline/`: classical SQI baselines. This line is preserved as-is.
2. `src/transformer_pipeline/`: waveform Transformer SQI research and the current v116/E31 mainline.

## Official Mainline

The Transformer mainline is data v1/v116 plus `E31_wave_mechanism_conformer`:

```text
BUT gap5 originals
  -> record-heldout split
  -> train-only medium/bad gap fill
  -> dual-view waveform channels
  -> named-query SQI Conformer
  -> mechanism auxiliary heads
  -> good / medium / bad
```

Validation and test stay pure `original_but`; only the training split is balanced.
The model input is waveform-derived channels only. SQI-like factors remain
auxiliary targets, not scalar input features.

Mainline command:

```bash
python -m src.transformer_pipeline.data_v1_gapfill pipeline --run --train E31
```

Useful checks:

```bash
python -m compileall -q src/transformer_pipeline/data_v1_gapfill src/transformer_pipeline/models
python -m src.transformer_pipeline.data_v1_gapfill audit
python -m src.transformer_pipeline.data_v1_gapfill train-check --model E31
```

Current data contract:

```text
policy: v116_gapfill_dual_goodorig_nm40_ms10_smc_s20260876
protocol rows: 31590 = 10530/10530/10530
train: 8310/8310/8310
val/test: original_but only
sampler: raw rows, no record-balanced sampler
```

## Current Snapshot

Latest full v116 + E31 run, seed `20260876`:

- E31 test acc: `0.9362`
- E31 macro F1: `0.9483`
- Good/medium/bad recall: `0.9250 / 0.9399 / 0.9939`

This is the current checked-in mainline; model soft-parameter tuning continues from here.

## Repository Layout

`src/sqi_pipeline/`
Classical SQI/ML pipeline and baseline command line entrypoint.

`src/transformer_pipeline/data_v1_gapfill/`
Current v116 data build, audit, plot, report, and E31 training-check entrypoint.

`src/transformer_pipeline/models/mtl_transformer.py`
Legacy multi-task Transformer baseline, retained for reproduction.

`src/transformer_pipeline/models/uformer1d.py`
Compatibility shim for archived experiments. The old Uformer mainline has been removed.

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

The old train step writes under `outputs/transformer/`; the official mainline is `data_v1_gapfill`.
