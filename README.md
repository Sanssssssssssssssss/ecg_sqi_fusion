# ECG SQI Fusion

Research code for ECG signal-quality assessment and waveform Transformer-based SQI classification.

`main` is the active official line. Use the v116 data pipeline and E31 model below unless you are working on the classical SQI baseline.

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
python -m compileall -q src/sqi_pipeline src/transformer_pipeline/data_v1_gapfill src/utils
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

Latest full v116 + soft-tuned E31 run, seed `20260876`:

- E31 test acc: `0.9432`
- E31 macro F1: `0.9525`
- Good/medium/bad recall: `0.9421 / 0.9320 / 0.9939`

This is the current checked-in mainline. The E31 architecture is unchanged; the
soft tuning only updates class weights to `[1.03, 1.05, 1.08]`.

## Repository Layout

`src/sqi_pipeline/`
Classical SQI/ML pipeline and baseline command line entrypoint.

`src/transformer_pipeline/data_v1_gapfill/`
Current v116 data build, audit, plot, report, and E31 training-check entrypoint.

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
python -m src.sqi_pipeline.validate_outputs --write tmp/sqi/validation/current_seed0.json
```
