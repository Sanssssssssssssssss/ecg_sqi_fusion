# CLI Reference

## Frozen inference

```text
python -m src.ecg_sqi_inference predict
  --model {12lead-conformer,singlelead-conformer,12lead-rbfsvm,singlelead-rbfsvm}
  --input PATH
  --fs HZ
  --out PATH
  [--device {cpu,cuda}]
```

| Option | Contract |
|---|---|
| `--model` | Required stable model identifier from the catalogue |
| `--input` | One supported ECG file or a recursively scanned directory |
| `--fs` | Positive source sampling frequency in hertz |
| `--out` | Output directory; created if absent |
| `--device` | Conformer device request; defaults to CPU |

Additional commands:

```bash
python -m src.ecg_sqi_inference verify-bundles
python -m src.ecg_sqi_inference export-inference-bundle --out PATH
```

## Classical SQI pipeline

```text
python -m src.sqi_pipeline.cli
  [--profile {baseline,paper_aligned}]
  [--artifacts_dir PATH]
  [--seed INT]
  [--only STEP,STEP]
  [--fresh] [--force] [--verbose]
```

`--fresh` deletes generated artifacts below the selected output root before
running. `--force` asks each stage to ignore its normal reuse checks. `--only`
executes named stages from the selected profile; dependencies must already
exist when an upstream stage is omitted.

## Transformer pipeline

```text
python -m src.transformer_pipeline.cli [GLOBAL OPTIONS] COMMAND
```

| Command | Purpose | Mutating execution gate |
|---|---|---|
| `audit` | Validate protocol and frozen contracts | None |
| `plot` | Render current diagnostic figures | None |
| `report` | Build current generated report outputs | None |
| `extract-but` | Extract BUT data | `--run` |
| `clean-smoke` | Run public-data smoke reconstruction | `--run` |
| `build-v116` | Construct v116 protocol | `--run` |
| `split` | Materialise frozen split | `--run` |
| `train --model E31` | Train selected model | `--run` |
| `pipeline --train E31` | Orchestrate full path | `--run` |

The explicit `--run` gates prevent an accidental expensive build from a help
or audit command.
