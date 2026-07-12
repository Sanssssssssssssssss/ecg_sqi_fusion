# Code architecture

This repository has three public workflows:

1. build and evaluate classical SQI models;
2. build and evaluate waveform/Conformer models;
3. run four frozen inference models locally or in Docker.

Generated data, reports, figures, and temporary detector binaries are outputs,
not source modules.

## Repository map

| Path | Responsibility |
|---|---|
| `src/sqi_pipeline/` | Classical SQI preprocessing, QRS detection, features, SVM, and LM-MLP experiments |
| `src/transformer_pipeline/` | Main waveform data construction and E31 training pipeline |
| `src/supplemental_transformer_experiments/` | Supplemental and Chapter 4 analyses |
| `src/ecg_sqi_inference/` | Inference-only CLI used by Docker |
| `pretrained/` | Frozen checkpoints and inference bundles |
| `docker/inference/` | CPU inference image and user instructions |
| `docker/reproduce/` | Container wrapper for isolated reproduction targets |
| `reproduce/` | Fresh-clone controller and expected-output audits |
| `tests/` | Reproducibility, metric-contract, and inference tests |
| `outputs/` | Generated evidence; ignored by Git and not a Python package |

## Main entry points

Classical SQI pipeline:

```bash
python -m src.sqi_pipeline.run_all --profile paper --run
```

Waveform pipeline:

```bash
python -m src.transformer_pipeline.run_all --run --train E31
```

Chapter 4 evidence:

```bash
python -m src.supplemental_transformer_experiments.chapter4_evidence.run pipeline --run
```

Frozen inference:

```bash
python -m src.ecg_sqi_inference verify-bundles
python -m src.ecg_sqi_inference predict --model MODEL --input INPUT --fs 125 --out OUTPUT
```

The inference command accepts `.npy`, `.npz`, numeric `.csv`, and WFDB
records. It resamples to 125 Hz and classifies non-overlapping 10-second
segments.

## Inference models

| Model | Leads | Output |
|---|---:|---|
| `12lead-conformer` | 12 | `acceptable` / `unacceptable` |
| `12lead-rbfsvm` | 12 | `acceptable` / `unacceptable` |
| `singlelead-conformer` | 1 | `good` / `medium` / `bad` |
| `singlelead-rbfsvm` | 1 | `good` / `medium` / `bad` |

Every inference artifact is listed with a SHA-256 digest in
`pretrained/inference/manifest.json`. Prediction verifies the manifest before
deserializing a model.

Build and run the same CLI in Docker by following
`docker/inference/README.md`.

## Pipeline flow

The classical path is:

```text
raw ECG -> resampling -> QRS detector cache -> SQI features -> split -> model -> tables/reports
```

The waveform path is:

```text
raw ECG -> protocol construction -> train/validation/test materialization -> E31 -> frozen evidence
```

Both paths use repository-relative paths. Data and generated outputs remain
outside importable source packages.

## Experiment lineage

The current waveform data builder is `run_v116_native_budget_repair.py`.
Earlier numbered scripts remain because the current implementation imports
parts of their validated construction logic:

```text
v2 -> v14 -> v21 -> v37 -> v81 -> v114 -> v115 -> v116
```

These files under `src/transformer_pipeline/data_v1_gapfill/support/` are
historical implementation dependencies, not separate user entry points. New
runs should start from `src.transformer_pipeline.run_all`, not from a numbered
support script.

Chapter 4 uses the existing `chapter4_evidence.run pipeline` entry point.
Unfinished `revision_*`, `audit_*`, `final_*`, and `*_exploratory` work files
are development material until their results and interfaces are frozen; they
are not part of the public workflow.

## Frozen and generated files

- `pretrained/` contains only the four model artifacts and runtime profiles
  required for inference.
- `outputs/` contains regenerable experiment evidence and is not submitted.
- `data/`, `tmp/`, generated `outputs/`, caches, logs, and virtual
  environments are excluded from commits.
- Do not stage ignored replacement outputs with `git add -f` unless the report
  explicitly references them.

## Validation

Run the complete repository test suite with:

```bash
python -m pytest -q
```

Docker validation additionally builds the image, verifies bundle hashes, and
runs each of the four model names against a mounted ECG input.
