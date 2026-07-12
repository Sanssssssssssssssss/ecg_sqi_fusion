# Reproduction Guide

The repository separates a quick public-data rebuild from exact replay of the
frozen historical support pool. Generated artifacts remain under ignored
`outputs/` or `reproduce/work/` directories.

## Clean-room controller

The controller clones `main`, creates an isolated environment, runs one target,
and audits expected outputs.

| Target | Purpose | Data |
|---|---|---|
| `baseline-cinc2011` | Classical 12-lead SQI pipeline | Set-A and NSTDB |
| `baseline-but` | Single-lead classical baseline | BUT QDB |
| `conformer-cinc2011` | Set-A waveform path | Set-A and public support data |
| `conformer-but` | BUT waveform path | BUT QDB |
| `inference-service` | Model hash and service check | None |
| `sqi-supplemental` | Classical diagnostic analyses | Existing baseline outputs |
| `transformer-supplemental` | Waveform diagnostic analyses | Existing transformer outputs |

```bash
python reproduce/run_reproduce.py --target inference-service
python reproduce/run_reproduce.py --target baseline-cinc2011
```

Every run writes `summary.json`, `summary.md`, `audit_matrix.csv`,
`artifact_checksums.csv`, and command logs.

## Classical pipeline

```bash
python -m src.sqi_pipeline.cli \
  --profile paper_aligned \
  --artifacts_dir outputs/sqi_paper_aligned \
  --seed 0
```

Paper-aligned QRS detection requires the external detector programs described
under [wfdb-qrs-kit](wfdb_qrs_kit.md). To rerun one stage, use
`--only manifest_raw,record84`; to remove generated outputs first, use
`--fresh` deliberately.

## Transformer pipeline

Audit before executing expensive stages:

```bash
python -m src.transformer_pipeline.cli audit
python -m src.transformer_pipeline.cli pipeline --train none
```

Execute data construction and E31 training explicitly:

```bash
python -m src.transformer_pipeline.cli pipeline --run --train E31
```

## Docker wrapper

```bash
docker build -f docker/inference/Dockerfile -t ecg-sqi-infer .
docker build -f docker/reproduce/Dockerfile -t ecg-reproduce .
docker run --rm -v "${PWD}/reproduce/work:/opt/reproduce/work" \
  ecg-reproduce --target inference-service
```

## Interpretation of a rebuild

A public-data rebuild demonstrates that the pipeline executes and satisfies
its contracts. Exact replay of frozen v116 construction requires the archived
support assets; without them the audit records
`historical_support_exact=false`. This distinction follows the limitations in
the [submitted report](report.md).

