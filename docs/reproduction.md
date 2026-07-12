# Reproduction Guide

The reproduction controller fresh-clones `main`, creates an isolated Python
environment, runs one target, and audits the expected outputs.

```bash
python reproduce/run_reproduce.py --target baseline-cinc2011
python reproduce/run_reproduce.py --target baseline-but
python reproduce/run_reproduce.py --target conformer-cinc2011
python reproduce/run_reproduce.py --target conformer-but
python reproduce/run_reproduce.py --target inference-service
```

Generated clones, data, logs, and summaries stay under the ignored
`reproduce/work/` directory. Each run writes `summary.json`, `summary.md`,
`audit_matrix.csv`, `artifact_checksums.csv`, and command logs.

## Docker wrapper

```bash
docker build -f docker/inference/Dockerfile -t ecg-sqi-infer .
docker build -f docker/reproduce/Dockerfile -t ecg-reproduce .
docker run --rm -v "${PWD}/reproduce/work:/opt/reproduce/work" \
  ecg-reproduce --target inference-service
```

Paper-aligned SQI targets require `wqrs` and EP Limited/Hamilton detector
executables. See [wfdb-qrs-kit](wfdb_qrs_kit.md) for their setup and licensing
boundary.

