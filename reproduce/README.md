# Full Reproduction

This controller fresh-clones `main`, creates an isolated environment, runs one
published pipeline target, and audits its expected outputs. Generated clones,
data, logs, and summaries stay below `reproduce/work/`, which is ignored by
Git.

From the repository root:

```bash
python reproduce/run_reproduce.py --target baseline-cinc2011
python reproduce/run_reproduce.py --target baseline-but
python reproduce/run_reproduce.py --target conformer-cinc2011
python reproduce/run_reproduce.py --target conformer-but
python reproduce/run_reproduce.py --target inference-service
```

The PowerShell and shell wrappers under `reproduce/scripts/` run the same
targets. Use `--repo-url` or `--branch` to reproduce another remote revision.

## Docker

Build the inference base first, then the reproduction controller:

```bash
docker build -f docker/inference/Dockerfile -t ecg-sqi-infer .
docker build -f docker/reproduce/Dockerfile -t ecg-reproduce .
docker run --rm -v "${PWD}/reproduce/work:/opt/reproduce/work" \
  ecg-reproduce --target inference-service
```

The base image supplies `wqrs` and `eplimited`. Data-bearing targets download
public data unless `--raw-copy` or `--no-download` is selected. To reuse local
raw data, mount it and pass `--raw-copy --raw-data-root /data`.

Each run writes `summary.json`, `summary.md`, `audit_matrix.csv`,
`artifact_checksums.csv`, and command logs below its run directory.
