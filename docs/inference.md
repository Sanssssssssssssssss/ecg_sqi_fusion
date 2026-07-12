# Inference and Docker

The image contains four frozen inference models and does not train models or
download datasets.

## Build and verify

```bash
docker build -f docker/inference/Dockerfile -t ecg-sqi-infer .
docker run --rm ecg-sqi-infer verify-bundles
```

## Predict

```bash
docker run --rm -v /host/data:/data ecg-sqi-infer predict \
  --model singlelead-conformer --input /data/input --fs 500 --out /data/output
```

Available model names are:

- `12lead-conformer`
- `singlelead-conformer`
- `12lead-rbfsvm`
- `singlelead-rbfsvm`

Inputs may be `.npy`, `.npz`, numeric `.csv`, or WFDB records. The service
resamples to 125 Hz and predicts non-overlapping 10-second windows. Twelve-lead
models accept `(samples, 12)` or `(12, samples)`; single-lead models accept a
vector or one-column array.

For WSL, mount Windows drives below `/mnt`, for example:

```bash
docker run --rm -v /mnt/e/ecg-data:/data ecg-sqi-infer predict \
  --model singlelead-rbfsvm --input /data/input --fs 500 --out /data/output
```

Each run writes per-record CSV files, `all_segments.csv`, and
`run_summary.json`.

